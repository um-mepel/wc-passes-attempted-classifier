"""Walk-forward backtest with strict train/test separation (see splits.py).

For each fold:
  1. Fit style clusters + minutes + rate model on TRAIN ONLY.
  2. Build features as-of-date and predict the held-out tournament.
  3. Score the DISTRIBUTION: CRPS + log-loss/Brier of P(over) at actual lines,
     plus calibration. Betting metrics computed only where Underdog lines exist.

CRPS and log-loss are the metrics that reward correct variance — the whole point
for over/under pricing — so model selection uses them, never RMSE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import betting
from .config import Config
from .features import build, fit_style_clusters, team_match_table
from .minutes_model import MinutesModel
from .predict import posterior_predictive, prob_over
from .rate_model import HierNB
from .splits import chunked_holdout, walk_forward


def crps_sample(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS estimated from predictive samples (energy form). Lower is better."""
    s = np.sort(samples, axis=1)
    n = s.shape[1]
    term1 = np.mean(np.abs(s - y[:, None]), axis=1)
    # E|X-X'| via sorted-sample identity
    idx = np.arange(1, n + 1)
    term2 = (2.0 / (n * n)) * np.sum((2 * idx - n - 1) * s, axis=1)
    return term1 - 0.5 * term2


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run(cfg: Config, pm: pd.DataFrame, lines: pd.DataFrame | None = None,
        use_actual_minutes: bool = False) -> pd.DataFrame:
    rows = []
    bt = cfg["backtest"]
    scheme = bt.get("scheme", "walk_forward")
    if scheme == "wc_chunked":
        # Live-tournament backtest: base = all data before the tournament, then step through
        # its matches in date chunks, retraining (per-fold fit below) after each chunk.
        folds = chunked_holdout(pm, tournament=bt.get("tournament", "World Cup 2026"),
                                chunk_days=int(bt.get("chunk_days", 1)),
                                chunk_mode=bt.get("chunk_mode", "days"))
        print(f"[backtest] {len(folds)} {bt.get('tournament', 'World Cup 2026')} chunks "
              f"(expanding window, retrain per chunk, {bt.get('chunk_days', 1)} date(s)/chunk)", flush=True)
    else:
        folds = walk_forward(pm, by="competition")
        print(f"[backtest] {len(folds)} walk-forward folds (train/test strictly separated)", flush=True)
    for i, fold in enumerate(folds, 1):
        train, test = fold.train, fold.test
        print(f"\n[backtest] fold {i}/{len(folds)} — hold out '{fold.name}' "
              f"(train {len(train)} rows < {fold.cutoff.date()} | test {len(test)})", flush=True)

        # fit everything on TRAIN ONLY
        km_style = fit_style_clusters(team_match_table(train), cfg["features"]["opponent_style_clusters"])[1]
        feats_all = build(pd.concat([train, test]), cfg, style_map=km_style)  # as-of-date; test sees only past
        # train on all PLAYED rows; evaluate only on STARTERS (who Underdog prices)
        ftrain = feats_all[feats_all["match_id"].isin(train["match_id"]) & (feats_all["minutes"] > 0)]
        ftest = feats_all[feats_all["match_id"].isin(test["match_id"]) & feats_all["started"].fillna(False)]

        minutes = MinutesModel(cfg["features"]["recency_halflife_matches"]).fit(train)
        model = HierNB(cfg).fit(ftrain)

        p_start = MinutesModel.recent_start_prob(feats_all, cfg["features"]["recency_halflife_matches"])
        p_start_test = p_start.loc[ftest.index].values

        samples = posterior_predictive(model, minutes, ftest, p_start_test,
                                       use_actual_minutes=use_actual_minutes, seed=cfg["model"]["seed"])
        y = ftest["passes_attempted"].values
        crps_rows = crps_sample(samples, y)
        crps = crps_rows.mean()
        ae = np.abs(samples.mean(1) - y)

        # split keepers vs outfield so the GK-head delta is visible (not blended away —
        # GKs are ~5.5% of rows, so a big GK fix barely moves the overall number).
        is_gk = ftest["position"].astype(str).str.contains("Goalkeep", case=False, na=False).values
        rec = {"fold": fold.name, "n_test": len(ftest), "crps": float(crps),
               "mae": float(ae.mean()),
               "n_gk": int(is_gk.sum()),
               "crps_gk": float(crps_rows[is_gk].mean()) if is_gk.any() else float("nan"),
               "mae_gk": float(ae[is_gk].mean()) if is_gk.any() else float("nan"),
               "mae_out": float(ae[~is_gk].mean()) if (~is_gk).any() else float("nan")}
        print(f"[backtest] fold {i}/{len(folds)} done — CRPS {rec['crps']:.2f}, MAE {rec['mae']:.1f} "
              f"| GK n={rec['n_gk']} CRPS {rec['crps_gk']:.2f} MAE {rec['mae_gk']:.1f} "
              f"(outfield MAE {rec['mae_out']:.1f})", flush=True)

        if lines is not None:
            merged = ftest.merge(lines, on=["match_id", "player_id"], how="inner")
            if len(merged):
                samp_m = posterior_predictive(model, minutes, merged,
                                              p_start.loc[merged.index].values if set(merged.index) <= set(p_start.index) else np.full(len(merged), 0.7),
                                              use_actual_minutes=use_actual_minutes, seed=cfg["model"]["seed"])
                po = prob_over(samp_m, merged["line"].values)
                yo = (merged["passes_attempted"].values >= np.ceil(merged["line"].values)).astype(float)
                rec["log_loss"] = log_loss(po, yo)
                edges = betting.edge_table(merged.assign(line=merged["line"]), po)
                rec.update({f"bet_{k}": v for k, v in betting.grade(edges).items()})
                if rec.get("bet_n_plays", 0) < cfg["backtest"]["min_plays_warn"]:
                    rec["warn"] = f"<{cfg['backtest']['min_plays_warn']} plays — ROI noisy"
        rows.append(rec)
    return pd.DataFrame(rows)
