"""Command-line entry point. Retraining on new data is ONE command:

    python3 -m src.cli retrain      # incremental ingest -> features -> fit -> save

Other commands:
    python3 -m src.cli list-competitions   # find StatsBomb comp/season ids for config
    python3 -m src.cli ingest              # pull new matches only (incremental)
    python3 -m src.cli backtest            # walk-forward, strict train/test separation
    python3 -m src.cli predict --upcoming upcoming.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .config import Config


def _load_corpus(cfg: Config) -> pd.DataFrame:
    """Assemble the modeling corpus. Prefers the combined StatsBomb+Fotmob corpus
    (data/raw/combined_player_match.parquet, built by scripts/build_fotmob_corpus.py)
    so training/calibration include live 2026 per-player passes; falls back to the
    StatsBomb (+optional FBref) build."""
    combined = cfg.path("raw") / "combined_player_match.parquet"
    if combined.exists():
        pm = pd.read_parquet(combined)
        print(f"[corpus] using combined StatsBomb+Fotmob corpus ({len(pm)} rows, "
              f"{(pm['provider'] == 'fotmob').sum()} Fotmob)")
        return pm
    from .ingest_statsbomb import build_player_match as sb_pm
    pm = sb_pm(cfg)                      # incremental; pass label lives here
    # FBref friendlies/quals add minutes/form rows (no pass label unless paid feed)
    try:
        from .ingest_fbref import build_player_match as fb_pm
        fb = fb_pm(cfg)
        if len(fb) and cfg["fbref"].get("passing_available", False):
            pm = pd.concat([pm, fb], ignore_index=True)
    except Exception as e:
        print(f"[corpus] FBref skipped: {e}")
    return pm


def cmd_list_competitions(cfg, args):
    from .ingest_statsbomb import list_competitions
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(list_competitions().to_string(index=False))


def cmd_ingest(cfg, args):
    pm = _load_corpus(cfg)
    print(f"Ingested {len(pm)} player-match rows across "
          f"{pm['match_id'].nunique()} matches, {pm['competition'].nunique()} competitions.")


def cmd_retrain(cfg, args):
    from .features import build
    from .minutes_model import MinutesModel
    from .rate_model import HierNB

    pm = _load_corpus(cfg)
    feats = build(pm, cfg)                         # as-of-date features (no leakage)
    labeled = feats[feats["minutes"] > 0]
    print(f"Fitting on {len(labeled)} labeled player-match rows...")
    MinutesModel(cfg["features"]["recency_halflife_matches"]).fit(pm)
    model = HierNB(cfg).fit(labeled)
    model.save(cfg.path("models") / "hiernb")
    print(f"Saved model to {cfg.path('models') / 'hiernb'}")


def cmd_backtest(cfg, args):
    from . import backtest
    from .ingest_underdog import load_lines
    pm = _load_corpus(cfg)
    lines = load_lines(cfg, pm)
    res = backtest.run(cfg, pm, lines=lines if len(lines) else None,
                       use_actual_minutes=args.actual_minutes)
    print(res.to_string(index=False))
    print("\nMean CRPS (lower=better):", round(res["crps"].mean(), 3))


def cmd_predict(cfg, args):
    from .features import build
    from .minutes_model import MinutesModel
    from .predict import posterior_predictive, prob_over, summarize
    from .rate_model import HierNB
    from . import betting

    pm = _load_corpus(cfg)
    upcoming = pd.read_csv(args.upcoming)          # rows describing tonight's player-matches
    feats = build(pd.concat([pm, upcoming], ignore_index=True), cfg)
    fu = feats[feats["match_id"].isin(upcoming["match_id"])]
    # passes-attempted props cover the starting XI — only predict likely starters
    if "started" in fu:
        fu = fu[fu["started"].fillna(True)]
    model = HierNB.load(cfg, cfg.path("models") / "hiernb")
    minutes = MinutesModel(cfg["features"]["recency_halflife_matches"]).fit(pm)
    p_start = upcoming["p_start"].values if "p_start" in upcoming else \
        MinutesModel.recent_start_prob(feats, cfg["features"]["recency_halflife_matches"]).loc[fu.index].values
    samples = posterior_predictive(model, minutes, fu, p_start)
    summ = summarize(samples).set_index(fu.index)
    out = fu.assign(**{c: summ[c] for c in ["pred", "ci_low", "ci_high", "moe", "std"]})
    out["interval_80"] = out.apply(lambda r: f"{r.pred} ± {r.moe}  [{r.ci_low}, {r.ci_high}]", axis=1)
    if "line" in fu:
        out["p_over"] = prob_over(samples, fu["line"].values)
        edges = betting.edge_table(out, out["p_over"].values)
        print(edges[["player", "line", "pred", "interval_80", "p_over", "pick", "edge"]].to_string(index=False))
    else:
        print(out[["player", "pred", "interval_80", "std"]].to_string(index=False))


COMMANDS = {
    "list-competitions": cmd_list_competitions,
    "ingest": cmd_ingest,
    "retrain": cmd_retrain,
    "backtest": cmd_backtest,
    "predict": cmd_predict,
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wc-passes-model")
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--config", default=None)
    ap.add_argument("--upcoming", default=None, help="CSV of upcoming player-matches (predict)")
    ap.add_argument("--actual-minutes", action="store_true",
                    help="backtest with actual minutes (isolates rate-model quality)")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
