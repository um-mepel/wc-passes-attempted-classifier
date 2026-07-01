"""A/B harness for the magnetism + pressure axes.

Fits each requested variant on IDENTICAL walk-forward folds and scores out-of-sample
CRPS/MAE + a targeted bias check (the build-up-defender over-prediction cohort that
magnetism is meant to fix). One feature build per fold, shared across variants, so the
comparison is apples-to-apples.

Variants:
  baseline : production HierNB, _FEATURES (5)
  way1     : HierNB, _FEATURES + [share_asof, team_vol_asof, oppposs_x_buildup, oppposs_x_att]
  way2     : TwoStage team_volume x share (src/volume_share.py)

Usage:
  PM_DRAWS=250 PM_TUNE=250 PM_CORES=2 PYTHONPATH=. python3 scripts/ab_axes.py \
      --variants baseline,way1 --folds 2
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import Config
from src.features import build, fit_style_clusters, load_corpus, team_match_table
from src.rate_model import HierNB, _FEATURES
from src.predict import posterior_predictive
from src.backtest import crps_sample
from src.splits import walk_forward

WAY1_FEATURES = list(_FEATURES) + [
    "share_asof", "team_vol_asof", "oppposs_x_buildup", "oppposs_x_att"]
# way1r: identical to way1 but RECENCY-weighted share replaces the flat expanding mean,
# to isolate whether tracking a share TREND (vs a flat average) adds OOS signal.
WAY1R_FEATURES = list(_FEATURES) + [
    "share_recencybiased", "team_vol_asof", "oppposs_x_buildup", "oppposs_x_att"]
# waybox: baseline + striker playing-style (box-touch ratio) — a poacher-vs-drop-deep axis
# the model lacks. Single feature, isolates whether it fixes the striker over-prediction.
WAYBOX_FEATURES = list(_FEATURES) + ["box_ratio_asof"]
# qualifier/friendly folds are a scaler-range artifact, not bet-relevant (see calibrate.py)
SKIP_FOLDS = {"WC Qualifier", "Intl Friendly"}
BUILDUP_ROLES = {"CB", "FB", "DM", "CM"}


def _hiernb_samples(cfg, ftrain, ftest, features, seed):
    model = HierNB(cfg, features=features).fit(ftrain)
    # use_actual_minutes=True -> minutes model + p_start are unused; eval on realized minutes
    return posterior_predictive(model, None, ftest, np.full(len(ftest), 1.0),
                                n_draws=1000, use_actual_minutes=True, seed=seed)


def _twostage_samples(cfg, ftrain, ftest, seed):
    from src.volume_share import TwoStage, predict_two_stage
    model = TwoStage(cfg).fit(ftrain)
    return predict_two_stage(model, ftest, n_draws=1000, use_actual_minutes=True, seed=seed)


def _score(name, samples, ftest):
    y = ftest["passes_attempted"].values.astype(float)
    crps = crps_sample(samples, y)
    pred = samples.mean(1)
    ae = np.abs(pred - y)
    is_gk = ftest["position"].astype(str).str.contains("Goalkeep", case=False, na=False).values
    return pd.DataFrame({
        "variant": name, "player": ftest["player"].values, "role": ftest["role"].values,
        "position": ftest["position"].values, "x_poss": ftest["x_poss"].values,
        "is_gk": is_gk, "y": y, "pred": pred, "ae": ae, "crps": crps,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="baseline,way1")
    ap.add_argument("--folds", type=int, default=2, help="most-recent N tournament folds")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    cfg = Config.load()
    pm = load_corpus(cfg)
    folds = [f for f in walk_forward(pm, by="competition") if f.name not in SKIP_FOLDS]
    folds = folds[-args.folds:]
    print(f"[ab] {len(folds)} folds: {[f.name for f in folds]}  variants={variants}", flush=True)

    per_row = []
    for i, fold in enumerate(folds, 1):
        train, test = fold.train, fold.test
        km = fit_style_clusters(team_match_table(train), cfg["features"]["opponent_style_clusters"])[1]
        feats = build(pd.concat([train, test]), cfg, style_map=km)   # as-of; test sees only past
        ftrain = feats[feats["match_id"].isin(train["match_id"]) & (feats["minutes"] > 0)]
        ftest = feats[feats["match_id"].isin(test["match_id"]) & feats["started"].fillna(False)
                      & (feats["minutes"] > 0)]
        print(f"\n[ab] fold {i}/{len(folds)} '{fold.name}': train {len(ftrain)} / test {len(ftest)}", flush=True)
        seed = cfg["model"]["seed"]
        for v in variants:
            print(f"[ab]   fitting {v} ...", flush=True)
            if v == "baseline":
                s = _hiernb_samples(cfg, ftrain, ftest, list(_FEATURES), seed)
            elif v == "way1":
                s = _hiernb_samples(cfg, ftrain, ftest, WAY1_FEATURES, seed)
            elif v == "way1r":
                s = _hiernb_samples(cfg, ftrain, ftest, WAY1R_FEATURES, seed)
            elif v == "waybox":
                s = _hiernb_samples(cfg, ftrain, ftest, WAYBOX_FEATURES, seed)
            elif v == "way2":
                s = _twostage_samples(cfg, ftrain, ftest, seed)
            else:
                raise SystemExit(f"unknown variant {v}")
            per_row.append(_score(v, s, ftest).assign(fold=fold.name))

    R = pd.concat(per_row, ignore_index=True)
    out = cfg.path("processed") / "ab_axes_rows.parquet"
    R.to_parquet(out)

    print("\n==================== A/B RESULTS (OOS, actual minutes) ====================")
    print(f"{'variant':10} {'CRPS':>7} {'MAE':>7} {'MAE_out':>8} {'MAE_gk':>7} {'buildup_bias':>13}")
    base_crps = None
    for v in variants:
        g = R[R.variant == v]
        # build-up over-prediction cohort: CB/FB/DM/CM in the top possession tercile
        thr = np.nanpercentile(g["x_poss"], 66)
        cohort = g[g.role.isin(BUILDUP_ROLES) & (g.x_poss >= thr)]
        bias = float((cohort.pred - cohort.y).mean()) if len(cohort) else float("nan")
        crps = g.crps.mean(); mae = g.ae.mean()
        mae_out = g[~g.is_gk].ae.mean(); mae_gk = g[g.is_gk].ae.mean() if g.is_gk.any() else float("nan")
        if v == "baseline":
            base_crps = crps
        tag = ""
        if base_crps is not None and v != "baseline":
            tag = f"  (CRPS {crps-base_crps:+.2f} vs baseline)"
        print(f"{v:10} {crps:7.2f} {mae:7.1f} {mae_out:8.1f} {mae_gk:7.1f} {bias:+13.1f}{tag}")
    print(f"\nrows -> {out}")
    print("buildup_bias = mean(pred - actual) for CB/FB/DM/CM in top possession tercile; "
          "closer to 0 is better (baseline over-predicts these — Kounde/Upamecano failure).")

    # STRIKER-specific view: box_ratio only touches forwards (~16% of rows), so the overall
    # CRPS is diluted. This is where a striker-style feature must show up if it works.
    print(f"\n--- STRIKERS only (role==ST) ---")
    print(f"{'variant':10} {'n':>5} {'CRPS':>7} {'MAE':>7} {'bias':>7}")
    for v in variants:
        st = R[(R.variant == v) & (R.role == "ST")]
        if not len(st):
            continue
        print(f"{v:10} {len(st):5} {st.crps.mean():7.2f} {st.ae.mean():7.2f} {float((st.pred-st.y).mean()):+7.2f}")


if __name__ == "__main__":
    main()
