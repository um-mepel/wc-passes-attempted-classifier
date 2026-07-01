"""Calibration WITH a box_ratio striker correction, compared head-to-head vs uncorrected.

The striker role-mean over-predicts stay-high poachers (+4) and under-predicts drop-deep
link strikers (-3); box_ratio (touches-in-box / touches) separates them. Here we estimate a
per-fold striker bias correction  err ~ a + b*box_ratio  on the fold's TRAIN strikers
(leakage-safe), shift the held-out strikers' predictive samples by -(a + b*box_ratio), then
calibrate. Reports Brier + reliability for uncorrected vs corrected, overall and strikers-only.

Saves models/calibrator_striker.pkl (does NOT touch production models/calibrator.pkl).
Run: PM_CORES=4 PYTHONPATH=. python3 scripts/calibrate_striker.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.features import build, fit_style_clusters, team_match_table, load_corpus
from src.minutes_model import MinutesModel
from src.predict import posterior_predictive
from src.rate_model import HierNB
from src.splits import walk_forward

PCTL = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
SKIP = {"WC Qualifier", "Intl Friendly"}


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _striker_correction(model, ftr, seed):
    """Fit err ~ a + b*box_ratio on TRAIN strikers (in-sample). Returns (a,b) or None."""
    st = ftr[(ftr.role == "ST") & ftr.box_ratio_asof.notna() & (ftr.minutes > 0)]
    if len(st) < 40:
        return None
    s = posterior_predictive(model, None, st, np.full(len(st), 1.0), n_draws=400,
                             use_actual_minutes=True, seed=seed)
    err = s.mean(1) - st.passes_attempted.values          # + = over-predict
    x = st.box_ratio_asof.values
    A = np.column_stack([np.ones(len(x)), x])
    (a, b), *_ = np.linalg.lstsq(A, err, rcond=None)
    return float(a), float(b)


def main():
    cfg = Config.load()
    pm = load_corpus(cfg)
    # accumulate calibration pairs for both variants + a striker mask
    U = dict(p=[], y=[], st=[])   # uncorrected
    C = dict(p=[], y=[], st=[])   # corrected
    for fold in walk_forward(pm, by="competition"):
        if fold.name in SKIP:
            continue
        train, test = fold.train, fold.test
        km = fit_style_clusters(team_match_table(train), cfg["features"]["opponent_style_clusters"])[1]
        feats = build(pd.concat([train, test]), cfg, style_map=km)
        ftr = feats[feats.match_id.isin(train.match_id) & (feats.minutes > 0)]
        fte = feats[feats.match_id.isin(test.match_id) & feats.started.fillna(False) & (feats.minutes > 0)]
        model = HierNB(cfg).fit(ftr)
        seed = cfg["model"]["seed"]
        samples = posterior_predictive(model, None, fte, np.full(len(fte), 1.0),
                                       n_draws=1000, use_actual_minutes=True, seed=seed)
        y = fte.passes_attempted.values
        is_st = (fte.role == "ST").values
        # striker correction estimated on TRAIN, applied to TEST strikers
        coef = _striker_correction(model, ftr, seed)
        corr = np.zeros(len(fte))
        if coef is not None:
            a, b = coef
            box = pd.to_numeric(fte.box_ratio_asof, errors="coerce").fillna(0.0).values
            corr = np.where(is_st, a + b * box, 0.0)         # per-row bias estimate
        samples_c = np.clip(samples - corr[:, None], 0, None)

        thr_u = np.percentile(samples, PCTL, axis=1).T
        thr_c = np.percentile(samples_c, PCTL, axis=1).T
        for i in range(len(fte)):
            for t in thr_u[i]:
                U["p"].append(float((samples[i] > t).mean())); U["y"].append(int(y[i] > t)); U["st"].append(bool(is_st[i]))
            for t in thr_c[i]:
                C["p"].append(float((samples_c[i] > t).mean())); C["y"].append(int(y[i] > t)); C["st"].append(bool(is_st[i]))
        cs = "on" if coef is not None else "OFF(no box data)"
        print(f"[calib] fold {fold.name}: {len(fte)} starters ({is_st.sum()} ST), correction {cs}"
              + (f" a={coef[0]:+.2f} b={coef[1]:+.1f}" if coef else ""), flush=True)

    def rep(D, label):
        p, y, st = np.array(D["p"]), np.array(D["y"]), np.array(D["st"])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        print(f"\n=== {label} ===")
        print(f"  Brier raw {brier(p,y):.4f} -> cal {brier(iso.predict(p),y):.4f}   (n={len(p)})")
        print(f"  STRIKERS raw {brier(p[st],y[st]):.4f} -> cal {brier(iso.predict(p[st]),y[st]):.4f}   (n={st.sum()})")
        print("  striker reliability (raw P -> empirical):")
        for lo in [0.5, 0.6, 0.7, 0.8]:
            m = st & (p >= lo) & (p < lo + 0.1)
            if m.sum():
                print(f"    {lo:.0%}-{lo+0.1:.0%}: empirical {y[m].mean():.0%}  (n={m.sum()})")
        return iso

    rep(U, "UNCORRECTED (baseline)")
    iso_c = rep(C, "WITH striker box_ratio correction")
    outp = cfg.path("models") / "calibrator_striker.pkl"
    with open(outp, "wb") as fh:
        pickle.dump(iso_c, fh)
    print(f"\nSaved corrected calibrator -> {outp}  (production calibrator.pkl untouched)")


if __name__ == "__main__":
    main()
