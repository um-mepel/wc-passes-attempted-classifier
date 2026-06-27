"""Fit an isotonic probability calibrator so model P(over) matches real frequencies.

Walk-forward (train on past tournaments, evaluate the held-out one) at ACTUAL minutes
— the confirmed-starter regime we bet. For each held-out starter we evaluate the
model's P(over t) at a grid of thresholds spanning its predictive distribution, paired
with the realized outcome (actual passes > t). Isotonic regression on those pairs maps
raw model probabilities -> calibrated probabilities, correcting over/under-confidence.

Saves models/calibrator.pkl (sklearn IsotonicRegression). parlay.py applies it to P(over).
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

PCTL = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])   # thresholds per row -> raw_P ~ 0.9..0.1


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main():
    cfg = Config.load()
    pm = load_corpus(cfg)
    raw_p, outcome = [], []

    for fold in walk_forward(pm, by="competition"):
        train, test = fold.train, fold.test
        km = fit_style_clusters(team_match_table(train), cfg["features"]["opponent_style_clusters"])[1]
        feats = build(pd.concat([train, test]), cfg, style_map=km)
        ftr = feats[feats.match_id.isin(train.match_id) & (feats.minutes > 0)]
        fte = feats[feats.match_id.isin(test.match_id) & feats.started.fillna(False) & (feats.minutes > 0)]
        model = HierNB(cfg).fit(ftr)
        samples = posterior_predictive(model, MinutesModel().fit(train), fte,
                                       np.full(len(fte), 1.0), n_draws=1000,
                                       use_actual_minutes=True, seed=cfg["model"]["seed"])
        y = fte.passes_attempted.values
        thr = np.percentile(samples, PCTL, axis=1).T            # (rows, 9)
        for i in range(len(fte)):
            for t in thr[i]:
                raw_p.append(float((samples[i] > t).mean()))
                outcome.append(int(y[i] > t))
        print(f"[calib] fold {fold.name}: {len(fte)} starters, {len(raw_p)} pairs so far", flush=True)

    raw_p, outcome = np.array(raw_p), np.array(outcome)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_p, outcome)

    print(f"\nBrier  raw {brier(raw_p, outcome):.4f}  ->  calibrated {brier(iso.predict(raw_p), outcome):.4f}")
    print("reliability (raw model P -> empirical hit rate):")
    for lo in [0.5, 0.6, 0.7, 0.8, 0.9]:
        m = (raw_p >= lo) & (raw_p < lo + 0.1)
        if m.sum():
            print(f"  model {lo:.0%}-{lo+0.1:.0%}: empirical {outcome[m].mean():.0%}  (cal -> {iso.predict([lo+0.05])[0]:.0%}, n={m.sum()})")
    with open(cfg.path("models") / "calibrator.pkl", "wb") as fh:
        pickle.dump(iso, fh)
    print(f"\nSaved -> {cfg.path('models')/'calibrator.pkl'}")


if __name__ == "__main__":
    main()
