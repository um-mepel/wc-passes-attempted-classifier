"""Binary-search the feature coefficients, tuned toward the World Cup.

The hierarchical model fits one global set of feature coefficients across ALL
tournaments. But we only care about the World Cup, where passing dynamics differ
(tougher fields, more cautious knockouts). This holds the fitted player/role/position
effects FIXED as an offset, then coordinate-wise binary-searches (golden-section) each
feature coefficient to minimise pass-prediction error on the World Cup specifically.

Honest split: TUNE on WC 2018, VALIDATE on WC 2022 (both World Cups, no peeking).
Uses actual minutes (the confirmed-starter regime we actually bet).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.features import build, fit_style_clusters, team_match_table
from src.rate_model import HierNB, _FEATURES


def offset_and_X(model: HierNB, fu: pd.DataFrame):
    """Return (offset_logmu_without_beta, X, y) using posterior-mean random effects."""
    d = model._design(fu, training=False)
    post = model.idata.posterior

    def pm(n):
        return post[n].mean(("chain", "draw")).values

    def g(arr, idx):
        idx = np.asarray(idx)
        return np.where(idx < 0, 0.0, arr[np.clip(idx, 0, len(arr) - 1)])

    off = (np.log(d["minutes"])
           + g(pm("a_role"), np.where(d["role"] < 0, 0, d["role"]))
           + g(pm("b_position"), d["position"])
           + g(pm("a_player"), d["player_id"])
           + g(pm("s_pstyle"), d["pstyle"])
           + g(pm("t_comp"), np.where(d["competition"] < 0, 0, d["competition"]))
           + g(pm("p_prov"), d["provider"]))
    return off, d["X"], d["y"]


def mae(beta, off, X, y):
    return float(np.abs(np.exp(off + X @ beta) - y).mean())


def golden_search(f, lo, hi, iters=40):
    """1D minimiser (golden-section ~ binary search) of f on [lo, hi]."""
    gr = (np.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    for _ in range(iters):
        if f(c) < f(d):
            hi, d = d, c
            c = hi - gr * (hi - lo)
        else:
            lo, c = c, d
            d = lo + gr * (hi - lo)
    return (lo + hi) / 2


def tune(off, X, y, beta0, sweeps=4, span=0.4):
    beta = beta0.copy()
    for _ in range(sweeps):
        for j in range(len(beta)):
            def f(v, j=j):
                b = beta.copy(); b[j] = v
                return mae(b, off, X, y)
            beta[j] = golden_search(f, beta0[j] - span, beta0[j] + span)
    return beta


def main():
    cfg = Config.load()
    pm = pd.read_parquet(cfg.path("raw") / "sb_player_match.parquet")
    model = HierNB.load(cfg, cfg.path("models") / "hiernb")
    feats = build(pm, cfg, style_map=fit_style_clusters(team_match_table(pm),
                                                         cfg["features"]["opponent_style_clusters"])[1])
    feats = feats[(feats.minutes >= 60) & feats.started]      # confirmed-starter regime

    wc18 = feats[feats.competition == "World Cup 2018"]
    wc22 = feats[feats.competition == "World Cup 2022"]
    beta0 = model.idata.posterior["beta"].mean(("chain", "draw")).values

    off_t, X_t, y_t = offset_and_X(model, wc18)
    off_v, X_v, y_v = offset_and_X(model, wc22)

    beta_wc = tune(off_t, X_t, y_t, beta0)

    print(f"Tune on WC2018 ({len(wc18)} starters), validate on WC2022 ({len(wc22)}). Actual minutes.\n")
    print(f"{'feature':24}{'learned':>9}{'WC-tuned':>10}")
    for f, a, b in zip(_FEATURES, beta0, beta_wc):
        print(f"  {f:22}{a:+9.3f}{b:+10.3f}")
    print(f"\nWC2022 validation MAE:  learned {mae(beta0, off_v, X_v, y_v):.2f}"
          f"  ->  WC-tuned {mae(beta_wc, off_v, X_v, y_v):.2f}")
    print(f"WC2018 (train) MAE:     learned {mae(beta0, off_t, X_t, y_t):.2f}"
          f"  ->  WC-tuned {mae(beta_wc, off_t, X_t, y_t):.2f}")
    np.save(cfg.path("models") / "beta_wc.npy", beta_wc)
    print(f"\nSaved WC-tuned coefficients -> {cfg.path('models')/'beta_wc.npy'}")


if __name__ == "__main__":
    main()
