"""Skill vs naive baseline: does the hierarchical model beat "predict the player's
own past passes"? Model-free, fast (no MCMC) — uses the SAME walk-forward folds and
the same held-out starters the model was scored on.

Baseline (as-of-date, strictly pre-cutoff): each held-out starter's predictive
distribution = empirical bootstrap of their OWN past match passes. If a player has
< MIN_HIST past matches, fall back to their position-role's past-passes pool, else
the global starter pool. This is a strong, realistic baseline.

Reports baseline CRPS per fold next to the model's CRPS (from the backtest).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.features import _role_bucket
from src.backtest import crps_sample
from src.splits import walk_forward

MIN_HIST = 3
N_DRAWS = 500

# model CRPS from the completed backtest (models/backtest.log)
MODEL_CRPS = {"World Cup 2022": 12.82, "AFCON 2023": 10.11, "Euro 2024": 13.44, "Copa America 2024": 10.84}


def main():
    cfg = Config.load()
    pm = pd.read_parquet(cfg.path("raw") / "sb_player_match.parquet")
    pm = pm[pm.minutes > 0].copy()
    pm["role"] = pm["position"].map(_role_bucket)
    rng = np.random.default_rng(0)

    print(f"{'fold (held-out)':20} {'n':>5} {'baseline':>9} {'model':>7} {'skill %':>8}")
    print("-" * 54)
    rows = []
    for fold in walk_forward(pm, by="competition"):
        cutoff = fold.cutoff
        hist = pm[pm.match_date < cutoff]                    # strictly pre-cutoff
        test = fold.test[fold.test.started].copy()           # same held-out starters as the model
        if not len(test):
            continue
        role_pool = {r: g.passes_attempted.values for r, g in hist.groupby("role")}
        global_pool = hist.passes_attempted.values

        samples = np.empty((len(test), N_DRAWS))
        for i, r in enumerate(test.itertuples(index=False)):
            past = hist.loc[hist.player_id == r.player_id, "passes_attempted"].values
            pool = past if len(past) >= MIN_HIST else role_pool.get(getattr(r, "role", None), global_pool)
            if len(pool) == 0:
                pool = global_pool
            samples[i] = rng.choice(pool, size=N_DRAWS)
        y = test.passes_attempted.values
        base = float(crps_sample(samples, y).mean())
        model = MODEL_CRPS.get(fold.name, float("nan"))
        skill = (1 - model / base) * 100 if base else float("nan")
        rows.append((fold.name, len(test), base, model, skill))
        print(f"{fold.name:20} {len(test):5d} {base:9.2f} {model:7.2f} {skill:7.1f}%")

    if rows:
        mb = np.mean([r[2] for r in rows]); mm = np.mean([r[3] for r in rows])
        print("-" * 54)
        print(f"{'MEAN':20} {'':5} {mb:9.2f} {mm:7.2f} {(1-mm/mb)*100:7.1f}%")
        print("\nskill % = how much lower the model's CRPS is vs the naive baseline.")
        print("positive => the model genuinely beats 'just use the player's past passes'.")


if __name__ == "__main__":
    main()
