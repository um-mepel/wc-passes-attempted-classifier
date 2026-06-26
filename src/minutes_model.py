"""Stage-1: expected-minutes distribution per player-match.

Passes ≈ minutes × rate, and minutes is the dominant source of variance, so we model
it explicitly as a DISTRIBUTION (not a point) and let predict.py integrate over it.

Two parts:
  • p_start  — probability the player starts (logistic on recent start rate / form)
  • minutes | start — empirical distribution, censored at 90+ (starters cluster high,
    subs low). Kept deliberately simple and data-driven; swap in a richer model later.

For backtesting on completed matches we can also use ACTUAL minutes (set use_actual=True)
to isolate the rate model's quality from minutes-projection error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MinutesModel:
    MIN_PLAYER_HIST = 3   # min past appearances to use a player's OWN minutes pool

    def __init__(self, halflife: float = 5.0):
        self.halflife = halflife
        self.start_minutes_pool: np.ndarray = np.array([90.0])
        self.sub_minutes_pool: np.ndarray = np.array([20.0])
        self.player_minutes: dict = {}   # player_id -> past minutes (as-of training)

    def fit(self, pm: pd.DataFrame) -> "MinutesModel":
        played = pm[pm["minutes"] > 0]
        self.start_minutes_pool = played.loc[played["started"], "minutes"].clip(upper=98).values
        self.sub_minutes_pool = played.loc[~played["started"], "minutes"].clip(upper=98).values
        if len(self.start_minutes_pool) == 0:
            self.start_minutes_pool = np.array([90.0])
        if len(self.sub_minutes_pool) == 0:
            self.sub_minutes_pool = np.array([20.0])
        # per-player minutes history — the empirical distribution of how long THIS
        # player actually plays (captures nailed-on starters vs rotation/sub risk),
        # which global pools cannot. Built from training rows only (as-of-date).
        self.player_minutes = {pid: g["minutes"].clip(upper=98).values
                               for pid, g in played.groupby("player_id")}
        return self

    def sample_minutes(self, p_start: np.ndarray, n: int, rng: np.random.Generator,
                       player_ids=None) -> np.ndarray:
        """Return an (len(p_start), n) array of sampled minutes.

        If player_ids are given and a player has >= MIN_PLAYER_HIST past appearances,
        sample from that player's OWN minutes distribution (player-specific). Otherwise
        fall back to the start/sub global mixture driven by p_start.
        """
        p_start = np.asarray(p_start)
        out = np.empty((len(p_start), n))
        for i in range(len(p_start)):
            own = self.player_minutes.get(player_ids[i]) if player_ids is not None else None
            if own is not None and len(own) >= self.MIN_PLAYER_HIST:
                out[i] = rng.choice(own, size=n)
            else:
                starts = rng.random(n) < p_start[i]
                out[i] = np.where(starts, rng.choice(self.start_minutes_pool, size=n),
                                  rng.choice(self.sub_minutes_pool, size=n))
        return out

    @staticmethod
    def recent_start_prob(pm: pd.DataFrame, halflife: float = 5.0) -> pd.Series:
        """As-of-date recency-weighted P(start) per row from the player's past starts."""
        pm = pm.sort_values("match_date")
        out = {}
        hist: dict = {}
        vals = []
        for r in pm.itertuples(index=False):
            h = hist.get(r.player_id, [])
            if h:
                ages = np.arange(len(h), 0, -1)
                w = 0.5 ** (ages / halflife)
                vals.append(float(np.sum(np.array(h) * w) / np.sum(w)))
            else:
                vals.append(0.5)
            hist.setdefault(r.player_id, []).append(1.0 if r.started else 0.0)
        return pd.Series(vals, index=pm.index)
