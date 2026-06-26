"""Shared fixtures: a small synthetic player-match corpus across three tournaments.

Deliberately dependency-light (pandas/numpy only) so the leakage guarantees can be
tested without installing pymc/statsbombpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def corpus() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    # three tournaments, chronological, overlapping players
    tournaments = [
        ("Euro 2020", "2021-06-11"),
        ("World Cup 2022", "2022-11-20"),
        ("Euro 2024", "2024-06-14"),
    ]
    players = [(1, "Alpha", "Center Midfield"), (2, "Bravo", "Right Center Back"),
               (3, "Charlie", "Left Wing"), (4, "Delta", "Center Forward")]
    mid = 1000
    for comp, start in tournaments:
        base = pd.Timestamp(start)
        for matchday in range(3):
            mid += 1
            date = base + pd.Timedelta(days=4 * matchday)
            for pid, name, pos in players:
                minutes = int(rng.integers(20, 95))
                rate = {1: 70, 2: 55, 3: 30, 4: 25}[pid]
                passes = int(rng.poisson(rate * minutes / 90))
                rows.append(dict(
                    match_id=mid, match_date=date, competition=comp, season=comp,
                    provider="statsbomb", has_360=False, team="X", opponent="Y",
                    player_id=pid, player=name, position=pos,
                    minutes=minutes, started=minutes >= 60,
                    passes_attempted=passes, passes_completed=int(passes * 0.85),
                ))
    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
