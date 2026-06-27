"""Train/test separation — enforced structurally, not by convention.

Guarantees (checked with assertions, so a leak fails loudly):
  1. Walk-forward by tournament+date: every test fold is held out ENTIRELY; the
     model for fold k trains only on matches that finished BEFORE the fold's
     first kickoff.
  2. Disjoint row-sets: no (match_id, player_id) row is ever in both train and
     test of the same fold. `assert_disjoint` verifies this.
  3. As-of-date features: feature building (features.py) is given ONLY the training
     slice's history for each fold, so no test-period information can leak in —
     including the recency-weighted player and player×style rates.

This module decides WHICH rows go where; features.py decides WHAT each row sees.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    name: str
    train: pd.DataFrame   # rows whose match_date < cutoff
    test: pd.DataFrame    # the held-out tournament/window
    cutoff: pd.Timestamp  # first kickoff of the test fold


def _key(df: pd.DataFrame) -> set:
    return set(map(tuple, df[["match_id", "player_id"]].itertuples(index=False, name=None)))


def assert_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Hard guarantee that no player-match row appears in both sides."""
    overlap = _key(train) & _key(test)
    if overlap:
        raise AssertionError(
            f"TRAIN/TEST LEAK: {len(overlap)} (match,player) rows in both sides, "
            f"e.g. {list(overlap)[:3]}"
        )
    # also forbid any test match_date earlier than the latest train date inside the fold
    if len(train) and len(test):
        assert test["match_date"].min() >= train["match_date"].max(), \
            "TRAIN/TEST LEAK: a test match predates the newest training match."


def walk_forward(df: pd.DataFrame, by: str = "competition",
                 min_train_tournaments: int = 2) -> list[Fold]:
    """Yield walk-forward folds: hold out one tournament at a time, train on all
    earlier tournaments only.

    `df` must have columns match_date, match_id, player_id, competition.
    """
    df = df.sort_values("match_date").copy()
    # order tournaments by their first kickoff
    order = (df.groupby(by)["match_date"].min().sort_values().index.tolist())

    folds: list[Fold] = []
    for i in range(min_train_tournaments, len(order)):
        test_comp = order[i]
        test = df[df[by] == test_comp]
        cutoff = test["match_date"].min()
        train = df[df["match_date"] < cutoff]          # strictly before the fold
        assert_disjoint(train, test)
        folds.append(Fold(name=str(test_comp), train=train, test=test, cutoff=cutoff))
    return folds


def final_holdout(df: pd.DataFrame, by: str = "competition") -> Fold:
    """The most recent tournament as a single untouched holdout (report metrics here)."""
    return walk_forward(df, by=by)[-1]


def chunked_holdout(df: pd.DataFrame, tournament: str, by: str = "competition",
                    chunk_days: int = 1, min_test_rows: int = 1) -> list[Fold]:
    """Live-tournament backtest: fix the base training set to everything BEFORE the
    tournament, then walk through the tournament's own matches in date-ordered chunks,
    EXPANDING the training set after each chunk (the model is refit per chunk in
    backtest.run). Fold k trains on (all pre-tournament data + every earlier chunk) and
    tests on chunk k — exactly how the model is used live during the World Cup.

    `chunk_days` groups that many distinct match-dates into one chunk (1 = a day's slate).
    Requires columns match_date, match_id, player_id, and `by`.
    """
    df = df.sort_values("match_date").copy()
    tour = df[df[by] == tournament]
    if tour.empty:
        raise ValueError(f"tournament '{tournament}' not found in corpus (column '{by}')")
    tour_day = pd.to_datetime(tour["match_date"]).dt.normalize()
    dates = sorted(tour_day.unique())
    chunks = [dates[i:i + chunk_days] for i in range(0, len(dates), chunk_days)]

    folds: list[Fold] = []
    for ci, cdates in enumerate(chunks, 1):
        test = tour[tour_day.isin(set(cdates))]
        if len(test) < min_test_rows:
            continue
        cutoff = test["match_date"].min()
        train = df[df["match_date"] < cutoff]          # pre-tournament + all earlier chunks
        assert_disjoint(train, test)
        d0, d1 = pd.Timestamp(cdates[0]).date(), pd.Timestamp(cdates[-1]).date()
        span = f"{d0}" if d0 == d1 else f"{d0}…{d1}"
        folds.append(Fold(name=f"{tournament} chunk {ci}/{len(chunks)} ({span})",
                          train=train, test=test, cutoff=cutoff))
    return folds
