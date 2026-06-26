"""Train/test separation guarantees."""
from __future__ import annotations

import pandas as pd
import pytest

from src.splits import assert_disjoint, walk_forward


def test_folds_are_chronological_and_disjoint(corpus):
    folds = walk_forward(corpus, by="competition", min_train_tournaments=1)
    assert len(folds) >= 1
    for f in folds:
        # every training match finished strictly before the fold's first kickoff
        assert f.train["match_date"].max() < f.cutoff
        assert f.test["match_date"].min() >= f.cutoff
        # no row overlap (raises if violated)
        assert_disjoint(f.train, f.test)


def test_no_player_match_row_in_both_sides(corpus):
    for f in walk_forward(corpus, by="competition", min_train_tournaments=1):
        train_keys = set(map(tuple, f.train[["match_id", "player_id"]].values))
        test_keys = set(map(tuple, f.test[["match_id", "player_id"]].values))
        assert train_keys.isdisjoint(test_keys)


def test_assert_disjoint_catches_a_planted_leak(corpus):
    folds = walk_forward(corpus, by="competition", min_train_tournaments=1)
    f = folds[0]
    leaked_train = pd.concat([f.train, f.test.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="LEAK"):
        assert_disjoint(leaked_train, f.test)


def test_assert_disjoint_catches_a_future_dated_train_row(corpus):
    folds = walk_forward(corpus, by="competition", min_train_tournaments=1)
    f = folds[0]
    # a train row dated after the test fold should trip the temporal check
    future = f.train.iloc[[0]].copy()
    future["match_date"] = f.test["match_date"].max() + pd.Timedelta(days=1)
    future["match_id"] = -999
    bad_train = pd.concat([f.train, future], ignore_index=True)
    with pytest.raises(AssertionError):
        assert_disjoint(bad_train, f.test)
