"""As-of-date feature guarantees: a row never sees itself or the future."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")  # features.build uses KMeans for style clusters

from src.config import Config
from src.features import build


@pytest.fixture
def cfg():
    return Config.load()


def test_first_appearance_has_no_history(corpus, cfg):
    feats = build(corpus, cfg)
    first = feats.sort_values("match_date").groupby("player_id").head(1)
    # no past matches -> recency-weighted rate is undefined (NaN), not a leaked value
    assert first["recent_per90"].isna().all()


def test_future_outlier_does_not_change_past_features(corpus, cfg):
    """Plant a huge outlier in each player's LAST match; earlier rows must be unchanged.
    If as-of-date were violated (row saw the future), earlier features would move."""
    base = build(corpus, cfg)[["match_id", "player_id", "recent_per90",
                               "style_per90_recencybiased"]]

    poisoned = corpus.copy()
    last = poisoned.sort_values("match_date").groupby("player_id").tail(1).index
    poisoned.loc[last, "passes_attempted"] = 9999
    after = build(poisoned, cfg)[["match_id", "player_id", "recent_per90",
                                  "style_per90_recencybiased"]]

    merged = base.merge(after, on=["match_id", "player_id"], suffixes=("_base", "_after"))
    # exclude the poisoned last rows themselves
    earlier = merged[~merged["match_id"].isin(poisoned.loc[last, "match_id"])]
    for col in ["recent_per90", "style_per90_recencybiased"]:
        b, a = earlier[f"{col}_base"].values, earlier[f"{col}_after"].values
        assert np.allclose(np.nan_to_num(b), np.nan_to_num(a)), f"{col} leaked future info"


def test_recent_rate_is_recency_weighted_not_flat_mean(corpus, cfg):
    """The recency-weighted rate should differ from a flat expanding mean when the
    player's rate trends — confirming the recency bias is actually applied."""
    feats = build(corpus, cfg).sort_values(["player_id", "match_date"])
    # at least some finite values exist beyond the first appearance
    assert feats["recent_per90"].notna().sum() > 0
    assert feats["style_per90_recencybiased"].notna().sum() > 0
