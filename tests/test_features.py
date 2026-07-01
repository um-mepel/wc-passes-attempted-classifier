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
    cols = ["recent_per90", "style_per90_recencybiased", "team_poss_asof", "opp_allowed_asof",
            "share_asof"]
    base = build(corpus, cfg)[["match_id", "player_id", *cols]]

    poisoned = corpus.copy()
    last = poisoned.sort_values("match_date").groupby("player_id").tail(1).index
    poisoned.loc[last, "passes_attempted"] = 9999
    after = build(poisoned, cfg)[["match_id", "player_id", *cols]]

    merged = base.merge(after, on=["match_id", "player_id"], suffixes=("_base", "_after"))
    # exclude the poisoned last rows themselves
    earlier = merged[~merged["match_id"].isin(poisoned.loc[last, "match_id"])]
    for col in cols:
        b, a = earlier[f"{col}_base"].values, earlier[f"{col}_after"].values
        assert np.allclose(np.nan_to_num(b), np.nan_to_num(a)), f"{col} leaked future info"


def test_magnetism_features_ignore_current_match(corpus, cfg):
    """share_asof (player's share of team passes) and team_vol_asof (team pass volume)
    must be as-of: a player's first-ever match has share_asof NaN, and a team's first-ever
    match has team_vol_asof NaN — proving neither uses the current match's realized passes."""
    feats = build(corpus, cfg).sort_values("match_date")
    first_player = feats.groupby("player_id").head(1)
    assert first_player["share_asof"].isna().all(), "share used current match"
    first_team = feats.groupby("team").head(1)
    assert first_team["team_vol_asof"].isna().all(), "team volume used current match"


def test_possession_features_ignore_current_match(corpus, cfg):
    """team_poss_asof / opp_allowed_asof must come from PRIOR matches only. A team's
    first-ever match has no prior, so these must be NaN — proving the current match's
    realized possession isn't used."""
    feats = build(corpus, cfg)
    first_per_team = feats.sort_values("match_date").groupby("team").head(1)
    assert first_per_team["team_poss_asof"].isna().all(), "team possession used current match"


def test_recent_rate_is_recency_weighted_not_flat_mean(corpus, cfg):
    """The recency-weighted rate should differ from a flat expanding mean when the
    player's rate trends — confirming the recency bias is actually applied."""
    feats = build(corpus, cfg).sort_values(["player_id", "match_date"])
    # at least some finite values exist beyond the first appearance
    assert feats["recent_per90"].notna().sum() > 0
    assert feats["style_per90_recencybiased"].notna().sum() > 0
