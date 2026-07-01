"""Possession Elo — rates each team's ability to WIN THE POSSESSION BATTLE.

Unlike result Elo, the "outcome" of each match is the realized possession SHARE (0-1),
not the score. So it captures a team's possession trend *opponent-adjusted and agnostic
of results*: keeping 60% against a possession-hungry side moves the rating more than 60%
against a low-block, and a team that dominates the ball even while losing still rates high.

Its expected possession share — logistic(rating gap) — is a better, head-to-head-calibrated
input to x_poss than an opponent-blind expanding mean of own possession (corr 0.80 vs 0.74
against realized WC2026 possession).

compute_poss_elo returns the same {norm_team: [(date, rating_after), ...]} shape as
team_ratings.compute_elo, so team_ratings._asof reads it leakage-safely (only pre-date games).
"""
from __future__ import annotations

import pandas as pd


def expected_share(r_team: float, r_opp: float, s: float = 400.0) -> float:
    """Expected possession share for `team` from the rating gap (Elo logistic)."""
    return 1.0 / (1.0 + 10 ** ((r_opp - r_team) / s))


def compute_poss_elo(matches: pd.DataFrame, k: float = 120.0, s: float = 350.0,
                     seed: float = 1500.0, g: float = 0.0) -> dict[str, list]:
    """matches: one row per match, columns tn, on (normalised team/opp names), poss
    (team's realised possession share 0-1), date (sorted-able). Returns the rating timeline.

    g bakes the result-Elo strength gap into the expected share: the update becomes
    exp = logistic((r_team - r_opp + g·elo_delta) / s), where elo_delta is the as-of
    result-Elo gap (same units as the poss-Elo rating). g=0 is the strength-BLIND original.
    A modest g (~0.3) stops minnow blowouts (72% vs San Marino) from inflating the rating —
    that possession is EXPECTED given the strength gap, so it no longer moves the rating.
    Validated: one-step-ahead possession corr 0.63->0.77 overall, 0.71->0.85 in mismatch games."""
    rating: dict[str, float] = {}
    timeline: dict[str, list] = {}
    has_ed = "elo_delta" in matches.columns
    for r in matches.sort_values("date").itertuples(index=False):
        t, o = r.tn, r.on
        rt, ro = rating.get(t, seed), rating.get(o, seed)
        ed = getattr(r, "elo_delta", 0.0) if (has_ed and g) else 0.0
        ed = 0.0 if ed is None or ed != ed else float(ed)   # NaN-safe (ed!=ed catches NaN)
        gap = (rt - ro) + g * ed                        # bake in strength gap (same units)
        exp = 1.0 / (1.0 + 10 ** (-gap / s))            # expected share incl. strength
        d = k * (float(r.poss) - exp)                  # over/under-performed expected share
        rating[t], rating[o] = rt + d, ro - d          # zero-sum (shares sum to 1)
        timeline.setdefault(t, []).append((r.date, rating[t]))
        timeline.setdefault(o, []).append((r.date, rating[o]))
    return timeline
