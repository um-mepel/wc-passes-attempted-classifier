"""As-of-date team features from ESPN, joined to any match (training or upcoming).

Two complementary signals, both strictly as-of-date (no look-ahead):
  • Elo strength from match RESULTS+margins — full coverage (every nation/era).
  • Recent average POSSESSION from ESPN team stats — used wherever ESPN has it
    (all major tournaments + recent internationals); NaN where unavailable.

attach() adds, for each (team, opponent, date) row:
    team_elo, opp_elo, team_poss_espn, opp_poss_espn
computed only from matches strictly BEFORE that date.
"""
from __future__ import annotations

import math
import unicodedata
from bisect import bisect_left

import pandas as pd

# StatsBomb -> ESPN name reconciliation for the join (only where they differ).
NAME_MAP = {
    "Cabo Verde": "Cape Verde", "South Korea": "Korea Republic", "Korea": "Korea Republic",
    "United States": "USA", "China PR": "China", "Ivory Coast": "Cote d'Ivoire",
    "Czech Republic": "Czechia", "IR Iran": "Iran", "Republic of Ireland": "Ireland",
    # FIFA-2002 / corpus spellings -> the spelling ESPN results use (so seeds, results,
    # and prediction rows all normalise to one key).
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "FYR Macedonia": "North Macedonia", "Macedonia": "North Macedonia",
    "Türkiye": "Turkey", "Turkiye": "Turkey",
    "Yugoslavia": "Serbia", "Serbia & Montenegro": "Serbia", "Serbia and Montenegro": "Serbia",
}


def _norm(s: str) -> str:
    s = NAME_MAP.get(str(s), str(s))
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower().strip()


def compute_elo(results: pd.DataFrame, k: float = 30.0, hfa: float = 55.0) -> dict[str, list]:
    """Return {team_norm: [(date, rating_after), ...]} from chronological results.
    Margin-of-victory weighted; cold start 1500."""
    rating: dict[str, float] = {}
    timeline: dict[str, list] = {}
    for r in results.sort_values("date").itertuples(index=False):
        h, a = _norm(r.home), _norm(r.away)
        Rh, Ra = rating.get(h, 1500.0), rating.get(a, 1500.0)
        Eh = 1.0 / (1.0 + 10 ** ((Ra - (Rh + hfa)) / 400.0))
        gd = (r.home_goals or 0) - (r.away_goals or 0)
        Sh = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        mult = math.log(abs(gd) + 1) or 1.0
        delta = k * mult * (Sh - Eh)
        rating[h], rating[a] = Rh + delta, Ra - delta
        timeline.setdefault(h, []).append((r.date, rating[h]))
        timeline.setdefault(a, []).append((r.date, rating[a]))
    return timeline


def _asof(timeline: dict, team: str, date) -> float:
    seq = timeline.get(_norm(team))
    if not seq:
        return 1500.0
    dates = [d for d, _ in seq]
    i = bisect_left(dates, pd.Timestamp(date))
    return seq[i - 1][1] if i > 0 else 1500.0


def _poss_lookup(poss: pd.DataFrame):
    """Pre-index possession rows by normalized team, sorted by date, for as-of avg."""
    idx = {}
    p = poss.dropna(subset=["possession"]).copy()
    p["tn"] = p["team"].map(_norm)
    p["date"] = pd.to_datetime(p["date"]).dt.tz_localize(None)
    for tn, g in p.sort_values("date").groupby("tn"):
        idx[tn] = (list(g["date"]), list(g["possession"]))
    return idx


def _poss_asof(idx, team, date, window=10):
    seq = idx.get(_norm(team))
    if not seq:
        return float("nan")
    dates, vals = seq
    i = bisect_left(dates, pd.Timestamp(date))
    past = vals[max(0, i - window):i]
    return sum(past) / len(past) if past else float("nan")


def attach(df: pd.DataFrame, results: pd.DataFrame, poss: pd.DataFrame, elo=None) -> pd.DataFrame:
    """Add team_elo/opp_elo/team_poss_espn/opp_poss_espn (as-of-date) to df.
    df needs columns: team, opponent, match_date. Pass `elo` (a precomputed
    {norm_team: [(date, rating), ...]} timeline, e.g. from elo_major) to override the
    default all-results Elo."""
    if elo is None:
        elo = compute_elo(results)
    pidx = _poss_lookup(poss)
    out = df.copy()
    d = pd.to_datetime(out["match_date"])
    out["team_elo"] = [_asof(elo, t, dt) for t, dt in zip(out["team"], d)]
    out["opp_elo"] = [_asof(elo, o, dt) for o, dt in zip(out["opponent"], d)]
    out["team_poss_espn"] = [_poss_asof(pidx, t, dt) for t, dt in zip(out["team"], d)]
    out["opp_poss_espn"] = [_poss_asof(pidx, o, dt) for o, dt in zip(out["opponent"], d)]
    return out
