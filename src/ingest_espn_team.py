"""All-nations ESPN team-level ingestion (possession / passes / passes-allowed).

ESPN exposes team-level Possession and Passes for every international match across
competitions (WC, friendlies, all confederation WC qualifiers, Nations Leagues) —
verified for friendlies + qualifiers, not just the World Cup. This pulls them
UNIFORMLY for all nations so the team features (SoS, possession, opp passes allowed)
are computed the same way for every team — no per-nation special-casing.

Output (data/raw/espn_team_match.parquet), one row per (match, team):
    date, competition, team, opponent, possession, team_passes, opp_passes_allowed

Incremental: cached per-event JSON is never re-fetched.
"""
from __future__ import annotations

import gzip
import json
import ssl
import time
import urllib.request
from pathlib import Path

# This Mac's Python lacks CA certs for urllib (curl works, urllib doesn't). ESPN is a
# public read-only API, so use an unverified context rather than failing every fetch.
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

import pandas as pd

from .config import Config

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# International competitions that, together, cover essentially every nation.
LEAGUES = [
    "fifa.world", "fifa.friendly",
    "fifa.worldq.uefa", "fifa.worldq.conmebol", "fifa.worldq.concacaf",
    "fifa.worldq.afc", "fifa.worldq.caf", "fifa.worldq.ofc",
    "uefa.nations", "concacaf.nations",
]


def _get(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
        raw = urllib.request.urlopen(req, timeout=25, context=_SSL).read()
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass
        return json.loads(raw)
    except Exception:
        return None


def _months(start: str, end: str):
    s = pd.Period(start, "M"); e = pd.Period(end, "M")
    p = s
    while p <= e:
        yield p.strftime("%Y%m")
        p += 1


def _event_ids(league: str, start="2024-06", end="2026-06") -> list[str]:
    ids = []
    for ym in _months(start, end):
        d = _get(f"{BASE}/{league}/scoreboard?dates={ym}01-{ym}28")
        if not d:
            continue
        for e in d.get("events", []):
            comp = e.get("competitions", [{}])[0]
            if comp.get("status", {}).get("type", {}).get("completed"):
                ids.append(e["id"])
        time.sleep(0.2)
    return list(dict.fromkeys(ids))


def _compact(summary: dict) -> dict:
    """Extract the bits we need (team stats + goals + date) into a small cacheable dict."""
    teams = summary.get("boxscore", {}).get("teams", [])
    out = {"date": None, "teams": {}}
    for t in teams:
        nm = t.get("team", {}).get("displayName")
        s = {x.get("label"): x.get("displayValue") for x in t.get("statistics", [])}
        out["teams"][nm] = {"poss": s.get("Possession"), "passes": s.get("Passes")}
    # goals + date from the header competitors
    comp = (summary.get("header", {}).get("competitions") or [{}])[0]
    out["date"] = comp.get("date")
    for c in comp.get("competitors", []):
        nm = c.get("team", {}).get("displayName")
        if nm in out["teams"]:
            try:
                out["teams"][nm]["goals"] = float(c.get("score"))
            except (TypeError, ValueError):
                out["teams"][nm]["goals"] = None
    return out


def _team_stats(league: str, eid: str, cache: Path) -> list[dict]:
    f = cache / f"{league}_{eid}.json"
    if f.exists():
        d = json.loads(f.read_text())
    else:
        summary = _get(f"{BASE}/{league}/summary?event={eid}")
        if summary is None:
            return []
        d = _compact(summary)
        f.write_text(json.dumps(d))
    parsed = d.get("teams", {})
    if len(parsed) != 2:
        return []
    rows = []
    names = list(parsed)
    for i, nm in enumerate(names):
        opp = names[1 - i]
        try:
            poss = float(str(parsed[nm].get("poss", "")).replace("%", ""))
            tp = float(parsed[nm].get("passes"))
            op = float(parsed[opp].get("passes"))
        except (TypeError, ValueError):
            continue
        gf, ga = parsed[nm].get("goals"), parsed[opp].get("goals")
        rows.append({"competition": league, "date": d.get("date"), "team": nm, "opponent": opp,
                     "possession": poss, "team_passes": tp, "opp_passes_allowed": op,
                     "goals_for": gf, "goals_against": ga,
                     "margin": (gf - ga) if (gf is not None and ga is not None) else None})
    return rows


def build(cfg: Config, start="2024-06", end="2026-06") -> pd.DataFrame:
    cache = cfg.path("raw") / "espn_summaries"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for lg in LEAGUES:
        ids = _event_ids(lg, start, end)
        print(f"[espn] {lg}: {len(ids)} completed matches", flush=True)
        for j, eid in enumerate(ids):
            rows.extend(_team_stats(lg, eid, cache))
            if not (cache / f"{lg}_{eid}.json").exists():
                time.sleep(0.25)
        print(f"[espn] {lg}: cumulative team-rows {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    out = cfg.path("raw") / "espn_team_match.parquet"
    df.to_parquet(out, index=False)
    print(f"[espn] wrote {len(df)} team-match rows ({df.team.nunique()} nations) -> {out}", flush=True)
    return df


def compute_sos(df: pd.DataFrame, margin_cap: float = 3.0, poss_weight: float = 0.04,
                iters: int = 60) -> pd.DataFrame:
    """Opponent-adjusted team strength + strength-of-schedule.

    Per match, a team's performance = capped goal margin + a possession-dominance term
    (who they beat, by how much, with how much control). Strength is then solved SRS-style
    so that rating_i = mean(performance_i + rating_of_opponent) — beating a strong team
    counts more than beating a weak one. SoS_i = average rating of opponents faced.
    """
    import numpy as np
    d = df.dropna(subset=["margin", "possession"]).copy()
    d["perf"] = d["margin"].clip(-margin_cap, margin_cap) + poss_weight * (d["possession"] - 50.0)
    teams = sorted(set(d.team) | set(d.opponent))
    rating = {t: 0.0 for t in teams}
    for _ in range(iters):
        new = {}
        for t in teams:
            g = d[d.team == t]
            new[t] = (g["perf"] + g["opponent"].map(rating)).mean() if len(g) else rating[t]
        m = np.nanmean(list(new.values()))
        rating = {t: (0.0 if np.isnan(v) else v) - m for t, v in new.items()}
    rows = []
    for t in teams:
        g = d[d.team == t]
        sos = g["opponent"].map(rating).mean() if len(g) else 0.0
        rows.append({"team": t, "games": len(g), "rating": rating[t], "sos": sos})
    return pd.DataFrame(rows).round(3).sort_values("rating", ascending=False)


def team_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-nation averages: possession, passing volume, and passes ALLOWED (SoS proxy)."""
    return (df.groupby("team")
            .agg(games=("team_passes", "size"), avg_possession=("possession", "mean"),
                 avg_passes=("team_passes", "mean"), avg_allowed=("opp_passes_allowed", "mean"))
            .round(1).sort_values("avg_possession", ascending=False))


if __name__ == "__main__":
    import sys
    cfg = Config.load()
    df = build(cfg)
    print(team_profiles(df).head(20).to_string())
