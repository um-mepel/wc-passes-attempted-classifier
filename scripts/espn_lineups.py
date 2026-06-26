"""Pull fixtures and CONFIRMED starting XIs from ESPN's public JSON API.

ESPN posts confirmed lineups in the `rosters` field of the match summary endpoint
~1 hour before kickoff. Run this close to kickoff to get real XIs automatically
(no HTML scraping, no 403s), then feed them to the passes model.

    python3 scripts/espn_lineups.py            # show today's fixtures + lineup status
    from espn_lineups import todays_lineups     # -> {team: [(player, pos), ...]}
"""
from __future__ import annotations

import json
import urllib.request

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def fixtures() -> list[dict]:
    """Today's fixtures: [{id, home, away, status}]."""
    d = _get(f"{BASE}/scoreboard")
    out = []
    for e in d.get("events", []):
        comp = e.get("competitions", [{}])[0]
        cs = comp.get("competitors", [])
        home = next((c["team"]["displayName"] for c in cs if c.get("homeAway") == "home"), "?")
        away = next((c["team"]["displayName"] for c in cs if c.get("homeAway") == "away"), "?")
        out.append({"id": e["id"], "home": home, "away": away,
                    "status": e.get("status", {}).get("type", {}).get("description", "")})
    return out


def lineup(event_id: str) -> dict[str, list[tuple[str, str]]]:
    """Confirmed XIs for one match, {team: [(player, position_abbr), ...]} (starters only).
    Empty until ESPN posts lineups (~1h pre-kickoff)."""
    d = _get(f"{BASE}/summary?event={event_id}")
    res = {}
    for r in d.get("rosters", []):
        team = r.get("team", {}).get("displayName", "?")
        starters = []
        for p in r.get("roster", []):
            if p.get("starter"):
                starters.append((p.get("athlete", {}).get("displayName", "?"),
                                 p.get("position", {}).get("abbreviation", "")))
        if starters:
            res[team] = starters
    return res


def todays_lineups() -> dict[str, list[tuple[str, str]]]:
    """All confirmed XIs available right now across today's fixtures."""
    out = {}
    for f in fixtures():
        for team, xi in lineup(f["id"]).items():
            out[team] = xi
    return out


if __name__ == "__main__":
    fx = fixtures()
    print(f"ESPN fixtures today ({len(fx)}):")
    any_lineups = False
    for f in fx:
        lu = lineup(f["id"])
        tag = "LINEUPS POSTED" if lu else "no lineup yet"
        print(f"  [{f['status']:11}] {f['home']} vs {f['away']}  ({tag})")
        for team, xi in lu.items():
            any_lineups = True
            print(f"      {team}: " + ", ".join(f"{n} ({p})" for n, p in xi))
    if not any_lineups:
        print("\nNo confirmed XIs yet — ESPN posts them ~1h before kickoff. Re-run closer to KO.")
