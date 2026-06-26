"""All-nations international RESULTS (scores only) from ESPN, for as-of-date Elo.

Results (goals) are available for every international match on ESPN — including old
friendlies that lack possession — so this is the right basis for a leakage-safe
team-strength feature. Lightweight: reads only scoreboards (no per-match summaries).

Output (data/raw/espn_results.parquet): date, competition, home, away, home_goals,
away_goals. Incremental per (league, month) cache.
"""
from __future__ import annotations

import gzip
import json
import ssl
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .config import Config
from .ingest_espn_team import LEAGUES, BASE, UA

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

# add the historical continental tournaments so pre-2024 strength is well-anchored
ALL_LEAGUES = LEAGUES + ["uefa.euro", "conmebol.america", "caf.nations", "afc.asian.cup"]


def _get(url):
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


def _months(start, end):
    p, e = pd.Period(start, "M"), pd.Period(end, "M")
    while p <= e:
        yield p.strftime("%Y%m")
        p += 1


def build(cfg: Config, start="2015-01", end="2026-06") -> pd.DataFrame:
    cache = cfg.path("raw") / "espn_results_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for lg in ALL_LEAGUES:
        n0 = len(rows)
        for ym in _months(start, end):
            f = cache / f"{lg}_{ym}.json"
            if f.exists():
                events = json.loads(f.read_text())
            else:
                d = _get(f"{BASE}/{lg}/scoreboard?dates={ym}01-{ym}28")
                events = []
                for e in (d or {}).get("events", []):
                    comp = e.get("competitions", [{}])[0]
                    if not comp.get("status", {}).get("type", {}).get("completed"):
                        continue
                    cs = comp.get("competitors", [])
                    rec = {"date": comp.get("date"), "competition": lg}
                    ok = True
                    for c in cs:
                        side = "home" if c.get("homeAway") == "home" else "away"
                        rec[side] = c.get("team", {}).get("displayName")
                        try:
                            rec[f"{side}_goals"] = int(c.get("score"))
                        except (TypeError, ValueError):
                            ok = False
                    if ok and "home" in rec and "away" in rec:
                        events.append(rec)
                f.write_text(json.dumps(events))
                time.sleep(0.15)
            rows.extend(events)
        print(f"[results] {lg}: +{len(rows)-n0} matches (total {len(rows)})", flush=True)
    df = pd.DataFrame(rows).dropna(subset=["home", "away"]).drop_duplicates(["date", "home", "away"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    out = cfg.path("raw") / "espn_results.parquet"
    df.to_parquet(out, index=False)
    print(f"[results] wrote {len(df)} matches, {pd.unique(df[['home','away']].values.ravel()).size} nations -> {out}", flush=True)
    return df


if __name__ == "__main__":
    build(Config.load())
