"""Pull MAJOR-tournament results (WC + continental championships) back to 2002 from
ESPN, for the seeded major-only Elo. Caches per (league, month); writes
data/raw/major_results.parquet with: date, competition, home, away, home_goals, away_goals.

Major leagues per spec: World Cup, Euro, Copa America, AFCON, Asian Cup, Gold Cup.
(Qualifiers/friendlies/Nations Leagues are deliberately excluded.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.ingest_espn_results import _get, _months
from src.ingest_espn_team import BASE

MAJOR = ["fifa.world", "uefa.euro", "conmebol.america", "caf.nations",
         "afc.asian.cup", "concacaf.gold"]


def build(start="2002-01", end="2026-06") -> pd.DataFrame:
    cfg = Config.load()
    cache = cfg.path("raw") / "espn_results_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for lg in MAJOR:
        n0 = len(rows)
        for ym in _months(start, end):
            f = cache / f"{lg}_{ym}.json"
            if f.exists():
                import json
                events = json.loads(f.read_text())
            else:
                d = _get(f"{BASE}/{lg}/scoreboard?dates={ym}01-{ym}28")
                events = []
                for e in (d or {}).get("events", []):
                    comp = e.get("competitions", [{}])[0]
                    if not comp.get("status", {}).get("type", {}).get("completed"):
                        continue
                    rec = {"date": comp.get("date"), "competition": lg}
                    ok = True
                    for c in comp.get("competitors", []):
                        side = "home" if c.get("homeAway") == "home" else "away"
                        rec[side] = c.get("team", {}).get("displayName")
                        try:
                            rec[f"{side}_goals"] = int(c.get("score"))
                        except (TypeError, ValueError):
                            ok = False
                    if ok and "home" in rec and "away" in rec:
                        events.append(rec)
                import json
                import time
                f.write_text(json.dumps(events))
                time.sleep(0.12)
            rows.extend(events)
        print(f"[major] {lg}: +{len(rows)-n0} matches (total {len(rows)})", flush=True)
    df = pd.DataFrame(rows).dropna(subset=["home", "away"]).drop_duplicates(["date", "home", "away"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    out = cfg.path("raw") / "major_results.parquet"
    df.to_parquet(out, index=False)
    nat = pd.unique(df[["home", "away"]].values.ravel()).size
    print(f"[major] wrote {len(df)} matches, {nat} nations, {df.date.min().date()}..{df.date.max().date()} -> {out}", flush=True)
    return df


if __name__ == "__main__":
    build()
