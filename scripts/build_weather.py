"""Build data/raw/weather.parquet = (date, team, temp) from the Fotmob match cache.

Fotmob carries content.weather.temperature (°C) on most 2024+ matches AND on upcoming
fixtures (forecasts), so the same table serves training (historical) and live prediction
(WC2026 forecasts). One row per (match-day, team); features.build joins it by (match_date,
team) and a missing row is treated as neutral (no heat effect).

Re-runnable and incremental in the sense that it just re-scans the on-disk cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config


def main():
    cfg = Config.load()
    cache = cfg.path("raw") / "fotmob_cache"
    rows = []
    for f in cache.glob("*.json"):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        w = (d.get("content", {}) or {}).get("weather") or {}
        temp = w.get("temperature")
        if temp is None:
            continue
        date = (d.get("general", {}) or {}).get("matchTimeUTCDate", "")[:10]
        if not date:
            continue
        for t in d.get("header", {}).get("teams", []) or []:
            if t.get("name"):
                rows.append({"date": pd.Timestamp(date).normalize(), "team": t["name"],
                             "temp": float(temp)})
    out = pd.DataFrame(rows).drop_duplicates(["date", "team"])
    p = cfg.path("raw") / "weather.parquet"
    out.to_parquet(p, index=False)
    print(f"wrote {len(out)} (date,team) weather rows -> {p}")
    print(f"temp range {out.temp.min():.0f}-{out.temp.max():.0f}C, median {out.temp.median():.0f}")


if __name__ == "__main__":
    main()
