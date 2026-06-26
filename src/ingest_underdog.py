"""Underdog pick'em line ingestion (for backtesting and live edge).

Underdog lines aren't a public API; collect them one of two ways and append to
the CSV at config.underdog.lines_csv:
  • daily manual/scripted scrape (e.g. the Apify "Underdog Player Props Scraper"),
  • or log them yourself before kickoff.

Expected CSV columns (one row per player-prop offered):
  date, competition, team, player, stat, line, multiplier_boost
We then resolve player -> player_id by name against the StatsBomb label table.

This keeps a permanent, append-only history so the walk-forward backtest always
has every line ever posted.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config


def load_lines(cfg: Config, player_match: pd.DataFrame) -> pd.DataFrame:
    """Read the lines CSV, filter to the Passes stat, and join player_id by name."""
    if not cfg["underdog"].get("enabled", False):
        return pd.DataFrame(columns=["match_id", "player_id", "line"])
    path = Path(cfg["underdog"]["lines_csv"])
    if not path.exists():
        print(f"[underdog] no lines file at {path} yet — backtest will skip betting metrics.")
        return pd.DataFrame(columns=["match_id", "player_id", "line"])

    lines = pd.read_csv(path, parse_dates=["date"])
    lines = lines[lines["stat"].str.lower() == cfg["underdog"]["stat_name"].lower()]

    # resolve to match_id + player_id by (player name, date) against the label table
    pm = player_match.copy()
    pm["match_date"] = pd.to_datetime(pm["match_date"]).dt.normalize()
    key = pm[["match_id", "player_id", "player", "match_date"]]
    merged = lines.merge(key, left_on=["player", "date"], right_on=["player", "match_date"], how="left")
    resolved = merged.dropna(subset=["match_id", "player_id"])
    if len(resolved) < len(lines):
        print(f"[underdog] {len(lines) - len(resolved)} lines unmatched by name/date "
              "(name mismatch or match not yet ingested).")
    return resolved[["match_id", "player_id", "line"]].astype({"match_id": int, "player_id": int})
