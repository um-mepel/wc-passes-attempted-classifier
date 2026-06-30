"""Snapshot today's live Underdog passes board for later grading.

Pulls the current passes props, runs the model, and writes two files:
  1. data/raw/underdog_lines.csv      — APPENDED, pipeline schema
       (date, competition, team, player, stat, line, multiplier_boost)
       consumed by src.ingest_underdog.load_lines for backtest/betting metrics.
  2. data/raw/lines_<DATE>.csv         — full grading sheet for this slate
       (adds model pred, pick, P(hit), history, flags) so we can grade our calls.

Run close to lock so the saved lines match what we actually bet.
Usage: python3 scripts/save_lines.py [YYYY-MM-DD]   (default: today CST guess via arg)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.features import load_corpus
from parlay import predict, pull_passes_lines, score  # reuse the live pipeline

DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-06-29"


def main():
    cfg = Config.load()
    pm = load_corpus(cfg)
    props = pull_passes_lines()
    if props.empty:
        print("No passes props live right now — nothing saved.")
        return
    up = score(predict(cfg, pm, props))

    # --- 1. pipeline-schema CSV (append; one row per offered prop) ---
    # Use Underdog's RAW player name (props.name) so grade-time name resolution against
    # Fotmob actuals works; map it back from the matched row via the line+team key.
    raw = props.rename(columns={"name": "ud_name"})[["ud_name", "team", "line"]]
    pipe = up.merge(raw, on=["team", "line"], how="left")
    pipe_out = pd.DataFrame({
        "date": DATE,
        "competition": "World Cup 2026",
        "team": pipe["team"],
        "player": pipe["ud_name"].fillna(pipe["player"]),
        "stat": cfg["underdog"]["stat_name"],     # "Passes"
        "line": pipe["line"],
        "multiplier_boost": 1.0,
    }).drop_duplicates(["date", "team", "player", "line"])

    path = Path(cfg["underdog"]["lines_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = pd.read_csv(path)
        combined = pd.concat([prior, pipe_out], ignore_index=True).drop_duplicates(
            ["date", "team", "player", "line"], keep="last")
    else:
        combined = pipe_out
    combined.to_csv(path, index=False)
    print(f"[save] {len(pipe_out)} props -> {path} (now {len(combined)} total rows)")

    # --- 2. full grading sheet for THIS slate ---
    sheet = up.assign(date=DATE)[
        ["date", "team", "player", "line", "pred", "pick", "p_over", "p_hit",
         "hist", "is_new", "agree"]].copy()
    sheet["p_over"] = sheet["p_over"].round(3)
    sheet["p_hit"] = sheet["p_hit"].round(3)
    sheet = sheet.sort_values("p_hit", ascending=False)
    grade_path = path.parent / f"lines_{DATE}.csv"
    sheet.to_csv(grade_path, index=False)
    print(f"[save] {len(sheet)} graded calls -> {grade_path}")
    print(f"\nTop 5 by P(hit):")
    print(sheet.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
