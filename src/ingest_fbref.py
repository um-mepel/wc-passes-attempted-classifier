"""FBref ingestion for international friendlies + WC qualifiers (form / minutes).

⚠️ Post-2026-01-20 FBref no longer serves per-player PASSING. So this pulls
minutes / starts / lineups / cards, which feed the stage-1 minutes model and
recency features — NOT the pass label. If you later mirror a paid feed that has
pass counts, flip `fbref.passing_available: true` in config and extend this to
emit passes_attempted (with provider='opta', handled by the model's p_provider).

Output (data/raw/fbref_player_match.parquet): match_id, match_date, competition,
provider, team, opponent, player_id, player, position, minutes, started,
passes_attempted (NaN unless passing_available).
"""
from __future__ import annotations

import time

import pandas as pd

from .config import Config


def build_player_match(cfg: Config) -> pd.DataFrame:
    if not cfg["fbref"].get("enabled", False):
        return pd.DataFrame()
    import soccerdata as sd  # lazy import

    pause = cfg["fbref"].get("rate_limit_seconds", 5)
    has_passing = cfg["fbref"].get("passing_available", False)
    frames = []
    for comp in cfg.fbref_competitions:
        fb = sd.FBref(leagues=comp["name"], seasons="2026")
        time.sleep(pause)
        try:
            summary = fb.read_player_match_stats(stat_type="summary")
        except Exception as e:  # FBref is brittle; log and continue
            print(f"[fbref] {comp['name']}: {e}")
            continue
        summary = summary.reset_index()
        summary["competition"] = comp["name"]
        summary["provider"] = comp.get("provider", "opta")
        summary["passes_attempted"] = float("nan")
        if has_passing:
            time.sleep(pause)
            try:
                passing = fb.read_player_match_stats(stat_type="passing").reset_index()
                summary = summary.merge(
                    passing[["game", "player", "Att"]].rename(columns={"Att": "passes_attempted"}),
                    on=["game", "player"], how="left", suffixes=("", "_p"))
            except Exception as e:
                print(f"[fbref] passing for {comp['name']}: {e}")
        frames.append(summary)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out.to_parquet(cfg.path("raw") / "fbref_player_match.parquet", index=False)
    return out
