"""Incremental StatsBomb open-data ingestion.

Pulls events + lineups for every configured men's international tournament and
builds the per-player-per-match PASS LABEL. Incremental: a match whose parquet
already exists in data/raw is never re-pulled, so retraining on new tournaments
only fetches the new matches.

Output (data/raw/sb_player_match.parquet), one row per player per match:
    match_id, match_date, competition, season, provider, has_360,
    team, opponent, player_id, player, position,
    minutes, passes_attempted, passes_completed, started
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import Config

# StatsBomb pass outcomes that mean "not completed".
_INCOMPLETE = {"Incomplete", "Out", "Pass Offside", "Injury Clearance", "Unknown"}


def _sb():
    # imported lazily so the module loads even before statsbombpy is installed
    from statsbombpy import sb
    return sb


def list_competitions() -> pd.DataFrame:
    """Print the StatsBomb competition/season catalog (to fill in config.yaml)."""
    comps = _sb().competitions()
    cols = [c for c in ["competition_id", "season_id", "competition_name",
                        "season_name", "competition_gender", "match_available_360"]
            if c in comps.columns]
    return comps[cols].sort_values(["competition_name", "season_name"])


def _match_label(match_id: int, meta: dict, cache_dir: Path) -> pd.DataFrame:
    """Build (and cache) the per-player label table for one match."""
    cache = cache_dir / f"match_{match_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    sb = _sb()
    events = sb.events(match_id=match_id)
    lineups = sb.lineups(match_id=match_id)  # dict {team: df}

    # passes attempted/completed per player
    passes = events[events["type"] == "Pass"].copy()
    if "pass_outcome" not in passes.columns:
        passes["pass_outcome"] = pd.NA
    grp = passes.groupby(["player_id", "player", "team"], dropna=True)
    label = grp.agg(passes_attempted=("id", "size")).reset_index()
    completed = (
        passes[~passes["pass_outcome"].isin(_INCOMPLETE)]
        .groupby("player_id").size().rename("passes_completed")
    )
    label = label.merge(completed, on="player_id", how="left")
    label["passes_completed"] = label["passes_completed"].fillna(0).astype(int)

    # granular position = the player's most-played position this match (from events)
    if "position" in events.columns:
        pos = (events.dropna(subset=["position"])
               .groupby("player_id")["position"]
               .agg(lambda s: s.value_counts().idxmax()).rename("position"))
        label = label.merge(pos, on="player_id", how="left")
    else:
        label["position"] = pd.NA

    # minutes + starter flag from lineups/events
    minutes = _minutes_from_events(events, lineups)
    label = label.merge(minutes, on="player_id", how="outer")
    label["passes_attempted"] = label["passes_attempted"].fillna(0).astype(int)
    label["passes_completed"] = label["passes_completed"].fillna(0).astype(int)

    teams = events["team"].dropna().unique().tolist()
    opp = {t: [o for o in teams if o != t][0] if len(teams) == 2 else pd.NA for t in teams}
    label["opponent"] = label["team"].map(opp)

    for k, v in {
        "match_id": match_id, "match_date": meta["match_date"],
        "competition": meta["competition"], "season": meta["season"],
        "provider": meta["provider"], "has_360": meta["has_360"],
    }.items():
        label[k] = v

    cache.parent.mkdir(parents=True, exist_ok=True)
    label.to_parquet(cache, index=False)
    return label


def _minutes_from_events(events: pd.DataFrame, lineups: dict) -> pd.DataFrame:
    """Accurate minutes played + starter flag from the lineups `positions` field.

    Each lineup row carries a `positions` list with clock-time `from`/`to` and a
    `start_reason`/`end_reason`. This gives exact minutes — including starters subbed
    off early or sent off (the cases that matter for predicting their pass volume).
    Only STARTERS are predicted downstream (passes-attempted props cover the XI), but
    we keep every player so the rate model trains on all observed pass counts.
    """
    match_end = int(events["minute"].max()) if "minute" in events.columns else 90

    def clock_to_min(s):
        if not s or not isinstance(s, str) or ":" not in s:
            return None
        mm, ss = s.split(":")[:2]
        return int(mm) + int(ss) / 60.0

    rows = []
    for df in lineups.values():
        for _, r in df.iterrows():
            pid = r.get("player_id")
            positions = r.get("positions") or []
            if not positions:                       # named in squad but never on pitch
                rows.append({"player_id": pid, "minutes": 0, "started": False})
                continue
            started = any(p.get("start_reason") == "Starting XI" for p in positions)
            on = clock_to_min(positions[0].get("from")) or 0.0
            last_to = positions[-1].get("to")       # null => played to the final whistle
            off = clock_to_min(last_to)
            off = match_end if off is None else off
            minutes = int(round(max(0.0, min(off, match_end) - on)))
            rows.append({"player_id": pid, "minutes": minutes, "started": started})
    return pd.DataFrame(rows)


def build_player_match(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Pull all configured tournaments incrementally and concatenate the label table."""
    os.environ.setdefault("SB_CORES", str(cfg["statsbomb"].get("sb_cores", 4)))
    sb = _sb()
    cache_dir = cfg.path("raw") / "statsbomb_matches"
    out_path = cfg.path("raw") / "sb_player_match.parquet"
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path)
    else:
        existing = pd.DataFrame()

    seen = set(existing["match_id"]) if len(existing) else set()
    frames = [existing] if len(existing) else []

    for comp in cfg.statsbomb_competitions:
        matches = sb.matches(competition_id=comp["competition_id"], season_id=comp["season_id"])
        for _, m in matches.iterrows():
            mid = int(m["match_id"])
            if mid in seen and not force:
                continue  # incremental: skip already-cached matches
            meta = {
                "match_date": m["match_date"], "competition": comp["name"],
                "season": comp["name"], "provider": comp.get("provider", "statsbomb"),
                "has_360": comp.get("has_360", False),
            }
            frames.append(_match_label(mid, meta, cache_dir))

    out = pd.concat(frames, ignore_index=True).drop_duplicates(["match_id", "player_id"])
    out["match_date"] = pd.to_datetime(out["match_date"])
    out = out.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    return out
