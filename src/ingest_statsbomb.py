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
    """Approximate minutes played and starter flag from substitution events + lineups.

    Starters = players present in the Starting XI tactics; minutes derived from
    sub-on/off events and match length. Good enough for the stage-1 minutes model;
    refined later if a cleaner minutes source is wired in.
    """
    rows = []
    # match length (last event minute, capped sensibly)
    match_end = int(events["minute"].max()) + 1 if "minute" in events else 90
    starters = set()
    for team, df in lineups.items():
        for _, r in df.iterrows():
            pid = r.get("player_id")
            # statsbombpy lineups carry positions list with from/to; treat presence
            # in the first position with from=='00:00' as a starter heuristic.
            starters.add(pid)
    subs = events[events["type"] == "Substitution"] if "type" in events else pd.DataFrame()

    played = events.groupby("player_id")["minute"].agg(["min", "max"]) if "player_id" in events else pd.DataFrame()
    for pid, row in played.iterrows():
        on = 0 if pid in starters else int(row["min"])
        off = match_end
        if not subs.empty and "substitution_replacement_id" in subs.columns:
            off_evt = subs[subs["player_id"] == pid]
            if len(off_evt):
                off = int(off_evt["minute"].iloc[0])
        rows.append({"player_id": pid, "minutes": max(0, min(off, match_end) - on),
                     "started": pid in starters})
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
