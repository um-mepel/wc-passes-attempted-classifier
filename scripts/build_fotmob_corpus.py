"""Pull WC 2026 per-player passes from Fotmob and merge into the training corpus.

Our StatsBomb corpus stops at 2024 — so 2026 roles are stale (Gueye averages 38 there
but passes 94 now) and some players are fragmented across IDs. This pulls every finished
WC 2026 match, **consolidates each Fotmob player onto the matching StatsBomb player_id by
name+team** (so their 2026 games attach to their history and lift the recent-rate anchor),
and writes data/raw/combined_player_match.parquet for retraining.

Run:  python3 scripts/build_fotmob_corpus.py            # pull + merge (cached)
      python3 scripts/build_fotmob_corpus.py --refresh  # re-pull from Fotmob
"""
from __future__ import annotations

import datetime as _dt
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.ingest_fotmob import FotmobTraining


def _expand(*ranges) -> list[str]:
    out = []
    for a, b in ranges:
        d, end = _dt.date.fromisoformat(a), _dt.date.fromisoformat(b)
        while d <= end:
            out.append(d.strftime("%Y%m%d"))
            d += _dt.timedelta(days=1)
    return out


# FIFA international windows (friendlies + qualifiers) through the WC — recent national-
# team form with per-player passes that no free source gave before.
DATES = _expand(
    ("2025-03-17", "2025-03-26"), ("2025-06-02", "2025-06-11"), ("2025-09-01", "2025-09-10"),
    ("2025-10-06", "2025-10-15"), ("2025-11-10", "2025-11-20"),
    ("2026-03-23", "2026-04-01"), ("2026-06-01", "2026-06-27"))


def _norm(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower().strip()


def _build_name_index(pm: pd.DataFrame) -> dict:
    """(team_norm, name_norm) and (team_norm, first, last) -> StatsBomb player_id."""
    idx = {}
    for pid, sub in pm.groupby("player_id"):
        tm, nmm = sub["team"].mode(), sub["player"].mode()
        if tm.empty or nmm.empty:
            continue
        team, nm = _norm(tm.iloc[0]), _norm(nmm.iloc[0])
        parts = nm.split()
        idx[(team, nm)] = pid
        if len(parts) >= 2:
            idx.setdefault((team, parts[0], parts[-1]), pid)   # first+last (handles middle names)
            idx.setdefault((team, parts[-1]), pid)             # last-only, lowest priority
    return idx


def _resolve(idx: dict, team: str, name: str):
    t, nm = _norm(team), _norm(name)
    parts = nm.split()
    if (t, nm) in idx:
        return idx[(t, nm)], False
    if len(parts) >= 2 and (t, parts[0], parts[-1]) in idx:
        return idx[(t, parts[0], parts[-1])], False
    if len(parts) >= 2 and (t, parts[-1]) in idx:           # surname-only: accept but flag
        return idx[(t, parts[-1])], False
    return None, True


def main():
    cfg = Config.load()
    raw = cfg.path("raw")
    cache = raw / "fotmob_wc2026.parquet"
    refresh = "--refresh" in sys.argv

    if cache.exists() and not refresh:
        fm = pd.read_parquet(cache)
        print(f"[cache] {len(fm)} Fotmob rows from {cache}")
    else:
        fm = FotmobTraining(cache_dir=raw / "fotmob_cache").wc_training_rows(DATES)
        fm.to_parquet(cache, index=False)
        print(f"[fotmob] pulled {len(fm)} rows -> {cache}")
    # derive type flags from competition if an older cache lacks them (avoids re-pull)
    if "is_qualifier" not in fm.columns:
        fm["is_qualifier"] = (fm["competition"] == "WC Qualifier").astype(int)
    if "is_friendly" not in fm.columns:
        fm["is_friendly"] = (fm["competition"] == "Intl Friendly").astype(int)
    fm = fm[fm.passes_attempted.notna() & fm.team.notna() & fm.opponent.notna()].copy()
    # one match per (date, teams): drop dup player rows
    fm = fm.drop_duplicates(["match_id", "fm_player_id"])

    pm = pd.read_parquet(raw / "sb_player_match.parquet")
    idx = _build_name_index(pm)
    sb_pos = pm[pm.position.notna()].groupby("player_id")["position"].agg(lambda s: s.mode().iloc[0])

    pids, newcaps, positions = [], [], []
    for _, r in fm.iterrows():
        pid, new = _resolve(idx, r.team, r.player)
        if pid is None:                                     # stable synthetic id for new caps
            pid = 900_000_000 + int(r.fm_player_id)
            pos = r.position
        else:
            pos = sb_pos.get(pid, r.position)               # prefer StatsBomb modal position
        pids.append(pid); newcaps.append(new); positions.append(pos)
    fm["player_id"], fm["matched"], fm["position"] = pids, [not n for n in newcaps], positions

    matched = fm["matched"].mean()
    print(f"[merge] {len(fm)} Fotmob rows | {matched:.0%} matched to StatsBomb players "
          f"({(~fm['matched']).sum()} new caps)")

    out = pd.DataFrame({
        "player_id": fm["player_id"], "player": fm["player"], "team": fm["team"],
        "passes_attempted": fm["passes_attempted"].astype(float), "passes_completed": np.nan,
        "position": fm["position"], "minutes": fm["minutes"].fillna(0).astype(float),
        "started": fm["minutes"].fillna(0) >= 45, "opponent": fm["opponent"],
        "match_id": fm["match_id"].astype(int) + 70_000_000, "match_date": pd.to_datetime(fm["match_date"]),
        "competition": fm["competition"], "season": fm["competition"], "provider": "fotmob",
        "has_360": False, "stage": fm["stage"], "depth": fm["depth"],
        "is_friendly": fm["is_friendly"].astype(int), "is_qualifier": fm["is_qualifier"].astype(int)})

    # StatsBomb rows are all competitive tournament football (never friendlies/qualifiers)
    combined = pd.concat([pm.assign(stage="unknown", depth=np.nan, is_friendly=0, is_qualifier=0),
                          out], ignore_index=True)
    print(f"[type] {int(out.is_friendly.sum())} friendly rows, "
          f"{int(out.is_qualifier.sum())} qualifier rows, "
          f"{int((out.competition == 'World Cup 2026').sum())} WC rows")
    combined = combined.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    cpath = raw / "combined_player_match.parquet"
    combined.to_parquet(cpath, index=False)
    print(f"[done] wrote {len(combined)} rows ({len(pm)} StatsBomb + {len(out)} Fotmob) -> {cpath}")
    # quick peek: Gueye's recent volume now visible?
    g = combined[(combined.provider == "fotmob") & combined.player.str.contains("Gueye", case=False, na=False)]
    if len(g):
        print("\nGueye Fotmob rows now in corpus:")
        print(g[["player", "team", "minutes", "passes_attempted", "opponent"]].to_string(index=False))


if __name__ == "__main__":
    main()
