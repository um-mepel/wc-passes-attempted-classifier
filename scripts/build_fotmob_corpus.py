"""Build the training corpus: StatsBomb PRIMARY, Fotmob fills the gaps.

StatsBomb (hand-coded event data) is the gold standard for the tournaments it covers
(WC 2018/2022, Euro 2020/2024, Copa America 2024, AFCON 2023). Fotmob supplies everything
StatsBomb LACKS — WC 2026, qualifiers, friendlies, Nations League, and the extra
tournaments — with per-player passes ATTEMPTED, realized possession, and scores. Each
Fotmob player is consolidated onto the matching StatsBomb player_id by name+team so a
player keeps one identity across sources; Fotmob rows for any competition StatsBomb already
covers are dropped. Writes data/raw/combined_player_match.parquet.

Run:  python3 scripts/build_fotmob_corpus.py            # build (uses caches)
      python3 scripts/build_fotmob_corpus.py --refresh  # re-pull from Fotmob (incremental)
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


# International windows. Fotmob is now the PRIMARY source, so we pull the full tournament
# history (WC/Euro/Copa/AFCON/Asian Cup finals) that StatsBomb used to supply, PLUS the
# 2025-26 friendly/qualifier/Nations-League windows for recent form. _classify_comp gates
# which leagues are kept; club matches are dropped.
DATES = _expand(
    # --- historical tournament finals (replacing the StatsBomb corpus) ---
    ("2018-06-14", "2018-07-15"),                       # World Cup 2018
    ("2021-06-11", "2021-07-11"),                       # Euro 2020 (+ Copa America 2021)
    ("2022-11-20", "2022-12-18"),                       # World Cup 2022
    ("2024-01-12", "2024-02-11"),                       # AFCON 2023 + Asian Cup 2023
    ("2024-06-14", "2024-07-15"),                       # Euro 2024 + Copa America 2024
    # --- recent form: friendlies, WC qualifiers, Nations League, WC 2026 ---
    ("2025-03-17", "2025-03-26"), ("2025-06-02", "2025-06-11"), ("2025-09-01", "2025-09-10"),
    ("2025-10-06", "2025-10-15"), ("2025-11-10", "2025-11-20"),
    ("2026-03-23", "2026-04-01"), ("2026-06-01", _dt.date.today().isoformat()))
# NOTE: the final WC-2026 window auto-extends to today() so a daily `--refresh` ingests
# newly-finished matches (Fotmob per-match cache keeps it incremental). Was hardcoded to
# 2026-06-27, which silently froze the corpus and made daily retrains no-ops.


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
    cache = raw / "fotmob_intl.parquet"            # full international history (was wc2026-only)
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
    for c in ("possession_for", "goals_for", "goals_against", "is_home"):   # older caches lack these
        if c not in fm.columns:
            fm[c] = np.nan
    fm = fm[fm.passes_attempted.notna() & fm.team.notna() & fm.opponent.notna()].copy()
    # one match per (date, teams): drop dup player rows
    fm = fm.drop_duplicates(["match_id", "fm_player_id"])
    # Fotmob's "Minutes played" is occasionally wrong (e.g. 0 min w/ 74 passes), giving
    # impossible per-90 that poisons the anchor/recency features and explodes the scaler.
    p90 = fm.passes_attempted / fm.minutes.clip(lower=1) * 90
    bad = (p90 > 150) | ((fm.minutes < 5) & (fm.passes_attempted > 8))
    print(f"[clean] dropping {int(bad.sum())} rows with corrupted minutes (impossible per-90)")
    fm = fm[~bad].copy()

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
        "is_friendly": fm["is_friendly"].astype(int), "is_qualifier": fm["is_qualifier"].astype(int),
        "possession_for": fm["possession_for"], "goals_for": fm["goals_for"],
        "goals_against": fm["goals_against"], "is_home": fm["is_home"]})

    # StatsBomb is the PRIMARY source (gold-standard hand-coded event data — more precise
    # pass counts, true positions, has_360). Keep ALL StatsBomb rows and use Fotmob only to
    # FILL GAPS: competitions StatsBomb doesn't have (WC 2026, qualifiers, friendlies,
    # Nations League, and the extra tournaments). Drop Fotmob rows for any competition
    # StatsBomb already covers. Competition-level dedup is robust (sb_player_match has ~37%
    # NULL team labels, so match/player keys are unreliable) and comp labels match by
    # construction (_classify_comp mirrors the StatsBomb names: 'World Cup 2022', etc.).
    # NOTE: recency still comes from Fotmob — StatsBomb has no 2025-26 data, so the last-2-
    # tournaments anchor for WC-2026 predictions is Fotmob-driven (staleness fix preserved).
    sb = pm.assign(stage="unknown", depth=np.nan, is_friendly=0, is_qualifier=0,
                   possession_for=np.nan, goals_for=np.nan, goals_against=np.nan, is_home=np.nan)
    sb_comps = set(sb["competition"].unique())
    fm_keep = ~out["competition"].isin(sb_comps)
    n_drop = int((~fm_keep).sum())
    out = out[fm_keep]
    print(f"[dedup] StatsBomb PRIMARY: kept {len(sb)} StatsBomb rows; dropped {n_drop} Fotmob "
          f"rows for comps StatsBomb covers. Fotmob fills gaps: {sorted(out.competition.unique())}")
    combined = pd.concat([sb, out], ignore_index=True)
    print(f"[type] {int(out.is_friendly.sum())} friendly rows, "
          f"{int(out.is_qualifier.sum())} qualifier rows, "
          f"{int((out.competition == 'World Cup 2026').sum())} WC2026 rows; "
          f"comps: {sorted(out.competition.unique())}")
    combined = combined.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    # de-conflate / de-fragment player_ids so each id maps to exactly one real person
    # (cross-source merges otherwise pool different players or split one across ids).
    from src.deconflate import clean_player_ids
    combined = clean_player_ids(combined)
    cpath = raw / "combined_player_match.parquet"
    combined.to_parquet(cpath, index=False)
    print(f"[done] wrote {len(combined)} rows ({len(sb)} StatsBomb fallback + {len(out)} Fotmob) -> {cpath}")
    # quick peek: Gueye's recent volume now visible?
    g = combined[(combined.provider == "fotmob") & combined.player.str.contains("Gueye", case=False, na=False)]
    if len(g):
        print("\nGueye Fotmob rows now in corpus:")
        print(g[["player", "team", "minutes", "passes_attempted", "opponent"]].to_string(index=False))


if __name__ == "__main__":
    main()
