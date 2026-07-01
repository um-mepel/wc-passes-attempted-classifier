"""Precompute the as-of box-touch ratio lookup used by features._attach_box_ratio.

box_ratio = touches in opposition box / total touches — a playing-STYLE axis that
separates stay-high poachers (high, ~0.11) from drop-deep link strikers (~0.05) and
build-up players (~0). Sourced from the Fotmob cache per-player stats; the model's
striker role-mean over-predicts poachers (+4 bias) and under-predicts link strikers (-3),
so a single learned slope on this feature corrects both without touching other positions.

Writes data/processed/box_ratio.parquet with columns (name, date, box_ratio_asof) where
box_ratio_asof is the leakage-safe expanding mean of the player's PAST games (shift()).
Merged into features.build() by (normalized name, date) via merge_asof.

Run: PYTHONPATH=. python3 scripts/build_box_ratio.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config
from parlay import _norm


def _gstat(pstats, name):
    """Pull a per-player stat value (total, else value) by its human-readable name."""
    for grp in pstats:
        sd = grp.get("stats")
        if isinstance(sd, dict) and name in sd:
            v = sd[name].get("stat") or {}
            return v.get("total", v.get("value"))
    return None


def main():
    cfg = Config.load()
    rows = []
    for f in glob.glob(str(cfg.path("raw") / "fotmob_cache" / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        dt = str((d.get("general", {}) or {}).get("matchTimeUTCDate", ""))[:10]
        ps = (d.get("content", {}) or {}).get("playerStats", {}) or {}
        if not dt:
            continue
        for _pid, p in ps.items():
            st = p.get("stats") or []
            tou = _gstat(st, "Touches")
            box = _gstat(st, "Touches in opposition box")
            mn = _gstat(st, "Minutes played")
            if not tou or not mn or mn < 30:      # need real minutes + touches for a stable ratio
                continue
            rows.append(dict(name=_norm(p.get("name")),
                             date=pd.Timestamp(dt).normalize(),
                             box_ratio=(box or 0) / max(tou, 1)))
    df = pd.DataFrame(rows).dropna().sort_values("date")
    # leakage-safe: each match's as-of value excludes itself (shift before expanding mean)
    df["box_ratio_asof"] = (df.groupby("name")["box_ratio"]
                            .transform(lambda s: s.shift().expanding().mean()))
    out = df[["name", "date", "box_ratio_asof"]].dropna()
    path = cfg.path("processed") / "box_ratio.parquet"
    out.to_parquet(path, index=False)
    print(f"saved {len(out)} (name,date) box_ratio_asof rows -> {path}")
    print(f"box_ratio_asof: p10={out.box_ratio_asof.quantile(.1):.3f} "
          f"median={out.box_ratio_asof.median():.3f} p90={out.box_ratio_asof.quantile(.9):.3f}")


if __name__ == "__main__":
    main()
