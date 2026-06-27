"""Auto-grade: pull Fotmob per-player passes (actuals) for finished matches and diff
them against what the model predicts, at the players' ACTUAL minutes — no manual entry.

Usage:
  python3 scripts/grade.py 20260626            # all finished WC matches that date
  python3 scripts/grade.py 20260626 4667776    # one match by Fotmob id

For each player we reconstruct the prediction row (team, opponent, position from our
StatsBomb history, actual minutes), run the live model, and compare to the Fotmob actual.
Appends every (date, match, player, model, actual, error) to data/grading_log.parquet so
the validation record builds itself over the tournament.
"""
from __future__ import annotations

import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.features import build, fit_style_clusters, team_match_table, load_corpus
from src.ingest_fotmob import Fotmob
from src.minutes_model import MinutesModel
from src.predict import posterior_predictive
from src.rate_model import HierNB


def _norm(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower().strip()


def _match_pid(pm_team: pd.DataFrame, name: str):
    """Find a StatsBomb player_id + modal position for a Fotmob name on a team."""
    nn = _norm(name)
    parts = set(nn.split())
    for _, r in pm_team.drop_duplicates("player_id").iterrows():
        pn = _norm(r.player)
        if nn == pn or nn in pn or pn in nn or (parts & set(pn.split()) and nn.split()[-1] == pn.split()[-1]):
            return r.player_id
    return None


def grade_match(fm: Fotmob, mid: int, label: str, pm, model, mins, sm, cfg) -> pd.DataFrame | None:
    act = fm.match_player_passes(mid)
    act = act[act.passes_attempted.notna() & act.minutes.notna() & act.team.notna()]
    teams = list(act.team.dropna().unique())
    if len(teams) != 2 or act.empty:
        return None
    gmid = 8_800_000 + mid % 100_000
    rows, meta = [], []
    for _, r in act.iterrows():
        opp = teams[1] if r.team == teams[0] else teams[0]
        pid = _match_pid(pm[pm.team == r.team], r.player)
        if pid is not None:
            ph = pm[(pm.player_id == pid) & pm.position.notna()]
            pos = ph.position.mode().iloc[0] if len(ph) else "Center Midfield"
            newcap = False
        else:
            pid = -(abs(hash(r.player)) % 10_000_000)
            pos, newcap = "Center Midfield", True
        rows.append(dict(match_id=gmid, match_date=pd.Timestamp("2026-06-27"), competition="WC 2026",
                         season="WC 2026", provider="statsbomb", has_360=False, team=r.team, opponent=opp,
                         player_id=pid, player=r.player, position=pos, minutes=float(r.minutes),
                         started=bool(r.minutes >= 45), passes_attempted=np.nan, passes_completed=np.nan))
        meta.append(dict(player=r.player, team=r.team, pos=pos, minutes=int(r.minutes),
                         actual=int(r.passes_attempted), pid=pid, newcap=newcap))
    up = pd.DataFrame(rows).drop_duplicates("player_id")
    fu = build(pd.concat([pm, up], ignore_index=True), cfg, style_map=sm)
    fu = fu[fu.match_id == gmid].copy()
    s = posterior_predictive(model, mins, fu, np.full(len(fu), 1.0), n_draws=1500,
                             use_actual_minutes=True, seed=1)
    pred = {pid: float(s[i].mean()) for i, pid in enumerate(fu.player_id)}
    out = []
    for m in meta:
        p = pred.get(m["pid"])
        if p is None:
            continue
        out.append({"match": label, "player": m["player"], "team": m["team"], "pos": m["pos"],
                    "min": m["minutes"], "model": round(p, 1), "actual": m["actual"],
                    "err": round(p - m["actual"], 1), "newcap": m["newcap"]})
    return pd.DataFrame(out)


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "20260626"
    only = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cfg = Config.load()
    pm = load_corpus(cfg)
    model = HierNB.load(cfg, cfg.path("models") / "hiernb")
    mins = MinutesModel().fit(pm[pm.minutes > 0])
    sm = fit_style_clusters(team_match_table(pm), cfg["features"]["opponent_style_clusters"])[1]
    fm = Fotmob()

    frames = []
    for m in fm.matches_on(date):
        if only and m["id"] != only:
            continue
        if not m["finished"] or "World" not in str(m["league"]):
            continue
        label = f"{m['home']}-{m['away']}"
        g = grade_match(fm, m["id"], label, pm, model, mins, sm, cfg)
        if g is None or g.empty:
            continue
        frames.append(g.assign(date=date, match_id=m["id"]))
        # confirmed starters (>=60 min) are the bet-eligible population
        starters = g[g["min"] >= 60]
        print(f"\n=== {label}  (MAE all {g.err.abs().mean():.1f} | starters≥60' {starters.err.abs().mean():.1f}) ===")
        print(g.sort_values("actual", ascending=False)[
            ["player", "pos", "min", "model", "actual", "err", "newcap"]].to_string(index=False))

    if frames:
        allg = pd.concat(frames, ignore_index=True)
        bet = allg[~allg.newcap & (allg["min"] >= 60)]
        print(f"\n##### OVERALL  MAE all={allg.err.abs().mean():.1f}  |  "
              f"data-backed starters≥60' (bet-eligible) MAE={bet.err.abs().mean():.1f}  "
              f"(bias {bet.err.mean():+.1f}) #####")
        logp = cfg.path("raw").parent / "grading_log.parquet"
        prior = pd.read_parquet(logp) if logp.exists() else pd.DataFrame()
        keep = pd.concat([prior, allg], ignore_index=True).drop_duplicates(["date", "match_id", "player"], keep="last")
        keep.to_parquet(logp, index=False)
        print(f"appended -> {logp}  ({len(keep)} rows total)")


if __name__ == "__main__":
    main()
