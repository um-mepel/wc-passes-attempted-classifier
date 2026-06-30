"""Persistent pick ledger so we can measure real edge over time.

Records one row per graded pick in data/results/ledger.csv and reports cumulative record,
hit-rate by confidence tier, model calibration (predicted P(hit) vs realized), and an EV
estimate vs Underdog parlay break-evens.

Workflow:
  - at lock:  python3 scripts/save_lines.py YYYY-MM-DD     (snapshots the board -> lines_<date>.csv)
  - after:    python3 scripts/grade_slate.py YYYY-MM-DD    (pulls actuals, appends to ledger)
  - anytime:  python3 scripts/grade_slate.py --summary     (cumulative edge)

Idempotent: re-grading a date won't double-count (dedup on date+player+line+pick).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config
from src.ingest_fotmob import Fotmob
from parlay import _norm

LEDGER = Path(__file__).resolve().parents[1] / "data" / "results" / "ledger.csv"
COLS = ["date", "match_id", "competition", "team", "player", "line", "pick",
        "p_hit", "model", "actual", "minutes", "result"]


def load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        return pd.read_csv(LEDGER)
    return pd.DataFrame(columns=COLS)


def append_rows(rows: pd.DataFrame):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    led = load_ledger()
    key = ["date", "player", "line", "pick"]
    both = pd.concat([led, rows[COLS]], ignore_index=True).drop_duplicates(key, keep="last")
    both.to_csv(LEDGER, index=False)
    return both


def grade_match(match_id: int, date: str, competition: str, picks: pd.DataFrame) -> pd.DataFrame:
    """picks: team, player, line, pick (O/U), p_hit, model. Returns graded rows for one match."""
    fm = Fotmob()
    act = fm.match_player_passes(int(match_id))
    if not len(act):
        return pd.DataFrame(columns=COLS)
    act["n"] = act.player.map(_norm)
    out = []
    for r in picks.itertuples(index=False):
        last = _norm(str(r.player).split()[-1])
        cand = act[act.n.apply(lambda s: last in s.split())]
        if len(cand) > 1:  # disambiguate by full-name token overlap
            toks = set(_norm(str(r.player)).split())
            cand = cand.iloc[[max(range(len(cand)),
                                  key=lambda i: len(toks & set(cand.iloc[i].n.split())))]]
        if not len(cand):
            continue
        a = float(cand.sort_values("minutes").iloc[-1].passes_attempted)
        mins = float(cand.sort_values("minutes").iloc[-1].minutes)
        res = "PUSH" if a == r.line else ("WIN" if ((a > r.line) == (r.pick == "O")) else "LOSS")
        out.append(dict(date=date, match_id=int(match_id), competition=competition,
                        team=r.team, player=r.player, line=r.line, pick=r.pick,
                        p_hit=getattr(r, "p_hit", np.nan), model=getattr(r, "model", np.nan),
                        actual=a, minutes=mins, result=res))
    return pd.DataFrame(out)


def summary():
    led = load_ledger()
    g = led[led.result.isin(["WIN", "LOSS"])]
    if not len(g):
        print("ledger empty — no graded picks yet."); return
    w, l = (g.result == "WIN").sum(), (g.result == "LOSS").sum()
    hit = w / (w + l)
    print(f"=== CUMULATIVE LEDGER ({g.date.nunique()} slates, {len(g)} graded picks) ===")
    print(f"  record: {w}-{l}   hit rate: {hit:.1%}")
    # break-evens: single ~52.4% (-110); 3-leg parlay 6x -> 55.0%/leg
    print(f"  break-even: 52.4% (std -110) | 55.0%/leg (3-leg 6x parlay)")
    print(f"  edge vs 52.4%: {hit-0.524:+.1%}   vs 55.0%: {hit-0.55:+.1%}")
    print(f"  model MAE on graded picks: {(g.model-g.actual).abs().mean():.1f}")
    # calibration: did model P(hit) match realized?
    if g.p_hit.notna().any():
        b = g.dropna(subset=["p_hit"]).copy()
        b["bucket"] = pd.cut(b.p_hit, [0.5, 0.6, 0.7, 0.8, 1.01], right=False)
        print("\n  model P(hit) vs realized:")
        for bk, gg in b.groupby("bucket"):
            if len(gg):
                print(f"    {str(bk):14} predicted~{gg.p_hit.mean():.0%}  realized {(gg.result=='WIN').mean():.0%}  (n={len(gg)})")
    print("\n  by team:")
    for t, gg in g.groupby("team"):
        print(f"    {t:22} {(gg.result=='WIN').sum()}-{(gg.result=='LOSS').sum()}")


def grade_date(date: str):
    """Grade a saved slate: read data/raw/lines_<date>.csv, resolve each pick's match via
    Fotmob, pull actuals, append graded rows to the ledger."""
    cfg = Config.load()
    lp = cfg.path("raw") / f"lines_{date}.csv"
    if not lp.exists():
        print(f"no saved lines for {date} ({lp}). Run save_lines.py {date} at lock time first.")
        return
    sheet = pd.read_csv(lp)
    sheet["pick"] = sheet["pick"].str[0].str.upper()              # OVER->O, UNDER->U
    sheet = sheet.rename(columns={"pred": "model"})
    fm = Fotmob()
    ms = {m["home"]: m for m in fm.matches_on(date.replace("-", ""))}
    ms.update({m["away"]: m for m in fm.matches_on(date.replace("-", ""))})
    all_rows = []
    for team, picks in sheet.groupby("team"):
        m = ms.get(team)
        if not m or not m.get("finished"):
            print(f"  {team}: no finished match found for {date} — skipped")
            continue
        all_rows.append(grade_match(m["id"], date, "World Cup 2026", picks))
    if all_rows:
        graded = pd.concat(all_rows, ignore_index=True)
        append_rows(graded)
        print(f"appended {len(graded)} graded picks for {date}\n")
    summary()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summary()
    elif len(sys.argv) > 1:
        grade_date(sys.argv[1])
    else:
        print("use: grade_slate.py YYYY-MM-DD  |  grade_slate.py --summary")
