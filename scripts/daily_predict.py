"""Daily predict + grade + Google-Drive spreadsheet. Run in the afternoon pre-lock.

Deterministic core (the hybrid Claude agent supervises + can override the slate):
  1. resolve TODAY's CST slate from Fotmob kickoff times (utcTime -> America/Chicago)
  2. pull the live Underdog passes board, filter to today's teams, predict + score
  3. save the board (lines_<date>.csv + append underdog_lines.csv)
  4. grade YESTERDAY's finished games into the ledger (full 120', per-90 MAE reported)
  5. write an .xlsx to the Google Drive synced folder (auto-uploads)

Usage:
  python3 scripts/daily_predict.py                      # today CST, auto slate
  python3 scripts/daily_predict.py --date 2026-07-03    # explicit CST date
  python3 scripts/daily_predict.py --teams Argentina,Colombia,Ghana,Cape Verde,Australia,Egypt
Env:
  WCP_DRIVE_DIR  target dir for the spreadsheet (default: local data/reports/)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from src.config import Config
from src.features import load_corpus
from src.ingest_fotmob import Fotmob
from parlay import predict, pull_passes_lines, score, build_parlays
from grade_slate import grade_match, append_rows, load_ledger

CST = ZoneInfo("America/Chicago")


def slate_for(cst_day: date, want_finished=None):
    """WC matches whose kickoff (CST) falls on cst_day. Pull UTC day & day+1 (CST=UTC-5)."""
    fm = Fotmob()
    out = {}
    for utc in (cst_day, cst_day + timedelta(days=1)):
        d = fm.get(f"/api/data/matches?date={utc.strftime('%Y%m%d')}")
        for lg in d.get("leagues", []):
            if "World Cup" not in (lg.get("name") or ""):
                continue
            for m in lg.get("matches", []):
                st = m.get("status", {}) or {}
                ts = st.get("utcTime")
                if not ts:
                    continue
                kick = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(CST)
                if kick.date() != cst_day:
                    continue
                if want_finished is not None and bool(st.get("finished")) != want_finished:
                    continue
                out[m["id"]] = {"id": m["id"], "home": m.get("home", {}).get("name"),
                                "away": m.get("away", {}).get("name"),
                                "finished": bool(st.get("finished")), "kick": kick}
    return list(out.values())


def predict_today(cfg, pm, teams):
    props = pull_passes_lines()
    props = props[props.team.isin(teams)].reset_index(drop=True)
    if props.empty:
        return None, None
    up = score(predict(cfg, pm, props))
    return up, props


def save_board(cfg, up, props, DATE):
    raw = props.rename(columns={"name": "ud_name"})[["ud_name", "team", "line"]]
    pipe = up.merge(raw, on=["team", "line"], how="left")
    pipe_out = pd.DataFrame({"date": DATE, "competition": "World Cup 2026", "team": pipe["team"],
                             "player": pipe["ud_name"].fillna(pipe["player"]), "stat": "Passes",
                             "line": pipe["line"], "multiplier_boost": 1.0}).drop_duplicates(
                             ["date", "team", "player", "line"])
    path = Path(cfg["underdog"]["lines_csv"])
    if path.exists():
        prior = pd.read_csv(path)
        combined = pd.concat([prior, pipe_out], ignore_index=True).drop_duplicates(
            ["date", "team", "player", "line"], keep="last")
    else:
        combined = pipe_out
    combined.to_csv(path, index=False)
    sheet = up.assign(date=DATE)[["date", "team", "player", "line", "pred", "pick", "p_over",
                                  "p_hit", "hist", "is_new", "agree"]].copy()
    sheet["p_over"] = sheet["p_over"].round(3); sheet["p_hit"] = sheet["p_hit"].round(3)
    sheet = sheet.sort_values("p_hit", ascending=False)
    sheet.to_csv(path.parent / f"lines_{DATE}.csv", index=False)
    return sheet


def grade_yesterday(cfg, cst_day):
    """Grade finished games from the prior CST day against its saved board."""
    ycst = cst_day - timedelta(days=1)
    lp = cfg.path("raw") / f"lines_{ycst.isoformat()}.csv"
    if not lp.exists():
        return None, f"no saved board for {ycst}"
    sheet = pd.read_csv(lp)
    sheet["pick"] = sheet["pick"].str[0].str.upper()
    sheet = sheet.rename(columns={"pred": "model"})
    finished = {t: m for m in slate_for(ycst, want_finished=True) for t in (m["home"], m["away"])}
    led = load_ledger()
    already = set(zip(led.date, led.team, led.player, led.line)) if len(led) else set()
    rows = []
    for team, picks in sheet.groupby("team"):
        m = finished.get(team)
        if not m:
            continue
        seen = np.array([(ycst.isoformat(), team, p, l) in already
                         for p, l in zip(picks.player, picks.line)])
        picks = picks[~seen]
        if len(picks):
            gm = grade_match(m["id"], ycst.isoformat(), "World Cup 2026", picks)
            if len(gm):
                rows.append(gm)
    if not rows:
        return None, f"nothing new to grade for {ycst}"
    graded = pd.concat(rows, ignore_index=True)
    append_rows(graded)
    return graded, f"graded {len(graded)} picks for {ycst}"


def standings():
    g = load_ledger()
    g = g[g.result.isin(["WIN", "LOSS"])].copy()
    if not len(g):
        return pd.DataFrame(), {}
    m = pd.to_numeric(g.minutes, errors="coerce")
    g["actual90"] = np.where(m.gt(0), g.actual * 90.0 / m, g.actual)
    w, l = (g.result == "WIN").sum(), (g.result == "LOSS").sum()
    summ = {"record": f"{w}-{l}", "hit": w / (w + l), "n": len(g),
            "mae90": (g.model - g.actual90).abs().mean(), "mae_raw": (g.model - g.actual).abs().mean()}
    per_slate = (g.assign(win=g.result.eq("WIN"))
                 .groupby("date").agg(W=("win", "sum"), L=("win", lambda s: (~s).sum()))
                 .reset_index())
    return per_slate, summ


def write_xlsx(DATE, sheet, up, parlays, graded, per_slate, summ, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"wc_passes_{DATE}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        # Today's board
        b = up[["team", "player", "pick", "line", "pred", "hist", "p_hit", "is_new", "agree"]].copy()
        b = b.rename(columns={"pred": "model", "p_hit": "P(hit)"})
        b["flag"] = np.where(b["is_new"], "new-cap",
                    np.where(~b["agree"], "model/hist disagree", ""))
        b.drop(columns=["is_new", "agree"]).to_excel(xl, sheet_name="Today Board", index=False)
        # Parlays
        prows = []
        for i, legs in enumerate(parlays, 1):
            pp = float(np.prod([L.p_hit for L in legs])); M = {2: 3.0, 3: 6.0}.get(len(legs), 0)
            for L in legs:
                prows.append({"parlay": i, "pick": L.pick, "player": L.player, "line": L.line,
                              "team": L.team, "model": round(L.pred, 1), "P(hit)": round(L.p_hit, 3)})
            prows.append({"parlay": i, "pick": f"P(all) {pp:.1%}", "player": f"pays {M:g}x",
                          "line": None, "team": f"EV {pp*M-1:+.0%}/$1", "model": None, "P(hit)": None})
        pd.DataFrame(prows).to_excel(xl, sheet_name="Parlays", index=False)
        # Yesterday grades
        if graded is not None and len(graded):
            gg = graded[["team", "player", "line", "pick", "model", "actual", "minutes", "result"]]
            gg.to_excel(xl, sheet_name="Yesterday Grades", index=False)
        # Standings
        if summ:
            hdr = pd.DataFrame([{"metric": "record", "value": summ["record"]},
                               {"metric": "hit rate", "value": f"{summ['hit']:.1%}"},
                               {"metric": "picks", "value": summ["n"]},
                               {"metric": "MAE per-90", "value": round(summ["mae90"], 1)},
                               {"metric": "edge vs 52.4%", "value": f"{summ['hit']-0.524:+.1%}"}])
            hdr.to_excel(xl, sheet_name="Standings", index=False, startrow=0)
            per_slate.to_excel(xl, sheet_name="Standings", index=False, startrow=len(hdr) + 2)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="CST date YYYY-MM-DD (default: today CST)")
    ap.add_argument("--teams", default=None, help="comma-sep override of today's teams")
    args = ap.parse_args()

    cst_day = date.fromisoformat(args.date) if args.date else datetime.now(CST).date()
    DATE = cst_day.isoformat()
    cfg = Config.load()
    pm = load_corpus(cfg)

    if args.teams:
        teams = set(t.strip() for t in args.teams.split(","))
        games = None
    else:
        games = slate_for(cst_day)
        teams = {t for g in games for t in (g["home"], g["away"])}
    print(f"[{DATE}] today's CST slate teams: {sorted(teams)}")

    up, props = predict_today(cfg, pm, teams)
    parlays = build_parlays(up, n_parlays=2, legs_per=3) if up is not None else []
    if up is not None:
        sheet = save_board(cfg, up, props, DATE)
        print(f"[save] board -> lines_{DATE}.csv ({len(sheet)} props)")
    else:
        sheet = pd.DataFrame(); print("[board] no props live yet for today's teams")

    graded, gmsg = grade_yesterday(cfg, cst_day)
    print(f"[grade] {gmsg}")

    per_slate, summ = standings()
    if summ:
        print(f"[standings] {summ['record']} ({summ['hit']:.1%}), MAE90 {summ['mae90']:.1f}")

    out_dir = os.environ.get("WCP_DRIVE_DIR", str(REPO / "data" / "reports"))
    if up is not None:
        path = write_xlsx(DATE, sheet, up, parlays, graded, per_slate, summ, out_dir)
        print(f"[xlsx] -> {path}")


if __name__ == "__main__":
    main()
