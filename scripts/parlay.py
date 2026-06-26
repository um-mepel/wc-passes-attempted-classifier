"""Build the best 3-leg Underdog passes parlay.

Strategy: same-team passes are correlated, so we take only the SINGLE best pick per
team, then combine the top 3 across DIFFERENT games into a parlay.

Per prop we score confidence from model + the player's own history:
  - both model and history must lean the SAME way vs the line (agreement),
  - edge = how far the line sits from the model prediction AND from history,
  - new caps (no StatsBomb history -> pooled) are heavily downweighted (unreliable).

Run close to kickoff once lines for the later games have posted.
"""
from __future__ import annotations

import json
import ssl
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.features import build, fit_style_clusters, team_match_table
from src.minutes_model import MinutesModel
from src.predict import posterior_predictive, prob_over, summarize
from src.rate_model import HierNB

_SSL = ssl.create_default_context(); _SSL.check_hostname = False; _SSL.verify_mode = ssl.CERT_NONE
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
# Underdog country code -> StatsBomb team name
CC = {"FRA": "France", "NOR": "Norway", "ESP": "Spain", "URU": "Uruguay", "BEL": "Belgium",
      "SEN": "Senegal", "EGY": "Egypt", "IRN": "Iran", "KSA": "Saudi Arabia",
      "POR": "Portugal", "GER": "Germany", "ENG": "England", "NED": "Netherlands", "ARG": "Argentina"}


def _norm(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower()


def pull_passes_lines() -> pd.DataFrame:
    req = urllib.request.Request("https://api.underdogfantasy.com/beta/v5/over_under_lines",
                                 headers={"User-Agent": UA})
    d = json.load(urllib.request.urlopen(req, timeout=20, context=_SSL))
    players = {p["id"]: p for p in d.get("players", [])}
    apps = {a["id"]: a for a in d.get("appearances", [])}
    rows = []
    for o in d.get("over_under_lines", []):
        ap = o.get("over_under", {}).get("appearance_stat", {}) or {}
        if ap.get("stat") != "period_1_2_passes":
            continue
        appr = apps.get(ap.get("appearance_id"), {})
        pl = players.get(appr.get("player_id"), {})
        cc = pl.get("country") or ""
        rows.append({"name": (pl.get("first_name", "") + " " + pl.get("last_name", "")).strip(),
                     "team": CC.get(cc, cc), "line": float(o["stat_value"]),
                     "match_id": appr.get("match_id")})
    df = pd.DataFrame(rows)
    # opponent = the other team sharing the same Underdog match_id
    opp = {}
    for mid, g in df.groupby("match_id"):
        teams = list(dict.fromkeys(g.team))
        for t in teams:
            opp[(mid, t)] = next((x for x in teams if x != t), "_OPP_")
    df["opponent"] = [opp.get((m, t), "_OPP_") for m, t in zip(df.match_id, df.team)]
    return df


def predict(cfg, pm, props) -> pd.DataFrame:
    rows = []
    for r in props.itertuples(index=False):
        pool = set(zip(pm[pm.team == r.team].player_id, pm[pm.team == r.team].player))
        hit = [(pid, nm) for pid, nm in pool if _norm(r.name.split()[-1]) in _norm(nm).split()]
        pid = hit[0][0] if hit else -(abs(hash(r.name)) % 10**7)
        phist = pm[(pm.player_id == pid) & pm.position.notna()]
        position = phist.position.mode().iloc[0] if len(phist) else "Center Midfield"
        hist = pm.loc[(pm.player_id == pid) & (pm.minutes >= 70), "passes_attempted"]
        rows.append(dict(match_id=9500000, match_date=pd.Timestamp("2026-06-26"), competition="WC 2026",
                         season="WC 2026", provider="statsbomb", has_360=False, team=r.team,
                         opponent=r.opponent, player_id=pid, player=hit[0][1] if hit else r.name,
                         position=position, minutes=90.0, started=True,
                         passes_attempted=np.nan, passes_completed=np.nan,
                         line=r.line, hist=hist.mean() if len(hist) else np.nan, is_new=not hit))
    up = pd.DataFrame(rows)
    feats = build(pd.concat([pm, up.drop(columns=["line", "hist", "is_new"])], ignore_index=True), cfg,
                  style_map=fit_style_clusters(team_match_table(pm), cfg["features"]["opponent_style_clusters"])[1])
    # build() re-sorts rows, so work in fu's order and merge line/hist back by player_id —
    # never assign predictions positionally onto `up` (that scrambles player<->prediction).
    fu = feats[feats.match_id == 9500000].copy().reset_index(drop=True)
    model = HierNB.load(cfg, cfg.path("models") / "hiernb")
    s = posterior_predictive(model, MinutesModel().fit(pm[pm.minutes > 0]), fu,
                             np.full(len(fu), 1.0), n_draws=1000, use_actual_minutes=True, seed=1)
    fu = fu.merge(up[["player_id", "line", "hist", "is_new"]].drop_duplicates("player_id"),
                  on="player_id", how="left")
    fu["pred"] = summarize(s).pred.values
    fu["p_over"] = prob_over(s, fu["line"].values)
    return fu


def score(up: pd.DataFrame) -> pd.DataFrame:
    """Confidence: model & history agree on side, large edge, real history."""
    up = up.copy()
    up["pick"] = np.where(up["p_over"] >= 0.5, "OVER", "UNDER")
    up["model_edge"] = np.abs(up["p_over"] - 0.5)                  # 0..0.5
    hist_over = up["hist"] > up["line"]
    up["agree"] = (hist_over == (up["pick"] == "OVER")) | up["hist"].isna()
    # confidence: model edge, require agreement, kill new caps
    up["confidence"] = up["model_edge"] * up["agree"] * np.where(up["is_new"], 0.25, 1.0)
    up.loc[up["hist"].notna() & ~up["agree"], "confidence"] *= 0.3  # model/history disagree -> distrust
    return up.sort_values("confidence", ascending=False)


def passes_corr(pm, pid_a, pid_b, min_co=5, default=0.6) -> float:
    """Correlation of two players' attempted passes across matches they BOTH started.
    Different-team players -> 0 (independent). Thin co-appearance history -> conservative default."""
    a = pm[(pm.player_id == pid_a) & (pm.minutes >= 60)][["match_id", "passes_attempted"]]
    b = pm[(pm.player_id == pid_b) & (pm.minutes >= 60)][["match_id", "passes_attempted"]]
    m = a.merge(b, on="match_id", suffixes=("_a", "_b"))
    if len(m) < min_co:
        return default
    c = np.corrcoef(m.passes_attempted_a, m.passes_attempted_b)[0, 1]
    return float(c) if not np.isnan(c) else default


def pick_parlay(pm, up, n=3, corr_cap=0.35):
    """Greedy by confidence. Different teams -> independent, always fine. Same team ->
    allowed ONLY if same pick direction AND genuinely low passes correlation; opposite
    directions on a team are a hedge (anti-correlated outcomes), never a parlay leg."""
    cand = up.sort_values("confidence", ascending=False)
    legs = []
    for r in cand.itertuples(index=False):
        if r.confidence <= 0:
            continue
        ok = True
        for L in legs:
            if r.team != L.team:
                continue                                   # different team -> independent
            if r.pick != L.pick:                           # opposite direction, same team
                ok = False; break                          #   -> hedge, exclude
            if passes_corr(pm, r.player_id, L.player_id) > corr_cap:
                ok = False; break                          # same dir but too correlated
        if ok:
            legs.append(r)
        if len(legs) == n:
            break
    return legs


def main():
    cfg = Config.load()
    pm = pd.read_parquet(cfg.path("raw") / "sb_player_match.parquet")
    props = pull_passes_lines()
    if props.empty:
        print("No passes props live right now."); return
    print(f"{len(props)} passes props across {props.team.nunique()} teams: {sorted(props.team.unique())}\n")
    up = score(predict(cfg, pm, props))
    print("=== all props ranked by confidence ===")
    for r in up.itertuples(index=False):
        h = f"{r.hist:.0f}" if not np.isnan(r.hist) else "n/a"
        flag = " [new cap-LOW TRUST]" if r.is_new else (" [model/hist disagree]" if not r.agree else "")
        print(f"  {r.team:8} {r.player[:22]:22} {r.pick} {r.line:5.1f}  (model {r.pred:4.1f}, hist {h:>3}, conf {r.confidence:.2f}){flag}")
    legs = pick_parlay(pm, up, n=3, corr_cap=0.5)
    print(f"\n=== recommended {len(legs)}-leg parlay (correlation-capped, not just 1/team) ===")
    for r in legs:
        same = [L.player for L in legs if L.team == r.team and L.player != r.player]
        note = f"  [same team as {same[0][:14]} but low corr]" if same else ""
        print(f"  {r.pick} {r.player} {r.line} passes  ({r.team}){note}")
    if len(legs) < 3:
        print(f"  ...only {len(legs)} uncorrelated +EV leg(s) available now — re-run when later games post lines.")


if __name__ == "__main__":
    main()
