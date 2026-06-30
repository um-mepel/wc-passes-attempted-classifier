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
from src.features import build, fit_style_clusters, team_match_table, load_corpus
from src.minutes_model import MinutesModel
from src.predict import posterior_predictive, prob_over, summarize
from src.rate_model import HierNB

_SSL = ssl.create_default_context(); _SSL.check_hostname = False; _SSL.verify_mode = ssl.CERT_NONE
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
# Underdog country code -> StatsBomb team name
CC = {"FRA": "France", "NOR": "Norway", "ESP": "Spain", "URU": "Uruguay", "BEL": "Belgium",
      "SEN": "Senegal", "EGY": "Egypt", "IRN": "Iran", "KSA": "Saudi Arabia", "IRQ": "Iraq",
      "POR": "Portugal", "PRT": "Portugal", "GER": "Germany", "ENG": "England", "NED": "Netherlands",
      "ARG": "Argentina", "COL": "Colombia", "PAN": "Panama", "AUT": "Austria", "DZA": "Algeria",
      "GHA": "Ghana", "HRV": "Croatia", "CRO": "Croatia", "ITA": "Italy", "BRA": "Brazil",
      "URY": "Uruguay", "ESP2": "Spain", "USA": "United States", "MEX": "Mexico",
      # ISO-3 codes Underdog actually sends (the abbreviations above are partly wrong/missing):
      "NLD": "Netherlands", "DEU": "Germany", "JPN": "Japan", "MAR": "Morocco",
      "CIV": "Ivory Coast", "ECU": "Ecuador", "SWE": "Sweden", "PRY": "Paraguay",
      "BIH": "Bosnia and Herzegovina", "DZA2": "Algeria"}


# Nordic/Germanic letters that NFD does NOT decompose (they're standalone letters,
# not accented bases) but feeds like Underdog strip to ASCII -> transliterate explicitly.
_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "å": "a", "Å": "A",
                           "ð": "d", "Ð": "D", "þ": "th", "Þ": "TH", "ł": "l", "Ł": "L"})


def _norm(s):
    s = str(s).translate(_TRANSLIT)
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


# ESPN displayName -> StatsBomb team name where they differ
_ESPN_FIX = {"Cape Verde": "Cabo Verde", "USA": "United States", "Korea Republic": "South Korea"}


def get_fixtures() -> dict:
    """Today's team -> opponent map from ESPN (so the opponent is the real fixture,
    not inferred from which teams happen to have props)."""
    req = urllib.request.Request("https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard",
                                 headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    import gzip
    raw = urllib.request.urlopen(req, timeout=20, context=_SSL).read()
    try:
        raw = gzip.decompress(raw)
    except Exception:
        pass
    d = json.loads(raw)
    fx = {}
    for e in d.get("events", []):
        cs = e.get("competitions", [{}])[0].get("competitors", [])
        if len(cs) == 2:
            a = _ESPN_FIX.get(cs[0]["team"]["displayName"], cs[0]["team"]["displayName"])
            b = _ESPN_FIX.get(cs[1]["team"]["displayName"], cs[1]["team"]["displayName"])
            fx[a], fx[b] = b, a
    return fx


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
    # opponent: prefer the other team sharing the Underdog match_id (works when both
    # teams have props); fall back to today's ESPN fixture otherwise.
    try:
        fx = get_fixtures()
    except Exception:
        fx = {}
    opp_by_match = {}
    for mid, g in df.groupby("match_id"):
        teams = list(dict.fromkeys(g.team))
        if len(teams) == 2:
            for t in teams:
                opp_by_match[(mid, t)] = next(x for x in teams if x != t)
    df["opponent"] = [opp_by_match.get((m, t)) or fx.get(t, "_OPP_")
                      for m, t in zip(df.match_id, df.team)]
    return df


def predict(cfg, pm, props) -> pd.DataFrame:
    rows = []
    for r in props.itertuples(index=False):
        pool = sorted(set(zip(pm[pm.team == r.team].player_id, pm[pm.team == r.team].player)),
                      key=lambda x: _norm(x[1]))  # deterministic order (set iteration is not)
        rtoks = set(_norm(r.name).split())
        cands = [(pid, nm) for pid, nm in pool if _norm(r.name.split()[-1]) in _norm(nm).split()]
        # disambiguate same-last-name collisions (Frenkie vs Luuk de Jong, Kaishu vs Kodai
        # Sano) by full-name token overlap — first name breaks the tie, not set order.
        hit = sorted(cands, key=lambda x: len(rtoks & set(_norm(x[1]).split())), reverse=True)
        pid = hit[0][0] if hit else -(abs(hash(r.name)) % 10**7)
        phist = pm[(pm.player_id == pid) & pm.position.notna()]
        position = phist.position.mode().iloc[0] if len(phist) else "Center Midfield"
        hist = pm.loc[(pm.player_id == pid) & (pm.minutes >= 70), "passes_attempted"]
        rows.append(dict(match_id=9500000, match_date=pd.Timestamp("2026-06-27"), competition="World Cup 2026",
                         season="World Cup 2026", provider="fotmob", has_360=False, team=r.team,
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
    p_over = prob_over(s, fu["line"].values)
    fu["p_over"] = _calibrate(cfg, p_over)        # map raw model prob -> calibrated frequency
    return fu


def _calibrate(cfg, p_over, raw_weight=0.7):
    """SOFT-BLEND the raw model probability with the isotonic calibrator: the rebuilt
    model is well-calibrated raw (70%->70% empirically), so the isotonic map over-tempers
    (pulls 70%->64%). Blend keeps a touch of extreme-shrinkage without killing +EV overs:
        p = raw_weight·raw + (1-raw_weight)·isotonic(raw).
    """
    import pickle
    raw = np.clip(p_over, 0.0, 1.0)
    path = cfg.path("models") / "calibrator.pkl"
    if not path.exists():
        return raw
    with open(path, "rb") as fh:
        iso = pickle.load(fh)
    cal = np.clip(iso.predict(raw), 0.0, 1.0)
    return np.clip(raw_weight * raw + (1.0 - raw_weight) * cal, 0.0, 1.0)


def score(up: pd.DataFrame) -> pd.DataFrame:
    """Confidence: model & history agree on side, large edge, real history."""
    up = up.copy()
    up["pick"] = np.where(up["p_over"] >= 0.5, "OVER", "UNDER")
    up["p_hit"] = np.where(up["pick"] == "OVER", up["p_over"], 1 - up["p_over"])  # model P(pick wins)
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


def build_parlays(up, n_parlays=2, legs_per=3, used=None):
    """Best 1-2 three-leg parlays across the WHOLE slate. Same-team players ARE allowed
    (legs can span games/slots), but: a player is never used twice across all parlays,
    and opposite-direction same-team legs (a hedge) are never combined in one parlay."""
    used = set(used or [])
    cand = up[up.confidence > 0].sort_values("confidence", ascending=False)
    parlays = []
    for _ in range(n_parlays):
        legs = []
        for r in cand.itertuples(index=False):
            if r.player_id in used or r.player_id in [L.player_id for L in legs]:
                continue
            if any(L.team == r.team and L.pick != r.pick for L in legs):  # hedge within parlay
                continue
            legs.append(r)
            if len(legs) == legs_per:
                break
        if len(legs) < legs_per:
            break                                      # not enough fresh legs for another parlay
        parlays.append(legs)
        used.update(L.player_id for L in legs)
    return parlays


def main():
    cfg = Config.load()
    pm = load_corpus(cfg)
    props = pull_passes_lines()
    if props.empty:
        print("No passes props live right now."); return
    print(f"{len(props)} passes props across {props.team.nunique()} teams: {sorted(props.team.unique())}\n")
    up = score(predict(cfg, pm, props))
    print("=== all props ranked by confidence ===")
    for r in up.itertuples(index=False):
        h = f"{r.hist:.0f}" if not np.isnan(r.hist) else "n/a"
        flag = " [new cap-LOW TRUST]" if r.is_new else (" [model/hist disagree]" if not r.agree else "")
        print(f"  {r.team:8} {r.player[:22]:22} {r.pick} {r.line:5.1f}  (model {r.pred:4.1f}, hist {h:>3}, "
              f"P(hit) {r.p_hit:.0%}, conf {r.confidence:.2f}){flag}")
    parlays = build_parlays(up, n_parlays=2, legs_per=3)
    if not parlays:
        n = int((up.confidence > 0).sum())
        print(f"\nNo full 3-leg parlay yet — only {n} confident leg(s) live (props for later games not posted).")
        return
    mult = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0}            # Underdog Standard payouts
    print(f"\n=== best {len(parlays)} three-leg parlay(s) for the rest of the day ===")
    for i, legs in enumerate(parlays, 1):
        p_each = [L.p_hit for L in legs]
        p_parlay = float(np.prod(p_each))               # independence assumption
        M = mult.get(len(legs), 0)
        ev = p_parlay * M - 1                            # per $1 staked
        breakeven = (1 / M) ** (1 / len(legs))           # per-leg P needed
        same_team = len({L.team for L in legs}) < len(legs)
        print(f"\nParlay {i}  ({len(legs)} legs, pays {M:g}x):")
        for r in legs:
            h = f"{r.hist:.0f}" if not np.isnan(r.hist) else "n/a"
            print(f"   {r.pick:5} {r.player[:24]:24} {r.line:5.1f}  ({r.team}, model {r.pred:.0f}, hist {h}, P(hit) {r.p_hit:.0%})")
        print(f"   -> P(all hit) {p_parlay:.1%}  x {M:g}  =  EV {ev:+.1%} per $1   (need {breakeven:.0%}/leg to break even)")
        if same_team:
            print(f"      NOTE: same-team legs are POSITIVELY correlated -> true P(all hit) is higher than the")
            print(f"      independent {p_parlay:.0%} if they move together (and lower variance the other way).")


if __name__ == "__main__":
    main()
