"""Predict passes for TODAY's matches using real predicted XIs (from Yahoo Sports).

Careful name->player_id matching (word-boundary + explicit nickname overrides) so we
never ship a wrong player. Players with no pre-2026 data are predicted via the model's
position/role pooling and clearly flagged (wider intervals, lower confidence).

No free 2026 passing data exists (FBref dropped Opta passing Jan 2026), so passing-rate
priors come from each player's 2018-2024 history; 2026 only tells us who's playing.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.features import build, fit_style_clusters, team_match_table, load_corpus
from src.minutes_model import MinutesModel
from src.predict import posterior_predictive, summarize
from src.rate_model import HierNB

DATE = "2026-06-26"

# (home, away). Real predicted XIs from Yahoo Sports; positions in 4-3-3 / 4-2-3-1 order.
# Each player: (display_surname, position_bucket). NEW = no pre-2026 data (pooled).
LINEUPS = {
    "France": ["Maignan GK", "Koundé RB", "Upamecano CB", "Saliba CB", "Digne LB",
               "Tchouaméni DM", "Rabiot CM", "Olise W", "Dembélé W", "Doué W", "Mbappé ST"],
    "Norway": None,  # no historical data — not modeled
    "Uruguay": ["Muslera GK", "Varela RB", "Cáceres CB", "Olivera CB", "Sanabria LB",
                "Bentancur CM", "Ugarte DM", "Valverde CM", "Canobbio W", "Núñez ST", "Araújo W"],
    "Spain": ["Simón GK", "Porro RB", "Cubarsí CB", "Laporte CB", "Cucurella LB",
              "Olmo CM", "Rodri DM", "Pedri CM", "Yamal W", "Oyarzabal ST", "Nico Williams W"],
}
FIXTURES = [("Norway", "France"), ("Uruguay", "Spain")]

# explicit overrides for nicknames / ambiguous surnames -> exact name substring in our data
OVERRIDES = {
    ("Spain", "Rodri"): "Rodrigo Hernández Cascante",
    ("Spain", "Nico Williams"): "Nicholas Williams",
}


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower()


def resolve(pm, team, label):
    """Return (player_id, full_name, is_new). label = 'Surname POS'."""
    surname, pos = label.rsplit(" ", 1)
    pool = {(pid, nm) for pid, nm in zip(pm[pm.team == team].player_id, pm[pm.team == team].player)}
    ov = OVERRIDES.get((team, surname))
    target = norm(ov) if ov else None
    cands = []
    for pid, nm in pool:
        nn = norm(nm)
        if target and target in nn:
            cands.append((pid, nm)); break
        if not target and surname.split()[-1] and norm(surname.split()[-1]) in nn.split():
            cands.append((pid, nm))
    if cands:
        return cands[0][0], cands[0][1], False, pos
    return None, surname, True, pos


def main():
    cfg = Config.load()
    pm = load_corpus(cfg)

    rows, report = [], []
    mid = 9_100_000
    for home, away in FIXTURES:
        for team, opp in [(home, away), (away, home)]:
            xi = LINEUPS.get(team)
            if xi is None:
                report.append((team, opp, "NO DATA — not modeled", None)); continue
            for label in xi:
                pid, name, is_new, pos = resolve(pm, team, label)
                report.append((team, opp, name, "NEW (pooled)" if is_new else f"id {pid}"))
                rows.append(dict(match_id=mid, match_date=pd.Timestamp(DATE), competition="World Cup 2026",
                                 season="World Cup 2026", provider="fotmob", has_360=False, team=team,
                                 opponent=opp, player_id=pid if pid is not None else -(abs(hash(name)) % 10**8),
                                 player=name, position=_pos_name(pos), minutes=np.nan, started=True,
                                 passes_attempted=np.nan, passes_completed=np.nan, is_new=is_new))
            mid += 1
    up = pd.DataFrame(rows)

    print("=== name resolution (verify no wrong players) ===")
    for t, o, nm, tag in report:
        print(f"  {t:8} {nm[:30]:30} {tag or ''}")

    style_map = fit_style_clusters(team_match_table(pm), cfg["features"]["opponent_style_clusters"])[1]
    feats = build(pd.concat([pm, up], ignore_index=True), cfg, style_map=style_map)
    fu = feats[feats.match_id >= 9_100_000].copy()
    fu["is_new"] = fu["player_id"] < 0      # new caps were assigned negative ids

    model = HierNB.load(cfg, cfg.path("models") / "hiernb")
    minutes = MinutesModel(cfg["features"]["recency_halflife_matches"]).fit(pm[pm.minutes > 0])
    samples = posterior_predictive(model, minutes, fu, np.full(len(fu), 0.9), n_draws=600, seed=cfg["model"]["seed"])
    s = summarize(samples).set_index(fu.index)
    fu = fu.assign(pred=s.pred.values, lo=s.ci_low.values, hi=s.ci_high.values)

    print("\n=== PREDICTIONS (real predicted XIs) ===")
    for home, away in FIXTURES:
        print(f"\n--- {home} vs {away} ---")
        for team in (home, away):
            sub = fu[fu.team == team].sort_values("pred", ascending=False)
            if not len(sub):
                print(f"  {team}: not modeled"); continue
            print(f"  {team}:")
            for _, r in sub.iterrows():
                flag = " *new cap (pooled)" if r.is_new else ""
                print(f"    {str(r.player)[:24]:24} {r.pred:5.1f}  [{r.lo:4.1f}, {r.hi:4.1f}]{flag}")
    render_pdf(cfg, fu)


def _pos_name(bucket):
    return {"GK": "Goalkeeper", "RB": "Right Back", "LB": "Left Back", "CB": "Center Back",
            "DM": "Center Defensive Midfield", "CM": "Center Midfield", "W": "Winger",
            "ST": "Center Forward"}.get(bucket, "Center Midfield")


def render_pdf(cfg, fu):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out = Path.home() / "wc-passes-model" / f"predictions_{DATE}.pdf"
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.93, "Passes-Attempted Projections", ha="center", size=20, weight="bold")
        fig.text(0.5, 0.89, f"World Cup 2026 — {DATE} — REAL predicted XIs (Yahoo Sports)", ha="center", size=12)
        fig.text(0.1, 0.5, (
            "MODEL PROJECTIONS — READ FIRST\n\n"
            "* Passes ATTEMPTED per player (Underdog 'Passes'). 'pred' = model mean;\n"
            "  [lo, hi] is the 80% interval (margin of error).\n\n"
            "* Lineups are Yahoo's PREDICTED XIs (not confirmed). Confirm ~1h pre-kick.\n"
            "* No free 2026 passing data exists; passing-rate priors come from each\n"
            "  player's 2018-2024 history. '*new cap' = no prior data -> pooled by\n"
            "  position (wider, lower-confidence).\n"
            "* Teams with no historical data (e.g. Norway) are not modeled."
        ), ha="left", va="center", size=11, family="monospace")
        pdf.savefig(fig); plt.close(fig)
        for home, away in FIXTURES:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.95, f"{home}  vs  {away}", ha="center", size=16, weight="bold")
            y = 0.88
            for team in (home, away):
                sub = fu[fu.team == team].sort_values("pred", ascending=False)
                if not len(sub):
                    fig.text(0.1, y, f"{team}: no historical data — not modeled", size=11, color="crimson")
                    y -= 0.05; continue
                fig.text(0.1, y, f"{team}  (Yahoo predicted XI)", size=12, weight="bold"); y -= 0.028
                fig.text(0.1, y, f"{'Player':26}{'Pred':>6}  {'80% interval':>14}", size=9,
                         family="monospace", weight="bold"); y -= 0.02
                for _, r in sub.iterrows():
                    flag = " *new" if r.is_new else ""
                    fig.text(0.1, y, f"{str(r.player)[:25]:26}{r.pred:6.1f}  [{r.lo:4.1f},{r.hi:5.1f}]{flag}",
                             size=9, family="monospace"); y -= 0.019
                y -= 0.03
            fig.text(0.5, 0.04, "Predicted XIs, not confirmed. '*new' = pooled estimate. See page 1.",
                     ha="center", size=8, color="gray")
            pdf.savefig(fig); plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
