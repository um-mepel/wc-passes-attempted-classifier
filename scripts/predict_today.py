"""Generate a shareable PDF of model projections for a day's fixtures.

IMPORTANT HONESTY NOTE (also printed on the PDF): the model is trained on
2018-2024 tournament data. There is no free live-2026 data and no confirmed
lineups, so for each covered team we project its MOST RECENT tournament starting
XI. Squads have changed; treat these as model projections from stale rosters,
not confirmed-lineup predictions. Teams never seen in our data are not modeled.

Usage: python3 scripts/predict_today.py
Edit FIXTURES / DATE below for other days.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.features import build, fit_style_clusters, team_match_table
from src.minutes_model import MinutesModel
from src.predict import posterior_predictive, summarize
from src.rate_model import HierNB

DATE = "2026-06-26"
FIXTURES = [
    ("Norway", "France"), ("Senegal", "Iraq"), ("Cabo Verde", "Saudi Arabia"),
    ("Uruguay", "Spain"), ("New Zealand", "Belgium"), ("Egypt", "Iran"),
]


def last_xi(pm: pd.DataFrame, team: str) -> pd.DataFrame:
    """Most-recent-match starting XI for a team (projected lineup)."""
    sub = pm[(pm.team == team) & pm.started]
    if not len(sub):
        return pd.DataFrame()
    last_mid = sub.sort_values("match_date").match_id.iloc[-1]
    return sub[sub.match_id == last_mid][["player_id", "player", "position"]].drop_duplicates("player_id")


def build_upcoming(pm: pd.DataFrame, fixtures, date) -> pd.DataFrame:
    """Construct as-of-today rows for each covered team's projected XI."""
    rows = []
    mid = 9_000_000
    for home, away in fixtures:
        for team, opp in [(home, away), (away, home)]:
            xi = last_xi(pm, team)
            for _, p in xi.iterrows():
                rows.append(dict(match_id=mid, match_date=pd.Timestamp(date), competition="WC 2026",
                                 season="WC 2026", provider="statsbomb", has_360=False,
                                 team=team, opponent=opp, player_id=p.player_id, player=p.player,
                                 position=p.position, minutes=np.nan, started=True,
                                 passes_attempted=np.nan, passes_completed=np.nan))
            mid += 1
    return pd.DataFrame(rows)


def main():
    cfg = Config.load()
    pm = pd.read_parquet(cfg.path("raw") / "sb_player_match.parquet")
    upcoming = build_upcoming(pm, FIXTURES, DATE)

    style_map = fit_style_clusters(team_match_table(pm), cfg["features"]["opponent_style_clusters"])[1]
    feats = build(pd.concat([pm, upcoming], ignore_index=True), cfg, style_map=style_map)
    fu = feats[feats.match_id >= 9_000_000].copy()

    model = HierNB.load(cfg, cfg.path("models") / "hiernb")
    minutes = MinutesModel(cfg["features"]["recency_halflife_matches"]).fit(pm[pm.minutes > 0])
    samples = posterior_predictive(model, minutes, fu, np.full(len(fu), 0.9), n_draws=600, seed=cfg["model"]["seed"])
    summ = summarize(samples).set_index(fu.index)
    fu = fu.assign(pred=summ.pred.values, lo=summ.ci_low.values, hi=summ.ci_high.values, moe=summ.moe.values)

    render_pdf(cfg, fu, pm)
    print(f"Wrote {cfg.path('models').parent / 'predictions_2026-06-26.pdf'}")


def render_pdf(cfg, fu, pm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    have = set(pm.team.dropna().unique())
    out = Path.home() / "wc-passes-model" / f"predictions_{DATE}.pdf"
    with PdfPages(out) as pdf:
        # cover / methodology page
        fig = plt.figure(figsize=(8.5, 11)); fig.clf()
        fig.text(0.5, 0.93, "Passes-Attempted Projections", ha="center", size=20, weight="bold")
        fig.text(0.5, 0.89, f"FIFA World Cup 2026  —  {DATE}", ha="center", size=13)
        caveat = (
            "MODEL PROJECTIONS — READ FIRST\n\n"
            "* Predicts passes ATTEMPTED per player (Underdog 'Passes' stat).\n"
            "* 'pred' is the model's mean; the 80% interval is the margin of error\n"
            "  (model is 80% confident the true value lands in that range).\n\n"
            "DATA CAVEATS (important if sharing):\n"
            "* Model trained on 2018-2024 tournaments. No live 2026 data exists\n"
            "  freely, and lineups are NOT confirmed.\n"
            "* Each team's projected XI = its MOST RECENT tournament starting XI.\n"
            "  Squads have changed, so treat these as projections from stale rosters.\n"
            "* Teams never seen in our data are NOT modeled (shown as such).\n"
            "* Confirm starters ~1h pre-kickoff before acting on any number."
        )
        fig.text(0.1, 0.5, caveat, ha="left", va="center", size=11, family="monospace")
        pdf.savefig(fig); plt.close(fig)

        for home, away in FIXTURES:
            fig = plt.figure(figsize=(8.5, 11)); fig.clf()
            fig.text(0.5, 0.95, f"{home}  vs  {away}", ha="center", size=16, weight="bold")
            y = 0.88
            for team, opp in [(home, away), (away, home)]:
                if team not in have:
                    fig.text(0.1, y, f"{team}: no data in model — not projected", size=11, color="crimson")
                    y -= 0.05; continue
                rows = fu[fu.team == team].sort_values("pred", ascending=False)
                fig.text(0.1, y, f"{team}  (projected XI)", size=12, weight="bold"); y -= 0.03
                fig.text(0.1, y, f"{'Player':28}{'Pos':10}{'Pred':>6}  {'80% interval':>16}",
                         size=9, family="monospace", weight="bold"); y -= 0.022
                for _, r in rows.iterrows():
                    line = f"{str(r.player)[:27]:28}{str(r.position)[:9]:10}{r.pred:6.1f}  [{r.lo:5.1f}, {r.hi:5.1f}]"
                    fig.text(0.1, y, line, size=9, family="monospace"); y -= 0.020
                y -= 0.03
            fig.text(0.5, 0.04, "Model projections — not confirmed lineups. See first page.",
                     ha="center", size=8, color="gray")
            pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
