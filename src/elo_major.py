"""Seeded, major-tournament-only Elo (2002 -> now).

Rationale: a plain all-results Elo lets teams blowout-farm weak confederation pools
(qualifiers/friendlies), so e.g. Japan floats above Brazil. This builder instead:

  1. Seeds every nation from its pre-2002-World-Cup FIFA ranking (data/raw/fifa_2002.csv),
     mapped onto an Elo scale — so absolute levels are calibrated ACROSS confederations
     from the start (no cold-start 1500 silo problem).
  2. Updates ratings ONLY on major-tournament matches: World Cup + continental
     championships (Euro, Copa America, AFCON, Asian Cup, Gold Cup).
  3. Treats FAILING TO QUALIFY for a major tournament as a modeled loss: every active
     member of a confederation that did not reach a given edition takes one loss against
     the average rating of that confederation's qualifiers — so perennial no-shows sink
     and strong teams that miss (e.g. Italy 2018) are penalised in proportion to how far
     above the qualifying bar they were.

Output mirrors team_ratings.compute_elo: {norm_team: [(date, rating_after), ...]}.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .team_ratings import _norm

# Continental championship league -> confederation; the World Cup spans all of them.
CONT_LEAGUE_CONF = {
    "uefa.euro": "uefa", "conmebol.america": "conmebol", "caf.nations": "caf",
    "afc.asian.cup": "afc", "concacaf.gold": "concacaf",
}
MAJOR_LEAGUES = set(CONT_LEAGUE_CONF) | {"fifa.world"}

# FIFA-2002-table names -> the spelling our results/corpus use, before _norm.
# (Most names already match ESPN once NAME_MAP/_norm is applied; only genuine
# divergences are listed here.)
FIFA_FIX = {
    "Cape Verde Islands": "Cabo Verde",
}


def load_seeds(path="data/raw/fifa_2002.csv", lo_pts=250, hi_pts=802,
               lo_elo=1300.0, hi_elo=2000.0) -> dict[str, float]:
    """{norm_team: seed_elo} from the 2002 FIFA points, linearly mapped to Elo."""
    df = pd.read_csv(path)
    m = (hi_elo - lo_elo) / (hi_pts - lo_pts)
    b = hi_elo - m * hi_pts
    seeds = {}
    for r in df.itertuples(index=False):
        elo = max(1000.0, min(2050.0, m * r.points + b))
        seeds[_norm(FIFA_FIX.get(r.team, r.team))] = elo
    return seeds


def build_confederations(espn_results: pd.DataFrame) -> dict[str, str]:
    """{norm_team: confederation} inferred from qualifier + continental labels.
    A team is assigned the confederation it appears in most often."""
    from collections import Counter
    conf_count: dict[str, Counter] = {}

    def bump(team, conf):
        conf_count.setdefault(_norm(team), Counter())[conf] += 1

    for r in espn_results.itertuples(index=False):
        comp = str(r.competition)
        conf = None
        if comp.startswith("fifa.worldq."):
            conf = comp.split(".")[-1]
        elif comp in CONT_LEAGUE_CONF:
            conf = CONT_LEAGUE_CONF[comp]
        elif comp == "uefa.nations":
            conf = "uefa"
        if conf:
            bump(r.home, conf); bump(r.away, conf)
    return {t: c.most_common(1)[0][0] for t, c in conf_count.items()}


def load_fifa_history(dirpath="data/raw/fifa_rankings",
                      lo_elo=1300.0, hi_elo=2020.0) -> dict:
    """{snapshot_date: {norm_team: fifa_implied_elo}} from the per-edition FIFA tables.
    Points scales differ by era, so each date is normalised internally: the date's top
    team -> hi_elo, its 5th-percentile team -> lo_elo (robust to tail outliers)."""
    out = {}
    for f in sorted(Path(dirpath).glob("*.csv")):
        df = pd.read_csv(f)
        lo, hi = df["points"].quantile(0.05), df["points"].max()
        rng = (hi - lo) or 1.0
        out[pd.Timestamp(f.stem)] = {
            _norm(t): max(1150.0, min(2060.0, lo_elo + (p - lo) / rng * (hi_elo - lo_elo)))
            for t, p in zip(df["team"], df["points"])}
    return dict(sorted(out.items()))


def _seed(seeds, team, default=1300.0):
    return seeds.get(team, default)


def _sg(x):
    """Signed concave goal value: sign(x)*log(1+|x|). Odd (keeps Elo zero-sum) and
    diminishing (a 5th goal matters less than the 1st, so blowouts can't be farmed)."""
    return math.copysign(math.log1p(abs(x)), x)


def compute_major_elo(major: pd.DataFrame, espn_results: pd.DataFrame,
                      k=35.0, hfa=0.0, gd_scale=130.0, q_weight=0.6, result_weight=0.5,
                      active_years=8, recency_halflife=5.0, wc_weight=1.3,
                      comp_weight=None, fifa_pull=0.50) -> dict[str, list]:
    """Seeded major-only Elo, scored on MARGIN vs EXPECTED margin.

    The update compares the actual goal difference to the difference expected from the
    rating gap: a heavy favourite expected to win by ~2 that only wins 1-0 UNDER-performs
    and loses rating, while beating a stronger side gains a lot. Concave goal value caps
    blowout-farming; it stays zero-sum.

    k                : base step.
    hfa              : home advantage in Elo pts (0 — majors are at neutral/host sites).
    gd_scale         : Elo points per expected goal of margin (gap/gd_scale = expected GD).
    q_weight         : non-qualification penalty as a fraction of k.
    recency_halflife : years; recent games/penalties move ratings more (form > reputation).
                       None disables. weight = 0.5 ** (age_years / halflife).
    wc_weight        : World Cup multiplier (set via comp_weight; cross-confederation signal).
    """
    # tournament weight by POOL STRENGTH — stops weak-pool farming (Gold Cup, Asian Cup)
    # from inflating a team the way qualifier-farming once floated Japan above Brazil.
    cw = {"fifa.world": wc_weight, "uefa.euro": 1.0, "conmebol.america": 1.0,
          "caf.nations": 0.85, "afc.asian.cup": 0.7, "concacaf.gold": 0.6}
    if comp_weight:
        cw.update(comp_weight)

    seeds = load_seeds()
    confeds = build_confederations(espn_results)
    fifa_hist = load_fifa_history() if fifa_pull else {}
    fifa_dates = sorted(fifa_hist)
    rating: dict[str, float] = dict(seeds)
    # Seed every ranked nation onto the timeline at the ranking date, so EVERY team is
    # modeled (has an as-of rating) even if it never reaches a major or takes a penalty.
    base = pd.Timestamp("2002-05-15")
    timeline: dict[str, list] = {t: [(base, e)] for t, e in seeds.items()}

    major = major[major["competition"].isin(MAJOR_LEAGUES)].copy()
    major["date"] = pd.to_datetime(major["date"]).dt.tz_localize(None)
    major = major.sort_values("date").reset_index(drop=True)
    major["year"] = major["date"].dt.year
    ref = major["date"].max()

    def rweight(date):
        if not recency_halflife:
            return 1.0
        age = (ref - date).days / 365.0
        return 0.5 ** (age / recency_halflife)

    # when did each team last play a major match (for the "active" test below)
    last_played: dict[str, pd.Timestamp] = {}

    def rec(team, date):
        timeline.setdefault(team, []).append((date, rating[team]))

    # iterate edition-by-edition in chronological order, so results and the
    # end-of-edition non-qualification penalties interleave correctly by date.
    editions = sorted(major.groupby(["competition", "year"]),
                      key=lambda kv: major.loc[kv[1].index, "date"].min())
    for (comp, year), g in editions:
        g = g.sort_values("date")
        estart = g["date"].min()
        # 0) pre-tournament pull toward the FIFA ranking: regress every team's rating a
        #    fraction of the way to its FIFA-implied Elo at this edition's snapshot. Anchors
        #    drift (e.g. a team inflated by plucky draws) back to FIFA reality and folds in
        #    qualifier/friendly form that FIFA tracks but we don't play.
        if fifa_pull and fifa_dates:
            prior = [d for d in fifa_dates if d <= estart + pd.Timedelta(days=3)]
            if prior:
                fe = fifa_hist[prior[-1]]
                for t, target in fe.items():
                    if t in rating:
                        rating[t] += fifa_pull * (target - rating[t])
                        rec(t, estart)
        # 1) play the edition's matches
        for r in g.itertuples(index=False):
            h, a = _norm(r.home), _norm(r.away)
            rating.setdefault(h, _seed(seeds, h)); rating.setdefault(a, _seed(seeds, a))
            Rh, Ra = rating[h], rating[a]
            exp_gd = (Rh + hfa - Ra) / gd_scale            # margin the gap predicts
            gd = (r.home_goals or 0) - (r.away_goals or 0)
            # performance = scoreline vs expectation. gd < exp_gd (a narrow win over a big
            # underdog) is NEGATIVE -> the favourite's rating drops, as it should. A small
            # result term keeps an ugly WIN from cratering.
            Eh = 1.0 / (1.0 + 10 ** ((Ra - (Rh + hfa)) / 400.0))
            Sh = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
            # Result is judged against OPPONENT STRENGTH: a narrow win over a big underdog
            # (gd < exp_gd) is an underperformance and DROPS the favourite (Brazil 1-0 Peru),
            # while a draw/upset over a much stronger side (Morocco 1-1 Brazil) lifts the
            # underdog. Beating weak teams barely moves you because exp_gd is already high.
            perf = (_sg(gd) - _sg(exp_gd)) + result_weight * (Sh - Eh)
            w = rweight(r.date) * cw.get(comp, 1.0)
            rating[h] += k * w * perf; rating[a] -= k * w * perf
            rec(h, r.date); rec(a, r.date)
            last_played[h] = last_played[a] = r.date

        # 2) non-qualification penalty at edition end
        edate = g["date"].max()
        qualified = set(_norm(t) for t in pd.unique(g[["home", "away"]].values.ravel()))
        target_confs = (list(CONT_LEAGUE_CONF.values()) if comp == "fifa.world"
                        else [CONT_LEAGUE_CONF.get(comp)])
        for conf in [c for c in target_confs if c]:
            members = [t for t, c in confeds.items() if c == conf]
            q_in_conf = [t for t in qualified if confeds.get(t) == conf]
            if len(q_in_conf) < 2:
                continue                                   # can't form a bar
            bar = sum(rating.get(t, _seed(seeds, t)) for t in q_in_conf) / len(q_in_conf)
            for t in members:
                if t in qualified:
                    continue
                # only penalise teams that are "active" (seeded, or played a major
                # recently) — don't keep flogging teams that have left the scene.
                lp = last_played.get(t)
                active = (t in seeds) or (lp is not None and (edate - lp).days <= active_years * 365)
                if not active:
                    continue
                R = rating.setdefault(t, _seed(seeds, t))
                E = 1.0 / (1.0 + 10 ** ((bar - R) / 400.0))   # expected vs the bar
                w = rweight(edate) * cw.get(comp, 1.0)
                rating[t] = R + (k * q_weight * w) * (0.0 - E)  # a loss to the bar
                rec(t, edate)
    return timeline
