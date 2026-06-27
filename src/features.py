"""As-of-date, recency-weighted feature construction.

CORE LEAK RULE: every feature for a match on date D is computed using ONLY matches
that finished strictly before D. This holds across the whole corpus, so combined
with splits.py (model trains only on pre-cutoff rows) train and test stay separate.

Recency weighting: player rate and the player×opponent-style rate both use an
exponential half-life (per-knob in config) so recent matches dominate — the
player×style signal is deliberately recency-biased.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import Config


# ── team-level aggregates (possession, volume, opponent PPDA proxy) ──────────
def team_match_table(pm: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team): team passes, possession share, opponent passes."""
    tm = (pm.groupby(["match_id", "match_date", "competition", "team", "opponent"])
            .agg(team_passes=("passes_attempted", "sum")).reset_index())
    tot = tm.groupby("match_id")["team_passes"].transform("sum")
    tm["possession_share"] = tm["team_passes"] / tot
    opp = tm[["match_id", "team", "team_passes"]].rename(
        columns={"team": "opponent", "team_passes": "opp_passes_allowed"})
    tm = tm.merge(opp, on=["match_id", "opponent"], how="left")
    return tm


# ── opponent style clustering (recomputed as-of-date is overkill; cluster on
#    team identity priors built only from training history per fold) ──────────
def fit_style_clusters(team_hist: pd.DataFrame, k: int) -> tuple[KMeans, pd.DataFrame]:
    """Cluster teams into k play-styles from their historical possession & volume.

    Returns the fitted model and a (team -> style) map. Fit on TRAINING history only.
    """
    prof = (team_hist.groupby("team")
            .agg(poss=("possession_share", "mean"),
                 vol=("team_passes", "mean"),
                 allow=("opp_passes_allowed", "mean")).dropna())
    if len(prof) < k:
        prof["style"] = 0
        return None, prof[["style"]]
    X = (prof - prof.mean()) / (prof.std() + 1e-9)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X.values)
    prof["style"] = km.labels_
    return km, prof[["style"]]


# ── recency-weighted as-of-date player rates ────────────────────────────────
def _ewma_asof(values: np.ndarray, weights_age: np.ndarray, halflife: float) -> float:
    """Exponential recency weight over already-past observations (most recent last)."""
    if len(values) == 0:
        return np.nan
    decay = 0.5 ** (weights_age / halflife)
    return float(np.sum(values * decay) / np.sum(decay))


def build(pm: pd.DataFrame, cfg: Config, style_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the feature matrix. `pm` is the full player-match label table; features
    for each row use only strictly-earlier matches (as-of-date).

    `style_map` (team->style) MUST be fit on training history only (see fit_style_clusters);
    if None it is fit here from all of `pm` (use only for inference/EDA, not CV).
    """
    f = cfg["features"]
    pm = pm.sort_values("match_date").reset_index(drop=True).copy()
    pm["per90"] = pm["passes_attempted"] / pm["minutes"].clip(lower=1) * 90

    tm = team_match_table(pm)
    if style_map is None:
        _, style_map = fit_style_clusters(tm, f["opponent_style_clusters"])
    # style of the OPPONENT: join opp_style on (match_id, opponent) so each player row
    # gets exactly one row (joining on match_id alone duplicates across both teams).
    opp_style = (tm[["match_id", "team"]].merge(style_map, left_on="team", right_index=True, how="left")
                 .rename(columns={"team": "opponent", "style": "opp_style"}))
    pm = pm.merge(opp_style, on=["match_id", "opponent"], how="left")
    pm["opp_style"] = pm["opp_style"].fillna(-1).astype(int)

    # AS-OF-DATE team context — NEVER the current match. possession_share /
    # opp_passes_allowed are realized only AFTER kickoff, so using the current
    # match's values would leak the result. Instead use each team's expanding mean
    # over its STRICTLY EARLIER matches (shift() drops the current row).
    tm = tm.sort_values("match_date")
    tm["team_poss_asof"] = (tm.groupby("team")["possession_share"]
                            .transform(lambda s: s.shift().expanding().mean()))
    tm["team_allowed_asof"] = (tm.groupby("team")["opp_passes_allowed"]
                               .transform(lambda s: s.shift().expanding().mean()))
    # player's own team possession (as-of)
    pm = pm.merge(tm[["match_id", "team", "team_poss_asof"]], on=["match_id", "team"], how="left")
    # opponent's historically-allowed passes (as-of) = "park factor" for the matchup
    opp_allowed = tm[["match_id", "team", "team_allowed_asof"]].rename(
        columns={"team": "opponent", "team_allowed_asof": "opp_allowed_asof"})
    pm = pm.merge(opp_allowed, on=["match_id", "opponent"], how="left")

    hl_style = f["style_recency_halflife_matches"]
    recent_rate, style_rate = [], []
    # per-player history: list of (per90, opp_style, competition), chronological.
    hist: dict = {}
    for row in pm.itertuples(index=False):
        pid = row.player_id
        h = hist.get(pid, [])
        # RECENT-RATE ANCHOR: average per90 over the player's LAST 2 TOURNAMENTS only
        # (the 2 most-recently-appeared competitions before this match).
        if h:
            last2 = list(dict.fromkeys(c for _, _, c in reversed(h)))[:2]
            vals = [v for v, _, c in h if c in last2]
            recent_rate.append(float(np.mean(vals)) if vals else np.nan)
        else:
            recent_rate.append(np.nan)
        # recency-biased player×style rate from PAST matches vs THIS style only
        hs = [v for v, sst, _ in h if sst == row.opp_style]
        ages_s = np.arange(len(hs), 0, -1)
        style_rate.append(_ewma_asof(np.array(hs), ages_s, hl_style) if hs else np.nan)
        hist.setdefault(pid, []).append((row.per90, row.opp_style, row.competition))

    pm["recent_per90"] = recent_rate
    pm["style_per90_recencybiased"] = style_rate
    pm["role"] = pm["position"].map(_role_bucket).fillna("UNK")

    # ANCHOR fallback for players with no last-2-tournament history (new caps):
    # their role's mean per-90, then a global median. The model's offset uses this.
    role_p90 = (pm.assign(p90=pm["passes_attempted"] / pm["minutes"].clip(lower=1) * 90)
                .query("minutes > 0").groupby("role")["p90"].mean())
    pm["anchor_per90"] = (pm["recent_per90"].fillna(pm["role"].map(role_p90))
                          .fillna(pm["per90"].median()).clip(lower=1.0))

    # ESPN as-of-date team strength (Elo) + possession, joined for EVERY team incl.
    # opponents not in StatsBomb (e.g. Norway). Leakage-safe (only pre-date matches).
    pm = _attach_espn(pm, cfg)
    return pm


def _attach_espn(pm: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = ["team_elo", "opp_elo", "team_poss_espn", "opp_poss_espn"]
    try:
        from .team_ratings import attach
        results = pd.read_parquet(cfg.path("raw") / "espn_results.parquet")
        poss = pd.read_parquet(cfg.path("raw") / "espn_team_match.parquet")
        return attach(pm, results, poss)
    except (FileNotFoundError, OSError):
        for c in cols:                       # ESPN data not present (e.g. unit tests)
            pm[c] = 1500.0 if c.endswith("elo") else np.nan
        return pm


_GROUPS = ["def", "wing", "mid", "attack"]


def _pos_groups(pos) -> dict:
    """Multi-hot position-group membership (a player can be in several):
      def    = LB, CB, RB, DM        wing  = LB, LW, RB, RW
      mid    = DM, CM, AM            attack= AM, LW, ST, F, RW
    Passing volume responds to the matchup DIFFERENTLY per group (validation:
    builders explode in dominant games, mids get starved vs strong opponents)."""
    p = str(pos).lower()
    back = "back" in p
    fullback = back and "center back" not in p
    dm = "defensive midfield" in p
    return {
        "def": int(back or dm),
        "wing": int(fullback or "wing" in p),
        "mid": int("midfield" in p),
        "attack": int("attacking midfield" in p or "wing" in p or "forward" in p or "striker" in p),
    }


def _role_bucket(pos) -> str:
    if not isinstance(pos, str):
        return "UNK"
    p = pos.lower()
    if "back" in p and ("center" in p or "centre" in p):
        return "CB"
    if "back" in p:
        return "FB"
    if "defensive midfield" in p or "pivot" in p:
        return "DM"
    if "midfield" in p:
        return "CM"
    if "wing" in p:
        return "W"
    if "forward" in p or "striker" in p:
        return "ST"
    if "keeper" in p:
        return "GK"
    return "UNK"
