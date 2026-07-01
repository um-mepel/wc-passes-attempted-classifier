"""Two-stage "team_volume x player_share" Bayesian model (an alternative to the
single-stage HierNB in rate_model.py).

    passes  =  team_volume(V)  x  player_share(S)  x  (minutes / 90)

Stage A — TEAM VOLUME (team-match rows):
    team_passes ~ NegBinomial(mu = exp(v0 + team_re[team] + gamma.Z), alpha_V)
    Z = standardized [x_poss, elo_delta, home, opp_allowed_asof]  (team-level as-of)

Stage B — PLAYER SHARE (player-match rows):
    share ~ Beta(mu.phi, (1-mu).phi),  mu = sigmoid(eta)
    eta   = m_pos[position] + m_player_dev[player] + press_pos[position].(opp_poss-0.5)
    - m_player_dev: per-player deviation, non-centered, pooled toward its POSITION mean
      (MAGNETISM — player share of team passes is a stable identity trait, split-half r~0.87).
    - press_pos: a per-POSITION slope on (opp_poss-0.5) — PRESSURE is noise per-player
      (r~0.03) but reliable per position (r~0.59), so it lives at the position level only.

Design rationale (validated this session) sits alongside rate_model.py's HierNB so a
shared eval harness can A/B them on out-of-sample CRPS. Interface mirrors HierNB:
`TwoStage(cfg).fit(df)`, `.save(path)`, `TwoStage.load(cfg, path)`, and the module
function `predict_two_stage(model, df, ...) -> (n_rows, n_draws) integer passes`.

Leakage rule (identical to features.build): only as-of columns feed the linear
predictors (x_poss, opp_poss, elo_delta, opp_allowed_asof, home). The realized
`team_passes` (Stage A) and `share` (Stage B) are TARGETS used only as `y`.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# Stage-A (team volume) design columns — all team-level as-of features (constant within
# a team-match). `home` is derived from is_home (0 when absent, e.g. WC prediction rows).
_STAGE_A_FEATURES = ["x_poss", "elo_delta", "home", "opp_allowed_asof"]
# A position needs at least this many training rows to get its own share level; sparser
# positions fall back to their role bucket so the level isn't estimated from ~nothing.
_MIN_POS_ROWS = 40


def _index(series: pd.Series) -> tuple[np.ndarray, list]:
    cats = series.astype("category")
    return cats.cat.codes.values, list(cats.cat.categories)


def _codes(series: pd.Series, cats: list) -> np.ndarray:
    """Map to stored training codes; unseen levels -> -1 (pooled group mean at predict)."""
    return series.map({c: i for i, c in enumerate(cats)}).fillna(-1).astype(int).values


def _gk_mask(df: pd.DataFrame) -> np.ndarray:
    return df["position"].astype(str).str.contains("Goalkeep", case=False, na=False).to_numpy()


class TwoStage:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.m = cfg["model"]
        self.idata_A = None       # Stage-A posterior (team volume)
        self.idata_B = None       # Stage-B posterior (player share)
        self.levels: dict = {}
        # scalers (Stage-A feature standardisation) + fitted passes-level dispersion
        self._z_mean = None
        self._z_std = None
        self._alpha_out = 8.0
        self._alpha_gk = 8.0

    # ── designs ─────────────────────────────────────────────────────────────
    def _home(self, df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(pd.Series(df.get("is_home", 0), index=df.index),
                             errors="coerce").fillna(0.0)

    def _team_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player-match rows to one row per (match_id, team). team_passes is the
        realized team total (TARGET); the design features are team-level as-of, so `first`
        of any player row equals the team-match value."""
        d = df.copy()
        d["home"] = self._home(d)
        for c in ("x_poss", "elo_delta", "opp_allowed_asof"):
            d[c] = pd.to_numeric(d.get(c, np.nan), errors="coerce")
        tf = (d.groupby(["match_id", "team"], as_index=False)
                .agg(team_passes=("team_passes", "first"),
                     passes_sum=("passes_attempted", "sum"),
                     x_poss=("x_poss", "first"),
                     elo_delta=("elo_delta", "first"),
                     home=("home", "first"),
                     opp_allowed_asof=("opp_allowed_asof", "first")))
        # team_passes column is the build()-merged team total; fall back to the summed
        # player passes if it's missing (e.g. a filtered slice).
        tf["team_passes"] = pd.to_numeric(tf["team_passes"], errors="coerce").fillna(tf["passes_sum"])
        return tf

    def _scale_A(self, frame: pd.DataFrame, training: bool) -> np.ndarray:
        """Standardize the Stage-A features with TRAIN mean/std (stored on the object).
        Median-impute missing, then clip to +/-8 SD to guard against out-of-range test
        values exploding exp(logV)."""
        raw = frame[_STAGE_A_FEATURES].apply(pd.to_numeric, errors="coerce")
        if training:
            self._impute_A = raw.median(numeric_only=True)
            filled = raw.fillna(self._impute_A).fillna(0.0)
            self._z_mean = filled.mean(0)
            self._z_std = filled.std(0) + 1e-9
        else:
            filled = raw.fillna(self._impute_A).fillna(0.0)
        return np.clip((filled - self._z_mean) / self._z_std, -8.0, 8.0).values

    def _pos_level(self, df: pd.DataFrame) -> pd.Series:
        """Share level = position string, but sparse positions fall back to their role
        bucket. `_pos_keep` (the well-populated positions) is fixed at training time."""
        keep = self.levels.get("_pos_keep", set(df["position"].unique()))
        pos = df["position"].astype(str)
        role = df["role"].astype(str) if "role" in df else pos
        return pos.where(pos.isin(keep), "role:" + role)

    def _share_design(self, df: pd.DataFrame, training: bool) -> dict:
        out = {}
        if training:
            keep = set(df["position"].astype(str).value_counts().pipe(
                lambda vc: vc[vc >= _MIN_POS_ROWS].index))
            self.levels["_pos_keep"] = keep
        poslev = self._pos_level(df)
        if training:
            codes, cats = _index(poslev); self.levels["poslevel"] = cats
            pcodes, pcats = _index(df["player_id"]); self.levels["player_id"] = pcats
        else:
            codes = _codes(poslev, self.levels["poslevel"])
            pcodes = _codes(df["player_id"], self.levels["player_id"])
        out["pos"] = codes
        out["player"] = pcodes
        opp = pd.to_numeric(df.get("opp_poss", 0.5), errors="coerce").fillna(0.5).to_numpy()
        out["opp_c"] = opp - 0.5
        if "share" in df:
            out["y"] = np.clip(pd.to_numeric(df["share"], errors="coerce").fillna(
                0.0).to_numpy(), 1e-4, 1 - 1e-4)
        return out

    # ── sampling helper (nutpie w/ pymc fallback, env overrides) ────────────
    def _sample(self, model, seed: int, tag: str):
        import pymc as pm
        cores = int(os.environ.get("PM_CORES", self.m.get("cores", 1)))
        draws = int(os.environ.get("PM_DRAWS", self.m["draws"]))
        tune = int(os.environ.get("PM_TUNE", self.m["tune"]))
        common = dict(draws=draws, tune=tune, chains=self.m["chains"],
                      target_accept=self.m["target_accept"], random_seed=seed)
        sampler = os.environ.get("PM_SAMPLER", self.m.get("sampler", "nutpie"))
        with model:
            try:
                if sampler != "nutpie":
                    raise RuntimeError("sampler!=nutpie")
                import nutpie  # noqa: F401
                print(f"[fit:2stage/{tag}] sampling with nutpie (Rust NUTS)...", flush=True)
                return pm.sample(**common, nuts_sampler="nutpie", progressbar=True)
            except Exception as e:
                print(f"[fit:2stage/{tag}] nutpie unavailable/failed "
                      f"({type(e).__name__}: {e}); falling back to pymc sampler", flush=True)
                return pm.sample(**common, cores=cores, progressbar=False,
                                 **({"mp_ctx": "spawn"} if cores > 1 else {}))

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "TwoStage":
        import pymc as pm

        # ---------- Stage A: team volume ----------
        tf = self._team_frame(df)
        tcodes, tcats = _index(tf["team"]); self.levels["team"] = tcats
        n_team = len(tcats)
        ZA = self._scale_A(tf, training=True)                     # (n_tm, nz)
        nz = ZA.shape[1]
        yV = np.clip(tf["team_passes"].to_numpy(), 1.0, None)
        v0_prior = float(np.log(np.clip(yV.mean(), 1.0, None)))   # ~log(mean team passes)

        with pm.Model() as model_A:
            # NON-CENTERED team random effect (avoids the hierarchical funnel).
            v0 = pm.Normal("v0", v0_prior, 1.0)
            sigma_team = pm.HalfNormal("sigma_team", 0.5)
            team_re = pm.Deterministic(
                "team_re", pm.Normal("team_re_z", 0, 1, shape=n_team) * sigma_team)
            gamma = pm.Normal("gamma", 0.0, 0.5, shape=nz)
            alpha_V = pm.Exponential("alpha_V", 1.0)
            logV = v0 + team_re[tcodes] + pm.math.dot(ZA, gamma)
            pm.NegativeBinomial("y", mu=pm.math.exp(logV), alpha=alpha_V, observed=yV)
        self.idata_A = self._sample(model_A, self.m["seed"], "A")

        # ---------- Stage B: player share ----------
        db = self._share_design(df, training=True)
        n_pos = max(1, len(self.levels["poslevel"]))
        n_player = len(self.levels["player_id"])
        share_prior = float(np.log(np.clip(db["y"].mean(), 1e-3, 1 - 1e-3) /
                                   (1 - np.clip(db["y"].mean(), 1e-3, 1 - 1e-3))))

        with pm.Model() as model_B:
            mu_share = pm.Normal("mu_share", share_prior, 1.0)     # global logit-share
            sigma_pos = pm.HalfNormal("sigma_pos", 0.75)
            m_pos = pm.Deterministic(
                "m_pos", mu_share + pm.Normal("m_pos_z", 0, 1, shape=n_pos) * sigma_pos)
            # MAGNETISM: per-player deviation pooled toward its position mean (added via
            # m_pos at predict); non-centered. Student-t for heavy-tailed hoggers.
            sigma_player = pm.HalfNormal("sigma_player", 0.5)
            m_player_dev = pm.Deterministic(
                "m_player_dev",
                pm.StudentT("m_player_z", nu=4, mu=0, sigma=1, shape=n_player) * sigma_player)
            # PRESSURE: position-level slope on (opp_poss-0.5); non-centered, its own sigma.
            mu_press = pm.Normal("mu_press", 0.0, 0.1)
            sigma_press = pm.HalfNormal("sigma_press", 0.1)
            press_pos = pm.Deterministic(
                "press_pos", mu_press + pm.Normal("press_pos_z", 0, 1, shape=n_pos) * sigma_press)
            phi = pm.Gamma("phi", alpha=2.0, beta=0.05)            # Beta concentration (~40)

            eta = (m_pos[db["pos"]] + m_player_dev[db["player"]]
                   + press_pos[db["pos"]] * db["opp_c"])
            mu = pm.math.sigmoid(eta)
            pm.Beta("y", alpha=mu * phi, beta=(1.0 - mu) * phi, observed=db["y"])
        self.idata_B = self._sample(model_B, self.m["seed"] + 1, "B")

        # ---------- passes-level dispersion (method-of-moments on the fitted mean) ----------
        # The combine uses the EXPECTED team volume x EXPECTED share, so the stage-internal
        # variance is dropped; a single fitted NB alpha (one outfield, one GK) carries the
        # passes-level observation noise. alpha from E[(y-mu)^2] = mu + mu^2/alpha.
        self._fit_dispersion(df)
        return self

    def _post_mean(self, post, name):
        return post[name].mean(dim=("chain", "draw")).values

    def _fit_dispersion(self, df: pd.DataFrame) -> None:
        A, B = self.idata_A.posterior, self.idata_B.posterior
        # Stage-A posterior-mean mu_V per player row (via its team-match).
        v0 = float(self._post_mean(A, "v0"))
        team_re = self._post_mean(A, "team_re")
        gamma = self._post_mean(A, "gamma")
        Z = self._scale_A(self._add_home(df), training=False)      # per-row team features
        tcode = _codes(df["team"], self.levels["team"])
        tr = np.where(tcode >= 0, team_re[np.where(tcode < 0, 0, tcode)], 0.0)
        mu_V = np.exp(np.clip(v0 + tr + Z @ gamma, None, 12))
        # Stage-B posterior-mean share.
        S = self._share_mean(df, B)
        minutes = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0).clip(lower=0).to_numpy()
        mu = np.clip(mu_V * S * (minutes / 90.0), 1e-6, None)
        y = pd.to_numeric(df["passes_attempted"], errors="coerce").fillna(0.0).to_numpy()
        gk = _gk_mask(df)
        self._alpha_out = self._mom_alpha(y[~gk], mu[~gk])
        self._alpha_gk = self._mom_alpha(y[gk], mu[gk])

    @staticmethod
    def _mom_alpha(y: np.ndarray, mu: np.ndarray, default: float = 8.0) -> float:
        if len(y) < 20:
            return default
        num = float(np.mean(mu ** 2))
        den = float(np.mean((y - mu) ** 2 - mu))
        if den <= 0 or num <= 0:
            return 1e4                      # ~Poisson (no over-dispersion detected)
        return float(np.clip(num / den, 0.5, 1e4))

    def _add_home(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy(); d["home"] = self._home(d)
        return d

    def _share_mean(self, df: pd.DataFrame, B) -> np.ndarray:
        """Posterior-mean share per row: sigmoid(m_pos[pos] + m_player_dev[player]
        + press_pos[pos]*(opp_poss-0.5)); unseen pos -> global mu_share, unseen player ->
        position mean (deviation 0)."""
        d = self._share_design(df, training=False)
        m_pos = self._post_mean(B, "m_pos"); mu_share = float(self._post_mean(B, "mu_share"))
        m_dev = self._post_mean(B, "m_player_dev")
        press = self._post_mean(B, "press_pos"); mu_press = float(self._post_mean(B, "mu_press"))
        pos, pl = d["pos"], d["player"]
        mp = np.where(pos >= 0, m_pos[np.where(pos < 0, 0, pos)], mu_share)
        md = np.where(pl >= 0, m_dev[np.where(pl < 0, 0, pl)], 0.0)
        pr = np.where(pos >= 0, press[np.where(pos < 0, 0, pos)], mu_press)
        eta = mp + md + pr * d["opp_c"]
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as fh:
            pickle.dump({"idata_A": self.idata_A, "idata_B": self.idata_B,
                         "levels": self.levels, "impute_A": self._impute_A,
                         "z_mean": self._z_mean, "z_std": self._z_std,
                         "alpha_out": self._alpha_out, "alpha_gk": self._alpha_gk}, fh)

    @classmethod
    def load(cls, cfg: Config, path: str | Path) -> "TwoStage":
        obj = cls(cfg)
        with open(Path(path) / "model.pkl", "rb") as fh:
            b = pickle.load(fh)
        obj.idata_A, obj.idata_B, obj.levels = b["idata_A"], b["idata_B"], b["levels"]
        obj._impute_A = b["impute_A"]
        obj._z_mean, obj._z_std = b["z_mean"], b["z_std"]
        obj._alpha_out, obj._alpha_gk = b["alpha_out"], b["alpha_gk"]
        return obj


# ── combine / predict ──────────────────────────────────────────────────────
def _stack_take(post, name, take):
    return post[name].stack(s=("chain", "draw")).values[..., take]


def _gather(arr: np.ndarray, idx: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """arr: (n_level, draws); fill: (draws,). Rows with idx<0 (unseen) get `fill`."""
    out = np.tile(fill, (len(idx), 1))
    ok = idx >= 0
    out[ok] = arr[idx[ok]]
    return out


def predict_two_stage(model: TwoStage, df: pd.DataFrame, n_draws: int = 1000,
                      use_actual_minutes: bool = True, seed: int = 0) -> np.ndarray:
    """Return an (n_rows, n_draws) integer sample of passes attempted, matching the
    shape/semantics of predict.posterior_predictive so a shared eval harness (CRPS,
    prob_over, summarize) can consume it identically.

    passes ~ NegBinomial(mu = mu_V . S . (minutes/90), alpha), where mu_V is the Stage-A
    expected team volume per posterior draw and S is the Stage-B expected (Beta-mean)
    share per draw. Stage variances are intentionally dropped and re-introduced as a
    single fitted passes-level NB dispersion (outfield/GK), keeping the combined
    variance sane and calibrated.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # ---------- Stage A draws: mu_V (rows, draws) ----------
    A = model.idata_A.posterior
    SA = A.dims["chain"] * A.dims["draw"]
    ta = rng.choice(SA, size=n_draws, replace=n_draws > SA)
    v0 = _stack_take(A, "v0", ta)                          # (draws,)
    team_re = _stack_take(A, "team_re", ta)                # (n_team, draws)
    gamma = _stack_take(A, "gamma", ta)                    # (nz, draws)
    tcode = _codes(df["team"], model.levels["team"])
    Z = model._scale_A(model._add_home(df), training=False)    # (rows, nz)
    tr = _gather(team_re, tcode, np.zeros(n_draws))            # unseen team -> re 0
    logV = v0[None, :] + tr + Z @ gamma                        # (rows, draws)
    mu_V = np.exp(np.clip(logV, None, 12))

    # ---------- Stage B draws: S (rows, draws) ----------
    B = model.idata_B.posterior
    SB = B.dims["chain"] * B.dims["draw"]
    tb = rng.choice(SB, size=n_draws, replace=n_draws > SB)
    m_pos = _stack_take(B, "m_pos", tb)                    # (n_pos, draws)
    m_dev = _stack_take(B, "m_player_dev", tb)             # (n_player, draws)
    press = _stack_take(B, "press_pos", tb)                # (n_pos, draws)
    mu_share = _stack_take(B, "mu_share", tb)              # (draws,)
    mu_press = _stack_take(B, "mu_press", tb)              # (draws,)
    d = model._share_design(df, training=False)
    mp = _gather(m_pos, d["pos"], mu_share)                # unseen pos -> global mu_share
    md = _gather(m_dev, d["player"], np.zeros(n_draws))    # unseen player -> position mean
    pr = _gather(press, d["pos"], mu_press)               # unseen pos -> global press
    eta = mp + md + pr * d["opp_c"][:, None]
    S = 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))       # Beta mean share (rows, draws)

    # ---------- combine + NB observation noise ----------
    if use_actual_minutes:
        minutes = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0).clip(lower=0).to_numpy()
    else:
        minutes = np.full(n, 90.0)
    mu = mu_V * S * (minutes / 90.0)[:, None]              # (rows, draws)
    mu = np.clip(mu, 1e-6, None)
    gk = _gk_mask(df)
    alpha = np.where(gk[:, None], model._alpha_gk, model._alpha_out)   # (rows, 1)
    a = np.clip(alpha, 1e-3, None)
    p = a / (a + mu)
    return rng.negative_binomial(a, np.clip(p, 1e-6, 1 - 1e-6))
