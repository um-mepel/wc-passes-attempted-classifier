"""Stage-2: hierarchical Bayesian Negative-Binomial passing-rate model (PyMC).

passes ~ NegBinomial(mu, alpha)
log(mu) = log(minutes) + a_player + b_role + s_player_style + g·X + t_competition + p_provider

Partial pooling: a_player ~ N(a_role, σ_p); s_player_style ~ N(s_role_style, σ_s).
The posterior predictive (sampled in predict.py) is what prices Underdog over/unders.

Retraining is one call: HierNB(cfg).fit(features_df). New players/teams just appear
as new factor levels; nothing else changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# All as-of-date (no current-match leakage). Possession enters as ONE own-possession
# feature (team_poss) and ONE opponent-possession feature (opp_poss), each Fotmob-realized
# first with ESPN / pass-proxy fallback (built in features.build). This replaces the old
# collinear pair (team_poss_asof + team_poss_espn, r=+0.72) whose coefficients cancelled
# and flipped the own-possession sign — see docs/MODEL_NOTES / model_fixes. The recent-rate
# anchor sets the per-player level via the player random-effect prior (see fit).
_FEATURES = ["style_per90_recencybiased", "x_poss", "opp_allowed_asof",
             "elo_delta", "opp_poss"]
# NOTE: match-type (qualifier / friendly / tournament) is NOT here — it lives in the
# t_comp effect via the `comp_effect` class (see features.build), so it generalises to
# held-out tournaments instead of being a collinear linear flag.


def _index(series: pd.Series) -> tuple[np.ndarray, list]:
    cats = series.astype("category")
    return cats.cat.codes.values, list(cats.cat.categories)


def _heartbeat(total: int, every: int = 100):
    """File-friendly progress callback for pm.sample — prints a timestamped line
    every `every` draws (instead of PyMC's \\r bar, which doesn't log to files),
    so a redirected log is watchable with `tail -f`."""
    import time as _t
    state = {"chain_t0": {}}

    def cb(trace, draw):
        i = draw.draw_idx + 1
        if i % every and i != total:
            return
        c = draw.chain
        state["chain_t0"].setdefault(c, _t.time())
        elapsed = _t.time() - state["chain_t0"][c]
        rate = i / elapsed if elapsed else 0
        phase = "tune" if getattr(draw, "tuning", False) else "sample"
        print(f"  [fit] chain {c}: {i}/{total} ({phase}) "
              f"{rate:.1f} draws/s, {elapsed:.0f}s elapsed", flush=True)

    return cb


class HierNB:
    def __init__(self, cfg: Config, features: list | None = None,
                 box_emphasis: float | None = None):
        self.cfg = cfg
        self.m = cfg["model"]
        self.idata = None
        self.levels: dict = {}
        # Feature list is overridable so A/B variants (e.g. + magnetism/pressure) can be
        # fit from the SAME class on identical folds. Defaults to the production _FEATURES.
        self.features = list(features) if features is not None else list(_FEATURES)
        # STRIKER box-touch emphasis (poacher vs false-9). >0 turns on a dedicated
        # ST-only slope on box_ratio_asof with prior N(0, box_emphasis); 0 = off, so
        # production is unchanged unless the knob is set. Bigger = more emphasis. Unlike
        # a plain feature (one global slope diluted across all positions and shrunk by
        # the N(0,0.5) beta prior), this slope is fit ONLY on ST rows where the signal
        # lives — mirrors the beta_gk mechanism.
        self.box_emphasis = float(box_emphasis if box_emphasis is not None
                                  else self.m.get("box_emphasis", 0.0))
        self._box_center, self._box_scale = 0.09, 0.05   # ST box_ratio train stats (defaults)

    def _design(self, df: pd.DataFrame, training: bool):
        """Map categorical levels to integer codes, remembering training levels so
        unseen test levels fall back to the pooled group mean (code = -1 handled in model)."""
        out = {}
        for col in ["player_id", "role", "position", "comp_effect", "provider"]:
            if training:
                codes, cats = _index(df[col])
                self.levels[col] = cats
            else:
                cats = self.levels[col]
                codes = df[col].map({c: i for i, c in enumerate(cats)}).fillna(-1).astype(int).values
            out[col] = codes
        # player×style key
        df = df.copy()
        df["pstyle"] = df["player_id"].astype(str) + "|" + df["opp_style"].astype(str)
        if training:
            codes, cats = _index(df["pstyle"]); self.levels["pstyle"] = cats
        else:
            cats = self.levels["pstyle"]
            codes = df["pstyle"].map({c: i for i, c in enumerate(cats)}).fillna(-1).astype(int).values
        out["pstyle"] = codes
        # Standardize/impute with TRAIN statistics only — fitting the scaler on the
        # test batch would leak the test distribution into the design matrix.
        raw = df[self.features]
        if training:
            self._impute = raw.median(numeric_only=True)
            filled = raw.fillna(self._impute).fillna(0.0)
            self._scale_mean = filled.mean(0)
            self._scale_std = filled.std(0) + 1e-9
        else:
            filled = raw.fillna(self._impute).fillna(0.0)
        # clip standardized features to ±8 SD: guards against out-of-range test values
        # (e.g. a StatsBomb-fit scaler meeting the wider Fotmob distribution) exploding
        # exp(log_mu). Genuine features sit well inside this; only artifacts are capped.
        out["X"] = np.clip((filled - self._scale_mean) / self._scale_std, -8.0, 8.0).values
        out["minutes"] = df["minutes"].clip(lower=1).values
        # GOALKEEPER flag (per row): keepers get their own feature slopes + dispersion
        # (see fit). Keyed off the position string so it works for train and predict.
        out["is_gk"] = df["position"].astype(str).str.contains(
            "Goalkeep", case=False, na=False).to_numpy(dtype=float)
        # DISPERSION covariate: build-up players (CB/FB/DM/CM) at high expected possession
        # have a heavy upper pass tail (Bensebaini-type). disp_z = is_buildup·(x_poss-0.5),
        # 0 for keepers/forwards, so only build-up dispersion scales with possession.
        _pos = df["position"].astype(str)
        _buildup = ((_pos.str.contains("Back", case=False, na=False)
                     | _pos.str.contains("Midfield", case=False, na=False))
                    & ~_pos.str.contains("Goalkeep", case=False, na=False)).to_numpy(dtype=float)
        _xp = pd.to_numeric(df.get("x_poss", pd.Series(0.5, index=df.index)),
                            errors="coerce").fillna(0.5).to_numpy()
        out["disp_z"] = _buildup * (_xp - 0.5)
        # STRIKER box-touch slope covariate: is_st · standardized(box_ratio_asof). The
        # is_st gate zeroes it for every non-striker, so beta_box (added in fit only when
        # box_emphasis>0) is estimated from ST rows alone — poachers (high box) pushed
        # down, false-9s (low box) pulled up. Standardized on TRAIN strikers.
        _is_st = (df["role"].astype(str) == "ST")
        _box = pd.to_numeric(df.get("box_ratio_asof", pd.Series(np.nan, index=df.index)),
                             errors="coerce")
        if training:
            _stb = _box[_is_st]
            if _stb.notna().any():
                self._box_center = float(_stb.median())
                self._box_scale = float(_stb.std()) if _stb.notna().sum() > 1 else 0.05
            self._box_scale = (self._box_scale or 0.05) + 1e-9
        _box_z = ((_box.fillna(self._box_center) - self._box_center) / self._box_scale).to_numpy()
        out["box_st"] = _is_st.to_numpy(dtype=float) * _box_z
        # recent-rate anchor (per-90): the model's level baseline (a log-offset).
        out["anchor"] = df.get("anchor_per90", pd.Series(30.0, index=df.index)).clip(lower=1.0).values
        out["y"] = df["passes_attempted"].values if "passes_attempted" in df else None
        return out

    def fit(self, df: pd.DataFrame) -> "HierNB":
        import pymc as pm

        d = self._design(df, training=True)
        n_player = len(self.levels["player_id"])
        n_role = max(1, len(self.levels["role"]))
        n_pos = max(1, len(self.levels["position"]))
        n_comp = len(self.levels["comp_effect"])
        n_prov = max(1, len(self.levels["provider"]))
        n_pstyle = len(self.levels["pstyle"])
        nfx = d["X"].shape[1]

        # INFORMATIVE player-prior centers: each player's recent rate (mean per-90 over
        # their LAST 2 TOURNAMENTS) relative to their role mean. a_player is shrunk toward
        # THIS instead of toward 0, so elite builders keep their own level (fixes the
        # systematic under-prediction) while new/low-data players still pool to the role.
        _t = df.copy()
        _t["_p90"] = _t["passes_attempted"] / _t["minutes"].clip(lower=1) * 90.0
        _t = _t.sort_values("match_date")
        role_mean = _t.groupby("role")["_p90"].mean()
        glob = float(_t["_p90"].mean())

        def _last2(g):
            comps = list(dict.fromkeys(g["competition"].tolist()[::-1]))[:2]
            return g.loc[g["competition"].isin(comps), "_p90"].mean()

        pl_rate = _t.groupby("player_id").apply(_last2)
        pl_role = _t.groupby("player_id")["role"].agg(lambda s: s.mode().iloc[0])
        pl_n = _t.groupby("player_id").size()
        K = 2.0   # data-count shrinkage: thin-data players pulled toward the role mean,
        prior_dev = np.zeros(len(self.levels["player_id"]))   # high-data keep their rate
        for i, pid in enumerate(self.levels["player_id"]):
            rm = float(role_mean.get(pl_role.get(pid), glob))
            pr = float(pl_rate.get(pid, glob))
            shrink = float(pl_n.get(pid, 0)) / (float(pl_n.get(pid, 0)) + K)
            prior_dev[i] = shrink * np.log(max(pr, 1.0) / max(rm, 1.0))

        with pm.Model() as model:
            # NON-CENTERED parameterization for every group effect: sample standard
            # normals and scale by sigma. This avoids hierarchical "funnels" that make
            # NUTS slow/divergent when groups (esp. ~5k sparse player x style levels)
            # have few observations. Deterministic names match what predict.py reads.
            mu = pm.Normal("mu", 3.5, 1.0)                       # ~log(33) global baseline rate
            sigma_role = pm.HalfNormal("sigma_role", 0.5)
            a_role = pm.Deterministic("a_role", mu + pm.Normal("a_role_z", 0, 1, shape=n_role) * sigma_role)
            sigma_pos = pm.HalfNormal("sigma_pos", 0.4)
            b_position = pm.Deterministic("b_position", pm.Normal("b_position_z", 0, 1, shape=n_pos) * sigma_pos)

            # PLAYER effect shrunk toward the player's OWN recent rate (prior_dev), not 0,
            # so genuinely high-volume passers (top-side CBs/DMs) keep their level instead
            # of being over-pooled to the role mean. Student-t for heavy tails.
            sigma_player = pm.HalfNormal("sigma_player", 0.5)
            a_player = pm.Deterministic(
                "a_player", prior_dev + pm.StudentT("a_player_z", nu=4, mu=0, sigma=1, shape=n_player) * sigma_player)

            sigma_pstyle = pm.HalfNormal("sigma_pstyle", 0.3)
            s_pstyle = pm.Deterministic("s_pstyle", pm.Normal("s_pstyle_z", 0, 1, shape=n_pstyle) * sigma_pstyle)

            t_comp = pm.Normal("t_comp", 0.0, 0.3, shape=n_comp)
            p_prov = pm.Normal("p_prov", 0.0, 0.3, shape=n_prov)
            beta = pm.Normal("beta", 0.0, 0.5, shape=nfx)
            # GOALKEEPER-SPECIFIC SLOPES (partial pooling): keepers respond to the
            # covariates differently from outfielders — most starkly, team possession
            # correlates NEGATIVELY with GK pass volume (-0.22) but POSITIVELY for
            # outfielders (+0.52), and the per-player anchor carries ~no GK signal. One
            # shared beta (outfielders outnumber GKs ~17:1) is therefore wrong-signed for
            # keepers. beta_gk is an additive deviation applied ONLY to GK rows; the
            # GK slope is (beta + beta_gk). sigma_beta_gk shrinks it toward the shared
            # slope when GK data is thin, so it can't overfit the ~2k GK rows.
            sigma_beta_gk = pm.HalfNormal("sigma_beta_gk", 0.5)
            beta_gk = pm.Deterministic("beta_gk", pm.Normal("beta_gk_z", 0, 1, shape=nfx) * sigma_beta_gk)
            # GK pass volume is ~2.5x less variable than outfielders' (sd 9.2 vs 23.1),
            # so a shared alpha makes GK over/under intervals too wide. Own dispersion.
            alpha_gk = pm.Exponential("alpha_gk", 1.0)
            # OUTFIELD dispersion is POSSESSION-SCALED: the mean can't catch the heavy
            # upper tail of high-possession build-up players (irreducible ~48% noise), so
            # the tail must live in the variance. alpha = exp(log_alpha + gamma_disp·disp_z);
            # gamma_disp<0 => higher possession -> smaller alpha -> fatter tail -> calibrated
            # over probabilities even when the point estimate undershoots a Bensebaini.
            log_alpha = pm.Normal("log_alpha", 1.5, 1.0)         # base outfield log-dispersion
            gamma_disp = pm.Normal("gamma_disp", 0.0, 0.5)
            alpha_out = pm.math.exp(log_alpha + gamma_disp * d["disp_z"])   # per-row

            # STRIKER box-touch slope (poacher vs false-9). Only added when box_emphasis>0;
            # prior N(0, box_emphasis) — a wider knob than the shared beta prior (0.5) so the
            # slope can carry the full ~7-pass poacher/false-9 swing instead of being shrunk.
            if self.box_emphasis and self.box_emphasis > 0:
                beta_box = pm.Normal("beta_box", 0.0, self.box_emphasis)
                box_term = beta_box * d["box_st"]
            else:
                box_term = 0.0

            def gather(arr, idx, fill=0.0):
                safe = np.where(idx < 0, 0, idx)
                val = arr[safe]
                return pm.math.switch(idx < 0, fill, val)

            log_mu = (
                np.log(d["minutes"])
                + a_role[np.where(d["role"] < 0, 0, d["role"])]
                + gather(b_position, d["position"])
                + gather(a_player, d["player_id"])           # centred on player's recent rate
                + gather(s_pstyle, d["pstyle"])
                + t_comp[np.where(d["comp_effect"] < 0, 0, d["comp_effect"])]
                + gather(p_prov, d["provider"])
                + pm.math.dot(d["X"], beta)
                + d["is_gk"] * pm.math.dot(d["X"], beta_gk)   # GK-only slope deviation
                + box_term                                    # ST-only box-touch slope
            )
            # per-row dispersion: keepers use their own (tighter) alpha_gk, outfielders
            # use the possession-scaled alpha_out.
            alpha_row = pm.math.switch(d["is_gk"] > 0.5, alpha_gk, alpha_out)
            pm.NegativeBinomial("y", mu=pm.math.exp(log_mu), alpha=alpha_row, observed=d["y"])

            # cores=1 -> sample chains sequentially in-process. This avoids the
            # macOS multiprocessing/Accelerate fork crash (EOFError) that kills
            # parallel chain workers; default to safe sequential sampling.
            # env overrides let parallel jobs tune their own resource use
            import os
            cores = int(os.environ.get("PM_CORES", self.m.get("cores", 1)))
            draws = int(os.environ.get("PM_DRAWS", self.m["draws"]))
            tune = int(os.environ.get("PM_TUNE", self.m["tune"]))
            total = tune + draws
            common = dict(draws=draws, tune=tune, chains=self.m["chains"],
                          target_accept=self.m["target_accept"], random_seed=self.m["seed"])
            sampler = os.environ.get("PM_SAMPLER", self.m.get("sampler", "nutpie"))
            try:
                if sampler != "nutpie":
                    raise RuntimeError("sampler!=nutpie")
                import nutpie  # noqa: F401
                print("[fit] sampling with nutpie (Rust NUTS)...", flush=True)
                # nutpie parallelises chains itself; no cores/mp_ctx/heartbeat callback.
                self.idata = pm.sample(**common, nuts_sampler="nutpie", progressbar=True)
            except Exception as e:                          # fall back to the stock sampler
                print(f"[fit] nutpie unavailable/failed ({type(e).__name__}: {e}); "
                      "falling back to pymc sampler", flush=True)
                self.idata = pm.sample(**common, cores=cores, progressbar=False,
                                       callback=_heartbeat(total, every=100),
                                       **({"mp_ctx": "spawn"} if cores > 1 else {}))
        self._model = model
        return self

    def save(self, path: str | Path) -> None:
        """Persist posterior + level maps + scaler in one pickle. Pickle avoids the
        fragile NetCDF backend chain (netCDF4/h5netcdf/h5py) — InferenceData wraps
        xarray Datasets which pickle cleanly."""
        import pickle
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as fh:
            pickle.dump({"idata": self.idata, "levels": self.levels, "impute": self._impute,
                         "scale_mean": self._scale_mean, "scale_std": self._scale_std,
                         "features": self.features, "box_emphasis": self.box_emphasis,
                         "box_center": self._box_center, "box_scale": self._box_scale}, fh)

    @classmethod
    def load(cls, cfg: Config, path: str | Path) -> "HierNB":
        import pickle
        obj = cls(cfg)
        with open(Path(path) / "model.pkl", "rb") as fh:
            blob = pickle.load(fh)
        obj.idata, obj.levels = blob["idata"], blob["levels"]
        obj._impute, obj._scale_mean, obj._scale_std = blob["impute"], blob["scale_mean"], blob["scale_std"]
        # older pickles predate the overridable feature list -> fall back to production _FEATURES
        obj.features = blob.get("features", list(_FEATURES))
        # older pickles predate the box-emphasis term -> off (0.0), defaults for center/scale
        obj.box_emphasis = blob.get("box_emphasis", 0.0)
        obj._box_center = blob.get("box_center", 0.09)
        obj._box_scale = blob.get("box_scale", 0.05)
        return obj
