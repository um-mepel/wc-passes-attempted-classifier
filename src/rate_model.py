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

# All as-of-date (computed from strictly earlier matches) — no current-match leakage.
_FEATURES = ["recent_per90", "style_per90_recencybiased", "team_poss_asof",
             "opp_allowed_asof"]


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
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.m = cfg["model"]
        self.idata = None
        self.levels: dict = {}

    def _design(self, df: pd.DataFrame, training: bool):
        """Map categorical levels to integer codes, remembering training levels so
        unseen test levels fall back to the pooled group mean (code = -1 handled in model)."""
        out = {}
        for col in ["player_id", "role", "position", "competition", "provider"]:
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
        raw = df[_FEATURES]
        if training:
            self._impute = raw.median(numeric_only=True)
            filled = raw.fillna(self._impute).fillna(0.0)
            self._scale_mean = filled.mean(0)
            self._scale_std = filled.std(0) + 1e-9
        else:
            filled = raw.fillna(self._impute).fillna(0.0)
        out["X"] = ((filled - self._scale_mean) / self._scale_std).values
        out["minutes"] = df["minutes"].clip(lower=1).values
        out["y"] = df["passes_attempted"].values if "passes_attempted" in df else None
        return out

    def fit(self, df: pd.DataFrame) -> "HierNB":
        import pymc as pm

        d = self._design(df, training=True)
        n_player = len(self.levels["player_id"])
        n_role = max(1, len(self.levels["role"]))
        n_pos = max(1, len(self.levels["position"]))
        n_comp = len(self.levels["competition"])
        n_prov = max(1, len(self.levels["provider"]))
        n_pstyle = len(self.levels["pstyle"])
        nfx = d["X"].shape[1]

        with pm.Model() as model:
            # NON-CENTERED parameterization for every group effect: sample standard
            # normals and scale by sigma. This avoids hierarchical "funnels" that make
            # NUTS slow/divergent when groups (esp. ~5k sparse player x style levels)
            # have few observations. Deterministic names match what predict.py reads.
            mu = pm.Normal("mu", 3.5, 1.0)                       # ~log(33) global baseline rate
            sigma_role = pm.HalfNormal("sigma_role", 0.5)
            a_role = pm.Deterministic("a_role", mu + pm.Normal("a_role_z", 0, 1, shape=n_role) * sigma_role)

            sigma_player = pm.HalfNormal("sigma_player", 0.5)
            a_player = pm.Deterministic("a_player", pm.Normal("a_player_z", 0, 1, shape=n_player) * sigma_player)

            # granular position effect (finer than role), partially pooled toward 0
            sigma_pos = pm.HalfNormal("sigma_pos", 0.4)
            b_position = pm.Deterministic("b_position", pm.Normal("b_position_z", 0, 1, shape=n_pos) * sigma_pos)

            sigma_pstyle = pm.HalfNormal("sigma_pstyle", 0.3)
            s_pstyle = pm.Deterministic("s_pstyle", pm.Normal("s_pstyle_z", 0, 1, shape=n_pstyle) * sigma_pstyle)

            t_comp = pm.Normal("t_comp", 0.0, 0.3, shape=n_comp)
            p_prov = pm.Normal("p_prov", 0.0, 0.3, shape=n_prov)
            beta = pm.Normal("beta", 0.0, 0.5, shape=nfx)
            alpha = pm.Exponential("alpha", 1.0)                 # NB dispersion

            def gather(arr, idx, fill=0.0):
                safe = np.where(idx < 0, 0, idx)
                val = arr[safe]
                return pm.math.switch(idx < 0, fill, val)

            log_mu = (
                np.log(d["minutes"])
                + a_role[np.where(d["role"] < 0, 0, d["role"])]
                + gather(b_position, d["position"])
                + gather(a_player, d["player_id"])
                + gather(s_pstyle, d["pstyle"])
                + t_comp[np.where(d["competition"] < 0, 0, d["competition"])]
                + gather(p_prov, d["provider"])
                + pm.math.dot(d["X"], beta)
            )
            pm.NegativeBinomial("y", mu=pm.math.exp(log_mu), alpha=alpha, observed=d["y"])

            # cores=1 -> sample chains sequentially in-process. This avoids the
            # macOS multiprocessing/Accelerate fork crash (EOFError) that kills
            # parallel chain workers; default to safe sequential sampling.
            # env overrides let parallel jobs tune their own resource use
            import os
            cores = int(os.environ.get("PM_CORES", self.m.get("cores", 1)))
            draws = int(os.environ.get("PM_DRAWS", self.m["draws"]))
            tune = int(os.environ.get("PM_TUNE", self.m["tune"]))
            total = tune + draws
            self.idata = pm.sample(
                draws=draws, tune=tune, chains=self.m["chains"],
                cores=cores, target_accept=self.m["target_accept"],
                random_seed=self.m["seed"], progressbar=False,
                callback=_heartbeat(total, every=100),
                **({"mp_ctx": "spawn"} if cores > 1 else {}),
            )
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
                         "scale_mean": self._scale_mean, "scale_std": self._scale_std}, fh)

    @classmethod
    def load(cls, cfg: Config, path: str | Path) -> "HierNB":
        import pickle
        obj = cls(cfg)
        with open(Path(path) / "model.pkl", "rb") as fh:
            blob = pickle.load(fh)
        obj.idata, obj.levels = blob["idata"], blob["levels"]
        obj._impute, obj._scale_mean, obj._scale_std = blob["impute"], blob["scale_mean"], blob["scale_std"]
        return obj
