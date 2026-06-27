"""Combine stage-1 (minutes) and stage-2 (rate) into a per-row posterior predictive
distribution of passes attempted, by Monte Carlo:

    for each posterior draw θ and each sampled minutes m:
        passes ~ NegBinomial(mean = exp(logμ(θ) with offset log m), alpha(θ))

The resulting sample per player-match IS what we price Underdog over/unders against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .minutes_model import MinutesModel
from .rate_model import HierNB, _FEATURES


def _linpred_samples(model: HierNB, df: pd.DataFrame, n_draws: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Return (eta, alpha) where eta is (n_rows, n_draws) log-rate WITHOUT the minutes
    offset, alpha is (n_draws,) dispersion. Unseen levels contribute 0 (pooled mean)."""
    post = model.idata.posterior
    d = model._design(df, training=False)

    def stack(name):
        return post[name].stack(s=("chain", "draw")).values  # (levels, S) or (S,)

    S = post.dims["chain"] * post.dims["draw"]
    take = rng.choice(S, size=n_draws, replace=n_draws > S)

    a_role = stack("a_role")[:, take]
    b_pos = stack("b_position")[:, take]
    a_player = stack("a_player")[:, take]
    s_pstyle = stack("s_pstyle")[:, take]
    t_comp = stack("t_comp")[:, take]
    p_prov = stack("p_prov")[:, take]
    beta = stack("beta")[:, take]            # (nfx, n_draws)
    alpha = stack("alpha")[take]

    def gather(arr, idx):
        out = np.zeros((len(idx), n_draws))
        ok = idx >= 0
        out[ok] = arr[idx[ok]]
        return out

    X = d["X"]                                # (n_rows, nfx)
    # a_player already carries each player's level (prior-centred on their recent rate);
    # unseen players (code -1) contribute 0 -> fall back to the role mean a_role.
    eta = (
        gather(a_role.T.T, np.where(d["role"] < 0, 0, d["role"]))  # role always present
        + gather(b_pos, d["position"])
        + gather(a_player, d["player_id"])
        + gather(s_pstyle, d["pstyle"])
        + gather(t_comp, np.where(d["comp_effect"] < 0, 0, d["comp_effect"]))
        + gather(p_prov, d["provider"])
        + X @ beta
    )
    return eta, alpha


def posterior_predictive(model: HierNB, minutes: MinutesModel, df: pd.DataFrame,
                         p_start: np.ndarray, n_draws: int = 400, n_min: int = 1,
                         use_actual_minutes: bool = False, seed: int = 0) -> np.ndarray:
    """Return an (n_rows, n_draws) sample of passes attempted."""
    rng = np.random.default_rng(seed)
    eta, alpha = _linpred_samples(model, df, n_draws, rng)            # (rows, draws), (draws,)
    if use_actual_minutes:
        m = np.repeat(df["minutes"].clip(lower=1).values[:, None], n_draws, axis=1)
    else:
        pids = df["player_id"].values if "player_id" in df else None
        m = minutes.sample_minutes(p_start, n_draws, rng, player_ids=pids)  # (rows, draws)
    mu = np.exp(eta + np.log(np.clip(m, 1, None)))
    # NB sampling: variance = mu + mu^2/alpha  (pymc alpha convention)
    p = alpha[None, :] / (alpha[None, :] + mu)
    return rng.negative_binomial(np.clip(alpha[None, :], 1e-3, None), np.clip(p, 1e-6, 1 - 1e-6))


def prob_over(samples: np.ndarray, line: np.ndarray) -> np.ndarray:
    """P(passes ≥ ceil(line)) per row from the predictive sample."""
    thr = np.ceil(np.asarray(line))[:, None]
    return (samples >= thr).mean(axis=1)


def summarize(samples: np.ndarray, ci: float = 0.80) -> "pd.DataFrame":
    """Per-row point estimate + credible interval from the predictive samples.

    The point estimate is the predictive MEAN (continuous, e.g. 36.2 — not an
    integer), and the interval is the central `ci` credible band (e.g. 80% -> the
    10th–90th percentiles), i.e. the margin of error around the prediction.
    """
    import pandas as pd
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    mean = samples.mean(axis=1)
    lo, med, hi = np.percentile(samples, [lo_q, 50, hi_q], axis=1)
    return pd.DataFrame({
        "pred": np.round(mean, 1),
        "ci_low": np.round(lo, 1),
        "ci_high": np.round(hi, 1),
        "moe": np.round((hi - lo) / 2, 1),       # +/- margin of error
        "p50": med,
        "std": np.round(samples.std(axis=1), 1),
    })
