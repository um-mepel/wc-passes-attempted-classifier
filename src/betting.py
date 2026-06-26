"""Underdog pick'em edge layer.

Underdog posts ONE projection line; you pick MORE/LESS at fixed parlay multipliers.
There is no two-sided price to de-vig, so the bar is the multiplier break-even.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_leg_breakeven(n_picks: int, multiplier: float) -> float:
    """Independent-legs break-even probability per leg for an N-pick parlay paying M×."""
    return (1.0 / multiplier) ** (1.0 / n_picks)


# Underdog Standard multipliers (typical; override per current promo).
STANDARD_MULTIPLIER = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0}


def edge_table(df: pd.DataFrame, p_over: np.ndarray, n_picks: int = 5) -> pd.DataFrame:
    """Given rows with a 'line' column and model P(over), return a ranked edge table.

    pick = MORE if P(over) > 0.5 else LESS; edge = |P - 0.5| surplus over the per-leg bar.
    """
    bar = per_leg_breakeven(n_picks, STANDARD_MULTIPLIER.get(n_picks, 20.0))
    out = df.copy()
    out["p_over"] = p_over
    out["pick"] = np.where(out["p_over"] >= 0.5, "MORE", "LESS")
    out["p_pick"] = np.where(out["pick"] == "MORE", out["p_over"], 1 - out["p_over"])
    out["breakeven"] = bar
    out["edge"] = out["p_pick"] - bar          # >0 ⇒ +EV leg at this parlay size
    return out.sort_values("edge", ascending=False)


def grade(df: pd.DataFrame) -> dict:
    """Backtest grading: hit-rate and naive flat-stake ROI of the selected legs."""
    graded = df.dropna(subset=["passes_attempted", "line", "pick"]).copy()
    actual_over = graded["passes_attempted"] >= np.ceil(graded["line"])
    won = np.where(graded["pick"] == "MORE", actual_over, ~actual_over)
    n = len(graded)
    return {
        "n_plays": int(n),
        "hit_rate": float(won.mean()) if n else float("nan"),
        # single-leg flat-stake ROI proxy at ~1.87 fair (≈ -110): payout 0.87 win, -1 loss
        "roi_singleleg": float(np.mean(np.where(won, 0.87, -1.0))) if n else float("nan"),
    }
