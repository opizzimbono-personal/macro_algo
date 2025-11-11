# features/transforms.py

import numpy as np
import pandas as pd


def compute_derivatives(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    First and second discrete differences of a Series.
    """
    s = s.sort_index()
    dx = s.diff()
    ddx = dx.diff()
    dx.name = f"{s.name}_d1"
    ddx.name = f"{s.name}_d2"
    return dx, ddx


def compute_range_stats(s: pd.Series,
                        band_width: float = 1.5) -> dict:
    """
    Basic stats and symmetric band around mean.

    band_width: multiple of std for [lower, upper].
    """
    s_clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s_clean.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "lower": np.nan,
            "upper": np.nan,
        }

    mu = s_clean.mean()
    sigma = s_clean.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        lower = upper = mu
    else:
        lower = mu - band_width * sigma
        upper = mu + band_width * sigma

    return {
        "mean": float(mu),
        "std": float(sigma),
        "lower": float(lower),
        "upper": float(upper),
    }


def compute_state_code(s: pd.Series,
                       lower: float,
                       upper: float,
                       edge_pct: float = 0.1) -> pd.Series:
    """
    Map each observation to:
      0 = comfortably inside band
      1 = near band edges
      2 = outside band

    edge_pct: thickness of edge zone as fraction of band width.
    """
    s = s.sort_index()
    state = pd.Series(index=s.index, dtype="float")

    if np.isnan(lower) or np.isnan(upper) or lower == upper:
        # Can't form a band; mark all NaN
        return state.rename(f"{s.name}_state012")

    span = upper - lower
    inner_low = lower + edge_pct * span
    inner_high = upper - edge_pct * span

    # Outside
    state[(s < lower) | (s > upper)] = 2
    # Near edges
    state[(s >= lower) & (s <= upper) &
          ((s <= inner_low) | (s >= inner_high))] = 1
    # Inside
    state[(s > inner_low) & (s < inner_high)] = 0

    return state.rename(f"{s.name}_state012")


def feature_bundle(s: pd.Series,
                   band_width: float = 1.5,
                   edge_pct: float = 0.1) -> tuple[pd.DataFrame, dict]:
    """
    Given a 1D series, return:
      - DataFrame with:
          level, d1, d2, z, state012
      - metadata dict with band + stats
    """
    s = s.sort_index()
    s = s.replace([np.inf, -np.inf], np.nan)

    dx, ddx = compute_derivatives(s)

    stats = compute_range_stats(s, band_width=band_width)
    state = compute_state_code(s, stats["lower"], stats["upper"], edge_pct=edge_pct)

    if stats["std"] and not np.isnan(stats["std"]) and stats["std"] != 0:
        z = (s - stats["mean"]) / stats["std"]
    else:
        z = pd.Series(index=s.index, dtype="float")
    z = z.rename(f"{s.name}_z")

    cols = [
        s.rename(s.name),
        dx,
        ddx,
        z,
        state,
    ]
    out = pd.concat(cols, axis=1)

    return out, stats
