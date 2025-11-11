from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd


# ========= Univariate signals =========

@dataclass
class UnivariateSignals:
    d1: pd.DataFrame
    d1_z: pd.DataFrame
    d1_large_move: pd.DataFrame
    d2: pd.DataFrame
    inflection_flag: pd.DataFrame          # any inflection (up or down)
    inflection_direction: pd.DataFrame     # "up", "down", or ""
    range_norm: pd.DataFrame
    range_edge_flag: pd.DataFrame
    range_mid_stable: pd.DataFrame


def _zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    win = max(5, window // 2)
    mu = df.rolling(window, min_periods=win).mean()
    sig = df.rolling(window, min_periods=win).std()
    return (df - mu) / sig


def compute_univariate_signals(
    X: pd.DataFrame,
    window_range: int = 24,
    window_z: int = 24,
    z_threshold: float = 2.0,
    edge_band: float = 0.1,
    mid_band: Tuple[float, float] = (0.2, 0.8),
    min_inflection_gap_months: int = 3,
) -> UnivariateSignals:
    """
    Per-series diagnostics:

      - d1: first difference
      - d1_z: rolling z-score of d1
      - d1_large_move: |d1_z| >= z_threshold
      - d2: second difference
      - inflection_flag / direction:
            slope sign flip + curvature + local extremum + min spacing
      - range_norm: position in rolling [min,max] (0..1)
      - range_edge_flag: near edges of rolling range
      - range_mid_stable: mostly in middle band of range
    """
    X = X.sort_index()

    # First & second differences
    d1 = X.diff()
    d1_z = _zscore(d1, window_z)
    d1_large = (d1_z.abs() >= z_threshold)

    d2 = d1.diff()
    d2_z = _zscore(d2, window_z)

    # Range-boundness
    win_r = max(6, window_range // 2)
    roll_min = X.rolling(window_range, min_periods=win_r).min()
    roll_max = X.rolling(window_range, min_periods=win_r).max()
    span = roll_max - roll_min

    range_norm = (X - roll_min) / span
    range_norm = range_norm.clip(lower=0.0, upper=1.0)

    lo, hi = mid_band
    range_edge = (range_norm <= edge_band) | (range_norm >= 1 - edge_band)

    mid_mask = (range_norm >= lo) & (range_norm <= hi)
    range_mid_stable = (
        mid_mask.rolling(window_range, min_periods=win_r)
        .mean()
        >= 0.8
    )

    # Inflections with extra discipline
    sign_d1 = np.sign(d1)

    cols = X.columns
    inflection_flag = pd.DataFrame(False, index=X.index, columns=cols)
    inflection_direction = pd.DataFrame("", index=X.index, columns=cols, dtype=object)

    def _enforce_min_gap(flag: pd.Series, months: int) -> pd.Series:
        if months <= 0:
            return flag
        idx = flag[flag].index
        if idx.empty:
            return flag * False
        keep = []
        last = None
        min_days = 30 * months
        for t in idx:
            if last is None or (t - last).days >= min_days:
                keep.append(t)
                last = t
        out = pd.Series(False, index=flag.index)
        out[keep] = True
        return out

    for col in cols:
        s = X[col]
        if s.dropna().size < 10:
            continue

        sd1 = sign_d1[col]
        curv = d2_z[col].abs() >= 1.0  # require meaningful curvature

        # Candidate flips
        up = (sd1.shift(1) < 0) & (sd1 > 0) & curv
        down = (sd1.shift(1) > 0) & (sd1 < 0) & curv

        # Local extremum around the turn (use t-1 as the extremum point)
        # Up (trough): s_{t-1} < s_{t-2}, s_{t-1} < s_t
        up = up & (s.shift(1) < s.shift(2)) & (s.shift(1) < s)

        # Down (peak): s_{t-1} > s_{t-2}, s_{t-1} > s_t
        down = down & (s.shift(1) > s.shift(2)) & (s.shift(1) > s)

        pre_z = d1_z[col].shift(1).abs()
        post_z = d1_z[col].abs()
        strength = pre_z.combine(post_z, func=max)
        up = up & (strength >= 1.0)
        down = down & (strength >= 1.0)

        # Enforce minimum spacing between inflections
        up = _enforce_min_gap(up.fillna(False), min_inflection_gap_months)
        down = _enforce_min_gap(down.fillna(False), min_inflection_gap_months)

        col_flag = up | down
        inflection_flag[col] = col_flag

        col_dir = pd.Series("", index=X.index, dtype=object)
        col_dir[up] = "up"
        col_dir[down] = "down"
        inflection_direction[col] = col_dir

    return UnivariateSignals(
        d1=d1,
        d1_z=d1_z,
        d1_large_move=d1_large.fillna(False),
        d2=d2,
        inflection_flag=inflection_flag.fillna(False),
        inflection_direction=inflection_direction,
        range_norm=range_norm,
        range_edge_flag=range_edge.fillna(False),
        range_mid_stable=range_mid_stable.fillna(False),
    )


# ========= Nonlinearity (linear vs quadratic trend) =========

@dataclass
class NonlinearitySignals:
    r2_linear: pd.DataFrame
    r2_quadratic: pd.DataFrame
    r2_gap: pd.DataFrame
    nonlinear_flag: pd.DataFrame


def _rolling_trend_r2(
    s: pd.Series,
    window: int,
    quad: bool = False,
) -> pd.Series:
    """
    Rolling R^2 of linear or quadratic trend of s on time index.
    R^2 is clamped to [0, 1]; invalid values -> NaN.
    """
    s = s.dropna()
    if s.empty:
        return pd.Series(index=[], dtype=float)

    idx = s.index
    y_all = s.values.astype(float)
    n = len(y_all)
    r2 = np.full(n, np.nan)

    for end in range(window - 1, n):
        start = end - window + 1
        y = y_all[start : end + 1]
        t = np.arange(len(y), dtype=float)

        if quad:
            Xmat = np.column_stack([np.ones_like(t), t, t**2])
        else:
            Xmat = np.column_stack([np.ones_like(t), t])

        try:
            beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
            y_hat = Xmat @ beta
            ss_res = ((y - y_hat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            if ss_tot <= 0:
                r2_val = np.nan
            else:
                r2_val = 1.0 - ss_res / ss_tot
                # clamp to [0,1]
                if not np.isfinite(r2_val) or r2_val < 0 or r2_val > 1:
                    r2_val = np.nan
            r2[end] = r2_val
        except Exception:
            r2[end] = np.nan

    return pd.Series(r2, index=idx, name=s.name)


def compute_nonlinearity_windows(
    X: pd.DataFrame,
    window: int = 36,
    min_r2_linear: float = 0.7,
    gap_threshold: float = 0.15,
) -> NonlinearitySignals:
    """
    For each series:

      - Rolling linear trend: x ~ t
      - Rolling quadratic: x ~ t + t^2
      - ΔR² = R²_quad - R²_lin

    nonlinear_flag:
      - R²_lin >= min_r2_linear
      - ΔR² >= gap_threshold
    """
    X = X.sort_index()
    cols = X.columns

    r2_lin_df = pd.DataFrame(index=X.index, columns=cols, dtype=float)
    r2_quad_df = pd.DataFrame(index=X.index, columns=cols, dtype=float)

    for col in cols:
        s = X[col].dropna()
        if s.size < window + 5:
            continue

        r2_lin = _rolling_trend_r2(s, window=window, quad=False)
        r2_quad = _rolling_trend_r2(s, window=window, quad=True)

        r2_lin_df.loc[r2_lin.index, col] = r2_lin
        r2_quad_df.loc[r2_quad.index, col] = r2_quad

    r2_gap = r2_quad_df - r2_lin_df

    # Only trust gaps where both R² are valid
    valid = r2_lin_df.notna() & r2_quad_df.notna()
    r2_gap = r2_gap.where(valid)

    nonlinear_flag = (
        (r2_lin_df >= min_r2_linear)
        & (r2_gap >= gap_threshold)
    )

    return NonlinearitySignals(
        r2_linear=r2_lin_df,
        r2_quadratic=r2_quad_df,
        r2_gap=r2_gap,
        nonlinear_flag=nonlinear_flag.fillna(False),
    )


# ========= Threshold nonlinearity (optional cross-series) =========

@dataclass
class ThresholdNonlinearityResult:
    x_col: str
    y_col: str
    best_threshold: float
    r2_linear: float
    r2_piecewise: float
    improvement: float
    nonlinear: bool


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    if y.size < 5:
        return np.nan
    X_design = np.column_stack([np.ones_like(X), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot <= 0:
        return np.nan
    r2 = 1.0 - ss_res / ss_tot
    if r2 < 0 or r2 > 1 or not np.isfinite(r2):
        return np.nan
    return r2


def _threshold_piecewise_r2(
    x: np.ndarray,
    y: np.ndarray,
    n_grid: int = 20,
) -> Tuple[float, float]:
    if y.size < 30:
        return np.nan, np.nan

    lo = np.quantile(x, 0.1)
    hi = np.quantile(x, 0.9)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return np.nan, np.nan

    best_r2 = -np.inf
    best_c = np.nan

    for c in np.linspace(lo, hi, n_grid):
        left = x <= c
        right = x > c
        if left.sum() < 10 or right.sum() < 10:
            continue

        r2_left = _ols_r2(y[left], x[left])
        r2_right = _ols_r2(y[right], x[right])
        if not np.isfinite(r2_left) or not np.isfinite(r2_right):
            continue

        n_total = y.size
        r2_combined = (
            r2_left * (left.sum() / n_total)
            + r2_right * (right.sum() / n_total)
        )

        if np.isfinite(r2_combined) and r2_combined > best_r2:
            best_r2 = r2_combined
            best_c = c

    if best_r2 <= 0 or not np.isfinite(best_r2):
        return np.nan, np.nan

    return best_c, best_r2


def compute_threshold_nonlinearity(
    X: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    min_obs: int = 60,
    improvement_min: float = 0.15,
    n_grid: int = 20,
) -> List[ThresholdNonlinearityResult]:
    results: List[ThresholdNonlinearityResult] = []

    for x_col, y_col in pairs:
        if x_col not in X.columns or y_col not in X.columns:
            continue

        df = X[[x_col, y_col]].dropna()
        if len(df) < min_obs:
            continue

        x = df[x_col].values
        y = df[y_col].values

        r2_lin = _ols_r2(y, x)
        if not np.isfinite(r2_lin):
            continue

        c_star, r2_piece = _threshold_piecewise_r2(x, y, n_grid=n_grid)
        if not np.isfinite(c_star) or not np.isfinite(r2_piece):
            continue

        improvement = r2_piece - r2_lin
        nonlinear = improvement >= improvement_min

        results.append(
            ThresholdNonlinearityResult(
                x_col=x_col,
                y_col=y_col,
                best_threshold=float(c_star),
                r2_linear=float(r2_lin),
                r2_piecewise=float(r2_piece),
                improvement=float(improvement),
                nonlinear=bool(nonlinear),
            )
        )

    return results


# ========= High-level pack =========

@dataclass
class SignalPack:
    univariate: UnivariateSignals
    nonlinear: NonlinearitySignals
    threshold_pairs: List[ThresholdNonlinearityResult]


def run_all_signals(
    X: pd.DataFrame,
    threshold_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> SignalPack:
    uni = compute_univariate_signals(X)
    nl = compute_nonlinearity_windows(X)
    th = compute_threshold_nonlinearity(X, threshold_pairs) if threshold_pairs else []
    return SignalPack(univariate=uni, nonlinear=nl, threshold_pairs=th)


# ========= Standardized signals =========

@dataclass
class StandardizedSignals:
    level_z: pd.DataFrame
    d1_z: pd.DataFrame
    range_pos: pd.DataFrame
    nonlinearity_score: pd.DataFrame
    big_move_flag: pd.DataFrame
    inflection_up: pd.DataFrame
    inflection_down: pd.DataFrame


def collect_standardized_signals(
    X: pd.DataFrame,
    pack: SignalPack,
    level_window: Optional[int] = None,
    nl_scale: float = 0.15,
    cap_nl: float = 3.0,
) -> StandardizedSignals:
    """
    Build standardized signal panels:

      - level_z: standardized level
      - d1_z: from univariate
      - range_pos: 0..1 within rolling range
      - nonlinearity_score: ΔR²-based, >0 only where nonlinear_flag True
      - big_move_flag: 0/1
      - inflection_up / inflection_down: 0/1
    """
    uni = pack.univariate
    nl = pack.nonlinear

    # Level z
    if level_window is not None:
        level_z = _zscore(X, level_window)
    else:
        mu = X.mean()
        sig = X.std(ddof=0).replace(0, np.nan)
        level_z = (X - mu) / sig

    # d1_z
    d1_z = uni.d1_z

    # Range position
    range_pos = uni.range_norm

    # Nonlinearity score: only where nonlinear_flag is True
    gap_pos = nl.r2_gap.clip(lower=0.0).fillna(0.0)

    if nl_scale and nl_scale > 0:
        scaled = gap_pos / nl_scale
    else:
        scaled = gap_pos.copy()

    if cap_nl is not None:
        scaled = scaled.clip(upper=cap_nl)

    nl_score = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    mask_nl = nl.nonlinear_flag & (gap_pos > 0)
    nl_score[mask_nl] = scaled[mask_nl]

    # Flags
    big = uni.d1_large_move.astype(int)
    inf_up = (uni.inflection_direction == "up").astype(int)
    inf_down = (uni.inflection_direction == "down").astype(int)

    return StandardizedSignals(
        level_z=level_z,
        d1_z=d1_z,
        range_pos=range_pos,
        nonlinearity_score=nl_score,
        big_move_flag=big,
        inflection_up=inf_up,
        inflection_down=inf_down,
    )
