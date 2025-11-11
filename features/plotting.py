from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .signal_engine import (
    UnivariateSignals,
    NonlinearitySignals,
    StandardizedSignals,
)


# ---------- helpers ----------

def _shade_flag_spans(
    ax,
    idx: pd.DatetimeIndex,
    flag: pd.Series,
    color: str = "red",
    alpha: float = 0.12,
) -> None:
    """
    Shade contiguous regions where `flag` is True over idx.
    """
    if flag is None:
        return

    f = pd.Series(flag, index=flag.index).reindex(idx).fillna(False)

    in_span = False
    start = None
    for t, on in f.items():
        if on and not in_span:
            in_span = True
            start = t
        elif not on and in_span:
            ax.axvspan(start, t, color=color, alpha=alpha)
            in_span = False

    if in_span and start is not None:
        ax.axvspan(start, idx[-1], color=color, alpha=alpha)


def _year_mask(
    idx: pd.DatetimeIndex,
    start_year: Optional[int],
    end_year: Optional[int],
) -> pd.Series:
    m = pd.Series(True, index=idx)
    if start_year is not None:
        m &= idx.year >= start_year
    if end_year is not None:
        m &= idx.year <= end_year
    return m


# ---------- single-series diagnostic (RAW view) ----------

def plot_series_diagnostics(
    X: pd.DataFrame,
    col: str,
    uni: UnivariateSignals,
    nl: NonlinearitySignals,
    meta: Optional[dict] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> None:
    """
    RAW VIEW: three-panel diagnostic for a single series `col`:

      Panel 1: Level
        - raw series
        - red shaded regions: nonlinear regime (nl.nonlinear_flag)
        - ▲ upward inflection (slope -→+)
        - ▼ downward inflection (slope +→-)

      Panel 2: First difference
        - d1
        - dots: large moves (|z(d1)| >= threshold)
        - ▲ / ▼ inflections

      Panel 3: ΔR² nonlinearity
        - line: R²_quad - R²_linear
        - dashed: threshold (0.15)
        - x: nonlinear_flag points
    """
    if col not in X.columns:
        raise KeyError(f"{col} not in X.columns")

    if meta is None:
        meta = X.attrs.get("column_meta", {})

    info = meta.get(col, {})
    base_name = info.get("name", col)
    units = info.get("units", "")
    category = info.get("category", "")

    level = X[col]
    d1 = uni.d1[col]
    d1_large = uni.d1_large_move[col]
    infl_flag = uni.inflection_flag[col]
    infl_dir = uni.inflection_direction[col]
    r2_gap = nl.r2_gap[col]
    nl_flag = nl.nonlinear_flag[col]

    mask = _year_mask(level.index, start_year, end_year)
    level = level[mask]
    d1 = d1[mask]
    d1_large = d1_large[mask]
    infl_flag = infl_flag[mask]
    infl_dir = infl_dir[mask]
    r2_gap = r2_gap[mask]
    nl_flag = nl_flag[mask]

    if level.dropna().empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # --- Panel 1: level --- #
    ax = axes[0]
    ax.plot(level.index, level.values, label="level")

    title_parts = [base_name]
    if category:
        title_parts.append(f"[{category}]")
    if units:
        title_parts.append(f"(units: {units})")
    ax.set_title("  ".join(title_parts))

    _shade_flag_spans(ax, level.index, nl_flag, color="red", alpha=0.10)

    up_idx = infl_flag & (infl_dir == "up")
    down_idx = infl_flag & (infl_dir == "down")

    if up_idx.any():
        ui = up_idx[up_idx].index
        ax.scatter(ui, level.loc[ui], marker="^", s=50, label="inflection up")
    if down_idx.any():
        di = down_idx[down_idx].index
        ax.scatter(di, level.loc[di], marker="v", s=50, label="inflection down")

    ax.set_ylabel(units if units else "Level")
    ax.grid(True)
    ax.legend(loc="best")

    # --- Panel 2: d1 --- #
    ax = axes[1]
    ax.plot(d1.index, d1.values, label="d1", alpha=0.85)

    lm_idx = d1_large[d1_large].index
    if len(lm_idx) > 0:
        ax.scatter(lm_idx, d1.loc[lm_idx], marker="o", s=25, label="large move")

    if up_idx.any():
        ui = [t for t in up_idx[up_idx].index if t in d1.index]
        if ui:
            ax.scatter(ui, d1.loc[ui], marker="^", s=40, label="inflection up")
    if down_idx.any():
        di = [t for t in down_idx[down_idx].index if t in d1.index]
        if di:
            ax.scatter(di, d1.loc[di], marker="v", s=40, label="inflection down")

    ax.set_title("First difference (d1), large moves & inflections")
    ax.set_ylabel("Δ (same units)")
    ax.grid(True)
    ax.legend(loc="best")

    # --- Panel 3: ΔR² --- #
    ax = axes[2]
    ax.plot(r2_gap.index, r2_gap.values, label="R²_quad - R²_linear")
    ax.axhline(0.15, linestyle="--", linewidth=1, label="nonlinear threshold")

    nl_idx = nl_flag[nl_flag].index
    if len(nl_idx) > 0:
        ax.scatter(
            nl_idx,
            r2_gap.loc[nl_idx],
            marker="x",
            s=30,
            label="nonlinear regime",
        )

    ax.set_title("Nonlinearity: improvement from quadratic vs linear trend")
    ax.set_ylabel("ΔR²")
    ax.grid(True)
    ax.legend(loc="best")

    plt.xlabel("Date (monthly, end-of-month)")
    plt.tight_layout()
    plt.show()


# ---------- all-series RAW diagnostics ----------

def plot_all_series_diagnostics(
    X: pd.DataFrame,
    uni: UnivariateSignals,
    nl: NonlinearitySignals,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    meta: Optional[dict] = None,
) -> None:
    """
    Loop raw-view diagnostics over all columns in X.
    """
    if meta is None:
        meta = X.attrs.get("column_meta", {})

    for col in X.columns:
        if X[col].dropna().empty:
            continue
        plot_series_diagnostics(
            X,
            col,
            uni=uni,
            nl=nl,
            meta=meta,
            start_year=start_year,
            end_year=end_year,
        )


# ---------- side-by-side compare: RAW or STANDARDIZED ----------

def plot_series_diagnostics_compare(
    X: pd.DataFrame,
    uni: UnivariateSignals,
    nl: NonlinearitySignals,
    left_col: str,
    right_col: str,
    std: Optional[StandardizedSignals] = None,
    use_standardized: bool = False,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    meta: Optional[dict] = None,
) -> None:
    """
    Compare two series (Left vs Right) in a 3x2 grid.

    If use_standardized = False (default):
        - left & right show RAW diagnostics (same as plot_series_diagnostics):
            Row 1: level + nonlinear shading + inflection dir
            Row 2: d1 + large moves + inflection dir
            Row 3: ΔR² (quad - lin) + nonlinear flags

    If use_standardized = True:
        Requires `std` (StandardizedSignals).
        For each side uses standardized signals:
            Row 1: level_z
            Row 2: d1_z
            Row 3: nonlinearity_score
        And flags from:
            - std.big_move_flag
            - std.inflection_up / std.inflection_down
            - score >= 1.0 as nonlinear regime
    """
    if meta is None:
        meta = X.attrs.get("column_meta", {})

    if use_standardized and std is None:
        raise ValueError("use_standardized=True but no StandardizedSignals `std` provided.")

    def _extract_raw(col: str):
        if not col or col not in X.columns:
            return None
        info = meta.get(col, {})
        return {
            "col": col,
            "name": info.get("name", col),
            "units": info.get("units", ""),
            "cat": info.get("category", ""),
            "level": X[col],
            "d1": uni.d1[col],
            "d1_large": uni.d1_large_move[col],
            "infl_flag": uni.inflection_flag[col],
            "infl_dir": uni.inflection_direction[col],
            "nl_metric": nl.r2_gap[col],
            "nl_flag": nl.nonlinear_flag[col],
            "mode": "raw",
        }

    def _extract_std(col: str):
        if (
            not col
            or col not in std.level_z.columns
        ):
            return None
        # we still read meta for names
        info = meta.get(col, {})
        # nonlinear regime from score >= 1 (since we scaled score so 1 ~ threshold)
        score = std.nonlinearity_score[col]
        nl_flag = score >= 1.0
        return {
            "col": col,
            "name": info.get("name", col) + " [std]",
            "units": "z-score",
            "cat": info.get("category", ""),
            "level": std.level_z[col],
            "d1": std.d1_z[col],
            "d1_large": std.big_move_flag[col].astype(bool),
            "infl_flag": (std.inflection_up[col].astype(bool)
                          | std.inflection_down[col].astype(bool)),
            "infl_dir": pd.Series(
                "",
                index=std.level_z.index,
            ).mask(std.inflection_up[col].astype(bool), "up").mask(
                std.inflection_down[col].astype(bool), "down"
            ),
            "nl_metric": score,
            "nl_flag": nl_flag,
            "mode": "std",
        }

    def _extract(col: str):
        return _extract_std(col) if use_standardized else _extract_raw(col)

    left = _extract(left_col)
    right = _extract(right_col)

    if left is None and right is None:
        print("Enter valid column names for Left and/or Right.")
        return

    def _apply_year_filter(obj):
        if obj is None:
            return None
        idx = obj["level"].index
        mask = _year_mask(idx, start_year, end_year)

        out = obj.copy()
        for key in ["level", "d1", "d1_large", "infl_flag", "infl_dir", "nl_metric", "nl_flag"]:
            out[key] = obj[key][mask]
        return out

    left = _apply_year_filter(left)
    right = _apply_year_filter(right)

    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex="row")
    fig.subplots_adjust(wspace=0.15, hspace=0.3)

    def _side(ax_level, ax_d1, ax_nl, data, label_side: str):
        if data is None or data["level"].dropna().empty:
            ax_level.set_title(f"{label_side}: (no / invalid series)")
            ax_level.axis("off")
            ax_d1.axis("off")
            ax_nl.axis("off")
            return

        lvl = data["level"]
        d1 = data["d1"]
        d1_large = data["d1_large"]
        infl_flag = data["infl_flag"]
        infl_dir = data["infl_dir"]
        nl_metric = data["nl_metric"]
        nl_flag = data["nl_flag"]

        name = data["name"]
        units = data["units"]
        cat = data["cat"]
        mode = data["mode"]  # "raw" or "std"

        # ---- Level panel ----
        ax_level.plot(lvl.index, lvl.values, label=f"{label_side}: {mode} level")
        title_parts = [f"{label_side}: {name}"]
        if cat:
            title_parts.append(f"[{cat}]")
        if units:
            title_parts.append(f"(units: {units})")
        ax_level.set_title("  ".join(title_parts), fontsize=10)

        _shade_flag_spans(ax_level, lvl.index, nl_flag, color="red", alpha=0.10)

        up_idx = infl_flag & (infl_dir == "up")
        down_idx = infl_flag & (infl_dir == "down")

        if up_idx.any():
            ui = up_idx[up_idx].index
            ax_level.scatter(ui, lvl.loc[ui], marker="^", s=40, label="inflection up")
        if down_idx.any():
            di = down_idx[down_idx].index
            ax_level.scatter(di, lvl.loc[di], marker="v", s=40, label="inflection down")

        ax_level.set_ylabel(units if units else "Level", fontsize=8)
        ax_level.grid(True)
        ax_level.legend(loc="best", fontsize=7)

        # ---- d1 panel ----
        ax_d1.plot(d1.index, d1.values, label="d1" if mode == "raw" else "d1_z", alpha=0.85)

        lm_idx = d1_large[d1_large].index if isinstance(d1_large, pd.Series) else d1_large.index[d1_large]
        if len(lm_idx) > 0:
            ax_d1.scatter(
                lm_idx,
                d1.loc[lm_idx],
                marker="o",
                s=20,
                label="large move",
            )

        if up_idx.any():
            ui = [t for t in up_idx[up_idx].index if t in d1.index]
            if ui:
                ax_d1.scatter(ui, d1.loc[ui], marker="^", s=35, label="inflection up")
        if down_idx.any():
            di = [t for t in down_idx[down_idx].index if t in d1.index]
            if di:
                ax_d1.scatter(di, d1.loc[di], marker="v", s=35, label="inflection down")

        ax_d1.set_title(
            "First diff (d1) / d1_z, large moves & inflections",
            fontsize=9,
        )
        ax_d1.set_ylabel("Δ" if mode == "raw" else "z", fontsize=8)
        ax_d1.grid(True)
        ax_d1.legend(loc="best", fontsize=7)

        # ---- nonlinearity panel ----
        ax_nl.plot(
            nl_metric.index,
            nl_metric.values,
            label="ΔR² (quad-lin)" if mode == "raw" else "nonlinearity_score",
        )

        if mode == "raw":
            ax_nl.axhline(0.15, linestyle="--", linewidth=1, label="threshold")
        else:
            ax_nl.axhline(1.0, linestyle="--", linewidth=1, label="score=1 (~threshold)")

        nl_idx = nl_flag[nl_flag].index
        if len(nl_idx) > 0:
            ax_nl.scatter(
                nl_idx,
                nl_metric.loc[nl_idx],
                marker="x",
                s=25,
                label="nonlinear regime",
            )

        ax_nl.set_title(
            "Nonlinearity ΔR²" if mode == "raw" else "Nonlinearity score",
            fontsize=9,
        )
        ax_nl.set_ylabel("ΔR²" if mode == "raw" else "score", fontsize=8)
        ax_nl.grid(True)
        ax_nl.legend(loc="best", fontsize=7)

    _side(axes[0, 0], axes[1, 0], axes[2, 0], left, "Left")
    _side(axes[0, 1], axes[1, 1], axes[2, 1], right, "Right")

    for ax in axes[2, :]:
        ax.set_xlabel("Date (monthly, end-of-month)", fontsize=9)

    plt.show()
