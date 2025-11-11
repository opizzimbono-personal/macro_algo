from __future__ import annotations

from typing import Tuple, List, Dict, Union

import pandas as pd

from .metric_registry import get_metric_specs, MetricSpec


# ---------- frequency mapping to monthly ----------

def _to_monthly(s: pd.Series, method: str) -> pd.Series:
    """
    Convert arbitrary freq series to monthly end-of-month (M) using a
    series-specific rule defined in MetricSpec.monthly_method.

    method:
      'daily_eom' : last daily observation in each month.
      'monthly'   : treat as monthly, use last obs in each month.
      'q_ffill'   : quarterly value held constant across its 3 months.
      'a_ffill'   : annual value held constant across 12 months.
      'a_12'      : annual flow split into 12 equal monthly values.

    Output index: MonthEnd dates (freq='M').
    """
    s = s.sort_index()
    if s.empty:
        return s

    m = (method or "").lower()

    if m == "daily_eom":
        return s.resample("M").last()

    if m == "monthly":
        return s.resample("M").last()

    if m == "q_ffill":
        q = s.resample("Q").last()
        return q.resample("M").ffill()

    if m == "a_ffill":
        y = s.resample("A").last()
        return y.resample("M").ffill()

    if m == "a_12":
        y = s.resample("A").last()
        monthly = (y / 12.0).resample("M").ffill()
        return monthly

    # conservative fallback: treat as monthly
    return s.resample("M").last()


# ---------- units → column suffix ----------

def _sanitize_units_for_col(units: str) -> str:
    if not units:
        return ""

    u = units.lower()
    u = u.replace("%", "pct")
    u = u.replace("/", "_per_")

    for ch in [",", "(", ")", "[", "]", "{", "}", ":", ";"]:
        u = u.replace(ch, " ")

    for ch in [" ", "-"]:
        u = u.replace(ch, "_")

    while "__" in u:
        u = u.replace("__", "_")

    u = u.strip("_")

    if len(u) > 40:
        u = u[:40].rstrip("_")

    return u


def _apply_unit_suffixes(
    X: pd.DataFrame,
    used_specs: List[MetricSpec],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """
    Rename columns of X to include units based on MetricSpecs and
    return (X_renamed, column_meta).

    New name: <spec.name>__<unit_suffix>
    """
    col_meta: Dict[str, Dict[str, str]] = {}
    rename_map: Dict[str, str] = {}

    spec_by_name = {spec.name: spec for spec in used_specs}

    for col in X.columns:
        spec = spec_by_name.get(col)
        if not spec:
            # Keep unknown columns but record minimal meta
            new_name = col
            col_meta[new_name] = {
                "name": col,
                "units": "",
                "category": "",
                "source": "",
                "description": "",
            }
            continue

        unit_suffix = _sanitize_units_for_col(spec.units)
        new_name = f"{spec.name}__{unit_suffix}" if unit_suffix else spec.name

        base = new_name
        k = 1
        while new_name in col_meta:
            k += 1
            new_name = f"{base}_{k}"

        rename_map[col] = new_name
        col_meta[new_name] = {
            "name": spec.name,
            "units": spec.units,
            "category": spec.category,
            "source": spec.source,
            "description": spec.description,
        }

    X_renamed = X.rename(columns=rename_map)
    return X_renamed, col_meta


# ---------- extend index to target end month ----------

def _extend_to_end(
    X: pd.DataFrame,
    end_date: Union[str, pd.Timestamp, None],
    extend_to_current: bool,
) -> pd.DataFrame:
    """
    Ensure X's monthly index runs through desired end month.

    - If extend_to_current=True and end_date=None:
          extend to last day of current month.
    - If end_date given:
          extend to that month-end.
    - Only extends forward; never truncates.
    """
    if X.empty:
        return X

    target_end = None

    if end_date is not None:
        if isinstance(end_date, str):
            ts = pd.to_datetime(end_date)
            target_end = ts.to_period("M").to_timestamp("M")
        elif isinstance(end_date, pd.Timestamp):
            target_end = end_date.to_period("M").to_timestamp("M")
        else:
            raise ValueError("end_date must be str, pd.Timestamp, or None")
    elif extend_to_current:
        today = pd.Timestamp.today()
        target_end = today.to_period("M").to_timestamp("M")

    if target_end is None:
        return X

    current_end = X.index.max()
    if target_end <= current_end:
        return X

    start = X.index.min()
    full_idx = pd.date_range(start=start, end=target_end, freq="M")
    return X.reindex(full_idx)


# ---------- derived series ----------

def _add_derived_series(
    X: pd.DataFrame,
    used_specs: List[MetricSpec],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[MetricSpec]]:
    """
    Add derived metrics (like CPI YoY) into X + used_specs BEFORE units renaming.
    """
    specs = list(used_specs)  # copy

    # CPI YoY from us_cpi_all_items
    if "us_cpi_all_items" in X.columns:
        base = X["us_cpi_all_items"]
        infl_yoy = base.pct_change(12) * 100.0

        if infl_yoy.notna().any():
            name = "us_cpi_all_items_yoy"
            X[name] = infl_yoy

            specs.append(
                MetricSpec(
                    name=name,
                    category="inflation",
                    source="Derived from us_cpi_all_items",
                    frequency="M",
                    units="percent",
                    monthly_method="monthly",
                    description="Year-on-year CPI inflation rate (headline).",
                    loader=lambda: X[name],  # not used in rebuild; placeholder
                )
            )
            if verbose:
                print("[INFO] Added derived series: us_cpi_all_items_yoy (YoY %).")

    return X, specs


# ---------- main builder ----------

def build_design_matrix(
    country: str = "US",
    dropna: bool = False,
    extend_to_current: bool = True,
    end_date: Union[str, pd.Timestamp, None] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[MetricSpec]]:
    """
    Build a MONTHLY (EOM) design matrix for `country` with:

      - All base metrics from MetricSpecs.
      - Derived series (e.g. CPI YoY).
      - Column names including units.
      - Metadata in X.attrs["column_meta"].

    Parameters
    ----------
    dropna : bool
        If True, keep only rows where all metrics are non-missing.
        If False, keep full panel; late periods may have NaNs for some series.
    extend_to_current : bool
        If True and end_date is None, extend index to current month-end.
    end_date : str or Timestamp
        Optional explicit end month (e.g. "2025-10").
    """
    specs = get_metric_specs(country)
    if not specs:
        raise RuntimeError(f"No metric specs defined for country '{country}'.")

    series_list: List[pd.Series] = []
    used_specs: List[MetricSpec] = []

    # Load & monthly-map each spec
    for spec in specs:
        try:
            raw = spec.loader()
            if raw is None:
                if verbose:
                    print(f"[WARN] {spec.name}: loader returned None, skipping.")
                continue

            if isinstance(raw, pd.DataFrame):
                if raw.empty:
                    if verbose:
                        print(f"[WARN] {spec.name}: empty DataFrame, skipping.")
                    continue
                if raw.shape[1] == 1:
                    raw = raw.iloc[:, 0]
                else:
                    if verbose:
                        print(
                            f"[WARN] {spec.name}: multi-col DataFrame from loader, skipping."
                        )
                    continue

            if not isinstance(raw, pd.Series):
                if verbose:
                    print(f"[WARN] {spec.name}: loader did not return Series, skipping.")
                continue

            raw = raw.sort_index().dropna()
            if raw.empty:
                if verbose:
                    print(f"[WARN] {spec.name}: empty after dropna, skipping.")
                continue

            m = _to_monthly(raw, spec.monthly_method).dropna()
            if m.empty:
                if verbose:
                    print(f"[WARN] {spec.name}: empty after monthly transform, skipping.")
                continue

            m.name = spec.name
            series_list.append(m)
            used_specs.append(spec)

        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to load {spec.name}: {e}")
            continue

    if not series_list:
        raise RuntimeError(
            "No metrics loaded successfully; check API keys/connectivity/loaders."
        )

    # Outer-join across all metrics (pre-extension)
    X = pd.concat(series_list, axis=1, join="outer").sort_index()

    # Extend index forward if requested
    X = _extend_to_end(X, end_date=end_date, extend_to_current=extend_to_current)

    # Add derived series like CPI YoY before renaming
    X, used_specs = _add_derived_series(X, used_specs, verbose=verbose)

    # Drop rows only AFTER all series (incl. derived) exist
    if dropna:
        X = X.dropna(how="any")

    # Apply units in column names + attach metadata
    X, col_meta = _apply_unit_suffixes(X, used_specs)
    X.attrs["column_meta"] = col_meta

    if verbose:
        print(
            f"[INFO] Design matrix built (monthly EOM): shape={X.shape} "
            f"with {len(used_specs)} metrics (including derived)."
        )

    return X, used_specs
