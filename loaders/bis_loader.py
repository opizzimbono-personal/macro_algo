import requests
import pandas as pd
from io import StringIO
from typing import Optional

BIS_BASE = "https://stats.bis.org"


class BISAPIError(Exception):
    pass


def fetch_bis_csv_from_url(url: str, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch BIS data when you already have the full Developer API URL.

    If no 'format=' is present, append '?format=csv' or '&format=csv'.
    """
    if "format=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}format=csv"

    resp = requests.get(url, timeout=timeout)
    if not resp.ok:
        raise BISAPIError(
            f"BIS API error {resp.status_code} for {url}: {resp.text[:300]}"
        )

    return pd.read_csv(StringIO(resp.text))


def fetch_bis_csv(
    path: str,
    params: Optional[dict] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch BIS data when you have the relative API path (no scheme/host).
    """
    if not path:
        raise ValueError("path must be a non-empty string")

    # If they accidentally pass full URL, delegate
    if path.startswith("http://") or path.startswith("https://"):
        return fetch_bis_csv_from_url(path, timeout=timeout)

    if params is None:
        params = {}
    params.setdefault("format", "csv")

    url = f"{BIS_BASE.rstrip('/')}/{path.lstrip('/')}"

    resp = requests.get(url, params=params, timeout=timeout)
    if not resp.ok:
        raise BISAPIError(
            f"BIS API error {resp.status_code} for {resp.url}: {resp.text[:300]}"
        )

    return pd.read_csv(StringIO(resp.text))


def bis_csv_to_timeseries(
    df: pd.DataFrame,
    value_col: str = "OBS_VALUE",
    time_col: str = "TIME_PERIOD",
    series_id_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Convert BIS-style flat CSV into a time-indexed DataFrame.
    """
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{time_col}' and '{value_col}' columns."
        )

    # Infer frequency
    sample = str(df[time_col].dropna().iloc[0])
    if "-Q" in sample:
        freq = "Q"
    elif sample.count("-") == 1:
        freq = "M"
    else:
        freq = "A"

    # Build series identifier
    if series_id_cols:
        for col in series_id_cols:
            if col not in df.columns:
                raise ValueError(f"Series id column '{col}' not in DataFrame.")
        df["_series_id"] = df[series_id_cols].astype(str).agg("|".join, axis=1)
    else:
        dim_cols = [c for c in df.columns if c not in (time_col, value_col)]
        if not dim_cols:
            df["_series_id"] = "series"
        else:
            df["_series_id"] = df[dim_cols].astype(str).agg("|".join, axis=1)

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    idx = pd.PeriodIndex(df[time_col].astype(str), freq=freq)

    wide = (
        df.assign(TIME_PERIOD=idx)
        .pivot_table(index="TIME_PERIOD", columns="_series_id", values=value_col)
        .sort_index()
    )

    try:
        wide.index = wide.index.to_timestamp()
    except Exception:
        pass

    if wide.shape[1] == 1:
        colname = wide.columns[0]
        wide.columns = [colname or "value"]

    return wide
