import pandas as pd

from .http_utils import get_json
from config import FRED_API_KEY, DEFAULT_START, DEFAULT_END

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(
    series_id: str,
    start_date: str = DEFAULT_START,
    end_date: str | None = DEFAULT_END,
) -> pd.Series:
    """
    Fetch a single FRED series as a clean pandas.Series with datetime index.
    Requires FRED_API_KEY in config.py or env.
    """
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
    }
    if end_date:
        params["observation_end"] = end_date

    data = get_json(BASE_URL, params=params)
    obs = data.get("observations", [])
    if not obs:
        return pd.Series(dtype=float, name=series_id)

    df = (
        pd.DataFrame(obs)[["date", "value"]]
        .assign(
            date=lambda d: pd.to_datetime(d["date"]),
            value=lambda d: pd.to_numeric(d["value"], errors="coerce"),
        )
        .dropna()
        .set_index("date")
        .sort_index()
    )
    s = df["value"]
    s.name = series_id
    return s
