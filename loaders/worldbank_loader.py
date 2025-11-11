import pandas as pd

from .http_utils import get_json

BASE_URL = "https://api.worldbank.org/v2"


def fetch_wb_indicator(
    indicator: str,
    countries,
    start_year: int = 1990,
    end_year: int = 2050,
) -> pd.DataFrame:
    """
    Fetches a World Bank indicator for one or many countries.
    Returns DataFrame with datetime index and one column per country code.
    """
    if isinstance(countries, str):
        countries = [countries]

    frames = []
    for c in countries:
        url = f"{BASE_URL}/country/{c}/indicator/{indicator}"
        params = {
            "format": "json",
            "per_page": 20000,
            "date": f"{start_year}:{end_year}",
        }

        page = 1
        rows = []
        while True:
            params["page"] = page
            data = get_json(url, params=params)
            if not data or len(data) < 2:
                break

            meta, values = data
            for v in values:
                if v.get("value") is not None:
                    rows.append(
                        {
                            "date": v["date"],
                            "value": float(v["value"]),
                        }
                    )

            if page >= meta.get("pages", 1):
                break
            page += 1

        if not rows:
            continue

        df = (
            pd.DataFrame(rows)
            .assign(date=lambda d: pd.to_datetime(d["date"]))
            .set_index("date")
            .sort_index()
            .rename(columns={"value": c})
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    return out
