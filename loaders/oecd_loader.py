# loaders/oecd_loader.py
import requests
import pandas as pd
from io import StringIO

OECD_BASE = "https://sdmx.oecd.org/public/rest"


class OECDAPIError(Exception):
    pass


def fetch_oecd_csv(
    dataflow: str,
    selection: str = "all",
    start_period: str | None = None,
    end_period: str | None = None,
    labels: bool = True,
    dimension_at_observation: str = "AllDimensions",
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Generic OECD SDMX CSV fetch.

    This is intentionally generic; for production you’ll normally:
      - Inspect the dataflow via OECD SDMX docs/portal
      - Pass a specific 'selection' key.

    We keep this here so OECD-backed metrics can plug in cleanly.
    """
    url = f"{OECD_BASE}/data/{dataflow}/{selection}"

    params = {
        "format": "csvfilewithlabels" if labels else "csvfile",
        "dimensionAtObservation": dimension_at_observation,
    }
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period

    resp = requests.get(url, params=params, timeout=timeout)
    if not resp.ok:
        raise OECDAPIError(
            f"OECD API error {resp.status_code} for {resp.url}: {resp.text[:300]}"
        )

    return pd.read_csv(StringIO(resp.text))
