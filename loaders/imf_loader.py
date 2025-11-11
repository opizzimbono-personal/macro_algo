# loaders/imf_loader.py
import pandas as pd
import sdmx

# Public client; no key needed for standard public datasets.
IMF_DATA = sdmx.Client("IMF_DATA")


def fetch_imf_series(
    dataset: str,
    key: str,
    start_period: str | None = None,
    end_period: str | None = None,
) -> pd.DataFrame:
    """
    Generic IMF SDMX fetch using sdmx1.

    Parameters
    ----------
    dataset : str
        IMF dataset ID, e.g. 'CPI', 'IFS', etc.
    key : str
        SDMX key string, e.g. 'USA.CPI.TOTL.ZF.A'
        (must be valid for the dataset DSD).
    start_period, end_period : SDMX time filters.

    Returns
    -------
    DataFrame
        Index: datetime-like;
        Columns: one per series key (e.g. country / concept).
    """
    params: dict = {}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period

    msg = IMF_DATA.data(dataset, key=key, params=params)
    df = sdmx.to_pandas(msg)

    # df is often a Series with a MultiIndex
    if isinstance(df, pd.Series):
        idx_names = list(df.index.names)
        time_level = [
            i for i, n in enumerate(idx_names) if "TIME" in (n or "").upper()
        ]
        if len(time_level) != 1:
            return df.to_frame("value")

        tlev = time_level[0]
        df = df.unstack(list(range(0, tlev)))
        if hasattr(df.index, "to_timestamp"):
            df.index = df.index.to_timestamp()
        df = df.sort_index()

    else:
        if hasattr(df.index, "to_timestamp"):
            try:
                df.index = df.index.to_timestamp()
            except Exception:
                pass
        df = df.sort_index()

    return df

