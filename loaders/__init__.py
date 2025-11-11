# Makes 'loaders' a proper package.
# Re-export commonly used functions if you want.

from .fred_loader import fetch_fred_series
from .worldbank_loader import fetch_wb_indicator
from .imf_loader import fetch_imf_series
from .oecd_loader import fetch_oecd_csv
from .bis_loader import fetch_bis_csv, fetch_bis_csv_from_url, bis_csv_to_timeseries
