# config.py
import os

# ---------- KEYS & GLOBALS ----------

# Required for FRED-based metrics
FRED_API_KEY = os.getenv("FRED_API_KEY", "ca5b932dcc8e725f14839fb4642a2c81")

if not FRED_API_KEY:
    raise RuntimeError(
        "FRED_API_KEY not set. Please export FRED_API_KEY in your environment."
    )

# Default sample window (you can change globally here)
DEFAULT_START = "1990-01-01"
DEFAULT_END = None  # use latest
