from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import pandas as pd

from loaders.fred_loader import fetch_fred_series
from loaders.worldbank_loader import fetch_wb_indicator


@dataclass
class MetricSpec:
    """
    Definition of one time series used in the design matrix.

    name            Column name.
    category        Logical group (immigration, wages, jobs, etc.).
    source          Human-readable source / series id.
    frequency       Native data frequency: 'D', 'M', 'Q', 'A'.
    units           Units BEFORE any monthly transform.
    monthly_method  How to map to monthly:
                        'daily_eom'  - last daily obs each month
                        'monthly'    - treat as monthly, take last-in-month
                        'q_ffill'    - quarterly value held constant across its 3 months
                        'a_ffill'    - annual value held constant across 12 months
                        'a_12'       - annual flow split into 12 equal monthly values
    description     Short explanation.
    loader          Zero-arg callable -> pandas.Series or 1-col DataFrame.
    """
    name: str
    category: str
    source: str
    frequency: str
    units: str
    monthly_method: str
    description: str
    loader: Callable[[], pd.Series]


def _ensure_series(x: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        if x.empty:
            return pd.Series(dtype=float, name=name)
        if x.shape[1] != 1:
            raise ValueError(f"Metric {name} loader returned multi-col DataFrame.")
        s = x.iloc[:, 0].copy()
    s.name = name
    return s.sort_index()


# ---------- US metrics implementing your outline (API-only) ----------

def _us_metrics() -> List[MetricSpec]:
    m: List[MetricSpec] = []

    # 1. Immigration: WB Net migration (UN DESA based), annual
    # We'll convert to monthly flow = annual / 12 via 'a_12'.
    def _us_net_migration():
        df = fetch_wb_indicator("SM.POP.NETM", "US", 1960, 2050)
        if df.empty or "US" not in df.columns:
            return pd.Series(dtype=float, name="us_net_migration_wb")
        return _ensure_series(df["US"], "us_net_migration_wb")

    m.append(
        MetricSpec(
            name="us_net_migration_wb",
            category="immigration",
            source="WorldBank: SM.POP.NETM",
            frequency="A",
            units="persons per year (net flow)",
            monthly_method="a_12",
            description="US net migration (annual); mapped to monthly as annual/12.",
            loader=_us_net_migration,
        )
    )

    # 2. Wages: Avg Hourly Earnings, Total Private (M)
    m.append(
        MetricSpec(
            name="us_avg_hourly_earnings",
            category="wages",
            source="FRED: CEU0500000003",
            frequency="M",
            units="USD per hour",
            monthly_method="monthly",
            description="Average Hourly Earnings of All Employees, Total Private.",
            loader=lambda: _ensure_series(
                fetch_fred_series("CEU0500000003"),
                "us_avg_hourly_earnings",
            ),
        )
    )

    # 3. Jobs: Unemployment rate (M)
    m.append(
        MetricSpec(
            name="us_unemployment_rate",
            category="jobs",
            source="FRED: UNRATE",
            frequency="M",
            units="percent",
            monthly_method="monthly",
            description="Unemployment Rate (U-3), monthly, percent.",
            loader=lambda: _ensure_series(
                fetch_fred_series("UNRATE"),
                "us_unemployment_rate",
            ),
        )
    )

    # 3b. Jobs: Nonfarm Payrolls (M)
    m.append(
        MetricSpec(
            name="us_nonfarm_payrolls",
            category="jobs",
            source="FRED: PAYEMS",
            frequency="M",
            units="thousands of persons",
            monthly_method="monthly",
            description="All Employees: Total Nonfarm Payrolls.",
            loader=lambda: _ensure_series(
                fetch_fred_series("PAYEMS"),
                "us_nonfarm_payrolls",
            ),
        )
    )

    # 4. Consumption: Real PCE (M)
    m.append(
        MetricSpec(
            name="us_real_pce",
            category="consumption",
            source="FRED: PCEC96",
            frequency="M",
            units="billions of chained 2017 USD, SAAR",
            monthly_method="monthly",
            description="Real Personal Consumption Expenditures.",
            loader=lambda: _ensure_series(
                fetch_fred_series("PCEC96"),
                "us_real_pce",
            ),
        )
    )

    # 5. Inflation: CPI All Items (M)
    m.append(
        MetricSpec(
            name="us_cpi_all_items",
            category="inflation",
            source="FRED: CPIAUCSL",
            frequency="M",
            units="index, 1982-84=100",
            monthly_method="monthly",
            description="CPI All Items, SA.",
            loader=lambda: _ensure_series(
                fetch_fred_series("CPIAUCSL"),
                "us_cpi_all_items",
            ),
        )
    )

    # 6. Inflation Expectations (survey): UMich (M)
    m.append(
        MetricSpec(
            name="us_mich_exp_inflation",
            category="inflation_expectations_survey",
            source="FRED: MICH",
            frequency="M",
            units="index",
            monthly_method="monthly",
            description="UMich inflation expectations index proxy.",
            loader=lambda: _ensure_series(
                fetch_fred_series("MICH"),
                "us_mich_exp_inflation",
            ),
        )
    )

    # 6b. Inflation Expectations (market): 5y5y (D)
    # We'll take month-end value via 'daily_eom'.
    m.append(
        MetricSpec(
            name="us_5y5y_fwd_infl",
            category="inflation_expectations_market",
            source="FRED: T5YIFR",
            frequency="D",
            units="percent",
            monthly_method="daily_eom",
            description="5-Year, 5-Year Forward Inflation Expectation Rate.",
            loader=lambda: _ensure_series(
                fetch_fred_series("T5YIFR"),
                "us_5y5y_fwd_infl",
            ),
        )
    )

    # 7. Cash & Assets: Households; Checkable Deposits & Currency (Q)
    # We'll hold constant within quarter via 'q_ffill'.
    def _us_household_liquid_assets():
        s = fetch_fred_series("BOGZ1FL193020005Q")
        return _ensure_series(s, "us_household_liquid_assets")

    m.append(
        MetricSpec(
            name="us_household_liquid_assets",
            category="cash_assets",
            source="FRED: BOGZ1FL193020005Q",
            frequency="Q",
            units="millions of USD",
            monthly_method="q_ffill",
            description="Households; checkable deposits & currency; asset, level (Z.1).",
            loader=_us_household_liquid_assets,
        )
    )

    # 8. IG Spreads: ICE BofA US Corp OAS (D)
    # Month-end level is standard.
    m.append(
        MetricSpec(
            name="us_ig_oas",
            category="ig_spreads",
            source="FRED: BAMLC0A0CM",
            frequency="D",
            units="percent (OAS)",
            monthly_method="daily_eom",
            description="ICE BofA US Corporate Index Option-Adjusted Spread (IG).",
            loader=lambda: _ensure_series(
                fetch_fred_series("BAMLC0A0CM"),
                "us_ig_oas",
            ),
        )
    )

    # 9. Productivity: OPHNFB (Q)
    # Hold constant within quarter.
    m.append(
        MetricSpec(
            name="us_nonfarm_productivity",
            category="productivity",
            source="FRED: OPHNFB",
            frequency="Q",
            units="index",
            monthly_method="q_ffill",
            description="Nonfarm Business Sector: Output per Hour.",
            loader=lambda: _ensure_series(
                fetch_fred_series("OPHNFB"),
                "us_nonfarm_productivity",
            ),
        )
    )

    # 10. Real-time Activity: Enplanements (M)
    m.append(
        MetricSpec(
            name="us_enplanements",
            category="activity_high_freq",
            source="FRED: ENPLANE",
            frequency="M",
            units="thousands of passengers",
            monthly_method="monthly",
            description="US air carrier enplanements.",
            loader=lambda: _ensure_series(
                fetch_fred_series("ENPLANE"),
                "us_enplanements",
            ),
        )
    )

    return m


def get_metric_specs(country: str = "US") -> List[MetricSpec]:
    if country.upper() == "US":
        return _us_metrics()
    return []
