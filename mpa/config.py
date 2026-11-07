"""Paths and readers shared by all steps. Stored files are read-only; reruns write to output/rerun/<run>/."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "inputs"
STORED = ROOT / "data" / "stored"
OUTPUT = ROOT / "output"
PAPER = OUTPUT / "paper"


def read_market() -> pd.DataFrame:
    """Hourly residual load (quantity, MW), day-ahead price (EUR/MWh) and the day index of the regimes; UTC."""
    df = pd.read_csv(INPUTS / "market_hourly.csv", index_col="datetime_utc", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["day", "real_day", "quantity", "price"]]


def read_fuel() -> pd.DataFrame:
    """Daily carbon-adjusted gas and coal prices (EUR/MWh thermal)."""
    return pd.read_csv(INPUTS / "fuel_prices_daily.csv", index_col="date", parse_dates=True)[["price_gas", "price_coal"]]


def read_load() -> pd.Series:
    s = pd.read_csv(INPUTS / "load_hourly.csv", index_col="DateTime", parse_dates=True)["load"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


# ---- stored results (what the paper used) ---------------------------------------------------------------------
def stored_change_points() -> list[int]:
    return json.loads((STORED / "change_points.json").read_text())["change_points"]


def stored_fits() -> pd.DataFrame:
    return pd.read_csv(STORED / "supply_fits.csv")


def stored_slopes() -> pd.DataFrame:
    return pd.read_parquet(STORED / "slopes.parquet")


def stored_dispatch() -> pd.DataFrame:
    return pd.read_parquet(STORED / "dispatch_results.parquet")


def stored_panel() -> pd.DataFrame:
    return pd.read_parquet(STORED / "analysis_panel.parquet")
