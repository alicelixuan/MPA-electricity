"""Step 4: analysis panel, one row per unit-hour (40 units, 2019-2024): deviation, net exposure, net profit, redispatch.
Cleaned from ML-Strom notebooks 5 (cell 8), 7 (cells 29-31), 8 (cell 3) and R&R_addition_redispatch_volume.
Rebuilt from the stored upstream results it equals data/stored/analysis_panel.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TECHS = ["Fossil Brown coal/Lignite", "Fossil Hard coal", "Fossil Gas Combined cycle", "Fossil Gas gas/steam turbine"]
ON, OFF = 0.95, 0.05                 # discretization of the Monte Carlo dispatch state (Eq. 4)
CONTROL_AREA = {"DE_50HzT": "50Hertz Transmission", "DE_Amprion": "Amprion",
                "DE_TenneT_GER": "TenneT TSO", "DE_TransnetBW": "TransnetBW"}
COLUMNS = ["production_type", "installed_capacity", "company", "competitive_dispatch_status_discretized",
           "deviation_discretized", "net_exposure", "margin", "price", "available_capacity", "actual_generation_output",
           "slope", "net_profit_withhold_marginal", "net_profit_pushin_marginal", "control_area",
           "redispatch_decrease_MWh", "redispatch_increase_MWh", "company_gen", "hedged_quantity"]


def net_exposure(units: pd.DataFrame, chp: pd.DataFrame) -> pd.DataFrame:
    """Eq. 12 (notebook 7, cells 29-31), over all units: company generation = all its units + its CHP units.
    Hedged quantity = monthly mean company generation, peak (Mon-Fri 8-20 h UTC) and off-peak separately."""
    df = units.merge(chp, on=["DateTime", "company"], how="left")
    df["CHP_generation"] = df.CHP_generation.fillna(0)
    df["company_gen"] = df.groupby(["company", "DateTime"]).ActualGenerationOutput.transform("sum") + df.CHP_generation
    t = pd.DatetimeIndex(df.DateTime)
    peak = (t.dayofweek < 5) & (t.hour >= 8) & (t.hour < 20)
    df["hedged_quantity"] = df.groupby([t.year, t.month, df.company, peak]).company_gen.transform("mean")
    df["net_exposure"] = df.company_gen - df.hedged_quantity
    return df


def deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Eqs. 4-5 (notebook 5, cell 8): discretized benchmark z and deviation y in {-1, 0, 1}."""
    df["competitive_dispatch_status_discretized"] = np.select([df.state >= ON, df.state <= OFF], [1.0, 0.0], np.nan)
    df["deviation_discretized"] = (df.ActualGenerationOutput > 0).astype(int) - df.competitive_dispatch_status_discretized
    df.loc[(df.deviation_discretized == -1) & (df.AvailableCapacity == 0), "deviation_discretized"] = 0  # outage
    return df


def hourly_redispatch(raw: pd.DataFrame) -> pd.DataFrame:
    """ENTSO-E 13.1.A redispatch intervals -> hourly MWh per control area and direction (R&R notebook, cells 2-11)."""
    raw = raw[raw.MapCode.str.contains("DE")].rename(columns={
        "StartRedispatching(UTC)": "start", "EndRedispatching(UTC)": "end", "CapacityImpact[MWh/MTU]": "mwh"})
    start, end = pd.to_datetime(raw.start, utc=True), pd.to_datetime(raw.end, utc=True)
    step = raw.ResolutionCode.map({"PT60M": 60, "PT30M": 30, "PT15M": 15}).to_numpy()
    minutes = (end - start).dt.total_seconds().to_numpy() / 60
    n = np.where(minutes >= 1, np.floor((minutes - 1) / step) + 1, 0).astype("int64")   # MTUs in [start, end - 1 min]
    i = np.repeat(np.arange(len(raw)), n)
    k = np.arange(len(i)) - np.repeat(np.cumsum(n) - n, n)
    mtu = start.dt.tz_convert(None).to_numpy()[i] + (k * step[i]).astype("timedelta64[m]")
    long = pd.DataFrame({"DateTime": pd.DatetimeIndex(mtu).tz_localize("UTC").floor("h"), "control_area": raw.MapCode.to_numpy()[i],
                         "direction": raw.ActionDirection.to_numpy()[i], "MWh": raw.mwh.astype(float).to_numpy()[i]})
    out = long.groupby(["DateTime", "control_area", "direction"]).MWh.sum().unstack(fill_value=0.0)
    out = out.rename(columns={"Increase": "redispatch_increase_MWh", "Decrease": "redispatch_decrease_MWh"}).reset_index()
    out["control_area"] = out.control_area.map(CONTROL_AREA)
    return out


def build_panel(dispatch: pd.DataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    """dispatch: EIC, DateTime, state, marginal_cost (step 3). slopes: DateTime, slope (step 2, min_space = 0)."""
    units = pd.read_parquet(config.INPUTS / "units_observed.parquet")
    chp = pd.read_parquet(config.INPUTS / "chp_generation.parquet")
    price = config.read_market().price.rename("price")
    area = pd.read_csv(config.INPUTS / "unit_control_area.csv")
    redispatch = hourly_redispatch(pd.read_parquet(config.INPUTS / "redispatch_entsoe.parquet"))

    df = net_exposure(units, chp)
    df = df[df.production_type.isin(TECHS)]                       # -> 40 units, 1,785,678 unit-hours
    df = df.merge(dispatch[["EIC", "DateTime", "state", "marginal_cost"]], on=["EIC", "DateTime"], how="left")
    df = df.merge(price, left_on="DateTime", right_index=True, how="left")
    df = df.merge(slopes[["DateTime", "slope"]], on="DateTime", how="left")
    df["margin"] = df.price - df.marginal_cost
    df = deviation(df)
    df["net_profit_withhold_marginal"] = df.slope * df.net_exposure - df.margin       # Eq. 9, per MW
    df["net_profit_pushin_marginal"] = -df.slope * df.net_exposure + df.margin       # Eq. 10, per MW
    df = df.merge(area, on="EIC", how="left").merge(redispatch, on=["DateTime", "control_area"], how="left")
    df[["redispatch_increase_MWh", "redispatch_decrease_MWh"]] = df[["redispatch_increase_MWh", "redispatch_decrease_MWh"]].fillna(0.0)
    df = df.rename(columns={"InstalledCapacity": "installed_capacity", "AvailableCapacity": "available_capacity",
                            "ActualGenerationOutput": "actual_generation_output"})
    return df.set_index(["EIC", "DateTime"]).sort_index()[COLUMNS]
