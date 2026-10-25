"""Loaders for every dataset used in v2 (all paths are relative to the repository)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "v2"
DATA = V2 / "data"
FIGURES = V2 / "figures"
MAPS = V2 / "maps"

# AirNow parameter names -> short codes used everywhere in v2
AIRNOW_PARAMS = {"PM2.5-24hr": "pm25", "OZONE-8HR": "o3_8h", "OZONE-1HR": "o3_1h",
                 "PM10-24hr": "pm10", "CO-8hr": "co", "SO2-24HR": "so2"}


def load_nyc_aqicn(path: Path | str = ROOT / "NYpollutants.csv") -> pd.DataFrame:
    """Daily New York City export of aqicn.org (values are US EPA AQI sub-indices)."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y/%m/%d")
    for c in df.columns.drop("date"):
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    df = df.sort_values("date").set_index("date").asfreq("D")
    return df


def load_uci(path: Path | str = ROOT / "AirQualityUCI.csv") -> pd.DataFrame:
    """UCI Air Quality (De Vito et al., 2008): hourly, Italian city, March 2004 - April 2005.

    Fixes the three parsing issues of v1: ``;`` separator with ``,`` decimals,
    -200 as the missing-value code (also in text columns) and the two empty
    trailing columns.
    """
    df = pd.read_csv(path, sep=";", decimal=",", na_values=[-200, "-200", "-200,0"])
    df = df.dropna(how="all").loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
    df["datetime"] = pd.to_datetime(df.Date + " " + df.Time.str.replace(".", ":", regex=False),
                                    format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    df = df.mask(df == -200)
    df = df.rename(columns={
        "CO(GT)": "co_gt", "PT08.S1(CO)": "s1_co", "NMHC(GT)": "nmhc_gt", "C6H6(GT)": "c6h6_gt",
        "PT08.S2(NMHC)": "s2_nmhc", "NOx(GT)": "nox_gt", "PT08.S3(NOx)": "s3_nox",
        "NO2(GT)": "no2_gt", "PT08.S4(NO2)": "s4_no2", "PT08.S5(O3)": "s5_o3",
        "T": "t", "RH": "rh", "AH": "ah"})
    return df.asfreq("h")


def load_airnow_daily(path: Path | str = DATA / "airnow" / "airnow_daily_nyc_region.csv.gz",
                      sites_path: Path | str = DATA / "airnow" / "airnow_sites_nyc_region.csv"):
    """Station-day AirNow values (tidy) and the station table."""
    df = pd.read_csv(path, dtype={"full_aqsid": str}, parse_dates=["date"])
    df["param"] = df.param.map(AIRNOW_PARAMS)
    df = df.dropna(subset=["param", "value"])
    sites = pd.read_csv(sites_path, dtype={"aqsid": str, "full_aqsid": str})
    return df, sites.set_index("full_aqsid")


# Automatic QC applied to the daily AirNow matrices (see aqi2.qc): an 8-hour ozone
# maximum below 1 ppb is physically implausible, and 5 identical consecutive daily
# values indicate a stuck instrument.
QC_RULES = {"pm25": {"floor": None, "min_run": 5}, "o3_8h": {"floor": 1.0, "min_run": 5},
            "o3_1h": {"floor": 1.0, "min_run": 5}}


def airnow_matrix(df: pd.DataFrame, param: str, min_days: int = 365, qc: bool = True) -> pd.DataFrame:
    """Wide date x station matrix for one parameter (stations with enough data)."""
    m = (df[df.param == param]
         .pivot_table(index="date", columns="full_aqsid", values="value", aggfunc="mean")
         .asfreq("D"))
    if qc and param in QC_RULES:
        from .qc import qc_matrix
        m, _ = qc_matrix(m, **QC_RULES[param])
    return m.loc[:, m.notna().sum() >= min_days]


def load_airnow_hourly(path: Path | str):
    """Hourly AirNow extract written by ``scripts/fetch_airnow.py hourly``."""
    path = Path(path)
    df = pd.read_csv(path, dtype={"full_aqsid": str})
    df["time"] = pd.to_datetime(df.time_utc, utc=True)
    sites = pd.read_csv(path.with_name(path.name.replace(".csv.gz", "_sites.csv")),
                        dtype={"aqsid": str, "full_aqsid": str}).set_index("full_aqsid")
    return df.drop(columns="time_utc"), sites


def load_weather(path: Path | str = DATA / "weather" / "nyc_weather_daily.csv.gz") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def regional_weather(w: pd.DataFrame, stations=("LGA", "JFK", "EWR", "NYC")) -> pd.DataFrame:
    """Daily regional mean of the airport observations (precipitation: Central Park)."""
    sub = w[w.station.isin(stations)].copy()
    if {"wdf2", "wsf2"} <= set(sub.columns):
        rad = np.deg2rad(sub.wdf2)
        sub["gust_u"] = -sub.wsf2 * np.sin(rad)
        sub["gust_v"] = -sub.wsf2 * np.cos(rad)
    num = sub.select_dtypes("number").columns
    reg = sub.groupby("date")[list(num)].mean()
    cp = w[w.station == "NYC"].set_index("date")
    for col in ("prcp", "snow", "snwd", "wt_haze", "wt_fog"):
        if col in cp:
            reg[col] = cp[col]
    reg["wt_haze_any"] = sub.groupby("date").wt_haze.max()
    if {"g_tmax", "g_tmin"} <= set(reg.columns):
        reg["tmax"] = reg.t_max.fillna(reg.g_tmax)
        reg["tmin"] = reg.t_min.fillna(reg.g_tmin)
    return reg.drop(columns=["wdf2"], errors="ignore").asfreq("D")


def load_basemap(names=("land", "lakes", "states", "counties", "coastline", "nyc_boroughs")) -> dict:
    import geopandas as gpd
    return {n: gpd.read_file(DATA / "basemap" / f"{n}.geojson") for n in names}
