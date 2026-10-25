"""Air quality indices: one engine, several national standards.

Implemented scales
------------------
* **US EPA AQI** (Technical Assistance Document, May 2024), with both the 2024
  PM2.5 breakpoints and the legacy 2012 ones (``pm25_2012``) so that the effect
  of the revision can be measured. Concentrations are truncated as the EPA
  prescribes before interpolation, and an exact inverse is provided to turn
  published AQI values (e.g. aqicn.org exports) back into concentrations.
* **European Air Quality Index** (EEA, bands used 2017-2023), six levels.
* **CAQI** (Citeair II common air quality index, hourly grid), continuous
  0-100+ scale used by many European cities.
* **AQHI** (Health Canada), an *additive* multi-pollutant health risk index,
  which contrasts with the "worst pollutant wins" logic of the AQI.
* **WHO 2021 Air Quality Guidelines** and interim targets, for exceedance
  statistics.

Units: PM in ug/m3, O3 / NO2 / SO2 in ppb, CO in ppm unless stated otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Unit conversions (25 degC, 1 atm: molar volume 24.45 L/mol)
# ---------------------------------------------------------------------------
MOLAR_VOLUME = 24.45
MOLECULAR_WEIGHT = {"no2": 46.0055, "o3": 47.997, "so2": 64.066, "co": 28.010,
                    "no": 30.006, "nox": 46.0055, "c6h6": 78.11}


def ugm3_to_ppb(x, pollutant: str):
    """ug/m3 -> ppb (for CO, mg/m3 -> ppm uses the same factor)."""
    return np.asarray(x, dtype=float) * MOLAR_VOLUME / MOLECULAR_WEIGHT[pollutant]


def ppb_to_ugm3(x, pollutant: str):
    """ppb -> ug/m3 (for CO, ppm -> mg/m3 uses the same factor)."""
    return np.asarray(x, dtype=float) * MOLECULAR_WEIGHT[pollutant] / MOLAR_VOLUME


# ---------------------------------------------------------------------------
# US EPA AQI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Category:
    name: str
    short: str
    lo: int
    hi: int
    color: str
    advice: str


EPA_CATEGORIES = (
    Category("Good", "Good", 0, 50, "#00E400",
             "Air quality is satisfactory and poses little or no risk."),
    Category("Moderate", "Moderate", 51, 100, "#FFFF00",
             "Acceptable; unusually sensitive people should consider reducing prolonged exertion."),
    Category("Unhealthy for Sensitive Groups", "USG", 101, 150, "#FF7E00",
             "Sensitive groups may experience health effects."),
    Category("Unhealthy", "Unhealthy", 151, 200, "#FF0000",
             "Some members of the general public may experience health effects."),
    Category("Very Unhealthy", "Very Unhealthy", 201, 300, "#8F3F97",
             "Health alert: the risk of health effects is increased for everyone."),
    Category("Hazardous", "Hazardous", 301, 500, "#7E0023",
             "Health warning of emergency conditions: everyone is more likely to be affected."),
)
EPA_COLORS = [c.color for c in EPA_CATEGORIES]
EPA_NAMES = [c.name for c in EPA_CATEGORIES]
EPA_SHORT = [c.short for c in EPA_CATEGORIES]

# (C_lo, C_hi, I_lo, I_hi). Canonical units listed in UNITS below.
EPA_BREAKPOINTS = {
    "pm25": [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
             (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)],
    "pm25_2012": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                  (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
                  (350.5, 500.4, 401, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200),
             (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)],
    "o3_8h": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200),
              (106, 200, 201, 300)],
    "o3_1h": [(125, 164, 101, 150), (165, 204, 151, 200), (205, 404, 201, 300),
              (405, 504, 301, 400), (505, 604, 401, 500)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200),
           (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200),
            (305, 604, 201, 300), (605, 804, 301, 400), (805, 1004, 401, 500)],
    "no2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200),
            (650, 1249, 201, 300), (1250, 1649, 301, 400), (1650, 2049, 401, 500)],
}
UNITS = {"pm25": "ug/m3 (24 h)", "pm25_2012": "ug/m3 (24 h)", "pm10": "ug/m3 (24 h)",
         "o3_8h": "ppb (8 h)", "o3_1h": "ppb (1 h)", "co": "ppm (8 h)",
         "so2": "ppb (1 h)", "no2": "ppb (1 h)"}
# EPA truncation: number of decimals kept before applying the breakpoints
_DECIMALS = {"pm25": 1, "pm25_2012": 1, "pm10": 0, "o3_8h": 0, "o3_1h": 0,
             "co": 1, "so2": 0, "no2": 0}


def _knots(table):
    c = np.array([[lo, hi] for lo, hi, _, _ in table], dtype=float).ravel()
    i = np.array([[lo, hi] for _, _, lo, hi in table], dtype=float).ravel()
    return c, i


def truncate(x, decimals: int):
    f = 10.0 ** decimals
    return np.floor(np.asarray(x, dtype=float) * f + 1e-9) / f


def epa_subindex(conc, pollutant: str, clip: bool = True, round_result: bool = True):
    """EPA sub-index for one pollutant.

    ``pollutant`` is a key of :data:`EPA_BREAKPOINTS`; concentrations are in the
    canonical unit of :data:`UNITS`. Values above the top breakpoint return 500
    when ``clip`` (the EPA convention for reporting) and are extrapolated
    along the last segment otherwise. NaN propagates.
    """
    table = EPA_BREAKPOINTS[pollutant]
    c_knots, i_knots = _knots(table)
    x = truncate(conc, _DECIMALS[pollutant])
    out = np.interp(x, c_knots, i_knots)
    if not clip:
        lo, hi, ilo, ihi = table[-1]
        above = x > hi
        out = np.where(above, ilo + (x - lo) * (ihi - ilo) / (hi - lo), out)
    lo0 = table[0][0]
    out = np.where(x < lo0, np.nan if pollutant == "o3_1h" else 0.0, out)
    out = np.where(np.isnan(x), np.nan, out)
    if round_result:
        out = np.floor(out + 0.5)
    return out if out.ndim else float(out)


def epa_inverse(aqi, pollutant: str):
    """Concentration that produces a given AQI value (inverse of the breakpoints).

    Works for non-integer AQI (e.g. medians) by interpolating linearly between
    the end of one segment and the start of the next.
    """
    c_knots, i_knots = _knots(EPA_BREAKPOINTS[pollutant])
    a = np.asarray(aqi, dtype=float)
    out = np.interp(a, i_knots, c_knots)
    out = np.where(np.isnan(a), np.nan, out)
    return out if out.ndim else float(out)


def epa_o3(o3_8h_ppb=None, o3_1h_ppb=None):
    """Ozone AQI: max of the 8-hour index and, when >= 125 ppb, the 1-hour index."""
    parts = []
    if o3_8h_ppb is not None:
        parts.append(epa_subindex(o3_8h_ppb, "o3_8h"))
    if o3_1h_ppb is not None:
        parts.append(epa_subindex(o3_1h_ppb, "o3_1h"))
    return np.fmax.reduce(np.vstack([np.atleast_1d(p) for p in parts]), axis=0)


def epa_category(aqi) -> np.ndarray:
    """Category index 0 (Good) ... 5 (Hazardous); -1 for NaN."""
    a = np.floor(np.asarray(aqi, dtype=float) + 0.5)
    cat = np.digitize(a, [51, 101, 151, 201, 301])
    return np.where(np.isnan(a), -1, cat)


def epa_category_name(aqi, short: bool = False):
    names = EPA_SHORT if short else EPA_NAMES
    cats = np.atleast_1d(epa_category(aqi))
    return np.array([names[c] if c >= 0 else None for c in cats], dtype=object)


def overall_aqi(subindices: pd.DataFrame) -> pd.DataFrame:
    """Overall AQI (max of sub-indices), dominant pollutant and number of inputs."""
    sub = subindices.astype(float)
    valid = sub.notna()
    aqi = sub.max(axis=1, skipna=True)
    dominant = sub.fillna(-np.inf).idxmax(axis=1).where(valid.any(axis=1))
    return pd.DataFrame({"aqi": aqi, "dominant": dominant, "n_pollutants": valid.sum(axis=1)})


# ---------------------------------------------------------------------------
# European Air Quality Index (EEA), six bands, ug/m3
# ---------------------------------------------------------------------------
EAQI_LEVELS = ("Good", "Fair", "Moderate", "Poor", "Very poor", "Extremely poor")
EAQI_COLORS = ("#50F0E6", "#50CCAA", "#F0E641", "#FF5050", "#960032", "#7D2181")
EAQI_BANDS = {  # upper bounds of the first five bands (ug/m3)
    "pm25": (10, 20, 25, 50, 75),     # 24 h running mean
    "pm10": (20, 40, 50, 100, 150),   # 24 h running mean
    "no2": (40, 90, 120, 230, 340),   # hourly
    "o3": (50, 100, 130, 240, 380),   # hourly
    "so2": (100, 200, 350, 500, 750),  # hourly
}


def eaqi_level(conc_ugm3, pollutant: str) -> np.ndarray:
    """EEA level index 0 (Good) ... 5 (Extremely poor); -1 for NaN."""
    x = np.asarray(conc_ugm3, dtype=float)
    lvl = np.digitize(x, EAQI_BANDS[pollutant], right=True)
    return np.where(np.isnan(x), -1, lvl)


# ---------------------------------------------------------------------------
# CAQI (Citeair II), hourly grid, continuous index
# ---------------------------------------------------------------------------
CAQI_INDEX = (0, 25, 50, 75, 100)
CAQI_GRID = {  # concentrations (ug/m3) at index 0, 25, 50, 75, 100
    "no2": (0, 50, 100, 200, 400),
    "pm10": (0, 25, 50, 90, 180),
    "pm25": (0, 15, 30, 55, 110),
    "o3": (0, 60, 120, 180, 240),
    "co_8h": (0, 5000, 7500, 10000, 20000),
    "so2": (0, 50, 100, 350, 500),
}
CAQI_LEVELS = ("Very low", "Low", "Medium", "High", "Very high")
CAQI_COLORS = ("#79BC6A", "#BBCF4C", "#EEC20B", "#F29305", "#E8416F")


def caqi_subindex(conc_ugm3, pollutant: str) -> np.ndarray:
    """Continuous CAQI sub-index, extrapolated linearly above 100."""
    grid = np.asarray(CAQI_GRID[pollutant], dtype=float)
    x = np.asarray(conc_ugm3, dtype=float)
    out = np.interp(x, grid, CAQI_INDEX)
    slope = 25.0 / (grid[-1] - grid[-2])
    out = np.where(x > grid[-1], 100 + (x - grid[-1]) * slope, out)
    return np.where(np.isnan(x), np.nan, out)


def caqi_level(index) -> np.ndarray:
    x = np.asarray(index, dtype=float)
    lvl = np.digitize(x, (25, 50, 75, 100), right=True)
    return np.where(np.isnan(x), -1, lvl)


# ---------------------------------------------------------------------------
# AQHI (Health Canada): additive risk from NO2, O3 and PM2.5 (3 h means)
# ---------------------------------------------------------------------------
AQHI_BETA = {"no2": 0.000871, "o3": 0.000537, "pm25": 0.000487}
AQHI_LEVELS = ("Low risk", "Moderate risk", "High risk", "Very high risk")


def aqhi(no2_ppb, o3_ppb, pm25_ugm3, round_result: bool = False) -> np.ndarray:
    """AQHI = 10/10.4 * 100 * sum(exp(beta * C) - 1)."""
    risk = ((np.exp(AQHI_BETA["no2"] * np.asarray(no2_ppb, float)) - 1)
            + (np.exp(AQHI_BETA["o3"] * np.asarray(o3_ppb, float)) - 1)
            + (np.exp(AQHI_BETA["pm25"] * np.asarray(pm25_ugm3, float)) - 1))
    out = 1000.0 / 10.4 * risk
    return np.floor(out + 0.5) if round_result else out


def aqhi_level(value) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    lvl = np.digitize(np.floor(x + 0.5), (4, 7, 11))
    return np.where(np.isnan(x), -1, lvl)


# ---------------------------------------------------------------------------
# WHO 2021 Air Quality Guidelines
# ---------------------------------------------------------------------------
WHO_AQG_2021 = {
    # pollutant: {averaging time: (AQG level, interim targets IT-1..IT-4)} in ug/m3 (CO in mg/m3)
    "pm25": {"annual": (5, (35, 25, 15, 10)), "24h": (15, (75, 50, 37.5, 25))},
    "pm10": {"annual": (15, (70, 50, 30, 20)), "24h": (45, (150, 100, 75, 50))},
    "o3": {"peak_season": (60, (100, 70)), "8h": (100, (160, 120))},
    "no2": {"annual": (10, (40, 30, 20)), "24h": (25, (120, 50))},
    "so2": {"24h": (40, (125, 50))},
    "co": {"24h": (4, (7,))},
}


def who_guideline(pollutant: str, averaging: str) -> float:
    return float(WHO_AQG_2021[pollutant][averaging][0])


# ---------------------------------------------------------------------------
# Convenience for daily AQI tables (e.g. the aqicn.org export of New York)
# ---------------------------------------------------------------------------
AQICN_TO_EPA = {"pm25": "pm25_2012", "pm10": "pm10", "o3": "o3_8h", "no2": "no2",
                "co": "co", "so2": "so2"}


def aqicn_to_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Invert aqicn.org sub-indices (US EPA scale, 2012 PM2.5 table) to concentrations."""
    out = {}
    for col in df.columns:
        if col in AQICN_TO_EPA:
            out[col] = epa_inverse(df[col].to_numpy(dtype=float), AQICN_TO_EPA[col])
    return pd.DataFrame(out, index=df.index)
