"""Daily weather covariates for the New York region from NOAA open data on AWS.

Two NOAA archives are combined:

* **Global Hourly / ISD** (``noaa-global-hourly-pds``): routine METAR reports
  (``FM-15``) give temperature, dew point, sea-level pressure, wind and
  visibility. They are aggregated to local (America/New_York) calendar days.
* **GHCN-Daily** (``noaa-ghcn-pds``): daily precipitation, snow and the
  "smoke or haze" (WT08) and "fog" (WT01) weather-type flags.

Example::

    python fetch_weather.py --start 2014-01-01 --end 2026-10-23
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ISD_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{usaf_wban}.csv"
GHCN_URL = "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{ghcn}.csv"
# The per-station GHCN files on AWS lag behind; the yearly files are refreshed
# every day, so recent years are read from them instead.
GHCN_YEAR_URL = "https://noaa-ghcn-pds.s3.amazonaws.com/csv.gz/by_year/{year}.csv.gz"
GHCN_ELEMENTS = {"TMAX": 10.0, "TMIN": 10.0, "PRCP": 10.0, "SNOW": 1.0, "SNWD": 1.0,
                 "AWND": 10.0, "WSF2": 10.0, "WDF2": 1.0, "WT01": 1.0, "WT08": 1.0}
GHCN_NAMES = {"TMAX": "g_tmax", "TMIN": "g_tmin", "PRCP": "prcp", "SNOW": "snow", "SNWD": "snwd",
              "AWND": "awnd", "WSF2": "wsf2", "WDF2": "wdf2", "WT01": "wt_fog", "WT08": "wt_haze"}
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "weather"

# code: (ISD USAF-WBAN id, GHCN-D id, name, lat, lon)
STATIONS = {
    "NYC": ("72505394728", "USW00094728", "New York Central Park", 40.7790, -73.9690),
    "LGA": ("72503014732", "USW00014732", "LaGuardia Airport", 40.7794, -73.8803),
    "JFK": ("74486094789", "USW00094789", "JFK Airport", 40.6386, -73.7622),
    "EWR": ("72502014734", "USW00014734", "Newark Airport", 40.6825, -74.1694),
    "HPN": ("72503794745", "USW00094745", "White Plains Airport", 41.0670, -73.7076),
    "ISP": ("72505004781", "USW00004781", "Islip Airport", 40.7939, -73.1017),
    "BDR": ("72504094702", "USW00094702", "Bridgeport Airport", 41.1583, -73.1289),
}

_session = requests.Session()


def _get(url: str, retries: int = 4) -> bytes | None:
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=120)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}")


def _scaled(field: pd.Series, idx: int, missing: str, scale: float) -> pd.Series:
    part = field.str.split(",", expand=True)[idx]
    out = pd.to_numeric(part, errors="coerce")
    return out.mask(part.str.lstrip("+-").eq(missing.lstrip("+-"))) / scale


def parse_isd(raw: bytes) -> pd.DataFrame:
    """Parse one Global Hourly CSV file into hourly physical quantities."""
    cols = ["DATE", "REPORT_TYPE", "WND", "VIS", "TMP", "DEW", "SLP"]
    df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in cols, dtype=str, low_memory=False)
    df = df[df.REPORT_TYPE.str.strip().eq("FM-15")].copy()
    t = pd.to_datetime(df.DATE, utc=True)
    out = pd.DataFrame({"time": t.dt.tz_convert("America/New_York")})
    out["t"] = _scaled(df.TMP, 0, "9999", 10).values
    out["td"] = _scaled(df.DEW, 0, "9999", 10).values
    out["slp"] = _scaled(df.SLP, 0, "99999", 10).values
    out["vis"] = _scaled(df.VIS, 0, "999999", 1).values
    wdir = _scaled(df.WND, 0, "999", 1)
    wtype = df.WND.str.split(",", expand=True)[2]
    ws = _scaled(df.WND, 3, "9999", 10)
    wdir = wdir.where(~wtype.eq("C"), np.nan)  # calm: direction undefined
    ws = ws.where(~wtype.eq("C"), 0.0)
    out["ws"] = ws.values
    rad = np.deg2rad(wdir.values)
    # meteorological convention: wind blowing FROM `wdir`
    out["u"] = np.where(ws.values == 0, 0.0, -ws.values * np.sin(rad))
    out["v"] = np.where(ws.values == 0, 0.0, -ws.values * np.cos(rad))
    return out.dropna(subset=["time"])


def relative_humidity(t: pd.Series, td: pd.Series) -> pd.Series:
    """Magnus formula, percent."""
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td / (b + td) - a * t / (b + t))


def daily_from_hourly(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    h["rh"] = relative_humidity(h.t, h.td)
    h["date"] = h.time.dt.tz_localize(None).dt.normalize()
    h["afternoon"] = h.time.dt.hour.between(12, 17)
    g = h.groupby("date")
    d = pd.DataFrame({
        "t_mean": g.t.mean(), "t_max": g.t.max(), "t_min": g.t.min(),
        "td_mean": g.td.mean(), "rh_mean": g.rh.mean(), "slp_mean": g.slp.mean(),
        "ws_mean": g.ws.mean(), "ws_max": g.ws.max(),
        "u_mean": g.u.mean(), "v_mean": g.v.mean(),
        "calm_frac": g.ws.apply(lambda s: (s < 1.5).mean()),
        "vis_mean": g.vis.mean() / 1000, "vis_min": g.vis.min() / 1000,
        "n_obs": g.t.count(),
    })
    aft = h[h.afternoon].groupby("date")
    d["t_aft"] = aft.t.mean()
    d["ws_aft"] = aft.ws.mean()
    d["slp_tendency"] = d.slp_mean.diff()
    return d


def fetch_isd_station(code: str, years: list[int]) -> pd.DataFrame:
    usaf_wban = STATIONS[code][0]
    frames = []
    for y in years:
        raw = _get(ISD_URL.format(year=y, usaf_wban=usaf_wban))
        if raw is not None:
            frames.append(parse_isd(raw))
    if not frames:
        return pd.DataFrame()
    d = daily_from_hourly(pd.concat(frames, ignore_index=True))
    print(f"  ISD {code}: {len(d)} days", flush=True)
    return d.assign(station=code)


def _ghcn_wide(g: pd.DataFrame) -> pd.DataFrame:
    g = g[g.Q_FLAG.isna() & g.ELEMENT.isin(GHCN_ELEMENTS)].copy()
    g["DATA_VALUE"] = pd.to_numeric(g.DATA_VALUE, errors="coerce") / g.ELEMENT.map(GHCN_ELEMENTS)
    w = g.pivot_table(index=["ID", "DATE"], columns="ELEMENT", values="DATA_VALUE", aggfunc="first")
    w = w.rename(columns=GHCN_NAMES).reset_index()
    w["DATE"] = pd.to_datetime(w.DATE, format="%Y%m%d")
    return w


def fetch_ghcn(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """GHCN-Daily for the selected stations: per-station files, then yearly files."""
    ids = {STATIONS[c][1]: c for c in codes}
    cols = ["ID", "DATE", "ELEMENT", "DATA_VALUE", "M_FLAG", "Q_FLAG", "S_FLAG", "OBS_TIME"]
    frames = []
    for gid in ids:
        raw = _get(GHCN_URL.format(ghcn=gid))
        if raw is not None:
            g = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False, header=0, names=cols)
            frames.append(_ghcn_wide(g[g.DATE >= start.replace("-", "")]))
    hist = pd.concat(frames, ignore_index=True)
    last = hist.groupby("ID").DATE.max().min()
    recent = []
    for year in range(last.year, pd.Timestamp(end).year + 1):
        raw = _get(GHCN_YEAR_URL.format(year=year))
        if raw is None:
            continue
        for chunk in pd.read_csv(io.BytesIO(raw), compression="gzip", header=None, names=cols,
                                 dtype=str, chunksize=4_000_000):
            chunk = chunk[chunk.ID.isin(ids)]
            if len(chunk):
                recent.append(chunk)
        print(f"  GHCN yearly file {year} scanned", flush=True)
    if recent:
        hist = pd.concat([_ghcn_wide(pd.concat(recent)), hist], ignore_index=True)
    w = hist.drop_duplicates(["ID", "DATE"]).sort_values(["ID", "DATE"])
    for flag in ("wt_fog", "wt_haze"):
        if flag in w:
            w[flag] = w[flag].fillna(0).clip(0, 1)
    w["station"] = w.ID.map(ids)
    w = w.drop(columns="ID").rename(columns={"DATE": "date"}).set_index("date")
    for code, d in w.groupby("station"):
        print(f"  GHCN {code}: {len(d)} days, last {d.index.max():%Y-%m-%d}", flush=True)
    return w


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default="2026-10-23")
    p.add_argument("--stations", nargs="*", default=list(STATIONS))
    p.add_argument("--out", type=Path, default=OUT_DIR / "nyc_weather_daily.csv.gz")
    a = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # one extra year on each side so that local-day aggregation is complete
    years = list(range(pd.Timestamp(a.start).year - 1, pd.Timestamp(a.end).year + 1))
    with ThreadPoolExecutor(max_workers=len(a.stations)) as pool:
        isd = list(pool.map(lambda c: fetch_isd_station(c, years), a.stations))
    ghcn = fetch_ghcn(a.stations, a.start, a.end)

    frames = []
    for code, di in zip(a.stations, isd):
        dg = ghcn[ghcn.station == code]
        if di.empty and dg.empty:
            continue
        di = di.drop(columns="station", errors="ignore")
        dg = dg.drop(columns="station", errors="ignore")
        d = di.join(dg, how="outer")
        d = d.loc[a.start:a.end]
        d.index.name = "date"
        frames.append(d.reset_index().assign(station=code))
    out = pd.concat(frames, ignore_index=True)
    first = ["date", "station"]
    out = out[first + [c for c in out.columns if c not in first]]
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    out.to_csv(a.out, index=False, compression="gzip")

    meta = pd.DataFrame([(k, *v) for k, v in STATIONS.items() if k in a.stations],
                        columns=["station", "isd_id", "ghcn_id", "name", "lat", "lon"])
    meta.to_csv(a.out.parent / "weather_stations.csv", index=False)
    print(f"wrote {len(out):,} station-days to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
