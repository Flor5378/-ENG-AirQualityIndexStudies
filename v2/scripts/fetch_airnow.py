"""Download AirNow public flat files and keep the stations inside a bounding box.

AirNow (US EPA) mirrors its real-time data as flat files on S3
(``files.airnowtech.org``). Three layouts are used here:

* ``daily_data.dat`` (until 2018):
  ``Date|AQSID|SiteName|Parameter|Units|Value|AveragingPeriod|DataSource``
* ``daily_data_v2.dat`` (2019 onwards): the same fields followed by
  ``AQI|Category|Latitude|Longitude|FullAQSID``
* ``HourlyData_YYYYMMDDHH.dat`` (hour in UTC):
  ``Date|Time|AQSID|SiteName|GMTOffset|Parameter|Units|Value|DataSource``

Station coordinates for the legacy and hourly layouts come from
``Monitoring_Site_Locations_V2.dat``, which also lists inactive sites.

Examples
--------
Daily files for the New York tri-state region::

    python fetch_airnow.py daily --start 2014-01-01 --end 2026-10-23

Hourly files for the June 2023 Canadian wildfire smoke episode::

    python fetch_airnow.py hourly --start 2023-06-05T00 --end 2023-06-09T23 \
        --bbox 38.5 47.5 -81.0 -67.0 --out ../data/airnow/airnow_hourly_smoke_2023.csv.gz

A year of hourly NO2 / O3 sampled every 5 hours (all hours of the day equally)::

    python fetch_airnow.py hourly --start 2024-01-01T00 --end 2024-12-31T23 --step-hours 5 \
        --params NO2 OZONE PM2.5 --out ../data/airnow/airnow_hourly_2024_every5h.csv.gz
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BASE = "https://s3-us-west-1.amazonaws.com/files.airnowtech.org/airnow"
SITES_URL = f"{BASE}/today/Monitoring_Site_Locations_V2.dat"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "airnow"

# New York tri-state region with a buffer, so that interpolation near the
# edges of the map still has neighbours on every side.
DEFAULT_BBOX = (39.5, 42.3, -76.0, -71.8)  # lat_min, lat_max, lon_min, lon_max

DAILY_V1 = ["date", "aqsid", "site", "param", "units", "value", "avg_hours", "source"]
DAILY_V2 = DAILY_V1 + ["aqi", "aqi_cat", "lat", "lon", "full_aqsid"]
HOURLY = ["date", "time", "aqsid", "site", "gmt_offset", "param", "units", "value", "source"]

_session = requests.Session()


def _get(url: str, retries: int = 4) -> bytes | None:
    """GET with exponential backoff. Returns None on 404/403 (file absent)."""
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}")


def load_sites() -> pd.DataFrame:
    """One row per AQSID with coordinates (first active record wins)."""
    raw = _get(SITES_URL)
    sites = pd.read_csv(io.BytesIO(raw), sep="|", encoding="latin-1", dtype=str)
    sites = sites.rename(columns={
        "AQSID": "aqsid", "FullAQSID": "full_aqsid", "SiteName": "site",
        "Latitude": "lat", "Longitude": "lon", "Elevation": "elevation",
        "AgencyName": "agency", "Status": "status", "StateAbbreviation": "state",
        "CountyName": "county", "CBSA_Name": "cbsa", "Parameter": "param",
    })
    for c in ("lat", "lon", "elevation"):
        sites[c] = pd.to_numeric(sites[c], errors="coerce")
    sites = sites.dropna(subset=["lat", "lon"])
    sites = sites[(sites.lat != 0) | (sites.lon != 0)]
    sites["active"] = sites["status"].eq("Active")
    params = sites.groupby("aqsid")["param"].apply(lambda s: ",".join(sorted(set(s))))
    sites = (sites.sort_values("active", ascending=False)
                  .drop_duplicates("aqsid")
                  .drop(columns=["param"])
                  .merge(params.rename("params"), on="aqsid"))
    keep = ["aqsid", "full_aqsid", "site", "lat", "lon", "elevation", "agency",
            "state", "county", "cbsa", "status", "params"]
    return sites[keep].reset_index(drop=True)


def in_bbox(lat: pd.Series, lon: pd.Series, bbox) -> pd.Series:
    la0, la1, lo0, lo1 = bbox
    return lat.between(la0, la1) & lon.between(lo0, lo1)


def to_full_id(aqsid: pd.Series, id_map: dict[str, str]) -> pd.Series:
    """Map 9-digit AQSIDs to the 12-digit FullAQSID (12-digit ids pass through)."""
    return aqsid.where(aqsid.str.len() == 12, aqsid.map(id_map))


def fetch_daily(day: pd.Timestamp, bbox, id_map: dict[str, str]) -> pd.DataFrame | None:
    ymd = day.strftime("%Y%m%d")
    raw = _get(f"{BASE}/{day.year}/{ymd}/daily_data_v2.dat")
    if raw is not None:
        df = pd.read_csv(io.BytesIO(raw), sep="|", header=None, names=DAILY_V2,
                         encoding="latin-1", dtype={"aqsid": str, "full_aqsid": str})
        df = df[in_bbox(df.lat, df.lon, bbox)]
        df["full_aqsid"] = df.full_aqsid.fillna(to_full_id(df.aqsid, id_map))
        fmt = "v2"
    else:
        raw = _get(f"{BASE}/{day.year}/{ymd}/daily_data.dat")
        if raw is None:
            return None
        df = pd.read_csv(io.BytesIO(raw), sep="|", header=None, names=DAILY_V1,
                         usecols=range(8), encoding="latin-1", dtype={"aqsid": str})
        df["full_aqsid"] = to_full_id(df.aqsid, id_map)
        df = df[df.full_aqsid.isin(set(id_map.values()))]
        fmt = "v1"
    df = df.assign(date=day.strftime("%Y-%m-%d"), fmt=fmt)
    cols = ["date", "full_aqsid", "param", "units", "value", "avg_hours", "aqi", "fmt", "site", "lat", "lon"]
    return df.reindex(columns=cols)


def fetch_hourly(ts: pd.Timestamp, id_map: dict[str, str], params: set[str]) -> pd.DataFrame | None:
    raw = _get(f"{BASE}/{ts.year}/{ts:%Y%m%d}/HourlyData_{ts:%Y%m%d%H}.dat")
    if raw is None:
        return None
    df = pd.read_csv(io.BytesIO(raw), sep="|", header=None, names=HOURLY,
                     encoding="latin-1", dtype={"aqsid": str})
    df = df[df.param.isin(params)]
    df["full_aqsid"] = to_full_id(df.aqsid, id_map)
    df = df[df.full_aqsid.isin(set(id_map.values()))]
    df = df.assign(time_utc=ts.strftime("%Y-%m-%dT%H:00Z"))
    return df[["time_utc", "full_aqsid", "param", "units", "value"]]


def run_parallel(func, items, workers: int, label: str) -> pd.DataFrame:
    frames, missing = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(func, it): it for it in items}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is None:
                missing.append(futures[fut])
            elif len(res):
                frames.append(res)
            if i % 200 == 0 or i == len(items):
                print(f"[{label}] {i}/{len(items)} files  {time.time() - t0:5.0f}s", flush=True)
    if missing:
        print(f"[{label}] {len(missing)} files not found, e.g. {missing[:5]}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["daily", "hourly"])
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX,
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    p.add_argument("--params", nargs="*", default=["PM2.5", "OZONE", "NO2", "CO", "PM10", "SO2"],
                   help="hourly mode only: parameters to keep")
    p.add_argument("--step-hours", type=int, default=1,
                   help="hourly mode only: keep one file every N hours (N coprime with 24 "
                        "samples every hour of the day equally often)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path)
    p.add_argument("--sites-out", type=Path)
    a = p.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sites = load_sites()
    box_sites = sites[in_bbox(sites.lat, sites.lon, a.bbox)]
    id_map = dict(zip(box_sites.aqsid, box_sites.full_aqsid))
    id_map.update({f: f for f in box_sites.full_aqsid})
    print(f"{len(box_sites)} monitoring sites inside bbox {a.bbox}")

    if a.mode == "daily":
        days = pd.date_range(a.start, a.end, freq="D")
        df = run_parallel(lambda d: fetch_daily(d, a.bbox, id_map), list(days), a.workers, "daily")
        df = df.dropna(subset=["full_aqsid"])
        df = df.drop_duplicates(["date", "full_aqsid", "param"]).sort_values(["date", "full_aqsid", "param"])
        # monitors absent from the site list still carry coordinates in the v2 files
        extra = (df.dropna(subset=["lat", "lon"]).drop_duplicates("full_aqsid", keep="last")
                   .loc[lambda x: ~x.full_aqsid.isin(box_sites.full_aqsid), ["full_aqsid", "site", "lat", "lon"]])
        if len(extra):
            extra = extra.assign(aqsid=extra.full_aqsid.str[-9:], status="unlisted")
            box_sites = pd.concat([box_sites, extra], ignore_index=True)
            print(f"{len(extra)} monitors not in the site list, coordinates taken from the daily files")
        df = df.drop(columns=["site", "lat", "lon"])
        out = a.out or DATA_DIR / "airnow_daily_nyc_region.csv.gz"
        sites_out = a.sites_out or DATA_DIR / "airnow_sites_nyc_region.csv"
    else:
        hours = pd.date_range(a.start, a.end, freq=f"{a.step_hours}h")
        df = run_parallel(lambda t: fetch_hourly(t, id_map, set(a.params)), list(hours), a.workers, "hourly")
        df = df.drop_duplicates(["time_utc", "full_aqsid", "param"]).sort_values(["time_utc", "full_aqsid", "param"])
        out = a.out or DATA_DIR / "airnow_hourly.csv.gz"
        sites_out = a.sites_out or out.with_name(out.name.replace(".csv.gz", "_sites.csv"))

    used = box_sites[box_sites.full_aqsid.isin(set(df.full_aqsid))]
    df.to_csv(out, index=False, compression="gzip")
    used.to_csv(sites_out, index=False)
    print(f"wrote {len(df):,} rows to {out} and {len(used)} sites to {sites_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
