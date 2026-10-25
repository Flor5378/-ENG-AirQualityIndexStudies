"""Basemap layers for the maps: Natural Earth 1:10m, clipped to the study area.

Writes small GeoJSON files to ``v2/data/basemap``:

* ``land.geojson``, ``lakes.geojson``, ``coastline.geojson`` and
  ``states.geojson`` for the north-eastern United States and southern Quebec
  (extent of the wildfire smoke animation);
* ``counties.geojson`` for the New York tri-state region;
* ``nyc_boroughs.geojson``: the five counties that make up New York City,
  with their 2020 Census population.

Example::

    python fetch_basemap.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box

NE = "https://naturalearth.s3.amazonaws.com/10m_{kind}/ne_10m_{name}.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "basemap"
NORTHEAST = box(-82.0, 38.0, -66.0, 48.0)
TRISTATE = box(-76.5, 39.3, -71.3, 42.5)

# 2020 Census (P1 table) resident population of the five boroughs.
BOROUGHS = {
    "36005": ("Bronx", 1_472_654),
    "36047": ("Brooklyn", 2_736_074),
    "36061": ("Manhattan", 1_694_251),
    "36081": ("Queens", 2_405_464),
    "36085": ("Staten Island", 495_747),
}


def read_ne(kind: str, name: str) -> gpd.GeoDataFrame:
    r = requests.get(NE.format(kind=kind, name=name), timeout=300)
    r.raise_for_status()
    tmp = OUT / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(tmp)
    gdf = gpd.read_file(next(tmp.glob(f"ne_10m_{name}.shp")))
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return gdf.to_crs(4326)


def clip(gdf: gpd.GeoDataFrame, area, keep: list[str], tol: float = 0.0) -> gpd.GeoDataFrame:
    out = gpd.clip(gdf[[c for c in keep if c in gdf.columns] + ["geometry"]], area)
    out = out[~out.geometry.is_empty]
    if tol:
        out["geometry"] = out.geometry.simplify(tol, preserve_topology=True)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    layers = {
        "land": ("physical", "land", [], 0.002),
        "lakes": ("physical", "lakes", ["name"], 0.002),
        "coastline": ("physical", "coastline", [], 0.002),
        "states": ("cultural", "admin_1_states_provinces", ["name", "postal", "admin"], 0.002),
    }
    for out_name, (kind, name, keep, tol) in layers.items():
        gdf = clip(read_ne(kind, name), NORTHEAST, keep, tol)
        gdf.to_file(OUT / f"{out_name}.geojson", driver="GeoJSON")
        print(f"{out_name}: {len(gdf)} features")

    counties = read_ne("cultural", "admin_2_counties")
    fips_col = next(c for c in ("CODE_LOCAL", "FIPS", "fips") if c in counties.columns)
    counties = counties.rename(columns={fips_col: "fips", "NAME": "name", "REGION": "state"})
    counties["fips"] = counties["fips"].astype(str).str.replace("US", "").str.zfill(5)
    tri = clip(counties, TRISTATE, ["fips", "name", "state"], 0.0005)
    tri.to_file(OUT / "counties.geojson", driver="GeoJSON")
    print(f"counties: {len(tri)} features")

    nyc = counties[counties.fips.isin(BOROUGHS)].copy()
    nyc["borough"] = nyc.fips.map(lambda f: BOROUGHS[f][0])
    nyc["population_2020"] = nyc.fips.map(lambda f: BOROUGHS[f][1])
    nyc = nyc[["fips", "borough", "population_2020", "geometry"]]
    nyc.to_file(OUT / "nyc_boroughs.geojson", driver="GeoJSON")
    print(f"nyc boroughs: {len(nyc)} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
