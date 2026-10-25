"""Export the data files behind the interactive dashboard.

Reads the processed outputs of the executed notebooks (``data/processed``), the
tables they displayed (forecast scores, spatial benchmark, COVID-19 effects,
trends) and the AirNow snapshot, and writes small row-shaped CSV and JSON files
to ``data/dashboard``. The maps are recomputed with the same Gaussian-process
settings as notebook 03, on its 0.02-degree grid.

    python export_dashboard.py
"""
from __future__ import annotations

import json
import sys
import warnings
from io import StringIO
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from aqi2 import data, indices as ix, spatial as sp  # noqa: E402

OUT = data.DATA / "dashboard"
PROCESSED = data.DATA / "processed"
METRO = (-75.05, -72.55, 40.15, 41.45)
NE = (-80.5, -67.5, 38.6, 47.4)
EPA_NAMES = list(ix.EPA_NAMES)
DOMINANT = {"pm25": "PM2.5", "o3": "Ozone"}


def displayed_table(notebook: str, columns: set[str], header=0) -> pd.DataFrame:
    """First table displayed by an executed notebook whose header contains ``columns``."""
    nb = nbformat.read(V2 / "notebooks" / notebook, as_version=4)
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            html = out.get("data", {}).get("text/html") if out.get("output_type") in ("display_data",
                                                                                      "execute_result") else None
            if not html or "<table" not in html:
                continue
            tab = pd.read_html(StringIO(html), header=header, index_col=0)[0]
            names = {str(c) for col in tab.columns for c in (col if isinstance(col, tuple) else (col,))}
            if columns <= names:
                return tab
    raise LookupError(f"no table with columns {sorted(columns)} in {notebook}")


def interval(text: str) -> tuple[float, float]:
    lo, hi = text.strip("[]").split(",")
    return float(lo), float(hi)


def daily() -> pd.DataFrame:
    d = pd.read_csv(PROCESSED / "nyc_aqi_v2_daily.csv", parse_dates=["date"])
    aqi_2012 = np.fmax(d.aqi_pm25_2012, d.aqi_o3)
    eaqi = np.maximum(ix.eaqi_level(d.pm25, "pm25"), ix.eaqi_level(ix.ppb_to_ugm3(d.o3, "o3"), "o3"))
    return pd.DataFrame({
        "date": d.date.dt.strftime("%Y-%m-%d"),
        "aqi": d.aqi.round().astype("Int64"),
        "category": [EPA_NAMES[c] for c in ix.epa_category(d.aqi)],
        "dominant": d.dominant.map(DOMINANT),
        "pm25": d.pm25.round(1), "o3": d.o3.round(1), "no2": d.no2.round(1),
        "aqi_2012": aqi_2012.round().astype("Int64"),
        "category_2012": [EPA_NAMES[c] for c in ix.epa_category(aqi_2012)],
        "eaqi": [ix.EAQI_LEVELS[k] for k in eaqi],
        "gap_filled": d.pm25_reconstructed | d.o3_reconstructed,
    })


def forecast(last_day: str) -> pd.DataFrame:
    f = pd.read_csv(PROCESSED / "nyc_live_forecast.csv")
    pm, oz = (f[c].str.split(" to ", expand=True).astype(float) for c in ("PM2.5 80% interval", "O3 80% interval"))
    lo = np.fmax(ix.epa_subindex(pm[0], "pm25"), ix.epa_subindex(oz[0], "o3_8h"))
    return pd.DataFrame({"date": f.date, "issued": last_day, "aqi": f.AQI.astype(int), "aqi_lo": lo.astype(int),
                         "aqi_hi": f["AQI upper (80%)"].astype(int), "category": f.category,
                         "pm25": f["PM2.5 (ug/m3)"], "pm25_lo": pm[0], "pm25_hi": pm[1],
                         "o3": f["O3 (ppb)"], "o3_lo": oz[0], "o3_hi": oz[1]})


def forecast_scores() -> pd.DataFrame:
    t = displayed_table("01_nyc_aqi_v2.ipynb", {"skill vs persistence", "coverage 80% (CQR)"})
    return pd.DataFrame({
        "pollutant": t.target.map(DOMINANT), "setting": t.setting, "lead_days": t.h,
        "mae": t.MAE.round(2), "mae_persistence": t["MAE persistence"].round(2),
        "mae_climatology": t["MAE climatology"].round(2),
        "skill_persistence": t["skill vs persistence"].round(3), "skill_climatology": t["skill vs climatology"].round(3),
        "coverage": t["coverage 80% (CQR)"].round(3), "coverage_raw": t["coverage 80% (raw)"].round(3),
    })


def spatial_benchmark() -> pd.DataFrame:
    t = displayed_table("03_geospatial_nyc.ipynb", {"RMSE", "R2", "MAE"}, header=[0, 1])
    family = {"nearest monitor": "baseline", "network mean": "baseline", "IDW": "geostatistics",
              "ordinary kriging": "geostatistics", "Gaussian process": "geostatistics",
              "RFSI-LightGBM": "machine learning", "kriging + boosted residual": "machine learning"}
    rows = []
    for method in family:
        for p in ("pm25", "o3"):
            rows.append({"pollutant": DOMINANT[p], "method": method, "family": family[method],
                         "rmse": t.loc[method, ("RMSE", p)], "mae": t.loc[method, ("MAE", p)],
                         "r2": t.loc[method, ("R2", p)]})
    return pd.DataFrame(rows)


def covid(r1: dict) -> pd.DataFrame:
    t = displayed_table("01_nyc_aqi_v2.ipynb", {"lockdown effect (%)", "90% CI"})
    rows = []
    for _, row in t.iterrows():
        key = "no2" if row.pollutant == "no2" else "pm"
        lo, hi = interval(row["90% CI"])
        rows.append({"pollutant": "NO2" if row.pollutant == "no2" else "PM2.5",
                     "effect": row["lockdown effect (%)"], "ci_lo": lo, "ci_hi": hi,
                     "placebo_min": r1[f"placebo_{key}_min"], "placebo_max": r1[f"placebo_{key}_max"],
                     "placebo_years": int(row["placebo years"])})
    return pd.DataFrame(rows)


def trends() -> pd.DataFrame:
    t = displayed_table("01_nyc_aqi_v2.ipynb", {"slope per year", "Mann-Kendall p"})
    ci = t["95% CI"].map(interval)
    return pd.DataFrame({"pollutant": t.pollutant.map(DOMINANT), "series": t.series,
                         "slope": t["slope per year"], "ci_lo": [a for a, _ in ci], "ci_hi": [b for _, b in ci],
                         "pct_per_year": t["% per year"].round(2), "p_value": t["Mann-Kendall p"],
                         "unit": t.pollutant.map({"pm25": "µg/m³ per year", "o3": "ppb per year"})})


def virtual_sensors(r2: dict) -> pd.DataFrame:
    names = {"co_gt": "CO", "no2_gt": "NO2", "nox_gt": "NOx"}
    order = ["static", "quarterly", "monthly", "weekly"]
    return pd.DataFrame([{"target": names[k.split("|")[0]], "model": "LightGBM" if k.endswith("lightgbm") else "linear",
                          "recalibration": s, "step": order.index(s), "r2": v[s]}
                         for k, v in r2["vs_r2"].items() for s in order])


def caqi_forecast(r2: dict) -> pd.DataFrame:
    return pd.DataFrame([{"lead_hours": int(h), "mae": v["mae"], "skill_persistence": v["skill_pers"] / 100,
                          "skill_climatology": v["skill_clim"] / 100, "coverage": v["cov"] / 100,
                          "coverage_raw": v["cov_raw"] / 100} for h, v in r2["forecast"].items()])


def key_results(r1: dict, r2: dict, r3: dict) -> pd.DataFrame:
    rows = [
        ("val_oz_n", "Official ozone AQI values recomputed", r1["val_oz_n"], "station-days"),
        ("val_oz_exact", "Ozone AQI values matched exactly", r1["val_oz_exact"] / 100, "share"),
        ("val_pm_within", "PM2.5 AQI values within one point", r1["val_pm_within"] / 100, "share"),
        ("v1_good", "Days rated Good by the v1 reading", r1["v1_good"] / 100, "share"),
        ("v2_good", "Same days rated Good, correct reading", r1["v2_good"] / 100, "share"),
        ("corr_before", "Export gases vs EPA monitors, as dated", r1["corr_before"], "correlation"),
        ("corr_after", "Export gases vs EPA monitors, shifted one day", r1["corr_after"], "correlation"),
        ("flag_oz", "Days of ozone stuck at the AQI ceiling, 2019", r1["flag_oz"], "days"),
        ("flag_pm", "Days of PM2.5 drift, July 2021 to January 2022", r1["flag_pm"], "days"),
        ("fill_r2_pm", "Gap filling R², PM2.5", r1["fill_r2_pm"], "R²"),
        ("fill_r2_oz", "Gap filling R², ozone", r1["fill_r2_oz"], "R²"),
        ("worst_aqi", "Worst daily AQI, June 7, 2023", r1["worst_aqi"], "AQI"),
        ("worst_pm", "City-wide 24 h PM2.5 that day", r1["worst_pm"], "µg/m³"),
        ("fireworks_pm", "PM2.5 added by July 4 fireworks", r1["fireworks_pm"], "µg/m³"),
        ("cat_acc", "Next-day AQI category right, with weather forecast", r1["cat_acc"] / 100, "share"),
        ("cat_acc_pers", "Next-day AQI category right, persistence", r1["cat_acc_pers"] / 100, "share"),
        ("raw_cov", "Coverage of raw 80% intervals", r1["raw_cov"] / 100, "share"),
        ("benzene_r2", "Benzene reference explained by one sensor", r2["benzene_r2"], "R²"),
        ("lowcost_exact", "Hours with the reference CAQI level, virtual sensors", r2["lowcost_exact"] / 100, "share"),
        ("lowcost_one", "Hours within one CAQI level", r2["lowcost_one"] / 100, "share"),
        ("pop_pm", "Population-weighted PM2.5, 2025", r3["pop_pm"], "µg/m³"),
        ("pop_ratio", "Times the WHO annual guideline", r3["pop_ratio"], "ratio"),
        ("n_pm", "PM2.5 monitors in the regional network", r3["n_pm"], "monitors"),
        ("n_o3", "Ozone monitors in the regional network", r3["n_o3"], "monitors"),
    ]
    return pd.DataFrame(rows, columns=["key", "label", "value", "unit"])


def boroughs(r3: dict, layers: dict) -> pd.DataFrame:
    pop = layers["nyc_boroughs"].set_index("borough").population_2020
    return pd.DataFrame([{"borough": b, "population": int(pop[b]), "pm25_2025": v["pm"], "days_above_who": v["days"],
                          "person_days_m": v["person_days_m"]} for b, v in r3["boroughs"].items()])


def maps(layers: dict):
    an, sites = data.load_airnow_daily()
    PM = data.airnow_matrix(an, "pm25", min_days=180).clip(lower=0)
    O3 = data.airnow_matrix(an, "o3_8h", min_days=180).clip(lower=0)
    lon_c, lat_c, LON, LAT = sp.make_grid(METRO, 0.02)
    land = sp.land_mask(LON, LAT, layers)
    near = land.copy()                     # land plus one cell, so the coastline clip leaves no gap
    near[1:] |= land[:-1]
    near[:-1] |= land[1:]
    near[:, 1:] |= near[:, :-1].copy()
    near[:, :-1] |= near[:, 1:].copy()
    XY = sp.to_xy_km(LAT.ravel(), LON.ravel())

    def gp_map(values: pd.Series, length_km: float):
        v = values.dropna()
        xy = sp.to_xy_km(sites.loc[v.index, "lat"], sites.loc[v.index, "lon"])
        return sp.make_gp(length_km=length_km).fit(xy, v.to_numpy()).predict(XY)

    def expected(M, year, threshold, months=None):
        total, n = np.zeros(LON.size), 0
        days = M.loc[str(year)].index
        if months:
            days = days[days.month.isin(months)]
        for d in days:
            z = M.loc[d].dropna()
            if len(z) < 15:
                continue
            xy = sp.to_xy_km(sites.loc[z.index, "lat"], sites.loc[z.index, "lon"])
            mu, sd = sp.make_gp(length_km=60).fit(xy, z.to_numpy()).predict(XY, return_std=True)
            total += sp.exceedance_probability(mu, sd, threshold)
            n += 1
        return total, n

    period = slice("2022-01-01", "2025-12-31")
    pm_mean = PM.loc[period].mean().where(PM.loc[period].notna().sum() > 600)
    summer = O3.loc[period]
    summer = summer[summer.index.month.isin([5, 6, 7, 8, 9])]
    o3_mean = summer.mean().where(summer.notna().sum() > 250)
    pm_mu, o3_mu = gp_map(pm_mean, 40), gp_map(o3_mean, 40)
    who = ix.who_guideline("pm25", "24h")
    pm_days, n_pm = expected(PM, 2025, who)
    o3_days, n_o3 = expected(O3, 2025, 70.5, months=[4, 5, 6, 7, 8, 9])
    print(f"exceedance maps: {n_pm} PM2.5 days, {n_o3} ozone days")
    keep = near.ravel()
    grid = pd.DataFrame({"lon": LON.ravel(), "lat": LAT.ravel(), "land": land.ravel().astype(int),
                         "pm25_mean": pm_mu, "o3_mean": o3_mu,
                         "pm25_days": pm_days, "o3_days": o3_days})[keep]
    grid = grid.round({"lon": 2, "lat": 2, "pm25_mean": 2, "o3_mean": 2, "pm25_days": 1, "o3_days": 1})

    y25 = PM.loc["2025"]
    o25 = O3.loc["2025"]
    o25 = o25[o25.index.month.isin([4, 5, 6, 7, 8, 9])]
    ids = pm_mean.dropna().index.union(o3_mean.dropna().index)
    st = sites.loc[ids, ["site", "state", "lat", "lon"]].copy()
    st = st[st.lon.between(METRO[0], METRO[1]) & st.lat.between(METRO[2], METRO[3])]
    st["pm25_mean"] = pm_mean.reindex(st.index).round(2)
    st["o3_mean"] = o3_mean.reindex(st.index).round(2)
    st["pm25_days"] = (y25 > who).sum().where(y25.notna().sum() > 300).reindex(st.index)
    st["o3_days"] = (o25 > 70.5).sum().where(o25.notna().sum() > 150).reindex(st.index)
    st = st.reset_index().rename(columns={"full_aqsid": "site_id", "site": "name"})
    st[["pm25_days", "o3_days"]] = st[["pm25_days", "o3_days"]].astype("Int64")
    return grid, st


def map_layers(layers: dict) -> list[dict]:
    import shapely
    from shapely.geometry import box

    def rings(geom, nd):
        polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for p in polys:
            if p.geom_type != "Polygon" or p.is_empty:
                continue
            yield [[[round(x, nd), round(y, nd)] for x, y in r.coords] for r in [p.exterior, *p.interiors]]

    def lines(geom, nd):
        parts = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))
        for ln in parts:
            if ln.geom_type == "LineString" and not ln.is_empty:
                yield [[round(x, nd), round(y, nd)] for x, y in ln.coords]

    land_all = layers["land"].union_all().difference(layers["lakes"].union_all())
    out = []
    for name, ext, tol, nd in (("metro", METRO, 0.002, 4), ("northeast", NE, 0.01, 3)):
        frame = box(ext[0], ext[2], ext[1], ext[3])
        inner = box(ext[0] + 1e-6, ext[2] + 1e-6, ext[1] - 1e-6, ext[3] - 1e-6)
        land = land_all.intersection(frame.buffer(0.05)).simplify(tol)
        for r in rings(land, nd):
            out.append({"map": name, "layer": "land", "name": "", "coords": r})
        coast = shapely.intersection(land_all.boundary, inner).simplify(tol)
        out.append({"map": name, "layer": "coast", "name": "", "coords": list(lines(coast, nd))})
        borders = shapely.intersection(layers["states"].boundary.union_all(), inner)
        borders = borders.difference(land_all.boundary.buffer(0.01)).simplify(tol)
        out.append({"map": name, "layer": "borders", "name": "", "coords": list(lines(borders, nd))})
    for _, b in layers["nyc_boroughs"].iterrows():
        for r in rings(b.geometry.simplify(0.0008), 4):
            out.append({"map": "metro", "layer": "boroughs", "name": b.borough, "coords": r})
    return out


def smoke() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, sites = data.load_airnow_hourly(data.DATA / "airnow" / "airnow_hourly_smoke_2023.csv.gz")
    hp = raw[raw.param == "PM2.5"].pivot_table(index="time", columns="full_aqsid", values="value")
    hp = hp.loc[:, hp.notna().mean() > 0.6].clip(lower=0)
    long = hp.stack().rename("pm25").reset_index()
    long["time"] = long.time.dt.strftime("%Y-%m-%dT%H:%MZ")
    long = long.rename(columns={"full_aqsid": "site_id"})[["time", "site_id", "pm25"]]
    st = sites.loc[hp.columns, ["site", "state", "lat", "lon"]].reset_index()
    st = st.rename(columns={"full_aqsid": "site_id", "site": "name"})
    st["country"] = np.where(st.site_id.str.startswith("124"), "Canada", "United States")
    st["nyc_area"] = (st.lon.between(-74.28, -73.68) & st.lat.between(40.48, 40.93)).astype(int)
    return long, st


def main() -> int:
    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    r1, r2, r3 = (json.loads((PROCESSED / f"results_0{i}.json").read_text()) for i in (1, 2, 3))
    layers = data.load_basemap()
    tables = {
        "nyc_daily.csv": daily(), "nyc_forecast.csv": forecast(r1["last_day"]),
        "forecast_scores.csv": forecast_scores(), "spatial_benchmark.csv": spatial_benchmark(),
        "covid_effect.csv": covid(r1), "trends.csv": trends(), "virtual_sensors.csv": virtual_sensors(r2),
        "caqi_forecast.csv": caqi_forecast(r2), "key_results.csv": key_results(r1, r2, r3),
        "borough_exposure.csv": boroughs(r3, layers),
    }
    tables["smoke_hourly.csv"], tables["smoke_sites.csv"] = smoke()
    tables["metro_grid.csv"], tables["metro_stations.csv"] = maps(layers)
    for name, t in tables.items():
        t.to_csv(OUT / name, index=False)
        print(f"{name}: {len(t)} rows, {(OUT / name).stat().st_size / 1e3:.0f} kB")
    (OUT / "map_layers.json").write_text(json.dumps(map_layers(layers), separators=(",", ":")))
    print(f"map_layers.json: {(OUT / 'map_layers.json').stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
