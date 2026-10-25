"""Plotting conventions shared by every v2 figure.

One visual system for the whole project: a warm off-white surface, hairline
solid grids, 2 px lines, a fixed categorical order for pollutants, the official
EPA colours whenever an AQI *category* is shown, and a single "heat" ramp for
concentrations.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

from .data import FIGURES
from .indices import EPA_CATEGORIES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WATER = "#e4ebf1"
LAND = "#f4f3ef"

# Categorical slots, fixed order (validated palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POLLUTANT_COLORS = {"pm25": SERIES[0], "o3": SERIES[1], "no2": SERIES[2], "co": SERIES[3],
                    "pm10": SERIES[4], "so2": SERIES[5]}
POLLUTANT_LABELS = {"pm25": "PM2.5", "o3": "O₃", "no2": "NO₂", "co": "CO",
                    "pm10": "PM10", "so2": "SO₂", "o3_8h": "O₃ (8 h)"}
# De-emphasis gray for context series
CONTEXT = "#c3c2b7"

# Sequential blue ramp (magnitude / uncertainty)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("aqi2_blues", BLUES)
# Semantic heat ramp for concentrations (analogous yellow -> red -> maroon)
HEAT = ["#fff6d5", "#fde3a1", "#fdbb6c", "#f78f4a", "#e85b35", "#c9302c", "#8c1d2f", "#4f0d26"]
HEAT_CMAP = LinearSegmentedColormap.from_list("aqi2_heat", HEAT)
# Diverging blue <-> red with a neutral gray midpoint
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "aqi2_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a99a", "#e34948", "#8c1d2f"])

AQI_CMAP = ListedColormap([c.color for c in EPA_CATEGORIES], name="epa_aqi")
# Continuous AQI colour ramp anchored on the official category colours (for contour maps)
AQI_CONT_CMAP = LinearSegmentedColormap.from_list(
    "epa_aqi_continuous", [(0, "#00E400"), (50 / 300, "#FFFF00"), (100 / 300, "#FF7E00"), (150 / 300, "#FF0000"),
                           (200 / 300, "#8F3F97"), (1.0, "#7E0023")])
AQI_BOUNDS = [0, 50.5, 100.5, 150.5, 200.5, 300.5, 500]
AQI_NORM = BoundaryNorm(AQI_BOUNDS, AQI_CMAP.N)


def set_theme(font_scale: float = 1.0):
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10 * font_scale,
        "axes.titlesize": 12.5 * font_scale, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 8, "axes.labelsize": 10 * font_scale, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "text.color": INK, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9 * font_scale,
        "figure.titlesize": 14 * font_scale, "figure.titleweight": "bold",
    })


def subtitle(ax, text: str, y: float = 1.0):
    """Secondary line under the axes title, in secondary ink."""
    ax.annotate(text, xy=(0, y), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=INK_2, ha="left", va="bottom")


def title(ax, main: str, sub: str | None = None):
    """Axes title, with an optional subtitle line between the title and the plot."""
    ax.set_title(main, pad=22 if sub else 8)
    ax._aqi2_subtitle = bool(sub)
    if sub:
        subtitle(ax, sub)


def fig_title(fig, main: str, sub: str | None = None, x: float = 0.01):
    """Figure-level title placed just above the highest axes and its titles
    (``bbox_inches='tight'`` keeps text outside the canvas in the saved image)."""
    h = fig.get_figheight()
    for a in fig.axes:
        a.apply_aspect()
    top_ax = max(fig.axes, key=lambda a: a.get_position().y1)
    top = top_ax.get_position().y1
    if getattr(top_ax, "_aqi2_subtitle", False):
        clearance = 0.62
    elif top_ax.get_title():
        clearance = 0.40
    else:
        clearance = 0.12
    y = top + clearance / h
    if sub:
        fig.text(x, y, sub, fontsize=9.5, color=INK_2, ha="left", va="bottom")
        y += 0.27 / h
    fig.text(x, y, main, fontsize=14.5, fontweight="bold", color=INK, ha="left", va="bottom")


def source_note(fig, text: str):
    fig.text(0.01, -0.02, text, fontsize=7.5, color=MUTED, ha="left", va="top")


def aqi_bands(ax, ymax: float | None = None, alpha: float = 0.13, labels: bool = True, x_label=None):
    """Shade the EPA AQI categories behind a time series of AQI values."""
    ymax = ymax or ax.get_ylim()[1]
    for c in EPA_CATEGORIES:
        if c.lo > ymax:
            break
        ax.axhspan(max(c.lo - 0.5, 0), min(c.hi + 0.5, ymax), color=c.color, alpha=alpha, lw=0, zorder=0)
        if labels:
            y = (max(c.lo - 0.5, 0) + min(c.hi + 0.5, ymax)) / 2
            ax.annotate(c.short, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK_2, va="center")
    ax.set_ylim(0, ymax)


def aqi_color(aqi):
    from .indices import epa_category
    cats = np.atleast_1d(epa_category(aqi))
    return [EPA_CATEGORIES[c].color if c >= 0 else "#dddddd" for c in cats]


def calendar_heatmap(ax, s: pd.Series, year: int, cmap=AQI_CMAP, norm=AQI_NORM, gap: float = 0.12,
                     show_months: bool = True, marked: pd.Series | None = None):
    """One-year calendar (weeks x weekdays) of a daily series.

    ``marked`` (boolean, same index) puts a small dot on days to flag, e.g.
    values reconstructed rather than measured.
    """
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    v = s.reindex(days)
    mk = marked.reindex(days).fillna(False).to_numpy() if marked is not None else np.zeros(len(days), bool)
    week = ((days.dayofyear - 1 + pd.Timestamp(f"{year}-01-01").dayofweek) // 7).to_numpy()
    dow = days.dayofweek.to_numpy()
    for w_, d_, val, m_ in zip(week, dow, v.to_numpy(), mk):
        color = "#ebeae4" if np.isnan(val) else cmap(norm(val))
        ax.add_patch(mpl.patches.FancyBboxPatch((w_ + gap / 2, 6 - d_ + gap / 2), 1 - gap, 1 - gap,
                                                boxstyle="round,pad=0,rounding_size=0.18",
                                                fc=color, ec="none"))
        if m_:
            ax.add_patch(mpl.patches.Circle((w_ + 0.5, 6 - d_ + 0.5), 0.13, fc=INK, ec="none", alpha=0.55))
    ax.set_xlim(0, 54)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    if show_months:
        for m in range(1, 13):
            first = pd.Timestamp(year=year, month=m, day=1)
            wk = (first.dayofyear - 1 + pd.Timestamp(f"{year}-01-01").dayofweek) // 7
            ax.text(wk + 0.1, 7.35, first.strftime("%b"), fontsize=7, color=INK_2, ha="left", va="bottom")
    ax.text(-0.6, 3.5, str(year), fontsize=9, color=INK, ha="right", va="center", fontweight="bold")


def aqi_legend(ax, loc="lower center", ncol=6, bbox=(0.5, -0.12), short=True, fontsize=8):
    handles = [mpl.patches.Patch(fc=c.color, ec="none", label=c.short if short else c.name)
               for c in EPA_CATEGORIES]
    return ax.legend(handles=handles, loc=loc, ncol=ncol, bbox_to_anchor=bbox, fontsize=fontsize,
                     handlelength=1.2, handleheight=1.0, columnspacing=1.2)


def draw_basemap(ax, layers: dict, extent=None, counties: bool = True, boroughs: bool = False,
                 land_color: str = LAND, water_color: str = WATER, lw: float = 0.5):
    """Water background, land, lakes, state/county lines from Natural Earth."""
    ax.set_facecolor(water_color)
    layers["land"].plot(ax=ax, color=land_color, lw=0, zorder=1)
    if "lakes" in layers:
        layers["lakes"].plot(ax=ax, color=water_color, lw=0, zorder=1.1)
    if counties and "counties" in layers:
        layers["counties"].boundary.plot(ax=ax, color="#d9d7ce", lw=lw * 0.6, zorder=1.2)
    if "states" in layers:
        layers["states"].boundary.plot(ax=ax, color=AXIS, lw=lw * 1.4, zorder=1.3)
    if boroughs and "nyc_boroughs" in layers:
        layers["nyc_boroughs"].boundary.plot(ax=ax, color=INK_2, lw=lw * 1.2, zorder=6)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(ax.get_ylim()))))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


_LAND_CACHE: dict = {}


def overlay_water(ax, layers: dict, water_color: str = WATER, zorder: float = 4, coast: bool = True):
    """Re-draw water above a raster so the surface is clipped to land with a smooth coastline."""
    import geopandas as gpd
    from shapely.geometry import box
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    key = id(layers["land"])
    if key not in _LAND_CACHE:
        _LAND_CACHE[key] = layers["land"].union_all()
    water = box(x0, y0, x1, y1).difference(_LAND_CACHE[key])
    gpd.GeoSeries([water], crs=4326).plot(ax=ax, color=water_color, lw=0, zorder=zorder)
    if "lakes" in layers:
        layers["lakes"].plot(ax=ax, color=water_color, lw=0, zorder=zorder + 0.01)
    if coast and "coastline" in layers:
        layers["coastline"].plot(ax=ax, color=AXIS, lw=0.5, zorder=zorder + 0.02)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def strip_geo_labels(fig):
    """geopandas labels map axes with the CRS axis names; maps carry no axis labels here."""
    for ax in fig.axes:
        if ax.get_xlabel().startswith("Geodetic"):
            ax.set_xlabel("")
        if ax.get_ylabel().startswith("Geodetic"):
            ax.set_ylabel("")


def save(fig, name: str, folder: Path = FIGURES, close: bool = False):
    strip_geo_labels(fig)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    if close:
        plt.close(fig)
    return path


_WX_NAMES = {"tmax": "max temperature", "tmin": "min temperature", "prcp": "precipitation",
             "awnd": "mean wind speed", "wsf2": "peak gust speed", "gust_u": "gust, east-west",
             "gust_v": "gust, north-south", "snwd": "snow depth", "t": "temperature", "rh": "relative humidity",
             "ah": "absolute humidity", "t_mean": "mean temperature", "ws_mean": "mean wind speed",
             "rh_mean": "relative humidity", "slp_mean": "sea-level pressure"}
_CAL_NAMES = {"doy_sin": "season (sine)", "doy_cos": "season (cosine)", "dow": "weekday",
              "year_frac": "time trend", "holiday": "public holiday", "fireworks": "fireworks day",
              "month": "month", "weekend": "weekend", "horizon": "lead time", "hour": "hour of day",
              "hour_sin": "hour (sine)", "hour_cos": "hour (cosine)", "how": "hour of week"}
_POLL_NAMES = {"pm25": "PM2.5", "o3": "O₃", "no2": "NO₂", "co": "CO", "aqi": "AQI"}


def pretty_feature(name: str, step: str = "day") -> str:
    """Readable label for the engineered feature names of aqi2.features."""
    suffix = ""
    if name.endswith("_tgt"):
        name, suffix = name[:-4], " (target day)"
    elif name.endswith("_t0"):
        name, suffix = name[:-3], " (issue day)"
    if name in _CAL_NAMES:
        return _CAL_NAMES[name] + suffix
    if name in _WX_NAMES:
        return _WX_NAMES[name] + suffix
    parts = name.split("_")
    base = _POLL_NAMES.get(parts[0], parts[0])
    if len(parts) == 2 and parts[1].startswith("lag"):
        k = int(parts[1][3:])
        now = "issue day" if step == "day" else "issue time"
        return f"{base}, {now}" if k == 0 else f"{base}, {k} {step}{'s' if k > 1 else ''} earlier"
    if len(parts) == 3 and parts[1].startswith("r"):
        return f"{base}, {parts[1][1:]}-{step} rolling {parts[2]}"
    if len(parts) == 2 and parts[1] == "ewm":
        return f"{base}, exponential average"
    return name.replace("_", " ") + suffix
