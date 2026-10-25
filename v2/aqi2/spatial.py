"""Spatial prediction of air quality between monitors.

Four families are compared with leave-one-station-out validation in notebook 03:

* inverse distance weighting (IDW);
* ordinary kriging with a pooled, standardised variogram (nugget treated as
  measurement noise, so maps are smooth rather than "bullseyes");
* Gaussian process regression with per-day maximum-likelihood hyperparameters,
  which also gives predictive uncertainty and exceedance probabilities;
* RFSI, random-forest spatial interpolation (Sekulic et al., 2020), here with
  LightGBM: the learner sees the values and distances of the nearest stations,
  network statistics, geography and season, and learns how to combine them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm

EARTH_R = 6371.0088
NYC_CENTER = (40.7580, -73.9855)  # Times Square


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.deg2rad(np.asarray(a, float)) for a in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def to_xy_km(lat, lon, lat0: float = NYC_CENTER[0], lon0: float = NYC_CENTER[1]) -> np.ndarray:
    """Local equirectangular projection in km (error < 0.5% over the region)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = np.deg2rad(lon - lon0) * EARTH_R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * EARTH_R
    return np.column_stack([x, y])


def pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


# ---------------------------------------------------------------------------
# IDW
# ---------------------------------------------------------------------------
def idw(xy_obs, z, xy_new, power: float = 2.0, max_km: float | None = None):
    d = pairwise(np.asarray(xy_new, float), np.asarray(xy_obs, float))
    w = 1.0 / np.maximum(d, 0.5) ** power
    if max_km is not None:
        w = np.where(d <= max_km, w, 0.0)
    z = np.asarray(z, float)
    return (w @ z) / w.sum(axis=1)


# ---------------------------------------------------------------------------
# Variograms and ordinary kriging
# ---------------------------------------------------------------------------
def exp_variogram(h, nugget, psill, rng):
    return nugget + psill * (1 - np.exp(-3 * h / rng))


def empirical_variogram(xy, Z, bins):
    """Pooled semivariance of standardised daily fields.

    ``Z`` is (days, stations); each day is standardised by its spatial mean
    and standard deviation so that clean and polluted days weigh equally.
    """
    D = pairwise(xy, xy)
    iu = np.triu_indices(len(xy), 1)
    h = D[iu]
    gam_sum = np.zeros(len(bins) - 1)
    cnt = np.zeros(len(bins) - 1)
    which = np.digitize(h, bins) - 1
    for z in np.atleast_2d(Z):
        ok = ~np.isnan(z)
        if ok.sum() < 5 or np.nanstd(z) == 0:
            continue
        zs = (z - np.nanmean(z)) / np.nanstd(z)
        g = 0.5 * (zs[:, None] - zs[None, :]) ** 2
        g = g[iu]
        valid = ~np.isnan(g) & (which >= 0) & (which < len(cnt))
        np.add.at(gam_sum, which[valid], g[valid])
        np.add.at(cnt, which[valid], 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    with np.errstate(invalid="ignore"):
        return pd.DataFrame({"h": centers, "gamma": gam_sum / cnt, "pairs": cnt})


def fit_variogram(emp: pd.DataFrame) -> dict:
    e = emp.dropna()
    e = e[e.pairs > 0]
    p0 = [0.1, max(e.gamma.max() - 0.1, 0.1), e.h.median()]
    popt, _ = curve_fit(exp_variogram, e.h, e.gamma, p0=p0, sigma=1 / np.sqrt(e.pairs),
                        bounds=([0, 1e-3, 1.0], [2.0, 5.0, 2000.0]), maxfev=20000)
    nugget, psill, rng = popt
    return {"nugget": nugget, "psill": psill, "range": rng,
            "nugget_frac": nugget / (nugget + psill)}


class OrdinaryKriging:
    """Ordinary kriging with an exponential covariance whose shape (range and
    nugget fraction) is fixed and whose sill is the variance of the day."""

    def __init__(self, rng_km: float, nugget_frac: float):
        self.rng, self.nf = rng_km, nugget_frac

    def _cov(self, d, sill):
        return (1 - self.nf) * sill * np.exp(-3 * d / self.rng)

    def fit(self, xy, z):
        self.xy, self.z = np.asarray(xy, float), np.asarray(z, float)
        self.sill = max(np.var(self.z), 1e-6)
        n = len(self.z)
        K = self._cov(pairwise(self.xy, self.xy), self.sill) + self.nf * self.sill * np.eye(n)
        A = np.ones((n + 1, n + 1))
        A[:n, :n], A[n, n] = K, 0.0
        self.A = A
        return self

    def predict(self, xy_new, return_std: bool = False):
        k0 = self._cov(pairwise(self.xy, np.asarray(xy_new, float)), self.sill)
        B = np.vstack([k0, np.ones((1, k0.shape[1]))])
        sol = np.linalg.solve(self.A, B)
        w, mu = sol[:-1], sol[-1]
        pred = w.T @ self.z
        if not return_std:
            return pred
        var = (1 - self.nf) * self.sill - np.sum(w * k0, axis=0) - mu
        return pred, np.sqrt(np.maximum(var, 0))


# ---------------------------------------------------------------------------
# Gaussian process regression
# ---------------------------------------------------------------------------
def make_gp(length_km: float = 60.0, noise: float = 0.1, optimize: bool = True):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_km, (5.0, 1000.0), nu=1.5)
              + WhiteKernel(noise, (1e-4, 2.0)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    optimizer="fmin_l_bfgs_b" if optimize else None,
                                    n_restarts_optimizer=0)


def gp_fixed(gp_fitted):
    """Clone of a fitted GP with its hyperparameters frozen (for fast LOO refits)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    return GaussianProcessRegressor(kernel=gp_fitted.kernel_, normalize_y=True, optimizer=None)


def exceedance_probability(mean, std, threshold):
    return 1 - norm.cdf((threshold - np.asarray(mean)) / np.maximum(np.asarray(std), 1e-6))


# ---------------------------------------------------------------------------
# Leave-one-station-out for the geostatistical methods
# ---------------------------------------------------------------------------
def loso_day(xy, z, methods: dict) -> pd.DataFrame:
    """Leave each station out in turn on one day. ``methods`` maps a name to
    ``f(xy_train, z_train, xy_test) -> prediction``."""
    ok = ~np.isnan(z)
    idx = np.flatnonzero(ok)
    rows = []
    for i in idx:
        tr = idx[idx != i]
        row = {"station": i, "obs": z[i]}
        for name, f in methods.items():
            row[name] = float(np.ravel(f(xy[tr], z[tr], xy[i:i + 1]))[0])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RFSI features
# ---------------------------------------------------------------------------
def rfsi_features(Z: np.ndarray, xy_obs: np.ndarray, xy_tgt: np.ndarray, k: int = 6,
                  self_index: np.ndarray | None = None, max_km: float = 250.0):
    """Neighbourhood features for every (day, target) pair.

    Z: (T, n) observed values (NaN = missing) at ``xy_obs``.
    xy_tgt: (m, 2) target locations; ``self_index[j]`` is the column of Z that
    corresponds to target j (excluded from its own neighbourhood) or -1.
    Returns an array of shape (T, m, F) and the feature names.
    """
    T, n = Z.shape
    D = pairwise(xy_tgt, xy_obs)                      # (m, n)
    if self_index is not None:
        rows = np.flatnonzero(self_index >= 0)
        D[rows, self_index[rows]] = np.inf
    D = np.where(D > max_km, np.inf, D)
    order = np.argsort(D, axis=1)                      # (m, n)
    Ds = np.take_along_axis(D, order, axis=1)          # sorted distances
    V = Z[:, order]                                    # (T, m, n)
    valid = ~np.isnan(V) & np.isfinite(Ds)[None]
    rank = np.cumsum(valid, axis=2)
    feats, names = [], []
    for j in range(1, k + 1):
        sel = valid & (rank == j)
        has = sel.any(axis=2)
        pos = sel.argmax(axis=2)
        val = np.take_along_axis(V, pos[..., None], axis=2)[..., 0]
        dist = np.take_along_axis(np.broadcast_to(Ds, V.shape), pos[..., None], axis=2)[..., 0]
        feats += [np.where(has, val, np.nan), np.where(has, dist, np.nan)]
        names += [f"nn{j}_val", f"nn{j}_dist"]
    Vz = np.where(valid, V, 0.0)
    W = np.where(valid, 1.0 / np.maximum(np.broadcast_to(Ds, V.shape), 0.5) ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        idw2 = (W * Vz).sum(2) / W.sum(2)
        W1 = np.where(valid, 1.0 / np.maximum(np.broadcast_to(Ds, V.shape), 0.5), 0.0)
        idw1 = (W1 * Vz).sum(2) / W1.sum(2)
        cnt = valid.sum(2)
        mean = Vz.sum(2) / cnt
        var = (np.where(valid, (V - mean[..., None]) ** 2, 0.0)).sum(2) / cnt
        near = valid & (np.broadcast_to(Ds, V.shape) <= 50)
        n50 = near.sum(2)
        mean50 = np.where(near, V, 0.0).sum(2) / n50
    feats += [idw2, idw1, mean, np.sqrt(var), cnt.astype(float), n50.astype(float), mean50]
    names += ["idw2", "idw1", "net_mean", "net_std", "net_n", "n_50km", "mean_50km"]
    return np.stack(feats, axis=-1), names


# ---------------------------------------------------------------------------
# Grids, masks and static covariates
# ---------------------------------------------------------------------------
def make_grid(extent, res: float):
    """Regular lon/lat grid. ``extent`` = (lon_min, lon_max, lat_min, lat_max)."""
    lon = np.arange(extent[0] + res / 2, extent[1], res)
    lat = np.arange(extent[2] + res / 2, extent[3], res)
    LON, LAT = np.meshgrid(lon, lat)
    return lon, lat, LON, LAT


def land_mask(LON, LAT, layers: dict) -> np.ndarray:
    """True over land (Natural Earth land minus lakes)."""
    import shapely
    land = layers["land"].union_all()
    if "lakes" in layers and len(layers["lakes"]):
        land = land.difference(layers["lakes"].union_all())
    return shapely.contains_xy(land, LON.ravel(), LAT.ravel()).reshape(LON.shape)


def coast_distance_km(lon, lat, layers: dict) -> np.ndarray:
    """Distance to the coastline in km (computed in UTM zone 18N)."""
    import geopandas as gpd
    import shapely
    coast = layers["coastline"].to_crs(32618).union_all()
    pts = gpd.GeoSeries(gpd.points_from_xy(np.ravel(lon), np.ravel(lat)), crs=4326).to_crs(32618)
    return shapely.distance(coast, pts.values).reshape(np.shape(lon)) / 1000.0


def polygon_mean(values: np.ndarray, LON, LAT, polygons, weights=None) -> np.ndarray:
    """Mean of a gridded field inside each polygon (NaN-aware)."""
    import shapely
    out = []
    for geom in polygons:
        inside = shapely.contains_xy(geom, LON.ravel(), LAT.ravel())
        v = values.reshape(-1)[inside]
        out.append(np.nanmean(v) if inside.any() else np.nan)
    return np.array(out)


def ok_loo_and_predict(xy, z, xy_new, rng_km: float, nugget_frac: float):
    """Ordinary kriging leave-one-out predictions at the data points (Dubrule, 1983)
    and predictions at new points, from a single factorisation.

    Returns (loo, pred) where ``loo[i]`` is the prediction at point i from all the
    other points.
    """
    xy, z = np.asarray(xy, float), np.asarray(z, float)
    n = len(z)
    sill = max(np.var(z), 1e-6)
    K = (1 - nugget_frac) * sill * np.exp(-3 * pairwise(xy, xy) / rng_km) + nugget_frac * sill * np.eye(n)
    A = np.ones((n + 1, n + 1))
    A[:n, :n], A[n, n] = K, 0.0
    B = np.linalg.inv(A)
    b = np.append(z, 0.0)
    Bb = B @ b
    loo = z - Bb[:n] / np.diag(B)[:n]
    pred = None
    if xy_new is not None and len(xy_new):
        k0 = (1 - nugget_frac) * sill * np.exp(-3 * pairwise(xy, np.asarray(xy_new, float)) / rng_km)
        rhs = np.vstack([k0, np.ones((1, k0.shape[1]))])
        w = (B @ rhs)[:n]
        pred = w.T @ z
    return loo, pred
