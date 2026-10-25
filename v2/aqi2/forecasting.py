"""Probabilistic forecasting: gradient boosting, conformal calibration, baselines, scores."""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

LGB_BASE = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                verbose=-1, n_jobs=4)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def regression_scores(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[m], yhat[m]
    err = yhat - y
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y - y.mean()) ** 2)
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err ** 2)),
            "bias": np.mean(err), "R2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "n": int(m.sum())}


def pinball(y, q, tau: float) -> float:
    y, q = np.asarray(y, float), np.asarray(q, float)
    d = y - q
    return float(np.nanmean(np.maximum(tau * d, (tau - 1) * d)))


def interval_scores(y, lo, hi, alpha: float) -> dict:
    """Coverage, mean width and Winkler (interval) score of a (1 - alpha) interval."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    m = ~(np.isnan(y) | np.isnan(lo) | np.isnan(hi))
    y, lo, hi = y[m], lo[m], hi[m]
    width = hi - lo
    penalty = (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return {"coverage": np.mean((y >= lo) & (y <= hi)), "width": np.mean(width),
            "winkler": np.mean(width + penalty)}


def exceedance_scores(y, yhat, threshold: float) -> dict:
    """Probability of detection, false alarm ratio and critical success index."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    obs, fc = y[m] > threshold, yhat[m] > threshold
    hits, misses, fa = np.sum(obs & fc), np.sum(obs & ~fc), np.sum(~obs & fc)
    return {"events": int(obs.sum()),
            "POD": hits / (hits + misses) if hits + misses else np.nan,
            "FAR": fa / (hits + fa) if hits + fa else np.nan,
            "CSI": hits / (hits + misses + fa) if hits + misses + fa else np.nan}


def skill(score_model: float, score_ref: float) -> float:
    """Skill score, 1 = perfect, 0 = as good as the reference."""
    return 1 - score_model / score_ref


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def persistence(y: pd.Series, horizon: int) -> pd.Series:
    return y.shift(horizon)


def seasonal_naive(y: pd.Series, horizon: int, season: int) -> pd.Series:
    k = int(np.ceil(horizon / season))
    return y.shift(k * season)


def climatology(y: pd.Series, train_end, key: str = "doy", window: int = 15) -> pd.Series:
    """Mean of the training years for the same day of year (+/- window) or hour of week."""
    train = y.loc[:train_end].dropna()
    if key == "doy":
        doy = train.index.dayofyear
        clim = pd.Series({d: train[np.minimum(np.abs(doy - d), 366 - np.abs(doy - d)) <= window].mean()
                          for d in range(1, 367)})
        return pd.Series(clim.reindex(y.index.dayofyear).to_numpy(), index=y.index)
    how = train.index.dayofweek * 24 + train.index.hour
    clim = train.groupby(how).mean()
    tgt = y.index.dayofweek * 24 + y.index.hour
    return pd.Series(clim.reindex(tgt).to_numpy(), index=y.index)


# ---------------------------------------------------------------------------
# Quantile gradient boosting with conformal calibration (CQR)
# ---------------------------------------------------------------------------
@dataclass
class QuantileBooster:
    """LightGBM median + lower/upper quantile models, calibrated with CQR.

    Conformalized quantile regression (Romano, Patterson & Candes, 2019): the
    raw quantile band is widened (or narrowed) by the (1 - alpha) empirical
    quantile of the conformity scores on a calibration block that follows the
    training block in time, which restores the nominal coverage.
    """
    alpha: float = 0.2
    params: dict = field(default_factory=dict)
    calib_frac: float = 0.2
    log_target: bool = False
    models: dict = field(default_factory=dict, init=False)
    q_hat: float = field(default=0.0, init=False)
    feature_names: list = field(default_factory=list, init=False)

    def _fwd(self, y):
        return np.log1p(np.clip(y, 0, None)) if self.log_target else y

    def _inv(self, y):
        return np.expm1(y) if self.log_target else y

    def _fit_models(self, X, y):
        p = {**LGB_BASE, **self.params}
        taus = {"lo": self.alpha / 2, "mid": 0.5, "hi": 1 - self.alpha / 2}
        models = {}
        for k, tau in taus.items():
            m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **p)
            m.fit(X, self._fwd(y))
            models[k] = m
        mean_model = lgb.LGBMRegressor(objective="regression_l1" if self.log_target else "regression", **p)
        mean_model.fit(X, self._fwd(y))
        models["mean"] = mean_model
        return models

    def fit(self, X: pd.DataFrame, y: pd.Series):
        m = y.notna()
        X, y = X[m], y[m].to_numpy(float)
        self.feature_names = list(X.columns)
        n_cal = int(len(y) * self.calib_frac)
        if n_cal >= 30:
            models = self._fit_models(X.iloc[:-n_cal], y[:-n_cal])
            Xc, yc = X.iloc[-n_cal:], y[-n_cal:]
            lo = self._inv(models["lo"].predict(Xc))
            hi = self._inv(models["hi"].predict(Xc))
            scores = np.maximum(lo - yc, yc - hi)
            level = min(1.0, np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal)
            self.q_hat = float(np.quantile(scores, level))
        self.models = self._fit_models(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_names]
        out = pd.DataFrame(index=X.index)
        out["mean"] = self._inv(self.models["mean"].predict(X))
        out["median"] = self._inv(self.models["mid"].predict(X))
        lo = self._inv(self.models["lo"].predict(X))
        hi = self._inv(self.models["hi"].predict(X))
        out["lo_raw"], out["hi_raw"] = lo, hi
        out["lo"] = np.minimum(lo - self.q_hat, out["median"])
        out["hi"] = np.maximum(hi + self.q_hat, out["median"])
        return out

    def shap_values(self, X: pd.DataFrame, which: str = "mean") -> pd.DataFrame:
        """Exact TreeSHAP contributions computed by LightGBM (last column = expected value)."""
        X = X[self.feature_names]
        contrib = self.models[which].booster_.predict(X, pred_contrib=True)
        return pd.DataFrame(contrib, index=X.index, columns=self.feature_names + ["expected_value"])


def walk_forward(X: pd.DataFrame, y: pd.Series, test_periods, make_model, min_train: int = 365):
    """Expanding-window evaluation: for each (start, end) test period, train on all
    rows strictly before ``start`` and predict the rows of the period."""
    preds = []
    for start, end in test_periods:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        tr = (X.index < start) & y.notna().to_numpy()
        te = (X.index >= start) & (X.index <= end)
        if tr.sum() < min_train or te.sum() == 0:
            continue
        model = make_model().fit(X[tr], y[tr])
        p = model.predict(X[te])
        if not isinstance(p, pd.DataFrame):
            p = pd.DataFrame({"mean": p}, index=X.index[te])
        p["y"] = y[te].to_numpy()
        p["fold"] = f"{start:%Y-%m-%d}"
        preds.append(p)
    return pd.concat(preds)
