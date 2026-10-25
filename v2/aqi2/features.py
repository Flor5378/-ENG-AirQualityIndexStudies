"""Feature engineering for daily and hourly air quality time series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def us_holidays(years) -> pd.DatetimeIndex:
    """Federal holidays that change traffic and fireworks activity in New York."""
    days = []
    for y in years:
        days += [f"{y}-01-01", f"{y}-07-04", f"{y}-12-25"]
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        days.append(nov[nov.dayofweek == 3][3])          # Thanksgiving
        may = pd.date_range(f"{y}-05-01", f"{y}-05-31")
        days.append(may[may.dayofweek == 0][-1])          # Memorial Day
        sep = pd.date_range(f"{y}-09-01", f"{y}-09-30")
        days.append(sep[sep.dayofweek == 0][0])           # Labor Day
    return pd.DatetimeIndex(pd.to_datetime(days)).normalize()


def calendar_features(index: pd.DatetimeIndex, hourly: bool = False) -> pd.DataFrame:
    """Cyclical calendar encodings plus holiday and fireworks flags."""
    idx = pd.DatetimeIndex(index)
    doy = idx.dayofyear.to_numpy()
    f = pd.DataFrame(index=idx)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["dow"] = idx.dayofweek
    f["weekend"] = (idx.dayofweek >= 5).astype(int)
    f["month"] = idx.month
    f["year_frac"] = idx.year + (doy - 1) / 365.25
    hol = us_holidays(range(idx.year.min() - 1, idx.year.max() + 2))
    day = idx.normalize()
    f["holiday"] = day.isin(hol).astype(int)
    md = idx.strftime("%m-%d")
    # Independence Day fireworks peak on the night of July 4 (daily means: July 4-5);
    # New Year fireworks on the night of Dec 31 - Jan 1.
    f["fireworks"] = np.isin(md, ["07-04", "07-05", "12-31", "01-01"]).astype(int)
    if hourly:
        h = idx.hour.to_numpy()
        f["hour"] = h
        f["hour_sin"] = np.sin(2 * np.pi * h / 24)
        f["hour_cos"] = np.cos(2 * np.pi * h / 24)
        f["how"] = idx.dayofweek * 24 + h   # hour of week
    return f


def lag_features(s: pd.Series, lags, name: str | None = None) -> pd.DataFrame:
    name = name or s.name
    return pd.DataFrame({f"{name}_lag{l}": s.shift(l) for l in lags}, index=s.index)


def rolling_features(s: pd.Series, windows, name: str | None = None, stats=("mean", "std", "max")) -> pd.DataFrame:
    """Rolling statistics ending at (and including) t."""
    name = name or s.name
    out = {}
    for w in windows:
        r = s.rolling(w, min_periods=max(1, w // 2))
        for st in stats:
            out[f"{name}_r{w}_{st}"] = getattr(r, st)()
    out[f"{name}_ewm"] = s.ewm(halflife=max(windows) / 4, min_periods=1).mean()
    return pd.DataFrame(out, index=s.index)


def make_direct_dataset(y: pd.Series, horizon: int, lags, windows,
                        exog_past: pd.DataFrame | None = None,
                        exog_future: pd.DataFrame | None = None,
                        hourly: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Supervised table for a direct ``horizon``-step forecast issued at origin t.

    * features describing the past are computed at t (``y`` lags and rolling
      statistics, ``exog_past`` lags);
    * features known in advance for the target time t+h come from
      ``exog_future`` (calendar and, in a perfect-prognosis setting, weather);
    * the target is ``y(t+h)``, indexed by the *target* time.
    """
    name = y.name or "y"
    past = [lag_features(y, lags, name), rolling_features(y, windows, name)]
    if exog_past is not None:
        past.append(exog_past.add_suffix("_t0"))
    X = pd.concat(past, axis=1)
    X.index = X.index + pd.tseries.frequencies.to_offset(y.index.freq) * horizon
    target_index = X.index
    cal = calendar_features(target_index, hourly=hourly)
    X = X.join(cal)
    if exog_future is not None:
        X = X.join(exog_future.reindex(target_index).add_suffix("_tgt"))
    X["horizon"] = horizon
    target = y.reindex(target_index)
    return X, target.rename("target")
