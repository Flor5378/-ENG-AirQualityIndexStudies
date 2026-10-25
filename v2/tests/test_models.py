import numpy as np
import pandas as pd

from aqi2 import features, forecasting, spatial


def test_direct_dataset_has_no_lookahead():
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    y = pd.Series(np.arange(60.0), index=idx, name="pm")
    X, t = features.make_direct_dataset(y, horizon=3, lags=[0, 1], windows=[7])
    row = X.loc["2020-01-20"]
    # issued at 2020-01-17: the most recent observation used is y(2020-01-17)
    assert row["pm_lag0"] == y["2020-01-17"]
    assert t["2020-01-20"] == y["2020-01-20"]
    assert row["pm_r7_max"] <= y["2020-01-17"]


def test_conformal_interval_coverage():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(size=n)})
    y = pd.Series(2 * X.a + rng.normal(scale=1 + X.b, size=n))
    qb = forecasting.QuantileBooster(alpha=0.2, params={"n_estimators": 200}).fit(X[:2000], y[:2000])
    p = qb.predict(X[2000:])
    cov = forecasting.interval_scores(y[2000:], p.lo, p.hi, 0.2)["coverage"]
    assert 0.74 < cov < 0.88


def test_kriging_and_idw_behave():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, size=(40, 2))
    z = np.sin(xy[:, 0] / 30) + 0.05 * rng.normal(size=40)
    ok = spatial.OrdinaryKriging(rng_km=60, nugget_frac=0.05).fit(xy, z)
    pred, sd = ok.predict(xy[:5] + 0.01, return_std=True)
    assert np.allclose(pred, z[:5], atol=0.15) and np.all(sd >= 0)
    far = ok.predict(np.array([[1000.0, 1000.0]]))
    assert abs(far[0] - z.mean()) < 0.3     # reverts to the (GLS) mean far away
    assert np.allclose(spatial.idw(xy, z, xy[:3]), z[:3], atol=0.1)


def test_rfsi_features_exclude_self():
    Z = np.array([[1.0, 2.0, 3.0]])
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
    F, names = spatial.rfsi_features(Z, xy, xy, k=2, self_index=np.array([0, 1, 2]))
    f = dict(zip(names, F[0, 0]))
    assert f["nn1_val"] == 2.0 and f["nn1_dist"] == 10.0 and f["nn2_val"] == 3.0
    assert f["net_n"] == 2


def test_kriging_closed_form_loo_matches_explicit():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 80, size=(25, 2))
    z = np.cos(xy[:, 1] / 25) + 0.1 * rng.normal(size=25)
    loo, pred = spatial.ok_loo_and_predict(xy, z, xy[:2] + 3.0, rng_km=50, nugget_frac=0.2)
    explicit = []
    for i in range(len(z)):
        keep = np.arange(len(z)) != i
        ok = spatial.OrdinaryKriging(50, 0.2)
        ok.fit(xy[keep], z[keep])
        ok.sill = max(np.var(z), 1e-6)  # same sill as the closed form (weights do not depend on it)
        explicit.append(ok.fit(xy[keep], z[keep]).predict(xy[i:i + 1])[0])
    assert np.allclose(loo, explicit, atol=1e-8)
    direct = spatial.OrdinaryKriging(50, 0.2).fit(xy, z).predict(xy[:2] + 3.0)
    assert np.allclose(pred, direct, atol=1e-8)
