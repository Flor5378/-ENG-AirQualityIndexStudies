import numpy as np
import pandas as pd
import pytest

from aqi2 import indices as ix


@pytest.mark.parametrize("conc, table, expected", [
    (35.9, "pm25_2012", 102),   # EPA Technical Assistance Document example
    (12.0, "pm25_2012", 50),
    (12.1, "pm25_2012", 51),
    (9.0, "pm25", 50),
    (9.1, "pm25", 51),
    (35.4, "pm25", 100),
    (125.4, "pm25", 200),
    (78, "o3_8h", 126),         # 0.078 ppm
    (8.4, "co", 90),
    (100, "no2", 100),
    (75, "so2", 100),
    (154, "pm10", 100),
])
def test_epa_subindex_reference_values(conc, table, expected):
    assert ix.epa_subindex(conc, table) == expected


def test_truncation_happens_before_interpolation():
    # 35.49 is truncated to 35.4 -> AQI 100, not 101
    assert ix.epa_subindex(35.49, "pm25") == 100
    assert ix.epa_subindex(54.9, "o3_8h") == 50


def test_vectorised_and_nan():
    out = ix.epa_subindex(np.array([5.0, np.nan, 600.0]), "pm25")
    assert out[0] == 28 and np.isnan(out[1]) and out[2] == 500


def test_ozone_one_hour_only_above_125():
    assert np.isnan(ix.epa_subindex(100, "o3_1h"))
    assert ix.epa_o3(o3_8h_ppb=[60], o3_1h_ppb=[130])[0] == max(ix.epa_subindex(60, "o3_8h"),
                                                               ix.epa_subindex(130, "o3_1h"))


@pytest.mark.parametrize("table", ["pm25", "pm25_2012", "pm10", "o3_8h", "co", "no2", "so2"])
def test_inverse_round_trip(table):
    for aqi in [0, 25, 50, 51, 75, 100, 101, 150, 151, 200]:
        if table == "o3_8h" and aqi > 300:
            continue
        conc = ix.epa_inverse(aqi, table)
        assert ix.epa_subindex(conc + 1e-6, table) == pytest.approx(aqi, abs=1)


def test_category_boundaries():
    assert list(ix.epa_category([0, 50, 51, 100, 101, 151, 201, 301, np.nan])) == [0, 0, 1, 1, 2, 3, 4, 5, -1]
    assert ix.epa_category_name([42])[0] == "Good"


def test_overall_and_dominant():
    sub = pd.DataFrame({"pm25": [40, np.nan, 80], "o3": [60, np.nan, 20]})
    out = ix.overall_aqi(sub)
    assert list(out.aqi[[0, 2]]) == [60, 80]
    assert out.dominant[0] == "o3" and out.dominant[2] == "pm25"
    assert np.isnan(out.aqi[1]) and out.n_pollutants[1] == 0


def test_eaqi_caqi_aqhi():
    assert ix.eaqi_level(22, "pm25") == 2            # Moderate
    assert ix.eaqi_level(10, "pm25") == 0            # Good (upper bound inclusive)
    assert ix.caqi_subindex(150, "no2") == pytest.approx(62.5)
    assert ix.caqi_subindex(600, "no2") == pytest.approx(125.0)
    assert ix.aqhi(20, 30, 10) == pytest.approx(3.72, abs=0.01)
    assert list(ix.aqhi_level([2, 5, 8, 12])) == [0, 1, 2, 3]


def test_unit_conversion():
    assert ix.ugm3_to_ppb(188.1, "no2") == pytest.approx(100, rel=1e-3)
    assert ix.ppb_to_ugm3(100, "o3") == pytest.approx(196.3, rel=1e-3)
