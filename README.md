# Air Quality Index Analysis and Prediction
---
*By Florian GROLLEAU* Sciences Po Saint-Germain-en-Laye
Date: August 2025 (version 1), October 2026 (version 2)

This repository contains notebooks, datasets, and reports for calculating and predicting the Air Quality Index (AQI). The project focuses on AQI computation for New York City and predictive modeling using data from an Italian city.

## Version 2 (October 2026)

**[`v2/`](v2/README.md)** rebuilds the project on public, verifiable data (US EPA AirNow monitors, NOAA weather, Natural Earth, UCI) up to October 23, 2026:

* **Quality control** of the version 1 data against independent sources: the New York file holds AQI sub-indices (not concentrations), its gases are stamped one day early, and it contains an ozone plateau and a PM2.5 sensor drift that changed the v1 conclusions.
* **A multi-standard AQI engine** (`v2/aqi2`) validated against the official AirNow values: EPA 2012 and 2024 breakpoints, European EAQI, CAQI, Canada's AQHI, WHO 2021 guidelines.
* **Trends without the weather**, attribution of wildfire smoke, fireworks and the COVID-19 lockdown (with placebo tests).
* **Probabilistic forecasting** with conformal prediction intervals and a live forecast.
* **Low-cost sensors** turned into virtual reference analyzers, with their drift measured.
* **Geospatial prediction**: kriging, Gaussian processes and machine-learning interpolators under spatial cross-validation, exceedance-probability maps, an hourly animation of the June 2023 smoke plume and interactive maps.
* **Completed report**: [`v2/report/AQI_v2_report.pdf`](v2/report/AQI_v2_report.pdf).

![New York City daily AQI, 2014 to 2026](v2/figures/01_calendar_aqi.png)

## Version 1 files

### Notebooks
- **`AQIoverNY.ipynb`**  
  Calculates AQI for New York City using historical pollutant data (`NYpollutants.csv`). Includes examples of computing AQI based on PM2.5, PM10, NO2, O3, and CO. Version 2 shows that this file already contains AQI sub-indices (see `v2/notebooks/01_nyc_aqi_v2.ipynb`).

- **`AirQualityIndex.ipynb`**  
  Predicts AQI using an Italian city dataset (`AirQualityUCI.csv`). Based on methods similar to [GeeksforGeeks AQI prediction tutorial](https://www.geeksforgeeks.org/python/predicting-air-quality-index-using-python/). Audited and extended in `v2/notebooks/02_uci_virtual_sensing.ipynb`.

### Datasets
- **`AirQualityUCI.csv`**: air quality data for an Italian city, used for AQI prediction modeling.
- **`NYpollutants.csv`**: historical pollutant data for New York City (2014 to 2025, AQI sub-indices from aqicn.org), used in `AQIoverNY.ipynb`.

### Documents
- **`AirQualityIndexPrediction.pdf`**: research document presenting methodology, results, and insights from AQI prediction studies (version 1).

### Archives
- **`AirQualityIndexPrediction.zip`**: LaTeX source and images of the version 1 report.

## Usage

### Version 1: compute AQI for New York City
```bash
# Open the notebook and run cells
jupyter notebook AQIoverNY.ipynb
```

### Version 2
```bash
pip install -r v2/requirements.txt
cd v2
python -m pytest tests                 # engine and model tests
python scripts/build_notebooks.py      # executes the three v2 notebooks
python scripts/make_report.py          # fills and compiles the v2 report
```
