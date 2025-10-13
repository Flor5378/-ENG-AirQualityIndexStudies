# Air Quality Index Analysis and Prediction
---
*By Florian GROLLEAU* Sciences Po Saint-Germain-en-Laye
Date: August 2025

This repository contains notebooks, datasets, and reports for calculating and predicting the Air Quality Index (AQI). The project focuses on AQI computation for New York City and predictive modeling using data from an Italian city.

## 📂 Files Overview

### Notebooks
- **`AQIoverNY.ipynb`**  
  Calculates AQI for New York City using historical pollutant data (`NYpollutants.csv`). Includes examples of computing AQI based on PM2.5, PM10, NO2, O3, and CO.

- **`AirQualityIndex.ipynb`**  
  Predicts AQI using an Italian city dataset (`AirQualityUCI.csv`). Based on methods similar to [GeeksforGeeks AQI prediction tutorial](https://www.geeksforgeeks.org/python/predicting-air-quality-index-using-python/).

### Datasets
- **`AirQualityUCI.csv`** – Air quality data for an Italian city, used for AQI prediction modeling.
- **`NYpollutants.csv`** – Historical pollutant data for New York City (2018–2025), used in `AQIoverNY.ipynb`.

### Documents
- **`AirQualityIndexPrediction.pdf`** – Research document presenting methodology, results, and insights from AQI prediction studies.

### Archives
- **`AirQualityIndexPrediction.zip`** – Contains supplementary files related to the AQI prediction project.

## ⚡ Usage

### Compute AQI for New York City
```bash
# Open the notebook and run cells
jupyter notebook AQIoverNY.ipynb
