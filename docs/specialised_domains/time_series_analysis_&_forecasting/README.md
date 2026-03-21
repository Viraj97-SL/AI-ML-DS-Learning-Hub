# Time Series Analysis & Forecasting

> A comprehensive guide to time series analysis — from classical ARIMA models through modern deep learning forecasters (TFT, PatchTST), with practical Python code for every technique.

---

## Overview

Time series data is everywhere: stock prices, sensor readings, website traffic, electricity demand, sales revenue, and heart rate monitors all produce sequences of values ordered in time. Unlike cross-sectional data where rows are independent, time series data has temporal dependencies — the value at time *t* is influenced by values at *t-1*, *t-2*, etc.

Time series analysis and forecasting is a core skill for data scientists in finance, supply chain, IoT, healthcare, energy, and e-commerce. Accurate forecasts directly drive business decisions: how much inventory to order, how many servers to provision, when to run promotions.

The field has evolved dramatically. Classical approaches (ARIMA, exponential smoothing) remain competitive for well-behaved univariate series. Neural approaches (LSTMs, N-BEATS, TFT, PatchTST) dominate on complex, multivariate, or large-scale forecasting tasks. Modern practitioners use both.

---

## Key Concepts

### Time Series Components

Every time series can be decomposed into:
- **Trend**: Long-term direction (upward, downward, flat)
- **Seasonality**: Regular repeating patterns (daily, weekly, yearly)
- **Cyclical**: Irregular longer-term fluctuations (business cycles)
- **Noise/Residual**: Random, unpredictable variation

**Additive**: `Y = Trend + Seasonality + Residual` (when seasonal amplitude is constant)
**Multiplicative**: `Y = Trend × Seasonality × Residual` (when seasonal amplitude grows with trend)

### Stationarity

A stationary time series has constant mean, variance, and autocorrelation over time. Most classical models require stationarity.

**Tests for stationarity**:
- **ADF (Augmented Dickey-Fuller)**: H₀ = unit root (non-stationary). Low p-value → stationary
- **KPSS**: H₀ = stationary. High p-value → stationary

**Making a series stationary**:
- **Differencing**: `Δy_t = y_t - y_{t-1}` removes trend
- **Log transformation**: Stabilizes variance
- **Seasonal differencing**: `y_t - y_{t-s}` removes seasonality

### Autocorrelation

- **ACF (Autocorrelation Function)**: Correlation between the series and its lags
- **PACF (Partial Autocorrelation)**: Correlation with lag *k* after removing effects of shorter lags
- **ACF/PACF plots** guide ARIMA order selection: p from PACF cutoff, q from ACF cutoff

---

## Learning Path

### Beginner
1. Understand time series components and decomposition (`statsmodels.seasonal_decompose`)
2. Plot ACF/PACF and interpret them
3. Test for stationarity (ADF test), apply differencing
4. Fit a simple ARIMA model in statsmodels

### Intermediate
5. SARIMA for seasonal data; model selection with AIC/BIC
6. Exponential Smoothing (Holt-Winters) for trend + seasonality
7. Prophet for business time series with holidays
8. Feature engineering for ML-based forecasting (lag features, rolling statistics)
9. LightGBM/XGBoost with time series features

### Advanced
10. Multivariate forecasting with VAR models
11. Neural forecasting: N-BEATS, TFT (Temporal Fusion Transformer)
12. PatchTST and other transformer-based forecasters
13. Hierarchical forecasting (forecasting at multiple aggregation levels simultaneously)
14. Conformal prediction for forecast uncertainty quantification

---

## Code Examples

### Decomposition and Stationarity

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# Load monthly airline passengers (classic benchmark dataset)
from statsmodels.datasets import get_rdataset
df = get_rdataset("AirPassengers").data
df.index = pd.date_range("1949-01", periods=len(df), freq="MS")
series = df["value"]

# Decomposition
result = seasonal_decompose(series, model="multiplicative", period=12)
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
for ax, (name, component) in zip(axes, [
    ("Observed", result.observed),
    ("Trend", result.trend),
    ("Seasonal", result.seasonal),
    ("Residual", result.resid),
]):
    ax.plot(component)
    ax.set_title(name)
plt.tight_layout()
plt.show()

# ADF test on log-differenced series (airline passengers needs log + diff)
log_diff = np.log(series).diff().dropna()
adf_stat, p_value, _, _, critical_values, _ = adfuller(log_diff)
print(f"ADF statistic: {adf_stat:.4f}")
print(f"p-value: {p_value:.4f}  {'→ Stationary ✓' if p_value < 0.05 else '→ Non-stationary ✗'}")
```

### ARIMA / SARIMA Modeling

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# Train/test split (last 24 months as test)
train = series[:-24]
test = series[-24:]

# SARIMA(1,1,1)(1,1,1,12) — auto identified from ACF/PACF
model = SARIMAX(
    np.log(train),
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fitted = model.fit(disp=False)
print(fitted.summary())

# Forecast
forecast_log = fitted.forecast(steps=24)
forecast = np.exp(forecast_log)  # undo log transform

mape = mean_absolute_percentage_error(test, forecast)
print(f"MAPE: {mape:.2%}")

# Plot
plt.figure(figsize=(12, 4))
plt.plot(train, label="Train")
plt.plot(test, label="Test", color="orange")
plt.plot(test.index, forecast, label=f"Forecast (MAPE={mape:.1%})", color="red", linestyle="--")
plt.legend()
plt.title("SARIMA Forecast — Airline Passengers")
plt.show()
```

### Prophet — Business Time Series

```python
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd

# Prophet requires columns 'ds' (datetime) and 'y' (value)
df_prophet = pd.DataFrame({
    "ds": series.index,
    "y": series.values,
})

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,    # monthly data — no weekly pattern
    daily_seasonality=False,
    changepoint_prior_scale=0.1, # flexibility of trend; higher = more flexible
    seasonality_prior_scale=10,
    interval_width=0.95,         # 95% uncertainty interval
)

# Add custom seasonality
model.add_seasonality(name="quarterly", period=365.25/4, fourier_order=5)

model.fit(df_prophet)

# Make future dataframe and forecast
future = model.make_future_dataframe(periods=24, freq="MS")
forecast = model.predict(future)

# Results
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(24))
fig = plot_plotly(model, forecast)
fig.show()
```

### LightGBM with Lag Features

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

def create_lag_features(series: pd.Series, lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
    """Create lag and rolling statistical features for ML-based forecasting."""
    df = pd.DataFrame({"y": series})
    df.index = series.index

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Rolling statistics
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["y"].shift(1).rolling(window).std()

    # Calendar features
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year

    return df.dropna()

# Build features
feature_df = create_lag_features(series, lags=[1, 2, 3, 6, 12], rolling_windows=[3, 6, 12])
X = feature_df.drop("y", axis=1)
y = feature_df["y"]

# Time-based split (no shuffling!)
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(20, verbose=False)])

preds = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, preds):.2%}")
```

---

## Evaluation Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **MAE** | mean(|y - ŷ|) | Easy to interpret; robust to outliers |
| **RMSE** | sqrt(mean((y - ŷ)²)) | Penalizes large errors more; common benchmark |
| **MAPE** | mean(|y - ŷ| / |y|) × 100 | Scale-independent; fails when y ≈ 0 |
| **sMAPE** | mean(2|y - ŷ| / (|y| + |ŷ|)) | Symmetric MAPE; handles near-zero better |
| **WAPE** | sum(|y - ŷ|) / sum(|y|) | Good for aggregated business metrics |
| **CRPS** | Proper scoring rule for distributions | Probabilistic forecasts |

### Backtesting (Cross-Validation for Time Series)

Never shuffle data for time series cross-validation. Use **walk-forward** or **expanding window** CV:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=0, test_size=12)  # 12-month test windows
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    # ... fit and evaluate
```

---

## Tools & Libraries

| Library | Purpose | Notes |
|---------|---------|-------|
| **statsmodels** | ARIMA, SARIMA, VAR, decomposition | Classical statistical models |
| **Prophet** | Business time series forecasting | Handles holidays, missing data |
| **NeuralProphet** | Neural Prophet — Prophet + AR-Net | Bridging classical and neural |
| **sktime** | Unified TS ML interface | scikit-learn API for time series |
| **darts** | Neural forecasting (TFT, N-BEATS, etc.) | PyTorch-based, comprehensive |
| **Nixtla / NeuralForecast** | State-of-art neural forecasters | N-BEATS, NHITS, PatchTST |
| **Nixtla / StatsForecast** | Fast classical models | 100x faster ARIMA than statsmodels |
| **GluonTS** | Probabilistic forecasting | Amazon Research library |
| **tsfresh** | Automated time series feature extraction | 700+ features automatically |
| **tslearn** | TS clustering and classification | DTW distance, k-Shape |

---

## Resources

### Courses & Tutorials
- [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) — Hyndman & Athanasopoulos — Free online; the definitive time series textbook
- [fast.ai Practical Deep Learning — Time Series](https://www.fast.ai/) — Modern DL approach to TS
- [Nixtla Blog](https://nixtlaverse.nixtla.io/) — State-of-art forecasting tutorials

### Books
- *Forecasting: Principles and Practice* — Hyndman & Athanasopoulos — Free at otexts.com/fpp3
- *Time Series Analysis* — Hamilton — Graduate-level statistical treatment
- *Deep Learning for Time Series Cookbook* — Peixeiro (Packt, 2023) — Practical neural approaches

### Key Papers
- [N-BEATS: Neural Basis Expansion Analysis](https://arxiv.org/abs/1905.10437) — Oreshkin et al. 2019
- [Temporal Fusion Transformers for Interpretable Multi-horizon Forecasting](https://arxiv.org/abs/1912.09363) — Lim et al. 2019
- [A Time Series is Worth 64 Words: PatchTST](https://arxiv.org/abs/2211.14730) — Nie et al. 2022
- [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) — Zeng et al. 2022 — Important skeptical analysis

---

## Projects & Exercises

**Project 1 — Retail Sales Forecasting**
Use the M5 Competition dataset (Walmart sales at store/item level). Forecast 28-day sales for 10 products. Build three models: SARIMA, Prophet, LightGBM with lag features. Evaluate with WAPE. Document which model wins and why. Use proper walk-forward backtesting.

**Project 2 — Energy Demand Prediction**
Forecast hourly electricity demand for a region (use PJM or European energy data from Kaggle). Include temperature as an exogenous variable. Compare: SARIMA-X (exogenous), LightGBM, and a simple TFT with darts. Plot prediction intervals.

**Project 3 — Anomaly Detection in Time Series**
Detect anomalies in a server metrics dataset (CPU usage, request latency). Implement 3 approaches: (1) z-score on rolling window, (2) Isolation Forest on lag features, (3) ARIMA residual monitoring. Evaluate false positive rate vs. detection latency for each.

---

## Related Topics
- [Time Series Notebook →](../../01_Data_Scientist/advanced/03_time_series.ipynb) — Hands-on time series with code
- [Statistics Foundations →](../../04_Foundations/statistics/05_experimental_design.ipynb)
- [Feature Engineering →](../../01_Data_Scientist/intermediate/03_feature_engineering.ipynb)
- [Computer Vision →](../computer_vision/README.md) — Video sequence modeling connects to time series
