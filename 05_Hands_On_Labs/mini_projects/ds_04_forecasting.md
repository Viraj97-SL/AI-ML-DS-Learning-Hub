# Sales Forecaster with Prophet

> **Difficulty:** Intermediate | **Time:** 1-2 days | **Track:** Data Science

## What You'll Build
A time-series sales forecasting tool using Facebook Prophet. You'll load retail sales data, handle seasonality and holidays, forecast 90 days ahead, and evaluate model accuracy with cross-validation.

## Learning Objectives
- Understand time-series data structures
- Configure Prophet for trend, seasonality, and holidays
- Evaluate forecasts with MAPE, RMSE, and coverage
- Visualize uncertainty intervals
- Cross-validate time-series models correctly

## Tech Stack
- `prophet`: Meta's forecasting library
- `pandas` / `numpy`: data prep
- `matplotlib` / `plotly`: visualization
- `scikit-learn`: error metrics

## Step-by-Step Guide

### Step 1: Load and Prepare Data
```python
import pandas as pd
import numpy as np

# Generate synthetic retail sales data (or load from Kaggle: Rossmann Sales)
np.random.seed(42)
dates = pd.date_range('2021-01-01', '2024-01-01', freq='D')
trend = np.linspace(1000, 1500, len(dates))
weekly = 200 * np.sin(2 * np.pi * pd.Series(dates).dt.dayofweek / 7)
yearly = 300 * np.sin(2 * np.pi * pd.Series(dates).dt.dayofyear / 365)
noise = np.random.normal(0, 50, len(dates))
sales = trend + weekly.values + yearly.values + noise

df = pd.DataFrame({'ds': dates, 'y': sales.clip(min=0)})
print(f'Dataset: {len(df)} days')
print(df.head())
```

### Step 2: Train Prophet Model
```python
from prophet import Prophet

# Configure model with multiple seasonalities
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',  # good for growing trends
    changepoint_prior_scale=0.05,       # flexibility of trend changes
    interval_width=0.95,                # 95% confidence intervals
)

# Add country holidays
model.add_country_holidays(country_name='US')

# Fit
model.fit(df)
print('Model training complete!')
```

### Step 3: Forecast and Visualize
```python
# Forecast 90 days ahead
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

import matplotlib.pyplot as plt

# Plot forecast with confidence intervals
fig1 = model.plot(forecast)
fig1.suptitle('Sales Forecast — 90 Days', y=1.02)
plt.show()

# Plot components (trend, weekly, yearly)
fig2 = model.plot_components(forecast)
plt.show()

# Extract key metrics
print(f"Forecast for next 30 days: {forecast.tail(90)['yhat'].head(30).mean():.0f} avg daily sales")
```

### Step 4: Cross-Validation and Evaluation
```python
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# Time-series cross-validation
df_cv = cross_validation(
    model,
    initial='365 days',    # training set for first fold
    period='90 days',      # how often to retrain
    horizon='90 days'      # forecast horizon
)

metrics = performance_metrics(df_cv)
print("\nForecast Performance Metrics:")
print(metrics[['horizon', 'mape', 'rmse', 'coverage']].to_string(index=False))

# Plot MAPE by horizon
fig = plot_cross_validation_metric(df_cv, metric='mape')
plt.title('MAPE by Forecast Horizon')
plt.show()
```

### Step 5: Add Regressors and Export
```python
# Add external regressor (e.g., promotional days)
df['promotion'] = (df['ds'].dt.dayofweek.isin([5, 6])).astype(int)  # weekend promotions

model2 = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model2.add_regressor('promotion')
model2.fit(df)

# Save forecast to CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('sales_forecast.csv', index=False)
print("Forecast saved to sales_forecast.csv")
```

## Expected Output
- Forecast chart with confidence intervals for 90 days
- Component decomposition (trend + seasonality)
- Cross-validation performance table (MAPE < 15% is good)
- Exported forecast CSV

## Stretch Goals
- [ ] Compare Prophet to ARIMA, ETS, and LSTM — which performs best on your data?
- [ ] Add a Monte Carlo simulation to estimate the range of total quarterly revenue
- [ ] Build a multi-store forecast (one model per store using a for loop) and compare accuracy by store size

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
