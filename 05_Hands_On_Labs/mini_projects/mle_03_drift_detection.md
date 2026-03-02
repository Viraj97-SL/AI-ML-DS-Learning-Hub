# Data Drift Detector with Evidently

> **Difficulty:** Intermediate | **Time:** 1-2 days | **Track:** ML Engineer

## What You'll Build
A data drift monitoring system using Evidently AI. Monitor feature distributions between a reference dataset and incoming production data, generate HTML drift reports, and trigger alerts when significant drift is detected.

## Learning Objectives
- Detect statistical drift in numerical and categorical features
- Generate automated drift reports with Evidently
- Implement alerting logic (email/Slack) on drift threshold
- Understand different drift tests (KS, chi-square, Jensen-Shannon)

## Tech Stack
- `evidently`: drift detection and reporting
- `pandas` / `numpy`: data manipulation
- `scipy`: statistical tests
- `smtplib` / `requests`: alerting

## Step-by-Step Guide

### Step 1: Create Reference and Production Datasets
```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# Reference dataset (training data distribution)
reference = pd.DataFrame({
    'age': np.random.normal(35, 10, n).clip(18, 70),
    'income': np.random.exponential(50000, n),
    'credit_score': np.random.normal(680, 80, n).clip(300, 850),
    'employment': np.random.choice(['employed', 'self-employed', 'unemployed'], n, p=[0.7, 0.2, 0.1]),
    'num_products': np.random.randint(1, 5, n),
    'churn': np.random.choice([0, 1], n, p=[0.8, 0.2])
})

# Production dataset (with drift introduced)
production = pd.DataFrame({
    'age': np.random.normal(42, 12, n).clip(18, 70),       # mean shifted +7
    'income': np.random.exponential(40000, n),               # lower income
    'credit_score': np.random.normal(650, 100, n).clip(300, 850),  # lower scores
    'employment': np.random.choice(['employed', 'self-employed', 'unemployed'], n, p=[0.55, 0.25, 0.2]),  # more unemployed
    'num_products': np.random.randint(1, 4, n),
    'churn': np.random.choice([0, 1], n, p=[0.65, 0.35])    # higher churn
})

print(f"Reference: {reference.shape} | Production: {production.shape}")
print("\nAge distribution shift:")
print(f"  Reference mean: {reference['age'].mean():.1f}")
print(f"  Production mean: {production['age'].mean():.1f}")
```

### Step 2: Generate Drift Report with Evidently
```python
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=reference, current_data=production)
    report.save_html('drift_report.html')
    print("Drift report saved to drift_report.html — open in browser!")

    # Get programmatic results
    results = report.as_dict()
    drift_detected = results['metrics'][0]['result']['dataset_drift']
    drifted_columns = results['metrics'][0]['result']['number_of_drifted_columns']
    print(f"\nDataset drift detected: {drift_detected}")
    print(f"Drifted columns: {drifted_columns}")
except ImportError:
    print("pip install evidently")
```

### Step 3: Manual Drift Tests (without Evidently)
```python
from scipy import stats

def detect_column_drift(reference: pd.Series, production: pd.Series, alpha: float = 0.05) -> dict:
    """Detect drift using KS test for numeric, chi-square for categorical."""
    if reference.dtype in ['float64', 'int64']:
        stat, p_value = stats.ks_2samp(reference.dropna(), production.dropna())
        test_name = 'KS Test'
    else:
        # Chi-square test for categorical
        ref_counts = reference.value_counts()
        prod_counts = production.value_counts()
        all_cats = set(ref_counts.index) | set(prod_counts.index)
        ref_counts = ref_counts.reindex(all_cats, fill_value=1)
        prod_counts = prod_counts.reindex(all_cats, fill_value=1)
        stat, p_value = stats.chi2_contingency(pd.DataFrame([ref_counts, prod_counts]))[0:2]
        test_name = 'Chi-Square'

    return {'test': test_name, 'statistic': round(stat, 4), 'p_value': round(p_value, 4), 'drift': p_value < alpha}

print("Column-by-column drift report:")
for col in ['age', 'income', 'credit_score', 'employment', 'num_products']:
    result = detect_column_drift(reference[col], production[col])
    status = "⚠ DRIFT" if result['drift'] else "✓ OK"
    print(f"  {col:15s}: {result['test']:12s} stat={result['statistic']:.4f}, p={result['p_value']:.4f} | {status}")
```

### Step 4: Alerting System
```python
import json
from datetime import datetime

def send_drift_alert(report: dict, channel: str = 'console') -> None:
    """Send drift alert via console, email, or Slack."""
    drifted = [col for col, info in report.items() if info['drift']]

    if not drifted:
        print("✓ No drift detected — monitoring continues.")
        return

    message = f"""
🚨 DATA DRIFT ALERT — {datetime.now().strftime('%Y-%m-%d %H:%M')}

Drifted features ({len(drifted)}/{len(report)} total):
{chr(10).join(f'  • {col}: p={report[col]["p_value"]:.4f}' for col in drifted)}

Action required: Investigate production data quality or retrain model.
"""
    if channel == 'console':
        print(message)
    elif channel == 'slack':
        # Uncomment with real webhook URL
        # import requests
        # requests.post(SLACK_WEBHOOK, json={'text': message})
        print(f"[Would send Slack alert]: {message[:100]}...")

# Build report and alert
drift_report = {col: detect_column_drift(reference[col], production[col])
                for col in ['age', 'income', 'credit_score', 'employment', 'num_products']}
send_drift_alert(drift_report)
```

### Step 5: Schedule Monitoring (Cron)
```python
# monitoring_job.py — run this daily via cron or Airflow
import schedule
import time

def monitoring_job():
    print(f"Running drift check at {datetime.now()}")
    # Load latest production data
    # production = load_from_database()
    # Run drift detection
    # send_drift_alert(...)
    print("Monitoring job complete.")

# Run every day at 9am
schedule.every().day.at("09:00").do(monitoring_job)

# while True:
#     schedule.run_pending()
#     time.sleep(60)

print("Scheduler configured.")
```

## Expected Output
- HTML drift report with visualizations per feature
- Console alert listing drifted features with p-values
- Scheduled monitoring job ready to deploy

## Stretch Goals
- [ ] Add a Population Stability Index (PSI) metric — PSI > 0.2 indicates significant drift
- [ ] Track drift metrics over time with MLflow and plot drift trend charts
- [ ] Integrate with Grafana: export drift scores as Prometheus metrics and create a dashboard

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
