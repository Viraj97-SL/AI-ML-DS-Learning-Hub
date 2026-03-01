# ML Engineer — Intermediate Phase

**Goal:** Build production ML pipelines — experiment tracking, versioning, automated retraining, model serving at scale, and monitoring.

**Duration:** 3–4 months at 10–15 hrs/week
**Prerequisites:** MLE Beginner Phase complete

---

## Curriculum Overview

```
Week 1–2   → Experiment Tracking with MLflow
Week 3–4   → Data & Model Versioning with DVC
Week 5–6   → Pipeline Orchestration (Airflow / Prefect)
Week 7–8   → Production Model Serving (FastAPI + Docker + Cloud)
Week 9–10  → Model Monitoring & Drift Detection
Week 11–12 → Feature Stores (Feast)
Week 13–14 → CI/CD for ML (GitHub Actions)
```

---

## Week 1–2: Experiment Tracking with MLflow

MLflow solves the "which model was that?" problem. It tracks every experiment with all parameters, metrics, and artifacts.

```python
# Install: pip install mlflow scikit-learn xgboost
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import xgboost as xgb

# Set tracking URI (local, remote server, or cloud)
mlflow.set_tracking_uri("mlruns")  # Local: stores in ./mlruns/

# Create or set experiment
mlflow.set_experiment("credit_risk_model_v2")

X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ============================================================
# Run 1: Random Forest baseline
# ============================================================
with mlflow.start_run(run_name="RandomForest_baseline"):
    # Log parameters
    params = {
        "model_type": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    mlflow.log_params(params)

    # Train
    model = RandomForestClassifier(**{k: v for k, v in params.items() if k != "model_type"})
    model.fit(X_train, y_train)

    # Evaluate and log metrics
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "test_auc": roc_auc_score(y_test, y_proba),
        "test_f1": f1_score(y_test, y_pred),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "cv_auc_mean": cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean(),
    }
    mlflow.log_metrics(metrics)
    print(f"RF AUC: {metrics['test_auc']:.4f}")

    # Log model with signature (input/output schema)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train[:5]
    )

    # Log custom artifacts
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={metrics['test_auc']:.4f}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png", dpi=100)
    mlflow.log_artifact("roc_curve.png")
    plt.close()

    run_id_rf = mlflow.active_run().info.run_id

# ============================================================
# Run 2: XGBoost
# ============================================================
with mlflow.start_run(run_name="XGBoost_tuned"):
    params_xgb = {
        "model_type": "XGBoost",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
    }
    mlflow.log_params(params_xgb)

    model_xgb = xgb.XGBClassifier(
        **{k: v for k, v in params_xgb.items() if k != "model_type"},
        random_state=42, n_jobs=-1, eval_metric="auc"
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                   early_stopping_rounds=20, verbose=False)

    y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
    y_pred_xgb = model_xgb.predict(X_test)

    metrics_xgb = {
        "test_auc": roc_auc_score(y_test, y_proba_xgb),
        "test_f1": f1_score(y_test, y_pred_xgb),
        "test_accuracy": accuracy_score(y_test, y_pred_xgb),
        "best_iteration": model_xgb.best_iteration,
    }
    mlflow.log_metrics(metrics_xgb)
    mlflow.xgboost.log_model(model_xgb, "model")
    print(f"XGB AUC: {metrics_xgb['test_auc']:.4f}")

# ============================================================
# Load best model from registry
# ============================================================
# Start MLflow UI: mlflow ui --port 5000 → http://localhost:5000

# Load by run_id
best_model = mlflow.sklearn.load_model(f"runs:/{run_id_rf}/model")
predictions = best_model.predict(X_test)

# Register model for production
result = mlflow.register_model(f"runs:/{run_id_rf}/model", "CreditRiskModel")
print(f"Model registered: version {result.version}")
```

```bash
# Launch MLflow tracking UI
mlflow ui --host 0.0.0.0 --port 5000
# Open http://localhost:5000 to compare experiments visually
```

---

## Week 3–4: Data Version Control (DVC)

DVC is Git for data and models. It lets you version large files without storing them in Git.

```bash
# Setup
pip install dvc dvc-s3  # or dvc-gcs, dvc-azure
git init
dvc init

# Configure remote storage (S3, GCS, Azure, or local)
dvc remote add -d myremote s3://my-ml-bucket/dvc-storage
# For local testing:
dvc remote add -d localremote /tmp/dvc-remote

# Track a dataset
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add raw dataset"

# Track a trained model
dvc add models/model.pkl
git add models/model.pkl.dvc
git commit -m "Add trained model v1"

# Push data to remote
dvc push

# Pull data on another machine
dvc pull
```

```python
# dvc.yaml — ML pipeline definition
# Defines stages, dependencies, outputs — enables reproducible pipelines

stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/dataset.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
    params:
      - params.yaml:
        - prepare.test_size
        - prepare.random_state

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.csv
    params:
      - params.yaml:
        - train.n_estimators
        - train.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/processed/test.csv
    metrics:
      - metrics/test_scores.json:
          cache: false
    plots:
      - metrics/roc_curve.csv:
          x: fpr
          y: tpr
```

```yaml
# params.yaml
prepare:
  test_size: 0.2
  random_state: 42

train:
  n_estimators: 100
  max_depth: 10
  learning_rate: 0.05
```

```bash
# Run the full pipeline
dvc repro

# Run only changed stages
dvc repro --downstream train

# Experiment tracking with DVC
dvc exp run --set-param train.n_estimators=200
dvc exp show  # Compare experiments in a table
dvc exp diff  # Diff between experiments
```

---

## Week 5–6: Pipeline Orchestration

### Prefect (Modern, Pythonic)

```python
# pip install prefect
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import joblib
import logging

logger = logging.getLogger(__name__)


@task(name="load-data", retries=3, retry_delay_seconds=30,
      cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_data(data_path: str) -> pd.DataFrame:
    """Load raw data from source."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} rows")
    return df


@task(name="validate-data")
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data quality before processing."""
    assert len(df) > 0, "Dataset is empty!"
    assert df.isnull().mean().max() < 0.5, "Too many missing values!"
    assert "target" in df.columns, "Target column missing!"

    logger.info(f"Data validation passed: {df.shape}")
    return df


@task(name="preprocess-data")
def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and preprocess data."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = df.drop("target", axis=1).select_dtypes(include="number")
    y = df["target"]

    # Fill missing values
    X = X.fillna(X.median())

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    return (X_train, y_train), (X_test, y_test)


@task(name="train-model")
def train_model(train_data: tuple, n_estimators: int = 100) -> RandomForestClassifier:
    """Train the ML model."""
    X_train, y_train = train_data
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info(f"Model trained with {n_estimators} trees")
    return model


@task(name="evaluate-model")
def evaluate_model(model: RandomForestClassifier, test_data: tuple) -> dict:
    """Evaluate model and log to MLflow."""
    X_test, y_test = test_data
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    metrics = {"test_auc": auc, "n_features": X_test.shape[1]}
    logger.info(f"Model AUC: {auc:.4f}")
    return metrics


@task(name="save-model")
def save_model(model: RandomForestClassifier, metrics: dict, output_path: str) -> str:
    """Save model if it meets quality threshold."""
    if metrics["test_auc"] < 0.70:
        raise ValueError(f"Model quality below threshold: AUC={metrics['test_auc']:.4f} < 0.70")

    joblib.dump(model, output_path)
    logger.info(f"Model saved to {output_path}")
    return output_path


@flow(name="ml-training-pipeline", log_prints=True)
def ml_training_pipeline(
    data_path: str = "data/processed/features.csv",
    model_output: str = "models/model.pkl",
    n_estimators: int = 100
):
    """End-to-end ML training pipeline."""
    # Load and validate
    raw_data = load_data(data_path)
    valid_data = validate_data(raw_data)

    # Preprocess
    train_data, test_data = preprocess_data(valid_data)

    # Train and evaluate
    model = train_model(train_data, n_estimators=n_estimators)
    metrics = evaluate_model(model, test_data)

    # Save if good enough
    save_model(model, metrics, model_output)

    return metrics


# Run locally
if __name__ == "__main__":
    results = ml_training_pipeline(n_estimators=200)
    print(f"Pipeline complete! Metrics: {results}")
```

```bash
# Schedule with Prefect Cloud (free tier available)
prefect cloud login
prefect deploy ml_pipeline.py:ml_training_pipeline \
    --name "daily-retraining" \
    --cron "0 2 * * *"   # Run at 2am daily
```

---

## Week 7–8: Production Model Serving

### Serving at Scale with FastAPI + Gunicorn

```python
# src/serving/app.py
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import logging

# ============================================================
# Metrics for monitoring
# ============================================================
PREDICTION_COUNT = Counter("predictions_total", "Total predictions made", ["model_version", "outcome"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency",
                                 buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
MODEL_LOADED = Gauge("model_loaded", "Whether model is loaded", ["version"])
BATCH_SIZE = Histogram("batch_size", "Size of batch predictions", buckets=[1, 5, 10, 50, 100, 500])

logger = logging.getLogger(__name__)


class PredictionInput(BaseModel):
    feature_1: float = Field(..., description="First feature")
    feature_2: float = Field(default=0.0, description="Second feature")
    feature_3: float = Field(default=0.0)
    feature_4: float = Field(default=0.0)
    feature_5: float = Field(default=0.0)


class PredictionOutput(BaseModel):
    prediction_id: str
    prediction: int
    probability: float
    model_version: str
    latency_ms: float


# Model store
class ModelStore:
    def __init__(self):
        self.model = None
        self.version = "unknown"
        self.feature_names = []

    def load(self, path: str):
        self.model = joblib.load(path)
        self.version = "1.0.0"
        logger.info(f"Model {self.version} loaded from {path}")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]


model_store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store.load("models/model.pkl")
    MODEL_LOADED.labels(version=model_store.version).set(1)
    yield
    MODEL_LOADED.labels(version=model_store.version).set(0)


app = FastAPI(title="ML Prediction Service", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(prometheus_client.generate_latest())


@app.post("/predict", response_model=PredictionOutput)
async def predict(body: PredictionInput, request: Request):
    t0 = time.perf_counter()
    X = np.array([[body.feature_1, body.feature_2, body.feature_3,
                   body.feature_4, body.feature_5]])
    try:
        preds, probas = model_store.predict(X)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {e}")

    latency = (time.perf_counter() - t0) * 1000
    PREDICTION_LATENCY.observe(latency / 1000)
    PREDICTION_COUNT.labels(version=model_store.version, outcome=str(preds[0])).inc()

    return PredictionOutput(
        prediction_id=str(uuid.uuid4())[:8],
        prediction=int(preds[0]),
        probability=round(float(probas[0]), 4),
        model_version=model_store.version,
        latency_ms=round(latency, 2)
    )
```

---

## Week 9–10: Model Monitoring & Drift Detection

```python
# pip install evidently
import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric, DatasetMissingValuesMetric,
    ColumnDriftMetric, ClassificationQualityMetric
)

# Simulate reference (training) data and production data
np.random.seed(42)
n = 1000

reference_data = pd.DataFrame({
    "feature_1": np.random.normal(50, 10, n),
    "feature_2": np.random.normal(25, 5, n),
    "feature_3": np.random.uniform(0, 1, n),
    "feature_4": np.random.randint(0, 5, n),
    "prediction": np.random.randint(0, 2, n),
    "target": np.random.randint(0, 2, n),
})

# Simulated production drift: feature_1 distribution shifted
current_data = pd.DataFrame({
    "feature_1": np.random.normal(65, 15, n),  # DRIFTED — mean shifted from 50 to 65!
    "feature_2": np.random.normal(25, 5, n),   # Same as training
    "feature_3": np.random.uniform(0, 1, n),
    "feature_4": np.random.randint(0, 5, n),
    "prediction": np.random.randint(0, 2, n),
    "target": np.random.randint(0, 2, n),
})

column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=["feature_1", "feature_2", "feature_3"],
    categorical_features=["feature_4"]
)

# ── Data Drift Report ────────────────────────────────────────
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    DatasetMissingValuesMetric(),
    ColumnDriftMetric(column_name="feature_1"),  # Monitor specific column
])

report.run(reference_data=reference_data, current_data=current_data,
           column_mapping=column_mapping)

report.save_html("monitoring/data_drift_report.html")
print("Drift report saved!")

# Extract metrics programmatically
result = report.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]
print(f"Dataset drift detected: {drift_detected}")

# ── Programmatic alerts ──────────────────────────────────────
def check_drift_and_alert(reference, current, threshold=0.05):
    """
    Check for drift and trigger alerts.
    In production: send to Slack, PagerDuty, email, etc.
    """
    from scipy.stats import ks_2samp

    alerts = []
    for col in reference.select_dtypes(include="number").columns:
        stat, pvalue = ks_2samp(reference[col].dropna(), current[col].dropna())
        if pvalue < threshold:
            alerts.append({
                "column": col,
                "ks_statistic": stat,
                "pvalue": pvalue,
                "severity": "HIGH" if pvalue < 0.001 else "MEDIUM"
            })

    if alerts:
        print(f"⚠️  DRIFT DETECTED in {len(alerts)} features:")
        for alert in alerts:
            print(f"  {alert['severity']}: {alert['column']} (KS={alert['ks_statistic']:.3f}, p={alert['pvalue']:.6f})")
        # In production: trigger_retraining_job()
    else:
        print("✅ No significant drift detected")

    return alerts

alerts = check_drift_and_alert(
    reference_data[["feature_1", "feature_2", "feature_3"]],
    current_data[["feature_1", "feature_2", "feature_3"]]
)
```

---

## Week 11–14: Feature Store + CI/CD

### GitHub Actions CI/CD for ML

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline CI/CD

on:
  push:
    branches: [main, develop]
    paths:
      - "src/**"
      - "params.yaml"
  schedule:
    - cron: "0 2 * * 1"   # Weekly retrain on Monday at 2am
  workflow_dispatch:        # Manual trigger

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  train-and-evaluate:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: iterative/setup-dvc@v1

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Pull data from DVC remote
        run: dvc pull data/

      - name: Run ML pipeline
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          dvc repro
          echo "Training complete!"

      - name: Check model quality gate
        run: |
          python scripts/check_model_quality.py \
            --min-auc 0.80 \
            --metrics-file metrics/scores.json

      - name: Push model to registry
        run: |
          python scripts/register_model.py \
            --model-path models/model.pkl \
            --run-name "CI-$(date +%Y%m%d)"

  deploy:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t ml-api:${{ github.sha }} .

      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker push $ECR_REGISTRY/ml-api:${{ github.sha }}

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster ml-prod \
            --service ml-api \
            --force-new-deployment
```

---

## Intermediate Phase Skills Checklist

- [ ] Track experiments with MLflow (params, metrics, artifacts, model registry)
- [ ] Version datasets and models with DVC
- [ ] Build a Prefect or Airflow pipeline with at least 4 tasks
- [ ] Deploy a model to a cloud endpoint (AWS/GCP/Azure)
- [ ] Set up drift detection with Evidently AI
- [ ] Write a GitHub Actions workflow that tests and trains your model
- [ ] Monitor model performance with custom metrics in Prometheus/Grafana

**Next:** [Advanced Phase →](../advanced/)