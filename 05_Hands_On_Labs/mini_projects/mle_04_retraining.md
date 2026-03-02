# Automated Model Retraining Pipeline

> **Difficulty:** Advanced | **Time:** 2-3 days | **Track:** ML Engineer

## What You'll Build
An automated ML retraining pipeline that monitors model performance, triggers retraining when accuracy degrades, logs new experiments to MLflow, and promotes models only when they beat the current production version.

## Learning Objectives
- Design a retraining trigger strategy
- Build a complete retrain-evaluate-promote pipeline
- Implement model rollback capability
- Orchestrate with Prefect or Airflow
- Add notifications on retraining events

## Tech Stack
- `mlflow`: experiment tracking and registry
- `scikit-learn`: model training
- `prefect`: pipeline orchestration (or Airflow)
- `evidently`: drift-based retraining triggers
- `pandas`: data management

## Step-by-Step Guide

### Step 1: Retraining Trigger Logic
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

class RetriggerReason(str, Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"

@dataclass
class RetriggerDecision:
    should_retrain: bool
    reason: Optional[RetriggerReason]
    details: dict

def check_performance_trigger(
    current_accuracy: float,
    baseline_accuracy: float,
    threshold_pct: float = 0.05
) -> RetriggerDecision:
    """Trigger retraining if accuracy drops more than threshold_pct."""
    drop = baseline_accuracy - current_accuracy
    drop_pct = drop / baseline_accuracy if baseline_accuracy > 0 else 0

    if drop_pct > threshold_pct:
        return RetriggerDecision(
            should_retrain=True,
            reason=RetriggerReason.PERFORMANCE_DEGRADATION,
            details={'accuracy_drop': round(drop_pct, 4), 'current': current_accuracy, 'baseline': baseline_accuracy}
        )
    return RetriggerDecision(should_retrain=False, reason=None, details={'accuracy_drop': round(drop_pct, 4)})

# Test
decision = check_performance_trigger(current_accuracy=0.82, baseline_accuracy=0.89, threshold_pct=0.05)
print(f"Retrain? {decision.should_retrain} | Reason: {decision.reason} | Details: {decision.details}")
```

### Step 2: Retraining Pipeline
```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime

mlflow.set_experiment('auto-retrain-pipeline')

def retrain_model(
    X_train, y_train, X_test, y_test,
    trigger_reason: RetriggerReason,
    params: dict = None
) -> dict:
    """Train a new model and log it to MLflow."""
    if params is None:
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}

    with mlflow.start_run(run_name=f'retrain_{datetime.now().strftime("%Y%m%d_%H%M")}') as run:
        mlflow.log_param('trigger_reason', trigger_reason.value)
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        mlflow.log_metrics({'accuracy': acc, 'auc_roc': auc})
        mlflow.sklearn.log_model(model, 'model', registered_model_name='production_model')

        return {'run_id': run.info.run_id, 'accuracy': acc, 'auc_roc': auc, 'model': model}
```

### Step 3: Promote or Rollback Logic
```python
def evaluate_and_promote(
    new_model_metrics: dict,
    production_metrics: dict,
    run_id: str,
    min_improvement: float = 0.01
) -> str:
    """Promote new model only if it beats production by min_improvement."""
    delta = new_model_metrics['accuracy'] - production_metrics['accuracy']

    if delta >= min_improvement:
        client = mlflow.tracking.MlflowClient()
        # Transition current production to archived
        current_versions = client.get_latest_versions('production_model', stages=['Production'])
        for v in current_versions:
            client.transition_model_version_stage('production_model', v.version, 'Archived')

        # Promote new model
        new_versions = client.get_latest_versions('production_model', stages=['None'])
        if new_versions:
            client.transition_model_version_stage('production_model', new_versions[-1].version, 'Production')
            print(f"✅ New model promoted! Delta: +{delta:.4f}")
            return 'promoted'

    print(f"⚠️ New model NOT promoted. Delta: {delta:.4f} < threshold {min_improvement}")
    return 'rejected'
```

### Step 4: Prefect Orchestration
```python
# pipeline.py — orchestrate with Prefect
from prefect import flow, task

@task(retries=2, retry_delay_seconds=30)
def load_fresh_data():
    X, y = make_classification(n_samples=2000, n_features=10, random_state=int(datetime.now().timestamp()))
    return train_test_split(X, y, test_size=0.2, random_state=42)

@task
def check_drift_trigger(X_new, X_reference):
    """Returns True if drift detected."""
    from scipy import stats
    p_values = [stats.ks_2samp(X_reference[:, i], X_new[:, i])[1] for i in range(X_new.shape[1])]
    return sum(p < 0.05 for p in p_values) > X_new.shape[1] * 0.3

@flow(name="auto-retrain-flow")
def auto_retrain_pipeline():
    X_train, X_test, y_train, y_test = load_fresh_data()
    drift = check_drift_trigger(X_train, X_train + 0.5)  # simulated drift

    if drift:
        print("Drift detected — triggering retraining")
        result = retrain_model(X_train, y_train, X_test, y_test, RetriggerReason.DATA_DRIFT)
        status = evaluate_and_promote(result, {'accuracy': 0.85}, result['run_id'])
        print(f"Pipeline complete: {status}")
    else:
        print("No drift — skipping retraining")

# Run: python pipeline.py
if __name__ == '__main__':
    auto_retrain_pipeline()
```

### Step 5: Monitoring Dashboard
```python
# view all retrains over time
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name('auto-retrain-pipeline')

if experiment:
    runs = client.search_runs(experiment.experiment_id, order_by=['start_time DESC'])
    history = [{
        'timestamp': r.info.start_time,
        'trigger': r.data.params.get('trigger_reason', 'N/A'),
        'accuracy': r.data.metrics.get('accuracy', 0),
        'promoted': 'Production' in str(r.info)
    } for r in runs]
    print(pd.DataFrame(history).to_string(index=False))
```

## Expected Output
- Pipeline that runs on schedule and retrains only when needed
- MLflow experiment history showing all retrain events with trigger reasons
- Promotion/rejection decision log per retrain
- Model registry showing current Production, Staging, and Archived versions

## Stretch Goals
- [ ] Add A/B testing: after retraining, route 10% of traffic to the new model and compare live accuracy for 24 hours before full promotion
- [ ] Add Slack notifications: send a message on every retrain event with before/after metrics
- [ ] Deploy with GitHub Actions: trigger the pipeline on a cron schedule using a workflow YAML

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
