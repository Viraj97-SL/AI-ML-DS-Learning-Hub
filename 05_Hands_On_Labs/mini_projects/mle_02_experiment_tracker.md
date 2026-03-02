# Experiment Tracker with MLflow

> **Difficulty:** Intermediate | **Time:** 1-2 days | **Track:** ML Engineer

## What You'll Build
Track 5+ ML experiments in MLflow, compare them in the UI, find the best model automatically, and promote it to the model registry.

## Learning Objectives
- Log metrics, params, and artifacts with MLflow
- Compare experiments in the MLflow UI
- Use the Model Registry for staging and production
- Automate best-model selection programmatically

## Tech Stack
- `mlflow`: experiment tracking and registry
- `scikit-learn`: ML models
- `optuna`: hyperparameter search (optional)
- `pandas`: data handling

## Step-by-Step Guide

### Step 1: Setup MLflow
```bash
pip install mlflow scikit-learn optuna
mlflow ui  # open http://localhost:5000 in browser
```

### Step 2: Run Tracked Experiments
```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

mlflow.set_experiment('breast-cancer-classification')
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

experiments = [
    ('LogisticRegression_C0.1', LogisticRegression(C=0.1, max_iter=200), {}),
    ('LogisticRegression_C1.0', LogisticRegression(C=1.0, max_iter=200), {}),
    ('RandomForest_n100', RandomForestClassifier(n_estimators=100, random_state=42), {}),
    ('RandomForest_n200_depth5', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42), {}),
    ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), {}),
]

for run_name, model, extra_params in experiments:
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({**model.get_params(), **extra_params})

        # Train
        model.fit(X_train, y_train)

        # Log metrics
        cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
        test_acc = accuracy_score(y_test, model.predict(X_test))
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_test))

        mlflow.log_metrics({'cv_accuracy': cv_score, 'test_accuracy': test_acc, 'test_auc': test_auc})

        # Log model
        mlflow.sklearn.log_model(model, 'model', registered_model_name='cancer_classifier')

        print(f'{run_name}: CV={cv_score:.4f} | TestAcc={test_acc:.4f} | AUC={test_auc:.4f}')
```

### Step 3: Find and Register Best Model
```python
import mlflow.tracking

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name('breast-cancer-classification')

# Get all runs sorted by AUC
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['metrics.test_auc DESC']
)

best_run = runs[0]
print(f'Best run: {best_run.info.run_name}')
print(f'Best AUC: {best_run.data.metrics["test_auc"]:.4f}')
print(f'Best params: {best_run.data.params}')

# Promote to production
model_version = client.get_latest_versions('cancer_classifier', stages=['None'])[0]
client.transition_model_version_stage(
    name='cancer_classifier',
    version=model_version.version,
    stage='Production'
)
print(f'Model v{model_version.version} promoted to Production!')
```

### Step 4: Load and Use Production Model
```python
# Load directly from registry (no hardcoded paths)
production_model = mlflow.sklearn.load_model('models:/cancer_classifier/Production')

import numpy as np
sample = X_test[:5]
predictions = production_model.predict(sample)
probabilities = production_model.predict_proba(sample)[:, 1]

print('Production model predictions:')
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f'  Sample {i}: class={pred}, confidence={prob:.3f}')
```

### Step 5: MLflow UI Exploration
```
In the MLflow UI (http://localhost:5000):
1. Navigate to "breast-cancer-classification" experiment
2. Click "Compare runs" and select all 5 runs
3. Use "Parallel Coordinates Plot" to visualize param-metric relationships
4. Click "Models" → "cancer_classifier" to see version history
5. Check the "Artifacts" tab for the serialized model files
```

## Expected Output
- 5 tracked experiments with full metrics in MLflow UI
- Best model automatically identified by AUC-ROC
- Model promoted to "Production" stage in registry
- Predictions served from registry (no file paths needed)

## Stretch Goals
- [ ] Add Optuna hyperparameter search: run 20 trials of GradientBoosting with Optuna, log each trial to MLflow, and find the optimal hyperparameter combination
- [ ] Create a custom metric: log a confusion matrix as an HTML artifact in MLflow
- [ ] Add data versioning: log the training data hash as a tag so you can always reproduce any experiment exactly

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
