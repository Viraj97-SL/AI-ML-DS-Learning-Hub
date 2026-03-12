# ML Model Benchmarking Suite

> **Difficulty:** Advanced | **Time:** 2-3 days | **Track:** ML Engineer

## What You'll Build
A standardized, reusable benchmarking framework that trains multiple ML models on any dataset, evaluates them across a rich set of metrics, tracks all experiments in MLflow, generates a comparative report with visualizations, and enforces quality gates via pytest.

## Learning Objectives
- Design a dataset-agnostic model benchmarking harness
- Track experiments, parameters, and artifacts with MLflow
- Implement statistical significance testing between models
- Generate HTML benchmark reports with matplotlib and pandas
- Write pytest fixtures and assertion tests for ML quality gates

## Prerequisites
- Comfortable with scikit-learn pipelines and model evaluation
- Basic MLflow tracking API knowledge
- Python testing with pytest

## Tech Stack
- `mlflow`: experiment tracking and model registry
- `scikit-learn`: diverse model zoo and evaluation utilities
- `pandas`: results aggregation, pivoting, and export
- `matplotlib` / `seaborn`: benchmark visualization plots
- `pytest`: quality gate tests with custom fixtures
- `scipy`: statistical significance tests between models

## Step-by-Step Guide

### Step 1: Dataset Loader and Benchmark Configuration
```python
# pip install mlflow scikit-learn pandas matplotlib seaborn pytest scipy

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification, make_regression,
    load_breast_cancer, load_diabetes,
)
from sklearn.model_selection import train_test_split

@dataclass
class BenchmarkConfig:
    name: str
    task_type: str                         # "classification" | "regression"
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    metrics: list[str] = field(default_factory=list)
    quality_gates: dict[str, float] = field(default_factory=dict)

@dataclass
class BenchmarkDataset:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    n_classes: Optional[int] = None

def load_dataset(dataset_name: str, config: BenchmarkConfig) -> BenchmarkDataset:
    """Load a named dataset and split it for benchmarking."""
    loaders = {
        "breast_cancer": lambda: load_breast_cancer(return_X_y=True, as_frame=False),
        "diabetes":      lambda: load_diabetes(return_X_y=True),
        "synthetic_clf": lambda: make_classification(n_samples=5000, n_features=20,
                                                       n_informative=10, random_state=config.random_state),
        "synthetic_reg": lambda: make_regression(n_samples=5000, n_features=20,
                                                  n_informative=10, random_state=config.random_state),
    }
    X, y = loaders[dataset_name]()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.test_size,
                                               random_state=config.random_state)
    print(f"Dataset '{dataset_name}': {X_tr.shape[0]} train / {X_te.shape[0]} test | {X.shape[1]} features")
    return BenchmarkDataset(dataset_name, X_tr, X_te, y_tr, y_te,
                            feature_names=[f"f{i}" for i in range(X.shape[1])],
                            n_classes=len(np.unique(y)) if config.task_type == "classification" else None)

cfg = BenchmarkConfig("clf_benchmark", "classification",
                       metrics=["accuracy", "roc_auc", "f1", "precision", "recall"],
                       quality_gates={"accuracy": 0.80, "roc_auc": 0.85})
ds = load_dataset("breast_cancer", cfg)
```

### Step 2: Model Zoo and Training Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier

def build_model_zoo(task_type: str) -> dict[str, Pipeline]:
    """Return a dict of named sklearn Pipelines ready for benchmarking."""
    scaler = StandardScaler()

    if task_type == "classification":
        return {
            "LogisticRegression": Pipeline([("scaler", scaler),
                                             ("model", LogisticRegression(max_iter=1000, random_state=42))]),
            "RandomForest":       Pipeline([("scaler", scaler),
                                             ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
            "GradientBoosting":   Pipeline([("scaler", scaler),
                                             ("model", GradientBoostingClassifier(n_estimators=100, random_state=42))]),
            "SVM":                Pipeline([("scaler", scaler),
                                             ("model", SVC(probability=True, random_state=42))]),
            "KNN":                Pipeline([("scaler", scaler),
                                             ("model", KNeighborsClassifier(n_neighbors=7))]),
        }
    else:
        return {
            "Ridge":            Pipeline([("scaler", scaler), ("model", Ridge())]),
            "RandomForest":     Pipeline([("scaler", scaler), ("model", RandomForestRegressor(n_estimators=100, random_state=42))]),
            "GradientBoosting": Pipeline([("scaler", scaler), ("model", GradientBoostingRegressor(n_estimators=100, random_state=42))]),
            "SVR":              Pipeline([("scaler", scaler), ("model", SVR())]),
        }

zoo = build_model_zoo("classification")
print(f"Model zoo: {list(zoo.keys())}")
```

### Step 3: Evaluation Engine with Cross-Validation
```python
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time

@dataclass
class ModelResult:
    model_name: str
    dataset_name: str
    metrics: dict[str, float]
    cv_scores: dict[str, float]
    train_time_s: float
    predict_time_ms: float
    n_params: Optional[int] = None

def evaluate_classifier(
    name: str, pipeline: Pipeline,
    dataset: BenchmarkDataset, cv_folds: int = 5,
) -> ModelResult:
    """Train, cross-validate, and evaluate a classifier."""
    # Training
    t0 = time.perf_counter()
    pipeline.fit(dataset.X_train, dataset.y_train)
    train_time = time.perf_counter() - t0

    # Prediction timing
    t1 = time.perf_counter()
    y_pred = pipeline.predict(dataset.X_test)
    predict_time = (time.perf_counter() - t1) * 1000

    y_proba = pipeline.predict_proba(dataset.X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics = {
        "accuracy":  accuracy_score(dataset.y_test, y_pred),
        "f1":        f1_score(dataset.y_test, y_pred, average="weighted"),
        "precision": precision_score(dataset.y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(dataset.y_test, y_pred, average="weighted"),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(dataset.y_test, y_proba)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_acc = cross_val_score(pipeline, dataset.X_train, dataset.y_train, cv=cv, scoring="accuracy")
    cv_scores = {"cv_mean": cv_acc.mean(), "cv_std": cv_acc.std()}

    print(f"  {name:<22} acc={metrics['accuracy']:.3f}  "
          f"auc={metrics.get('roc_auc', 0):.3f}  "
          f"cv={cv_scores['cv_mean']:.3f}±{cv_scores['cv_std']:.3f}  "
          f"train={train_time:.2f}s")

    return ModelResult(name, dataset.name, metrics, cv_scores, train_time, predict_time)

print("Evaluating all classifiers...")
results = [evaluate_classifier(name, pipe, ds) for name, pipe in zoo.items()]
```

### Step 4: MLflow Experiment Tracking
```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("model-benchmarking-suite")

def log_benchmark_to_mlflow(result: ModelResult, pipeline: Pipeline) -> str:
    """Log a single model benchmark result to MLflow."""
    with mlflow.start_run(run_name=f"{result.model_name}_{result.dataset_name}") as run:
        # Log all metrics
        mlflow.log_metrics(result.metrics)
        mlflow.log_metrics(result.cv_scores)
        mlflow.log_metric("train_time_s", result.train_time_s)
        mlflow.log_metric("predict_time_ms", result.predict_time_ms)

        # Log params from the pipeline
        model_obj = pipeline.named_steps.get("model")
        if model_obj:
            mlflow.log_params(model_obj.get_params())

        # Log tags
        mlflow.set_tags({"dataset": result.dataset_name, "task": "classification"})

        # Log the model
        mlflow.sklearn.log_model(pipeline, "pipeline",
                                  registered_model_name=f"benchmark_{result.model_name}")
        return run.info.run_id

# Log all results
for result, (name, pipe) in zip(results, zoo.items()):
    run_id = log_benchmark_to_mlflow(result, pipe)
    print(f"Logged {name}: run_id={run_id[:8]}...")

print("\nMLflow UI: mlflow ui --port 5000")
```

### Step 5: Statistical Testing and Report Generation
```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def build_results_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    """Flatten all ModelResult objects into a tidy DataFrame."""
    rows = []
    for r in results:
        row = {"model": r.model_name, "dataset": r.dataset_name,
               "train_time_s": r.train_time_s, "predict_time_ms": r.predict_time_ms,
               **r.metrics, **r.cv_scores}
        rows.append(row)
    return pd.DataFrame(rows)

def statistical_comparison(
    results_df: pd.DataFrame,
    metric: str = "accuracy",
    baseline_model: str = "LogisticRegression",
) -> pd.DataFrame:
    """Compute pairwise statistical significance vs baseline using Wilcoxon test."""
    comparisons = []
    for model in results_df["model"].unique():
        if model == baseline_model:
            continue
        base_score = results_df[results_df["model"] == baseline_model][metric].values
        test_score = results_df[results_df["model"] == model][metric].values
        delta = test_score.mean() - base_score.mean() if len(test_score) > 1 else test_score[0] - base_score[0]
        comparisons.append({"model": model, "delta_vs_baseline": delta,
                             "baseline": baseline_model, "metric": metric})
    return pd.DataFrame(comparisons)

def plot_benchmark_report(results_df: pd.DataFrame, output_path: str = "benchmark_report.png"):
    """Generate a 4-panel benchmark visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ML Model Benchmark Report", fontsize=16, fontweight="bold")

    models = results_df["model"]
    colors = sns.color_palette("husl", len(models))

    # Panel 1: Accuracy comparison
    axes[0, 0].barh(models, results_df["accuracy"], color=colors)
    axes[0, 0].set_title("Accuracy (Test Set)")
    axes[0, 0].set_xlim(0, 1)
    for i, v in enumerate(results_df["accuracy"]):
        axes[0, 0].text(v + 0.005, i, f"{v:.3f}", va="center")

    # Panel 2: CV mean ± std
    axes[0, 1].barh(models, results_df["cv_mean"], xerr=results_df["cv_std"],
                    color=colors, capsize=4)
    axes[0, 1].set_title("Cross-Validation Accuracy (mean ± std)")
    axes[0, 1].set_xlim(0, 1)

    # Panel 3: Train time
    axes[1, 0].barh(models, results_df["train_time_s"], color=colors)
    axes[1, 0].set_title("Training Time (seconds)")
    axes[1, 0].set_xlabel("Seconds")

    # Panel 4: Radar-style multi-metric (displayed as grouped bars)
    metric_cols = [c for c in ["accuracy", "roc_auc", "f1", "precision", "recall"] if c in results_df.columns]
    melted = results_df[["model"] + metric_cols].melt(id_vars="model", var_name="metric", value_name="score")
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=axes[1, 1], palette="husl")
    axes[1, 1].set_title("All Metrics Comparison")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Report saved to: {output_path}")
    return fig

results_df = build_results_dataframe(results)
print("\nBenchmark Summary:")
print(results_df[["model", "accuracy", "roc_auc", "cv_mean", "train_time_s"]].to_string(index=False))
```

## Expected Output
- A complete benchmark run across 4-5 classifiers (or regressors) on any sklearn-compatible dataset
- MLflow experiment with one run per model, tracking all metrics, params, and logged model artifacts
- A 4-panel matplotlib benchmark report saved as PNG
- A `results.csv` export with all metrics for downstream analysis
- Statistical comparison table showing which models beat the baseline at p < 0.05

## Stretch Goals
- [ ] **Pytest quality gates:** Write a `test_benchmark.py` that loads the results CSV and asserts that: (1) at least one model exceeds the configured accuracy threshold, (2) the best model's CV std is below 0.05 (no high variance), and (3) the training time for any single model stays under 60 seconds; fail the CI pipeline if any gate fails
- [ ] **Hyperparameter sweep integration:** Add an `--optimize` CLI flag that runs Optuna to tune the top-2 models from the first pass (by accuracy), logs each trial as an MLflow child run, and reports the delta between default and optimized performance
- [ ] **Dataset-agnostic loader:** Extend `load_dataset()` to accept any CSV or Parquet file path, auto-detect whether the task is classification or regression based on the target column's cardinality, and automatically apply appropriate preprocessing (ordinal encoding, missing value imputation)

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`