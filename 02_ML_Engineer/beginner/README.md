# ML Engineer — Beginner Phase

**Goal:** Write production-quality Python, containerize applications, serve ML models via APIs, and understand the full ML lifecycle.

**Duration:** 2–3 months at 10–15 hrs/week
**Prerequisites:** Python basics, familiarity with at least one ML library (e.g., sklearn)

---

## Curriculum Overview

```
Week 1–2  → Production Python (OOP, testing, packaging, type hints)
Week 3–4  → Docker & Containerization
Week 5–6  → REST APIs with FastAPI
Week 7–8  → ML Model Training & Serialization
Week 9–10 → Your First ML API (train → package → serve)
Week 11–12→ Git workflows, CI basics, project wrap-up
```

---

## Week 1–2: Production Python for ML Engineers

Unlike DS work in notebooks, MLE code lives in production systems. It needs to be testable, maintainable, and reproducible.

### 1.1 Object-Oriented Python for ML

```python
# ============================================================
# Example: A well-designed ML pipeline class
# ============================================================
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging — always use logging, never print() in production!
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Immutable configuration for an ML model."""
    model_name: str
    version: str = "1.0.0"
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = "target"
    hyperparams: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "hyperparams": self.hyperparams,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**d)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> ModelConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible preprocessor.
    Subclasses BaseEstimator + TransformerMixin to get:
    - fit() / transform() / fit_transform() interface
    - get_params() / set_params() for hyperparameter tuning
    """

    def __init__(self, impute_strategy: str = "median", scale: bool = True):
        self.impute_strategy = impute_strategy
        self.scale = scale

    def fit(self, X: pd.DataFrame, y=None) -> DataPreprocessor:
        """Learn parameters from training data."""
        logger.info(f"Fitting preprocessor on {X.shape}")
        self._numeric_cols = X.select_dtypes(include="number").columns.tolist()
        self._cat_cols = X.select_dtypes(include="object").columns.tolist()

        # Learn imputation values from training data ONLY
        if self.impute_strategy == "median":
            self._fill_values = X[self._numeric_cols].median()
        elif self.impute_strategy == "mean":
            self._fill_values = X[self._numeric_cols].mean()

        # Learn scaling parameters from training data ONLY
        if self.scale:
            self._means = X[self._numeric_cols].mean()
            self._stds = X[self._numeric_cols].std().replace(0, 1)  # Avoid div by 0

        return self  # Always return self from fit!

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Apply learned parameters to new data."""
        logger.info(f"Transforming {X.shape}")
        X = X.copy()

        # Impute missing values
        X[self._numeric_cols] = X[self._numeric_cols].fillna(self._fill_values)

        # Scale
        if self.scale:
            X[self._numeric_cols] = (X[self._numeric_cols] - self._means) / self._stds

        return X[self._numeric_cols].values


class MLPipeline:
    """End-to-end ML pipeline: preprocess → train → predict → save/load."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.preprocessor = DataPreprocessor()
        self.model = None
        self._is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> MLPipeline:
        """Train the pipeline on data."""
        logger.info(f"Training pipeline: {self.config.model_name} v{self.config.version}")

        # Import model type dynamically based on config
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        model_map = {
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "logistic_regression": LogisticRegression,
        }

        model_cls = model_map.get(self.config.model_name)
        if model_cls is None:
            raise ValueError(f"Unknown model: {self.config.model_name}. Choose from {list(model_map)}")

        self.model = model_cls(**self.config.hyperparams)

        # Preprocess
        X_processed = self.preprocessor.fit_transform(X_train)

        # Train
        self.model.fit(X_processed, y_train)
        self._is_fitted = True

        logger.info("Training complete!")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(f"{self.config.model_name} doesn't support probabilities")
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before predicting. Call .fit() first.")

    def save(self, directory: Path | str) -> Path:
        """Save pipeline artifacts to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save config as JSON (human-readable)
        self.config.save(directory / "config.json")

        # Save pipeline as pickle (binary)
        artifacts = {
            "preprocessor": self.preprocessor,
            "model": self.model,
            "_is_fitted": self._is_fitted,
        }
        with open(directory / "pipeline.pkl", "wb") as f:
            pickle.dump(artifacts, f)

        logger.info(f"Pipeline saved to {directory}")
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> MLPipeline:
        """Load a saved pipeline."""
        directory = Path(directory)
        config = ModelConfig.load(directory / "config.json")
        pipeline = cls(config)

        with open(directory / "pipeline.pkl", "rb") as f:
            artifacts = pickle.load(f)

        pipeline.preprocessor = artifacts["preprocessor"]
        pipeline.model = artifacts["model"]
        pipeline._is_fitted = artifacts["_is_fitted"]

        logger.info(f"Pipeline loaded from {directory}")
        return pipeline
```

### 1.2 Testing with pytest

Testing is non-negotiable in production ML systems. Every function you write should have tests.

```python
# tests/test_pipeline.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from your_module import MLPipeline, ModelConfig, DataPreprocessor


@pytest.fixture
def sample_data():
    """Create reusable sample data for tests."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "age": np.random.randint(20, 60, n),
        "income": np.random.normal(60000, 20000, n),
        "score": np.random.uniform(0, 100, n),
        "missing_col": [None if i % 5 == 0 else np.random.randn() for i in range(n)],
    })
    y = pd.Series((X["income"] > 65000).astype(int), name="high_income")
    return X, y


@pytest.fixture
def config():
    """Sample model config."""
    return ModelConfig(
        model_name="random_forest",
        version="1.0.0",
        feature_columns=["age", "income", "score"],
        hyperparams={"n_estimators": 10, "random_state": 42}
    )


class TestDataPreprocessor:
    def test_fit_transform_shape(self, sample_data):
        X, _ = sample_data
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(X.select_dtypes(include="number"))
        assert result.shape[0] == len(X)  # Same number of rows

    def test_no_missing_values_after_transform(self, sample_data):
        X, _ = sample_data
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(X.select_dtypes(include="number"))
        assert not np.isnan(result).any(), "Missing values remain after transform!"

    def test_transform_uses_fit_statistics(self, sample_data):
        """Transform should use TRAINING statistics, not test statistics."""
        X, _ = sample_data
        X_train, X_test = X[:80], X[80:]

        preprocessor = DataPreprocessor()
        X_cols = X_train.select_dtypes(include="number")
        preprocessor.fit(X_cols)

        # Transform test set using training statistics
        result_1 = preprocessor.transform(X_test.select_dtypes(include="number"))

        # The means should be training means, not test means
        assert preprocessor._means is not None


class TestMLPipeline:
    def test_predict_before_fit_raises(self, config, sample_data):
        X, _ = sample_data
        pipeline = MLPipeline(config)
        with pytest.raises(RuntimeError, match="must be fitted"):
            pipeline.predict(X)

    def test_fit_predict_basic(self, config, sample_data):
        X, y = sample_data
        pipeline = MLPipeline(config)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_save_load_roundtrip(self, config, sample_data):
        X, y = sample_data
        pipeline = MLPipeline(config)
        pipeline.fit(X, y)
        original_preds = pipeline.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save(tmpdir)
            loaded = MLPipeline.load(tmpdir)
            loaded_preds = loaded.predict(X)

        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_invalid_model_raises(self, sample_data):
        X, y = sample_data
        bad_config = ModelConfig(model_name="non_existent_model")
        pipeline = MLPipeline(bad_config)
        with pytest.raises(ValueError, match="Unknown model"):
            pipeline.fit(X, y)

    @pytest.mark.parametrize("model_name", ["random_forest", "logistic_regression"])
    def test_multiple_models(self, sample_data, model_name):
        X, y = sample_data
        config = ModelConfig(model_name=model_name, hyperparams={"random_state": 42})
        pipeline = MLPipeline(config)
        pipeline.fit(X, y)
        assert pipeline._is_fitted


# Run with: pytest tests/ -v --cov=your_module --cov-report=term-missing
```

---

## Week 3–4: Docker for ML Engineers

Docker lets you package your model + code + dependencies into a portable container that runs identically everywhere.

### Why Docker in ML?

```
Without Docker:               With Docker:
"It works on my machine!"     "It works in the container, everywhere!"

Dev env ≠ Prod env            Dev env = Prod env = Test env
Dependency conflicts          Isolated dependencies
"What Python version?"        Always Python 3.11
Manual setup = 2 hours        docker build . = 10 minutes
```

### Dockerfile for an ML API

```dockerfile
# Dockerfile
# Start from an official Python image (slim = smaller size)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies (for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies FIRST (cached layer — faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code AFTER dependencies
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user for security
RUN adduser --disabled-password --gecos "" mluser && \
    chown -R mluser:mluser /app
USER mluser

# Expose the port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml — orchestrate multiple services
version: "3.9"

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - MODEL_PATH=/app/models/pipeline.pkl
    volumes:
      - ./models:/app/models:ro  # Read-only model mount
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: add a monitoring stack
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

```bash
# Essential Docker commands for ML Engineers
docker build -t my-ml-api:v1.0 .          # Build image
docker build --no-cache -t my-ml-api .     # Force rebuild (no cache)
docker run -p 8000:8000 my-ml-api:v1.0    # Run container
docker run -d -p 8000:8000 my-ml-api      # Run in background (detached)
docker logs <container_id>                  # View logs
docker exec -it <container_id> bash        # Shell into running container
docker ps                                   # List running containers
docker ps -a                               # List all containers (including stopped)
docker stop <container_id>                  # Stop container
docker rm <container_id>                    # Remove container
docker images                              # List images
docker rmi <image_id>                      # Remove image
docker system prune                        # Clean up unused resources
docker-compose up -d                       # Start all services (background)
docker-compose down                        # Stop all services
docker-compose logs -f ml-api              # Follow logs for a service
```

---

## Week 5–6: REST APIs with FastAPI

FastAPI is the standard for serving ML models. It's fast, automatic docs, type-safe, and built on async Python.

```python
# src/api.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from .pipeline import MLPipeline

logger = logging.getLogger(__name__)

# ============================================================
# Models (Pydantic schemas for request/response validation)
# ============================================================
class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age in years")
    income: float = Field(..., ge=0, description="Annual income in USD")
    score: float = Field(..., ge=0, le=100, description="Credit score 0-100")
    missing_col: Optional[float] = Field(None, description="Optional feature")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 75000,
                "score": 72.5,
                "missing_col": None
            }
        }

class BatchPredictionRequest(BaseModel):
    instances: List[PredictionRequest] = Field(..., min_items=1, max_items=1000)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    prediction_label: str
    model_version: str
    latency_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_instances: int
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


# ============================================================
# App Lifecycle (load model at startup)
# ============================================================
pipeline: Optional[MLPipeline] = None
start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global pipeline
    model_path = Path("models/pipeline")
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        pipeline = MLPipeline.load(model_path)
        logger.info(f"Model {pipeline.config.model_name} v{pipeline.config.version} loaded!")
    else:
        logger.warning(f"No model found at {model_path} — predictions will fail")
    yield
    logger.info("Shutting down ML API")


# ============================================================
# Create FastAPI App
# ============================================================
app = FastAPI(
    title="ML Model API",
    description="Production-grade ML model serving API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Endpoints
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy" if pipeline else "degraded",
        "model_loaded": pipeline is not None,
        "model_version": pipeline.config.version if pipeline else "N/A",
        "uptime_seconds": time.time() - start_time,
    }

@app.get("/model-info")
async def model_info():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "name": pipeline.config.model_name,
        "version": pipeline.config.version,
        "features": pipeline.config.feature_columns,
        "hyperparams": pipeline.config.hyperparams,
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    X = pd.DataFrame([request.model_dump()])

    try:
        prediction = int(pipeline.predict(X)[0])
        proba = float(pipeline.predict_proba(X)[0][prediction])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    latency = (time.time() - t0) * 1000
    return {
        "prediction": prediction,
        "probability": round(proba, 4),
        "prediction_label": "high_income" if prediction == 1 else "low_income",
        "model_version": pipeline.config.version,
        "latency_ms": round(latency, 2),
    }

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    X = pd.DataFrame([r.model_dump() for r in request.instances])

    predictions_raw = pipeline.predict(X)
    probas = pipeline.predict_proba(X)

    predictions = []
    for pred, proba_row in zip(predictions_raw, probas):
        pred = int(pred)
        predictions.append({
            "prediction": pred,
            "probability": round(float(proba_row[pred]), 4),
            "prediction_label": "high_income" if pred == 1 else "low_income",
            "model_version": pipeline.config.version,
            "latency_ms": 0,  # Will be set to total/n
        })

    total_latency = (time.time() - t0) * 1000
    return {
        "predictions": predictions,
        "total_instances": len(predictions),
        "total_latency_ms": round(total_latency, 2),
    }
```

```bash
# Run the API
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Test it (use the auto-generated Swagger UI at http://localhost:8000/docs)
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 75000, "score": 72.5}'
```

---

## Week 7–10: Training, Serialization & Your First ML API

### Train a Model and Serve It

```python
# train.py — Run this script to train and save your model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path

from src.pipeline import MLPipeline, ModelConfig

# ============================================================
# 1. Load and prepare data
# ============================================================
print("Loading data...")
# Using UCI Adult Income dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ["age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
df = pd.read_csv(url, names=cols, skipinitialspace=True)

# Prepare features and target
X = df[["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]].copy()
y = (df["income"] == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================
# 2. Train the pipeline
# ============================================================
config = ModelConfig(
    model_name="random_forest",
    version="1.0.0",
    feature_columns=list(X.columns),
    hyperparams={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1  # Use all CPU cores
    }
)

pipeline = MLPipeline(config)
pipeline.fit(X_train, y_train)

# ============================================================
# 3. Evaluate
# ============================================================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n📊 Evaluation Results:")
print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

# ============================================================
# 4. Save the model
# ============================================================
save_path = pipeline.save("models/pipeline")
print(f"\n✅ Model saved to {save_path}")
print("Now run: uvicorn src.api:app --reload")
```

---

## Week 11–12: Git Workflows for ML Teams

### ML-Specific Git Practices

```bash
# Standard feature branch workflow
git checkout -b feature/add-xgboost-model
# ... make changes ...
git add src/models/xgboost_model.py tests/test_xgboost.py
git commit -m "feat: add XGBoost model option to pipeline

- Add GradientBoostingClassifier and XGBoost to model_map
- Add unit tests for both model types
- Update ModelConfig hyperparams documentation"

git push origin feature/add-xgboost-model
# Open Pull Request on GitHub

# Tag releases
git tag -a v1.0.0 -m "Release v1.0.0: Random Forest baseline"
git push origin v1.0.0
```

### Pre-commit Hooks for Code Quality

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-merge-conflict
      - id: detect-private-key   # Prevent committing secrets!

  - repo: local
    hooks:
      - id: pytest
        name: Run tests
        entry: pytest tests/ -x -q
        language: system
        pass_filenames: false
        always_run: true
```

```bash
# Setup pre-commit (run once)
pip install pre-commit
pre-commit install
# Now every git commit will auto-run these checks!
```

---

## Beginner Phase Skills Checklist

- [ ] Write Python classes with proper OOP (init, methods, properties)
- [ ] Use type hints on all function signatures
- [ ] Write pytest tests for at least 5 functions
- [ ] Build and run a Docker container
- [ ] Build a FastAPI endpoint that serves a model
- [ ] Train a model and save/load it correctly
- [ ] Can explain what a Docker image vs container is
- [ ] Set up pre-commit hooks on a project
- [ ] Know git branching workflow for teams

**Next:** [Intermediate Phase →](../intermediate/)
