# ML Engineer — Portfolio Projects

> Each project demonstrates production-ready ML engineering: not just training models, but deploying, monitoring, and maintaining them at scale.

---

## What Makes an MLE Project Different from a DS Project?

| Data Scientist Project | ML Engineer Project |
|------------------------|---------------------|
| Jupyter notebook | Production Python package |
| Accuracy metrics | Latency, throughput, cost metrics |
| Model training | Model training + serving + monitoring |
| Local execution | Docker + Kubernetes deployment |
| Manual runs | Automated pipelines |

---

## Project 1: Production ML Serving System (Beginner MLE)

**Objective:** Take a trained scikit-learn model and build a production-grade serving system around it.

**Dataset:** [UCI Adult Income](https://archive.ics.uci.edu/ml/datasets/adult)

**What to build:**

```
income-prediction-service/
├── src/
│   ├── model/
│   │   ├── train.py           ← Training script
│   │   ├── predict.py         ← Prediction logic
│   │   └── evaluate.py        ← Evaluation metrics
│   ├── api/
│   │   ├── main.py            ← FastAPI app
│   │   ├── schemas.py         ← Pydantic models
│   │   └── middleware.py      ← Logging, rate limiting
│   └── utils/
│       ├── preprocessing.py   ← Feature pipeline
│       └── monitoring.py      ← Prometheus metrics
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_preprocessing.py
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── k8s/
│       ├── deployment.yaml
│       └── service.yaml
├── notebooks/
│   └── 01_eda_and_training.ipynb
├── Makefile
├── requirements.txt
└── README.md
```

**Core implementation:**

```python
# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import joblib
import time
import logging

from .schemas import PredictionRequest, PredictionResponse, BatchRequest, BatchResponse
from ..model.predict import Predictor
from ..utils.monitoring import record_prediction

logger = logging.getLogger(__name__)
predictor: Predictor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading ML model...")
    predictor = Predictor.load("models/income_model_v1.pkl")
    logger.info(f"Model loaded: {predictor.model_version}")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Income Prediction Service",
    version="1.0.0",
    lifespan=lifespan
)

REQUEST_COUNT = Counter("predictions_total", "Total predictions", ["result"])
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
ERROR_COUNT = Counter("prediction_errors_total", "Total prediction errors")

@app.get("/health")
def health():
    return {"status": "healthy", "model_version": predictor.model_version if predictor else None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    with REQUEST_LATENCY.time():
        try:
            result = predictor.predict(request.dict())
            REQUEST_COUNT.labels(result=str(result.prediction)).inc()
            return result
        except Exception as e:
            ERROR_COUNT.inc()
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

**Key deliverables:**
- API with <20ms p99 latency
- 99% uptime with health checks + auto-restart
- Prometheus metrics dashboard in Grafana
- GitHub Actions CI/CD pipeline
- Load test results (1000 req/s sustained)

---

## Project 2: End-to-End MLOps Pipeline (Intermediate MLE)

**Objective:** Build a fully automated ML pipeline from data ingestion to model deployment with monitoring.

**Dataset:** [NYC Taxi (TLC)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — predict tip percentage

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOPS PIPELINE ARCHITECTURE                   │
│                                                                 │
│  GitHub Push                                                    │
│      ↓                                                          │
│  GitHub Actions CI                                              │
│    ├── Run tests (pytest)                                       │
│    ├── Data quality checks                                      │
│    └── Model quality gate (AUC > 0.85)                         │
│          ↓                                                      │
│  DVC Pipeline (dvc repro)                                       │
│    ├── Stage 1: Download raw data                               │
│    ├── Stage 2: Preprocess + feature engineering                │
│    ├── Stage 3: Train model (logged to MLflow)                  │
│    └── Stage 4: Evaluate + compare to production                │
│          ↓                                                      │
│  If evaluation passes:                                          │
│    ├── Register model in MLflow Registry                        │
│    ├── Build Docker image → Push to ECR                         │
│    └── Deploy to ECS/Kubernetes (blue-green)                    │
│          ↓                                                      │
│  Production monitoring:                                         │
│    ├── Evidently data drift reports (daily)                     │
│    ├── Model performance monitoring (weekly)                    │
│    └── PagerDuty alert if AUC drops >5%                         │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation highlights:**

```python
# dvc.yaml
stages:
  download:
    cmd: python src/data/download.py
    params:
      - data.year
      - data.months
    outs:
      - data/raw/

  preprocess:
    cmd: python src/features/build_features.py
    deps:
      - data/raw/
      - src/features/build_features.py
    params:
      - features.target_column
      - features.categorical_cols
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet

  train:
    cmd: python src/models/train_model.py
    deps:
      - data/processed/train.parquet
      - src/models/train_model.py
    params:
      - model.n_estimators
      - model.learning_rate
      - model.max_depth
    outs:
      - models/model.pkl
    metrics:
      - reports/metrics.json:
          cache: false

  evaluate:
    cmd: python src/models/evaluate_model.py
    deps:
      - data/processed/test.parquet
      - models/model.pkl
    metrics:
      - reports/eval_metrics.json:
          cache: false
    plots:
      - reports/feature_importance.csv
      - reports/roc_curve.csv
```

**Portfolio talking points:**
- "Reduced model deployment time from 2 days to 15 minutes"
- "Implemented automated drift detection that caught a data quality issue before it impacted production"
- "Blue-green deployment with zero downtime model updates"

---

## Project 3: Real-Time Feature Store (Intermediate MLE)

**Objective:** Build a production feature store that serves features with <10ms latency.

**Problem:** Credit card fraud detection where features must be computed in real-time.

**Tech stack:** Redis + Kafka + Flink + FastAPI

```python
# Feature definitions (using Feast)
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Define entities
user = Entity(
    name="user_id",
    value_type=ValueType.STRING,
    description="User identifier"
)

# Batch source (for offline training)
user_stats_source = FileSource(
    path="s3://feast-data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

# Feature view
user_activity_fv = FeatureView(
    name="user_activity_features",
    entities=["user_id"],
    ttl=timedelta(hours=24),
    features=[
        Feature(name="txn_count_1h", dtype=ValueType.INT64),
        Feature(name="txn_count_24h", dtype=ValueType.INT64),
        Feature(name="avg_amount_7d", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_txn", dtype=ValueType.INT64),
        Feature(name="unique_merchants_7d", dtype=ValueType.INT64),
    ],
    online=True,
    source=user_stats_source,
)

# Real-time feature computation (Flink job)
# Consumes Kafka transaction stream, computes sliding-window features,
# writes to Redis for <10ms online retrieval
```

**Key metrics to achieve:**
- Feature retrieval: <5ms p99
- Feature freshness: <1 minute lag from event
- Write throughput: 50,000 feature updates/second
- 99.9% uptime SLA

---

## Project 4: Distributed Training Pipeline (Advanced MLE)

**Objective:** Train a large model on 100M+ samples using distributed computing.

**Dataset:** Custom tabular dataset or [Criteo Click-Through Rate](https://www.kaggle.com/c/criteo-display-ad-challenge) (1B rows)

**What to demonstrate:**

```python
# Multi-GPU training with DDP + gradient accumulation
# Target: Train on 100M rows in <2 hours on 4 GPUs

training_config = {
    "architecture": "Wide & Deep (Google, 2016) for CTR prediction",
    "training": {
        "gpus": 4,
        "ddp": True,
        "batch_size_per_gpu": 2048,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 65536,
        "learning_rate": 3e-4,
        "scheduler": "cosine with warmup",
        "mixed_precision": "bfloat16",
    },
    "data": {
        "total_rows": "100M+",
        "format": "Parquet on S3",
        "prefetch_workers": 8,
        "streaming": True,  # Don't load all data into RAM
    },
    "results": {
        "training_time": "1h 42min",
        "peak_memory_per_gpu": "14.2 GB",
        "final_auc": 0.792,
        "throughput": "280K samples/second",
    }
}
```

**System design document to write:**
- GPU memory calculation (model params + activations + gradients + optimizer states)
- Bottleneck analysis: compute-bound vs memory-bound vs IO-bound
- Cost estimate: AWS p3.8xlarge vs spot instances vs Spot with checkpointing

---

## Project 5: LLM Fine-Tuning Service (Advanced MLE)

**Objective:** Build a service that allows customers to fine-tune an open-source LLM on their own data.

**Inspired by:** OpenAI fine-tuning API, Anyscale Endpoints, Modal

**Architecture:**

```
Customer uploads JSONL file (instruction-response pairs)
      ↓
Upload validation service (format check, PII detection, content moderation)
      ↓
Fine-tuning job queue (Redis Queue / Celery)
      ↓
Fine-tuning worker (QLoRA on A100/H100)
  - Load base model (Llama/Mistral/Phi)
  - Train with customer data (QLoRA)
  - Evaluate on held-out test set
  - Save adapter weights to S3
      ↓
Deployment service
  - Load base model + customer adapter
  - Serve via FastAPI with /completions endpoint
  - Auto-scale based on request volume
      ↓
Monitoring dashboard
  - Per-customer usage (tokens, latency)
  - Model quality metrics
  - Cost tracking
```

**Business metrics to track:**
- Fine-tuning time: target <30 minutes for 10K examples
- Inference cost: <$0.002 per 1K tokens
- Customer satisfaction: measured by re-use rate

---

## Project 6: ML Platform (Staff/Principal Level)

**Objective:** Design and build a self-serve ML platform for a 50-person data science team.

**The problem:** Data scientists are blocked for weeks waiting for MLOps/DevOps help to train models, deploy them, and monitor them.

**What to build:**

```
SELF-SERVE ML PLATFORM
    │
    ├── Experiment Tracking (MLflow)
    │   └── Any DS can track experiments, compare runs, register models
    │
    ├── Batch Training Infrastructure
    │   ├── Submit training jobs via simple Python API
    │   ├── Auto-provisions GPU/CPU clusters
    │   └── Results stored in MLflow automatically
    │
    ├── Online Serving
    │   ├── One-click deployment from MLflow registry
    │   ├── Auto-scales based on traffic
    │   └── Automatic A/B testing framework
    │
    ├── Data Platform
    │   ├── Feature store (Feast)
    │   ├── Data catalog (DataHub or Amundsen)
    │   └── Automated data quality checks (Great Expectations)
    │
    └── Monitoring & Alerting
        ├── Per-model dashboards (auto-generated)
        ├── Drift detection (Evidently)
        └── PagerDuty integration
```

**Success metrics for your platform:**
- DS time to first deployment: from 2 weeks → 2 hours
- Model deployment frequency: from 1/month → 10/week
- Mean time to detect model degradation: from 1 week → 1 day
- Infrastructure cost reduction: 30% (via better GPU utilization)

---

## MLE Project Evaluation Rubric

| Criteria | 1 (Basic) | 3 (Good) | 5 (Excellent) |
|----------|-----------|----------|----------------|
| **Code Quality** | Works but messy | Clean, documented | Production-grade with tests |
| **System Design** | Single service | Well-structured | Scalable architecture |
| **MLOps** | Manual runs | CI/CD pipeline | Full automation + monitoring |
| **Performance** | Functional | Meets basic SLAs | Benchmarked with optimization |
| **Documentation** | README exists | Clear setup guide | Architecture docs + decisions |

---

## How to Present MLE Projects in Interviews

**The STAR-E format (add Engineering to STAR):**
- **Situation:** Business problem + technical constraints
- **Task:** Engineering goals with specific SLAs
- **Action:** Design decisions with tradeoffs explained
- **Result:** Quantified outcomes (latency, throughput, cost, reliability)
- **Engineering decisions:** "I chose X over Y because..."

**Example answer:**
> "I built a real-time fraud scoring service with a <50ms P99 latency SLA. I chose LightGBM over XGBoost because LightGBM's leaf-wise growth gives better accuracy on tabular data with the same training time. I used Redis for feature caching to reduce the p99 from 180ms to 35ms. I deployed on Kubernetes with HPA scaling from 2 to 20 replicas, which handled Black Friday traffic spikes without manual intervention. The system processed 2M transactions/day at 99.95% uptime."

---

*Back to: [MLE Track](../README.md) | [Main README](../../README.md)*