# ML Engineer Interview Preparation

> MLE interviews test your software engineering skills AND your ML knowledge. You need to be strong in both.

---

## MLE Interview Structure (Typical)

```
Round 1: Recruiter Screen (30 min) → Background, motivation
Round 2: Coding Screen (45-60 min) → LeetCode-style coding (Medium/Hard)
Round 3: ML Concepts (60 min) → ML fundamentals + system design intro
Round 4: ML System Design (60 min) → Design an ML system end-to-end
Round 5: Practical ML Coding (60 min) → Train/evaluate model, debug pipeline
Round 6: Behavioral + Team Fit (45 min) → Leadership principles, collaboration
```

---

## Software Engineering Questions

### Data Structures & Algorithms

**Q1: Implement a function to detect data drift between two distributions.**

```python
def detect_drift(reference: list[float], current: list[float],
                 method: str = "ks", alpha: float = 0.05) -> dict:
    """
    Detect distribution drift between reference and current data.

    Methods:
    - 'ks': Kolmogorov-Smirnov test (non-parametric, general)
    - 'psi': Population Stability Index (common in finance)
    - 'chi2': Chi-square test (categorical features)
    """
    import numpy as np
    from scipy import stats

    result = {"method": method, "drift_detected": False}

    if method == "ks":
        stat, pvalue = stats.ks_2samp(reference, current)
        result.update({"statistic": stat, "pvalue": pvalue,
                        "drift_detected": pvalue < alpha})

    elif method == "psi":
        # PSI > 0.25 → significant drift, 0.1-0.25 → moderate
        reference, current = np.array(reference), np.array(current)
        bins = np.percentile(reference, np.linspace(0, 100, 11))
        bins[0] -= 1e-10; bins[-1] += 1e-10
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)
        ref_pct = (ref_hist + 0.0001) / len(reference)
        cur_pct = (cur_hist + 0.0001) / len(current)
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        result.update({"psi": psi, "drift_detected": psi > 0.25})

    return result

# Test it
import numpy as np
reference = np.random.normal(50, 10, 1000).tolist()
current_no_drift = np.random.normal(50, 10, 1000).tolist()   # Same distribution
current_with_drift = np.random.normal(65, 12, 1000).tolist() # Shifted!

print(detect_drift(reference, current_no_drift, method="ks"))
print(detect_drift(reference, current_with_drift, method="ks"))
print(detect_drift(reference, current_with_drift, method="psi"))
```

**Q2: Design and implement a feature store (simplified).**

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import json

@dataclass
class FeatureView:
    name: str
    features: List[str]
    ttl_seconds: Optional[int] = None  # Time-to-live for cached features
    description: str = ""

@dataclass
class Feature:
    name: str
    value: Any
    timestamp: datetime
    entity_id: str

class SimpleFeatureStore:
    """Simplified feature store for learning purposes."""

    def __init__(self):
        self._online_store: Dict[str, Dict[str, Feature]] = {}  # entity_id → feature_name → Feature
        self._feature_views: Dict[str, FeatureView] = {}
        self._offline_data: Optional[pd.DataFrame] = None

    def register_feature_view(self, feature_view: FeatureView):
        self._feature_views[feature_view.name] = feature_view

    def write_to_online_store(self, entity_id: str, features: Dict[str, Any]):
        if entity_id not in self._online_store:
            self._online_store[entity_id] = {}
        for name, value in features.items():
            self._online_store[entity_id][name] = Feature(
                name=name, value=value,
                timestamp=datetime.utcnow(), entity_id=entity_id
            )

    def get_online_features(self, entity_id: str, feature_names: List[str]) -> Dict[str, Any]:
        """Low-latency feature retrieval for model serving (<10ms SLA)."""
        result = {}
        entity_features = self._online_store.get(entity_id, {})
        for name in feature_names:
            feature = entity_features.get(name)
            if feature is None:
                result[name] = None  # Missing feature
            else:
                # Check TTL
                view = self._find_view_for_feature(name)
                if view and view.ttl_seconds:
                    age = (datetime.utcnow() - feature.timestamp).total_seconds()
                    if age > view.ttl_seconds:
                        result[name] = None  # Expired
                        continue
                result[name] = feature.value
        return result

    def _find_view_for_feature(self, feature_name: str) -> Optional[FeatureView]:
        for view in self._feature_views.values():
            if feature_name in view.features:
                return view
        return None

    def materialize_to_online(self, df: pd.DataFrame, entity_col: str, feature_cols: List[str]):
        """Batch-write features to online store (e.g., from daily training run)."""
        for _, row in df.iterrows():
            self.write_to_online_store(
                entity_id=str(row[entity_col]),
                features={col: row[col] for col in feature_cols}
            )

# Usage example
fs = SimpleFeatureStore()

view = FeatureView(
    name="user_features",
    features=["age", "total_purchases", "days_since_last_order", "avg_order_value"],
    ttl_seconds=3600  # 1 hour
)
fs.register_feature_view(view)

# Materialize from batch DataFrame
user_data = pd.DataFrame({
    "user_id": ["u001", "u002", "u003"],
    "age": [28, 45, 33],
    "total_purchases": [5, 23, 12],
    "days_since_last_order": [3, 45, 7],
    "avg_order_value": [85.0, 120.0, 65.0]
})
fs.materialize_to_online(user_data, "user_id", view.features)

# Online retrieval during inference
features = fs.get_online_features("u001", view.features)
print(f"Features for u001: {features}")
```

---

## ML System Design Questions

### Q3: Design a Real-Time Recommendation System for Netflix

**Framework to always use: TRAPS**
- **T**raffic estimation
- **R**equirements (functional + non-functional)
- **A**rchitecture overview
- **P**ipeline details
- **S**cale and tradeoffs

**Answer framework:**

**Scale:**
- 250M users, 15,000+ titles, 100M daily active users
- Latency SLA: <100ms for recommendations

**Functional Requirements:**
1. Given a user, return top-K recommended items
2. Recommendations update based on recent viewing history
3. Support cold start (new users, new items)

**Architecture:**

```
┌────────────────────────────────────────────────────────┐
│ OFFLINE (batch, runs daily)                            │
│                                                        │
│ Raw data → Feature engineering → Train model          │
│ (Spark)     (Feast)              (PyTorch/TF on GPUs) │
│                                                        │
│ → Generate candidate embeddings for all users+items   │
│ → Store in vector database (FAISS/Milvus)             │
└────────────────────────────────────────────────────────┘
         ↓ embeddings                    ↓ features
┌─────────────────────┐    ┌─────────────────────────────┐
│ ONLINE SERVING       │    │ FEATURE STORE (Redis)       │
│                     │    │ user_recent_watches         │
│ User request        │    │ user_demographics           │
│    ↓                │    │ item_popularity_score       │
│ Retrieve user emb.  │←──│ user_engagement_rate        │
│    ↓                │    └─────────────────────────────┘
│ ANN search (FAISS)  │
│ → Top-500 candidates│
│    ↓                │
│ Ranking model       │← Features from store
│ (LightGBM/DNN)     │
│ → Top-20 results   │
│    ↓               │
│ Business rules     │← Diversity filter, A/B test
│ → Final 10 items   │
└─────────────────────┘

Total latency budget:
  Feature retrieval: 10ms
  ANN search: 20ms
  Ranking: 30ms
  Business rules: 5ms
  Total: ~65ms (well within 100ms SLA)
```

**Key ML Components:**
- **Two-tower model:** User tower + Item tower → learn embeddings
- **Approximate Nearest Neighbor (ANN):** FAISS HNSW for fast similarity search
- **Ranking model:** Learns from implicit feedback (watch time, completion rate)
- **Diversity:** Add items from different genres to avoid filter bubble

**Tradeoffs:**
- Offline vs online features (freshness vs latency)
- Model accuracy vs inference speed
- Exploration vs exploitation (Thompson sampling)
- Collaborative filtering vs content-based vs hybrid

---

### Q4: Design a Fraud Detection System

**Constraints:** 1M transactions/day, <100ms latency, $0.01/transaction cost to review, $200 average fraud loss

**Architecture:**

```
Transaction Stream
        ↓
Real-time Feature Engineering (Kafka + Flink)
  - Amount deviation from user's typical spend
  - Time since last transaction (in same location)
  - Velocity: # transactions in last 1h
  - Geo-distance from previous transaction
        ↓
Fraud Scoring Service (<50ms budget)
  - Online features from Redis Feature Store
  - Gradient Boosting Model (LightGBM)
  - Return fraud_score ∈ [0, 1]
        ↓
Decision Engine
  - score > 0.9 → Block automatically
  - score > 0.7 → Send to human review queue
  - score < 0.7 → Allow (log for monitoring)
        ↓
Feedback Loop
  - Human decisions → label confirmed frauds
  - Weekly model retraining
  - Drift monitoring on feature distributions
```

**Key ML Challenges:**
- Class imbalance (fraud rate ~0.1%) → SMOTE + class weights + AUC-PR metric
- Adversarial adaptation (fraudsters adapt) → frequent retraining + concept drift detection
- Low latency → feature pre-computation, model quantization
- Ground truth delay → need chargeback data (3-7 day lag)

---

## MLOps / Production Questions

**Q5: Your model's accuracy dropped from 92% to 84% in production. How do you debug this?**

```
Step 1: Check data quality first
  - Are there new null/NaN values in any feature?
  - Did feature distributions shift? (run KS test on each)
  - Is there a schema change (new feature values, missing columns)?

Step 2: Check for label drift
  - Is the class distribution different from training?
  - Has the definition of "positive" changed?

Step 3: Check model behavior
  - Are confidence scores calibrated? (are 90% confident predictions right 90% of time?)
  - Which segments/cohorts are performing worst?
  - Is there temporal drift? (performance drop correlates with time?)

Step 4: Compare feature importance
  - Have top features changed?
  - Are important features showing drift?

Step 5: Check infrastructure
  - Any data pipeline changes recently?
  - Are features being computed correctly?
  - Is there train-serve skew (features differ between training and serving)?

Root cause categories:
  A. Feature distribution shift (data drift)     → Retrain with new data
  B. Label shift (target distribution changed)   → Retrain with new labels
  C. Concept drift (relationship changed)        → New model + features
  D. Data quality issue                          → Fix pipeline
  E. Train-serve skew                            → Fix feature computation
```

**Q6: How would you reduce model inference latency from 200ms to <20ms?**

```python
# Optimization techniques in order of impact:

# 1. Model quantization (8-bit vs 32-bit → 4x speedup, minimal accuracy loss)
import torch
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 2. ONNX export (standardized format, optimized runtime)
import torch.onnx
dummy_input = torch.randn(1, input_size)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}})

# 3. TensorRT (NVIDIA GPU optimization — 5-10x speedup on GPU)
# 4. Batching (process 32 requests together → amortize overhead)
# 5. Caching (cache predictions for common inputs)
# 6. Feature pre-computation (compute expensive features offline)
# 7. Model distillation (train small student from large teacher)
# 8. Pruning (remove near-zero weights)
```

---

## Behavioral Questions for MLEs

- Tell me about a model you deployed to production. What challenges did you face?
- Describe a time you had to balance model accuracy vs latency/cost tradeoffs.
- Tell me about a time your data pipeline had a bug. How did you find and fix it?
- How do you approach retraining strategy for a production model?
- Describe a technical disagreement with a team member and how you resolved it.

---

## MLE 30-Day Interview Prep Plan

| Week | Focus |
|------|-------|
| 1 | LeetCode medium: Arrays, HashMaps, Trees, Sorting (2 problems/day) |
| 2 | ML fundamentals: bias-variance, regularization, evaluation metrics, algorithm internals |
| 3 | ML System Design: practice 5 system design questions using TRAPS framework |
| 4 | Mock interviews + behavioral stories (prepare 8 STAR stories) |

**Top resources:**
- [Designing ML Systems (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [Made With ML — MLOps course](https://madewithml.com)
- [LeetCode — Top Interview 150](https://leetcode.com/studyplan/top-interview-150/)
- [ML System Design Interview](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/B09YQWX59Z)

---

*Back to: [Interview Prep](../) | [MLE Track](../../02_ML_Engineer/) | [Main README](../../README.md)*