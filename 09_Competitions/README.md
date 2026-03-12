<div align="center">

# Competitive ML & AI — The Complete Guide

### From your first submission to your first medal

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![DrivenData](https://img.shields.io/badge/DrivenData-FF6B35?style=for-the-badge)](https://drivendata.org)
[![Zindi](https://img.shields.io/badge/Zindi-5C2D91?style=for-the-badge)](https://zindi.africa)

</div>

---

## Why Compete?

Competitive ML is the fastest way to go from "I know ML theory" to "I can solve real problems under pressure." Here's what competitions give you that courses don't:

- **Real, messy data** — not cleaned textbook datasets
- **Leaderboard feedback** — know exactly where you rank against thousands of practitioners worldwide
- **Forcing function** — deadlines force you to ship instead of endlessly tweaking
- **Community knowledge** — discussion forums are a goldmine of feature ideas and techniques
- **Portfolio signal** — a Kaggle medal or top-10 finish is immediately legible to hiring managers
- **Prize money** — top competitions offer $25,000–$1,000,000 in prizes

> **The fastest path to a DS/ML/AI job is: coursework + competitions + portfolio. Skip any one and you're slower.**

---

## Platform Comparison

| Platform | Best For | Prize Pool | Community Size | Difficulty | Free Compute |
|----------|----------|------------|----------------|------------|--------------|
| [Kaggle](https://kaggle.com) | Tabular, CV, NLP, all types | $0–$1M | 17M+ users | All levels | Yes (GPU/TPU) |
| [DrivenData](https://drivendata.org) | Social good, health, climate | $5K–$50K | Smaller, focused | Intermediate+ | No |
| [Zindi](https://zindi.africa) | Africa-focused problems | $500–$10K | 50K+ | Beginner-friendly | No |
| [AIcrowd](https://aicrowd.com) | Reinforcement Learning, Research | Varies | 30K+ | Advanced | Partial |
| [HuggingFace Spaces](https://huggingface.co/spaces) | LLM, generative AI | Community | 1M+ | All levels | Yes (via ZeroGPU) |
| [Analytics Vidhya](https://datahack.analyticsvidhya.com) | Indian market, hackathons | $1K–$20K | 500K+ | Beginner-intermediate | No |
| [Numerai](https://numer.ai) | Quantitative finance | Weekly payouts (NMR) | 10K+ | Advanced | No |
| [CodaLab](https://codalab.org) | Academic, research | Varies | Research-focused | Advanced | No |

**Recommendation for beginners:** Start on Kaggle. The free compute, community discussions, and shared notebooks (Kernels) make it uniquely beginner-friendly.

---

## Getting Started: Step-by-Step Guide for Beginners

### Step 1: Choose Your First Competition (Week 1)

Do NOT jump into an active $100K competition on your first day. Instead:

1. Go to [kaggle.com/competitions](https://kaggle.com/competitions)
2. Filter by: **Getting Started** or **Playground**
3. Start with the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) — it's the "Hello World" of Kaggle
4. Then try [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### Step 2: Understand the Problem (Days 1–2)

Before writing a single line of code:
- Read the **Overview** tab completely — understand the business context
- Study the **Data** tab — understand every column, file, and format
- Read the **Evaluation** tab — know exactly how you'll be scored (accuracy, AUC, RMSE, etc.)
- Browse the **Discussion** tab — see what experienced competitors are saying

### Step 3: Run the Baseline (Days 2–3)

```python
# Standard competition starter template
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load data
train = pd.read_csv('/kaggle/input/competition-name/train.csv')
test  = pd.read_csv('/kaggle/input/competition-name/test.csv')

# Quick EDA
print(train.shape, test.shape)
print(train.dtypes)
print(train.isnull().sum())
print(train.describe())

# Minimal baseline — get a submission in!
# (replace with actual target and feature columns)
X = train.drop(['id', 'target'], axis=1).select_dtypes(include='number').fillna(-999)
y = train['target']
X_test = test.select_dtypes(include='number').fillna(-999)

model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
print(f"CV AUC: {cv_score:.4f}")

model.fit(X, y)
preds = model.predict_proba(X_test)[:, 1]

# Submit
submission = pd.DataFrame({'id': test['id'], 'target': preds})
submission.to_csv('submission.csv', index=False)
```

> **Rule #1:** Get a valid submission in within the first 48 hours. The leaderboard is your compass.

### Step 4: Explore & Improve (Days 3–14)

Once you have a baseline score, iterate in this order:
1. **EDA** — visualize distributions, correlations, target balance
2. **Feature engineering** — create new features from existing ones
3. **Model tuning** — try XGBoost, LightGBM, CatBoost
4. **Cross-validation** — validate properly before submitting
5. **Ensembling** — combine your best models

### Step 5: Study Shared Notebooks (Ongoing)

The single highest ROI activity on Kaggle is reading top public notebooks:
- Sort by "Most Votes" in the **Code** tab
- Read EDA notebooks to understand the data better
- Study modeling notebooks to learn new techniques

### Step 6: Engage with the Community

- Post questions in **Discussion** — the community is welcoming
- Share your findings (even if not winning — karma/votes matter)
- Read winning solutions *after* the competition ends

---

## Competition Lifecycle

Every well-run competition attempt follows this arc:

```
Phase 1: Understand (Days 1–3)
    ↓
    Read problem statement thoroughly
    Study evaluation metric (AUC, F1, RMSE, mAP, etc.)
    Download and inspect all data files
    Read top discussion threads

Phase 2: EDA (Days 3–7)
    ↓
    Distribution analysis of features and target
    Missing value analysis
    Correlation heatmaps
    Time-based splits (for time series)
    Target leakage check

Phase 3: Baseline (Days 5–7)
    ↓
    Minimal working submission (even naive model)
    Establish cross-validation strategy
    Verify CV ↔ LB correlation

Phase 4: Feature Engineering (Days 7–20)
    ↓
    Domain-specific features
    Aggregations and group statistics
    Interaction features
    Target encoding (with care)
    External data sources

Phase 5: Modeling (Days 15–35)
    ↓
    Gradient boosting (LightGBM/XGBoost/CatBoost)
    Neural networks if applicable
    Proper hyperparameter tuning (Optuna)
    Model diversity for ensembling

Phase 6: Ensembling (Days 30–Final)
    ↓
    Simple averaging of diverse models
    Stacking with meta-learner
    Blending based on OOF predictions

Phase 7: Final Submission (Last 2 days)
    ↓
    Select 2 submissions strategically
    One "safe" (best CV), one "risky" (best LB)
    Double-check submission format
```

---

## Evaluation Metrics Quick Reference

| Metric | Used When | Notes |
|--------|-----------|-------|
| Accuracy | Balanced classification | Misleading on imbalanced data |
| AUC-ROC | Binary classification | Threshold-independent, robust |
| Log Loss | Probability predictions | Penalizes confident wrong predictions |
| F1 Score | Imbalanced classification | Harmonic mean of precision/recall |
| RMSE | Regression | Penalizes large errors heavily |
| MAE | Regression | More robust to outliers than RMSE |
| RMSLE | Regression (positive values) | Good for exponential distributions |
| mAP | Object detection | Mean Average Precision |
| NDCG | Ranking/recommender | Normalized Discounted Cumulative Gain |
| CER/WER | Speech/OCR | Character/Word Error Rate |

---

## Key Resources

| Resource | What It Is | Link |
|----------|------------|------|
| Kaggle Learn | Free micro-courses | [kaggle.com/learn](https://kaggle.com/learn) |
| Kaggle Grandmaster interviews | Strategy insights | [Kaggle Blog](https://medium.com/kaggle-blog) |
| ML Competitions Reddit | Community | [r/mlcompetitions](https://reddit.com/r/mlcompetitions) |
| Top Solutions GitHub | Solution codebases | [github.com/topics/kaggle](https://github.com/topics/kaggle) |
| Winning Solutions List | Curated top solutions | [kagglesolutions.com](http://kagglesolutions.com) |
| DrivenData Blog | Social good ML | [blog.drivendata.org](https://blog.drivendata.org) |

---

## Contents of This Section

| File | Description |
|------|-------------|
| [README.md](README.md) | You are here — competition overview and getting started |
| [kaggle_guide.md](kaggle_guide.md) | Deep-dive Kaggle guide: setup, strategy, tricks |
| [winning_strategies.md](winning_strategies.md) | Meta-guide: ensembling, feature engineering, Optuna |
| [past_competitions.md](past_competitions.md) | 15+ landmark competitions worth studying |

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [Career Guide](../08_Career_Guide/) | [Main README](../README.md) | [Learning Roadmaps](../00_Overview/learning_roadmaps.md) |