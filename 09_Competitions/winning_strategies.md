<div align="center">

# Winning Strategies for ML Competitions

### The meta-guide to finishing in the top 10%

</div>

---

## Overview

Top competitors don't just know ML — they have a systematic process. This guide distills the patterns observed across thousands of winning solutions. Read this before starting any competition.

> **Key insight:** Most top solutions are not one magical model. They are a disciplined process of EDA → feature engineering → ensembling → not overfitting the leaderboard.

---

## 1. Feature Engineering Tricks by Data Type

### 1a. Tabular / Structured Data

```python
import pandas as pd
import numpy as np
from itertools import combinations

def engineer_tabular_features(df: pd.DataFrame,
                               numeric_cols: list[str],
                               cat_cols: list[str]) -> pd.DataFrame:
    """Comprehensive tabular feature engineering."""
    df = df.copy()

    # --- Numeric interactions ---
    for col_a, col_b in combinations(numeric_cols[:5], 2):  # limit combinations
        df[f'{col_a}_x_{col_b}'] = df[col_a] * df[col_b]
        df[f'{col_a}_div_{col_b}'] = df[col_a] / (df[col_b] + 1e-9)
        df[f'{col_a}_minus_{col_b}'] = df[col_a] - df[col_b]

    # --- Aggregation features ---
    for group_col in cat_cols:
        for agg_col in numeric_cols:
            grouped = df.groupby(group_col)[agg_col]
            df[f'{agg_col}_mean_by_{group_col}'] = df[group_col].map(grouped.mean())
            df[f'{agg_col}_std_by_{group_col}']  = df[group_col].map(grouped.std())
            df[f'{agg_col}_max_by_{group_col}']  = df[group_col].map(grouped.max())
            df[f'{agg_col}_min_by_{group_col}']  = df[group_col].map(grouped.min())
            # Deviation from group mean
            df[f'{agg_col}_dev_from_{group_col}_mean'] = (
                df[agg_col] - df[f'{agg_col}_mean_by_{group_col}']
            )

    # --- Polynomial features (selective) ---
    for col in numeric_cols:
        df[f'{col}_sq'] = df[col] ** 2
        df[f'{col}_log1p'] = np.log1p(np.abs(df[col]))
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))

    # --- Frequency encoding ---
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

    # --- Count encoding ---
    for col in cat_cols:
        counts = df[col].value_counts()
        df[f'{col}_count'] = df[col].map(counts)

    return df
```

### 1b. NLP / Text Data

```python
import re
from collections import Counter

def engineer_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Statistical text features without deep learning."""
    df = df.copy()
    text = df[text_col].fillna('')

    # Basic statistics
    df['char_count']       = text.str.len()
    df['word_count']       = text.str.split().str.len()
    df['sentence_count']   = text.str.count(r'[.!?]') + 1
    df['avg_word_length']  = df['char_count'] / (df['word_count'] + 1)
    df['unique_word_ratio'] = text.apply(lambda x: len(set(x.lower().split())) / (len(x.split()) + 1))

    # Punctuation and caps
    df['exclamation_count'] = text.str.count('!')
    df['question_count']    = text.str.count(r'\?')
    df['caps_ratio']        = text.apply(lambda x: sum(c.isupper() for c in x) / (len(x) + 1))
    df['digit_count']       = text.str.count(r'\d')

    # Sentiment-adjacent features
    positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome'}
    negative_words = {'bad', 'terrible', 'hate', 'worst', 'poor', 'horrible', 'awful'}

    df['positive_word_count'] = text.str.lower().apply(
        lambda x: sum(w in x.split() for w in positive_words)
    )
    df['negative_word_count'] = text.str.lower().apply(
        lambda x: sum(w in x.split() for w in negative_words)
    )

    return df
```

### 1c. Time Series Data

```python
def engineer_ts_features(df: pd.DataFrame,
                           date_col: str,
                           value_col: str,
                           id_col: str | None = None) -> pd.DataFrame:
    """Comprehensive time series feature engineering."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Calendar features
    df['year']        = df[date_col].dt.year
    df['month']       = df[date_col].dt.month
    df['week']        = df[date_col].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['quarter']     = df[date_col].dt.quarter
    df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end']   = df[date_col].dt.is_month_end.astype(int)

    # Cyclical encoding (important for periodicity!)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Lag features (per group if id_col provided)
    group = df.groupby(id_col) if id_col else df.groupby(lambda _: 0)

    for lag in [1, 2, 3, 7, 14, 28]:
        df[f'{value_col}_lag_{lag}'] = group[value_col].shift(lag)

    # Rolling features
    for window in [7, 14, 30, 90]:
        shifted = group[value_col].shift(1)
        df[f'{value_col}_roll_mean_{window}'] = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'{value_col}_roll_std_{window}'] = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    return df
```

### 1d. Image / Computer Vision Data

For image competitions, feature engineering happens at the **augmentation** and **embedding** level:

```python
import timm
import torch

def extract_embeddings(image_paths: list, model_name: str = 'convnext_base.fb_in22k') -> np.ndarray:
    """Extract pretrained embeddings for use as tabular features."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))

    embeddings = []
    from PIL import Image

    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            emb = model(tensor).squeeze().numpy()
            embeddings.append(emb)

    return np.array(embeddings)
```

---

## 2. Model Selection Cheatsheet

| Data Type | First Try | Second Try | Deep Learning Option |
|-----------|-----------|------------|---------------------|
| Tabular (classification) | LightGBM | XGBoost + CatBoost | TabNet, SAINT |
| Tabular (regression) | LightGBM | Ridge + LightGBM ensemble | TabTransformer |
| NLP (classification) | DeBERTa-v3-base | RoBERTa-large | T5, GPT-2 |
| NLP (generation) | LLaMA-3 / Mistral | Fine-tuned GPT | Seq2Seq T5 |
| CV (classification) | EfficientNet-B4 | ConvNeXt-Base | ViT-Base |
| CV (detection) | YOLOv8 | DINO | Co-DETR |
| CV (segmentation) | SegFormer | SAM + finetune | Mask2Former |
| Time series (forecasting) | LightGBM + lags | N-BEATS / NHITS | PatchTST, iTransformer |
| Recommender | LightFM | XGBoost + features | Two-Tower NN |
| Graph | LightGBM + graph features | GraphSAGE | GAT, GNN |

---

## 3. Ensemble Methods with Code

### 3a. Simple Averaging

```python
def simple_average(predictions: dict[str, np.ndarray],
                    weights: dict[str, float] | None = None) -> np.ndarray:
    """Weighted or unweighted average of predictions."""
    if weights is None:
        weights = {k: 1.0 / len(predictions) for k in predictions}

    total_weight = sum(weights.values())
    weighted_sum = sum(pred * weights[name] for name, pred in predictions.items())
    return weighted_sum / total_weight

# Example
ensemble = simple_average({
    'lgb':  lgb_preds,
    'xgb':  xgb_preds,
    'cat':  cat_preds,
    'nn':   nn_preds,
}, weights={'lgb': 0.3, 'xgb': 0.3, 'cat': 0.25, 'nn': 0.15})
```

### 3b. Stacking with Out-of-Fold Predictions

```python
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

class StackingEnsemble:
    def __init__(self, base_models: list, meta_model=None, n_folds: int = 5):
        self.base_models = base_models
        self.meta_model  = meta_model or LogisticRegression()
        self.n_folds     = n_folds

    def fit_predict(self, X_train, y_train, X_test):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        oof_features  = np.zeros((len(X_train), len(self.base_models)))
        test_features = np.zeros((len(X_test), len(self.base_models)))

        for model_idx, model in enumerate(self.base_models):
            fold_test_preds = np.zeros((len(X_test), self.n_folds))

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr        = y_train[tr_idx]

                model.fit(X_tr, y_tr)

                oof_features[val_idx, model_idx] = (
                    model.predict_proba(X_val)[:, 1]
                    if hasattr(model, 'predict_proba')
                    else model.predict(X_val)
                )
                fold_test_preds[:, fold] = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, 'predict_proba')
                    else model.predict(X_test)
                )

            test_features[:, model_idx] = fold_test_preds.mean(axis=1)

        # Train meta-learner on OOF features
        self.meta_model.fit(oof_features, y_train)
        return self.meta_model.predict_proba(test_features)[:, 1]
```

### 3c. Blending (Simpler Alternative to Stacking)

```python
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

def optimize_blend_weights(oof_preds: list[np.ndarray], y_true: np.ndarray) -> np.ndarray:
    """Find optimal blend weights by minimizing validation loss."""
    n_models = len(oof_preds)

    def neg_auc(weights):
        weights = np.array(weights)
        weights = np.abs(weights) / np.abs(weights).sum()  # normalize
        blended = sum(w * p for w, p in zip(weights, oof_preds))
        return -roc_auc_score(y_true, blended)

    initial_weights = [1.0 / n_models] * n_models
    bounds = [(0, 1)] * n_models

    result = minimize(neg_auc, initial_weights, method='SLSQP', bounds=bounds,
                      constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1})

    optimal_weights = np.abs(result.x) / np.abs(result.x).sum()
    print(f"Optimal weights: {optimal_weights}")
    print(f"Best blend AUC: {-result.fun:.4f}")

    return optimal_weights
```

---

## 4. Hyperparameter Tuning at Scale with Optuna

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def run_optuna_study(X_train, y_train, n_trials: int = 200, n_folds: int = 5):
    """Full Optuna hyperparameter search with pruning."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective':         'binary',
            'metric':            'auc',
            'verbosity':         -1,
            'boosting_type':     trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'n_estimators':      trial.suggest_int('n_estimators', 200, 3000),
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 16, 512),
            'max_depth':         trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'feature_fraction':  trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq':      trial.suggest_int('bagging_freq', 1, 10),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(X_train))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])

            oof[val_idx] = model.predict_proba(X_val)[:, 1]

            # Optuna pruning
            intermediate = roc_auc_score(y_train[val_idx], oof[val_idx])
            trial.report(intermediate, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return roc_auc_score(y_train, oof)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner()
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    print(f"\nBest trial: AUC = {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study
```

---

## 5. Common Pitfalls and Leakage Prevention

### The Most Dangerous Mistakes

| Mistake | Impact | How to Detect | How to Fix |
|---------|--------|---------------|------------|
| **Target leakage** | Artificially inflated CV score | Adversarial validation, feature importance | Remove leaky features |
| **Temporal leakage** | Model sees the future | Check that no feature uses future data | Shift all temporal features by 1+ period |
| **Group leakage** | Same entity in train and val | Check if IDs appear in both splits | Use GroupKFold |
| **Preprocessing leakage** | Scaling/encoding fits on all data | CV must fit preprocessing on train only | Use sklearn Pipelines |
| **LB overfitting** | Great on public LB, terrible private | Trust CV over LB, limit submissions | Hold out validation set never shown to LB |

### Detecting and Fixing Target Leakage

```python
def check_for_leakage(X_train: pd.DataFrame, y_train: pd.Series,
                       threshold: float = 0.95) -> list[str]:
    """Find features that are suspiciously well-correlated with target."""
    suspicious = []

    for col in X_train.select_dtypes(include='number').columns:
        # Correlation with target
        corr = abs(X_train[col].corr(y_train))

        if corr > threshold:
            suspicious.append((col, corr))
            print(f"SUSPICIOUS: {col} has correlation {corr:.3f} with target!")

    return suspicious

def safe_pipeline_example():
    """Show how to prevent preprocessing leakage with sklearn Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    import lightgbm as lgb

    # WRONG: fit scaler on all data, then cross-validate
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)  # LEAKS!
    # scores = cross_val_score(model, X_scaled, y, cv=5)

    # RIGHT: use Pipeline — preprocessing fits only on training fold
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', lgb.LGBMClassifier(verbose=-1))
    ])

    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    return scores
```

---

## 6. Post-Competition Analysis: How to Study Winning Solutions

The highest-leverage learning activity in competitive ML is studying winning solutions after a competition ends. Here is a systematic approach:

### Step 1: Find the Solution Write-ups

- **Kaggle Discussion tab:** Winners typically post write-ups. Sort by date (end of competition).
- **GitHub search:** `kaggle + competition-name + solution`
- **Kaggle Blog:** Major competition winners are often interviewed
- **ArXiv:** Research-track competitions often result in papers

### Step 2: Analyze the Solution Structure

For each top solution, document:

```markdown
## Solution Analysis Template

**Competition:** [Name] | **Rank:** [Rank] | **Score:** [Score]

### Data Preprocessing
- Missing value handling:
- Outlier treatment:
- Feature scaling:

### Feature Engineering (most important!)
- Key features created:
- External data used:
- Surprising insights:

### Model Architecture
- Primary model:
- Secondary models:
- Architecture choices:

### Validation Strategy
- CV scheme used:
- CV-LB correlation:

### Ensembling
- How many models:
- Ensemble method:
- Diversity strategy:

### Key Insight
- What made this solution special:
- What would I replicate:
```

### Step 3: Reproduce Key Elements

Don't just read — reproduce. Take 1–2 key ideas from a winning solution and implement them from scratch. This cements the learning far more than passive reading.

### Step 4: Track Your Learning

```python
# Maintain a personal competition notebook
competition_learnings = {
    "Titanic (2024 attempt)": {
        "rank": "Top 15%",
        "key_feature": "Title extraction from Name column",
        "what_worked": "Random Forest with 5-fold CV",
        "what_didnt": "GBDT without feature engineering underfit",
        "winning_insight": "Cabin deck letter was highly predictive",
        "next_time": "Better cabin feature engineering, try XGBoost",
    }
}
```

---

## Resources

| Resource | Link |
|----------|------|
| "Winning Solutions" Repository | [github.com/sokrypton/kaggle-solutions](https://github.com/kownse/kaggle_realtimewi) |
| MLWave Ensemble Guide | [mlwave.com](https://mlwave.com/kaggle-ensembling-guide/) |
| Optuna Documentation | [optuna.readthedocs.io](https://optuna.readthedocs.io) |
| "KAGGLE Book" by Bojan Tunguz | [packt.com](https://www.packtpub.com/product/the-kaggle-book/9781801817479) |

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [Kaggle Guide](kaggle_guide.md) | [Main README](../README.md) | [Past Competitions](past_competitions.md) |
