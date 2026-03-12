<div align="center">

# The Definitive Kaggle Guide

### From account creation to Grandmaster strategy

</div>

---

## 1. Account Setup

1. Go to [kaggle.com](https://kaggle.com) and sign up (Google login recommended)
2. Complete your profile — add a bio, link your GitHub
3. Verify your phone number (required to use free GPU/TPU compute)
4. Browse [Kaggle Learn](https://kaggle.com/learn) for free micro-courses (Python, ML, SQL)

---

## 2. Kaggle API Setup

The Kaggle API lets you download competition data, submit predictions, and manage datasets from your terminal.

### Installation

```bash
pip install kaggle
```

### Authentication

1. Go to [kaggle.com/settings](https://kaggle.com/settings) → API → **Create New Token**
2. This downloads `kaggle.json`
3. Place it at:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
4. Set permissions (Linux/macOS only):

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Common API Commands

```bash
# List active competitions
kaggle competitions list

# Download competition data
kaggle competitions download -c titanic
kaggle competitions download -c titanic -p /path/to/folder

# Unzip
unzip titanic.zip -d ./data/

# Submit a prediction file
kaggle competitions submit -c titanic -f submission.csv -m "XGBoost baseline v1"

# Check your submission history
kaggle competitions submissions -c titanic

# List public notebooks (kernels) for a competition
kaggle kernels list --competition titanic --sort-by voteCount

# Download a specific public notebook
kaggle kernels pull username/notebook-name -p ./notebooks/ --wp
```

---

## 3. Notebook Environment Tips

### Kaggle Notebooks (free, browser-based)

| Resource | Limit |
|----------|-------|
| CPU | 2 cores, 13 GB RAM |
| GPU (T4 x2) | 30 hours/week |
| TPU (v3-8) | 20 hours/week |
| Disk | 20 GB |
| Internet | Can be enabled for competitions that allow it |

**Pro tips for Kaggle notebooks:**

```python
# Check if you're running on Kaggle
import os
ON_KAGGLE = os.path.exists('/kaggle/input')

# Standard path setup
DATA_DIR  = '/kaggle/input/competition-name/' if ON_KAGGLE else './data/'
OUTPUT_DIR = '/kaggle/working/' if ON_KAGGLE else './output/'
MODEL_DIR  = '/kaggle/working/models/' if ON_KAGGLE else './models/'
os.makedirs(MODEL_DIR, exist_ok=True)

# Disable GPU memory growth warnings (TensorFlow)
import tensorflow as tf
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True
)

# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

**Save and load intermediate results to avoid re-computation:**

```python
import pickle

# Save
with open(f'{OUTPUT_DIR}oof_predictions.pkl', 'wb') as f:
    pickle.dump(oof_preds, f)

# Load
with open(f'{OUTPUT_DIR}oof_predictions.pkl', 'rb') as f:
    oof_preds = pickle.load(f)
```

---

## 4. Competition Strategy by Type

### 4a. Tabular Competitions

The bread and butter of Kaggle. 80% of your time should be on feature engineering.

**Toolkit:**
- LightGBM, XGBoost, CatBoost (gradient boosting trio)
- Optuna for hyperparameter tuning
- sklearn for preprocessing and cross-validation

**Strategy:**
1. Start with LightGBM as baseline (fastest iteration)
2. Add feature engineering aggressively
3. CatBoost excels with categorical features
4. Ensemble all three at the end
5. Neural networks (TabNet, SAINT) for final boost

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np

def train_lgbm(X, y, X_test, params=None, n_folds=5):
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1,
        }

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    feature_importance = np.zeros(X.shape[1])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval   = lgb.Dataset(X_val,   label=y_val)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(500)
            ]
        )

        oof_preds[val_idx]  = model.predict(X_val)
        test_preds          += model.predict(X_test) / n_folds
        feature_importance  += model.feature_importance('gain') / n_folds

        from sklearn.metrics import roc_auc_score
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

    from sklearn.metrics import roc_auc_score
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall OOF AUC: {overall_auc:.4f}")

    return oof_preds, test_preds, feature_importance
```

---

### 4b. NLP Competitions

Dominated by transformer-based models since 2018.

**Toolkit:**
- HuggingFace Transformers
- Sentence-Transformers for embeddings
- `deberta-v3-large` is often the best starting point

**Strategy:**
1. Start with a pretrained base model (DeBERTa, RoBERTa)
2. Fine-tune with careful learning rate scheduling
3. Use gradient checkpointing to fit larger models
4. Ensemble diverse model architectures
5. Pseudo-labeling for unlabeled test data

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding='max_length'
    )

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    report_to="none",
)
```

---

### 4c. Computer Vision Competitions

**Toolkit:**
- PyTorch + timm (Py**T**orch **Im**age **M**odels)
- Albumentations for augmentation
- TIMM models: `convnext_base`, `vit_base_patch16_224`, `efficientnet_b4`

**Strategy:**
1. Start with a mid-size EfficientNet or ConvNeXt
2. Heavy augmentation is critical (Albumentations)
3. Test-Time Augmentation (TTA) always helps
4. Mixup/CutMix during training
5. Pseudo-labeling if test set is large

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

---

### 4d. Time Series Competitions

**Toolkit:**
- LightGBM with lag features (still the dominant approach)
- NeuralForecast (NHITS, N-BEATS)
- statsforecast for classical methods

**Strategy:**
1. Lag features are king — create many lags
2. Rolling statistics (mean, std, min, max)
3. Respect the temporal split — **never** use future data in your features
4. External data (weather, holidays, economic indicators) often matters
5. Hierarchical reconciliation for multi-level forecasting

```python
def create_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    """Create lag and rolling features for time series."""
    df = df.sort_values('date').copy()

    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby('id')[target_col].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'{target_col}_roll_mean_{window}'] = (
            df.groupby('id')[target_col]
            .shift(1)
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f'{target_col}_roll_std_{window}'] = (
            df.groupby('id')[target_col]
            .shift(1)
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # Date features
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month']       = pd.to_datetime(df['date']).dt.month
    df['quarter']     = pd.to_datetime(df['date']).dt.quarter
    df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)

    return df
```

---

## 5. Medal Strategy

Kaggle medals are awarded to top-finishing teams:

| Medal | Classification | Regression |
|-------|---------------|------------|
| Bronze | Top 40% | Top 40% |
| Silver | Top 20% | Top 20% |
| Gold | Top 10% (min 10 teams) | Top 10% (min 10 teams) |

**Progression path:**
1. **First 5 competitions:** Focus on learning, not medals. Read discussions, study notebooks.
2. **Competitions 5–15:** Aim for top 50%. Get your first bronze.
3. **Competitions 15–30:** Target silver medals. Focus on ensembling.
4. **30+ competitions:** Target gold. Specialize in 1–2 competition types.

**Rank progression:**
- Novice → Contributor → Expert → Master → Grandmaster
- Each rank has requirements based on medals earned

---

## 6. Top Tips from Kaggle Grandmasters

> "The most important thing is a reliable cross-validation setup. If your CV doesn't correlate with the leaderboard, you're flying blind." — **Chris Deotte, Kaggle Grandmaster**

> "Feature engineering beats model complexity every time for tabular data. Spend 70% of your time there." — **Jean-François Puget (CPMP)**

> "Read all the discussion posts from Day 1. Often a single insight from the community saves you a week of wrong directions." — **Bojan Tunguz**

> "Solo is fine for learning. But for medals, team up. Two solid competitors can combine OOF predictions in hours to jump 50+ ranks." — **Abhishek Thakur**

> "Don't trust the public leaderboard too much. Always trust your CV more — especially late in the competition when people overfit the public LB." — **Dmytro Danevskyi**

---

## 7. 10 Must-Know Kaggle Tricks

### Trick 1: Stratified K-Fold Cross Validation

```python
from sklearn.model_selection import StratifiedKFold

# Always use stratified splits for classification to maintain class balance
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # train model on train_idx, validate on val_idx
    pass
```

### Trick 2: Adversarial Validation (Check Train/Test Similarity)

```python
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# If this AUC is > 0.9, train and test are very different distributions!
train['is_test'] = 0
test['is_test']  = 1
combined = pd.concat([train, test], ignore_index=True)

X_adv = combined.drop('is_test', axis=1).select_dtypes(include='number').fillna(-999)
y_adv = combined['is_test']

clf = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
adv_auc = cross_val_score(clf, X_adv, y_adv, cv=5, scoring='roc_auc').mean()
print(f"Adversarial AUC: {adv_auc:.4f}")
# AUC ~0.5 = good (train and test look similar)
# AUC ~0.9+ = bad (train and test are very different)
```

### Trick 3: Target Encoding (With Leak Prevention)

```python
from sklearn.model_selection import KFold
import numpy as np

def target_encode(train: pd.DataFrame, test: pd.DataFrame,
                   col: str, target: str, n_folds: int = 5) -> tuple:
    """Target encoding with cross-validation to prevent leakage."""
    train_encoded = train[col].copy().astype(float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(train):
        means = train.iloc[tr_idx].groupby(col)[target].mean()
        train_encoded.iloc[val_idx] = train.iloc[val_idx][col].map(means)

    # Fill NaN with global mean
    global_mean = train[target].mean()
    train_encoded.fillna(global_mean, inplace=True)

    # Test uses all training data
    test_encoded = test[col].map(train.groupby(col)[target].mean()).fillna(global_mean)

    return train_encoded, test_encoded
```

### Trick 4: Pseudo-Labeling

```python
def pseudo_label(model, X_train, y_train, X_test, threshold: float = 0.95):
    """Add high-confidence test predictions as training data."""
    model.fit(X_train, y_train)
    test_probs = model.predict_proba(X_test)

    # Keep only high-confidence predictions
    confident_mask = test_probs.max(axis=1) > threshold
    confident_X    = X_test[confident_mask]
    confident_y    = test_probs[confident_mask].argmax(axis=1)

    print(f"Pseudo-labeled {confident_mask.sum()} samples out of {len(X_test)}")

    # Combine with original training data
    X_combined = pd.concat([X_train, confident_X], ignore_index=True)
    y_combined = np.concatenate([y_train, confident_y])

    return X_combined, y_combined
```

### Trick 5: Test-Time Augmentation (TTA) for CV

```python
import torch
import torch.nn.functional as F

def tta_predict(model, images: torch.Tensor, n_augments: int = 5) -> torch.Tensor:
    """Average predictions over multiple augmented versions of test images."""
    model.eval()
    predictions = []

    with torch.no_grad():
        # Original
        predictions.append(F.softmax(model(images), dim=1))

        # Horizontal flip
        predictions.append(F.softmax(model(torch.flip(images, dims=[3])), dim=1))

        # Vertical flip
        predictions.append(F.softmax(model(torch.flip(images, dims=[2])), dim=1))

        # 90° rotations
        for k in range(1, 4):
            predictions.append(F.softmax(model(torch.rot90(images, k, dims=[2, 3])), dim=1))

    return torch.stack(predictions).mean(dim=0)
```

### Trick 6: Stacking (Model Ensembling)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

def build_stacking_ensemble(base_oof_preds: dict, test_preds: dict, y_train):
    """Stack OOF predictions as meta-features."""
    # Build meta-feature matrix from OOF predictions
    meta_train = np.column_stack(list(base_oof_preds.values()))
    meta_test  = np.column_stack(list(test_preds.values()))

    # Train meta-learner (simple logistic regression works well)
    meta_model = LogisticRegression(C=1.0, max_iter=1000)
    meta_model.fit(meta_train, y_train)

    final_preds = meta_model.predict_proba(meta_test)[:, 1]
    return final_preds, meta_model
```

### Trick 7: Optuna for Hyperparameter Tuning

```python
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)
print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Trick 8: Feature Selection with SHAP

```python
import shap
import lightgbm as lgb

# Fit model
model = lgb.LGBMClassifier(n_estimators=500, verbose=-1)
model.fit(X_train, y_train)

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Get mean absolute SHAP values per feature
feature_importance_shap = pd.DataFrame({
    'feature': X_train.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

# Drop low-importance features
low_importance_features = feature_importance_shap[
    feature_importance_shap['importance'] < 0.001
]['feature'].tolist()

print(f"Dropping {len(low_importance_features)} low-importance features")
X_train_selected = X_train.drop(columns=low_importance_features)
```

### Trick 9: Rank-Based Ensemble

```python
from scipy.stats import rankdata

def rank_average(predictions: list[np.ndarray]) -> np.ndarray:
    """
    Rank-based averaging — more robust than simple averaging
    when models have different score scales.
    """
    ranked = [rankdata(p) / len(p) for p in predictions]
    return np.mean(ranked, axis=0)

# Example: combine 3 model predictions
final = rank_average([lgb_preds, xgb_preds, cat_preds])
```

### Trick 10: Stratified Group K-Fold (for grouped data)

```python
from sklearn.model_selection import StratifiedGroupKFold

# When you have groups (e.g., same patient, same user) that must not
# span train and validation sets
sgkf = StratifiedGroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(
    sgkf.split(X, y, groups=df['patient_id'])
):
    # train_idx and val_idx will never share a patient_id
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
```

---

## Resources

| Resource | Link |
|----------|------|
| Kaggle Official Docs | [kaggle.com/docs](https://www.kaggle.com/docs) |
| Kaggle API GitHub | [github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api) |
| Kaggle Grandmaster Notebooks | [kaggle.com/notebooks](https://kaggle.com/notebooks?sortBy=voteCount&language=Python) |
| "Approaching Almost Any ML Problem" book | [github.com/abhishekkrthakur/approachingalmost](https://github.com/abhishekkrthakur/approachingalmost) |
| H2O AutoML | [docs.h2o.ai](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) |

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [Competitions Overview](README.md) | [Main README](../README.md) | [Winning Strategies](winning_strategies.md) |