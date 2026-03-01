# Data Scientist — Intermediate Phase

**Goal:** Build, evaluate, and improve machine learning models for real-world tabular, text, and time-series problems.

**Duration:** 3–4 months at 10–15 hrs/week
**Prerequisites:** Completed beginner phase or equivalent skills

---

## Curriculum Overview

```
Week 1–2   → ML Fundamentals (theory + intuition)
Week 3–4   → Supervised Learning Algorithms Deep Dive
Week 5–6   → Feature Engineering (the #1 lever)
Week 7–8   → Model Evaluation & Validation
Week 9–10  → Ensemble Methods (XGBoost, LightGBM, Stacking)
Week 11–12 → Unsupervised Learning (Clustering, Dimensionality Reduction)
Week 13–14 → sklearn Pipelines & Reproducible ML Workflows
```

---

## Week 1–2: ML Fundamentals

### The Machine Learning Landscape

```
MACHINE LEARNING
├── Supervised Learning (labeled data)
│   ├── Classification (discrete output)
│   │   ├── Binary: Spam/Not spam, Fraud/Not fraud
│   │   └── Multi-class: Image category, sentiment (pos/neg/neutral)
│   └── Regression (continuous output)
│       └── House prices, stock forecasts, demand prediction
│
├── Unsupervised Learning (unlabeled data)
│   ├── Clustering (K-means, DBSCAN, Hierarchical)
│   ├── Dimensionality Reduction (PCA, UMAP, t-SNE)
│   └── Anomaly Detection (Isolation Forest, Autoencoders)
│
├── Semi-supervised (small labeled + large unlabeled)
│   └── Label propagation, self-training
│
└── Reinforcement Learning (agent + environment + reward)
    └── Game playing, robotics, recommendation systems
```

### The Bias-Variance Tradeoff (Visualized)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(42)

# True function: y = sin(x) + noise
x = np.linspace(0, 2*np.pi, 30)
y = np.sin(x) + np.random.normal(0, 0.3, len(x))

x_plot = np.linspace(0, 2*np.pi, 300).reshape(-1, 1)
x_fit = x.reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, degree, title in zip(axes, [1, 4, 20], ["Underfit (degree=1)", "Just Right (degree=4)", "Overfit (degree=20)"]):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_fit, y)
    y_pred = model.predict(x_plot)

    ax.scatter(x, y, color="black", s=20, zorder=5, label="Training data")
    ax.plot(x_plot, np.sin(x_plot), "g--", alpha=0.5, label="True function")
    ax.plot(x_plot, y_pred, "r-", linewidth=2, label=f"Model (degree={degree})")
    ax.set_ylim(-3, 3)
    ax.set_title(title)
    ax.legend(fontsize=8)

    # Annotate bias vs variance
    if degree == 1:
        ax.text(3, 2, "HIGH BIAS\nLow Variance\n(Underfitting)", ha="center", fontsize=9, color="red",
                bbox=dict(boxstyle="round", facecolor="mistyrose"))
    elif degree == 20:
        ax.text(3, 2, "Low Bias\nHIGH VARIANCE\n(Overfitting)", ha="center", fontsize=9, color="red",
                bbox=dict(boxstyle="round", facecolor="mistyrose"))
    else:
        ax.text(3, 2, "Balanced Bias\n& Variance\n✅ Good!", ha="center", fontsize=9, color="green",
                bbox=dict(boxstyle="round", facecolor="honeydew"))

plt.suptitle("The Bias-Variance Tradeoff", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

### Linear Regression: From Scratch

```python
import numpy as np

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using gradient descent.
    This is how every ML algorithm fundamentally works.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            # Forward pass: make predictions
            y_pred = X @ self.weights + self.bias  # Matrix multiplication

            # Compute loss (Mean Squared Error)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # Backward pass: compute gradients
            dw = (2 / n_samples) * X.T @ (y_pred - y)  # Gradient w.r.t. weights
            db = (2 / n_samples) * np.sum(y_pred - y)   # Gradient w.r.t. bias

            # Update parameters (gradient descent step)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot  # R² score

# Test it
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Our implementation
model_scratch = LinearRegressionScratch(learning_rate=0.1, n_iterations=500)
model_scratch.fit(X_train_scaled, y_train)
print(f"Our R²:     {model_scratch.score(X_test_scaled, y_test):.4f}")

# sklearn (should be very similar!)
model_sklearn = LinearRegression()
model_sklearn.fit(X_train_scaled, y_train)
print(f"sklearn R²: {model_sklearn.score(X_test_scaled, y_test):.4f}")

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(model_scratch.loss_history)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Iterations")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Week 3–4: Supervised Learning Algorithms

### The Algorithm Cheat Sheet

| Algorithm | Type | When to use | Pros | Cons |
|-----------|------|-------------|------|------|
| Linear/Logistic Regression | Linear | Baseline, interpretability required | Fast, interpretable | Assumes linearity |
| Decision Tree | Tree | Interpretability, mixed features | Easy to explain | Overfits easily |
| Random Forest | Ensemble | General purpose | Robust, handles missing | Slower, less interpretable |
| XGBoost/LightGBM | Boosting | Tabular data competitions | Very accurate | Needs tuning |
| SVM | Kernel | High-dimensional, small data | Effective in high dim | Slow on large data |
| KNN | Instance-based | Simple baseline | No training phase | Slow at inference |
| Naive Bayes | Probabilistic | Text classification, fast baseline | Very fast | Strong independence assumption |
| Neural Network | Deep Learning | Images, text, complex patterns | Extremely flexible | Needs lots of data |

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np

# Generate a realistic classification problem
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=10,
    n_redundant=5, n_clusters_per_class=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# Compare Multiple Algorithms
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(probability=True, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

results = []
for name, model in models.items():
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="roc_auc")
    model.fit(X_train_s, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])

    results.append({
        "Model": name,
        "CV AUC (mean)": cv_scores.mean(),
        "CV AUC (std)": cv_scores.std(),
        "Test AUC": test_auc
    })

results_df = pd.DataFrame(results).sort_values("Test AUC", ascending=False)
print(results_df.round(4).to_string(index=False))
```

---

## Week 5–6: Feature Engineering

Feature engineering is the most impactful skill in data science. Better features beat better algorithms every time.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# ============================================================
# Feature Engineering Techniques
# ============================================================

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# ── 1. EXTRACT INFORMATION FROM EXISTING FEATURES ──────────
# Title from Name (very predictive on Titanic!)
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")
title_map = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
df["Title"] = df["Title"].map(title_map).fillna("Rare")

# Family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # +1 for self
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Deck from Cabin
df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

# ── 2. BINNING CONTINUOUS FEATURES ─────────────────────────
# Age groups (domain knowledge: children, young adults, adults, elderly)
df["AgeGroup"] = pd.cut(df["Age"],
                         bins=[0, 12, 18, 35, 60, 200],
                         labels=["Child", "Teen", "YoungAdult", "Adult", "Elder"],
                         right=False)
df["AgeGroup"].fillna("Unknown", inplace=True)

# Fare quartiles
df["FareBand"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Mid", "High", "VeryHigh"])

# ── 3. INTERACTION FEATURES ────────────────────────────────
# Some combinations are more predictive than either feature alone
df["Pclass_Sex"] = df["Pclass"].astype(str) + "_" + df["Sex"]
df["Age_Pclass"] = df["Age"] * df["Pclass"]  # Continuous interaction

# ── 4. AGGREGATION FEATURES (group-level stats) ────────────
# Average survival by Title, Pclass
group_survival = df.groupby("Title")["Survived"].transform("mean")
df["TitleSurvivalRate"] = group_survival

group_fare = df.groupby("Pclass")["Fare"].transform("mean")
df["FareVsClassAvg"] = df["Fare"] - group_fare  # Is this person paying above average for their class?

# ── 5. ENCODING CATEGORICAL VARIABLES ──────────────────────
# One-hot encoding (for nominal categories with no ordering)
df = pd.get_dummies(df, columns=["Sex", "Embarked", "Title", "AgeGroup", "FareBand",
                                   "Pclass_Sex", "Deck"], drop_first=True)

# ── 6. HANDLE REMAINING MISSING VALUES ─────────────────────
df["Age"].fillna(df["Age"].median(), inplace=True)

# ── 7. SELECT FEATURES ─────────────────────────────────────
feature_cols = [col for col in df.columns
                if col not in ["Survived", "PassengerId", "Name", "Ticket", "Cabin"]]
X = df[feature_cols]
y = df["Survived"]

print(f"Original features: 11")
print(f"Engineered features: {len(feature_cols)}")
print(f"\nTop features:\n{X.head()}")
```

### Feature Selection

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
import matplotlib.pyplot as plt

# ── Method 1: Feature Importance from Tree Models ──────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X.fillna(0), y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
top_20 = importances.nlargest(20)

plt.figure(figsize=(10, 6))
top_20.plot(kind="barh", color="steelblue")
plt.xlabel("Feature Importance")
plt.title("Top 20 Features by Random Forest Importance")
plt.tight_layout()
plt.show()

# ── Method 2: Mutual Information ───────────────────────────
mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)
mi_df = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features by mutual information:")
print(mi_df.head(10))

# ── Method 3: Correlation with target ──────────────────────
correlations = X.fillna(0).corrwith(y).abs().sort_values(ascending=False)
print("\nTop 10 features by correlation with target:")
print(correlations.head(10))

# ── Method 4: Recursive Feature Elimination (RFE) ──────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.fillna(0))

rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
rfe.fit(X_scaled, y)
selected_features = X.columns[rfe.support_].tolist()
print(f"\nRFE selected {len(selected_features)} features: {selected_features}")
```

---

## Week 7–8: Model Evaluation & Validation

### Cross-Validation Strategies

```python
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold,
    LeaveOneOut, TimeSeriesSplit, cross_validate
)
import numpy as np

# ── Standard K-Fold (for regression) ───────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── Stratified K-Fold (for classification — preserves class ratio!) ──
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Time Series Split (for temporal data — NO leakage!) ────
tscv = TimeSeriesSplit(n_splits=5)

# ── Cross-validate with multiple metrics ───────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, random_state=42)

cv_results = cross_validate(
    RandomForestClassifier(random_state=42),
    X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring={
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall"
    },
    return_train_score=True
)

for metric in ["accuracy", "roc_auc", "f1"]:
    train = cv_results[f"train_{metric}"]
    test = cv_results[f"test_{metric}"]
    print(f"{metric:12s}: Train {train.mean():.3f}±{train.std():.3f} | Test {test.mean():.3f}±{test.std():.3f}")
```

### Classification Metrics Deep Dive

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
y_proba = np.array([0.9, 0.1, 0.85, 0.4, 0.2, 0.75, 0.6, 0.15, 0.95, 0.05, 0.8, 0.35, 0.25, 0.7, 0.88])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ── 1. Confusion Matrix ─────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Predicted Neg", "Predicted Pos"],
            yticklabels=["Actual Neg", "Actual Pos"])
axes[0].set_title("Confusion Matrix")

tn, fp, fn, tp = cm.ravel()
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"Precision = TP/(TP+FP) = {tp}/{tp+fp} = {tp/(tp+fp):.3f}")
print(f"Recall    = TP/(TP+FN) = {tp}/{tp+fn} = {tp/(tp+fn):.3f}")
print(f"F1        = 2*P*R/(P+R) = {2*tp/(2*tp+fp+fn):.3f}")

# ── 2. ROC Curve ────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
auc = roc_auc_score(y_true, y_proba)
axes[1].plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC={auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
axes[1].fill_between(fpr, tpr, alpha=0.1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

# ── 3. Precision-Recall Curve (better for imbalanced!) ─────
precision, recall, _ = precision_recall_curve(y_true, y_proba)
ap = average_precision_score(y_true, y_proba)
axes[2].plot(recall, precision, "g-", linewidth=2, label=f"PR Curve (AP={ap:.3f})")
axes[2].axhline(y_true.mean(), color="k", linestyle="--", alpha=0.5, label=f"Baseline ({y_true.mean():.2f})")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].set_title("Precision-Recall Curve")
axes[2].legend()

plt.tight_layout()
plt.show()
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

# ── Grid Search (exhaustive, slow but thorough) ─────────────
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5, scoring="roc_auc",
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# ── Random Search (faster — sample random combinations) ─────
param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": [5, 10, 20, None],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50, cv=5, scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1
)
random_search.fit(X_train, y_train)
print(f"\nRandom Search best params: {random_search.best_params_}")
print(f"Random Search best CV AUC: {random_search.best_score_:.4f}")

# ── Optuna (Bayesian optimization — best approach) ──────────
# pip install optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    model = RandomForestClassifier(random_state=42, **params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(f"\nOptuna best params: {study.best_params}")
print(f"Optuna best AUC: {study.best_value:.4f}")
```

---

## Week 9–10: Ensemble Methods

```python
# XGBoost — the king of tabular ML competitions
import xgboost as xgb
from sklearn.metrics import roc_auc_score

model_xgb = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,   # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    early_stopping_rounds=50,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

print(f"XGBoost AUC: {roc_auc_score(y_test, model_xgb.predict_proba(X_test)[:, 1]):.4f}")
print(f"Best iteration: {model_xgb.best_iteration}")

# Plot feature importance
xgb.plot_importance(model_xgb, max_num_features=15, figsize=(8, 6))
plt.show()
```

---

## Week 11–12: Unsupervised Learning

```python
# K-Means Clustering with proper evaluation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=1000, n_features=10, centers=4, random_state=42)

# Find optimal K using Elbow Method + Silhouette
k_range = range(2, 11)
inertias, silhouette_scores = [], []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(k_range, inertias, "bo-")
axes[0].set_xlabel("K"), axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method — look for the 'elbow'")

axes[1].plot(k_range, silhouette_scores, "go-")
axes[1].set_xlabel("K"), axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score — higher is better")
plt.show()

# PCA — Dimensionality Reduction
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Color by K-means cluster labels
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init="auto")
labels = kmeans_final.fit_predict(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", alpha=0.6, s=20)
plt.colorbar(scatter)
plt.title(f"Clusters visualized in PCA space\n(explains {pca.explained_variance_ratio_.sum():.1%} of variance)")
plt.show()
```

---

## Week 13–14: sklearn Pipelines

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# ============================================================
# A production-quality sklearn Pipeline
# ============================================================

# Define feature types
numeric_features = ["age", "income", "score"]
categorical_features = ["occupation", "education", "marital_status"]

# Numeric transformer: impute → scale
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical transformer: impute → encode
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
], remainder="drop")

# Full pipeline: preprocess → model
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(n_estimators=200, random_state=42))
])

# Train
full_pipeline.fit(X_train, y_train)

# Evaluate
print(f"Pipeline AUC: {roc_auc_score(y_test, full_pipeline.predict_proba(X_test)[:, 1]):.4f}")

# Save the ENTIRE pipeline (including preprocessor state!)
joblib.dump(full_pipeline, "model_pipeline.pkl")

# Load and predict — zero preprocessing required!
loaded_pipeline = joblib.load("model_pipeline.pkl")
predictions = loaded_pipeline.predict(X_test)
```

---

## Intermediate Phase Skills Checklist

- [ ] Understand bias-variance tradeoff and can diagnose over/underfitting
- [ ] Have implemented linear regression from scratch with gradient descent
- [ ] Can compare 5+ ML algorithms for the same problem
- [ ] Engineer 10+ new features from raw data (dates, text, interactions)
- [ ] Can use cross-validation correctly (stratified for classification, time-series for temporal)
- [ ] Understand precision vs recall and when to use each
- [ ] Can plot and interpret ROC curves and PR curves
- [ ] Have tuned hyperparameters with RandomSearchCV and Optuna
- [ ] Have used XGBoost or LightGBM on a real problem
- [ ] Build a full sklearn Pipeline with ColumnTransformer
- [ ] Have entered at least one Kaggle competition

**Next:** [Advanced Phase →](../advanced/)