# Data Scientist — Advanced Phase

**Goal:** Master deep learning, NLP, causal inference, and Bayesian methods. Develop specialization and work at a senior level.

**Duration:** 4–6 months at 10–15 hrs/week
**Prerequisites:** Intermediate Phase complete, comfortable with PyTorch basics

---

## Curriculum Overview

```
Week 1–3   → Deep Learning Fundamentals (PyTorch)
Week 4–6   → NLP: From TF-IDF to Transformers
Week 7–9   → Time Series Analysis & Forecasting
Week 10–12 → A/B Testing, Causal Inference & Experiment Design
Week 13–15 → Bayesian Statistics in Practice
Week 16–18 → Model Interpretability (XAI/SHAP)
Week 19–21 → Capstone: End-to-End Specialization Project
```

---

## Week 1–3: Deep Learning with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ── Neural Network from Scratch ───────────────────────────────
class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for binary classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout_rate: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))  # Output: single logit
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def train_model(model, train_loader, val_loader, n_epochs=50, lr=1e-3):
    """Complete training loop with early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(n_epochs):
        # ── Training ──────────────────────────────────────────
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────────
        model.eval()
        val_losses, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                val_losses.append(criterion(logits, y_batch).item())
                probs = torch.sigmoid(logits).numpy()
                all_probs.extend(probs)
                all_labels.extend(y_batch.numpy())

        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_probs)
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if epoch % 10 == 0:
            lr_current = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | LR: {lr_current:.6f}")

        # ── Early Stopping ────────────────────────────────────
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    return history


# Usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=10000, n_features=30, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

train_dataset = TabularDataset(X_train_s, y_train)
val_dataset = TabularDataset(X_val_s, y_val)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

model = MLPClassifier(input_dim=30, hidden_dims=[256, 128, 64], dropout_rate=0.3)
history = train_model(model, train_loader, val_loader, n_epochs=100)
```

---

## Week 4–6: NLP — Modern Text Analysis

### From Bag of Words to BERT

```python
# ── Traditional NLP: TF-IDF + Logistic Regression ─────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import pandas as pd

# Load sentiment dataset (AG News or SST-2)
# Using synthetic example here
texts = [
    "This product is absolutely amazing! Best purchase ever!",
    "Terrible quality. Broke after one week. Do not buy!",
    "Okay product. Does what it says, nothing more.",
    "Outstanding customer service and fast shipping!",
    "Disappointed. Expected much better for the price.",
] * 200  # Scale up

labels = [1, 0, 1, 1, 0] * 200

# TF-IDF baseline
tfidf_pipeline = make_pipeline(
    TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True),
    LogisticRegression(C=1.0, max_iter=500)
)

from sklearn.model_selection import cross_val_score, train_test_split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels)
tfidf_pipeline.fit(X_train, y_train)
print("TF-IDF + LR:")
print(classification_report(y_test, tfidf_pipeline.predict(X_test)))

# ── Modern NLP: Hugging Face Transformers ────────────────────
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Zero-shot classification (no training needed!)
classifier = pipeline("zero-shot-classification",
                       model="facebook/bart-large-mnli",
                       device=0 if torch.cuda.is_available() else -1)

candidate_labels = ["positive", "negative", "neutral"]
for text in texts[:3]:
    result = classifier(text, candidate_labels)
    print(f"\nText: {text[:60]}...")
    print(f"Predicted: {result['labels'][0]} ({result['scores'][0]:.3f})")

# Sentiment analysis pipeline (pre-trained)
sentiment_pipeline = pipeline("sentiment-analysis",
                               model="distilbert-base-uncased-finetuned-sst-2-english")
results = sentiment_pipeline(texts[:5])
for text, result in zip(texts[:5], results):
    print(f"{result['label']:8s} ({result['score']:.3f}): {text[:60]}")

# ── Fine-tuning BERT ─────────────────────────────────────────
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

model_name = "distilbert-base-uncased"  # Small, fast BERT variant
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

# Create HuggingFace dataset
hf_dataset = Dataset.from_dict({"text": texts, "label": labels})
hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
tokenized = hf_dataset.map(tokenize_function, batched=True)
tokenized = tokenized.remove_columns(["text"])
tokenized.set_format("torch")

model_bert = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none",  # Set to "wandb" for tracking
)

trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
)
trainer.train()
print("Fine-tuning complete!")
```

---

## Week 7–9: Time Series Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# ── Time series decomposition ─────────────────────────────────
# Generate realistic retail sales data
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=156, freq="W")
trend = np.linspace(1000, 1500, len(dates))
seasonality = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)  # 52-week cycle
noise = np.random.normal(0, 50, len(dates))
sales = pd.Series(trend + seasonality + noise, index=dates, name="sales")

# Decompose: trend + seasonality + residual
decomposition = seasonal_decompose(sales, model="additive", period=52)
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for ax, component, name in zip(axes, [sales, decomposition.trend, decomposition.seasonal, decomposition.resid],
                                ["Original", "Trend", "Seasonal", "Residual"]):
    ax.plot(component, lw=1.5)
    ax.set_ylabel(name)
    ax.grid(True, alpha=0.3)
plt.suptitle("Time Series Decomposition", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ── Stationarity Test ─────────────────────────────────────────
result = adfuller(sales, autolag="AIC")
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print(f"Stationary: {'Yes' if result[1] < 0.05 else 'No (need to difference)'}")

# ── SARIMA: Statistical approach ─────────────────────────────
train = sales[:-13]
test = sales[-13:]

sarima_model = SARIMAX(
    train,
    order=(2, 1, 2),          # (p, d, q) — non-seasonal
    seasonal_order=(1, 1, 1, 52),  # (P, D, Q, S) — seasonal
    trend="c"
)
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

forecast = sarima_fit.forecast(steps=13)
mae_sarima = np.mean(np.abs(test - forecast))
print(f"\nSARIMA MAE: {mae_sarima:.2f}")

# ── Prophet: Facebook's forecasting library ───────────────────
# pip install prophet
from prophet import Prophet

# Prophet expects "ds" (date) and "y" (value) columns
df_prophet = sales.reset_index().rename(columns={"index": "ds", "sales": "y"})
df_train = df_prophet[:-13]
df_test = df_prophet[-13:]

model_prophet = Prophet(
    seasonality_mode="additive",
    yearly_seasonality=True,
    weekly_seasonality=False,
    changepoint_prior_scale=0.05,  # Flexibility of trend changepoints
)
model_prophet.fit(df_train)

future = model_prophet.make_future_dataframe(periods=13, freq="W")
forecast_prophet = model_prophet.predict(future)

# Plot forecast with components
model_prophet.plot(forecast_prophet)
plt.title("Prophet Forecast")
plt.show()

model_prophet.plot_components(forecast_prophet)
plt.show()

mae_prophet = np.mean(np.abs(df_test["y"].values - forecast_prophet["yhat"].tail(13).values))
print(f"Prophet MAE: {mae_prophet:.2f}")
```

---

## Week 10–12: A/B Testing & Causal Inference

```python
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# PART 1: Proper A/B Test Design
# ============================================================

def power_analysis(baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Calculate required sample size per group for a proportions test.

    Args:
        baseline_rate: Current conversion rate (e.g., 0.10 = 10%)
        mde: Minimum Detectable Effect — smallest relative lift you care about (e.g., 0.05 = 5%)
        alpha: Type I error rate (false positive rate, default 5%)
        power: 1 - Type II error rate (default 80%)
    """
    from statsmodels.stats.proportion import proportion_effectsize, zt_ind_solve_power

    treatment_rate = baseline_rate * (1 + mde)
    effect_size = proportion_effectsize(baseline_rate, treatment_rate)

    n = zt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,  # Equal group sizes
        alternative="two-sided"
    )
    return int(np.ceil(n))

# Example: 10% baseline, want to detect 10% relative lift (10% → 11%)
required_n = power_analysis(baseline_rate=0.10, mde=0.10)
print(f"Required sample size per group: {required_n:,}")
print(f"Total experiment size: {required_n*2:,}")
print(f"Days needed (assuming 10k users/day): {required_n*2/10000:.1f}")

# ============================================================
# PART 2: Run the Experiment
# ============================================================
np.random.seed(42)
n_per_group = required_n

# Simulate actual experiment results
control_conversions = np.random.binomial(1, 0.10, n_per_group)
treatment_conversions = np.random.binomial(1, 0.112, n_per_group)  # True lift = 12%

print(f"\nControl:   {control_conversions.mean():.4f} ({control_conversions.sum():,} conversions)")
print(f"Treatment: {treatment_conversions.mean():.4f} ({treatment_conversions.sum():,} conversions)")

# ============================================================
# PART 3: Statistical Analysis
# ============================================================
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

count = np.array([treatment_conversions.sum(), control_conversions.sum()])
nobs = np.array([n_per_group, n_per_group])

stat, pvalue = proportions_ztest(count, nobs)

absolute_lift = treatment_conversions.mean() - control_conversions.mean()
relative_lift = absolute_lift / control_conversions.mean()

ci_treatment = proportion_confint(treatment_conversions.sum(), n_per_group)
ci_control = proportion_confint(control_conversions.sum(), n_per_group)

print(f"\n{'='*50}")
print(f"EXPERIMENT RESULTS")
print(f"{'='*50}")
print(f"Z-statistic: {stat:.4f}")
print(f"P-value: {pvalue:.6f}")
print(f"Statistical significance: {'YES ✅' if pvalue < 0.05 else 'NO ❌'}")
print(f"\nEffect Size:")
print(f"  Absolute lift: {absolute_lift:+.4f} ({absolute_lift/control_conversions.mean()*100:+.1f}%)")
print(f"  Relative lift: {relative_lift:+.1%}")
print(f"\n95% Confidence Intervals:")
print(f"  Control:   [{ci_control[0]:.4f}, {ci_control[1]:.4f}]")
print(f"  Treatment: [{ci_treatment[0]:.4f}, {ci_treatment[1]:.4f}]")

# Business impact estimation
revenue_per_conversion = 50  # $50 average order value
monthly_users = 300000
monthly_revenue_increase = monthly_users * absolute_lift * revenue_per_conversion
print(f"\nBusiness Impact:")
print(f"  Monthly revenue increase: ${monthly_revenue_increase:,.0f}")

# ============================================================
# PART 4: Causal Inference (Observational Data)
# ============================================================
# When you can't run an A/B test, use causal inference

# Difference-in-Differences (DiD) — for policy changes
np.random.seed(42)
n = 1000

# Simulate: users in "treatment city" got feature, "control city" didn't
df = pd.DataFrame({
    "user_id": range(n),
    "treated": np.random.binomial(1, 0.5, n),
    "post": np.random.binomial(1, 0.5, n),
    "age": np.random.normal(35, 10, n),
    "income": np.random.normal(60000, 20000, n),
})

# Outcome model: revenue ~ treated + post + treated*post + controls
df["revenue"] = (
    500 + 100 * df["treated"] + 50 * df["post"]
    + 75 * df["treated"] * df["post"]  # True treatment effect = $75
    + 0.002 * df["income"] + np.random.normal(0, 30, n)
)

import statsmodels.formula.api as smf

# DiD regression
did_model = smf.ols("revenue ~ treated + post + treated:post + age + income", data=df).fit()
print(did_model.summary().tables[1])
print(f"\nEstimated treatment effect (DiD): ${did_model.params['treated:post']:.2f}")
print(f"True treatment effect: $75.00")
```

---

## Advanced Phase Skills Checklist

- [ ] Train a neural network with PyTorch from scratch with proper training loop
- [ ] Fine-tune a BERT-based model for a custom classification task
- [ ] Build a forecasting model with Prophet and SARIMA
- [ ] Design and analyze a proper A/B test (power analysis → results → business recommendation)
- [ ] Apply Difference-in-Differences for causal analysis
- [ ] Use SHAP values to explain a black-box model
- [ ] Have a specialization project (NLP, CV, forecasting, or experimentation)

**Projects:** [See Advanced Projects →](../projects/)
