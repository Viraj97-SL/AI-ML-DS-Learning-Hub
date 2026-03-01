# Mini-Projects — Quick Hands-On Practice

> Bite-sized projects you can complete in 2-4 hours. Each one practices a specific skill and produces a shareable artifact.

---

## How to Use This Section

Each mini-project:
- **Teaches one specific skill deeply**
- **Takes 2-4 hours** (longer optional stretch goals)
- **Has a defined "done" state** you can share
- **Builds toward larger portfolio projects**

---

## Beginner Mini-Projects

### Mini-1: Titanic EDA Notebook

**Skill:** Exploratory Data Analysis
**Time:** 2 hours
**Output:** Jupyter notebook with 8+ charts and written insights

```python
# Starter code: run this to set up your environment
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Your tasks:
# 1. Basic info: shape, dtypes, missing values
# 2. Survival rate by: Sex, Pclass, Age group, Embarked
# 3. Age distribution (histogram + boxplot)
# 4. Fare distribution (identify outliers)
# 5. Correlation heatmap
# 6. Family size vs survival
# 7. Title extraction from Name column
# 8. Write 5 insights from your analysis

# Starter: Missing value heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
plt.title("Missing Values in Titanic Dataset")
plt.show()
print(f"Missing values:\n{df.isnull().sum()}")
```

**Done when:** Notebook has 8+ visualizations, each with a written insight below it.

**Stretch goal:** Add interactive Plotly charts, deploy to Google Colab with a shareable link.

---

### Mini-2: SQL Analytics Challenge

**Skill:** SQL for Data Analysis
**Time:** 2-3 hours
**Output:** SQL script with 10 analytical queries

```sql
-- Setup: Install DuckDB (pip install duckdb) and run this
-- duckdb setup:
-- INSTALL httpfs; LOAD httpfs;
-- CREATE TABLE titanic AS
-- SELECT * FROM read_csv_auto('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv');

-- Your 10 queries to write:
-- Q1: Survival rate by passenger class (Pclass)
-- Q2: Average fare by class and embarkation port
-- Q3: Top 10 most common titles (from Name column)
-- Q4: Survival rate by title
-- Q5: Family size distribution (SibSp + Parch + 1)
-- Q6: Survival rate by family size (solo vs small vs large family)
-- Q7: Age bucket analysis (child/teen/adult/senior)
-- Q8: Percentile analysis of Fare (p25, p50, p75, p90, p99)
-- Q9: Correlation proxy: average age by survival status
-- Q10: Missing value summary (count NULLs per column)

-- Starter: Q1
SELECT
    Pclass,
    COUNT(*) as total,
    SUM(Survived) as survived,
    ROUND(100.0 * SUM(Survived) / COUNT(*), 1) as survival_rate_pct
FROM titanic
GROUP BY Pclass
ORDER BY Pclass;
```

**Done when:** All 10 queries written, each with a brief comment explaining the business question.

---

### Mini-3: Your First ML Pipeline

**Skill:** Scikit-learn pipeline, train/test split, model evaluation
**Time:** 3 hours
**Output:** Python script that trains, evaluates, and saves a model

```python
# Complete this skeleton:
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Load data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# 2. Feature engineering (add at least 3 new features)
# TODO: Add Title, FamilySize, IsAlone, AgeGroup, FareBand

# 3. Define features and target
TARGET = "Survived"
NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Sex", "Pclass", "Embarked"]

# 4. Build preprocessing pipeline (TODO: complete this)
numeric_transformer = Pipeline(steps=[
    # TODO: imputer + scaler
])
categorical_transformer = Pipeline(steps=[
    # TODO: imputer + encoder
])
preprocessor = ColumnTransformer(transformers=[
    # TODO: numeric and categorical pipelines
])

# 5. Build full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Train/test split and fit
# TODO: split, fit, predict

# 7. Evaluate (at least 3 metrics)
# TODO: classification_report, ROC-AUC, confusion matrix

# 8. Cross-validation
# TODO: 5-fold cross-validation + print mean ± std

# 9. Save model
# TODO: joblib.dump(model, "titanic_model.pkl")

print("Model saved!")
```

**Done when:** Pipeline runs end-to-end, AUC > 0.80, model saved to disk.

---

## Intermediate Mini-Projects

### Mini-4: A/B Test Simulator

**Skill:** Statistical hypothesis testing, A/B test analysis
**Time:** 3 hours
**Output:** Interactive Jupyter notebook with real A/B analysis

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import zt_ind_solve_power

# Scenario: You work at an e-commerce company.
# Current checkout button (Control): red, 5.2% conversion rate
# New button (Treatment): blue, we want to detect a 15% relative lift
# i.e., new rate should be 5.98% or higher

# Part 1: Power Analysis
# Q: How many users do we need per variant?
effect_size = ?  # TODO: calculate Cohen's h for 5.2% vs 5.98%
n_per_variant = ?  # TODO: use zt_ind_solve_power

# Part 2: Simulate the experiment
np.random.seed(42)
n_users = ?  # Use your calculated sample size
control_conversions = ?   # TODO: simulate from Bernoulli(0.052)
treatment_conversions = ? # TODO: simulate from Bernoulli(0.062)

# Part 3: Frequentist test
# TODO: proportions_ztest

# Part 4: Bayesian test
# TODO: Beta distribution posteriors, P(treatment > control)

# Part 5: Practical significance
# Q: Is the lift economically meaningful?
# Average order value: $85
# Monthly visitors: 50,000
# TODO: Calculate monthly revenue impact

# Part 6: Visualization
# TODO: Show posterior distributions, lift distribution
```

**Done when:** Both frequentist and Bayesian tests implemented, revenue impact calculated.

---

### Mini-5: Build a Text Classifier

**Skill:** NLP preprocessing, TF-IDF, BERT fine-tuning
**Time:** 3-4 hours
**Output:** Two models (baseline + BERT) with comparison

```python
# Dataset: AG News (4-class text classification)
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
dataset = load_dataset("ag_news")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# Part 1: TF-IDF + Logistic Regression baseline
# TODO: TfidfVectorizer + LR pipeline
# Target: >90% accuracy

# Part 2: Error analysis
# Q: Which class is hardest to predict? Which examples are misclassified?
# TODO: Confusion matrix + 5 misclassified examples per class

# Part 3: Hugging Face zero-shot classifier (no training!)
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["World", "Sports", "Business", "Sci/Tech"]
# TODO: Test on 50 examples, compare to fine-tuned baseline

# Part 4: Fine-tune DistilBERT (optional if you have GPU)
# pip install transformers datasets accelerate
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments, Trainer
)
# TODO: Fine-tune for 2 epochs
# Target: >94% accuracy
```

**Done when:** TF-IDF baseline runs, comparison table between models, error analysis complete.

---

### Mini-6: FastAPI ML Service

**Skill:** ML model serving, API design, Docker
**Time:** 4 hours
**Output:** Dockerized API with /predict endpoint + tests

```python
# main.py — complete this skeleton

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Optional

app = FastAPI(title="Titanic Survival Predictor", version="1.0.0")

# Load model at startup (hint: use lifespan)
model = None

class PassengerFeatures(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: str = Field(..., pattern="^(male|female)$")
    age: float = Field(..., ge=0, le=120)
    fare: float = Field(..., ge=0)
    embarked: Optional[str] = Field(None, pattern="^(S|C|Q)$")
    sib_sp: int = Field(0, ge=0)
    parch: int = Field(0, ge=0)

class PredictionResponse(BaseModel):
    survived: bool
    survival_probability: float
    confidence: str  # "high", "medium", "low"

@app.get("/health")
def health():
    # TODO: return model loaded status
    pass

@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerFeatures):
    # TODO: call model, return prediction
    pass

# test_api.py — write these tests:
# - test_health_endpoint returns 200
# - test_predict_returns_valid_schema
# - test_predict_known_survivor (1st class female)
# - test_predict_known_victim (3rd class male)
# - test_invalid_pclass raises 422
# - test_missing_required_field raises 422

# Dockerfile — complete:
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install ...
# COPY . .
# HEALTHCHECK ...
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Done when:** `docker build` succeeds, `docker run`, `curl /predict` returns valid response, all tests pass.

---

## Advanced Mini-Projects

### Mini-7: Build a RAG Pipeline from Scratch

**Skill:** Vector embeddings, similarity search, RAG architecture
**Time:** 4 hours
**Output:** Working Q&A system over a document set

```python
# Build without LangChain — understand the internals

import numpy as np
from openai import OpenAI
from typing import List, Tuple
import json

client = OpenAI()

class SimpleVectorStore:
    """Minimal vector store — understand what LangChain's vectorstore does."""

    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []

    def embed(self, text: str) -> List[float]:
        """Get embedding for a text."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def add_documents(self, texts: List[str]):
        """Embed and store documents."""
        for text in texts:
            self.documents.append(text)
            self.embeddings.append(self.embed(text))

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Find k most similar documents to query."""
        query_embedding = self.embed(query)
        scores = [(doc, self.cosine_similarity(query_embedding, emb))
                  for doc, emb in zip(self.documents, self.embeddings)]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    # TODO: implement sliding window chunking

def answer_question(question: str, store: SimpleVectorStore) -> str:
    """Retrieve context and generate answer."""
    # TODO: search store, build context, call GPT

# Test with Wikipedia articles
documents = [
    # TODO: add 3-5 paragraphs from Wikipedia on a topic of your choice
]

store = SimpleVectorStore()
store.add_documents(documents)

# Test questions
questions = [
    "What is [topic]?",  # TODO: fill in
    "When did [event] happen?",
    "Who invented [thing]?",
]

for q in questions:
    answer = answer_question(q, store)
    print(f"Q: {q}\nA: {answer}\n")
```

**Done when:** RAG pipeline answers 3 questions correctly from provided documents. Compare answer quality with/without context.

---

### Mini-8: Prompt Engineering Challenge

**Skill:** Prompt design, evaluation, iteration
**Time:** 2-3 hours
**Output:** Before/after prompt comparison with evaluation results

**Challenge:** The following prompt gives inconsistent, often wrong results. Fix it and measure the improvement.

```python
from openai import OpenAI
import json

client = OpenAI()

# BROKEN PROMPT — produces inconsistent results
BROKEN_SYSTEM = "You are a helpful assistant."
BROKEN_USER = """Look at this review and tell me if it's positive or negative.
Also say why.

Review: {review}"""

# YOUR TASK:
# 1. Run the broken prompt on 20 reviews, measure accuracy
# 2. Fix the prompt (structured output, clear instructions, etc.)
# 3. Run the fixed prompt on the same 20 reviews
# 4. Compare accuracy and consistency

test_reviews = [
    ("This product is amazing! Best purchase I've made this year.", "positive"),
    ("Terrible quality, broke after 2 days. Complete waste of money.", "negative"),
    ("It's okay. Not great, not terrible. Does what it's supposed to do.", "neutral"),
    ("Fast shipping, product looks exactly like the photos. Happy with purchase.", "positive"),
    ("Disappointed. Expected much better based on the reviews.", "negative"),
    # Add 15 more...
]

def evaluate_prompt(system: str, user_template: str, reviews: list) -> dict:
    """Evaluate prompt accuracy on test reviews."""
    correct = 0
    results = []

    for review_text, true_label in reviews:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_template.format(review=review_text)}
            ],
        )
        # TODO: Parse response, compare to true_label
        # Challenge: The broken prompt doesn't give structured output
        # so parsing is hard — that's part of the problem to fix!

    return {"accuracy": correct / len(reviews), "results": results}

# Run experiment
broken_results = evaluate_prompt(BROKEN_SYSTEM, BROKEN_USER, test_reviews)
# TODO: Write FIXED_SYSTEM and FIXED_USER, run evaluation, compare
```

**Done when:** Fixed prompt achieves >95% accuracy on 20 reviews, vs <80% for broken prompt.

---

## Mini-Project Submission Checklist

Before sharing any mini-project:

- [ ] Code runs from clean environment (no hidden dependencies)
- [ ] Requirements clearly stated (Python version, packages)
- [ ] Results documented (what accuracy/metric did you achieve?)
- [ ] One interesting insight or lesson learned noted
- [ ] Uploaded to GitHub with descriptive README

---

*Back to: [Labs Overview](../README.md) | [Main README](../../README.md)*
