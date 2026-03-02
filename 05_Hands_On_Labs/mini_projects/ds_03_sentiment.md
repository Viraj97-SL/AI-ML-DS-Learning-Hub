# Text Sentiment Analyzer

> **Difficulty:** Beginner-Intermediate | **Time:** 1-2 days | **Track:** Data Science

## What You'll Build
A sentiment analysis tool that classifies text (tweets, reviews, comments) using both a traditional ML approach (TF-IDF + Logistic Regression) and a Hugging Face transformer pipeline. Compare both approaches and deploy as a simple web app.

## Learning Objectives
- Preprocess and tokenize text data
- Build a TF-IDF + classifier baseline
- Use HuggingFace transformers for zero-shot and fine-tuned sentiment
- Compare classical ML vs transformer approaches
- Deploy as a Gradio or Streamlit app

## Prerequisites
- Basic Python and pandas
- Familiarity with scikit-learn

## Tech Stack
- `transformers`: HuggingFace models
- `scikit-learn`: TF-IDF + Logistic Regression baseline
- `pandas` / `numpy`: data handling
- `streamlit` or `gradio`: deployment

## Step-by-Step Guide

### Step 1: Load and Prepare Data
```python
import pandas as pd
from datasets import load_dataset

# Load Twitter Sentiment140 or IMDb
dataset = load_dataset('imdb')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Quick look
print(train_df.head(3))
print(train_df['label'].value_counts())
```

### Step 2: TF-IDF Baseline
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# Sample for speed
train_sample = train_df.sample(5000, random_state=42)
test_sample = test_df.sample(1000, random_state=42)

tfidf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
    ('clf', LogisticRegression(max_iter=200)),
])

tfidf_pipeline.fit(train_sample['text'], train_sample['label'])
preds = tfidf_pipeline.predict(test_sample['text'])
print(classification_report(test_sample['label'], preds, target_names=['Negative', 'Positive']))
```

### Step 3: Transformer Pipeline
```python
from transformers import pipeline

# Zero-shot (no fine-tuning needed)
sentiment_pipeline = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    truncation=True,
    max_length=512
)

# Run on test samples
sample_texts = test_sample['text'].head(100).tolist()
results = sentiment_pipeline(sample_texts)

# Map to 0/1 labels
transformer_preds = [1 if r['label'] == 'POSITIVE' else 0 for r in results]
transformer_scores = [r['score'] if r['label'] == 'POSITIVE' else 1 - r['score'] for r in results]

from sklearn.metrics import accuracy_score
print(f'Transformer accuracy (100 samples): {accuracy_score(test_sample["label"].head(100), transformer_preds):.3f}')
```

### Step 4: Compare Approaches
```python
import matplotlib.pyplot as plt

comparison = {
    'TF-IDF + LR': {'accuracy': 0.87, 'inference_ms': 2, 'model_size_mb': 5},
    'DistilBERT': {'accuracy': 0.93, 'inference_ms': 45, 'model_size_mb': 260},
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, metric in enumerate(['accuracy', 'inference_ms', 'model_size_mb']):
    values = [comparison[m][metric] for m in comparison]
    axes[i].bar(list(comparison.keys()), values, color=['steelblue', 'coral'])
    axes[i].set_title(metric)
plt.tight_layout()
plt.show()
```

### Step 5: Streamlit App
```python
# app.py
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

st.title('Sentiment Analyzer')
model = load_model()

text = st.text_area('Enter text to analyze:', height=150)
if st.button('Analyze') and text:
    result = model(text[:512])[0]
    sentiment = result['label']
    confidence = result['score']
    color = 'green' if sentiment == 'POSITIVE' else 'red'
    st.markdown(f'**Sentiment:** :{color}[{sentiment}]')
    st.progress(confidence)
    st.write(f'Confidence: {confidence:.1%}')
```

## Expected Output
- Accuracy comparison: TF-IDF baseline vs transformer
- Speed/accuracy tradeoff visualization
- Interactive sentiment analyzer web app

## Stretch Goals
- [ ] Fine-tune DistilBERT on a domain-specific dataset (e.g., financial news, product reviews)
- [ ] Add aspect-based sentiment (positive about price but negative about quality)
- [ ] Handle multilingual input using `xlm-roberta-base-sentiment`

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
