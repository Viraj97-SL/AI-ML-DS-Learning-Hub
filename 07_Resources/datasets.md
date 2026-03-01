# Datasets for Practice Projects

> 50+ curated datasets organized by task type. All are free and publicly available.

---

## Tabular / Structured Data

### For Beginners

| Dataset | Size | Task | Link | Notes |
|---------|------|------|------|-------|
| **Titanic** | 891 rows | Classification | [Kaggle](https://www.kaggle.com/competitions/titanic) | The classic first ML project |
| **Iris** | 150 rows | Classification | sklearn.datasets | Simplest multi-class problem |
| **Wine Quality** | 6,497 rows | Regression/Classification | [UCI](https://archive.ics.uci.edu/ml/datasets/wine+quality) | Good for feature importance |
| **Diabetes** | 768 rows | Classification | sklearn.datasets | Medical prediction basics |
| **Boston Housing** | 506 rows | Regression | sklearn.datasets | Classic regression problem |
| **Breast Cancer** | 569 rows | Classification | sklearn.datasets | Binary classification |

### For Intermediate/Advanced

| Dataset | Size | Task | Link | Notes |
|---------|------|------|------|-------|
| **House Prices (Ames)** | 1,460 rows | Regression | [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | Feature engineering playground |
| **Adult Income** | 48,842 rows | Classification | [UCI](https://archive.ics.uci.edu/ml/datasets/adult) | Classic income prediction |
| **Credit Card Fraud** | 284,807 rows | Imbalanced classification | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Class imbalance techniques |
| **Telco Customer Churn** | 7,043 rows | Classification | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Business-relevant churn |
| **Bank Marketing** | 45,211 rows | Classification | [UCI](https://archive.ics.uci.edu/ml/datasets/bank+marketing) | Direct marketing data |
| **Lending Club Loans** | 2.2M rows | Classification | [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | Financial risk modeling |
| **NYC Taxi Trips** | 100M+ rows | Regression/Forecasting | [NYC Open Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Real big data challenge |
| **E-commerce Reviews** | 23,486 rows | NLP + Classification | [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) | Sentiment + recommendation |

---

## NLP Datasets

| Dataset | Size | Task | Link | Notes |
|---------|------|------|------|-------|
| **IMDB Reviews** | 50K reviews | Sentiment | [Hugging Face](https://huggingface.co/datasets/imdb) | Binary sentiment benchmark |
| **SST-2** | 67K sentences | Sentiment | [Hugging Face](https://huggingface.co/datasets/sst2) | Stanford fine-grained sentiment |
| **AG News** | 120K articles | Text classification | [Hugging Face](https://huggingface.co/datasets/ag_news) | 4-class news categorization |
| **SQuAD 2.0** | 150K Q&A pairs | Extractive QA | [HF Datasets](https://huggingface.co/datasets/squad_v2) | Reading comprehension |
| **MultiNLI** | 433K pairs | NLI / Entailment | [Hugging Face](https://huggingface.co/datasets/multi_nli) | Sentence pair classification |
| **CommonCrawl** | Petabytes | Pre-training | [commoncrawl.org](https://commoncrawl.org/) | Raw web text |
| **Wikipedia** | 6.7M articles | QA, summarization | [HF Datasets](https://huggingface.co/datasets/wikipedia) | High-quality text |
| **Amazon Reviews** | 82M reviews | Sentiment, recommendation | [UCSD JMCAULEY](https://nijianmo.github.io/amazon/) | Product review analysis |
| **Yelp Reviews** | 8M reviews | Sentiment + regression | [Yelp Dataset](https://www.yelp.com/dataset) | 1-5 star ratings + text |
| **Twitter Sentiment 140** | 1.6M tweets | Sentiment | [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) | Social media NLP |
| **BookCorpus** | 11K books | Language modeling | [Hugging Face](https://huggingface.co/datasets/bookcorpus) | Long-form text |
| **Enron Emails** | 500K emails | Classification, NLP | [Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) | Email classification, spam |

---

## Computer Vision Datasets

| Dataset | Size | Task | Link | Notes |
|---------|------|------|------|-------|
| **MNIST** | 70K images | Classification | `torchvision.datasets` | Handwritten digit recognition |
| **Fashion-MNIST** | 70K images | Classification | `torchvision.datasets` | Clothing item classification |
| **CIFAR-10/100** | 60K images | Classification | `torchvision.datasets` | 10/100 class image recognition |
| **ImageNet** | 1.2M images | Classification | [image-net.org](https://image-net.org/) | The benchmark for CV |
| **COCO** | 200K images | Detection, segmentation | [cocodataset.org](https://cocodataset.org/) | Object detection standard |
| **Open Images V7** | 9M images | Detection, segmentation | [Google](https://storage.googleapis.com/openimages/web/index.html) | Largest multi-label dataset |
| **Chest X-Ray** | 108K images | Medical classification | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) | Pneumonia detection |
| **Cats vs Dogs** | 25K images | Binary classification | [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats) | Classic binary CV |
| **CelebA** | 200K face images | Attribute classification | [Drive](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Face attributes |

---

## Time Series Datasets

| Dataset | Frequency | Task | Link | Notes |
|---------|-----------|------|------|-------|
| **Air Passengers** | Monthly | Forecasting | `statsmodels` | Classic SARIMA example |
| **Electricity Consumption** | Hourly | Forecasting | [Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) | Energy demand forecasting |
| **Store Sales (Kaggle)** | Daily | Forecasting | [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | Retail sales competition |
| **Stock Prices** | Daily | Forecasting | [Yahoo Finance (yfinance)](https://pypi.org/project/yfinance/) | Financial time series |
| **M4 Competition** | Various | Forecasting | [M4 Competition](https://github.com/Mcompetitions/M4-methods) | 100K time series benchmark |
| **ETTh1/ETTm1** | Hourly | Long-term forecasting | [GitHub](https://github.com/zhouhaoyi/ETDataset) | Transformer TS benchmark |
| **NYC Traffic** | Hourly | Forecasting | [NYC Open Data](https://data.cityofnewyork.us/) | Traffic volume prediction |
| **Australian Weather** | Daily | Classification | [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) | Rain prediction |

---

## Recommendation System Datasets

| Dataset | Domain | Link | Notes |
|---------|--------|------|-------|
| **MovieLens 20M** | Movies | [grouplens.org](https://grouplens.org/datasets/movielens/) | Standard CF benchmark |
| **Amazon Product Reviews** | E-commerce | [UCSD](https://nijianmo.github.io/amazon/) | 82M reviews, 24 categories |
| **Yelp Dataset** | Restaurants | [yelp.com/dataset](https://www.yelp.com/dataset) | Business reviews |
| **LastFM** | Music | [HetRec](https://grouplens.org/datasets/hetrec-2011/) | Music listening history |
| **Book-Crossing** | Books | [Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) | Book ratings |
| **Steam Games** | Gaming | [Kaggle](https://www.kaggle.com/datasets/tamber/steam-video-games) | Game playtime data |

---

## Government & Social Datasets

| Dataset | Topic | Link | Notes |
|---------|-------|------|-------|
| **World Bank Open Data** | Global economics | [data.worldbank.org](https://data.worldbank.org/) | 1,600+ development indicators |
| **Our World in Data** | Global health, climate | [ourworldindata.org](https://ourworldindata.org/) | High-quality charts + data |
| **US Census Bureau** | Demographics | [census.gov/data](https://www.census.gov/data.html) | US population data |
| **WHO Global Health** | Healthcare | [who.int/data](https://www.who.int/data) | Global health indicators |
| **Google Trends** | Search trends | [trends.google.com](https://trends.google.com) | Temporal search patterns |
| **Stack Overflow Survey** | Developer data | [insights.stackoverflow.com](https://insights.stackoverflow.com/survey) | Annual developer survey |
| **GitHub Archive** | Code repositories | [gharchive.org](https://www.gharchive.org/) | Open source activity |
| **Open Payments (CMS)** | Healthcare payments | [openpaymentsdata.cms.gov](https://openpaymentsdata.cms.gov/) | Pharma-doctor payments |

---

## Dataset Platforms

| Platform | What's There | Link |
|----------|-------------|------|
| **Kaggle Datasets** | 250,000+ datasets across all domains | kaggle.com/datasets |
| **Hugging Face Datasets** | 100K+ NLP + vision + audio datasets | huggingface.co/datasets |
| **UC Irvine ML Repository** | 600+ classic ML datasets | archive.ics.uci.edu/ml |
| **Google Dataset Search** | Search engine for datasets | datasetsearch.research.google.com |
| **Data.gov** | US government open data | data.gov |
| **OpenML** | ML benchmark datasets | openml.org |
| **Papers With Code** | Datasets + benchmarks + SOTA | paperswithcode.com/datasets |
| **AWS Open Data** | Large-scale open datasets on S3 | registry.opendata.aws |
| **GDELT Project** | Global news events | gdeltproject.org |

---

## How to Load Datasets Efficiently

```python
# ── Hugging Face Datasets (best for NLP) ────────────────────
from datasets import load_dataset

# Load with streaming (don't download everything at once!)
dataset = load_dataset("imdb", streaming=True)
for example in dataset["train"].take(5):
    print(example["text"][:100], "→", example["label"])

# Standard load
dataset = load_dataset("ag_news", split="train")
print(dataset.features)

# ── Kaggle API ───────────────────────────────────────────────
# pip install kaggle
# Set up ~/.kaggle/kaggle.json with your API token
import subprocess
subprocess.run(["kaggle", "datasets", "download",
                "datasciencedojo/titanic", "--unzip"])

# ── DuckDB for large files (no pandas memory limits!) ────────
import duckdb
conn = duckdb.connect()
df = conn.execute("""
    SELECT * FROM read_csv_auto('large_file.csv')
    WHERE column_a > 100
    LIMIT 10000
""").df()

# ── Polars for fast DataFrame operations ─────────────────────
import polars as pl
df = pl.scan_csv("large_file.csv")  # Lazy evaluation
result = df.filter(pl.col("age") > 30).select(["name", "age", "salary"]).collect()
```

---

## Dataset Best Practices

| Practice | Why |
|----------|-----|
| Always check for class imbalance | Naive models predict majority class |
| Check for train/test leakage | Temporal features must be split correctly |
| Profile missing values before modeling | Different handling strategies needed |
| Check for duplicate rows | Can inflate metrics |
| Understand data provenance | Know when/how/why data was collected |
| Verify label quality | ML can't fix bad ground truth |
| Check for privacy/PII | Don't share personal data in public repos |

---

*Back to: [Resources](.) | [Main README](../README.md)*
