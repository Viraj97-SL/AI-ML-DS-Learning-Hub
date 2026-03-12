<div align="center">

# Landmark ML Competitions Worth Studying

### The curriculum of competitive data science

</div>

---

## Why Study Past Competitions?

Each major Kaggle competition is a concentrated lesson in a specific ML challenge. Studying winning solutions from these competitions will teach you more practical ML in a month than most courses will in a year.

> **How to use this list:** Pick 3–5 competitions relevant to your area, read the winning solutions, then try to reproduce the key techniques on the original data.

---

## The 20 Landmark Competitions

### Tabular Data

| Competition | Platform | Year | Topic | Winner's Approach | Key Technique | Link |
|-------------|----------|------|-------|-------------------|---------------|------|
| **Titanic: Machine Learning from Disaster** | Kaggle | Ongoing | Binary classification | Feature engineering on names/cabins + ensemble | Title extraction, cabin deck features | [Explore](https://www.kaggle.com/c/titanic) |
| **House Prices: Advanced Regression** | Kaggle | Ongoing | Regression | LightGBM + Ridge blend, heavy feature engineering | Log-transform target, polynomial features | [Explore](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| **Porto Seguro's Safe Driver Prediction** | Kaggle | 2017 | Imbalanced binary classification | Stacking 300+ models, target encoding | GAM features, balanced sampling | [Solutions](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629) |
| **Santander Customer Transaction Prediction** | Kaggle | 2019 | Binary classification (200 features) | Magic features (value frequency), LightGBM | Identifying "fake" test rows | [1st Place](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89003) |
| **IEEE-CIS Fraud Detection** | Kaggle | 2019 | Fraud detection (imbalanced) | LightGBM, uid-based aggregation features | Client identity construction | [1st Place](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284) |

**What to learn from tabular competitions:**
- Feature engineering matters more than model choice
- Always check for "magic features" (data-specific patterns)
- Proper CV strategy is critical for imbalanced data

---

### Natural Language Processing

| Competition | Platform | Year | Topic | Winner's Approach | Key Technique | Link |
|-------------|----------|------|-------|-------------------|---------------|------|
| **Jigsaw Toxic Comment Classification** | Kaggle | 2018 | Multi-label text classification | LSTM + fastText + CNN ensemble | Character-level models for OOV | [1st Place](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557) |
| **Google QUEST Q&A Labeling** | Kaggle | 2020 | Multi-output regression (NLP) | RoBERTa + BERT ensemble, Spearman optimization | Multi-target joint training | [1st Place](https://www.kaggle.com/c/google-quest-challenge/discussion/129840) |
| **CommonLit Readability Prize** | Kaggle | 2021 | Regression on text difficulty | DeBERTa-large fine-tuning | MCRMSE optimization, pseudo-labeling | [1st Place](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/260729) |
| **Feedback Prize — Evaluating Writing** | Kaggle | 2022 | Token classification (argument mining) | DeBERTa-v3-large, ensemble of 5 seeds | Strided prediction on long documents | [1st Place](https://www.kaggle.com/c/feedback-prize-2021/discussion/313177) |
| **LLM Science Exam** | Kaggle | 2023 | Multiple choice QA with LLMs | Fine-tuned Llama-2 + DeBERTa + RAG | Wikipedia retrieval + reranking | [1st Place](https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446303) |

**What to learn from NLP competitions:**
- Transformer fine-tuning best practices (learning rate, warmup, batch size)
- Handling long documents (striding, hierarchical models)
- Multi-task and multi-label training tricks

---

### Computer Vision

| Competition | Platform | Year | Topic | Winner's Approach | Key Technique | Link |
|-------------|----------|------|-------|-------------------|---------------|------|
| **Dogs vs. Cats** | Kaggle | 2013 | Binary image classification | VGG16 fine-tuning (revolutionary at time) | Transfer learning from ImageNet | [Explore](https://www.kaggle.com/c/dogs-vs-cats) |
| **APTOS Diabetic Retinopathy Detection** | Kaggle | 2019 | Medical image classification | EfficientNet-B5, Ben Graham preprocessing | Ben Graham image preprocessing | [1st Place](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065) |
| **RSNA Intracranial Hemorrhage Detection** | Kaggle | 2019 | Medical image multi-label classification | EfficientNet ensemble, windowing | Multi-window image stacking | [1st Place](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117223) |
| **SIIM-ISIC Melanoma Classification** | Kaggle | 2020 | Medical CV + tabular fusion | EfficientNet + metadata, heavy TTA | External data from prior years | [1st Place](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412) |
| **Google Landmark Recognition** | Kaggle | 2021 | Large-scale image retrieval | ArcFace + GeM pooling | Metric learning, re-ranking | [1st Place](https://www.kaggle.com/c/landmark-recognition-2021/discussion/277099) |

**What to learn from CV competitions:**
- Transfer learning from ImageNet-21k pre-trained models
- Augmentation strategies (Albumentations)
- Metric learning (ArcFace) for retrieval/verification tasks

---

### Time Series

| Competition | Platform | Year | Topic | Winner's Approach | Key Technique | Link |
|-------------|----------|------|-------|-------------------|---------------|------|
| **Walmart Store Sales Forecasting** | Kaggle | 2014 | Retail demand forecasting | Time-weighted regression + holiday features | Bayesian structural time series | [Explore](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) |
| **Web Traffic Time Series Forecasting** | Kaggle | 2017 | Multi-variate web traffic | Dilated causal CNNs (WaveNet-inspired) | WaveNet architecture for time series | [1st Place](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795) |
| **M5 Forecasting — Accuracy** | Kaggle | 2020 | Hierarchical retail demand | LightGBM with lag features, quantile regression | Hierarchical reconciliation | [1st Place](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684) |

**What to learn from time series competitions:**
- Proper temporal cross-validation (never shuffle!)
- Lag and rolling feature construction
- Hierarchical forecasting reconciliation

---

### Recommender Systems

| Competition | Platform | Year | Topic | Winner's Approach | Key Technique | Link |
|-------------|----------|------|-------|-------------------|---------------|------|
| **Netflix Prize** | Netflix | 2009 | Collaborative filtering | 800+ model ensemble (Bellkor's Pragmatic Chaos) | Matrix factorization + temporal dynamics | [Paper](https://www.kdd.org/kdd2009/docs/NetflixPrize-2009.pdf) |
| **Otto Group Product Classification** | Kaggle | 2015 | Multi-class classification | Stacking neural networks with GBDT | Feature hashing, stacked generalization | [1st Place](https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335) |

---

## Summary: What to Learn Column

| Technique | Best Taught By |
|-----------|----------------|
| Target encoding without leakage | Porto Seguro, Santander |
| Transformer fine-tuning | CommonLit, Feedback Prize |
| Temporal CV strategy | M5 Forecasting |
| Medical image preprocessing | APTOS, RSNA, SIIM-ISIC |
| Large-scale ensembling | Porto Seguro, Netflix Prize |
| LLM + RAG for competitions | LLM Science Exam |
| Metric learning (ArcFace) | Google Landmark |
| WaveNet for sequences | Web Traffic Forecasting |
| Feature importance + SHAP | IEEE-CIS Fraud |
| Pseudo-labeling | CommonLit, Jigsaw |

---

## How to Study a Winning Solution: A Template

```
1. Read the overview/discussion write-up (15 min)
2. Clone or download the winning code
3. Read the feature engineering code first (most valuable)
4. Understand the validation strategy
5. Trace the ensembling logic
6. Re-implement the top 2-3 ideas from scratch in a clean notebook
7. Write a 1-page summary of: what problem, what worked, why
```

---

## Resources

| Resource | Link |
|----------|------|
| All Kaggle Competition Solutions | [github.com/lbniesz/kaggle-solutions](https://github.com/lbniesz/kaggle-solutions) |
| M5 Competition Papers | [arxiv.org](https://arxiv.org/abs/2104.00786) |
| Netflix Prize Papers | [netflixprize.com](https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf) |
| DrivenData Past Competitions | [drivendata.org/competitions/past](https://www.drivendata.org/competitions/past/) |

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [Winning Strategies](winning_strategies.md) | [Main README](../README.md) | [Career Guide](../08_Career_Guide/) |