# Data Scientist — Portfolio Projects

> Each project below is designed to be **portfolio-worthy**: a real problem, real data, and a result you can explain in an interview.

---

## Project 1: Titanic Survival Prediction (Beginner)

**Objective:** Predict passenger survival using the famous Titanic dataset. Perfect first end-to-end ML project.

**Dataset:** [Kaggle Titanic](https://www.kaggle.com/competitions/titanic)

**What to build:**
1. Full EDA with 8+ visualizations and written insights
2. Feature engineering: title extraction, family size, deck, fare bands
3. Compare 5+ models (LR, RF, XGBoost, SVM, KNN)
4. Hyperparameter tuning with Optuna
5. Submit to Kaggle (aim for top 10%)

**Key learning:** End-to-end ML workflow, feature engineering, model comparison

**Portfolio talking points:**
- "I used feature engineering to improve AUC from 0.78 to 0.85"
- "I found that passenger title was more predictive than raw name"
- "I scored in the top X% of Kaggle competition"

---

## Project 2: House Price Prediction (Intermediate)

**Objective:** Predict house sale prices on the Ames Housing dataset — a more complex regression problem.

**Dataset:** [Kaggle House Prices - Advanced Regression](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

**What to build:**
```python
# High-level pipeline
1. EDA: distribution of SalePrice (log-transform), correlation heatmap, missing value map
2. Feature engineering:
   - TotalSF = GrLivArea + TotalBsmtSF
   - HouseAge = YrSold - YearBuilt
   - HasGarage, HasPool, HasFireplace (binary)
   - Neighborhood quality encoding (target encoding)
3. Preprocessing pipeline with ColumnTransformer:
   - Numerical: impute (median) → StandardScaler
   - Categorical: impute (most_frequent) → OrdinalEncoder (for tree models)
4. Models: Lasso, Ridge, ElasticNet, XGBoost, LightGBM
5. Ensemble: weighted average of best models
6. Submission: RMSLE metric
```

**Target metric:** RMSLE < 0.12 (top 10% benchmark)

---

## Project 3: Customer Churn Prediction (Intermediate)

**Full business framing from problem → recommendation.**

**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Project structure:**
```
churn_prediction/
├── notebooks/
│   ├── 01_eda.ipynb          ← EDA + business insights
│   ├── 02_features.ipynb     ← Feature engineering
│   ├── 03_modeling.ipynb     ← Model training + evaluation
│   └── 04_business.ipynb     ← Cost-benefit analysis
├── src/
│   ├── features.py
│   ├── model.py
│   └── evaluate.py
├── reports/
│   └── executive_summary.pdf ← Non-technical summary
└── README.md
```

**Business framing to include:**
- Churn costs $500/customer to re-acquire. Early retention costs $50.
- What probability threshold maximizes ROI?
- Which customers should we target first? (highest predicted churn + highest LTV)
- Build a customer segment heatmap

**Key deliverable:** A slide deck with 5 slides: Problem → Data → Model → Impact → Recommendations

---

## Project 4: NYC Taxi Demand Forecasting (Advanced)

**Objective:** Predict taxi demand by zone and hour for operational planning.

**Dataset:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

**Skills demonstrated:**
- Large dataset handling (100M+ rows with Dask/DuckDB)
- Time series feature engineering (lags, rolling stats, cyclical encoding)
- Spatial analysis (visualize demand heatmaps)
- Proper time series cross-validation (no future leakage!)
- XGBoost for time series

```python
# Key features to engineer
features = {
    # Time features
    "hour_sin": np.sin(2 * np.pi * hour / 24),         # Cyclical encoding!
    "hour_cos": np.cos(2 * np.pi * hour / 24),
    "day_of_week": ...,
    "is_weekend": ...,
    "is_holiday": ...,

    # Lag features (historical demand)
    "demand_lag_1h": ...,    # 1 hour ago
    "demand_lag_24h": ...,   # Same hour yesterday
    "demand_lag_168h": ...,  # Same hour last week

    # Rolling statistics
    "demand_rolling_mean_24h": ...,
    "demand_rolling_std_24h": ...,

    # Weather (join with NYC weather data)
    "temperature": ...,
    "precipitation": ...,
    "wind_speed": ...,
}
```

---

## Project 5: Sentiment Analysis Pipeline (Advanced NLP)

**Objective:** Build a production-quality sentiment analysis system.

**Dataset:** [Amazon Product Reviews](https://nijianmo.github.io/amazon/index.html)

**Full implementation:**
```python
# 1. Data pipeline: 5M+ reviews → cleaned + sampled dataset
# 2. Baseline: TF-IDF + Logistic Regression
# 3. Advanced: Fine-tune DistilBERT
# 4. Evaluation: Confusion matrix, per-class metrics, error analysis
# 5. Deploy: Streamlit app with batch prediction CSV upload
# 6. Demo: "Paste any review, get instant sentiment + confidence"
```

**Stretch goals:**
- Aspect-based sentiment (not just overall, but per product aspect)
- Multilingual support using multilingual BERT
- Real-time Streamlit demo deployed to Hugging Face Spaces

---

## Project 6: COVID-19 Data Story (Beginner–Intermediate)

**Objective:** Tell a compelling data story about pandemic trends using public data.

**Data:** [Our World in Data COVID-19 dataset](https://github.com/owid/covid-19-data)

**Story structure:**
1. Global spread: choropleth maps, animated timeline
2. Vaccination rollout: by country, by income group
3. Economic impact correlation
4. Policy analysis: lockdown timing vs case curves
5. Prediction: simple ARIMA forecast

**Tools:** pandas, Plotly, Folium (maps), statsmodels

---

## Capstone Project: Your Own Idea

The best portfolio project is one you genuinely care about. Pick a domain you find interesting:

| Domain | Project Ideas |
|--------|---------------|
| Healthcare | Disease prediction from patient records |
| Finance | Credit scoring, fraud detection, stock forecasting |
| Sports | Player performance prediction, match outcome |
| Social Media | Trending topic detection, bot identification |
| Environment | Air quality forecasting, climate data analysis |
| E-commerce | Recommendation engine, demand forecasting |
| Education | Student performance prediction, topic modeling on papers |

**Capstone checklist:**
- [ ] Real, publicly available dataset (not toy data)
- [ ] Clear problem framing (what decision does this inform?)
- [ ] EDA with 8+ meaningful visualizations
- [ ] Feature engineering (at least 5 new features)
- [ ] 3+ models compared with proper cross-validation
- [ ] Final model interpretation (SHAP or feature importance)
- [ ] Business recommendations (not just model metrics)
- [ ] Clean GitHub repo with:
  - [ ] Descriptive README with results
  - [ ] requirements.txt
  - [ ] Well-organized notebooks
  - [ ] A deployed demo (Streamlit, HF Spaces, etc.)

---

## How to Present Your Projects in Interviews

**The STAR format for DS projects:**
- **Situation:** "I wanted to solve [problem] for [stakeholder]"
- **Task:** "My goal was to [specific measurable objective]"
- **Action:** "I [specific technical steps — don't just say 'I trained a model']"
- **Result:** "The model achieved [metric], which translates to [business impact]"

**The 3-metric rule:** Always have 3 metrics ready:
1. A model metric (AUC, RMSE, F1)
2. A business metric (revenue, cost, time saved)
3. A comparison baseline (vs. random, vs. previous model, vs. human)

---

*Back to: [DS Track](../README.md) | [Main README](../../README.md)*
