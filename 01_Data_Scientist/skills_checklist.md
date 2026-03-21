# Data Scientist Skills Checklist

Use this checklist to track your progress and identify gaps. Fork the repo, then check off items as you master each skill. When **80%+ of a level is complete**, you're ready to move to the next.

---

## Level 1 — Beginner Foundations

### Python for Data Science
- [ ] Write Python functions with type hints
- [ ] Use list comprehensions and generators
- [ ] Handle exceptions and file I/O
- [ ] Understand OOP basics (classes, inheritance)
- [ ] Use virtual environments and pip

### Data Manipulation
- [ ] Load CSV, JSON, Excel with pandas
- [ ] Filter, sort, and group DataFrames
- [ ] Handle missing values (drop, fill, impute)
- [ ] Merge and join multiple DataFrames
- [ ] Reshape data (pivot, melt, stack)

### SQL
- [ ] Write SELECT, WHERE, GROUP BY, ORDER BY queries
- [ ] Use JOIN (INNER, LEFT, RIGHT, FULL)
- [ ] Use window functions (ROW_NUMBER, RANK, LAG, LEAD)
- [ ] Write subqueries and CTEs
- [ ] Use aggregation functions (COUNT, SUM, AVG, MIN, MAX)

### Mathematics & Statistics
- [ ] Understand mean, median, mode, variance, std deviation
- [ ] Know common distributions (normal, binomial, Poisson)
- [ ] Understand conditional probability and Bayes' theorem
- [ ] Perform and interpret hypothesis tests (t-test, chi-square)
- [ ] Understand correlation vs causation

### Data Visualization
- [ ] Create basic plots with matplotlib (line, bar, scatter, histogram)
- [ ] Use seaborn for statistical plots
- [ ] Build interactive charts with plotly
- [ ] Choose the right chart type for the data
- [ ] Label axes, add titles, make charts presentation-ready

---

## Level 2 — Intermediate ML Skills

### Machine Learning Fundamentals
- [ ] Explain the bias-variance tradeoff
- [ ] Implement train/validation/test splits correctly
- [ ] Perform cross-validation (k-fold, stratified)
- [ ] Tune hyperparameters with GridSearchCV or RandomizedSearchCV
- [ ] Understand overfitting and regularization (L1, L2)

### Supervised Learning
- [ ] Train and evaluate Linear/Logistic Regression
- [ ] Use Decision Trees and understand how they split
- [ ] Train Random Forests and Gradient Boosting (XGBoost, LightGBM)
- [ ] Build classification pipelines with scikit-learn
- [ ] Evaluate models: accuracy, precision, recall, F1, AUC-ROC

### Unsupervised Learning
- [ ] Apply K-Means clustering and choose optimal K
- [ ] Use DBSCAN for density-based clustering
- [ ] Perform PCA for dimensionality reduction
- [ ] Use t-SNE/UMAP for visualization
- [ ] Evaluate clustering (silhouette score, inertia)

### Feature Engineering
- [ ] Encode categorical variables (one-hot, label, target encoding)
- [ ] Scale features (StandardScaler, MinMaxScaler, RobustScaler)
- [ ] Create interaction features and polynomial features
- [ ] Handle imbalanced datasets (SMOTE, class weights, resampling)
- [ ] Perform feature selection (correlation, mutual information, RFE)

### Experiment Design & Evaluation
- [ ] Design and analyze A/B tests
- [ ] Compute statistical power and sample size
- [ ] Understand p-values and avoid p-hacking
- [ ] Use confusion matrices correctly
- [ ] Explain model performance to non-technical stakeholders

---

## Level 3 — Advanced Topics

### Deep Learning
- [ ] Build and train neural networks with PyTorch
- [ ] Understand backpropagation and gradient descent variants
- [ ] Use CNNs for image tasks
- [ ] Use RNNs/LSTMs for sequential data
- [ ] Apply transfer learning with pretrained models

### Natural Language Processing
- [ ] Preprocess text (tokenization, stemming, lemmatization, stopwords)
- [ ] Use TF-IDF and word embeddings (Word2Vec, GloVe)
- [ ] Fine-tune a transformer (BERT, DistilBERT) for classification
- [ ] Build a text classification pipeline end-to-end
- [ ] Understand attention mechanisms conceptually

### Time Series Analysis
- [ ] Decompose time series (trend, seasonality, residual)
- [ ] Test for stationarity (ADF test) and make series stationary
- [ ] Fit ARIMA and SARIMA models
- [ ] Use Prophet for forecasting
- [ ] Evaluate time series models (MAE, RMSE, MAPE)

### Bayesian Methods
- [ ] Understand prior, likelihood, and posterior
- [ ] Perform Bayesian A/B testing
- [ ] Use PyMC or Stan for probabilistic modeling
- [ ] Interpret credible intervals vs confidence intervals
- [ ] Apply Bayesian optimization for hyperparameter search

### Model Interpretability
- [ ] Use SHAP values to explain predictions
- [ ] Apply LIME for local explanations
- [ ] Build partial dependence plots (PDP) and ICE plots
- [ ] Conduct a model audit for fairness and bias
- [ ] Write a model card for a deployed model

### Causal Inference
- [ ] Distinguish correlation from causation with DAGs
- [ ] Use propensity score matching
- [ ] Apply difference-in-differences analysis
- [ ] Understand instrumental variables
- [ ] Run and analyze natural experiments

---

## Soft Skills & Professional Practice

### Communication
- [ ] Present findings to a non-technical audience convincingly
- [ ] Write a clear executive summary for an analysis
- [ ] Build a stakeholder-ready dashboard (Streamlit, Tableau, or Looker)
- [ ] Document your methodology and assumptions clearly
- [ ] Translate business questions into data problems

### Engineering Best Practices
- [ ] Use Git for version control (branches, PRs, commit messages)
- [ ] Write modular, testable Python code
- [ ] Write unit tests for data transformation functions
- [ ] Use logging instead of print statements in production code
- [ ] Follow PEP 8 and use a linter (ruff, flake8)

### Data Intuition
- [ ] Spot data quality issues immediately during EDA
- [ ] Know when a model is "too good to be true"
- [ ] Sanity-check results against business intuition
- [ ] Ask "what could go wrong?" before deploying
- [ ] Know when NOT to use ML (simple rules may be better)

---

## Portfolio Milestones

- [ ] 1 complete EDA project (Jupyter notebook, published)
- [ ] 1 end-to-end ML project (data → model → evaluation)
- [ ] 1 project with a deployed model or interactive dashboard
- [ ] 1 Kaggle competition entry (any ranking counts)
- [ ] 1 project with real-world data you collected yourself
- [ ] GitHub profile with pinned repos, README, and activity graph
- [ ] LinkedIn profile with skills and featured projects section

---

## How to Use This Checklist

1. **Fork this repo** to your own GitHub account
2. Copy the relevant sections into your `my_progress.md` file
3. Check off items as you genuinely master them (not just "read about")
4. When stuck on a skill, open a GitHub Discussion or check the corresponding notebook
5. Use unchecked items as your **personal study backlog**

> **Tip:** Mastery means you can explain it to someone else and implement it from scratch. "Familiar with" is not mastery.

---

*Return to [Data Scientist Track →](README.md)*
*See also: [ML Engineer Checklist →](../02_ML_Engineer/README.md) | [AI Engineer Checklist →](../03_AI_Engineer/README.md)*
