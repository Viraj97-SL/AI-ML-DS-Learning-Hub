# Data Scientist Interview Preparation

> Comprehensive guide to ace your DS interviews — from phone screens to final rounds.

---

## DS Interview Structure (Typical)

```
Round 1: Recruiter Screen (30 min)
    → Background, motivation, salary expectation

Round 2: Technical Screen (45-60 min)
    → Python/SQL coding OR stats questions

Round 3: Take-Home Assignment (4-8 hrs)
    → EDA + modeling on a provided dataset

Round 4: Technical Interview (60 min)
    → ML concepts, math, your take-home

Round 5: Case Study / Business Case (60 min)
    → How would you approach X problem?

Round 6: Behavioral Interview (45 min)
    → STAR-format questions about past work
```

---

## Statistics & Probability Questions

### Fundamentals
**Q1: Explain the Central Limit Theorem and why it's important for DS.**
> The CLT states that the sampling distribution of the sample mean approaches a normal distribution as sample size increases, regardless of the underlying population distribution (given finite variance). This matters because it justifies using normal-based statistical tests even when our data isn't normally distributed, and underpins confidence intervals and hypothesis tests.

**Q2: What's the difference between Type I and Type II error?**
> - Type I (False Positive / α): Rejecting a null hypothesis when it's actually true. Probability controlled by significance level (e.g., α = 0.05).
> - Type II (False Negative / β): Failing to reject a null hypothesis when it's false. Related to statistical power (Power = 1 - β).
> *Analogy: Type I = convicting an innocent person. Type II = acquitting a guilty person.*

**Q3: Explain p-value in plain English.**
> The p-value is the probability of observing results at least as extreme as our actual results, *assuming the null hypothesis is true*. A small p-value (e.g., < 0.05) means our observed result would be very unlikely under the null hypothesis, giving us evidence to reject it. **Common misconception:** p-value is NOT the probability that the null hypothesis is true.

**Q4: What's the difference between correlation and causation? Give an example.**
> Correlation measures the linear relationship between two variables (−1 to +1). Causation means one variable directly causes changes in another. Classic example: ice cream sales and drowning rates are positively correlated — because both increase in summer (common cause: hot weather). Neither causes the other.

**Q5: Explain confidence interval.**
> A 95% CI means: if we repeated our sampling process 100 times, approximately 95 of those intervals would contain the true population parameter. **Common misconception:** it does NOT mean "there's a 95% probability the true value is in this interval" — the true value is either in it or not.

**Q6: What is the difference between frequentist and Bayesian statistics?**
> - **Frequentist:** Probability = long-run frequency of events. Parameters are fixed unknown constants. Uses p-values, confidence intervals.
> - **Bayesian:** Probability = degree of belief. Parameters have probability distributions. Start with prior, update with data to get posterior. Uses credible intervals.

---

## Machine Learning Questions

**Q7: Explain bias-variance tradeoff.**
> Total error = Bias² + Variance + Irreducible error.
> - **Bias:** Error from wrong assumptions. High bias → underfitting (model too simple)
> - **Variance:** Error from sensitivity to training data. High variance → overfitting (model too complex)
> - As model complexity increases, bias decreases but variance increases. Optimal models balance both.

**Q8: How does a Random Forest work?**
> Random Forest is an ensemble of decision trees using two sources of randomness:
> 1. **Bagging:** Each tree is trained on a bootstrap sample (random sample with replacement) of the training data
> 2. **Feature randomness:** At each split, only a random subset of features is considered
> Final prediction: majority vote (classification) or mean (regression) across all trees.
> Why does it work? Averaging uncorrelated trees reduces variance without increasing bias much.

**Q9: Explain gradient boosting.**
> Gradient boosting builds trees sequentially where each tree corrects the errors of the previous ones. Specifically, each new tree is fit to the *negative gradient* of the loss function — essentially fitting the residuals. XGBoost, LightGBM, and CatBoost are popular implementations with regularization and speed improvements.

**Q10: When would you use L1 vs L2 regularization?**
> - **L1 (Lasso):** Adds |weights| to loss. Drives some weights exactly to zero → feature selection. Use when you suspect many features are irrelevant.
> - **L2 (Ridge):** Adds weights² to loss. Shrinks all weights toward zero but rarely exactly zero. Better when all features might be relevant.
> - **ElasticNet:** Combines both. Good default when unsure.

**Q11: How do you handle class imbalance?**
> 1. **Resample:** Oversample minority (SMOTE) or undersample majority
> 2. **Class weights:** `class_weight='balanced'` in sklearn gives minority class more weight
> 3. **Change threshold:** Lower decision threshold to catch more positives
> 4. **Different metrics:** Use precision-recall curve, F1, AUC-ROC instead of accuracy
> 5. **Anomaly detection approach:** Treat as novelty detection for extreme imbalance

**Q12: What's the difference between precision and recall? When do you prefer each?**
> - **Precision** = TP / (TP + FP): Of predicted positives, how many are correct?
> - **Recall** = TP / (TP + FN): Of actual positives, how many did we catch?
> - Prefer **high recall** when missing positives is costly (cancer screening, fraud detection)
> - Prefer **high precision** when false positives are costly (spam filter — don't want to miss real email)

---

## SQL Questions

**Q13: Write a query to find the second highest salary from an employees table.**
```sql
-- Method 1: Using LIMIT/OFFSET
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;

-- Method 2: Using subquery (works across SQL dialects)
SELECT MAX(salary)
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Method 3: Using window function (most robust)
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rnk
    FROM employees
) ranked
WHERE rnk = 2
LIMIT 1;
```

**Q14: Write a query to find users who logged in on both consecutive days.**
```sql
WITH daily_logins AS (
    SELECT DISTINCT user_id, DATE(login_time) as login_date
    FROM user_events
    WHERE event_type = 'login'
)
SELECT DISTINCT a.user_id
FROM daily_logins a
JOIN daily_logins b
    ON a.user_id = b.user_id
    AND a.login_date = b.login_date + INTERVAL '1 day';
```

**Q15: Calculate 7-day rolling average of daily revenue.**
```sql
SELECT
    date,
    revenue,
    AVG(revenue) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7day_avg
FROM daily_revenue
ORDER BY date;
```

---

## Python / Coding Questions

**Q16: Implement k-means clustering from scratch.**
```python
import numpy as np

def kmeans(X, k, max_iters=100):
    """
    Simple k-means implementation.
    X: (n_samples, n_features)
    k: number of clusters
    """
    # Initialize centroids randomly
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx]

    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.array([
            np.linalg.norm(X - c, axis=1) for c in centroids
        ])  # (k, n_samples)
        labels = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([
            X[labels == i].mean(axis=0) for i in range(k)
        ])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

# Test
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
labels, centroids = kmeans(X, k=3)
```

**Q17: Calculate AUC-ROC from scratch.**
```python
import numpy as np

def auc_roc(y_true, y_scores):
    """Calculate AUC-ROC using trapezoidal integration."""
    thresholds = sorted(set(y_scores), reverse=True)
    tprs, fprs = [0], [0]

    pos = sum(y_true)
    neg = len(y_true) - pos

    for threshold in thresholds:
        predicted = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(p == 1 and t == 1 for p, t in zip(predicted, y_true))
        fp = sum(p == 1 and t == 0 for p, t in zip(predicted, y_true))
        tprs.append(tp / pos)
        fprs.append(fp / neg)

    tprs.append(1)
    fprs.append(1)

    # Trapezoidal rule
    auc = sum(
        (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2
        for i in range(1, len(fprs))
    )
    return auc
```

---

## Case Study / Business Questions

**Q18: Netflix asks you to measure the impact of a recommendation algorithm change. How do you approach this?**

Structure your answer using these steps:
1. **Clarify the goal:** What metric matters most? Watch time? Retention? Engagement?
2. **Design the experiment:** A/B test — control (old algo) vs treatment (new algo). Randomize at user level.
3. **Sample size:** Power analysis to determine how many users needed to detect a meaningful effect.
4. **Duration:** Long enough to capture weekly viewing patterns. Typically 2-4 weeks.
5. **Primary metric:** Watch time per user per week
6. **Guardrail metrics:** Ensure we don't degrade other KPIs (signup rate, diversity of content)
7. **Analysis:** T-test or Mann-Whitney U on the primary metric. Check for novelty effect.
8. **Decision:** If statistically significant AND practically significant (effect size matters), ship it.

**Q19: Instagram asks how you'd detect bot accounts.**

1. **Frame as ML problem:** Binary classification (bot vs real)
2. **Features to consider:**
   - Account age vs follower count ratio
   - Profile completeness (photo, bio)
   - Posting frequency and patterns
   - Time between posts (very regular = suspicious)
   - Follower/following ratio
   - Engagement rate on posts
   - Content similarity (repeated/copied)
   - Network features (mutual connections)
3. **Model:** Start with logistic regression for interpretability, then try XGBoost/neural net
4. **Evaluation:** Precision-recall at different thresholds — prefer high precision (avoid banning real users)
5. **Ground truth:** Sample of known bots (previously banned) + known real accounts
6. **Deployment:** Score daily, manual review queue for ambiguous cases

---

## Behavioral Questions (STAR Format)

**Situation** → **Task** → **Action** → **Result**

Common DS behavioral questions:
- Tell me about a project where your analysis changed a business decision
- Describe a time you had to communicate complex findings to a non-technical audience
- Tell me about a model that performed poorly in production — how did you handle it?
- How do you handle disagreement with stakeholders about data interpretation?
- Describe a time you had to work with messy, incomplete data

---

## Take-Home Assignment Tips

1. **Document your EDA** — show your thought process, not just results
2. **Baseline first** — simple model before complex
3. **Feature engineering** — this separates good DSs from great ones
4. **Interpret the model** — SHAP values, feature importance
5. **Business recommendations** — always connect back to the problem
6. **Clean code** — use functions, docstrings, meaningful variable names
7. **Slide deck or report** — your communication skills matter as much as the code

---

## 30-Day Interview Prep Plan

| Week | Focus |
|------|-------|
| Week 1 | Statistics review, probability, A/B testing |
| Week 2 | ML algorithms, model evaluation, bias-variance |
| Week 3 | SQL practice (StrataScratch, LeetCode), Python coding |
| Week 4 | Mock take-homes, case studies, behavioral prep |

**Resources:**
- [Ace the Data Science Interview](https://www.acethedatascienceinterview.com/) — 200+ questions
- [StrataScratch](https://stratascratch.com) — real company interview questions
- [LeetCode SQL 50](https://leetcode.com/studyplan/top-sql-50/) — free SQL problems

---

*Back to: [Interview Prep](../) | [DS Track](../../01_Data_Scientist/) | [Main README](../../README.md)*
