# Data Scientist — Beginner Phase

**Goal:** By end of this phase, you can independently load, clean, analyze, and visualize any dataset, and explain your findings clearly.

**Duration:** 2–3 months at 10–15 hrs/week
**Prerequisites:** Basic Python (variables, loops, functions)

---

## Curriculum Overview

```
Week 1–2  → Python for Data Science (pandas + NumPy)
Week 3–4  → Data Visualization (matplotlib + seaborn + Plotly)
Week 5–6  → Statistics Fundamentals (probability, distributions, inference)
Week 7–8  → Exploratory Data Analysis (EDA) Workflow
Week 9–10 → SQL for Data Analysis
Week 11–12→ Capstone: End-to-End EDA Project
```

---

## Week 1–2: Python for Data Science

### Learning Objectives
By the end of Week 2 you should be able to:
- Load any CSV, JSON, or Excel file into pandas
- Select, filter, and transform columns and rows
- Handle missing values confidently
- Perform groupby aggregations

### Topics

#### 1.1 NumPy — The Numerical Foundation

NumPy arrays are the backbone of all Python data science. Every ML library (pandas, scikit-learn, PyTorch) is built on NumPy.

**Key Concepts:**
```python
import numpy as np

# Creating arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Shape and dimensions
print(arr.shape)       # (5,)   — 1D with 5 elements
print(matrix.shape)    # (3, 3) — 3 rows, 3 columns
print(matrix.ndim)     # 2

# Creating special arrays
zeros = np.zeros((3, 4))         # 3×4 matrix of zeros
ones = np.ones((2, 3))           # 2×3 matrix of ones
identity = np.eye(4)             # 4×4 identity matrix
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 11) # 11 evenly spaced points from 0 to 1

# Mathematical operations (vectorized — NO loops needed!)
arr * 2               # Element-wise multiplication: [2, 4, 6, 8, 10]
arr ** 2              # Squared: [1, 4, 9, 16, 25]
np.sqrt(arr)          # Square root
np.log(arr)           # Natural log

# Aggregations
np.sum(arr)           # 15
np.mean(arr)          # 3.0
np.std(arr)           # 1.4142...
np.min(arr), np.max(arr)  # 1, 5
np.argmin(arr), np.argmax(arr)  # 0, 4 (indices)

# Slicing (same as Python lists but multi-dimensional)
matrix[0, :]          # First row: [1, 2, 3]
matrix[:, 1]          # Second column: [2, 5, 8]
matrix[1:3, 0:2]      # Sub-matrix rows 1-2, cols 0-1

# Boolean indexing (very powerful!)
arr[arr > 3]          # [4, 5]
arr[arr % 2 == 0]     # [2, 4] — even numbers
```

**Why this matters:** When you train a 1-billion parameter neural network, it's manipulating a very large NumPy/tensor array. Understanding this from the start builds the right mental model.

#### 1.2 pandas — The Data Science Workhorse

```python
import pandas as pd

# === LOADING DATA ===
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", index_col=0, parse_dates=["date_column"])
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_json("data.json")

# === FIRST LOOK ===
df.head(5)        # First 5 rows
df.tail(5)        # Last 5 rows
df.shape          # (rows, columns)
df.columns        # Column names
df.dtypes         # Data types of each column
df.info()         # Summary: dtypes + non-null counts
df.describe()     # Stats: mean, std, min, quartiles, max

# === MISSING VALUES ===
df.isnull().sum()             # Count NaN per column
df.isnull().mean() * 100      # % missing per column

# Strategies for handling missing data:
df.dropna()                           # Drop rows with any NaN
df.dropna(subset=["important_col"])   # Drop only if specific column is NaN
df.fillna(0)                          # Fill with constant
df.fillna(df["price"].median())       # Fill with median (robust to outliers)
df["category"].fillna("Unknown")      # Fill categorical with placeholder
df.interpolate(method="linear")       # Interpolate (good for time series)

# === SELECTING DATA ===
df["salary"]                          # Single column → Series
df[["name", "salary", "age"]]         # Multiple columns → DataFrame
df.iloc[0]                            # First row by position
df.iloc[0:5]                          # Rows 0–4 by position
df.iloc[0:5, 1:3]                     # Rows 0–4, columns 1–2
df.loc[df.index[0]]                   # First row by label
df.loc[df["department"] == "Engineering"]  # Filter rows

# === FILTERING ===
# Single condition
engineers = df[df["department"] == "Engineering"]

# Multiple conditions — use & (and), | (or), ~ (not)
senior_eng = df[(df["department"] == "Engineering") & (df["years_exp"] > 5)]
not_intern = df[df["level"] != "Intern"]
high_earners = df[df["salary"] > 100000]

# String operations
df[df["name"].str.startswith("A")]
df[df["email"].str.contains("@gmail.com")]
df[df["city"].str.lower().isin(["new york", "san francisco"])]

# isin for multiple values
df[df["department"].isin(["Engineering", "Data Science", "AI"])]

# === SORTING ===
df.sort_values("salary", ascending=False)
df.sort_values(["department", "salary"], ascending=[True, False])

# === FEATURE ENGINEERING (Creating New Columns) ===
df["salary_k"] = df["salary"] / 1000
df["is_senior"] = df["years_exp"] > 5
df["full_name"] = df["first_name"] + " " + df["last_name"]
df["name_length"] = df["name"].str.len()
df["salary_band"] = pd.cut(df["salary"], bins=[0, 50000, 100000, 200000],
                            labels=["Low", "Mid", "High"])

# Apply custom function
def classify_age(age):
    if age < 30: return "Young"
    elif age < 45: return "Mid"
    else: return "Senior"

df["age_group"] = df["age"].apply(classify_age)
# Or with lambda:
df["age_group"] = df["age"].apply(lambda x: "Young" if x < 30 else "Senior")

# === GROUPBY — the SQL GROUP BY of pandas ===
# Average salary by department
df.groupby("department")["salary"].mean()

# Multiple aggregations at once
df.groupby("department").agg({
    "salary": ["mean", "median", "std", "count"],
    "years_exp": "mean",
    "age": ["min", "max"]
})

# Groupby with multiple columns
df.groupby(["department", "level"])["salary"].mean().unstack()

# Value counts — frequency of each category
df["department"].value_counts()
df["department"].value_counts(normalize=True) * 100  # As percentages

# === MERGING / JOINING ===
# Like SQL JOINs
merged = pd.merge(df_employees, df_departments, on="dept_id", how="inner")
merged = pd.merge(df_left, df_right, left_on="emp_id", right_on="employee_id")

# Concat — stacking DataFrames
combined = pd.concat([df_2023, df_2024], axis=0, ignore_index=True)
wide_df = pd.concat([df_features, df_target], axis=1)
```

### Practice Exercises — Week 1–2

**Exercise 1:** Load [this Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) and answer:
- How many passengers survived?
- What was the average age by passenger class?
- What's the survival rate by sex?
- Which column has the most missing values?

**Exercise 2:** Create a DataFrame of your choice (at least 50 rows using pandas + random data), then:
- Add 3 derived columns
- Group by a categorical column and aggregate a numeric one
- Save to CSV and reload it

**Exercise 3:** Load [this movies dataset](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-05-28/winemag-data-130k-v2.csv) and find the top 10 most expensive wines per country.

---

## Week 3–4: Data Visualization

### Learning Objectives
- Create 10+ different chart types
- Know which chart type to use for which situation
- Build publication-quality visualizations
- Create interactive charts

### The Chart Selection Guide

```
Your data has...                    Use this chart
─────────────────────────────────────────────────
1 numeric variable (distribution)  → Histogram, KDE plot, box plot
2 numeric variables (relationship) → Scatter plot, line chart
1 numeric + 1 category             → Bar chart, box plot, violin plot
2 categories (composition)         → Stacked bar, pie chart (use sparingly)
Change over time                   → Line chart, area chart
Correlation matrix                 → Heatmap
High-dimensional data              → Pair plot, parallel coordinates
Geographic data                    → Choropleth map
```

### matplotlib — The Foundation

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Visualization Gallery", fontsize=16, fontweight="bold")

# 1. Line chart
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), color="steelblue", linewidth=2, label="sin(x)")
axes[0, 0].plot(x, np.cos(x), color="tomato", linestyle="--", label="cos(x)")
axes[0, 0].set_title("Line Chart")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Histogram
data = np.random.randn(1000)
axes[0, 1].hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
axes[0, 1].set_title("Histogram")
axes[0, 1].set_xlabel("Value")
axes[0, 1].set_ylabel("Frequency")

# 3. Scatter plot
x_scatter = np.random.randn(200)
y_scatter = x_scatter * 2 + np.random.randn(200) * 0.5
scatter = axes[0, 2].scatter(x_scatter, y_scatter, c=y_scatter, cmap="viridis",
                              alpha=0.6, s=30)
plt.colorbar(scatter, ax=axes[0, 2])
axes[0, 2].set_title("Scatter Plot")

# 4. Bar chart
categories = ["DS", "MLE", "AIE", "SWE", "PM"]
values = [95000, 130000, 120000, 110000, 100000]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
bars = axes[1, 0].bar(categories, values, color=colors, edgecolor="white")
axes[1, 0].set_title("Average Salary by Role")
axes[1, 0].set_ylabel("Salary ($)")
# Add value labels on top of bars
for bar, val in zip(bars, values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f"${val:,}", ha="center", va="bottom", fontsize=9)

# 5. Box plot
data_groups = [np.random.normal(0, std, 100) for std in [0.5, 1, 1.5, 2, 2.5]]
axes[1, 1].boxplot(data_groups, labels=["A", "B", "C", "D", "E"],
                   patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.7))
axes[1, 1].set_title("Box Plot (Distribution Comparison)")

# 6. Heatmap
correlation_matrix = np.random.uniform(-1, 1, (5, 5))
np.fill_diagonal(correlation_matrix, 1)
im = axes[1, 2].imshow(correlation_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
axes[1, 2].set_xticks(range(5))
axes[1, 2].set_yticks(range(5))
axes[1, 2].set_xticklabels(["A", "B", "C", "D", "E"])
axes[1, 2].set_yticklabels(["A", "B", "C", "D", "E"])
plt.colorbar(im, ax=axes[1, 2])
axes[1, 2].set_title("Heatmap")

plt.tight_layout()
plt.savefig("gallery.png", dpi=150, bbox_inches="tight")
plt.show()
```

### seaborn — Statistical Visualization Made Easy

```python
import seaborn as sns
import pandas as pd

# Load a built-in dataset
tips = sns.load_dataset("tips")     # Restaurant tips
titanic = sns.load_dataset("titanic")
iris = sns.load_dataset("iris")

# Set style (do this once at the top of your notebook)
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)

# Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram with KDE
sns.histplot(tips["total_bill"], kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Distribution of Total Bill")

# Box plot — compare distributions across categories
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, ax=axes[1])
axes[1].set_title("Bill by Day & Sex")

# Violin plot — richer than boxplot
sns.violinplot(x="day", y="total_bill", data=tips, ax=axes[2],
               inner="quartile", palette="Set2")
axes[2].set_title("Violin Plot by Day")
plt.tight_layout(); plt.show()

# Pair plot — see all pairwise relationships at once (great for EDA!)
g = sns.pairplot(iris, hue="species", diag_kind="kde", plot_kws={"alpha": 0.6})
g.fig.suptitle("Iris Dataset — Pair Plot", y=1.02)
plt.show()

# Heatmap (correlation matrix)
plt.figure(figsize=(10, 8))
corr = tips.select_dtypes(include="number").corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # Hide upper triangle (redundant)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, mask=mask, square=True, linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# FacetGrid — small multiples (same chart, different subsets)
g = sns.FacetGrid(tips, col="time", row="sex", margin_titles=True, height=4)
g.map(sns.histplot, "total_bill", kde=True)
g.add_legend()
plt.show()
```

### Plotly — Interactive Visualizations

```python
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = px.data.gapminder()

# Animated scatter plot (GDP vs Life Expectancy over time — famous Gapminder viz!)
fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                 animation_frame="year",
                 animation_group="country",
                 size="pop", color="continent",
                 hover_name="country",
                 log_x=True, size_max=55,
                 title="GDP per Capita vs Life Expectancy (1952-2007)")
fig.show()

# Interactive choropleth map
fig = px.choropleth(df[df["year"] == 2007],
                    locations="iso_alpha",
                    color="lifeExp",
                    hover_name="country",
                    color_continuous_scale="viridis",
                    title="Life Expectancy by Country (2007)")
fig.show()

# Sunburst chart (hierarchical data)
fig = px.sunburst(df[df["year"] == 2007],
                  path=["continent", "country"],
                  values="pop",
                  color="lifeExp",
                  title="Population by Continent & Country")
fig.show()
```

---

## Week 5–6: Statistics Fundamentals

### Learning Objectives
- Understand and apply probability distributions
- Perform hypothesis tests and interpret p-values
- Understand confidence intervals and sampling
- Apply the Central Limit Theorem

### Probability Distributions

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 1. Normal Distribution
x = np.linspace(-4, 4, 100)
for mu, sigma, label in [(0, 1, "N(0,1)"), (0, 2, "N(0,2)"), (2, 1, "N(2,1)")]:
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), label=label, lw=2)
axes[0, 0].set_title("Normal Distribution")
axes[0, 0].legend()
axes[0, 0].fill_between(x, stats.norm.pdf(x), where=(x > -1) & (x < 1),
                         alpha=0.2, label="68% of data")

# 2. Binomial Distribution (coin flips, conversion rates)
n, p = 20, 0.5
x_binom = np.arange(0, n+1)
axes[0, 1].bar(x_binom, stats.binom.pmf(x_binom, n, p), color="steelblue", alpha=0.7)
axes[0, 1].set_title(f"Binomial (n={n}, p={p})")
axes[0, 1].set_xlabel("Number of successes")

# 3. Poisson Distribution (events per time period)
for lam in [1, 5, 10]:
    x_pois = np.arange(0, 30)
    axes[0, 2].plot(x_pois, stats.poisson.pmf(x_pois, lam), "o-", label=f"λ={lam}")
axes[0, 2].set_title("Poisson Distribution")
axes[0, 2].legend()

# 4. Central Limit Theorem — CRITICAL CONCEPT
np.random.seed(42)
# Start with a non-normal (uniform) distribution
population = np.random.uniform(0, 100, 100000)
sample_means = [np.random.choice(population, size=30).mean() for _ in range(5000)]

axes[1, 0].hist(population[:1000], bins=30, alpha=0.5, label="Population (Uniform)", color="gray")
axes[1, 0].set_title("Population Distribution")
axes[1, 0].legend()

axes[1, 1].hist(sample_means, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
axes[1, 1].set_title("CLT: Sample Means (n=30) → Normal!")
axes[1, 1].set_xlabel("Sample Mean")

plt.tight_layout()
plt.show()
```

### Hypothesis Testing Step-by-Step

```python
from scipy import stats
import numpy as np

# ======================================================
# Example: Did our website redesign improve conversion?
# Control: Old design | Treatment: New design
# ======================================================

np.random.seed(42)
# Simulate conversion rates (1 = converted, 0 = didn't)
control = np.random.binomial(1, 0.10, size=1000)    # 10% baseline conversion
treatment = np.random.binomial(1, 0.12, size=1000)  # 12% (hoped for improvement)

print(f"Control conversion rate:   {control.mean():.3f} ({control.sum()} conversions)")
print(f"Treatment conversion rate: {treatment.mean():.3f} ({treatment.sum()} conversions)")

# Step 1: State hypotheses
# H₀ (null): No difference between groups (conversion_control = conversion_treatment)
# H₁ (alternative): Treatment converts better

# Step 2: Choose test — 2-sample z-test for proportions
from statsmodels.stats.proportion import proportions_ztest

count = np.array([treatment.sum(), control.sum()])
nobs  = np.array([len(treatment), len(control)])
stat, pvalue = proportions_ztest(count, nobs, alternative="larger")

print(f"\nZ-statistic: {stat:.4f}")
print(f"P-value:     {pvalue:.4f}")

# Step 3: Interpret
alpha = 0.05  # Significance level (5% false positive rate)
if pvalue < alpha:
    print(f"\n✅ Reject H₀ (p={pvalue:.4f} < α={alpha})")
    print("The new design significantly improves conversion!")
else:
    print(f"\n❌ Fail to reject H₀ (p={pvalue:.4f} ≥ α={alpha})")
    print("No statistically significant improvement detected.")

# Step 4: Calculate confidence interval for the difference
from statsmodels.stats.proportion import proportion_confint

# 95% CI for control rate
ci_control = proportion_confint(control.sum(), len(control), alpha=0.05)
ci_treatment = proportion_confint(treatment.sum(), len(treatment), alpha=0.05)

print(f"\n95% CI Control:   {ci_control[0]:.3f} – {ci_control[1]:.3f}")
print(f"95% CI Treatment: {ci_treatment[0]:.3f} – {ci_treatment[1]:.3f}")

# Step 5: Effect size (don't forget — statistical ≠ practical significance!)
lift = (treatment.mean() - control.mean()) / control.mean() * 100
print(f"\nLift: {lift:.1f}% relative improvement")
```

---

## Week 7–8: EDA Workflow

The Exploratory Data Analysis (EDA) workflow is the core skill of a Data Scientist. Here is a repeatable, professional process:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def full_eda(df, target_col=None):
    """Professional EDA template for any dataset."""

    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print("\nColumn types:")
    print(df.dtypes.value_counts())

    print("\n" + "=" * 60)
    print("2. MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
    missing_df = missing_df[missing_df["count"] > 0].sort_values("percent", ascending=False)
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("✅ No missing values!")

    print("\n" + "=" * 60)
    print("3. NUMERIC FEATURES")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    print(df[numeric_cols].describe().round(2).to_string())

    print("\n" + "=" * 60)
    print("4. CATEGORICAL FEATURES")
    print("=" * 60)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols[:5]:  # Show first 5
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head(5).to_string())

    print("\n" + "=" * 60)
    print("5. OUTLIERS (IQR method)")
    print("=" * 60)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

    if target_col:
        print("\n" + "=" * 60)
        print(f"6. TARGET VARIABLE: {target_col}")
        print("=" * 60)
        print(df[target_col].describe())
        if df[target_col].dtype == "object":
            print("\nClass distribution:")
            print(df[target_col].value_counts(normalize=True).round(3) * 100)

# Usage
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
full_eda(df, target_col="Survived")
```

---

## Week 9–10: SQL for Data Analysis

### Why SQL Matters
Every data science job requires SQL. You'll use it to:
- Query data warehouses (BigQuery, Snowflake, Redshift)
- Aggregate millions of rows without loading into Python
- Join tables from different sources
- Build data pipelines

### Practice with DuckDB (local, no setup)

```python
import duckdb
import pandas as pd

# DuckDB lets you run SQL directly on pandas DataFrames — great for practice!
con = duckdb.connect()

# Load the Titanic data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
con.register("titanic", df)

# Basic SELECT
result = con.execute("""
    SELECT PassengerId, Name, Age, Survived, Pclass
    FROM titanic
    WHERE Survived = 1
      AND Age < 30
    ORDER BY Age DESC
    LIMIT 10
""").df()
print(result)

# Aggregation with GROUP BY
survival_by_class = con.execute("""
    SELECT
        Pclass,
        COUNT(*) AS total_passengers,
        SUM(Survived) AS survivors,
        ROUND(AVG(Survived) * 100, 1) AS survival_rate_pct,
        ROUND(AVG(Age), 1) AS avg_age,
        ROUND(AVG(Fare), 2) AS avg_fare
    FROM titanic
    GROUP BY Pclass
    ORDER BY Pclass
""").df()
print(survival_by_class)

# Window functions (advanced SQL — very common in interviews!)
ranked = con.execute("""
    SELECT
        Name,
        Fare,
        Pclass,
        RANK() OVER (PARTITION BY Pclass ORDER BY Fare DESC) AS rank_in_class,
        AVG(Fare) OVER (PARTITION BY Pclass) AS avg_fare_in_class,
        Fare - AVG(Fare) OVER (PARTITION BY Pclass) AS fare_vs_class_avg
    FROM titanic
    ORDER BY Pclass, rank_in_class
""").df()
print(ranked.head(15))

# Subqueries
high_rollers = con.execute("""
    SELECT *
    FROM titanic
    WHERE Fare > (SELECT AVG(Fare) * 2 FROM titanic)
      AND Survived = 1
    ORDER BY Fare DESC
""").df()
print(high_rollers[["Name", "Fare", "Pclass", "Survived"]])

# CTE (Common Table Expression) — cleaner than subqueries
survival_stats = con.execute("""
    WITH class_stats AS (
        SELECT
            Pclass,
            COUNT(*) AS total,
            SUM(Survived) AS survived,
            ROUND(AVG(Survived) * 100, 1) AS survival_pct
        FROM titanic
        GROUP BY Pclass
    ),
    gender_stats AS (
        SELECT
            Sex,
            COUNT(*) AS total,
            SUM(Survived) AS survived,
            ROUND(AVG(Survived) * 100, 1) AS survival_pct
        FROM titanic
        GROUP BY Sex
    )
    SELECT 'Class' AS dimension, CAST(Pclass AS VARCHAR) AS value,
           total, survived, survival_pct
    FROM class_stats
    UNION ALL
    SELECT 'Gender', Sex, total, survived, survival_pct
    FROM gender_stats
    ORDER BY dimension, survival_pct DESC
""").df()
print(survival_stats)
```

### Top 20 SQL Interview Patterns

```sql
-- 1. Running total / cumulative sum
SELECT date, revenue,
       SUM(revenue) OVER (ORDER BY date) AS cumulative_revenue
FROM daily_sales;

-- 2. Month-over-month growth rate
SELECT month, revenue,
       LAG(revenue) OVER (ORDER BY month) AS prev_month,
       ROUND((revenue - LAG(revenue) OVER (ORDER BY month)) /
             LAG(revenue) OVER (ORDER BY month) * 100, 2) AS mom_growth_pct
FROM monthly_revenue;

-- 3. Rank within groups (top N per category)
SELECT *
FROM (
    SELECT *,
           RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
    FROM employees
) ranked
WHERE rnk <= 3;  -- Top 3 earners per department

-- 4. Find duplicates
SELECT email, COUNT(*) AS occurrences
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- 5. Find users who did action A then B (funnel analysis)
SELECT DISTINCT a.user_id
FROM events a
JOIN events b ON a.user_id = b.user_id
WHERE a.event_type = 'signup'
  AND b.event_type = 'purchase'
  AND b.event_time > a.event_time;

-- 6. Retention: users active in both month 1 and month 2
WITH month1 AS (SELECT DISTINCT user_id FROM events WHERE month = 1),
     month2 AS (SELECT DISTINCT user_id FROM events WHERE month = 2)
SELECT COUNT(m2.user_id) * 100.0 / COUNT(m1.user_id) AS retention_rate
FROM month1 m1
LEFT JOIN month2 m2 ON m1.user_id = m2.user_id;

-- 7. Pivot / crosstab using CASE WHEN
SELECT
    user_id,
    SUM(CASE WHEN channel = 'email' THEN revenue ELSE 0 END) AS email_rev,
    SUM(CASE WHEN channel = 'search' THEN revenue ELSE 0 END) AS search_rev,
    SUM(CASE WHEN channel = 'social' THEN revenue ELSE 0 END) AS social_rev
FROM conversions
GROUP BY user_id;
```

---

## Week 11–12: Capstone EDA Project

### Project: Comprehensive EDA on Real-World Data

**Choose one:**
1. [NYC Taxi Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — analyze trip patterns, pricing, demand
2. [Stack Overflow Developer Survey](https://insights.stackoverflow.com/survey) — analyze developer demographics and salaries
3. [World Development Indicators](https://databank.worldbank.org/source/world-development-indicators) — explore global development trends

**Your EDA should include:**
- [ ] Data loading and initial profiling
- [ ] Missing value analysis and imputation decisions
- [ ] Univariate analysis (distributions of all key variables)
- [ ] Bivariate analysis (relationships between variables)
- [ ] Multivariate analysis (3+ variables together)
- [ ] Outlier detection and handling
- [ ] At least 10 insightful visualizations
- [ ] 5 specific findings/insights written as business recommendations
- [ ] A clean, narrative Jupyter notebook (tell a story!)

**Deliverable:** A GitHub repo with a README, notebook, and 3–5 slide visual summary.

---

## Resources for This Phase

| Resource | What for | Link |
|----------|----------|------|
| Python Data Science Handbook | pandas + NumPy mastery | [Free online](https://jakevdp.github.io/PythonDataScienceHandbook/) |
| Kaggle pandas course | Structured exercises | [kaggle.com/learn](https://www.kaggle.com/learn/pandas) |
| StatQuest (YouTube) | Statistics intuition | [YouTube: StatQuest](https://www.youtube.com/@statquest) |
| Mode SQL Tutorial | SQL for analysts | [mode.com/sql-tutorial](https://mode.com/sql-tutorial/) |
| Storytelling with Data | Visualization best practices | Book by Cole Nussbaumer Knaflic |
| seaborn gallery | Chart inspiration | [seaborn.pydata.org](https://seaborn.pydata.org/examples/index.html) |
| Plotly documentation | Interactive charts | [plotly.com/python](https://plotly.com/python/) |

---

## Skills Checklist — Beginner Phase Complete When:
- [ ] Can load and profile any CSV/Excel/JSON in under 5 minutes
- [ ] Can handle missing values with justification (not just `dropna()`)
- [ ] Can create histograms, scatter plots, box plots, heatmaps, bar charts
- [ ] Can write SQL with GROUP BY, JOIN, and at least one window function
- [ ] Understand what a p-value actually means
- [ ] Have completed one full EDA project in a Jupyter notebook
- [ ] Can explain your findings to someone without a data background

**Next:** [Intermediate Phase →](../intermediate/)
