# Statistics for Data Science & ML

> Statistics is the language of data science. You don't need a stats degree — you need the key concepts and when to apply them.

---

## Why Statistics Matters

```
Without Statistics:              With Statistics:
"Sales went up this month"       "Sales increased 12% ± 3% (p=0.03)"
"Users prefer blue button"       "Blue converts 2.3% better (n=50K, power=90%)"
"Our model is 92% accurate"      "Model achieves 0.91 AUC with well-calibrated probabilities"
"Revenue seems seasonal"         "Strong weekly + yearly seasonality (STL decomposition)"
```

---

## Core Statistical Concepts

### 1. Descriptive Statistics

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
data = np.concatenate([np.random.normal(70, 10, 800), np.random.normal(90, 5, 200)])

# ── Central Tendency ─────────────────────────────────────────
mean = np.mean(data)              # Sensitive to outliers
median = np.median(data)          # Robust to outliers
mode = stats.mode(data).mode     # Most frequent value

print(f"Mean:   {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode:   {mode:.2f}")
print(f"Skew:   {stats.skew(data):.3f}  (positive = right-skewed)")

# ── Spread / Variability ─────────────────────────────────────
std = np.std(data, ddof=1)        # ddof=1 for sample std
variance = np.var(data, ddof=1)
iqr = stats.iqr(data)             # Interquartile Range (robust spread)
cv = std / mean * 100             # Coefficient of Variation (% spread)

print(f"\nStd Dev: {std:.2f}")
print(f"IQR:     {iqr:.2f}")
print(f"CV:      {cv:.1f}%")

# ── Percentiles & Quartiles ──────────────────────────────────
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p:3d}th: {np.percentile(data, p):.2f}")
```

### 2. Probability Distributions

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# ── Normal Distribution ──────────────────────────────────────
x = np.linspace(-4, 4, 200)
axes[0, 0].plot(x, stats.norm.pdf(x), "b-", lw=2, label="PDF")
axes[0, 0].fill_between(x, stats.norm.pdf(x), where=(x >= -1) & (x <= 1),
                          alpha=0.3, label="68% (±1σ)")
axes[0, 0].fill_between(x, stats.norm.pdf(x), where=(x >= -2) & (x <= 2),
                          alpha=0.2, label="95% (±2σ)")
axes[0, 0].set_title("Normal Distribution")
axes[0, 0].legend(fontsize=8)

# ── t-Distribution vs Normal (small samples) ────────────────
for df_val, ls in [(2, "--"), (5, "-."), (30, ":"), (np.inf, "-")]:
    if df_val == np.inf:
        label, dist = "Normal", stats.norm
    else:
        label, dist = f"t(df={df_val})", stats.t(df=df_val)
    axes[0, 1].plot(x, dist.pdf(x), ls=ls, lw=2, label=label)
axes[0, 1].set_title("t-Distribution (heavier tails for small n)")
axes[0, 1].set_ylim(0, 0.42)
axes[0, 1].legend(fontsize=8)

# ── Chi-squared Distribution (test statistics) ──────────────
x_chi = np.linspace(0, 30, 200)
for df_val in [2, 5, 10, 15]:
    axes[0, 2].plot(x_chi, stats.chi2.pdf(x_chi, df=df_val), lw=2, label=f"df={df_val}")
axes[0, 2].set_title("Chi-squared Distribution")
axes[0, 2].legend(fontsize=8)

# ── Central Limit Theorem Demonstration ─────────────────────
# Start with a uniform distribution
population = np.random.uniform(0, 100, 100000)

for i, n in enumerate([1, 5, 30, 100]):
    sample_means = [np.random.choice(population, n).mean() for _ in range(10000)]
    ax = axes[1, i % 3] if i < 3 else axes[1, 2]
    axes[1, i].hist(sample_means, bins=50, density=True, alpha=0.7, color=f"C{i}")
    mu, sigma = np.mean(sample_means), np.std(sample_means)
    x_norm = np.linspace(min(sample_means), max(sample_means), 100)
    axes[1, i].plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), "r-", lw=2)
    axes[1, i].set_title(f"CLT: n={n} (SE={sigma:.2f})")

plt.suptitle("Probability Distributions & CLT", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

### 3. Hypothesis Testing Framework

```python
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms

# ============================================================
# The 5-Step Hypothesis Testing Framework
# ============================================================

print("""
HYPOTHESIS TESTING — 5 STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. State H₀ and H₁
2. Choose significance level α (typically 0.05)
3. Select the appropriate test
4. Compute test statistic and p-value
5. Make decision and interpret in context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ── Test 1: One-sample t-test ────────────────────────────────
# Q: Does this website have average load time = 2 seconds?
np.random.seed(42)
load_times = np.random.normal(2.3, 0.8, 50)  # True mean = 2.3s

t_stat, p_value = stats.ttest_1samp(load_times, popmean=2.0)
print(f"One-sample t-test (H₀: μ = 2.0s)")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
ci = stats.t.interval(0.95, len(load_times)-1, loc=np.mean(load_times),
                       scale=stats.sem(load_times))
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

# ── Test 2: Two-sample t-test ────────────────────────────────
# Q: Do two groups have different means?
group_A = np.random.normal(50, 10, 100)
group_B = np.random.normal(55, 12, 100)

# Check variance equality first (Levene's test)
levene_stat, levene_p = stats.levene(group_A, group_B)
equal_var = levene_p > 0.05

t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=equal_var)
print(f"\nTwo-sample t-test (H₀: μ_A = μ_B)")
print(f"Levene's p-value: {levene_p:.4f} → {'Equal variance assumed' if equal_var else 'Welch t-test (unequal variance)'}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"Conclusion: {'Statistically significant difference' if p_value < 0.05 else 'No significant difference'}")

# Effect size (Cohen's d) — statistical vs practical significance
effect_size = (np.mean(group_B) - np.mean(group_A)) / np.sqrt(
    (np.std(group_A)**2 + np.std(group_B)**2) / 2
)
print(f"Cohen's d: {effect_size:.3f} ({'Small' if abs(effect_size) < 0.5 else 'Medium' if abs(effect_size) < 0.8 else 'Large'})")

# ── Test 3: Chi-square test (categorical variables) ──────────
# Q: Is button color independent of conversion?
observed = np.array([
    [40, 460],   # Red: 40 converted, 460 didn't
    [55, 445],   # Blue: 55 converted, 445 didn't
])
chi2, p_val, dof, expected = stats.chi2_contingency(observed)
print(f"\nChi-square test (H₀: color and conversion are independent)")
print(f"Chi2: {chi2:.4f}, df: {dof}, p-value: {p_val:.4f}")
print(f"Conclusion: {'Reject H₀ — color affects conversion' if p_val < 0.05 else 'No significant association'}")

# ── Multiple Testing Problem ──────────────────────────────────
# If you run 20 tests at α=0.05, expect 1 false positive by chance!
# Solution: Bonferroni correction or Benjamini-Hochberg (FDR)
from statsmodels.stats.multitest import multipletests

# Simulate 20 p-values (1 real effect, 19 noise)
pvalues = [0.03, 0.52, 0.11, 0.04, 0.73, 0.01, 0.89, 0.45, 0.02, 0.66,
           0.92, 0.31, 0.07, 0.55, 0.18, 0.41, 0.09, 0.83, 0.06, 0.74]

reject_bonf, pvals_bonf, _, _ = multipletests(pvalues, alpha=0.05, method="bonferroni")
reject_bh, pvals_bh, _, _ = multipletests(pvalues, alpha=0.05, method="fdr_bh")

print(f"\nMultiple Testing (20 tests at α=0.05):")
print(f"  Uncorrected: {sum(p < 0.05 for p in pvalues)} significant")
print(f"  Bonferroni:  {sum(reject_bonf)} significant (very conservative)")
print(f"  B-H (FDR):   {sum(reject_bh)} significant (recommended)")
```

### 4. Bayesian Statistics

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Bayesian A/B Testing (better than frequentist for business decisions)
# ============================================================

def bayesian_ab_test(
    control_conversions: int, control_total: int,
    treatment_conversions: int, treatment_total: int,
    n_samples: int = 100000
):
    """
    Bayesian A/B test using Beta distribution (conjugate prior for Bernoulli).
    Beta(α, β) is the posterior distribution of a conversion rate.
    """
    # Prior: Beta(1, 1) = uniform = no prior knowledge
    alpha_prior = 1
    beta_prior = 1

    # Posterior update: Beta(α + conversions, β + non-conversions)
    alpha_control = alpha_prior + control_conversions
    beta_control = beta_prior + (control_total - control_conversions)

    alpha_treat = alpha_prior + treatment_conversions
    beta_treat = beta_prior + (treatment_total - treatment_conversions)

    # Sample from posteriors (Monte Carlo)
    control_samples = np.random.beta(alpha_control, beta_control, n_samples)
    treat_samples = np.random.beta(alpha_treat, beta_treat, n_samples)

    # P(treatment > control)
    prob_treatment_better = np.mean(treat_samples > control_samples)

    # Expected lift and credible interval
    lift_samples = (treat_samples - control_samples) / control_samples
    lift_mean = np.mean(lift_samples)
    lift_ci = np.percentile(lift_samples, [2.5, 97.5])

    # Risk of choosing treatment (expected loss)
    expected_loss = np.mean(np.maximum(0, control_samples - treat_samples))

    return {
        "prob_treatment_better": prob_treatment_better,
        "lift_mean": lift_mean,
        "lift_ci_95": lift_ci,
        "expected_loss": expected_loss,
        "control_posterior": (alpha_control, beta_control),
        "treatment_posterior": (alpha_treat, beta_treat),
    }

# Example
result = bayesian_ab_test(
    control_conversions=120, control_total=1200,
    treatment_conversions=145, treatment_total=1200
)

print(f"Bayesian A/B Test Results:")
print(f"  Control rate:   {120/1200:.3f}")
print(f"  Treatment rate: {145/1200:.3f}")
print(f"  P(treatment > control): {result['prob_treatment_better']:.3f}")
print(f"  Expected lift: {result['lift_mean']:+.1%}")
print(f"  95% Credible Interval: [{result['lift_ci_95'][0]:.1%}, {result['lift_ci_95'][1]:.1%}]")
print(f"  Expected loss if we ship treatment: {result['expected_loss']:.4f}")
print(f"\nDecision: ", end="")
if result["prob_treatment_better"] > 0.95:
    print("🚀 Ship treatment (>95% probability of being better)")
elif result["prob_treatment_better"] > 0.80:
    print("🤔 Collect more data (80-95% probability)")
else:
    print("❌ Stick with control (<80% probability of improvement)")

# Visualize posteriors
x = np.linspace(0, 0.25, 1000)
a_c, b_c = result["control_posterior"]
a_t, b_t = result["treatment_posterior"]

plt.figure(figsize=(10, 5))
plt.plot(x, stats.beta.pdf(x, a_c, b_c), "b-", lw=2, label=f"Control (α={a_c}, β={b_c})")
plt.plot(x, stats.beta.pdf(x, a_t, b_t), "r-", lw=2, label=f"Treatment (α={a_t}, β={b_t})")
plt.fill_between(x, stats.beta.pdf(x, a_c, b_c), alpha=0.2, color="blue")
plt.fill_between(x, stats.beta.pdf(x, a_t, b_t), alpha=0.2, color="red")
plt.xlabel("Conversion Rate")
plt.ylabel("Probability Density")
plt.title(f"Bayesian A/B Test Posteriors\nP(Treatment > Control) = {result['prob_treatment_better']:.1%}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Statistics Cheat Sheet

### Which Test to Use?

| Scenario | Test |
|----------|------|
| Compare mean to a known value | One-sample t-test |
| Compare 2 group means (continuous) | Two-sample t-test (Welch's) |
| Compare 2 group means (non-normal) | Mann-Whitney U test |
| Compare 3+ group means | ANOVA → Tukey post-hoc |
| Test independence (categorical) | Chi-square test |
| Test correlation (continuous) | Pearson (normal) or Spearman (non-normal) |
| Test before/after (same subjects) | Paired t-test |
| A/B test for proportions | Z-test for proportions |
| Distribution comparison | Kolmogorov-Smirnov test |
| Multiple tests simultaneously | Bonferroni or B-H correction |

### The Most Common Mistakes

| Mistake | Correct Approach |
|---------|-----------------|
| p < 0.05 = "important" | Always report effect size (Cohen's d) |
| Run test before collecting data is done | Pre-register sample size before running |
| Test after every data point | Fixed horizon OR use sequential testing |
| Run 10 tests at α=0.05 → 1 false positive | Use multiple testing correction |
| "Fail to reject" = "null hypothesis is true" | Absence of evidence ≠ evidence of absence |
| Use p-value as probability null is true | p-value is NOT P(H₀ is true) |

---

## Practice Problems

**Problem 1:** Your app's checkout time is currently 3.2 seconds. You deployed a change. 50 users measured: mean=2.9s, std=0.7s. Did the change help? (State hypotheses, run test, interpret)

**Problem 2:** Survey 200 men and 200 women about a product. 45% of men and 52% of women said "Yes". Is there a significant gender difference?

**Problem 3:** You ran 10 marketing campaigns simultaneously. 2 of them show p < 0.05. Are these real effects or false positives? What correction should you apply?

**Problem 4:** Design a Bayesian A/B test for a button that currently converts at 8%. You want to detect a 15% relative lift with 95% confidence. How would you structure this?

---

## Recommended Resources

| Resource | Focus | Free? |
|----------|-------|-------|
| StatQuest with Josh Starmer (YouTube) | Visual intuitions | Free |
| Think Stats by Allen Downey | Statistics with Python | [Free PDF](https://greenteapress.com/thinkstats2/) |
| Statistical Rethinking (McElreath) | Bayesian statistics | Lectures on YouTube |
| Khan Academy Statistics | Fundamentals | Free |
| statsmodels documentation | Applied stats in Python | Free |

---

*Back to: [Foundations](../README.md) | [DS Track](../../01_Data_Scientist/) | [Main README](../../README.md)*
