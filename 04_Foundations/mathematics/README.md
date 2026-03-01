# Mathematics & Statistics Foundations

> **"You don't need a math degree. You need enough math to understand what your models are doing."**

This section covers the mathematical foundations required for Data Science and Machine Learning. We focus on *intuition first*, then formalism — using Python to visualize concepts.

---

## What Math Do You Actually Need?

| Role | Math Depth Required |
|------|-------------------|
| Data Scientist | Statistics (high), Linear Algebra (medium), Calculus (low) |
| ML Engineer | Linear Algebra (medium), Calculus (medium), Statistics (medium) |
| AI Engineer | Statistics (low), Linear Algebra (low-medium), Calculus (low) |

**Good news:** You don't need to master all of this before starting. Learn what you need, when you need it.

---

## Topics Covered

### 1. Linear Algebra
- Vectors and matrices — the language of data
- Matrix operations (multiplication, transpose, inverse)
- Eigenvalues and eigenvectors (critical for PCA)
- Norms and distances (L1, L2, cosine similarity)
- Dot products and projections

**Why it matters:** Every ML model operates on matrices. Neural networks are just chains of matrix multiplications.

### 2. Calculus & Optimization
- Derivatives and the chain rule
- Partial derivatives and gradients
- Gradient descent (the algorithm behind all deep learning)
- Convexity and local vs global optima

**Why it matters:** Training any ML model = running gradient descent = following derivatives downhill.

### 3. Probability & Statistics
- Probability fundamentals (sample space, events, independence)
- Probability distributions (Normal, Binomial, Poisson, etc.)
- Bayesian thinking (prior, likelihood, posterior)
- Hypothesis testing and p-values
- Confidence intervals
- Maximum Likelihood Estimation (MLE)

**Why it matters:** All ML models are probabilistic. Understanding uncertainty is core to DS.

### 4. Information Theory
- Entropy and information
- Cross-entropy (the loss function of most classifiers)
- KL divergence (measuring distribution distance)
- Mutual information

**Why it matters:** These concepts underpin loss functions, model evaluation, and understanding why models fail.

---

## Learning Resources

### Linear Algebra
| Resource | Type | Time |
|----------|------|------|
| 3Blue1Brown "Essence of Linear Algebra" (YouTube) | Video series | 4-5 hrs |
| MIT OpenCourseWare 18.06 | Course | 20-30 hrs |
| "Linear Algebra" by Strang (free MIT) | Book | Self-paced |
| fast.ai Computational Linear Algebra | Course | 10 hrs |

### Calculus
| Resource | Type | Time |
|----------|------|------|
| 3Blue1Brown "Essence of Calculus" (YouTube) | Video series | 3-4 hrs |
| Khan Academy Calculus | Course | 20-40 hrs |
| "Calculus Made Easy" by Thompson | Book | 5-10 hrs |

### Statistics & Probability
| Resource | Type | Time |
|----------|------|------|
| StatQuest with Josh Starmer (YouTube) | Video series | 10-20 hrs |
| Khan Academy Statistics | Course | 20-40 hrs |
| "Think Stats" by Downey | Book (free) | 10-15 hrs |
| "Statistical Thinking in Python" (DataCamp) | Course | 8 hrs |

---

## Notebooks in This Section

| Notebook | Topics | Level |
|----------|--------|-------|
| [01_vectors_matrices.ipynb](01_vectors_matrices.ipynb) | Vectors, matrix ops, visualizations | Beginner |
| [02_linear_algebra_ml.ipynb](02_linear_algebra_ml.ipynb) | PCA, SVD, regression via matrices | Intermediate |
| [03_calculus_gradient_descent.ipynb](03_calculus_gradient_descent.ipynb) | Derivatives, gradient descent from scratch | Beginner-Intermediate |
| [04_probability_basics.ipynb](04_probability_basics.ipynb) | Distributions, probability rules, sampling | Beginner |
| [05_statistics_hypothesis.ipynb](05_statistics_hypothesis.ipynb) | Hypothesis testing, p-values, A/B testing | Intermediate |
| [06_bayesian_thinking.ipynb](06_bayesian_thinking.ipynb) | Bayes theorem, prior/posterior, MCMC intro | Intermediate-Advanced |
| [07_information_theory.ipynb](07_information_theory.ipynb) | Entropy, cross-entropy, KL divergence | Advanced |

---

## The Minimum Viable Math

If you're in a hurry, learn these in order:

1. **Vectors** — What they are, dot product, cosine similarity
2. **Matrices** — Multiplication, transpose
3. **Derivatives** — What they mean (slope), chain rule
4. **Gradient descent** — How models learn
5. **Normal distribution** — Bell curve, mean, std
6. **Hypothesis testing** — p-value, statistical significance

This will get you through 80% of what you need for junior-level roles.

---

*Back to: [Main README](../../README.md) | [DS Track](../../01_Data_Scientist/) | [ML Track](../../02_ML_Engineer/)*
