# Software Engineering Best Practices for Data Scientists

> A practical guide to writing production-quality Python code — covering project structure, testing, documentation, CI/CD, and the engineering habits that separate notebook prototypes from reliable data systems.

---

## Overview

Most data scientists learn Python through tutorials and notebooks, which are excellent for exploration. But production data systems demand more: reproducible environments, testable code, readable documentation, automated quality checks, and reliable deployment pipelines.

Adopting software engineering best practices is the single biggest lever a data scientist can pull to advance their career. It's what differentiates a junior analyst from someone who can ship models to production. It also makes collaborative development possible — code that only you can run and understand is a liability, not an asset.

This guide covers the engineering practices most relevant to data scientists: those that apply to Python scripts, pipelines, notebooks, and ML systems without requiring a full software engineering background.

---

## Key Concepts

### Why SE Practices Matter for DS
- **Reproducibility**: Your analysis must produce the same result when run by a colleague 6 months later on a different machine
- **Maintainability**: Code you wrote 3 months ago must be understandable without reverse-engineering it
- **Reliability**: Production data pipelines must handle edge cases, bad data, and failures gracefully
- **Collaboration**: Teams need shared standards to review, extend, and debug each other's code

---

## 1. Project Structure

The **Cookiecutter Data Science** pattern is the de facto standard:

```
my_project/
├── README.md
├── pyproject.toml          ← project metadata + dependencies
├── .env                    ← secrets (NEVER commit this)
├── .gitignore
├── data/
│   ├── raw/                ← immutable original data
│   ├── processed/          ← cleaned, transformed data
│   └── external/           ← data from third parties
├── notebooks/
│   ├── 01_eda.ipynb        ← numbered, purpose-named notebooks
│   └── 02_modeling.ipynb
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── data/           ← data loading and preprocessing
│       ├── features/       ← feature engineering
│       ├── models/         ← model training and evaluation
│       └── visualization/  ← plotting functions
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── configs/
│   └── model_config.yaml   ← hyperparameters and settings
└── Makefile                ← common commands
```

**Key rule**: `data/raw/` is read-only. Never modify raw data in place — always write transformations to `data/processed/`.

---

## 2. Dependency Management

Use `pyproject.toml` with `poetry` or plain `pip`:

```toml
# pyproject.toml
[tool.poetry]
name = "my-project"
version = "0.1.0"
description = "Customer churn prediction pipeline"
python = "^3.11"

[tool.poetry.dependencies]
pandas = "^2.0"
scikit-learn = "^1.3"
xgboost = "^2.0"
pydantic = "^2.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4"
ruff = "^0.1"
mypy = "^1.5"
pre-commit = "^3.4"
```

```bash
# Pin ALL dependencies for full reproducibility
poetry export -f requirements.txt --output requirements.txt --without-hashes

# Or with pip
pip freeze > requirements-lock.txt
```

---

## 3. Type Hints and Static Analysis

Type hints make code self-documenting and catch bugs before runtime:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

def load_and_clean(
    filepath: Path,
    drop_nulls: bool = True,
    date_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load CSV and apply standard cleaning steps.

    Args:
        filepath: Path to the CSV file.
        drop_nulls: If True, drop rows with any null values.
        date_column: Column name to parse as datetime, or None to skip.

    Returns:
        Cleaned DataFrame with consistent dtypes.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If date_column is specified but not present.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    if date_column is not None:
        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' not in DataFrame")
        df[date_column] = pd.to_datetime(df[date_column])

    if drop_nulls:
        df = df.dropna()

    return df
```

Run mypy for static type checking:
```bash
mypy src/ --ignore-missing-imports --strict
```

---

## 4. Testing Data Pipelines

Testing data code requires slightly different patterns than general software:

```python
# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from src.my_project.features.engineer import compute_rolling_features

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame for feature engineering tests."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="D"),
        "sales": np.random.rand(30) * 1000 + 500,
        "customer_id": [f"C{i:04d}" for i in range(30)],
    })

def test_rolling_features_shape(sample_df):
    result = compute_rolling_features(sample_df, window=7)
    # Output has same number of rows
    assert len(result) == len(sample_df)
    # New feature columns were added
    assert "sales_rolling_mean_7d" in result.columns
    assert "sales_rolling_std_7d" in result.columns

def test_rolling_features_no_nan_after_warmup(sample_df):
    result = compute_rolling_features(sample_df, window=7)
    # After warmup period (7 rows), no NaNs should appear
    assert result.iloc[7:]["sales_rolling_mean_7d"].isna().sum() == 0

def test_rolling_features_values_are_reasonable(sample_df):
    result = compute_rolling_features(sample_df, window=7)
    # Rolling mean should be within the range of actual sales
    assert result["sales_rolling_mean_7d"].min() >= 0
    assert result["sales_rolling_mean_7d"].max() <= sample_df["sales"].max() * 1.01

@pytest.mark.parametrize("window", [3, 7, 14, 30])
def test_rolling_features_various_windows(sample_df, window):
    result = compute_rolling_features(sample_df, window=window)
    assert f"sales_rolling_mean_{window}d" in result.columns
```

Run tests:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 5. Pre-commit Hooks

Enforce code quality automatically before every commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff           # linting (replaces flake8, isort, pyupgrade)
        args: [--fix]
      - id: ruff-format    # formatting (replaces black)

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pandas-stubs, types-requests]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
      - id: detect-private-key    # blocks accidental credential commits
      - id: no-commit-to-branch   # prevents committing directly to main
        args: [--branch, master, --branch, main]
```

```bash
pip install pre-commit
pre-commit install          # installs hooks into .git/hooks/
pre-commit run --all-files  # run manually on all files
```

---

## 6. Logging Over Print Statements

```python
import logging
from pathlib import Path

# Configure once at the entry point of your application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),                    # console
        logging.FileHandler("logs/pipeline.log"),   # file
    ],
)

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, config: dict) -> object:
    logger.info("Starting model training with config: %s", config)
    try:
        # ... training code ...
        logger.info("Training complete. AUC: %.4f", auc)
        return model
    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        raise
```

**Why**: `print()` disappears in production. Logs persist, have timestamps, have severity levels, and can be shipped to centralized log aggregators (CloudWatch, Datadog, etc.).

---

## 7. Configuration Management

Never hardcode parameters. Use config files:

```python
# configs/model_config.yaml
model:
  name: xgboost_v2
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8

data:
  train_path: data/processed/train.parquet
  test_path: data/processed/test.parquet
  target_column: churn
  features:
    - tenure_months
    - monthly_charges
    - total_charges
```

```python
import yaml
from pydantic import BaseModel, FilePath

class ModelConfig(BaseModel):
    name: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float

class Config(BaseModel):
    model: ModelConfig

def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)  # Pydantic validates types on load

config = load_config("configs/model_config.yaml")
print(config.model.n_estimators)  # 200, typed as int
```

---

## 8. CI/CD for Data Science (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check src/ tests/

      - name: Type check with mypy
        run: mypy src/

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Tools & Libraries

| Tool | Purpose | Install |
|------|---------|---------|
| **ruff** | Linting + formatting (replaces flake8, black, isort) | `pip install ruff` |
| **mypy** | Static type checking | `pip install mypy pandas-stubs` |
| **pytest** | Testing framework | `pip install pytest pytest-cov` |
| **pre-commit** | Git hooks for quality gates | `pip install pre-commit` |
| **poetry** | Dependency management | `pip install poetry` |
| **pydantic** | Data validation + config | `pip install pydantic` |
| **loguru** | Better logging | `pip install loguru` |
| **hydra** | Config management | `pip install hydra-core` |
| **DVC** | Data version control | `pip install dvc` |
| **nbstripout** | Strip notebook outputs before commit | `pip install nbstripout` |

---

## Resources

### Courses & Guides
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) — The project structure standard
- [The Pragmatic Programmer](https://pragprog.com/titles/tpp20/the-pragmatic-programmer-20th-anniversary-edition/) — Timeless SE principles
- [Python Testing with pytest](https://pragprog.com/titles/bopytest2/python-testing-with-pytest-second-edition/) — Brian Okken — Best pytest book

### Key Resources
- [Real Python: Python Best Practices](https://realpython.com/) — Practical articles on all topics above
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) — Comprehensive style reference

---

## Projects & Exercises

**Project 1 — Refactor a Notebook**
Take one of your existing Jupyter notebook analyses. Refactor it into a proper package with `src/` structure. Extract functions, add type hints and docstrings, write pytest tests for the core logic, add pre-commit hooks, and wire up a GitHub Actions CI pipeline. Compare before/after.

**Project 2 — Typed ML Pipeline**
Write a complete training pipeline (`load → preprocess → feature engineer → train → evaluate → save`) as a typed Python package. Every function must have type hints. Run mypy with `--strict`. Zero mypy errors is the goal.

**Project 3 — Test a Data Pipeline**
Write a suite of 20+ pytest tests for a data preprocessing pipeline: test happy paths, edge cases (empty DataFrames, single rows, all-null columns), type expectations, and property-based tests using the `hypothesis` library. Achieve 95%+ coverage.

---

## Related Topics
- [Python for DS Guide →](../../../04_Foundations/programming/python_for_ds.md)
- [MLOps & CI/CD Notebook →](../../../02_ML_Engineer/intermediate/07_mlops_cicd.ipynb)
- [ML Engineer Track →](../../../02_ML_Engineer/README.md)
