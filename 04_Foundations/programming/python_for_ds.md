# Python for Data Science & ML

> The essential Python skills every DS, MLE, and AIE must have.

---

## Why Python?

Python dominates data science and ML because:
- Readable syntax — focus on ideas, not boilerplate
- Massive ecosystem (pandas, NumPy, PyTorch, scikit-learn, LangChain...)
- Huge community — answers to every question are online
- "Glue language" — integrates with C, Java, R, Rust via bindings
- First-class Jupyter notebook support

---

## What Python Level Do You Need?

| Role | Python Level Needed |
|------|-------------------|
| Data Scientist | Intermediate — scripting, data manipulation, notebook fluency |
| ML Engineer | Advanced — OOP, testing, packaging, async, performance |
| AI Engineer | Intermediate-Advanced — async, decorators, API integration |

---

## Part 1: Python Fundamentals

### Core Types & Operators
```python
# Numbers
x = 42          # int
y = 3.14        # float
z = 1 + 2j      # complex

# Strings
name = "Claude"
multi = """
Multiple
lines
"""
# f-strings (use these!)
greeting = f"Hello, {name}! You are {len(name)} chars long."

# Collections
my_list = [1, 2, 3, "mix", True]       # Ordered, mutable
my_tuple = (1, 2, 3)                   # Ordered, immutable
my_set = {1, 2, 3, 2, 1}              # Unordered, unique: {1, 2, 3}
my_dict = {"key": "value", "num": 42}  # Key-value pairs
```

### Control Flow
```python
# Conditionals
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# Loops
for i in range(10):
    print(i)

# List comprehensions (very Pythonic — use them!)
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
matrix = [[i*j for j in range(5)] for i in range(5)]

# Dict comprehension
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}
```

### Functions
```python
# Basic function
def add(a, b):
    return a + b

# Default arguments
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# *args and **kwargs
def flexible(*args, **kwargs):
    print(f"Positional: {args}")
    print(f"Keyword: {kwargs}")

flexible(1, 2, 3, name="Alice", role="DS")

# Lambda (anonymous function)
double = lambda x: x * 2
squares = list(map(lambda x: x**2, range(10)))

# Decorators (very important for ML Engineers!)
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(0.1)
    return "done"
```

---

## Part 2: Python for Data Science

### NumPy — The Foundation
```python
import numpy as np

# Arrays (the DS workhorse)
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Key operations
print(arr.shape)          # (5,)
print(matrix.shape)       # (2, 3)
print(matrix.dtype)       # int64
print(matrix.T)           # Transpose

# Math operations (vectorized — much faster than loops)
arr * 2                   # Element-wise: [2, 4, 6, 8, 10]
arr + np.array([10,20,30,40,50])  # Element-wise addition
np.dot(arr, arr)          # Dot product: 55

# Statistical operations
np.mean(arr)              # 3.0
np.std(arr)               # 1.4142...
np.median(arr)            # 3.0
np.percentile(arr, 75)    # 4.0

# Indexing and slicing
arr[0]                    # First element
arr[-1]                   # Last element
arr[1:4]                  # Elements 1,2,3
matrix[0, :]              # First row
matrix[:, 1]              # Second column
matrix[matrix > 3]        # Boolean indexing

# Creating arrays
np.zeros((3, 4))          # 3x4 zeros
np.ones((2, 3))           # 2x3 ones
np.eye(4)                 # 4x4 identity matrix
np.linspace(0, 1, 100)    # 100 evenly spaced values
np.random.randn(1000)     # Standard normal distribution
```

### pandas — Data Manipulation
```python
import pandas as pd

# Creating DataFrames
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [28, 34, 24],
    "salary": [95000, 110000, 78000]
})

# Reading data
df = pd.read_csv("data.csv")
df = pd.read_json("data.json")
df = pd.read_excel("data.xlsx")
df = pd.read_sql("SELECT * FROM table", connection)

# Basic exploration
df.head(10)           # First 10 rows
df.tail(5)            # Last 5 rows
df.shape              # (rows, columns)
df.dtypes             # Column data types
df.describe()         # Summary statistics
df.info()             # Info about missing values, types
df.isnull().sum()     # Count missing values per column

# Selecting data
df["name"]                    # Series (single column)
df[["name", "salary"]]       # DataFrame (multiple columns)
df.iloc[0]                   # First row by position
df.iloc[0:5]                 # First 5 rows
df.loc[df["age"] > 30]       # Filter by condition

# Data cleaning (the most important real-world skill!)
df.dropna()                              # Drop rows with any NaN
df.fillna(0)                            # Fill NaN with 0
df.fillna(df.mean())                    # Fill with column mean
df["salary"] = df["salary"].astype(float)  # Change type
df.drop_duplicates()                    # Remove duplicate rows
df.rename(columns={"name": "full_name"})  # Rename columns
df["age"] = df["age"].str.strip()       # Remove whitespace (if string)

# Feature engineering
df["is_senior"] = df["age"] > 30
df["salary_k"] = df["salary"] / 1000
df["name_length"] = df["name"].str.len()

# Groupby (the SQL GROUP BY of pandas)
df.groupby("department")["salary"].mean()
df.groupby(["department", "level"]).agg({
    "salary": ["mean", "min", "max"],
    "age": "mean"
})

# Merging DataFrames (like SQL JOINs)
merged = pd.merge(df1, df2, on="id", how="inner")

# Sorting
df.sort_values("salary", ascending=False)
df.sort_values(["department", "salary"])

# Apply custom functions
df["category"] = df["salary"].apply(lambda x: "high" if x > 100000 else "low")

# Time series
df["date"] = pd.to_datetime(df["date"])
df.set_index("date").resample("M").mean()  # Monthly averages
```

---

## Part 3: Object-Oriented Python (Critical for ML Engineers)

```python
from dataclasses import dataclass
from typing import Optional, List
import json

@dataclass
class ModelConfig:
    """Configuration for an ML model."""
    model_name: str
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    hidden_layers: List[int] = None

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64]

class MLModel:
    """Base class for ML models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._is_trained = False

    def fit(self, X, y):
        """Train the model. Must be implemented by subclasses."""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting")
        return self._predict(X)

    def _predict(self, X):
        raise NotImplementedError

    def save(self, path: str):
        """Save model config to JSON."""
        with open(path, "w") as f:
            json.dump(vars(self.config), f)

    @classmethod
    def load(cls, path: str):
        """Load model from saved config."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
        return cls(config)

    def __repr__(self):
        return f"MLModel(name={self.config.model_name}, trained={self._is_trained})"
```

---

## Part 4: Python for Production (ML Engineers)

### Type Hints
```python
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

def preprocess(
    data: np.ndarray,
    normalize: bool = True,
    clip_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """Preprocess input data for model inference."""
    if normalize:
        data = (data - data.mean()) / data.std()
    if clip_range:
        data = np.clip(data, clip_range[0], clip_range[1])
    return data
```

### Testing with pytest
```python
# test_preprocessing.py
import pytest
import numpy as np
from your_module import preprocess

def test_normalize_output():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = preprocess(data, normalize=True)
    assert abs(result.mean()) < 1e-6  # Mean should be ~0
    assert abs(result.std() - 1.0) < 1e-6  # Std should be ~1

def test_clip_range():
    data = np.array([-100.0, 0.0, 100.0])
    result = preprocess(data, normalize=False, clip_range=(-1.0, 1.0))
    assert result.min() >= -1.0
    assert result.max() <= 1.0

def test_raises_on_empty():
    with pytest.raises(ValueError):
        preprocess(np.array([]))
```

### Async Python (for AI Engineers building APIs)
```python
import asyncio
import aiohttp

async def fetch_completion(session, prompt: str) -> str:
    """Async LLM API call."""
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}]
        }
    ) as response:
        data = await response.json()
        return data["choices"][0]["message"]["content"]

async def batch_completions(prompts: list[str]) -> list[str]:
    """Process multiple prompts concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_completion(session, p) for p in prompts]
        return await asyncio.gather(*tasks)

# Run it
results = asyncio.run(batch_completions(["Tell me a joke", "What is AI?"]))
```

---

## Part 5: Essential Libraries Quick Reference

```bash
# Install the core DS/ML stack
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
pip install torch torchvision  # or: pip install tensorflow
pip install xgboost lightgbm
pip install plotly streamlit fastapi uvicorn
pip install openai anthropic langchain langchain-openai
pip install sentence-transformers chromadb
```

---

## Practice Exercises

Try these to solidify your skills:

1. **Beginner:** Load a CSV, find missing values, fill them with column medians, and export cleaned data
2. **Intermediate:** Write a class `DataCleaner` with methods for each preprocessing step, with full type hints
3. **Advanced:** Write a decorator that caches function results to disk and add tests for it
4. **Expert:** Build an async function that calls 3 different LLM APIs in parallel and returns the fastest response

---

## Recommended Resources

| Resource | Best For | Free? |
|----------|----------|-------|
| [Automate the Boring Stuff](https://automatetheboringstuff.com) | Complete beginners | Yes |
| [Real Python](https://realpython.com) | Intermediate Python | Mostly |
| [Python Tricks (Dan Bader)](https://realpython.com/products/python-tricks-book/) | Clean Python patterns | No |
| [Fluent Python (Luciano Ramalho)](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) | Expert Python | No |
| [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) | DS-specific Python | Yes |

---

*Back to: [Foundations](../README.md) | [DS Track](../../01_Data_Scientist/) | [MLE Track](../../02_ML_Engineer/)*
