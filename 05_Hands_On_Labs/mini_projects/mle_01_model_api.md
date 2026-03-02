# Model API with FastAPI + Docker

> **Difficulty:** Intermediate | **Time:** 1-2 days | **Track:** ML Engineer

## What You'll Build
Package a trained scikit-learn model as a production-ready REST API using FastAPI, containerize it with Docker, and add health checks, input validation, and error handling.

## Learning Objectives
- Serialize and load ML models with joblib
- Build a typed REST API with FastAPI and Pydantic
- Containerize a Python service with Docker
- Add health checks and proper error handling
- Test the API with automated tests

## Tech Stack
- `fastapi`: REST API framework
- `pydantic`: input validation
- `joblib`: model serialization
- `scikit-learn`: model
- `Docker`: containerization
- `pytest` + `httpx`: API testing

## Step-by-Step Guide

### Step 1: Train and Save the Model
```python
# train.py
import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=5000, n_features=10, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f'Test accuracy: {acc:.4f}')

# Save model and metadata
joblib.dump(model, 'model.joblib')
import json
with open('model_meta.json', 'w') as f:
    json.dump({'accuracy': acc, 'n_features': 10, 'version': '1.0.0'}, f)
print('Model saved!')
```

### Step 2: Build the FastAPI Service
```python
# app/main.py
import joblib
import json
from pathlib import Path
from typing import Any
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

app = FastAPI(title='ML Prediction API', version='1.0.0')

# Load model at startup
MODEL = joblib.load('model.joblib')
with open('model_meta.json') as f:
    META = json.load(f)

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_items=10, max_items=10,
                                   description='10 numerical features for prediction')

    @validator('features', each_item=True)
    def check_finite(cls, v):
        if not (v == v) or abs(v) > 1e9:  # NaN check and range check
            raise ValueError('Feature values must be finite numbers')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

@app.get('/health')
def health_check() -> dict[str, Any]:
    return {'status': 'healthy', 'model_version': META['version'], 'accuracy': META['accuracy']}

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = int(MODEL.predict(X)[0])
        probability = float(MODEL.predict_proba(X)[0, prediction])
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            model_version=META['version']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')
```

### Step 3: Write Tests
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json()['status'] == 'healthy'

def test_valid_prediction():
    resp = client.post('/predict', json={'features': [0.5] * 10})
    assert resp.status_code == 200
    data = resp.json()
    assert data['prediction'] in [0, 1]
    assert 0 <= data['probability'] <= 1

def test_invalid_input_too_few_features():
    resp = client.post('/predict', json={'features': [1.0] * 5})
    assert resp.status_code == 422  # validation error

def test_invalid_input_nan():
    resp = client.post('/predict', json={'features': [float('nan')] + [0.5] * 9})
    assert resp.status_code == 422
```

### Step 4: Dockerfile
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 5: Build and Run
```bash
# Build image
docker build -t ml-prediction-api .

# Run container
docker run -d -p 8000:8000 --name ml-api ml-prediction-api

# Test it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -1.2, 0.3, 2.1, -0.5, 1.1, -0.8, 0.2, 1.5, -0.3]}'

# Check logs
docker logs ml-api

# Run tests
pytest tests/ -v
```

## Expected Output
- FastAPI server running in Docker container
- `/health` endpoint returning model metadata
- `/predict` endpoint with validated input/output
- All tests passing
- OpenAPI docs at `http://localhost:8000/docs`

## Stretch Goals
- [ ] Add `/batch_predict` endpoint that accepts a list of feature arrays and returns multiple predictions
- [ ] Add Prometheus metrics (request count, latency p99) and visualize in Grafana
- [ ] Add a model version header and build a `/v2/predict` endpoint with a different model, routing between versions based on the header

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
