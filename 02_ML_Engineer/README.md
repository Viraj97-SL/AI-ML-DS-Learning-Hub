# ML Engineer Learning Track

> **"Build the systems that make machine learning work in the real world."**

ML Engineers build, deploy, and maintain production-grade ML systems at scale. They bridge the gap between data science research and real-world software products. This is one of the highest-paying engineering roles in tech.

---

## Track Overview

```
PHASE 1: Software Engineering Foundations (2-3 months)
    Python SWE → Git → Docker → System Design
         ↓
PHASE 2: ML Fundamentals (3-4 months)
    ML Algorithms → Deep Learning → Model Evaluation → PyTorch
         ↓
PHASE 3: MLOps & Production (4-6 months)
    Pipelines → Model Serving → Monitoring → Cloud ML
         ↓
PHASE 4: Advanced (3-5 months)
    Distributed Training → LLM Fine-tuning → Platform Engineering
```

---

## Prerequisites

- [ ] Solid Python programming (OOP, decorators, testing, type hints)
- [ ] Basic Linux/terminal comfort
- [ ] Understanding of REST APIs
- [ ] Basic Git workflow (commit, branch, merge)
- [ ] Basic ML concepts (what a model is, train/test split, etc.)

> **Don't have these?** Start at [Foundations →](../04_Foundations/)

---

## Skills You Will Build

### Technical Skills
| Category | Skills |
|----------|--------|
| Programming | Python (expert), Bash, SQL, Scala (optional) |
| ML Frameworks | PyTorch, TensorFlow, scikit-learn, XGBoost |
| MLOps | MLflow, Weights & Biases, DVC, Kubeflow |
| Model Serving | FastAPI, BentoML, TorchServe, Triton |
| Infrastructure | Docker, Kubernetes, Terraform |
| Data Engineering | Apache Airflow, Spark, Kafka, dbt |
| Cloud | AWS SageMaker / GCP Vertex AI / Azure ML |
| Monitoring | Evidently AI, Prometheus, Grafana |
| Feature Stores | Feast, Tecton |
| Testing | pytest, Great Expectations, model testing |

---

## Beginner Phase — Software Engineering for ML

**Goal:** Write production-quality Python, understand containers, and build simple ML APIs.

**Duration:** 2-3 months (10-15 hrs/week)

| Week | Topic | Resource | Project |
|------|-------|----------|---------|
| 1-2 | Advanced Python (OOP, testing, packaging) | [Python SWE Guide](beginner/01_python_swe.ipynb) | Refactor a messy script |
| 3-4 | Git for teams | [Git Deep Dive](beginner/02_git_workflow.md) | Collaborate on a shared repo |
| 5-6 | Docker fundamentals | [Docker Guide](beginner/03_docker_intro.ipynb) | Containerize a Python app |
| 7-8 | REST APIs with FastAPI | [FastAPI Guide](beginner/04_fastapi_intro.ipynb) | Build a prediction API |
| 9-10 | ML model basics (sklearn + PyTorch) | [ML Refresher](beginner/05_ml_refresher.ipynb) | Train + serialize a model |
| 11-12 | End-to-end: Train + Serve | [First ML API Project](beginner/06_first_ml_api.ipynb) | Full mini-project |

**[→ Start Beginner Phase](beginner/)**

---

## Intermediate Phase — MLOps Core

**Goal:** Track experiments, build ML pipelines, and deploy models to production.

**Duration:** 3-4 months (10-15 hrs/week)

| Week | Topic | Resource | Project |
|------|-------|----------|---------|
| 1-2 | Experiment Tracking (MLflow) | [MLflow Guide](intermediate/01_mlflow.ipynb) | Track 10 experiments |
| 3-4 | Data Version Control (DVC) | [DVC Guide](intermediate/02_dvc.ipynb) | Version a dataset + model |
| 5-6 | Pipeline Orchestration (Airflow) | [Airflow Guide](intermediate/03_airflow.ipynb) | Automated retraining pipeline |
| 7-8 | Model Serving at Scale | [Serving Guide](intermediate/04_model_serving.ipynb) | Deploy model to prod |
| 9-10 | Model Monitoring | [Monitoring Guide](intermediate/05_model_monitoring.ipynb) | Detect data drift |
| 11-12 | Feature Stores (Feast) | [Feature Store Guide](intermediate/06_feature_stores.ipynb) | Build feature pipeline |
| 13-14 | CI/CD for ML | [ML CI/CD Guide](intermediate/07_mlops_cicd.ipynb) | GitHub Actions for ML |

**[→ Start Intermediate Phase](intermediate/)**

---

## Advanced Phase — Production Scale & LLM Systems

**Goal:** Design ML platforms, train at scale, fine-tune LLMs, and handle enterprise ML challenges.

**Duration:** 4-6 months (10-15 hrs/week)

| Week | Topic | Resource | Project |
|------|-------|----------|---------|
| 1-3 | Kubernetes for ML | [K8s Guide](advanced/01_kubernetes_ml.ipynb) | Deploy ML on K8s |
| 4-6 | Distributed Training | [Distributed Training](advanced/02_distributed_training.ipynb) | Multi-GPU training job |
| 7-9 | LLM Fine-tuning (LoRA/QLoRA) | [LLM Fine-tuning Guide](advanced/03_llm_finetuning.ipynb) | Fine-tune Llama on custom data |
| 10-12 | Spark for ML | [Spark ML Guide](advanced/04_spark_ml.ipynb) | Process 100M row dataset |
| 13-15 | ML Platform Design | [Platform Architecture](advanced/05_ml_platform_design.md) | Design ML platform spec |
| 16-18 | Inference Optimization | [Optimization Guide](advanced/06_inference_optimization.ipynb) | Reduce latency by 50% |

**[→ Start Advanced Phase](advanced/)**

---

## Projects

### Beginner Projects
- [ ] ML Model as REST API (FastAPI + Docker)
- [ ] Model Comparison Dashboard (MLflow + 5 models)
- [ ] Automated Data Quality Checker

### Intermediate Projects
- [ ] End-to-End ML Pipeline (Airflow + MLflow + FastAPI)
- [ ] Model Monitoring System with Drift Detection
- [ ] A/B Testing Framework for ML Models

### Advanced Projects
- [ ] LLM Fine-tuning Pipeline (Llama/Mistral on custom data)
- [ ] Real-time Feature Store with Kafka + Feast
- [ ] ML Platform Design Document + Proof of Concept
- [ ] Multi-model Serving System with Kubernetes

**[→ See All Projects](projects/)**

---

## Skills Checklist

### Beginner Level
- [ ] Can write testable, modular Python code
- [ ] Understand Docker: build images, run containers, docker-compose
- [ ] Can build a REST API with FastAPI
- [ ] Can serialize and load ML models (pickle, joblib, ONNX)
- [ ] Know Git workflows: branching, PRs, code review

### Intermediate Level
- [ ] Track ML experiments with MLflow
- [ ] Version datasets and models with DVC
- [ ] Build Airflow DAGs for ML workflows
- [ ] Deploy a model API with Docker + cloud
- [ ] Detect data drift and model degradation
- [ ] Build a basic feature pipeline

### Advanced Level
- [ ] Deploy ML workloads on Kubernetes
- [ ] Run distributed training across multiple GPUs
- [ ] Fine-tune an LLM using LoRA
- [ ] Design an end-to-end ML platform architecture
- [ ] Optimize model inference latency (quantization, batching, caching)

---

## Recommended Resources

### Free Books & Docs
| Resource | Notes |
|----------|-------|
| [Full Stack Deep Learning](https://fullstackdeeplearning.com) | Best-in-class MLOps curriculum |
| [Made With ML](https://madewithml.com) | End-to-end production ML (GitHub: GokuMohandas/Made-With-ML) |
| [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) | Free 9-week MLOps course |
| [Designing ML Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) | Chip Huyen's seminal book |

### Certifications Worth Getting
- AWS Certified Machine Learning — Specialty
- Google Professional ML Engineer
- Databricks Certified ML Professional

---

*Back to: [Main README](../README.md) | [Role Comparison](../00_Overview/role_comparison.md)*
