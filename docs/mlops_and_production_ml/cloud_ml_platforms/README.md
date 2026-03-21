# Cloud ML Platforms

> A practical comparison guide to AWS SageMaker, GCP Vertex AI, and Azure ML — covering managed training, model serving, feature stores, and cost optimization strategies.

---

## Overview

Cloud ML platforms provide fully managed infrastructure for the entire machine learning lifecycle: data preparation, feature engineering, distributed training, experiment tracking, model registry, serving, and monitoring. They eliminate the need to manage Kubernetes clusters, GPU drivers, and autoscaling infrastructure manually.

The three dominant platforms — **AWS SageMaker**, **GCP Vertex AI**, and **Azure ML** — each cover the same core use cases but with different APIs, pricing models, and ecosystem integrations. Choosing one depends on your existing cloud footprint, team expertise, and specific workloads.

For ML engineers, proficiency in at least one cloud ML platform is now table-stakes. Most production ML systems live in the cloud, and organizations rely on these platforms to reduce operational overhead and accelerate experimentation.

---

## Platform Comparison

| Feature | AWS SageMaker | GCP Vertex AI | Azure ML |
|---------|--------------|---------------|----------|
| **Training jobs** | SageMaker Training | Custom Training Jobs | Compute Clusters |
| **Managed notebooks** | SageMaker Studio | Vertex Workbench | Azure ML Studio |
| **Pipelines** | SageMaker Pipelines | Vertex Pipelines (Kubeflow) | Azure ML Pipelines |
| **Feature store** | SageMaker Feature Store | Vertex Feature Store | Azure ML Feature Store |
| **Model registry** | SageMaker Model Registry | Vertex Model Registry | Azure ML Model Registry |
| **Serving** | SageMaker Endpoints | Vertex AI Endpoints | Azure ML Endpoints |
| **AutoML** | SageMaker Autopilot | Vertex AutoML | Azure AutoML |
| **Experiment tracking** | SageMaker Experiments | Vertex Experiments | Azure ML Experiments |
| **Batch inference** | SageMaker Batch Transform | Vertex Batch Prediction | Azure ML Batch Endpoints |
| **Spot/Preemptible** | Spot Instances (up to 90% savings) | Spot VMs (up to 91%) | Low-priority VMs (up to 80%) |

---

## AWS SageMaker

### Core Components
- **SageMaker Studio**: Integrated IDE for notebooks, experiments, and pipelines
- **Training Jobs**: Managed distributed training on any instance type
- **SageMaker Experiments**: Automatic tracking of hyperparameters and metrics
- **Model Registry**: Versioned model artifacts with approval workflows
- **Real-time Endpoints**: Auto-scaling model serving with A/B testing
- **SageMaker Pipelines**: ML workflow orchestration (DAG-based)
- **Feature Store**: Online (low-latency) + offline (S3-backed) feature storage

### Key Code Example

```python
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role

role = get_execution_role()
session = sagemaker.Session()

# Define training job
estimator = SKLearn(
    entry_point="train.py",          # Your training script
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    output_path=f"s3://{session.default_bucket()}/model-artifacts",
    hyperparameters={
        "n-estimators": 100,
        "max-depth": 5,
        "learning-rate": 0.1,
    },
    use_spot_instances=True,         # Up to 90% cost savings
    max_wait=3600,
    max_run=1800,
)

# Start training
estimator.fit({"train": "s3://my-bucket/data/train"})

# Deploy to real-time endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name="my-model-endpoint",
)

# Inference
import json
result = predictor.predict([[5.1, 3.5, 1.4, 0.2]])
print(result)

# Clean up
predictor.delete_endpoint()
```

---

## GCP Vertex AI

### Core Components
- **Vertex Workbench**: Managed JupyterLab environment
- **Custom Training**: Training on managed compute with auto-packaging
- **Vertex Pipelines**: Kubeflow Pipelines-compatible ML workflow orchestration
- **Model Registry**: Centralized model versioning and metadata
- **Vertex Endpoints**: Online prediction with traffic splitting
- **Feature Store**: Managed feature storage with online/offline serving
- **Vertex Experiments**: Track and compare training runs

### Key Code Example

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

aiplatform.init(project="my-project", location="us-central1")

# Launch a custom training job
job = aiplatform.CustomTrainingJob(
    display_name="xgboost-classifier",
    script_path="trainer/task.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-2:latest",
    requirements=["xgboost==2.0.0"],
    model_serving_container_image_uri=(
        "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
    ),
)

model = job.run(
    dataset=None,
    replica_count=1,
    machine_type="n1-standard-4",
    args=["--n-estimators=200", "--max-depth=6"],
    sync=True,  # block until complete
)

# Deploy to endpoint
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=5,           # auto-scaling
    traffic_percentage=100,
)

# Online prediction
predictions = endpoint.predict(
    instances=[{"feature1": 1.5, "feature2": 2.3}]
)
print(predictions)
```

---

## Azure ML

### Core Components
- **Azure ML Studio**: Web-based UI for experiments, datasets, models
- **Compute Clusters**: Auto-scaling training clusters
- **Environments**: Docker-based reproducible training environments
- **Pipelines**: Multi-step ML workflows with caching
- **Model Registry**: Versioned model artifacts with tags and metadata
- **Online/Batch Endpoints**: Real-time and batch inference

### Key Code Example

```python
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute, Environment
from azure.identity import DefaultAzureCredential

# Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="<rg-name>",
    workspace_name="<ws-name>",
)

# Create compute cluster (auto-scales 0→4 nodes)
compute = AmlCompute(
    name="cpu-cluster",
    type="amlcompute",
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=120,
    tier="LowPriority",              # ~80% cost savings
)
ml_client.begin_create_or_update(compute).result()

# Define and submit training job
job = command(
    code="./src",
    command="python train.py --learning-rate ${{inputs.lr}}",
    inputs={"lr": 0.01},
    environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
    compute="cpu-cluster",
    display_name="sklearn-training-run",
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
```

---

## Cost Optimization Strategies

### 1. Use Spot/Preemptible Instances for Training
Training jobs are interruptible — use managed spot instances for up to 90% savings. Always checkpoint your model during training so you can resume if the spot instance is reclaimed.

| Platform | Spot Type | Typical Savings |
|----------|-----------|-----------------|
| AWS | `use_spot_instances=True` | 60–90% |
| GCP | `scheduling.preemptible=True` | 60–91% |
| Azure | `tier="LowPriority"` | 60–80% |

### 2. Right-size Inference Endpoints
- Start with the smallest instance that meets latency SLA
- Enable auto-scaling (scale to 0 for non-production)
- Use serverless inference for unpredictable traffic (SageMaker Serverless, Vertex Serverless)
- Batch similar requests together (dynamic batching)

### 3. Multi-framework Containers
Use pre-built containers (SKLearn, PyTorch, TensorFlow managed images) instead of building custom Docker images — faster startup, lower storage costs.

### 4. Experiment with Smaller Instances First
Debug on `ml.t3.medium` (SageMaker) or `e2-standard-2` (Vertex) before scaling up. GPU instances are 10–50x more expensive than CPU — only use them when training time savings justify the cost.

---

## When to Use Managed vs Custom

| Scenario | Recommendation |
|----------|---------------|
| Team is <5 people | Managed platform (less ops overhead) |
| Large org with dedicated MLOps team | Custom on Kubernetes (more control, lower cost) |
| Compliance/data residency required | Check platform's data handling guarantees |
| Research / experimentation | Vertex AI or SageMaker Studio (great notebooks) |
| Tight latency SLA (<5ms) | Custom deployment closer to application |
| Batch inference only | All three platforms are equivalent |

---

## Resources

### Courses & Tutorials
- [AWS SageMaker Immersion Day](https://catalog.workshops.aws/sagemaker-immersion-day) — Official hands-on workshop
- [Google Cloud Vertex AI Quickstarts](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform) — Official getting started guides
- [Microsoft Learn: Azure ML](https://learn.microsoft.com/en-us/training/paths/use-azure-machine-learning-pipelines/) — Free official Azure ML learning path

### Books & Papers
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107963/) — Chip Huyen — Best book on production ML; cloud-agnostic
- [Practical MLOps](https://www.oreilly.com/library/view/practical-mlops/9781098103002/) — Noah Gift — Covers SageMaker and Vertex in depth

---

## Projects & Exercises

**Project 1 — End-to-End SageMaker Pipeline**
Train an XGBoost model on the SageMaker built-in container. Define a SageMaker Pipeline with: data preprocessing step → training step → model evaluation step → conditional registration (only register if AUC > 0.85). Deploy the registered model to a real-time endpoint.

**Project 2 — Cost Comparison Experiment**
Train the same model 3 ways: (1) local CPU, (2) cloud on-demand GPU, (3) cloud spot GPU. Record wall-clock time and total cost for each. Document your findings as a decision guide for when to use each approach.

**Project 3 — Serverless Inference**
Deploy a scikit-learn model to SageMaker Serverless Inference or Vertex AI Serverless Endpoints. Build a simple load test with `locust` to characterize cold start times and throughput. Compare cost vs. always-on endpoint for different request volumes.

---

## Related Topics
- [MLOps & CI/CD Notebook →](../../02_ML_Engineer/intermediate/07_mlops_cicd.ipynb)
- [Distributed Training →](../../02_ML_Engineer/advanced/01_distributed_training.ipynb)
- [Inference Optimization →](../../02_ML_Engineer/advanced/06_inference_optimization.ipynb)
- [Data Quality & Observability →](../data_engineering/data_quality_&_observability/README.md)
