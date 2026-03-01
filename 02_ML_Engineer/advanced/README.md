# ML Engineer — Advanced Track

> Advanced ML Engineering: distributed training, Kubernetes, LLM fine-tuning, ML platform architecture, and production-grade systems that scale to millions.

---

## Prerequisites

Before starting this track, ensure you can confidently:
- Build and deploy ML models with FastAPI + Docker
- Use MLflow, DVC, and Prefect for MLOps
- Write production Python with proper testing and logging
- Design basic ML systems (recommendation, fraud detection)

---

## Advanced Track Phases

| Phase | Weeks | Focus |
|-------|-------|-------|
| **A1** | 1–3 | Distributed Training & Large-Scale ML |
| **A2** | 4–6 | Kubernetes for ML Workloads |
| **A3** | 7–9 | LLM Fine-Tuning & PEFT |
| **A4** | 10–12 | ML Platform Architecture |

---

## Phase A1: Distributed Training & Large-Scale ML

### 1. PyTorch Distributed Data Parallel (DDP)

```python
# ── Multi-GPU training with DDP ──────────────────────────────
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset

# ── Setup & Cleanup ──────────────────────────────────────────
def setup(rank: int, world_size: int):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ── Model ────────────────────────────────────────────────────
class LargeModel(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 4096, output_dim: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.network(x)

# ── Training function (runs per GPU) ─────────────────────────
def train_on_gpu(rank: int, world_size: int, epochs: int = 5):
    setup(rank, world_size)

    # Create model and move to GPU
    model = LargeModel().to(rank)
    model = DDP(model, device_ids=[rank])  # Wrap with DDP

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Dataset with DistributedSampler (ensures each GPU gets different data)
    X = torch.randn(10000, 1024)
    y = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(X, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=256, sampler=sampler, num_workers=4)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # IMPORTANT: ensures different shuffles
        total_loss = 0.0

        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(rank), batch_y.to(rank)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # Only rank 0 prints
        if rank == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save checkpoint from rank 0 only
    if rank == 0:
        torch.save(model.module.state_dict(), "distributed_model.pt")

    cleanup()

# Launch: torchrun --nproc_per_node=4 train_ddp.py
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_on_gpu,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

### 2. DeepSpeed for Billion-Parameter Models

```python
# pip install deepspeed
# ── DeepSpeed ZeRO-3 for model parallelism ───────────────────
import deepspeed
import torch
import torch.nn as nn

# ds_config.json
DS_CONFIG = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 2e-5, "weight_decay": 0.01}
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {"warmup_min_lr": 0, "warmup_max_lr": 2e-5, "warmup_num_steps": 200}
    },
    "zero_optimization": {
        "stage": 3,                          # ZeRO-3: shard params + grads + optimizer states
        "offload_optimizer": {"device": "cpu"},  # Offload optimizer to CPU RAM
        "offload_param": {"device": "cpu"},      # Offload params to CPU RAM
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "wall_clock_breakdown": False
}

# Initialize DeepSpeed engine
def train_with_deepspeed(model, dataset):
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=DS_CONFIG
    )

    for step, batch in enumerate(trainloader):
        inputs, labels = batch
        outputs = model_engine(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model_engine.backward(loss)    # DeepSpeed handles gradient scaling
        model_engine.step()            # DeepSpeed handles optimizer step

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

# Launch: deepspeed --num_gpus=8 train_ds.py
```

### 3. Apache Spark for Large-Scale Feature Engineering

```python
# pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
)
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ── Initialize Spark ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("MLFeatureEngineering") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ── Load large dataset (from S3, HDFS, or local) ─────────────
df = spark.read.parquet("s3://my-bucket/transactions/*.parquet")
print(f"Dataset size: {df.count():,} rows, {len(df.columns)} columns")

# ── Feature Engineering at Scale ─────────────────────────────
# Window functions for time-based features
user_window_7d = Window.partitionBy("user_id").orderBy("timestamp") \
    .rangeBetween(-7 * 86400, 0)  # 7 days in seconds

df_features = df.withColumn(
    "txn_count_7d", F.count("transaction_id").over(user_window_7d)
).withColumn(
    "avg_amount_7d", F.avg("amount").over(user_window_7d)
).withColumn(
    "max_amount_7d", F.max("amount").over(user_window_7d)
).withColumn(
    "days_since_first_txn",
    F.datediff(F.col("timestamp"), F.min("timestamp").over(
        Window.partitionBy("user_id")
    ))
)

# Aggregate features
user_agg = df.groupBy("user_id").agg(
    F.count("transaction_id").alias("total_txns"),
    F.sum("amount").alias("total_spend"),
    F.avg("amount").alias("avg_txn_amount"),
    F.stddev("amount").alias("std_txn_amount"),
    F.countDistinct("merchant_category").alias("category_diversity"),
    F.max("timestamp").alias("last_txn_date"),
    F.min("timestamp").alias("first_txn_date"),
)

# ── Spark ML Pipeline ─────────────────────────────────────────
categorical_cols = ["merchant_category", "card_type", "country"]
numeric_cols = ["amount", "txn_count_7d", "avg_amount_7d", "max_amount_7d"]

# String indexing + OHE for categoricals
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
            for c in categorical_cols]

# Assemble all features
all_feature_cols = numeric_cols + [f"{c}_ohe" for c in categorical_cols]
assembler = VectorAssembler(inputCols=all_feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")

# Gradient Boosted Trees
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    maxDepth=5,
    stepSize=0.1,
)

pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, gbt])

# Train / Test Split
train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

print("Training Spark ML Pipeline...")
model = pipeline.fit(train_df)

# Evaluate
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc:.4f}")

# Save model
model.write().overwrite().save("s3://my-bucket/models/fraud_pipeline")
```

---

## Phase A2: Kubernetes for ML Workloads

### Kubernetes Fundamentals for ML

```yaml
# ── Deployment for ML inference service ─────────────────────
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference-service
  namespace: ml-production
  labels:
    app: ml-inference
    version: v1.2.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
    type: RollingUpdate
  template:
    metadata:
      labels:
        app: ml-inference
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: inference
        image: your-registry/ml-inference:v1.2.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: "0"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: MODEL_PATH
          value: "/models/fraud_model_v2"
        - name: LOG_LEVEL
          value: "INFO"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ml-secrets
              key: redis-url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      nodeSelector:
        node-type: inference
      tolerations:
      - key: "inference-only"
        operator: "Exists"
        effect: "NoSchedule"

---
# ── Horizontal Pod Autoscaler ─────────────────────────────────
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
  namespace: ml-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: inference_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 4
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 120
```

### Kubeflow Pipelines

```python
# pip install kfp
import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (
    component, pipeline, Input, Output, Dataset, Model, Metrics, Artifact
)

# ── Define pipeline components ───────────────────────────────
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def preprocess_data(
    raw_data: Input[Dataset],
    processed_data: Output[Dataset],
    test_size: float = 0.2,
):
    """Preprocess training data."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(raw_data.path)

    # Feature engineering
    df["feature_ratio"] = df["feature_a"] / (df["feature_b"] + 1)
    df = df.fillna(df.median(numeric_only=True))

    train, test = train_test_split(df, test_size=test_size, random_state=42)
    train.to_parquet(processed_data.path + "_train.parquet")
    test.to_parquet(processed_data.path + "_test.parquet")

    processed_data.metadata["train_rows"] = len(train)
    processed_data.metadata["test_rows"] = len(test)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "lightgbm", "pyarrow", "scikit-learn"]
)
def train_model(
    processed_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_output: Output[Metrics],
    n_estimators: int = 300,
    learning_rate: float = 0.05,
):
    """Train LightGBM model."""
    import pandas as pd
    import lightgbm as lgb
    import pickle
    from sklearn.metrics import roc_auc_score

    train = pd.read_parquet(processed_data.path + "_train.parquet")
    test = pd.read_parquet(processed_data.path + "_test.parquet")

    feature_cols = [c for c in train.columns if c != "label"]
    X_train, y_train = train[feature_cols], train["label"]
    X_test, y_test = test[feature_cols], test["label"]

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    metrics_output.log_metric("auc_roc", auc)
    print(f"AUC-ROC: {auc:.4f}")

    with open(model_artifact.path, "wb") as f:
        pickle.dump(model, f)

    model_artifact.metadata["framework"] = "lightgbm"
    model_artifact.metadata["auc_roc"] = auc


@component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3", "pickle-mixin"]
)
def deploy_model(
    model_artifact: Input[Model],
    endpoint_url: str,
    deployment_name: str,
):
    """Deploy model to serving endpoint."""
    import requests
    import json

    payload = {
        "model_uri": model_artifact.uri,
        "deployment_name": deployment_name,
        "framework": model_artifact.metadata["framework"],
        "auc_roc": model_artifact.metadata["auc_roc"],
    }

    response = requests.post(
        f"{endpoint_url}/v1/deployments",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    print(f"Deployed: {response.json()}")


# ── Define the pipeline ───────────────────────────────────────
@pipeline(
    name="fraud-detection-pipeline",
    description="End-to-end fraud detection training pipeline"
)
def fraud_detection_pipeline(
    raw_data_path: str,
    endpoint_url: str,
    test_size: float = 0.2,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
):
    preprocess_task = preprocess_data(
        raw_data=raw_data_path,
        test_size=test_size
    )

    train_task = train_model(
        processed_data=preprocess_task.outputs["processed_data"],
        n_estimators=n_estimators,
        learning_rate=learning_rate,
    )
    train_task.set_cpu_limit("4")
    train_task.set_memory_limit("8G")
    train_task.set_gpu_limit(1)

    deploy_task = deploy_model(
        model_artifact=train_task.outputs["model_artifact"],
        endpoint_url=endpoint_url,
        deployment_name="fraud-model-v1",
    )
    deploy_task.after(train_task)


# Compile and submit
kfp.compiler.Compiler().compile(
    pipeline_func=fraud_detection_pipeline,
    package_path="fraud_pipeline.yaml"
)

client = kfp.Client(host="http://kubeflow-pipelines-host:80")
run = client.create_run_from_pipeline_func(
    fraud_detection_pipeline,
    arguments={
        "raw_data_path": "gs://my-bucket/data/transactions.parquet",
        "endpoint_url": "http://model-serving-service:8080",
    }
)
```

---

## Phase A3: LLM Fine-Tuning with LoRA/QLoRA

### QLoRA Fine-Tuning (Efficient Fine-Tuning on Consumer GPUs)

```python
# pip install transformers peft bitsandbytes trl datasets accelerate
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# ── 1. Configure 4-bit Quantization ──────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # Double quantization saves memory
)

# ── 2. Load base model in 4-bit ───────────────────────────────
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # or mistralai/Mistral-7B-Instruct-v0.3
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",       # Automatically distributes layers across GPUs
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)  # Enable gradient checkpointing

# ── 3. Configure LoRA ─────────────────────────────────────────
lora_config = LoraConfig(
    r=16,                          # Rank (higher = more params, better fit, slower)
    lora_alpha=32,                 # Scaling factor (alpha/r = effective learning rate)
    target_modules=[               # Which layers to adapt
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Trainable params: ~20M (vs 3B total) = 0.67% — HUGE memory saving!

# ── 4. Prepare dataset ────────────────────────────────────────
def format_instruction(example):
    """Format as Llama-3 instruction template."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that answers questions accurately.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{example['instruction']}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{example['output']}<|eot_id|>"""

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# ── 5. Training configuration ─────────────────────────────────
training_args = TrainingArguments(
    output_dir="./llama3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # Effective batch size = 4 * 4 = 16
    gradient_checkpointing=True,     # Save memory by recomputing activations
    optim="paged_adamw_32bit",       # Memory-efficient optimizer
    save_steps=100,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,                       # bfloat16 (better for modern GPUs)
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,            # Group similar-length sequences → fewer pads
    lr_scheduler_type="cosine",
    report_to="wandb",
    run_name="llama3-dolly-finetune",
)

# ── 6. SFT Trainer ────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=format_instruction,
    packing=True,   # Pack multiple short sequences into one → faster training
)

print("Starting QLoRA fine-tuning...")
trainer.train()

# ── 7. Save and merge ─────────────────────────────────────────
# Save LoRA adapter (tiny — ~100MB vs full model 13GB)
trainer.model.save_pretrained("./llama3-lora-adapter")
tokenizer.save_pretrained("./llama3-lora-adapter")

# Optionally merge adapter into base model for faster inference
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
merged_model = PeftModel.from_pretrained(base_model, "./llama3-lora-adapter")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("./llama3-finetuned-merged")
print("Fine-tuning complete!")
```

### Evaluate Fine-Tuned Model

```python
from transformers import pipeline
from datasets import load_dataset
import json

# Load fine-tuned model
generator = pipeline(
    "text-generation",
    model="./llama3-finetuned-merged",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

def evaluate_model(test_examples: list[dict], max_new_tokens: int = 256) -> dict:
    """Evaluate fine-tuned model on test examples."""
    results = []

    for example in test_examples:
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{example['instruction']}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        generated_text = output[0]["generated_text"].split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

        results.append({
            "instruction": example["instruction"],
            "expected": example.get("output", ""),
            "generated": generated_text,
        })

    return results

# Load test set
test_data = load_dataset("databricks/databricks-dolly-15k", split="train[-100:]")
results = evaluate_model(test_data.to_list()[:20])

for r in results[:3]:
    print(f"\nInstruction: {r['instruction'][:100]}...")
    print(f"Generated: {r['generated'][:200]}...")
    print("-" * 50)
```

---

## Phase A4: ML Platform Architecture

### Design Principles

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML PLATFORM ARCHITECTURE                     │
├──────────────────┬──────────────────┬──────────────────────────┤
│  DATA LAYER       │  COMPUTE LAYER    │  SERVING LAYER           │
│                  │                  │                          │
│ Data Lake (S3)   │ Training Cluster  │ Online Serving           │
│ Feature Store    │  - Kubernetes     │  - FastAPI               │
│  (Redis/Feast)   │  - Kubeflow       │  - TorchServe            │
│ Data Catalog     │  - Spark          │  - Triton Inference      │
│  (DataHub)       │                  │                          │
├──────────────────┼──────────────────┼──────────────────────────┤
│  ORCHESTRATION   │  MODEL REGISTRY   │  MONITORING              │
│                  │                  │                          │
│ Prefect / Airflow│ MLflow Registry  │ Prometheus + Grafana     │
│ GitHub Actions   │ Model Cards      │ Evidently AI             │
│ ArgoCD           │ A/B Experiments  │ PagerDuty Alerts         │
└──────────────────┴──────────────────┴──────────────────────────┘
```

### Model Serving with NVIDIA Triton

```python
# Triton Inference Server — ultra-high performance serving
# Supports: PyTorch, TensorFlow, ONNX, TensorRT, custom backends

# Model repository structure:
# model_repository/
# ├── fraud_detection/
# │   ├── config.pbtxt          ← Model configuration
# │   └── 1/                   ← Version 1
# │       └── model.onnx
# └── embedding_model/
#     ├── config.pbtxt
#     └── 1/
#         └── model.plan       ← TensorRT plan

# config.pbtxt (Triton model config)
TRITON_CONFIG = """
name: "fraud_detection"
platform: "onnxruntime_onnx"
max_batch_size: 256

input [
  {
    name: "input_features"
    data_type: TYPE_FP32
    dims: [50]
  }
]

output [
  {
    name: "fraud_probability"
    data_type: TYPE_FP32
    dims: [1]
  }
]

dynamic_batching {
  preferred_batch_size: [32, 64, 128]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

optimization {
  execution_accelerators {
    gpu_execution_accelerator : [
      {
        name : "tensorrt"
        parameters { key: "precision_mode" value: "FP16" }
        parameters { key: "max_workspace_size_bytes" value: "1073741824" }
      }
    ]
  }
}
"""

# Python client to query Triton
import tritonclient.grpc as triton_grpc
import numpy as np

def predict_with_triton(features: np.ndarray, server_url: str = "localhost:8001"):
    """Send inference request to Triton."""
    client = triton_grpc.InferenceServerClient(url=server_url)

    # Check server health
    assert client.is_server_live(), "Triton server is not live!"
    assert client.is_model_ready("fraud_detection"), "Model is not ready!"

    # Prepare input
    input_tensor = triton_grpc.InferInput("input_features", features.shape, "FP32")
    input_tensor.set_data_from_numpy(features)

    output = triton_grpc.InferRequestedOutput("fraud_probability")

    # Send request
    response = client.infer(
        model_name="fraud_detection",
        model_version="1",
        inputs=[input_tensor],
        outputs=[output],
    )

    predictions = response.as_numpy("fraud_probability")
    return predictions


# Benchmark Triton throughput
import time
import threading

def benchmark_triton(n_requests: int = 1000, n_threads: int = 10):
    results = []

    def worker():
        for _ in range(n_requests // n_threads):
            features = np.random.randn(32, 50).astype(np.float32)  # Batch of 32
            start = time.perf_counter()
            _ = predict_with_triton(features)
            results.append(time.perf_counter() - start)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    overall_start = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    total_time = time.perf_counter() - overall_start

    print(f"Total requests: {len(results)}")
    print(f"Throughput: {len(results) / total_time:.0f} req/s")
    print(f"P50 latency: {sorted(results)[len(results)//2]*1000:.1f}ms")
    print(f"P99 latency: {sorted(results)[int(len(results)*0.99)]*1000:.1f}ms")
```

### Shadow Mode Deployment (Safe Model Updates)

```python
import asyncio
import httpx
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class ShadowConfig:
    primary_url: str       # Production model (serves real traffic)
    shadow_url: str        # New model (receives mirrored traffic, responses ignored)
    shadow_percentage: float = 1.0  # 100% of traffic is shadowed
    log_discrepancies: bool = True
    discrepancy_threshold: float = 0.1  # Flag if predictions differ by >10%

class ShadowModeProxy:
    """
    Proxy that mirrors traffic to a shadow model.
    Primary model response is always returned to the client.
    Shadow model runs in background for comparison.
    """

    def __init__(self, config: ShadowConfig):
        self.config = config
        self.primary_client = httpx.AsyncClient(base_url=config.primary_url)
        self.shadow_client = httpx.AsyncClient(base_url=config.shadow_url)
        self.shadow_metrics = {"total": 0, "discrepancies": 0, "shadow_errors": 0}

    async def predict(self, features: dict) -> dict:
        """Forward to primary; shadow runs in background."""
        import random

        # Always call primary (blocking)
        primary_response = await self.primary_client.post("/predict", json=features)
        primary_result = primary_response.json()

        # Shadow call (fire and forget if within shadow percentage)
        if random.random() < self.config.shadow_percentage:
            asyncio.create_task(self._shadow_predict(features, primary_result))

        return primary_result

    async def _shadow_predict(self, features: dict, primary_result: dict):
        """Call shadow model and compare results."""
        try:
            shadow_response = await self.shadow_client.post("/predict", json=features)
            shadow_result = shadow_response.json()

            self.shadow_metrics["total"] += 1

            # Check for significant discrepancy
            primary_score = primary_result.get("fraud_probability", 0)
            shadow_score = shadow_result.get("fraud_probability", 0)
            diff = abs(primary_score - shadow_score)

            if diff > self.config.discrepancy_threshold:
                self.shadow_metrics["discrepancies"] += 1
                if self.config.log_discrepancies:
                    logger.warning(
                        f"Shadow discrepancy: primary={primary_score:.3f}, "
                        f"shadow={shadow_score:.3f}, diff={diff:.3f}"
                    )

        except Exception as e:
            self.shadow_metrics["shadow_errors"] += 1
            logger.error(f"Shadow call failed: {e}")

    def get_metrics(self) -> dict:
        total = self.shadow_metrics["total"]
        if total == 0:
            return self.shadow_metrics

        return {
            **self.shadow_metrics,
            "discrepancy_rate": self.shadow_metrics["discrepancies"] / total,
            "shadow_error_rate": self.shadow_metrics["shadow_errors"] / total,
        }
```

---

## Advanced ML Concepts Quick Reference

| Concept | Use Case | Key Library |
|---------|----------|-------------|
| **DDP** | Multi-GPU single node | `torch.distributed` |
| **FSDP** | Model too large for 1 GPU | `torch.distributed.fsdp` |
| **DeepSpeed ZeRO** | Billion-parameter training | `deepspeed` |
| **LoRA** | Efficient fine-tuning | `peft` |
| **QLoRA** | 4-bit fine-tuning on consumer GPUs | `peft + bitsandbytes` |
| **ONNX** | Cross-framework portability | `torch.onnx` |
| **TensorRT** | NVIDIA GPU inference optimization | `tensorrt` |
| **Triton** | High-throughput model serving | `tritonclient` |
| **Kubeflow** | K8s-native ML pipelines | `kfp` |
| **Feast** | Production feature store | `feast` |

---

*Back to: [MLE Track](../README.md) | [Main README](../../README.md)*