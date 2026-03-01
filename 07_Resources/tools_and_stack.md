# The Modern AI/ML/DS Toolbox (2025)

> The definitive guide to tools you actually use on the job — organized by category and role.

---

## Programming Languages

| Language | Relevance | Who Uses It |
|----------|-----------|------------|
| **Python** | Essential | DS, MLE, AIE — primary language |
| **SQL** | Essential | DS, MLE — data querying |
| **R** | Useful for DS | DS in academia, pharma, finance |
| **Scala** | Useful for MLE | MLE working with Spark |
| **Julia** | Niche | Academic/scientific computing |
| **Bash/Shell** | Important | MLE — scripting, automation |
| **JavaScript/TS** | Growing | AIE — web-integrated AI products |
| **Rust** | Emerging | MLE — high-performance inference |
| **Go** | Niche | MLE — infrastructure tools |
| **C++** | Specialized | MLE — custom CUDA kernels, inference |

---

## Core Python Data Science Libraries

### Data Manipulation
| Library | Use Case | Level |
|---------|----------|-------|
| **pandas** | DataFrames, data cleaning, analysis | Essential |
| **NumPy** | Numerical arrays, math operations | Essential |
| **polars** | Fast pandas alternative (Rust-based) | Growing fast |
| **Dask** | Parallel/distributed pandas-like | Large data |
| **Vaex** | Out-of-core DataFrames | Very large datasets |
| **PyArrow** | Columnar memory format | Interoperability |

### Visualization
| Library | Use Case | Level |
|---------|----------|-------|
| **matplotlib** | Base plotting library | Essential |
| **seaborn** | Statistical visualization (on top of matplotlib) | Essential |
| **plotly** | Interactive charts | Very useful |
| **altair** | Declarative, elegant charts | Good alternative |
| **bokeh** | Interactive web visualization | Useful |
| **streamlit** | Quick data apps and dashboards | Very useful |
| **gradio** | ML model demos | AI Engineers |

---

## Machine Learning Libraries

### Classical ML
| Library | Use Case |
|---------|----------|
| **scikit-learn** | The ML workhorse — everything from LinearRegression to SVMs |
| **XGBoost** | Gradient boosting — often wins Kaggle tabular competitions |
| **LightGBM** | Faster than XGBoost for large datasets |
| **CatBoost** | Best for categorical features |
| **statsmodels** | Statistical models, hypothesis testing |

### Deep Learning
| Library | Use Case | Notes |
|---------|----------|-------|
| **PyTorch** | Deep learning research + production | The industry standard now |
| **TensorFlow** | Deep learning | Still widely used, especially in legacy |
| **Keras** | High-level DL API (now part of TF + standalone) | Easiest to start with |
| **JAX** | High-performance numerical computing | Google, research focus |
| **Flax** | Neural networks on JAX | Google research |
| **Lightning (PyTorch)** | Clean PyTorch training loops | Highly recommended |

### NLP
| Library | Use Case |
|---------|----------|
| **Hugging Face Transformers** | Pre-trained models for NLP/vision/audio |
| **PEFT** | Parameter-efficient fine-tuning (LoRA, etc.) |
| **TRL** | RLHF and fine-tuning with human feedback |
| **Tokenizers** | Fast tokenization |
| **Datasets (HF)** | Easy dataset loading |
| **spaCy** | Industrial NLP pipeline |
| **NLTK** | Educational NLP toolkit |
| **Gensim** | Topic modeling, word2vec |

---

## MLOps Tools

### Experiment Tracking
| Tool | Notes |
|------|-------|
| **MLflow** | Open-source, most widely used |
| **Weights & Biases (W&B)** | Best UI/UX, industry favorite |
| **Neptune.ai** | Good alternative |
| **Comet ML** | Feature-rich |
| **DVC** | Data and model versioning (Git for ML) |
| **ClearML** | Full MLOps platform |

### Pipeline Orchestration
| Tool | Notes |
|------|-------|
| **Apache Airflow** | Most mature, most widely deployed |
| **Prefect** | Modern, Python-native, excellent DX |
| **Dagster** | Asset-based orchestration, growing fast |
| **Metaflow** | Netflix-built, great for DS |
| **Kubeflow Pipelines** | Kubernetes-native |
| **ZenML** | MLOps framework (pipeline + artifact tracking) |

### Model Serving
| Tool | Use Case |
|------|----------|
| **FastAPI** | Build ML APIs — the go-to choice |
| **BentoML** | Model serving with built-in packaging |
| **TorchServe** | PyTorch model serving |
| **TF Serving** | TensorFlow model serving |
| **Triton Inference Server** | NVIDIA, high-performance multi-framework |
| **vLLM** | Fast LLM inference (PagedAttention) |
| **Ollama** | Run LLMs locally |
| **Modal** | Serverless GPU functions for ML |
| **Replicate** | Run ML models via API |

### Feature Stores
| Tool | Notes |
|------|-------|
| **Feast** | Open-source, widely used |
| **Tecton** | Managed feature platform |
| **Hopsworks** | Open-source full ML platform |

### Model Monitoring
| Tool | Use Case |
|------|----------|
| **Evidently AI** | Open-source data/model drift detection |
| **Arize AI** | Production ML monitoring |
| **WhyLabs** | ML observability |
| **Grafana + Prometheus** | General metrics monitoring |

---

## AI Engineering Tools

### LLM APIs
| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | GPT-4o, GPT-4.1, o3, o4 | Most widely used |
| **Anthropic** | Claude Sonnet 4.6, Claude Opus 4.6 | Strong reasoning, long context |
| **Google** | Gemini 2.0, Gemini 2.5 Pro | Strong for multimodal |
| **Mistral AI** | Mistral Large, Codestral | Good open-source models |
| **Cohere** | Command R+, Embed | RAG-optimized |
| **Together AI** | Llama, Mixtral, etc. | Cheap open-source model API |
| **Groq** | Llama, Mixtral | Extremely fast inference (LPU) |
| **Ollama** | Any open-source model | Local inference, free |
| **Hugging Face** | Thousands of models | Model hub + inference API |

### LLM Application Frameworks
| Tool | Use Case |
|------|----------|
| **LangChain** | LLM application framework — most popular |
| **LlamaIndex** | Data framework for LLM apps — especially RAG |
| **AutoGen** | Multi-agent conversational AI (Microsoft) |
| **CrewAI** | Role-based multi-agent framework |
| **Haystack** | Production NLP pipelines |
| **Semantic Kernel** | LLM orchestration (Microsoft) |
| **Instructor** | Structured LLM output with Pydantic |
| **Outlines** | Structured text generation |
| **DSPy** | Programmatic prompting (Stanford) |
| **LangGraph** | Stateful agent graphs (LangChain) |

### Vector Databases
| Tool | Notes |
|------|-------|
| **Chroma** | Local-first, excellent for dev/prototyping |
| **Pinecone** | Managed cloud vector DB, production-ready |
| **Weaviate** | Open-source, feature-rich |
| **Qdrant** | High-performance, Rust-based |
| **Milvus** | Large-scale vector search |
| **pgvector** | Vector extension for PostgreSQL |
| **Faiss** | Facebook's in-memory similarity search |

### LLM Evaluation & Observability
| Tool | Use Case |
|------|----------|
| **RAGAS** | RAG evaluation metrics |
| **DeepEval** | LLM testing framework |
| **LangSmith** | Tracing, debugging, evaluation for LangChain apps |
| **Arize Phoenix** | LLM observability and tracing |
| **Promptfoo** | CLI tool for testing prompts |
| **Helicone** | LLM observability + cost tracking |
| **Langfuse** | Open-source LLM observability |

---

## Infrastructure & DevOps (for MLEs)

| Tool | Category |
|------|----------|
| **Docker** | Containerization — essential |
| **Kubernetes** | Container orchestration at scale |
| **Terraform** | Infrastructure as Code |
| **Helm** | Kubernetes package manager |
| **GitHub Actions** | CI/CD — most common for ML |
| **GitLab CI** | CI/CD alternative |
| **ArgoCD** | GitOps for Kubernetes |
| **Ray** | Distributed Python/ML computing |
| **Spark** | Distributed data processing |
| **Kafka** | Real-time data streaming |
| **dbt** | Data transformation |
| **Airflow** | Workflow orchestration |

---

## Cloud ML Platforms

| Platform | Provider | Key Services |
|----------|----------|-------------|
| **AWS SageMaker** | Amazon | Training, endpoint, pipelines, Studio |
| **GCP Vertex AI** | Google | Training, serving, Feature Store, Workbench |
| **Azure ML** | Microsoft | Designer, SDK, pipelines, compute |
| **Databricks** | Open + Azure/AWS/GCP | MLflow + Spark + notebooks |
| **Modal** | Startup | Serverless GPU for ML inference |
| **Lambda Labs** | Startup | Cheap GPU cloud |
| **Vast.ai** | Marketplace | GPU rental marketplace |

---

## Development Environment

| Tool | Use |
|------|-----|
| **VS Code** | Best general-purpose IDE for DS/ML |
| **JupyterLab** | Notebooks — essential |
| **Google Colab** | Free GPU, zero setup |
| **Kaggle Kernels** | Free GPU, integrated with data |
| **GitHub Codespaces** | Cloud VS Code environment |
| **PyCharm** | Python IDE (heavier than VS Code) |
| **Cursor** | AI-powered code editor |
| **Conda** | Environment management |
| **uv** | Ultra-fast Python package manager (2024) |
| **Poetry** | Dependency management |
| **pre-commit** | Automated code quality checks |
| **ruff** | Fast Python linter |
| **black** | Python code formatter |
| **mypy** | Static type checking |

---

## Databases

| Tool | Type | Notes |
|------|------|-------|
| **PostgreSQL** | Relational | Most popular open-source RDBMS |
| **MySQL** | Relational | Widely used, simpler than PG |
| **SQLite** | Embedded relational | Local, zero config |
| **DuckDB** | Analytical SQL | In-process analytical DB — amazing for DS |
| **BigQuery** | Cloud data warehouse | Google, SQL interface |
| **Snowflake** | Cloud data warehouse | Cloud-native |
| **Redshift** | Cloud data warehouse | AWS |
| **MongoDB** | Document (NoSQL) | Flexible schema |
| **Redis** | Key-value (in-memory) | Caching, session, pub/sub |
| **Cassandra** | Wide-column | High write throughput |
| **Neo4j** | Graph database | Knowledge graphs |

---

*Back to: [Resources](.) | [Main README](../README.md)*
