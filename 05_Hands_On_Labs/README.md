# Hands-On Labs

> **"The only way to truly learn is by doing."**

This section contains standalone notebooks and projects that you can use to practice concepts from any track. Each lab is self-contained with its own instructions and data.

---

## How to Run These Labs

### Option 1: Google Colab (Zero Setup — Recommended)
Every notebook has a "Open in Colab" button. Click it, no installation needed.

### Option 2: GitHub Codespaces (Consistent Environment)
1. Click the green "Code" button on the main repo page
2. Select "Codespaces" → "Create codespace on main"
3. Wait ~2 minutes for setup
4. Navigate to any notebook and run it

### Option 3: Local
```bash
git clone https://github.com/yourusername/ai-ml-ds-hub.git
cd "ai-ml-ds-hub"
pip install -r requirements.txt
jupyter lab
```

---

## Interactive Notebooks

### Beginner Level

| Notebook | Topics | Time | Track |
|----------|--------|------|-------|
| [01_eda_from_scratch.ipynb](notebooks/01_eda_from_scratch.ipynb) | pandas, visualization, EDA workflow | 2-3 hrs | DS |
| [02_stats_intuition.ipynb](notebooks/02_stats_intuition.ipynb) | Distributions, CLT, hypothesis testing | 2 hrs | DS |
| [03_python_data_structures.ipynb](notebooks/03_python_data_structures.ipynb) | Lists, dicts, sets, comprehensions | 1-2 hrs | All |
| [04_sql_for_analysts.ipynb](notebooks/04_sql_for_analysts.ipynb) | SQLite + pandas + common queries | 2 hrs | DS, MLE |
| [05_first_ml_model.ipynb](notebooks/05_first_ml_model.ipynb) | sklearn, train/test split, metrics | 1-2 hrs | DS, MLE |
| [06_first_llm_app.ipynb](notebooks/06_first_llm_app.ipynb) | OpenAI API, prompting, chatbot | 1-2 hrs | AIE |

### Intermediate Level

| Notebook | Topics | Time | Track |
|----------|--------|------|-------|
| [07_feature_engineering.ipynb](notebooks/07_feature_engineering.ipynb) | Feature creation, encoding, scaling | 2-3 hrs | DS |
| [08_model_evaluation.ipynb](notebooks/08_model_evaluation.ipynb) | CV, AUC-ROC, precision-recall, SHAP | 2-3 hrs | DS, MLE |
| [09_neural_network_scratch.ipynb](notebooks/09_neural_network_scratch.ipynb) | Backprop + numpy neural net | 2-3 hrs | DS, MLE |
| [10_pytorch_intro.ipynb](notebooks/10_pytorch_intro.ipynb) | Tensors, autograd, training loop | 2-3 hrs | MLE |
| [11_build_ml_api.ipynb](notebooks/11_build_ml_api.ipynb) | FastAPI + sklearn model serving | 2 hrs | MLE |
| [12_rag_pipeline.ipynb](notebooks/12_rag_pipeline.ipynb) | RAG with Chroma + LangChain | 2-3 hrs | AIE |
| [13_prompt_engineering.ipynb](notebooks/13_prompt_engineering.ipynb) | Zero/few-shot, CoT, system prompts | 2-3 hrs | AIE |

### Advanced Level

| Notebook | Topics | Time | Track |
|----------|--------|------|-------|
| [14_fine_tune_bert.ipynb](notebooks/14_fine_tune_bert.ipynb) | BERT fine-tuning with Hugging Face | 3-4 hrs | DS, MLE |
| [15_llm_fine_tuning_lora.ipynb](notebooks/15_llm_fine_tuning_lora.ipynb) | LoRA fine-tuning on Llama | 4-6 hrs | MLE, AIE |
| [16_mlops_pipeline.ipynb](notebooks/16_mlops_pipeline.ipynb) | MLflow + DVC + automated retraining | 3-4 hrs | MLE |
| [17_multi_agent_system.ipynb](notebooks/17_multi_agent_system.ipynb) | LangGraph multi-agent workflow | 3-4 hrs | AIE |
| [18_ab_testing_stats.ipynb](notebooks/18_ab_testing_stats.ipynb) | Full A/B test design + analysis | 3-4 hrs | DS |
| [19_transformer_from_scratch.ipynb](notebooks/19_transformer_from_scratch.ipynb) | Build transformer in PyTorch | 4-6 hrs | MLE |
| [20_llm_eval_framework.ipynb](notebooks/20_llm_eval_framework.ipynb) | Build custom LLM evaluation suite | 3-4 hrs | AIE |

---

## Mini-Projects (1-3 Days Each)

Quick projects that cement a specific skill without requiring a full end-to-end effort.

### Data Science Mini-Projects
| Project | What You Build | Skills |
|---------|---------------|--------|
| [Titanic Survival Analysis](mini_projects/ds_01_titanic.md) | Complete EDA + prediction + visualization | pandas, sklearn, matplotlib |
| [COVID Dashboard](mini_projects/ds_02_covid_dashboard.md) | Interactive dashboard with Streamlit | pandas, Plotly, Streamlit |
| [Text Sentiment Analyzer](mini_projects/ds_03_sentiment.md) | Twitter sentiment with HF transformers | NLP, Transformers |
| [Sales Forecaster](mini_projects/ds_04_forecasting.md) | Time series model with Prophet | Prophet, pandas, evaluation |
| [Customer Segments](mini_projects/ds_05_clustering.md) | K-means clustering with interpretation | sklearn, viz, business framing |

### ML Engineer Mini-Projects
| Project | What You Build | Skills |
|---------|---------------|--------|
| [Model API](mini_projects/mle_01_model_api.md) | FastAPI endpoint with Docker | FastAPI, Docker, sklearn |
| [Experiment Tracker](mini_projects/mle_02_experiment_tracker.md) | MLflow tracking for 5 experiments | MLflow, sklearn |
| [Data Drift Detector](mini_projects/mle_03_drift_detection.md) | Drift detection with Evidently | Evidently AI, pandas |
| [Automated Retraining](mini_projects/mle_04_retraining.md) | Scheduled model retraining | Airflow/Prefect, MLflow |
| [Model Benchmark](mini_projects/mle_05_benchmark.md) | Latency benchmark for 3 serving methods | FastAPI, BentoML, Docker |

### AI Engineer Mini-Projects
| Project | What You Build | Skills |
|---------|---------------|--------|
| [Personal Q&A Bot](mini_projects/aie_01_qa_bot.md) | RAG over your own PDF docs | LangChain, Chroma, OpenAI |
| [Code Review Agent](mini_projects/aie_02_code_review.md) | AI code reviewer via GitHub Actions | LLM API, GitHub Actions |
| [Research Agent](mini_projects/aie_03_research_agent.md) | Agent that searches + summarizes | LangChain Agents, Tavily |
| [AI-Powered CLI](mini_projects/aie_04_cli.md) | Command-line AI assistant | Anthropic API, Typer |
| [LLM Comparison Tool](mini_projects/aie_05_comparison.md) | Compare GPT vs Claude vs Gemini | Multiple APIs, Streamlit |

---

## Challenge of the Week

> Fork this repo, solve the challenge, and share your solution in GitHub Discussions!

Current challenge: **Build a Mini-RAG system with 3 chunking strategies and benchmark which retrieves better on 10 test questions.**

Past challenges are archived in [challenges/](challenges/).

---

## Tips for Getting the Most from Labs

1. **Don't just run the cells** — understand each line before moving to the next
2. **Break things intentionally** — remove a step and see what breaks
3. **Change the data** — try the notebook on a different dataset
4. **Extend the notebook** — add one more analysis beyond what's shown
5. **Share what you build** — post your extended version in GitHub Discussions

---

*Back to: [Main README](../README.md)*
