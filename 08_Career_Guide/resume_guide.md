# Resume Guide for DS / MLE / AIE Roles

> A resume is a marketing document, not a biography. Every word should argue why you're the best candidate for this role.

---

## Resume Structure (Order Matters)

```
YOUR NAME
Location | Email | LinkedIn | GitHub | Portfolio URL
─────────────────────────────────────────────────────────
SUMMARY (3 lines max — optional but recommended)

SKILLS (Technical skills, not soft skills)

EXPERIENCE (Most important section)
  Company | Title | Dates
  • Impact bullet 1
  • Impact bullet 2

PROJECTS (Critical for entry/mid-level)

EDUCATION (Below experience if you have 2+ years)

CERTIFICATIONS (If relevant)
```

---

## The Impact Formula for Bullet Points

**Formula:** `Action verb → [What you did] → [How you did it] → [Quantified result]`

| Weak | Strong |
|------|--------|
| "Worked on ML models" | "Built fraud detection model (LightGBM) reducing false positives by 35% at 99th percentile recall" |
| "Used Python for data analysis" | "Processed 50M daily events with DuckDB + Polars, cutting pipeline runtime from 4h to 18min" |
| "Helped improve recommendation system" | "Redesigned two-tower embedding model (PyTorch DDP, 4xA100), lifting CTR 8.3% ($2.1M ARR impact)" |
| "Built a chatbot" | "Deployed RAG-based customer support agent (Claude API + RAGAS eval 0.91) automating 62% of Tier-1 tickets" |
| "Improved model accuracy" | "Increased AUC from 0.82 → 0.91 via feature engineering (20 new features) + Optuna hypertuning" |

---

## Bullet Point Bank by Role

### Data Scientist Bullets

```
ANALYSIS
• Conducted EDA on 15M+ customer records, uncovering 3 key churn signals
  that drove retention strategy (12% churn reduction, $1.8M saved annually)

A/B TESTING
• Designed and analyzed 6 A/B tests using Bayesian methods, reducing
  experiment runtime by 40% vs. frequentist approach (n=200K+/experiment)

MACHINE LEARNING
• Built gradient boosting model (XGBoost + Optuna) for credit risk scoring;
  improved Gini coefficient from 0.52 → 0.68 on holdout set

FEATURE ENGINEERING
• Engineered 45 features from raw click-stream data using DuckDB window
  functions; top-3 features contributed 60% of model lift

FORECASTING
• Developed SARIMA + XGBoost ensemble for 52-week demand forecasting;
  reduced MAPE from 18% to 9.3% vs. baseline
```

### ML Engineer Bullets

```
MLOps
• Migrated ML training workflows from ad-hoc scripts to Prefect pipelines;
  reduced model-to-production time from 3 weeks to 2 days

SERVING
• Built real-time inference API (FastAPI + Triton) serving 2M predictions/day
  at <35ms P99 latency; implemented auto-scaling (2→20 replicas)

DISTRIBUTED TRAINING
• Implemented PyTorch DDP training across 8 A100 GPUs; reduced training time
  from 14h to 2h for 500M-parameter model

MONITORING
• Deployed Evidently AI drift detection with automated weekly reports;
  caught 2 data quality incidents before they impacted model performance

CI/CD
• Built GitHub Actions pipeline (test → train → evaluate → deploy) with
  automated model quality gate; eliminated manual deployment steps
```

### AI Engineer Bullets

```
LLM SYSTEMS
• Built document Q&A system (RAG + reranking + RAGAS eval) for 50K legal docs;
  achieved 0.89 faithfulness, cut lawyer research time by 65%

FINE-TUNING
• Fine-tuned Llama-3-8B with QLoRA on 15K domain-specific examples;
  outperformed GPT-4o on internal benchmarks while reducing API cost 90%

AGENTS
• Designed multi-agent research pipeline (LangGraph + 4 specialized agents);
  automated 70% of competitive intelligence reports

EVALUATION
• Built LLM evaluation harness (100-case golden dataset + LLM-as-judge);
  used to compare 5 models across accuracy, faithfulness, and cost metrics

PROMPT ENGINEERING
• Reduced hallucination rate from 23% to 4% via structured output enforcement,
  chain-of-thought prompting, and context grounding
```

---

## Data Science Resume Template

```
ALEX CHEN
San Francisco, CA | alex@email.com
linkedin.com/in/alexchen | github.com/alexchen | alexchen.xyz

──────────────────────────────────────────────────────────────
SUMMARY
Data Scientist with 3 years of experience building ML models and
data pipelines for e-commerce and fintech. Strong in Python, SQL,
and classical ML. Passionate about turning messy data into decisions.
──────────────────────────────────────────────────────────────
SKILLS
Python (NumPy, pandas, scikit-learn, XGBoost, LightGBM, PyTorch)
SQL (PostgreSQL, BigQuery, DuckDB, window functions, CTEs)
ML/Stats: A/B testing, time series, NLP, feature engineering, SHAP
MLOps: MLflow, DVC, Docker, GitHub Actions, Prefect
Visualization: matplotlib, seaborn, Plotly, Streamlit, Tableau
──────────────────────────────────────────────────────────────
EXPERIENCE

Data Scientist | Acme Corp | Jan 2023 – Present
• Built churn prediction model (LightGBM) identifying 15K at-risk customers
  monthly; retention campaigns reduced churn by 18% ($3.2M ARR impact)
• Designed 12 A/B experiments with power analysis, reducing false positives
  by 30% vs previous methodology
• Automated weekly data quality checks (Great Expectations), catching 4
  data incidents before they reached production dashboards
• Created Streamlit dashboard for marketing team; eliminated 8h/week of
  manual reporting

Data Analyst | StartupXYZ | Jun 2021 – Dec 2022
• Wrote 50+ complex SQL queries analyzing 100M+ events/month in BigQuery
• Built product analytics dashboard (Looker); adopted by 30+ stakeholders
• Identified $400K revenue leak through cohort analysis of subscription data
──────────────────────────────────────────────────────────────
PROJECTS

Titanic Survival Prediction | github.com/alexchen/titanic
• Ranked top 8% on Kaggle (public score: 0.804)
• Feature engineering: title extraction, family size, fare binning → AUC 0.78→0.85

Customer Churn Prediction | github.com/alexchen/churn
• End-to-end project with business framing, EDA, modeling, and deployment
• Deployed Streamlit app on HuggingFace Spaces; 500+ demo users
──────────────────────────────────────────────────────────────
EDUCATION
B.S. Computer Science | State University | 2021 | GPA: 3.7
──────────────────────────────────────────────────────────────
CERTIFICATIONS
AWS Machine Learning Specialty | Google Professional Data Engineer
```

---

## ML Engineer Resume Template

```
MORGAN PATEL
Remote | morgan@email.com
linkedin.com/in/morganpatel | github.com/morganpatel

──────────────────────────────────────────────────────────────
SUMMARY
ML Engineer with 5 years building and deploying production ML systems.
Expert in Python, PyTorch, and MLOps. Shipped models serving 10M+ daily
predictions. Prior background in backend engineering.
──────────────────────────────────────────────────────────────
SKILLS
Languages: Python (expert), Go, SQL
ML Frameworks: PyTorch, scikit-learn, XGBoost, LightGBM, Hugging Face
MLOps: MLflow, DVC, Prefect/Airflow, Evidently, Great Expectations
Serving: FastAPI, TorchServe, Triton Inference, ONNX, TensorRT
Infrastructure: Kubernetes, Docker, AWS (ECS, SageMaker, S3), Terraform
Monitoring: Prometheus, Grafana, PagerDuty
──────────────────────────────────────────────────────────────
EXPERIENCE

Senior ML Engineer | TechCorp | Mar 2022 – Present
• Led migration from Jupyter-notebook-based training to Kubeflow Pipelines;
  reduced deployment cycle from 3 weeks to 4 hours
• Built real-time fraud scoring service (FastAPI + Redis + LightGBM);
  2M transactions/day at 28ms P99, 99.97% uptime over 18 months
• Implemented QLoRA fine-tuning pipeline for Llama-3-8B on 50K domain
  examples; reduced GPT-4 API costs by $120K/year
• Designed feature store (Feast + Redis) serving 35 features at <8ms P99;
  eliminated train-serve skew across 12 production models

ML Engineer | FinanceStartup | Jul 2020 – Feb 2022
• Built credit risk model pipeline with DVC + GitHub Actions + SageMaker;
  went from monthly to daily model retraining
• Implemented Evidently drift monitoring catching 3 data quality issues
  within 2 hours vs previous 1-week detection lag
• Reduced inference cost 60% via model distillation (student from LGBM →
  linear model with maintained Gini coefficient)
──────────────────────────────────────────────────────────────
PROJECTS
github.com/morganpatel/ml-platform: Open-source ML platform template
  (200+ stars) — MLflow + DVC + FastAPI + Docker + GitHub Actions

github.com/morganpatel/qlora-recipes: QLoRA fine-tuning cookbook
  for Llama/Mistral/Phi models (150+ stars)
──────────────────────────────────────────────────────────────
EDUCATION
B.S. Software Engineering | Tech University | 2020
```

---

## AI Engineer Resume Template

```
SAM RODRIGUEZ
New York, NY | sam@email.com
linkedin.com/in/samrodriguez | github.com/samrodriguez | samrodriguez.ai

──────────────────────────────────────────────────────────────
SUMMARY
AI Engineer with 3 years building LLM-powered applications. Expert in
RAG, agents, and prompt engineering. Shipped production AI systems with
measurable business impact. Strong evaluator — I care about what works,
not just what demos well.
──────────────────────────────────────────────────────────────
SKILLS
LLMs: OpenAI (GPT-4o/o1), Anthropic (Claude), Llama, Mistral, Gemini
Frameworks: LangChain, LlamaIndex, LangGraph, CrewAI, Guardrails AI
RAG Stack: ChromaDB, Pinecone, Weaviate, pgvector, FAISS, BM25
Fine-Tuning: QLoRA, PEFT, TRL, Unsloth, Axolotl
Evaluation: RAGAS, DeepEval, LangSmith, Braintrust, custom LLM-as-judge
Backend: FastAPI, Python, PostgreSQL, Redis, Docker
──────────────────────────────────────────────────────────────
EXPERIENCE

AI Engineer | LegalTech Inc | Feb 2023 – Present
• Built document Q&A system over 200K legal documents (RAG + reranking);
  RAGAS faithfulness 0.91, reduced lawyer research time by 65%
• Designed multi-agent contract review system (LangGraph, 4 agents);
  automated 80% of standard contract redlines, saving 4h/contract
• Led prompt optimization initiative (50-case golden dataset + GPT-4 judge);
  improved answer accuracy from 71% to 89% over 3 months
• Implemented hallucination detection pipeline (cross-reference + citation
  grounding); reduced ungrounded claims from 15% to 2.3%

AI Engineer | StartupAI | Jan 2022 – Jan 2023
• Built customer support chatbot (Claude API + RAG) automating 62% of
  Tier-1 tickets; CSAT maintained at 4.3/5 vs 4.1 for human agents
• Fine-tuned Llama-3-8B on 15K domain examples (QLoRA); matched
  GPT-4 performance on internal benchmarks at 1/10th the API cost
• Created eval framework comparing 6 LLM providers across 8 metrics;
  used by team for all model selection decisions
──────────────────────────────────────────────────────────────
PROJECTS

github.com/samrodriguez/rag-evals: RAG evaluation toolkit (300+ stars)
  — RAGAS wrappers, golden dataset builder, dashboard

samrodriguez.ai/demos: Live demos of 5 AI projects

github.com/samrodriguez/agent-cookbook: LangGraph agent patterns
──────────────────────────────────────────────────────────────
EDUCATION
B.S. Computer Science | University | 2022
```

---

## ATS (Applicant Tracking System) Optimization

Most large companies use ATS to filter resumes before a human reads them.

**ATS checklist:**
- [ ] Use standard section headers (EXPERIENCE, EDUCATION, SKILLS)
- [ ] No tables, columns, graphics, or headers/footers
- [ ] Submit as PDF (preserves formatting)
- [ ] Include job-specific keywords from the JD
- [ ] Spell out acronyms at least once (ML, LLM, CI/CD)
- [ ] Use standard fonts (Arial, Calibri, Georgia)
- [ ] Don't put important info in headers/footers

**Keyword extraction from job descriptions:**

```python
# Quick script to extract keywords from job description
from collections import Counter
import re

def extract_keywords(job_description: str) -> list[tuple[str, int]]:
    """Extract technical keywords from a job description."""
    # Common tech keywords to look for
    tech_terms = re.findall(
        r'\b(Python|SQL|PyTorch|TensorFlow|MLflow|Kubernetes|Docker|'
        r'AWS|GCP|Azure|Spark|Kafka|RAG|LLM|FastAPI|scikit-learn|'
        r'XGBoost|LightGBM|BERT|GPT|Transformer|A/B test(?:ing)?|'
        r'LangChain|LangGraph|Pinecone|ChromaDB|FAISS|OpenAI|'
        r'Anthropic|Hugging Face|Prefect|Airflow|DVC|Evidently)\b',
        job_description,
        re.IGNORECASE
    )
    return Counter(tech_terms).most_common(20)

# Example usage:
jd = """We are looking for an ML Engineer with Python, PyTorch, MLflow,
Kubernetes experience. Knowledge of LLMs and RAG is a plus."""
print(extract_keywords(jd))
```

---

## Common Resume Mistakes

| Mistake | Fix |
|---------|-----|
| Responsibilities, not impact | Lead with impact metrics (%, $, time saved) |
| "Proficient in Python" | Show Python in work bullet points instead |
| 3-page resume | Keep to 1 page (< 5 years), 2 pages (5+ years) |
| Generic objective statement | Targeted 3-line summary or skip it |
| No GitHub link | Always include — interviewers check it |
| GPA after 3+ years | Remove (or only include if >3.7) |
| "References available" | This is a given — waste of space |
| Listing every tool ever used | Curate to what you're confident discussing |
| Using passive voice | Start every bullet with a strong action verb |
| Too much jargon | Balance technical depth with readability |

---

## Action Verb Bank

**Built/Created:** Architected, Built, Developed, Designed, Implemented, Engineered

**Improved:** Optimized, Reduced, Accelerated, Improved, Enhanced, Boosted

**Led/Managed:** Led, Spearheaded, Directed, Coordinated, Mentored, Trained

**Analyzed:** Analyzed, Investigated, Diagnosed, Evaluated, Benchmarked, Profiled

**Deployed:** Deployed, Released, Shipped, Launched, Automated, Migrated

---

*Back to: [Career Guide](../README.md) | [Main README](../../README.md)*