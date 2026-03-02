# AI Engineer Learning Track

> **"Build intelligent products that millions of people use every day."**

AI Engineers build AI-powered applications and systems using Large Language Models (LLMs) and other foundation models. This is the fastest-growing engineering specialty in 2024-2025, and the barrier to entry is lower than you might think.

---

## Track Overview

```
PHASE 1: LLM Foundations (1-2 months)
    Python → LLM APIs → Prompt Engineering → RAG Basics
         ↓
PHASE 2: Core AI Engineering (2-3 months)
    RAG Systems → LangChain → Agents → Vector Databases
         ↓
PHASE 3: Advanced AI Systems (3-5 months)
    Fine-tuning → Multi-agent → Evals → Production AI
         ↓
PHASE 4: Expert (Ongoing)
    Alignment → Research → Open Source Contributions
```

---

## Prerequisites

- [ ] Python proficiency (functions, classes, async/await basics)
- [ ] Basic understanding of REST APIs
- [ ] Git basics (clone, commit, push)
- [ ] API key from OpenAI or Anthropic (free tier is fine)

> **Good news:** This is the most accessible of the three tracks for Python developers. You can build real AI apps within your first week!

---

## What Makes AI Engineering Different

| Traditional Software | AI Engineering |
|---------------------|---------------|
| Deterministic output | Probabilistic / non-deterministic output |
| Unit testable | Requires custom evaluation frameworks |
| Bug = code error | Bug = prompt issue, model hallucination, context problem |
| Deploy once, stable | Models drift, need continuous evaluation |
| Clear correctness | "Good enough" is often the standard |

---

## Skills You Will Build

| Category | Skills |
|----------|--------|
| LLM APIs | OpenAI, Anthropic Claude, Google Gemini, Mistral, open-source |
| Frameworks | LangChain, LlamaIndex, AutoGen, CrewAI |
| Vector Databases | Chroma, Pinecone, Weaviate, Qdrant, pgvector |
| Embeddings | OpenAI Embeddings, SentenceTransformers |
| Prompt Engineering | Zero-shot, few-shot, CoT, ReAct, meta-prompting |
| RAG | Naive RAG → Advanced RAG → GraphRAG |
| Agents | Tool use, function calling, agent frameworks |
| Fine-tuning | LoRA, QLoRA, DPO, RLHF basics |
| Evaluation | RAGAS, DeepEval, LangSmith, custom evals |
| Deployment | FastAPI, Modal, Hugging Face Spaces, Streamlit |
| Observability | LangSmith, Arize Phoenix, Helicone |

---

## Beginner Phase — LLM Foundations

**Goal:** Call LLM APIs confidently, understand prompt engineering, and build your first AI app.

**Duration:** 1-2 months (8-12 hrs/week)

| Week | Topic | Resource | Project |
|------|-------|----------|---------|
| 1 | LLM fundamentals & APIs (OpenAI, Claude, Gemini) | [LLM APIs Guide](beginner/01_llm_apis.ipynb) | Call 3 different LLM APIs |
| 2-3 | Prompt Engineering fundamentals | [Prompt Eng Guide](beginner/02_prompt_engineering.ipynb) | Improve a bad prompt |
| 4 | Chatbot with memory | [Chatbot Guide](beginner/03_chatbot_with_memory.ipynb) | Multi-turn chatbot |
| 5-6 | Basic RAG pipeline | [RAG Basics](beginner/04_basic_rag.ipynb) | Q&A over your own docs |
| 7-8 | Deploy AI App (Streamlit + cloud) | [Deployment Guide](beginner/05_deploy_ai_app.ipynb) | Streamlit app + deploy |

**[→ Start Beginner Phase](beginner/)**

---

## Intermediate Phase — Core AI Engineering

**Goal:** Build production-quality RAG systems, understand agents, and deploy AI features.

**Duration:** 2-3 months (10-15 hrs/week)

| Week | Topic | Resource | Project |
|------|-------|----------|---------|
| 1-2 | LangChain deep dive | [LangChain Guide](intermediate/02_langchain_deep_dive.ipynb) | Build a chain |
| 3-4 | Advanced RAG techniques | [Advanced RAG](intermediate/01_advanced_rag.ipynb) | Hybrid search + reranking |
| 5-6 | Vector databases | [Vector DB Guide](intermediate/03_vector_databases.ipynb) | Build semantic search |
| 7-8 | AI Agents | [Agents Guide](intermediate/04_ai_agents.ipynb) | Build tool-using agent |
| 9-10 | Function calling & tool use | [Tool Use Guide](intermediate/05_function_calling.ipynb) | Agent with 5+ tools |
| 11-12 | LlamaIndex | [LlamaIndex Guide](intermediate/05_llamaindex.ipynb) | Complex document QA |
| 13-14 | AI Evaluation | [Eval Guide](intermediate/06_ai_evaluation.ipynb) | Build eval suite |
| 15-16 | Production AI Systems | [Production Guide](intermediate/07_production_ai.ipynb) | Prod-ready AI service |

**[→ Start Intermediate Phase](intermediate/)**

---

## Advanced Phase — Expert AI Systems

**Goal:** Fine-tune LLMs, build multi-agent systems, and design production-grade AI architectures.

**Duration:** 3-5 months (12-15 hrs/week)

| Week | Topic | Resource | Project |
|------|-------|----------|---------|
| 1-3 | LLM Fine-tuning (LoRA/QLoRA + Alignment) | [Fine-tuning Guide](advanced/01_llm_finetuning_alignment.ipynb) | Fine-tune Llama 3 |
| 4-6 | Multi-agent systems | [Multi-agent Guide](advanced/02_multi_agent_systems.ipynb) | Research + report agent |
| 7-9 | Advanced Retrieval techniques | [Advanced Retrieval](advanced/04_advanced_retrieval.ipynb) | Multi-strategy retrieval |
| 10-12 | AI Safety & Guardrails | [Safety Guide](advanced/03_safety_guardrails.ipynb) | Build guardrails system |
| 13-15 | GraphRAG & Knowledge Graphs | [GraphRAG Guide](advanced/05_graphrag.ipynb) | Knowledge graph RAG |
| 16-18 | Custom LLM Evaluation | [Advanced Evals](advanced/06_advanced_evals.ipynb) | Custom eval framework |
| 19-21 | AI Product Case Studies | [Case Studies](advanced/07_case_studies.md) | Analyze + present |

**[→ Start Advanced Phase](advanced/)**

---

## Projects

### Beginner Projects
- [ ] AI Chatbot with memory (OpenAI / Claude API)
- [ ] Document Q&A System (basic RAG)
- [ ] AI Writing Assistant (prompt engineering showcase)
- [ ] YouTube Transcript Summarizer

### Intermediate Projects
- [ ] Personal Knowledge Base with RAG (your notes → searchable AI)
- [ ] AI Research Assistant Agent (searches web + summarizes)
- [ ] Multi-model AI API Gateway (route to different models)
- [ ] Code Review Bot (GitHub Actions + LLM)

### Advanced Projects
- [ ] Fine-tuned LLM for a specific domain (legal, medical, etc.)
- [ ] Multi-agent research system (AutoGen / CrewAI)
- [ ] AI-powered SaaS product (idea → full product)
- [ ] Custom Evaluation Framework for LLM outputs

**[→ See All Projects](projects/)**

---

## Key Concepts You Must Understand

### The AI Engineering Stack

```
┌──────────────────────────────────────────────────┐
│              AI Application Layer                 │
│  (Streamlit / FastAPI / Next.js / Slack bot)     │
├──────────────────────────────────────────────────┤
│              Orchestration Layer                  │
│  (LangChain / LlamaIndex / AutoGen / CrewAI)     │
├──────────────────────────────────────────────────┤
│              Memory & Storage Layer               │
│  (Vector DB / SQL / Redis / Knowledge Graph)     │
├──────────────────────────────────────────────────┤
│                 LLM Layer                         │
│  (OpenAI / Claude / Gemini / Llama / Mistral)   │
├──────────────────────────────────────────────────┤
│             Evaluation & Monitoring               │
│  (RAGAS / LangSmith / Arize / DeepEval)         │
└──────────────────────────────────────────────────┘
```

### RAG Architecture
```
User Query
    ↓
Query Embedding
    ↓
Vector Similarity Search (Vector DB)
    ↓
Retrieved Chunks (Top-K)
    ↓
[Query + Context] → LLM
    ↓
Answer
```

### Agent Loop (ReAct Pattern)
```
Thought → Action → Observation → Thought → ... → Final Answer
```

---

## Skills Checklist

### Beginner Level
- [ ] Can call OpenAI and Anthropic APIs
- [ ] Understand tokens, context windows, temperature
- [ ] Can write effective prompts (zero-shot, few-shot, system prompts)
- [ ] Built a basic chatbot with conversation history
- [ ] Can build a basic RAG pipeline
- [ ] Deployed one AI app (Streamlit, Hugging Face Spaces, etc.)

### Intermediate Level
- [ ] Built a production-quality RAG system with hybrid search
- [ ] Built an AI agent that uses tools (web search, calculator, code execution)
- [ ] Understand chunking strategies for RAG
- [ ] Can evaluate RAG systems (faithfulness, relevance, completeness)
- [ ] Know when to use LangChain vs LlamaIndex vs raw API calls
- [ ] Can handle streaming responses in a web app

### Advanced Level
- [ ] Fine-tuned an open-source LLM with LoRA
- [ ] Built and orchestrated a multi-agent system
- [ ] Designed and ran a comprehensive LLM evaluation suite
- [ ] Understand alignment basics (RLHF, DPO, Constitutional AI)
- [ ] Can architect a production AI system with monitoring + guardrails
- [ ] Contributed to an AI open-source project

---

## Recommended Resources

### Courses (Mostly Free)
| Course | Provider | Duration |
|--------|----------|----------|
| Neural Networks: Zero to Hero | Andrej Karpathy (YouTube) | 20 hrs |
| Prompt Engineering for Developers | DeepLearning.AI | 1-2 hrs |
| LangChain for LLM Apps | DeepLearning.AI | 1-2 hrs |
| Building RAG Agents with LLMs | NVIDIA DLI | 8 hrs |
| Generative AI for Beginners | Microsoft (GitHub) | 18 lessons |
| LangChain Academy | LangChain | Self-paced |
| Hugging Face NLP Course | Hugging Face | Self-paced |

### Essential Reading (Free)
| Resource | Link |
|----------|------|
| Prompt Engineering Guide | [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) |
| OpenAI Cookbook | [openai/openai-cookbook](https://github.com/openai/openai-cookbook) |
| LangChain Docs | [python.langchain.com](https://python.langchain.com) |
| RAGAS Paper | [arXiv: 2309.15217](https://arxiv.org/abs/2309.15217) |
| Attention Is All You Need | [arXiv: 1706.03762](https://arxiv.org/abs/1706.03762) |

### Key Papers Every AI Engineer Should Read
1. "Attention Is All You Need" (Transformer architecture)
2. "Language Models are Few-Shot Learners" (GPT-3)
3. "RLHF: Learning to summarize from human feedback"
4. "Constitutional AI: Harmlessness from AI Feedback"
5. "Retrieval-Augmented Generation for Knowledge-Intensive NLP"
6. "LoRA: Low-Rank Adaptation of Large Language Models"

---

*Back to: [Main README](../README.md) | [Role Comparison](../00_Overview/role_comparison.md)*
