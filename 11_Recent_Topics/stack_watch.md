<div align="center">

# 🛠️ Stack Watch: 2025–2026

### Infrastructure and Tooling Movements That Matter

[![Updated](https://img.shields.io/badge/Updated-June%202026-red?style=for-the-badge)](.)

</div>

> **How to read this**: if you're about to build something that depends on a tool or API in this section, check the version and any breaking changes listed. Hackathon ideas that reference specific versions are pinned to the version current at June 2026.

---

## MongoDB Atlas (2024–2026)

| Feature | Status | Relevance |
|---------|--------|-----------|
| Vector Search GA | 🟢 Production | Core for all RAG ideas |
| Hybrid Search (`$rankFusion`) | 🟢 Production | Dense + sparse fusion in one query |
| Atlas Stream Processing | 🟢 Production | Replaces self-managed Kafka for many use cases |
| Time-Series Collections | 🟢 Production | Sensor data, AIS, market prices — no TTL hacks needed |
| ColPali-Compatible Dense Vectors | 🟢 Production | 128-dim to 4096-dim; works with nomic-embed-text, Voyage |
| Change Streams | 🟢 Production | Real-time event bus for multi-agent coordination |
| GridFS | 🟢 Production | Large artifacts (PDFs, DICOMs, video frames) |
| LangGraph MongoDB Checkpointer | 🟢 Production | Official package: `langgraph-checkpoint-mongodb` |
| Atlas Search (Lucene-based) | 🟢 Production | Full-text search, fuzzy, autocomplete |

**Key tip for hackathons**: Use an M10 cluster ($0.08/hr). Free M0 has rate limits that will kill a demo. Budget $4–8 for the cluster, it's worth it.

---

## AWS (2024–2026)

| Service | Update | Relevance |
|---------|--------|-----------|
| **Bedrock — Claude 4.x / Fable 5** | Available in us-east-1, us-west-2 | Primary reasoning model for all ideas |
| **Bedrock — Nova models** | Nov 2024 | Cost-efficient alternatives to Claude for high-volume steps |
| **Amazon Strands** | May 2025 GA | Tool-first agent framework from AWS; simpler than LangGraph for AWS-native stacks |
| **AgentCore** | 2025 | Managed agent execution environment; handles scaling and state |
| **Bedrock Knowledge Bases** | 🟢 Production | Managed RAG with MongoDB Atlas connector |
| **Bedrock Guardrails** | 🟢 Production | Content filtering, PII detection — use for healthcare/legal ideas |
| **Lambda + EventBridge** | 🟢 Production | Sleep-time consolidation, scheduled agent triggers |
| **Step Functions** | 🟢 Production | Complex multi-step workflows with retry and compensation |

**Key tip**: For hackathons, `us-east-1` has the highest Bedrock model availability. Always set fallback to `us-west-2`.

---

## LLM Model Landscape (June 2026)

> Models are listed by tier. Within each tier, use the first option unless you have a specific reason to switch.

### Frontier / Reasoning

| Model | Provider | Key Capability | Cost Tier | Best For |
|-------|---------|----------------|-----------|----------|
| **Claude Fable 5** | Anthropic | State-of-the-art on nearly all reasoning benchmarks (Jun 2026) | Very High | Novel research, hardest reasoning tasks |
| **Claude Opus 4.8** | Anthropic | #1 coding model; 88.6% SWE-bench Verified (May 2026) | High | Coding agents, complex analysis |
| **GPT-5.5** | OpenAI | Strong agentic workflows, native computer use (Apr 2026) | High | Agentic automation, computer use |
| **o3** | OpenAI | Extended chain-of-thought reasoning | High | Math, science, hard logical reasoning |
| **Gemini 3.1 Pro** | Google | Best general-purpose; 1M token context (Feb 2026) | High | Long-context analysis, multimodal |

### Production Workhorse (Best Cost/Performance)

| Model | Provider | Key Capability | Cost Tier | Best For |
|-------|---------|----------------|-----------|----------|
| **Claude Sonnet 4.6** | Anthropic | Best balance of capability and cost | Mid | Primary reasoning, most production tasks |
| **Gemini 3.5 Flash** | Google | Fast multimodal, GA production-ready (Jun 2026) | Low-Mid | RAG pipelines, long-document processing |
| **GPT-4o** | OpenAI | Vision + text, low latency | Mid | Multimodal pipelines, real-time |

### Efficient / High-Volume

| Model | Provider | Key Capability | Cost Tier | Best For |
|-------|---------|----------------|-----------|----------|
| **Claude Haiku 4.5** | Anthropic | Fast, ~90% of Sonnet capability | Low | Worker agents in swarms, classification |
| **Gemini 3.1 Flash-Lite** | Google | Lowest cost multimodal, high throughput (2026) | Very Low | High-volume structured tasks |
| **MAI-Code-1-Flash** | Microsoft | Coding-specialized, efficient (Jun 2026) | Low | Code generation, review at scale |
| **Mistral Small 3.1** | Mistral | 24B, multilingual, EU data-resident option | Low | GDPR-sensitive European apps |

### Open Weight / Self-Hosted

| Model | Provider | Key Capability | License | Best For |
|-------|---------|----------------|---------|----------|
| **DeepSeek-R1** | DeepSeek | Open-weight reasoning model | Apache 2.0 | Reasoning agents on a budget |
| **Llama 4** | Meta | Multimodal, strong general-purpose | Llama license | Self-hosted multimodal |
| **Phi-4** | Microsoft | 14B, high capability per parameter | MIT | Edge / on-device inference |
| **Gemma 3 (4B/12B/27B)** | Google | Efficient, open, well-documented | Gemma license | Resource-constrained environments |
| **Kimi K2.7 Code** | Moonshot AI | Coding specialist (Jun 2026) | Open | Code-focused self-hosted tasks |

**Model routing strategy** (cost-aware, June 2026):
1. Route simple / high-volume tasks → Haiku 4.5 or Gemini 3.1 Flash-Lite
2. Route standard reasoning → Sonnet 4.6 or Gemini 3.5 Flash
3. Route complex multi-step → Opus 4.8 or Gemini 3.1 Pro
4. Route frontier research problems → Fable 5 or o3
5. Route real-time voice / video → Gemini 3.5 Flash
6. Route sensitive data on-prem → Phi-4 or DeepSeek-R1 (self-hosted)
7. Route code generation at scale → Opus 4.8 or MAI-Code-1-Flash

---

## AI Coding Tools (2025–2026)

> AI-assisted development has become a primary workflow for AI/ML engineers in 2026. These tools are now expected skills in job descriptions.

| Tool | Key Feature | Notes |
|------|------------|-------|
| **Claude Code (Anthropic)** | Agentic CLI; reads and edits your codebase end-to-end | Runs in terminal; supports hooks, MCP servers |
| **Cursor** | AI-first IDE; Tab completion + Agent mode for multi-file edits | Most popular AI IDE as of 2026 |
| **GitHub Copilot Workspace** | PR-centric; plans changes across files from an issue | Integrated with GitHub flow |
| **Windsurf (Codeium)** | Cascade agent; autonomous multi-file refactoring | Strong on refactoring tasks |
| **Aider** | Terminal-based pair programmer; works with any LLM backend | Open source, highly configurable |
| **Devin** | Fully autonomous software engineer agent (cloud) | Best for long-horizon tasks |

**Vibe Coding** — the 2025–2026 term for using LLMs to build software primarily through natural-language prompting with minimal manual code editing. Practical for prototypes and non-critical tooling; requires careful review for production.

---

## Retrieval Stack (2024–2026)

| Tool | Version | Key Feature | Status |
|------|---------|-------------|--------|
| **Voyage AI rerank-2.5** | Aug 2025 | Instruction-following reranker; best-in-class for specialized domains | 🟢 Production |
| **Voyage multilingual-3** | 2025 | Cross-lingual embeddings; strong for multilingual apps | 🟢 Production |
| **ColPali v1.2** | 2025 | Page-level VLM embeddings; no OCR; works on scanned docs | 🟢 Production |
| **MongoDB Atlas $rankFusion** | 2024 | Hybrid dense + sparse in a single aggregation | 🟢 Production |
| **RAGAS** | 2025 | RAG evaluation framework (faithfulness, relevance, context recall) | 🟢 Production |
| **DeepEval** | 2025 | LLM output evaluation with 14+ metrics | 🟢 Production |
| **Qdrant** | 1.9+ | Alternative vector DB; strong filtering + payload indexing | 🟢 Production |
| **Weaviate** | 1.25+ | Multi-modal, GraphQL interface | 🟢 Production |
| **PGVector 0.7** | 2025 | Postgres extension; good for existing Postgres users | 🟢 Production |

---

## Agent Frameworks (2024–2026)

| Framework | Version | Key Feature | Hackathon Fit |
|-----------|---------|-------------|--------------|
| **LangGraph** | 0.2+ | Stateful agent graphs; MongoDB checkpointer; conditional edges | ⭐⭐⭐⭐⭐ |
| **AutoGen v0.4** | 0.4 | Actor-model async; typed messages; works well for swarms | ⭐⭐⭐⭐ |
| **CrewAI** | 1.x | Role-based multi-agent; easy to prototype; less flexible | ⭐⭐⭐⭐ |
| **Amazon Strands** | GA May 2025 | Tool-first AWS-native agents; simple API | ⭐⭐⭐⭐ |
| **Magentic-One** | 2024 | Orchestrator + 4 specialist agents; browser + code | ⭐⭐⭐ |
| **Pydantic AI** | 2025 | Type-safe agent framework; excellent for structured outputs | ⭐⭐⭐⭐ |
| **browser-use** | 2025 | LLM-controlled browser automation; key for web-scraping agents | ⭐⭐⭐⭐⭐ |
| **LiveKit Agents** | 2025 | Real-time voice + video agent SDK | ⭐⭐⭐⭐ |

**Recommendation**: Use **LangGraph + MongoDB checkpointer** as the base for stateful agent ideas. Use **AutoGen v0.4** or **CrewAI** for multi-agent swarms. Use **LangGraph + Voyage AI** for retrieval-augmented agents.

---

## LLMOps Tooling (2024–2026)

| Tool | Use Case | Free Tier |
|------|---------|-----------|
| **LangSmith** | Trace LLM calls; debug agent loops; prompt versioning | Yes (limited) |
| **Arize Phoenix** | Open-source LLM observability; embedding drift detection | Yes (self-host) |
| **Helicone** | Proxy-based logging; latency tracking; cost per request | Yes (1M calls/mo) |
| **Guardrails AI** | Schema validation, content filtering, hallucination detection | Yes |
| **Instructor** | Structured output reliability (Pydantic models from LLM) | Yes (open source) |
| **Outlines** | Grammar-constrained LLM generation | Yes (open source) |
| **RAGAS** | RAG pipeline evaluation | Yes (open source) |
| **DeepEval** | LLM output evaluation suite | Yes (limited) |

---

## Fine-Tuning Tooling (2024–2026)

| Tool | Key Feature | When to Use |
|------|------------|------------|
| **Unsloth** | 2× faster LoRA fine-tuning; 80% less VRAM | Any LoRA fine-tuning job |
| **Axolotl** | YAML-configured fine-tuning; supports all major formats | When you need reproducible fine-tuning configs |
| **GRPO** | Group Relative Policy Optimization; DeepSeek-R1's training approach | When you want reasoning-style fine-tuning |
| **DPO** | Direct Preference Optimization; simpler than RLHF | When you have preference data |
| **QLoRA** | 4-bit quantized LoRA; fine-tune 70B on a single A100 | Large model fine-tuning with limited GPU |
| **DoRA** | Weight-Decomposed LoRA; better than LoRA for many tasks | Drop-in LoRA replacement |

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← Trends](2025_2026_trends.md) | [🏠 README](../README.md) | [Paper Digests →](paper_digests/) |
