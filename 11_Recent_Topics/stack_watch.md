<div align="center">

# 🛠️ Stack Watch: 2025–2026

### Infrastructure and Tooling Movements That Matter

[![Updated](https://img.shields.io/badge/Updated-May%202026-red?style=for-the-badge)](.)

</div>

> **How to read this**: if you're about to build something that depends on a tool or API in this section, check the version and any breaking changes listed. Hackathon ideas that reference specific versions are pinned to the version current at May 2026.

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
| **Bedrock — Claude 4.x / 3.5 Sonnet** | Available in us-east-1, us-west-2 | Primary reasoning model for all ideas |
| **Bedrock — Nova models** | Nov 2024 | Cost-efficient alternatives to Claude for high-volume steps |
| **Amazon Strands** | May 2025 GA | Tool-first agent framework from AWS; simpler than LangGraph for AWS-native stacks |
| **AgentCore** | 2025 | Managed agent execution environment; handles scaling and state |
| **Bedrock Knowledge Bases** | 🟢 Production | Managed RAG with MongoDB Atlas connector |
| **Bedrock Guardrails** | 🟢 Production | Content filtering, PII detection — use for healthcare/legal ideas |
| **Lambda + EventBridge** | 🟢 Production | Sleep-time consolidation, scheduled agent triggers |
| **Step Functions** | 🟢 Production | Complex multi-step workflows with retry and compensation |

**Key tip**: For hackathons, `us-east-1` has the highest Bedrock model availability. Always set fallback to `us-west-2`.

---

## LLM Model Landscape (May 2026)

| Model | Provider | Key Capability | Cost Tier | Hackathon Use |
|-------|---------|----------------|-----------|---------------|
| **Claude Sonnet 4.6** | Anthropic | Best coding, instruction-following | Mid | Primary reasoning |
| **Claude Opus 4.7** | Anthropic | Complex multi-step reasoning | High | Deep analysis agents |
| **Claude Haiku 4.5** | Anthropic | Fast, cheap | Low | Worker agents in swarms |
| **GPT-4o** | OpenAI | Vision + text, fast | Mid | Multimodal pipelines |
| **o3 / o1** | OpenAI | Reasoning / extended thinking | High | Reasoning-heavy agents |
| **DeepSeek-R1** | DeepSeek | Open-weight reasoning model; Apache 2.0 | Free (self-host) | Reasoning agents on a budget |
| **Gemini 2.0 Flash** | Google | Multimodal, fast, long context (1M tokens) | Low-Mid | Long-document processing |
| **Gemini 2.0 Pro** | Google | Complex reasoning, multimodal | High | Complex analysis |
| **Phi-4** | Microsoft | 14B, high capability per parameter | Free (self-host) | Edge / on-device inference |
| **Gemma 2 (9B/27B)** | Google | Open-weight, efficient | Free (self-host) | Low-resource language SLMs |
| **Llama 3.3 (70B)** | Meta | Open-weight, strong general purpose | Free (self-host) | Self-hosted everything |
| **Mistral Small 3.1** | Mistral | 24B, multilingual, efficient | Low | European GDPR-sensitive apps |

**Model routing strategy** (cost-aware):
1. Route simple tasks → Haiku 4.5 or Gemma 2
2. Route complex reasoning → Sonnet 4.6
3. Route multi-step analysis → Opus 4.7 or o3
4. Route real-time voice → Gemini 2.0 Flash
5. Route sensitive data on-prem → Phi-4 or DeepSeek-R1 (self-hosted)

---

## Retrieval Stack (2024–2026)

| Tool | Version | Key Feature | Status |
|------|---------|-------------|--------|
| **Voyage AI rerank-2.5** | Aug 2025 | Instruction-following reranker; best-in-class for specialized domains | 🟢 Production |
| **Voyage multilingual-3** | 2025 | Cross-lingual embeddings; strong for #51 RefugeeVoice, #90 EdgarLinguist | 🟢 Production |
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
| **LangGraph** | 0.2 | Stateful agent graphs; MongoDB checkpointer; conditional edges | ⭐⭐⭐⭐⭐ |
| **AutoGen v0.4** | 0.4 | Actor-model async; typed messages; works well for swarms | ⭐⭐⭐⭐ |
| **CrewAI** | 1.x | Role-based multi-agent; easy to prototype; less flexible | ⭐⭐⭐⭐ |
| **Amazon Strands** | GA May 2025 | Tool-first AWS-native agents; simple API | ⭐⭐⭐⭐ |
| **Magentic-One** | 2024 | Orchestrator + 4 specialist agents; browser + code | ⭐⭐⭐ |
| **Pydantic AI** | 2025 | Type-safe agent framework; excellent for structured outputs | ⭐⭐⭐⭐ |
| **browser-use** | 2025 | LLM-controlled browser automation; key for web-scraping agents | ⭐⭐⭐⭐⭐ |
| **LiveKit Agents** | 2025 | Real-time voice + video agent SDK | ⭐⭐⭐⭐ |

**Recommendation**: Use **LangGraph + MongoDB checkpointer** as the base for Theme 1 ideas. Use **AutoGen v0.4** or **CrewAI** for Theme 2 swarms. Use **LangGraph + Voyage AI** for Theme 3 retrieval ideas.

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
