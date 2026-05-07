<div align="center">

# 🔥 Recent Topics: AI/ML/DS Trends 2025–2026

### What's New Since 2024 — The Bleeding Edge Every Practitioner Needs

[![Updated](https://img.shields.io/badge/Updated-May%202026-red?style=for-the-badge)](.)
[![Papers](https://img.shields.io/badge/Paper%20Digests-12-purple?style=for-the-badge)](paper_digests/)
[![Hot%20Topics](https://img.shields.io/badge/Hot%20Topics-20%2B-orange?style=for-the-badge)](2025_2026_trends.md)

</div>

---

## What This Section Is (and Isn't)

**This is**: the field's changelog. It tracks what shifted, what shipped, and what became foundational since mid-2024.

**This is not**: tutorials. You won't find "Introduction to RAG" here. For that, go to the main tracks:
- Full RAG guide → [03_AI_Engineer/intermediate/](../03_AI_Engineer/intermediate/)
- LLM fine-tuning → [02_ML_Engineer/advanced/](../02_ML_Engineer/advanced/)
- Agent frameworks → [03_AI_Engineer/advanced/](../03_AI_Engineer/advanced/)

> **Rule of thumb**: if it was in a course syllabus before 2024, it belongs in the main tracks. If it first appeared on arXiv or in a product changelog in 2024–2026, it belongs here.

---

## Sections

| Section | What It Contains |
|---------|----------------|
| [📈 2025–2026 Trend Tracker](2025_2026_trends.md) | 20+ hot topics table with maturity ratings, hackathon-readiness, and cross-links |
| [🛠️ Stack Watch](stack_watch.md) | Infrastructure movements: MongoDB, AWS Bedrock, model releases, frameworks |
| [📄 Paper Digests](paper_digests/) | Deep dives on 12 anchor papers — each with a "Hackathon Angle" section |

---

## The 12 Anchor Papers

These are the papers powering the `10_Hackathons/` ideas. Each has a full digest:

| Paper | TL;DR | Hackathon Theme |
|-------|-------|-----------------|
| [ReasoningBank](paper_digests/reasoningbank.md) | Distills successful reasoning patterns into a reusable memory bank | Theme 1 |
| [MIRIX](paper_digests/mirix.md) | Multi-type memory architecture (episodic/semantic/procedural/working/resource) | Theme 1 |
| [Zep Temporal KG](paper_digests/zep.md) | Bi-temporal knowledge graph for agent memory with `valid_from`/`valid_to` | Theme 1 |
| [SagaLLM](paper_digests/sagallm.md) | Transactional agents with compensating actions for long-running workflows | Theme 1 |
| [VIGIL](paper_digests/vigil.md) | Reflective sibling supervisor that catches agent self-errors | Theme 1 |
| [Magentic-One](paper_digests/magentic_one.md) | Generalist multi-agent system: orchestrator + 4 specialist agents | Theme 2 |
| [A2A Protocol](paper_digests/a2a_protocol.md) | Google's open standard for agent-to-agent communication | Theme 2 |
| [MCP](paper_digests/mcp.md) | Anthropic's Model Context Protocol — de-facto tool exposure standard | Theme 2 |
| [ColPali](paper_digests/colpali.md) | VLM-based document retrieval using page-level embeddings; no OCR needed | Theme 3 |
| [Search-R1](paper_digests/search_r1.md) | RL-trained retriever that learns when and how to search | Theme 3 |
| [GraphRAG](paper_digests/graphrag.md) | Knowledge graph communities + LLM summarization for global reasoning | Theme 3 |
| [HippoRAG 2](paper_digests/hipporag.md) | Hippocampus-inspired Personalized PageRank for multi-hop retrieval | Theme 3 |

---

## How to Use in Hackathons

1. **Pick a theme** in `10_Hackathons/` that fits your domain interest
2. **Read the paper digest** for the anchor paper of your chosen idea (30 minutes)
3. **Check the "Hackathon Angle"** section in the digest — it tells you exactly which part of the paper to implement
4. **Read the Stack Watch** to confirm the libraries and APIs are still current
5. **Build** — the ideas files give you the MongoDB schema and demo script

> The digests are designed to be read in 30 minutes, not 3 hours. If a paper has 40 pages, the digest gives you the 4 pages that matter for a hackathon.

---

## Update Cadence

This section is updated **monthly**. When a major model, framework, or paper lands:
1. A row is added to `2025_2026_trends.md`
2. A new digest is added to `paper_digests/` if it's a hackathon-relevant paper
3. `stack_watch.md` is updated with version changes

To contribute an update, see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← 10_Hackathons](../10_Hackathons/README.md) | [🏠 README](../README.md) | [Trend Tracker →](2025_2026_trends.md) |
