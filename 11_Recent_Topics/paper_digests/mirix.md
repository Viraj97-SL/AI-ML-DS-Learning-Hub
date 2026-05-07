<div align="center">

# 📄 Paper Digest: MIRIX — Multi-Instance Reflective Intelligence eXchange

**2025 · Memory architecture for persistent multi-agent systems**

</div>

---

## One-Line Summary
A six-tier memory architecture for LLM agents that separates core identity, procedural knowledge, semantic facts, episodic events, active working memory, and resource cache — each with different retention and retrieval policies.

## Problem It Solves
Most agent memory systems are monolithic (everything in one vector store or context). MIRIX recognizes that different types of memory have radically different characteristics: your agent's identity and core values change rarely; procedural skills change slowly; semantic world knowledge changes moderately; episodic events change constantly. One storage mechanism is wrong for all of these.

## The Six Memory Tiers

| Tier | Content | Retention | Update Frequency |
|------|---------|-----------|-----------------|
| **Core** | Agent identity, values, role definition | Permanent | Rare |
| **Procedural** | Skills, workflows, tool-use patterns | Long-term | Occasional |
| **Semantic** | World knowledge, domain facts | Medium-term | Regular |
| **Episodic** | Event log, interaction history | Short-term + TTL | Constant |
| **Working** | Current context, in-progress reasoning | Session-only | Per-step |
| **Resource Cache** | Retrieved documents, tool outputs | Transient | On-demand |

## MongoDB Mapping

```python
# Each tier maps to a MongoDB collection with different TTL and indexing
TIER_COLLECTIONS = {
    "core": db["agent_core"],              # No TTL, rarely updated
    "procedural": db["agent_procedures"],  # TTL: 90 days, vector + text index
    "semantic": db["agent_semantic"],      # TTL: 30 days, vector index
    "episodic": db["agent_episodic"],      # TTL: 7 days, time-series
    "working": db["agent_working"],        # TTL: 24 hours, in-memory preferred
    "resource_cache": db["agent_cache"]    # TTL: 1 hour, vector index
}
```

## How to Use in a Hackathon

Start with just 3 tiers for a 24h hackathon:
1. **Semantic**: long-term facts about the domain
2. **Episodic**: recent interactions with TTL
3. **Working**: current context for the active task

Add Core if your agent has a persistent persona (customer service bot, legal advisor).

## Key Insight
The reflective step: after each interaction, the agent asks itself: "Should this experience update my procedural knowledge? Did I learn a new semantic fact? Should my core values change?" This controlled self-modification prevents memory drift.

## Relevant Hackathon Ideas
Theme 1: All prolonged coordination ideas — especially #15 ChronicleGuard, #22 PatientPath, #31 TeacherMind

---

*See also: [ReasoningBank](reasoningbank.md) · [Zep temporal KG](zep.md)*
