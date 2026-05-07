<div align="center">

# 📄 Paper Digest: ReasoningBank

**arXiv:2504.09762 · April 2025**

</div>

---

## One-Line Summary
A persistent memory architecture for LLM agents that separates episodic (short-term) from semantic (long-term) memory, with sleep-time consolidation and bi-temporal provenance.

## Problem It Solves
Standard LLM agents have no memory beyond their context window. Each session starts from scratch. ReasoningBank gives agents a MongoDB-backed memory store that persists across sessions, consolidates experiences into durable facts, and retrieves relevant context before each reasoning step.

## Key Contributions

| Contribution | Detail |
|--------------|--------|
| **Episodic memory** | Short-term event log with TTL; recent agent interactions stored as timestamped records |
| **Semantic memory** | Long-term fact store extracted from episodic memory via nightly consolidation |
| **Sleep-time consolidation** | Offline batch process (Lambda/EventBridge) distills recent episodes into semantic facts |
| **Bi-temporal provenance** | Each memory has `valid_from`/`valid_to` so outdated beliefs can be tracked |
| **Retrieval-augmented context** | Before each agent step, relevant memories are retrieved and prepended to the prompt |

## Architecture

```
[Episodic Store] → [Consolidation Agent] → [Semantic Store]
                                                    ↓
                            [Query] → [Retriever] → [Context Prepend] → [LLM]
```

## How to Use in a Hackathon

```python
# 1. Store every agent interaction as an episode
db.episodic_memory.insert_one({
    "agent_id": "agent-001",
    "event": "analyzed quarterly report",
    "outcome": "revenue down 12%, cost of goods up 8%",
    "timestamp": datetime.now(timezone.utc),
    "expires_at": datetime.now(timezone.utc) + timedelta(days=7)
})

# 2. Nightly consolidation: extract durable facts
# (run via EventBridge Lambda trigger)

# 3. Before each reasoning step, prepend retrieved memory
context = recall_context(agent_id="agent-001", current_task=query)
```

## Benchmark Results
- Agents with ReasoningBank outperform stateless agents by +23% on multi-session task completion
- Memory retrieval adds ~180ms per step (acceptable for most applications)

## Gotchas
- TTL index must be set on `expires_at` field or episodic memory grows unbounded
- Consolidation can hallucinate facts if the base LLM is low quality; use Sonnet-class models
- Semantic memory needs conflict detection: two contradictory facts can coexist

## Relevant Hackathon Ideas
Theme 1: All 33 ideas, especially #1 ProsecuteIQ, #15 ChronicleGuard, #28 SleepTimeRL

---

*See also: [Zep temporal KG](zep.md) · [MIRIX multi-memory](mirix.md)*
