<div align="center">

# 📄 Paper Digest: Zep — Temporal Knowledge Graph for Agents

**arXiv:2501.13956 · January 2025**

</div>

---

## One-Line Summary
A bi-temporal knowledge graph framework for LLM agents that separates *when a fact was valid* (valid time) from *when the system learned it* (transaction time), enabling precise point-in-time memory queries.

## Problem It Solves
Agent memory systems either keep all facts forever (stale beliefs accumulate) or discard old facts (lose historical context). Zep solves this with bi-temporal indexing: you can ask "what did the agent know on March 1st about the state of the project on February 15th?" — two independent time axes.

## Key Contributions

| Contribution | Detail |
|--------------|--------|
| **Bi-temporal schema** | `valid_from`/`valid_to` (fact validity) + `inserted_at`/`superseded_at` (system knowledge) |
| **Temporal graph queries** | Point-in-time consistent graph traversal |
| **Entity resolution** | Automatically merges duplicate entity mentions across documents |
| **Contradiction detection** | Flags when a new fact contradicts a currently-valid fact |
| **GraphQL-style query** | Simple API: `get_facts_at_time(entity, time)` |

## The Bi-Temporal Model

```
        Valid Time →
  T  ┌─────────────────────────────────
  r  │  "AMOC risk = 0.6"  (2025-01-01 to 2025-06-30)
  a  │  "AMOC risk = 0.7"  (2025-06-30 to present)
  n  │
  s  │  "AMOC risk = 0.6"  ← what we knew until paper revision
  a  │  "AMOC risk = 0.68" ← what we know now (revised upward)
  c  │
  t  └─────────────────────────────────
```

## MongoDB Implementation

```python
# Store a new fact with bi-temporal validity
db.facts.insert_one({
    "entity": "AMOC",
    "predicate": "collapse_risk",
    "value": 0.68,
    "valid_time": {
        "valid_from": datetime(2026, 1, 1),
        "valid_to": None  # Currently valid
    },
    "transaction_time": {
        "inserted_at": datetime.now(timezone.utc),
        "superseded_at": None
    }
})

# Query: what was the AMOC risk on June 15, 2025?
query_date = datetime(2025, 6, 15)
db.facts.find_one({
    "entity": "AMOC",
    "predicate": "collapse_risk",
    "valid_time.valid_from": {"$lte": query_date},
    "$or": [
        {"valid_time.valid_to": None},
        {"valid_time.valid_to": {"$gt": query_date}}
    ]
})
```

## How to Use in a Hackathon
- **ChronoLaw** (Blueprint 06): law versioning with bi-temporal validity
- **Tipping Oracle** (Blueprint 04): climate risk scores that update as new evidence arrives
- Any Theme 1 idea that needs "what did the agent know at time T" semantics

## Gotchas
- Double-entry bookkeeping: every update requires closing the old record and opening a new one
- Query complexity increases; index both `valid_from` and `transaction_time.inserted_at`
- Transaction time is append-only — never update it

## Relevant Hackathon Ideas
Theme 1: #15 ChronicleGuard, #19 IncidentMemory · Theme 3: #78 PrecedentBrain, #82 ComplianceCompass

---

*See also: [ReasoningBank](reasoningbank.md) · [SagaLLM](sagallm.md)*
