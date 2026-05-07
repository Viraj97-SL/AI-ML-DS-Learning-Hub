<div align="center">

# 📄 Paper Digest: SagaLLM — Distributed Transactions for LLM Agents

**arXiv:2312.05382 · December 2023 (updated 2024)**

</div>

---

## One-Line Summary
Applies the classic database Saga pattern to LLM agent workflows — when a multi-step agent action fails partway through, compensating transactions automatically undo the completed steps.

## Problem It Solves
Multi-agent pipelines are brittle: if step 4 of a 6-step workflow fails, steps 1–3 have already modified state. Without compensation, you're left in an inconsistent partial state. SagaLLM adds a compensating action to each step; on failure, the system runs compensations in reverse order to restore consistency.

## Key Contributions

| Contribution | Detail |
|--------------|--------|
| **Saga choreography** | Each agent step registers a compensation action before executing |
| **Forward recovery** | If a step fails, retry with alternative strategy before triggering rollback |
| **Backward recovery** | Execute compensations in reverse order to undo completed steps |
| **LLM-generated compensations** | For novel workflows, Claude can generate the compensation action |
| **MongoDB audit log** | Full saga transaction log with step status and compensation results |

## The Saga Pattern

```
Forward:  T1 → T2 → T3 → [FAIL at T4]
                              ↓
Backward: C3 → C2 → C1  (compensating transactions)
```

Where each `Cx` undoes the effect of `Tx`.

## MongoDB Schema

```python
db.saga_transactions.insert_one({
    "saga_id": "saga_001",
    "steps": [
        {
            "step": 1, "name": "reserve_berth",
            "action": {"type": "mongodb_update", ...},
            "compensation": {"type": "mongodb_update", "reverse": True},
            "status": "completed"
        },
        {
            "step": 2, "name": "charge_fee",
            "action": {...},
            "compensation": {"type": "refund", ...},
            "status": "failed"  # Triggers backward recovery
        }
    ],
    "overall_status": "rolling_back"
})
```

## How to Use in a Hackathon

```python
class SagaOrchestrator:
    def __init__(self):
        self.completed_steps = []
    
    def execute_step(self, step_fn, compensation_fn, *args):
        result = step_fn(*args)
        self.completed_steps.append(compensation_fn)
        return result
    
    def rollback(self):
        for compensation in reversed(self.completed_steps):
            try:
                compensation()
            except Exception as e:
                print(f"Compensation failed: {e}")  # Log but continue
        self.completed_steps.clear()
```

## Typical Hackathon Use Cases
- **Portfall** (Blueprint 03): vessel reroute + insurer deal — if insurer backs out, undo vessel's course change
- **CityReview** (#74): zoning proposal passes then gets legal veto — undo all stakeholder commitments
- **MarsControl** (#75): science experiment started then power fails — compensate by saving partial data

## Gotchas
- Compensations must be idempotent (safe to run twice)
- Some actions are not reversible (sending an email, issuing a public statement) — these need semantic compensations ("send a correction email")
- Don't use for operations that must be atomic — use 2PC instead

## Relevant Hackathon Ideas
Theme 1: #7 PortfolioSaga, #20 ContractAgent · Theme 2: #44 TradeSettlement, #74 CityReview

---

*See also: [ReasoningBank](reasoningbank.md) · [Zep temporal KG](zep.md)*
