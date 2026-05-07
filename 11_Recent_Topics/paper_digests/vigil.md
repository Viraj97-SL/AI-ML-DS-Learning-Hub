<div align="center">

# 📄 Paper Digest: VIGIL — Reflective Supervisor for Multi-Agent Systems

**2025 · Adversarial oversight for autonomous agent swarms**

</div>

---

## One-Line Summary
A supervisor agent that continuously monitors a multi-agent system for goal drift, unsafe actions, and coordination failures — intervening before harm rather than auditing after the fact.

## Problem It Solves
Multi-agent systems can drift from their intended goal when individual agents optimize locally. VIGIL adds a persistent meta-agent layer that observes all inter-agent communication, detects anomalies (goal misalignment, escalating resource use, unexpected tool calls), and either corrects behavior directly or escalates to a human operator.

## Key Contributions

| Contribution | Detail |
|--------------|--------|
| **Goal consistency monitor** | Embeds each agent's stated goal and checks semantic drift from the original task |
| **Action risk scorer** | Classifies each tool call by risk level: read-only, reversible-write, irreversible |
| **Communication graph analysis** | Detects unusual message patterns (agent isolation, circular dependency, broadcast storms) |
| **Reflective intervention** | VIGIL can inject corrective messages into agent conversations |
| **Escalation protocol** | When VIGIL confidence < threshold, pauses the swarm and alerts a human |

## The Monitoring Loop

```
[All A2A Messages] → [VIGIL Observer] → [Anomaly Detector]
                                                ↓
                              [Low risk] → [Log + continue]
                              [Med risk] → [Inject correction]
                              [High risk] → [Pause swarm + alert human]
```

## MongoDB Integration

```python
# VIGIL logs every agent action with risk score
db.vigil_audit.insert_one({
    "agent_id": "agent-007",
    "action": "delete_database_collection",
    "risk_level": "HIGH",
    "vigil_decision": "blocked",
    "reason": "Irreversible action not in original task scope",
    "timestamp": datetime.now(timezone.utc)
})

# Goal drift detection
def check_goal_drift(agent_id: str, current_output: str, original_goal: str) -> float:
    """Returns cosine distance between current output and original goal embeddings."""
    # High distance = goal drift
    return cosine_distance(embed(current_output), embed(original_goal))
```

## How to Use in a Hackathon

Add VIGIL as a passive observer to any multi-agent demo:
1. Log all A2A messages to a `vigil_feed` collection
2. Run a simple goal-drift check before each agent step
3. Demo the intervention: show VIGIL catching a rogue agent in the demo

This dramatically increases judge confidence in safety properties of your system.

## Relevant Hackathon Ideas
Theme 2: All multi-agent ideas — essential for #46 TumorBoardSim (medical safety), #72 ICUTriage, #75 MarsControl

---

*See also: [Magentic-One](magentic_one.md) · [A2A Protocol](a2a_protocol.md)*
