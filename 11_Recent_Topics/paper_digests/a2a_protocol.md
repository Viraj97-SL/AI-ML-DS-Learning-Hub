<div align="center">

# 📄 Paper Digest: A2A Protocol — Agent-to-Agent Communication

**Google 2025 · Open standard for inter-agent communication**

</div>

---

## One-Line Summary
An open protocol for agents to discover each other's capabilities, negotiate task assignments, and exchange results — analogous to HTTP for agents.

## Problem It Solves
Multi-agent systems today are proprietary silos: a LangGraph agent cannot easily talk to an AutoGen agent, which cannot talk to a CrewAI agent. A2A provides a common language for agent capability advertisement and task delegation.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **AgentCard** | JSON document describing an agent's capabilities, input/output types, resource limits |
| **Task** | Unit of work with a defined type, payload, and lifecycle (pending → active → completed/failed) |
| **Artifact** | Output produced by an agent (text, file, structured data) |
| **Message** | Synchronous or async communication between agents |
| **Agent Registry** | Directory of available agents with their AgentCards |

## AgentCard Schema (simplified)

```json
{
  "agent_id": "summarizer-001",
  "name": "Document Summarizer",
  "version": "1.0",
  "capabilities": ["summarize", "extract_entities", "translate"],
  "input_types": ["text/plain", "application/pdf"],
  "output_types": ["text/markdown", "application/json"],
  "resource_limits": {
    "max_payload_mb": 10,
    "max_tokens": 8192,
    "rate_limit_per_minute": 60
  },
  "endpoint": "https://agent.example.com/a2a",
  "authentication": {"type": "bearer"}
}
```

## MongoDB as the A2A Message Bus

```python
# Agent registry
db.agent_registry.create_index([("capabilities", 1)])
db.agent_registry.create_index([("agent_id", 1)], unique=True)

# Task queue — Change Streams for real-time delivery
db.task_queue.watch([
    {"$match": {
        "operationType": "insert",
        "fullDocument.to_agent": "summarizer-001"
    }}
])
```

## vs MCP

| Aspect | A2A | MCP |
|--------|-----|-----|
| **Who talks** | Agent ↔ Agent | Agent → Tool |
| **Direction** | Bidirectional delegation | Unidirectional invocation |
| **Discovery** | AgentCard registry | Tool manifest |
| **State** | Stateful task lifecycle | Stateless tool calls |
| **Best for** | Multi-agent workflows | Extending agent with tools |

They are complementary: an agent uses MCP to access tools, and A2A to delegate tasks to other agents.

## How to Use in a Hackathon

1. Register all agents at startup (capability cards → MongoDB)
2. Use MongoDB Change Streams as the message bus
3. Add `to_agent` field to every task message so agents can filter their queue
4. Demo: show two independent agents coordinating without knowing each other's internals

## Relevant Hackathon Ideas
Theme 2: All 34 ideas — A2A is the foundational protocol for all multi-agent collaboration

---

*See also: [MCP](mcp.md) · [Magentic-One](magentic_one.md) · [VIGIL](vigil.md)*
