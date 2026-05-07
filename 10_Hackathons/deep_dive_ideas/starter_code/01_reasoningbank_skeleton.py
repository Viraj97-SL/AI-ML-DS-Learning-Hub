"""
ReasoningBank Skeleton — LangGraph Agent with MongoDB Durable State
=====================================================================
Starter code for Theme 1 (Prolonged Coordination) hackathon ideas.

What this gives you:
- LangGraph agent loop persisted to MongoDB via official checkpointer
- Episodic memory: recent events stored with TTL
- Semantic memory: long-term facts extracted from episodes
- Sleep-time consolidation: triggered via EventBridge (cron)

Paper anchor: ReasoningBank (arXiv:2504.09762), Zep temporal KG (arXiv:2501.13956)

Install:
    pip install langgraph langgraph-checkpoint-mongodb pymongo anthropic
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, TypedDict

from anthropic import Anthropic
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.environ.get("MONGODB_DB", "reasoningbank_demo")
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------------------------------------------------------------------------
# MongoDB collections
# ---------------------------------------------------------------------------

episodic_col = db["episodic_memory"]    # Short-term, with TTL index
semantic_col = db["semantic_memory"]    # Long-term facts
strategy_col = db["strategy_rewards"]  # Bandit router history

# Create TTL index on episodic memory (events expire after 7 days)
episodic_col.create_index("expires_at", expireAfterSeconds=0)


# ---------------------------------------------------------------------------
# Agent state (LangGraph TypedDict)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: list[dict]
    current_task: str
    task_id: str
    recalled_context: str
    agent_id: str


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def store_episode(agent_id: str, task_id: str, event: str, outcome: str) -> None:
    """Store a short-term episodic memory with 7-day TTL."""
    episodic_col.insert_one({
        "agent_id": agent_id,
        "task_id": task_id,
        "event": event,
        "outcome": outcome,
        "timestamp": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc).replace(
            day=datetime.now(timezone.utc).day + 7
        ),
    })


def consolidate_to_semantic(agent_id: str) -> int:
    """
    Sleep-time consolidation: extract durable facts from recent episodes.
    Called by EventBridge nightly trigger.
    Returns number of facts extracted.
    """
    recent = list(episodic_col.find(
        {"agent_id": agent_id},
        sort=[("timestamp", -1)],
        limit=50
    ))
    if not recent:
        return 0

    episode_text = "\n".join(
        f"- [{e['timestamp']}] {e['event']} → {e['outcome']}"
        for e in recent
    )

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Extract 3-5 durable facts from these agent episodes that should be "
                f"remembered long-term. Return as JSON array of strings.\n\n{episode_text}"
            )
        }]
    )

    import json
    facts = json.loads(response.content[0].text)
    for fact in facts:
        semantic_col.update_one(
            {"agent_id": agent_id, "fact": fact},
            {"$set": {
                "agent_id": agent_id,
                "fact": fact,
                "extracted_from_n_episodes": len(recent),
                "last_updated": datetime.now(timezone.utc),
                "valid_from": datetime.now(timezone.utc),
                "valid_to": None,
            }},
            upsert=True
        )
    return len(facts)


def recall_context(agent_id: str, current_task: str) -> str:
    """Retrieve relevant memory to prepend to the agent's context."""
    # Pull recent episodes
    recent_episodes = list(episodic_col.find(
        {"agent_id": agent_id},
        sort=[("timestamp", -1)],
        limit=5
    ))
    # Pull semantic memory
    semantic_facts = list(semantic_col.find(
        {"agent_id": agent_id, "valid_to": None},
        limit=10
    ))

    context_parts = []
    if recent_episodes:
        context_parts.append("Recent episodes:\n" + "\n".join(
            f"- {e['event']}: {e['outcome']}" for e in recent_episodes
        ))
    if semantic_facts:
        context_parts.append("Remembered facts:\n" + "\n".join(
            f"- {f['fact']}" for f in semantic_facts
        ))
    return "\n\n".join(context_parts) if context_parts else "No prior context."


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def recall_node(state: AgentState) -> AgentState:
    """Retrieve memory context before reasoning."""
    context = recall_context(state["agent_id"], state["current_task"])
    return {**state, "recalled_context": context}


def reason_node(state: AgentState) -> AgentState:
    """Main reasoning step using Claude."""
    system_prompt = f"""You are a persistent agent with the following memory:

{state["recalled_context"]}

Use this context to inform your response. Be concise."""

    messages = state["messages"] + [{
        "role": "user",
        "content": state["current_task"]
    }]

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=messages
    )

    answer = response.content[0].text
    new_messages = messages + [{"role": "assistant", "content": answer}]

    # Store this interaction as an episode
    store_episode(
        agent_id=state["agent_id"],
        task_id=state["task_id"],
        event=f"Task: {state['current_task'][:100]}",
        outcome=f"Response: {answer[:200]}"
    )

    return {**state, "messages": new_messages}


def should_continue(state: AgentState) -> str:
    """Routing: end after one round for demo; extend for multi-turn."""
    return END


# ---------------------------------------------------------------------------
# LangGraph graph with MongoDB checkpointer
# ---------------------------------------------------------------------------

def build_agent(agent_id: str) -> Any:
    """Build and return a compiled LangGraph agent with MongoDB persistence."""
    graph = StateGraph(AgentState)
    graph.add_node("recall", recall_node)
    graph.add_node("reason", reason_node)

    graph.add_edge(START, "recall")
    graph.add_edge("recall", "reason")
    graph.add_conditional_edges("reason", should_continue)

    # MongoDB checkpointer: agent state survives restarts
    checkpointer = MongoDBSaver(client, DB_NAME)
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uuid

    AGENT_ID = "demo-agent-001"
    agent = build_agent(AGENT_ID)

    # First invocation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [],
        "current_task": "Summarize the key risks of the project we've been tracking.",
        "task_id": str(uuid.uuid4()),
        "recalled_context": "",
        "agent_id": AGENT_ID,
    }

    result = agent.invoke(initial_state, config=config)
    print("Agent response:", result["messages"][-1]["content"])

    # Simulate sleep-time consolidation
    n = consolidate_to_semantic(AGENT_ID)
    print(f"Consolidated {n} facts to semantic memory.")

    # Second invocation — agent remembers
    second_state: AgentState = {
        **initial_state,
        "current_task": "What do you remember about our previous discussion?",
        "task_id": str(uuid.uuid4()),
    }
    result2 = agent.invoke(second_state, config=config)
    print("Agent with memory:", result2["messages"][-1]["content"])
