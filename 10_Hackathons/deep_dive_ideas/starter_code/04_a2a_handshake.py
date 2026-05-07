"""
A2A Handshake Skeleton — Agent-to-Agent Protocol (Google 2025)
=============================================================
Starter code for Theme 2 (Multi-Agent Collaboration) hackathon ideas.

What this gives you:
- Agent capability advertisement via A2A AgentCard
- Task negotiation: one agent sends a task, another accepts/rejects
- MongoDB Change Streams as the A2A message bus
- Async multi-agent event loop

Paper anchor: A2A Protocol (Google 2025), Magentic-One (arXiv:2411.04468)

Install:
    pip install pymongo anthropic python-dotenv
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic
from pymongo import MongoClient
from pymongo.change_stream import ChangeStream

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.environ.get("MONGODB_DB", "a2a_demo")

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB_NAME]
anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# MongoDB collections acting as the A2A message bus
agent_registry_col = db["agent_registry"]
task_queue_col = db["task_queue"]
result_col = db["task_results"]


# ---------------------------------------------------------------------------
# A2A Data Structures (simplified from Google A2A spec)
# ---------------------------------------------------------------------------

@dataclass
class AgentCard:
    """Capability advertisement — what this agent can do."""
    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    input_types: list[str]
    output_types: list[str]
    resource_limits: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskMessage:
    """A2A task message sent from orchestrator to specialist."""
    task_id: str
    from_agent: str
    to_agent: str
    task_type: str
    payload: dict[str, Any]
    priority: int = 5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"


@dataclass
class TaskResult:
    """Result message sent back after task completion."""
    task_id: str
    from_agent: str
    result: dict[str, Any]
    status: str  # "completed" | "failed" | "partial"
    completed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None


# ---------------------------------------------------------------------------
# Base Agent class
# ---------------------------------------------------------------------------

class A2AAgent:
    def __init__(self, card: AgentCard):
        self.card = card
        self._running = False

    def register(self) -> None:
        """Publish capability card to MongoDB agent registry."""
        agent_registry_col.replace_one(
            {"agent_id": self.card.agent_id},
            {**asdict(self.card), "registered_at": datetime.now(timezone.utc)},
            upsert=True
        )
        print(f"[{self.card.name}] Registered with capabilities: {self.card.capabilities}")

    def send_task(self, to_agent_id: str, task_type: str, payload: dict) -> str:
        """Send a task to another agent via the MongoDB task queue."""
        task = TaskMessage(
            task_id=str(uuid.uuid4()),
            from_agent=self.card.agent_id,
            to_agent=to_agent_id,
            task_type=task_type,
            payload=payload
        )
        task_queue_col.insert_one(asdict(task))
        print(f"[{self.card.name}] Sent task {task.task_id[:8]}... to {to_agent_id}")
        return task.task_id

    def send_result(self, task_id: str, result: dict, status: str = "completed") -> None:
        """Return result to the requesting agent."""
        res = TaskResult(
            task_id=task_id,
            from_agent=self.card.agent_id,
            result=result,
            status=status
        )
        result_col.insert_one(asdict(res))
        # Mark task as completed in queue
        task_queue_col.update_one(
            {"task_id": task_id},
            {"$set": {"status": status}}
        )

    async def handle_task(self, task: dict) -> None:
        """Override in subclasses to implement task-specific logic."""
        raise NotImplementedError

    async def listen(self) -> None:
        """Listen for incoming tasks via MongoDB Change Stream."""
        self._running = True
        pipeline = [{"$match": {
            "operationType": "insert",
            "fullDocument.to_agent": self.card.agent_id,
            "fullDocument.status": "pending"
        }}]
        print(f"[{self.card.name}] Listening for tasks...")
        with task_queue_col.watch(pipeline, full_document="updateLookup") as stream:
            async for change in self._async_stream(stream):
                if not self._running:
                    break
                task = change["fullDocument"]
                print(f"[{self.card.name}] Received task: {task['task_type']}")
                await self.handle_task(task)

    async def _async_stream(self, stream: ChangeStream):
        """Wrap synchronous Change Stream in async generator."""
        loop = asyncio.get_event_loop()
        while self._running:
            change = await loop.run_in_executor(None, lambda: next(stream, None))
            if change:
                yield change
            else:
                await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Example specialist agents
# ---------------------------------------------------------------------------

class SummarizerAgent(A2AAgent):
    """Specialist: summarizes text documents."""

    async def handle_task(self, task: dict) -> None:
        if task["task_type"] != "summarize":
            self.send_result(task["task_id"], {}, status="failed")
            return

        text = task["payload"].get("text", "")
        response = await anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": f"Summarize in 3 bullet points:\n\n{text}"}]
        )
        summary = response.content[0].text
        self.send_result(task["task_id"], {"summary": summary})
        print(f"[{self.card.name}] Summary completed for task {task['task_id'][:8]}")


class OrchestratorAgent(A2AAgent):
    """Orchestrator: decomposes work and dispatches to specialists."""

    def find_capable_agent(self, capability: str) -> str | None:
        """Look up agent registry for a capable specialist."""
        agent = agent_registry_col.find_one({
            "capabilities": capability,
            "agent_id": {"$ne": self.card.agent_id}
        })
        return agent["agent_id"] if agent else None

    async def handle_task(self, task: dict) -> None:
        """Orchestrator receives high-level tasks and delegates."""
        if task["task_type"] == "process_document":
            # Delegate summarization to a SummarizerAgent
            summarizer_id = self.find_capable_agent("summarize")
            if summarizer_id:
                sub_task_id = self.send_task(
                    to_agent_id=summarizer_id,
                    task_type="summarize",
                    payload=task["payload"]
                )
                # Wait for result
                await self._wait_for_result(sub_task_id)

    async def _wait_for_result(self, task_id: str, timeout_s: float = 30.0) -> dict | None:
        """Poll for a task result (simple demo; production uses Change Stream)."""
        import time
        start = time.time()
        while time.time() - start < timeout_s:
            result = result_col.find_one({"task_id": task_id})
            if result:
                print(f"[{self.card.name}] Got result for {task_id[:8]}: {result['status']}")
                return result
            await asyncio.sleep(0.5)
        print(f"[{self.card.name}] Timeout waiting for {task_id[:8]}")
        return None


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

async def main():
    # Create agents with capability cards
    orchestrator = OrchestratorAgent(AgentCard(
        agent_id="orchestrator-001",
        name="Orchestrator",
        description="Decomposes work and dispatches to specialists",
        capabilities=["orchestrate", "decompose"],
        input_types=["task_request"],
        output_types=["task_delegation"]
    ))

    summarizer = SummarizerAgent(AgentCard(
        agent_id="summarizer-001",
        name="Summarizer",
        description="Summarizes text documents",
        capabilities=["summarize"],
        input_types=["text"],
        output_types=["summary"],
        resource_limits={"max_payload_mb": 10}
    ))

    # Register both agents
    orchestrator.register()
    summarizer.register()

    # Start summarizer listener in background
    listener_task = asyncio.create_task(summarizer.listen())

    # Orchestrator sends a document for processing
    await asyncio.sleep(0.5)  # let listener start
    orchestrator.send_task(
        to_agent_id="summarizer-001",
        task_type="summarize",
        payload={"text": (
            "The A2A protocol enables agents to discover each other via capability cards, "
            "negotiate task assignments, and exchange results through a standardized message format. "
            "Unlike RPC, A2A is asynchronous and agent-initiated. Unlike MCP, A2A focuses on "
            "agent-to-agent delegation rather than tool invocation."
        )}
    )

    # Wait for completion
    await asyncio.sleep(5)
    listener_task.cancel()

    # Show registry
    print("\n--- Agent Registry ---")
    for agent in agent_registry_col.find():
        print(f"  {agent['name']}: {agent['capabilities']}")


if __name__ == "__main__":
    asyncio.run(main())
