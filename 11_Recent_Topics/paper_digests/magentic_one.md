<div align="center">

# 📄 Paper Digest: Magentic-One

**arXiv:2411.04468 · November 2024 · Microsoft Research**

</div>

---

## One-Line Summary
A generalist multi-agent architecture with a central Orchestrator managing four specialist agents (WebSurfer, FileSurfer, Coder, ComputerTerminal) that together can complete real-world web and computer tasks.

## Problem It Solves
Single-agent systems struggle with tasks requiring multiple tools across different interfaces (browser, file system, code execution, terminal). Magentic-One decomposes these into specialist agents and provides an Orchestrator that plans, delegates, monitors, and recovers from failures.

## Key Contributions

| Contribution | Detail |
|--------------|--------|
| **Orchestrator** | Maintains a task ledger; plans sub-tasks; routes to specialists; detects stalls |
| **WebSurfer** | LLM-controlled browser automation (built on browser-use/Playwright) |
| **FileSurfer** | File system navigation, document reading, search |
| **Coder** | Writes and reviews code; generates test cases |
| **ComputerTerminal** | Executes shell commands; manages processes |
| **Task ledger** | Persistent record of completed sub-tasks; avoids re-doing completed work |

## Architecture

```
                    [Orchestrator]
                    /    |    |    \
            [WebSurfer] [FileSurfer] [Coder] [Terminal]
                    \    |    |    /
                    [Shared Workspace]
                    (MongoDB / file system)
```

## How to Use in a Hackathon

Magentic-One is available as `autogen-ext[magentic-one]`:

```python
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

client = AnthropicChatCompletionClient(model="claude-sonnet-4-6")
team = MagenticOne(client=client)

# Run a complex task
result = await team.run(task="Research the top 5 papers on AMOC and summarize key findings")
```

For MongoDB integration: store the task ledger in MongoDB so the team can resume after interruption (pairs with LangGraph checkpointer).

## Benchmark Results
- GAIA benchmark: 38.0% (first generalist system to score >35%)
- WebArena: competitive with specialized web agents
- HumanEval: 85%+ on coding tasks via Coder agent

## Gotchas
- WebSurfer requires a Chrome/Chromium browser on the execution environment
- Orchestrator can get into loops on ambiguous tasks; add a max_turns limit
- Each specialist is a full LLM call; costs add up quickly — use Haiku for specialist agents

## Relevant Hackathon Ideas
Theme 2: #50 DeepResearch, #55 DevOpsSwarm, #60 AuditCrawler, #70 AgentMarket

---

*See also: [A2A Protocol](a2a_protocol.md) · [VIGIL](vigil.md)*
