<div align="center">

# 📄 Paper Digest: MCP — Model Context Protocol

**Anthropic 2024 · Open standard for agent-tool communication**

</div>

---

## One-Line Summary
A standardized protocol for connecting LLM agents to external tools, data sources, and APIs — the USB-C of AI agent tooling.

## Problem It Solves
Every agent framework reimplements tool integration from scratch. Connecting an agent to a database, an API, a file system, or a code interpreter requires framework-specific code. MCP provides one protocol that any LLM client can use to discover and invoke tools, and any tool server can implement.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **MCP Server** | Exposes tools, resources, and prompts via the MCP protocol |
| **MCP Client** | The LLM agent that discovers and calls tools |
| **Tool** | A callable function with a schema (like OpenAPI for agent tools) |
| **Resource** | A data source the agent can read (file, database, API endpoint) |
| **Prompt** | A reusable prompt template the server exposes |
| **Transport** | stdio (local) or SSE (remote over HTTP) |

## Tool Schema Example

```json
{
  "name": "search_pubmed",
  "description": "Search PubMed for biomedical literature",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"},
      "max_results": {"type": "integer", "default": 10}
    },
    "required": ["query"]
  }
}
```

## MCP Server in Python (FastMCP)

```python
from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient

mcp = FastMCP("MongoDB Atlas MCP Server")
db = MongoClient(os.environ["MONGODB_URI"])["my_database"]

@mcp.tool()
def search_papers(query: str, max_results: int = 10) -> list[dict]:
    """Search the research paper corpus."""
    return list(db.papers.find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(max_results))

if __name__ == "__main__":
    mcp.run()
```

## Claude Code + MCP

```json
// ~/.claude/settings.json — add MCP servers
{
  "mcpServers": {
    "mongodb-atlas": {
      "command": "python",
      "args": ["my_mcp_server.py"],
      "env": {"MONGODB_URI": "..."}
    }
  }
}
```

## vs A2A

MCP and A2A are complementary:
- **MCP**: Agent → Tool (agent invokes a function, gets back a result)
- **A2A**: Agent → Agent (agent delegates a task to another agent with full lifecycle management)

Use MCP when you're extending a single agent's capabilities. Use A2A when you need autonomous peer agents that can initiate work independently.

## Relevant Hackathon Ideas
All themes benefit from MCP: use it to expose MongoDB queries, satellite data APIs, PubMed search, and offset registries as agent tools without reinventing tool integration.

---

*See also: [A2A Protocol](a2a_protocol.md) · [Magentic-One](magentic_one.md)*
