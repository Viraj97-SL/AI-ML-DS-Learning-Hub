<div align="center">

# 📄 Paper Digest: Search-R1 — Training LLMs to Reason and Retrieve

**arXiv:2503.09516 · March 2025**

</div>

---

## One-Line Summary
Train an LLM with RL to decide *when* to search, *what* to search for, and *how* to use retrieved results — closing the loop between retrieval and reasoning.

## Problem It Solves
Standard RAG pipelines retrieve once before the LLM reasons. Complex queries require interleaved retrieval and reasoning: "Search for X, reason about it, then search for Y based on what you learned." Search-R1 trains this capability directly via reinforcement learning, using answer quality as the reward signal.

## Key Contributions

| Contribution | Detail |
|--------------|--------|
| **Process reward model** | Rewards the agent for each good search decision, not just the final answer |
| **Interleaved reasoning** | Agent alternates `<think>`, `<search>`, `<result>`, `<think>`, `<answer>` |
| **GRPO training** | Group Relative Policy Optimization — same approach as DeepSeek-R1 |
| **Multi-hop capability** | Learns to decompose complex queries into sequential searches |
| **Domain transfer** | Fine-tuned on one domain transfers to others |

## The Thinking-Searching Loop

```
Query: "Who won the gold medal in the event where the 2023 world record was broken?"

<think> I need to find which event had a world record broken in 2023 </think>
<search> "2023 world record broken" </search>
<result> 100m women's world record broken at 2023 Worlds by Faith Kipyegon </result>
<think> Now I need to find who won the gold medal in that specific event </think>
<search> "Faith Kipyegon 2023 Worlds 100m result" </search>
<result> Faith Kipyegon won gold with world record 4:07.64 in 1500m (correction needed) </result>
<answer> Faith Kipyegon won gold and set the world record in the 1500m at the 2023 World Championships </answer>
```

## How to Implement Without Training

You can get Search-R1-style behavior from Claude with a prompt pattern:

```python
SEARCH_R1_SYSTEM = """You are a research assistant. For complex queries, use this format:
<think> reason about what you need to search for </think>
<search> your search query </search>
[tool will provide results]
<think> reason about the results </think>
<search> follow-up search if needed </search>
<answer> your final answer with citations </answer>"""

# Then use tool_use to intercept <search> tags and call your retriever
```

## Benchmark Results
- BRIGHT (reasoning-intensive retrieval): +31% over standard RAG
- HotpotQA: +18% on multi-hop questions
- PopQA: competitive with much larger models

## Relevant Hackathon Ideas
Theme 3: All 23 ideas benefit from Search-R1-style decomposition.
Especially: #78 PrecedentBrain (multi-hop legal citation chains), #80 BiomedHive, Blueprint 02 Replicant, Blueprint 05 TruthWeight

---

*See also: [ColPali](colpali.md) · [HippoRAG](hipporag.md) · [GraphRAG](graphrag.md)*
