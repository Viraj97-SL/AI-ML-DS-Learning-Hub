<div align="center">

# 📄 Paper Digest: GraphRAG — Graph-Based Retrieval-Augmented Generation

**arXiv:2404.16130 · April 2024 · Microsoft Research**

</div>

---

## One-Line Summary
Build a knowledge graph from your document corpus, detect communities of related entities, summarize each community with Claude, and answer queries against community summaries — enabling global reasoning across an entire corpus.

## Problem It Solves
Standard RAG answers queries about specific passages but struggles with global questions: "What are the main themes in this corpus?", "How does concept X relate to concept Y across multiple documents?", "What is the overall state of research on topic Z?" These require synthesizing across the entire corpus, not just finding the nearest passage.

## Key Contributions

| Contribution | Detail |
|--------------|--------|
| **Entity extraction** | LLM extracts entities and relations from every document chunk |
| **Community detection** | Leiden algorithm groups related entities into communities |
| **Community summarization** | LLM generates a paragraph-level summary of each community |
| **Global query** | Query is answered against all community summaries (not individual passages) |
| **Local query** | Standard vector search, enhanced by graph context |
| **Multi-level hierarchy** | Communities at multiple granularities (fine to coarse) |

## Two Query Modes

```
Global query: "What are the main findings about climate tipping points?"
→ Retrieve all relevant community summaries
→ Claude synthesizes across all communities
→ Answer reflects entire corpus understanding

Local query: "What does paper X say about AMOC thresholds?"
→ Standard vector search + graph context from entity neighborhood
→ Answer grounded in specific passages
```

## MongoDB Implementation

```python
# Step 1: Extract entities from documents
def extract_entities(text: str) -> list[dict]:
    # Use Claude to extract entities and relations
    # Store in db.kg_entities and db.kg_relations
    pass

# Step 2: Community summarization (after Leiden on the graph)
def summarize_community(entity_ids: list[str]) -> str:
    # Fetch entities and their relations
    # Use Claude to generate a community summary
    pass

# Step 3: Global query against community summaries
def global_query(question: str) -> str:
    all_summaries = list(db.communities.find())
    context = "\n\n".join(c["summary"] for c in all_summaries)
    # Claude synthesizes across all summaries
    pass
```

## When to Use GraphRAG vs Standard RAG

| Query Type | Standard RAG | GraphRAG |
|-----------|-------------|---------|
| "What does passage X say?" | ✅ Better | ❌ Overkill |
| "What are the main themes?" | ❌ Misses global | ✅ Better |
| "How do X and Y relate?" | ❌ Single passage | ✅ Multi-hop graph |
| "Summarize the corpus" | ❌ No global view | ✅ Community summaries |

## Benchmark Results (DRIFT benchmark)
- GraphRAG: 72.1% on global comprehension questions (vs 45.3% for standard RAG)
- Local queries: comparable to standard RAG with graph context bonus

## Relevant Hackathon Ideas
Theme 3: #78 PrecedentBrain (legal citation graph), #80 BiomedHive, #89 GenomeNav, Blueprint 09 Protocol Darwin, Blueprint 10 Carbon Lie Detector

---

*See also: [HippoRAG](hipporag.md) · [Search-R1](search_r1.md) · [ColPali](colpali.md)*
