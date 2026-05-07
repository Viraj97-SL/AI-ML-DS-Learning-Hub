<div align="center">

# 📄 Paper Digest: HippoRAG 2 — Multi-Hop Associative Memory for RAG

**arXiv:2502.14802 · February 2025**

</div>

---

## One-Line Summary
Inspired by the hippocampus's role in human memory, HippoRAG 2 uses Personalized PageRank over a knowledge graph to find multi-hop connections that standard embedding search misses.

## Problem It Solves
Standard vector search finds semantically similar passages but fails at multi-hop reasoning: "Which protein is produced by the gene that regulates the pathway affected by drug X?" requires 3 hops across different documents. HippoRAG builds a knowledge graph from your corpus and uses PPR to traverse it, finding indirect connections.

## Key Contributions

| Contribution | Detail |
|--------------|--------|
| **OpenIE extraction** | Extracts (subject, predicate, object) triples from all documents |
| **Synonym node merging** | Merges equivalent entity mentions across documents |
| **Personalized PageRank** | PPR from query-matched nodes finds multi-hop relevant paths |
| **Dense passage retrieval** | PPR results augmented with standard dense retrieval |
| **HippoRAG 2 improvements** | Better entity linking, improved synonym detection vs. v1 |

## The PPR Intuition

```
Query: "What disease does the pathway regulated by BRCA1 affect?"

1. Find seed nodes: {BRCA1} (exact match in KG)
2. Run PPR from BRCA1: 
   BRCA1 → DNA_repair_pathway (high PPR score)
   DNA_repair_pathway → genome_stability (medium PPR)
   genome_stability → breast_cancer (high PPR — many edges)
3. Top nodes by PPR: {DNA_repair_pathway, breast_cancer, BRCA2, ...}
4. Retrieve passages mentioning these nodes
```

## MongoDB Implementation

```python
def build_hipporag_kg(documents: list[dict]) -> None:
    for doc in documents:
        triples = extract_triples(doc["text"])  # OpenIE
        for subj, pred, obj in triples:
            # Upsert entities
            db.entities.update_one({"name": subj}, {"$set": {"name": subj}}, upsert=True)
            db.entities.update_one({"name": obj}, {"$set": {"name": obj}}, upsert=True)
            # Add relation
            db.relations.insert_one({
                "from": subj, "to": obj, "predicate": pred,
                "doc_id": doc["_id"]
            })

def ppr_query(seed_entities: list[str], top_k: int = 10) -> list[str]:
    """Run PPR from seed entities, return top-k entity names."""
    # Build adjacency from relations collection
    # Run PageRank with personalization on seed entities
    # Return top-k by PPR score
    pass  # See starter_code/09_graphrag_query.py for full implementation
```

## HippoRAG vs GraphRAG

| Aspect | HippoRAG | GraphRAG |
|--------|----------|---------|
| **Strength** | Multi-hop entity traversal | Global corpus understanding |
| **Graph type** | Entity-relation KG (fine-grained) | Community summaries (coarse-grained) |
| **Query type** | "Find the path from X to Y" | "What are the themes?" |
| **Best for** | Gene→protein→disease chains | Cross-document synthesis |

They are complementary: use GraphRAG for global queries, HippoRAG for multi-hop entity queries.

## Benchmark Results
- MuSiQue (multi-hop): +22% over standard RAG
- HotpotQA: +15% on bridge questions
- 2WikiMultiHop: +19%

## Relevant Hackathon Ideas
Theme 3: #80 BiomedHive (gene→protein→pathway→disease), #89 GenomeNav, #78 PrecedentBrain, Blueprint 01 Viral Autopsy (mutation→account→cascade), Blueprint 08 Exodus Mapper

---

*See also: [GraphRAG](graphrag.md) · [Search-R1](search_r1.md)*
