"""
GraphRAG + HippoRAG PPR Query Skeleton
=======================================
Starter code for Theme 3 (Adaptive Retrieval) hackathon ideas.

What this gives you:
- GraphRAG community summaries built from a document corpus
- HippoRAG 2 Personalized PageRank for multi-hop entity traversal
- MongoDB graphLookup for entity hop queries
- Combined: community-level global reasoning + entity-level multi-hop

Paper anchors:
  GraphRAG (arXiv:2404.16130) — community detection + LLM summaries
  HippoRAG 2 (arXiv:2502.14802) — PPR for multi-hop associative retrieval

Install:
    pip install pymongo anthropic numpy scikit-learn
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import numpy as np
from anthropic import Anthropic
from pymongo import MongoClient

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.environ.get("MONGODB_DB", "graphrag_demo")

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB_NAME]
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Collections
entities_col = db["kg_entities"]          # Nodes: entities extracted from docs
relations_col = db["kg_relations"]        # Edges: entity relationships
communities_col = db["kg_communities"]    # GraphRAG community summaries
docs_col = db["documents"]               # Source documents


# ---------------------------------------------------------------------------
# Step 1: Entity + relation extraction (GraphRAG Phase 1)
# ---------------------------------------------------------------------------

def extract_entities_and_relations(doc_id: str, text: str) -> tuple[list[dict], list[dict]]:
    """
    Use Claude to extract entities and relations from a text chunk.
    Returns (entities, relations).
    """
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Extract entities and relations from this text.
Return JSON with two arrays:
- "entities": [{{"id": "...", "name": "...", "type": "PERSON|ORG|CONCEPT|LOCATION"}}]
- "relations": [{{"from": "entity_id", "to": "entity_id", "type": "relation_type"}}]

Text: {text[:2000]}"""
        }]
    )

    import json
    try:
        data = json.loads(response.content[0].text)
        entities = [{"doc_id": doc_id, **e} for e in data.get("entities", [])]
        relations = [{"doc_id": doc_id, **r} for r in data.get("relations", [])]
        return entities, relations
    except (json.JSONDecodeError, KeyError):
        return [], []


def build_knowledge_graph(documents: list[dict]) -> None:
    """Index all documents into the entity-relation knowledge graph."""
    for doc in documents:
        entities, relations = extract_entities_and_relations(doc["_id"], doc["text"])
        if entities:
            entities_col.insert_many(entities, ordered=False)
        if relations:
            relations_col.insert_many(relations, ordered=False)
        print(f"KG: indexed {len(entities)} entities, {len(relations)} relations from {doc['_id']}")


# ---------------------------------------------------------------------------
# Step 2: Community detection + summarization (GraphRAG Phase 2)
# ---------------------------------------------------------------------------

def detect_communities() -> list[list[str]]:
    """
    Simple community detection via connected components.
    Production: use Leiden algorithm (graspologic library).
    """
    # Build adjacency list
    adj = defaultdict(set)
    for rel in relations_col.find():
        adj[rel["from"]].add(rel["to"])
        adj[rel["to"]].add(rel["from"])

    visited = set()
    communities = []

    def dfs(node: str, community: list):
        visited.add(node)
        community.append(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, community)

    for entity_id in adj:
        if entity_id not in visited:
            community: list[str] = []
            dfs(entity_id, community)
            if len(community) > 1:
                communities.append(community)

    return communities


def summarize_community(entity_ids: list[str]) -> str:
    """Generate a community summary using Claude."""
    entities = list(entities_col.find({"id": {"$in": entity_ids}}, limit=20))
    relations = list(relations_col.find({
        "from": {"$in": entity_ids},
        "to": {"$in": entity_ids}
    }, limit=30))

    entity_text = "\n".join(f"- {e['name']} ({e.get('type', 'unknown')})" for e in entities)
    relation_text = "\n".join(f"- {r['from']} --[{r['type']}]--> {r['to']}" for r in relations)

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"Summarize this knowledge graph community in 2-3 sentences:\n\n"
                f"Entities:\n{entity_text}\n\nRelations:\n{relation_text}"
            )
        }]
    )
    return response.content[0].text


def build_community_summaries() -> int:
    """Build and store community summaries (GraphRAG Phase 2)."""
    communities = detect_communities()
    for i, entity_ids in enumerate(communities):
        summary = summarize_community(entity_ids)
        community_id = f"community_{i}"
        communities_col.update_one(
            {"_id": community_id},
            {"$set": {
                "_id": community_id,
                "entity_ids": entity_ids,
                "summary": summary,
                "size": len(entity_ids)
            }},
            upsert=True
        )
        print(f"Community {i}: {len(entity_ids)} entities — {summary[:80]}...")
    return len(communities)


# ---------------------------------------------------------------------------
# Step 3: HippoRAG Personalized PageRank
# ---------------------------------------------------------------------------

def personalized_pagerank(
    seed_entities: list[str],
    damping: float = 0.85,
    iterations: int = 30,
    top_k: int = 10
) -> list[tuple[str, float]]:
    """
    Run Personalized PageRank from seed entities to find related nodes.
    Seed entities are the query-matched entities.
    Returns sorted list of (entity_id, score).
    """
    # Build adjacency list with weights
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    all_entities: set[str] = set()

    for rel in relations_col.find():
        u, v = rel["from"], rel["to"]
        weight = rel.get("weight", 1.0)
        adj[u][v] = adj[u].get(v, 0) + weight
        adj[v][u] = adj[v].get(u, 0) + weight
        all_entities.update([u, v])

    if not all_entities:
        return []

    entity_list = sorted(all_entities)
    entity_idx = {e: i for i, e in enumerate(entity_list)}
    n = len(entity_list)

    # Personalization vector: uniform over seed entities
    personalization = np.zeros(n)
    valid_seeds = [s for s in seed_entities if s in entity_idx]
    if not valid_seeds:
        return []
    for seed in valid_seeds:
        personalization[entity_idx[seed]] = 1.0 / len(valid_seeds)

    # Build row-normalized transition matrix
    scores = personalization.copy()
    for _ in range(iterations):
        new_scores = np.zeros(n)
        for entity, neighbors in adj.items():
            if entity not in entity_idx:
                continue
            i = entity_idx[entity]
            total_weight = sum(neighbors.values())
            for neighbor, weight in neighbors.items():
                if neighbor in entity_idx:
                    j = entity_idx[neighbor]
                    new_scores[j] += damping * scores[i] * (weight / total_weight)
        new_scores += (1 - damping) * personalization
        scores = new_scores

    # Return top-k entities by PPR score
    ranked = sorted(zip(entity_list, scores.tolist()), key=lambda x: -x[1])
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Step 4: Combined GraphRAG + HippoRAG query
# ---------------------------------------------------------------------------

def graphrag_query(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Full GraphRAG + HippoRAG query pipeline:
    1. Find seed entities from query (BM25 on entity names)
    2. Run PPR to find related entities (multi-hop)
    3. Find community summaries covering those entities
    4. Generate final answer grounded in community summaries
    """
    # Step 1: Find seed entities
    seed_results = list(entities_col.find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}, "id": 1, "name": 1}
    ).sort([("score", {"$meta": "textScore"})]).limit(5))

    seed_entities = [e["id"] for e in seed_results]
    print(f"Seed entities: {[e['name'] for e in seed_results]}")

    # Step 2: PPR traversal
    ppr_results = personalized_pagerank(seed_entities, top_k=top_k * 2)
    relevant_entity_ids = [entity_id for entity_id, _ in ppr_results]
    print(f"PPR top entities: {relevant_entity_ids[:5]}")

    # Step 3: Find communities covering relevant entities
    relevant_communities = list(communities_col.find({
        "entity_ids": {"$in": relevant_entity_ids}
    }).limit(3))

    community_context = "\n\n".join(
        f"Community {i+1}: {c['summary']}"
        for i, c in enumerate(relevant_communities)
    )

    # Step 4: Generate grounded answer
    if not community_context:
        return {"answer": "No relevant context found.", "communities_used": 0, "entities_traversed": 0}

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Answer this question using ONLY the provided context.\n\n"
                f"Context:\n{community_context}\n\n"
                f"Question: {query}"
            )
        }]
    )

    return {
        "answer": response.content[0].text,
        "communities_used": len(relevant_communities),
        "entities_traversed": len(ppr_results),
        "seed_entities": [e["name"] for e in seed_results],
        "community_summaries": [c["summary"] for c in relevant_communities]
    }


# ---------------------------------------------------------------------------
# MongoDB graphLookup helper (alternative to PPR for simple hop queries)
# ---------------------------------------------------------------------------

def graphlookup_hops(start_entity_id: str, max_depth: int = 3) -> list[dict]:
    """
    Use MongoDB $graphLookup for direct graph traversal.
    Faster than PPR for small graphs or simple queries.
    """
    pipeline = [
        {"$match": {"id": start_entity_id}},
        {"$graphLookup": {
            "from": "kg_relations",
            "startWith": "$id",
            "connectFromField": "to",
            "connectToField": "from",
            "as": "path",
            "maxDepth": max_depth,
            "depthField": "hop_depth"
        }},
        {"$project": {
            "name": 1,
            "path_length": {"$size": "$path"},
            "path": {
                "$map": {
                    "input": "$path",
                    "as": "p",
                    "in": {"to": "$$p.to", "type": "$$p.type", "depth": "$$p.hop_depth"}
                }
            }
        }}
    ]
    return list(entities_col.aggregate(pipeline))


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Seed some sample data
    sample_docs = [
        {
            "_id": "doc_001",
            "text": (
                "AMOC weakening is linked to Arctic sea ice loss. "
                "The Atlantic Meridional Overturning Circulation distributes heat globally. "
                "Ditlevsen and Ditlevsen (2023) found early-warning signals in North Atlantic SST data."
            )
        },
        {
            "_id": "doc_002",
            "text": (
                "Amazon dieback threatens to release 90 billion tonnes of CO2. "
                "Deforestation accelerates the feedback loop between drought and forest loss. "
                "The tipping point may occur at 20-25% deforestation."
            )
        }
    ]

    # Create text index for entity search
    try:
        entities_col.create_index([("name", "text"), ("type", "text")])
    except Exception:
        pass

    print("Building knowledge graph...")
    build_knowledge_graph(sample_docs)

    print("\nBuilding community summaries...")
    n = build_community_summaries()
    print(f"Built {n} community summaries")

    print("\nRunning GraphRAG + HippoRAG query...")
    result = graphrag_query("What are the climate tipping point cascades?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Communities used: {result['communities_used']}")
    print(f"Entities traversed: {result['entities_traversed']}")
