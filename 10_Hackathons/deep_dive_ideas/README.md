<div align="center">

# 🔬 Deep Dive Blueprints

### Maximum-Wow Ideas — Full Implementation Blueprints

[![Count](https://img.shields.io/badge/Blueprints-10-brightgreen?style=for-the-badge)](.)
[![Difficulty](https://img.shields.io/badge/Difficulty-Expert-red?style=for-the-badge)](.)
[![Starter%20Code](https://img.shields.io/badge/Starter%20Code-4%20Skeletons-blue?style=for-the-badge)](starter_code/)

</div>

---

## What Makes a Deep Dive Different

Standard hackathon ideas are single-paragraph sparks. These blueprints go further:

- **Full architecture diagram** — data flow, agent graph, MongoDB schema
- **Anchor paper justification** — each design choice tied to a peer-reviewed source
- **90-second demo script** — beat-by-beat with expected screen output
- **Starter code** — runnable Python skeleton, not pseudocode
- **Evaluation rubric** — how judges will score it, how to maximize each criterion

These are the ideas that win "Best Technical Implementation" and "Most Innovative." They require a team of 2–3 people and 24–72 hours.

---

## The 10 Blueprints

| # | Name | Core Innovation | Themes Used | Difficulty |
|---|------|----------------|-------------|-----------|
| 01 | [Viral Autopsy](#01-viral-autopsy) | Mutation provenance graph for mis/disinformation spread | T1 + T2 + T3 | ⭐⭐⭐⭐⭐ |
| 02 | [Replicant](#02-replicant) | Live arXiv reproducibility scoring with xKG evidence | T1 + T3 | ⭐⭐⭐⭐ |
| 03 | [Portfall](#03-portfall) | Maritime disruption economic cascade model | T1 + T2 | ⭐⭐⭐⭐⭐ |
| 04 | [Tipping Oracle](#04-tipping-oracle) | Climate tipping point cross-institution consensus | T1 + T2 + T3 | ⭐⭐⭐⭐⭐ |
| 05 | [TruthWeight](#05-truthweight) | Real-time health claim credibility from TikTok + JMIR | T2 + T3 | ⭐⭐⭐⭐ |
| 06 | [ChronoLaw](#06-chronolaw) | Bi-temporal law-as-living-graph | T1 + T3 | ⭐⭐⭐⭐ |
| 07 | [Ghostwriter Forensics](#07-ghostwriter-forensics) | Cognitive trace detection in AI-written essays | T1 + T3 | ⭐⭐⭐⭐ |
| 08 | [Exodus Mapper](#08-exodus-mapper) | Climate displacement multi-source adaptive retrieval | T1 + T3 | ⭐⭐⭐⭐ |
| 09 | [Protocol Darwin](#09-protocol-darwin) | Self-evolving protocol spec from agent failures | T1 + T2 | ⭐⭐⭐⭐⭐ |
| 10 | [Carbon Lie Detector](#10-carbon-lie-detector) | Greenwash score from satellite + supply chain + academic | T2 + T3 | ⭐⭐⭐⭐⭐ |

**T1** = Theme 1 (Prolonged Coordination) · **T2** = Theme 2 (Multi-Agent Collaboration) · **T3** = Theme 3 (Adaptive Retrieval)

---

## Starter Code

Four runnable Python skeletons are in [`starter_code/`](starter_code/):

| File | What It Gives You |
|------|------------------|
| [`01_reasoningbank_skeleton.py`](starter_code/01_reasoningbank_skeleton.py) | LangGraph agent loop + MongoDB checkpointer + episodic memory with TTL |
| [`04_a2a_handshake.py`](starter_code/04_a2a_handshake.py) | A2A capability advertisement + task negotiation between two agents |
| [`07_colpali_index.py`](starter_code/07_colpali_index.py) | ColPali PDF indexing → MongoDB Atlas Vector Search pipeline |
| [`09_graphrag_query.py`](starter_code/09_graphrag_query.py) | GraphRAG community summaries + HippoRAG PPR multi-hop query |

---

## Which Blueprint to Pick

| Your Situation | Blueprint |
|----------------|-----------|
| Strongest team skill: NLP + graph | 01 Viral Autopsy or 10 Carbon Lie Detector |
| Strongest team skill: distributed systems | 03 Portfall or 09 Protocol Darwin |
| Strongest team skill: retrieval / RAG | 02 Replicant or 05 TruthWeight |
| Want a clear societal story for judges | 04 Tipping Oracle or 08 Exodus Mapper |
| Solo with 48h | 06 ChronoLaw or 07 Ghostwriter Forensics |
| Maximum technical depth | 09 Protocol Darwin |
| Maximum "wow" in demo | 10 Carbon Lie Detector |

---

## Navigation

| Previous | Home |
|----------|------|
| [← 10_Hackathons](../README.md) | [🏠 All Themes](../README.md) |
