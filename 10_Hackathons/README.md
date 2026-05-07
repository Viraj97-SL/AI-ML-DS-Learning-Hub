<div align="center">

# 🏆 Hackathon Ideas Hub

### 100+ Research-Grounded Ideas Across 3 Themes

[![Ideas](https://img.shields.io/badge/Ideas-100%2B-brightgreen?style=for-the-badge)](.)
[![Themes](https://img.shields.io/badge/Themes-3-blue?style=for-the-badge)](.)
[![Stack](https://img.shields.io/badge/Stack-MongoDB%20Atlas%20%2B%20AWS-orange?style=for-the-badge)](.)
[![Papers](https://img.shields.io/badge/Paper%20Anchors-12%2B-purple?style=for-the-badge)](.)
[![Updated](https://img.shields.io/badge/Updated-May%202026-red?style=for-the-badge)](.)

</div>

---

## Why This Section Exists

Hackathons are the fastest forcing function for applying cutting-edge research to real problems. A 48-hour sprint forces you to ship, demo, and defend your technical choices — skills that no MOOC teaches.

Every idea in this section is:
- **Grounded in a 2024–2026 paper** — not vibe-coded from a blog post
- **Tied to a named industry pain point** — so the judges' impact criteria are pre-loaded
- **Demoable in 90 seconds** — with a specific "wow moment" identified
- **Built on MongoDB Atlas + AWS** — the dominant agentic-AI stack as of 2026

> **The 3 signature demo moves**: (a) crash + restart from MongoDB checkpointer *(proves durability)*, (b) stream a fresh event into a live panel via Change Streams *(proves reactivity)*, (c) show ReasoningBank entries growing across a "before/after" run *(proves learning)*. Layer any two of these in 3 minutes and you have a credible agent showcase.

---

## Three Themes at a Glance

| Theme | Ideas | Core Concept | Key Paper Anchors |
|-------|-------|-------------|-------------------|
| [🧠 Theme 1: Prolonged Coordination](theme_1_prolonged_coordination/README.md) | 33 | Durable agents surviving weeks, months, years | ReasoningBank · MIRIX · SagaLLM · VIGIL · Zep |
| [🤝 Theme 2: Multi-Agent Collaboration](theme_2_multi_agent_collaboration/README.md) | 34 | Specialist swarms, A2A protocol, orchestrators | A2A Protocol · MCP · Magentic-One · AutoGen |
| [🔍 Theme 3: Adaptive Retrieval](theme_3_adaptive_retrieval/README.md) | 23 | RAG that learns, self-improving search | ColPali · Search-R1 · GraphRAG · HippoRAG |

---

## Choose Your Idea

### By Time Budget

| Time Budget | Solo | 2–3 Person Team | 4+ Person Team |
|-------------|------|-----------------|----------------|
| **24 hours** | Theme 3 (Difficulty ⭐⭐) | Theme 2 (Difficulty ⭐⭐⭐) | Theme 2 (Difficulty ⭐⭐⭐⭐) |
| **48 hours** | Theme 1 (Difficulty ⭐⭐⭐) | Any theme (⭐⭐⭐) | Deep Dives |
| **1 week** | Any Deep Dive | Full Deep Dive + paper | Full stack production |

### By Domain

| You care about... | Start with |
|-------------------|------------|
| Healthcare / biomedical | #2 NemoRecall · #36 EndPoint · #80 BiomedHive |
| Finance / fintech | #8 MarketWatch13F · #57 FederatedFraud · #90 EdgarLinguist |
| Climate / sustainability | #4 QueueClear · #44 CarbonNetwork · #83 SatelliteMind |
| Legal / compliance | #1 ProsecuteIQ · #42 CourtBench · #78 PrecedentBrain |
| Cybersecurity | #6 APT-Hunter · #38 RedBlueLoop · #81 ThreatLens |
| Social impact | #45 DisasterCluster · #49 CivicJustice · #92 AsylumLens |
| Developer tools | #9 EvergreenIDE · #52 CodeForge · #84 RepoSeer |
| Gen Z / media | Deep Dives: Viral Autopsy · TruthWeight · MemeFossil |

---

## 🔬 Deep Dive Blueprints (Maximum Wow)

Ten ideas with complete architecture diagrams, MongoDB schemas, AWS service maps, and timestamp-by-timestamp 90-second demo scripts:

| # | Title | Theme | Domain | Wow Moment |
|---|-------|-------|--------|-----------|
| 1 | [Viral Autopsy](deep_dive_ideas/01_viral_autopsy.md) | Multi-Agent | Journalism / Gen Z | Mutation provenance graph freezes node-by-node as agents work |
| 2 | [Replicant](deep_dive_ideas/02_replicant.md) | Multi-Agent | Open Science | Live arXiv paper fails reproducibility check on-screen |
| 3 | [Portfall](deep_dive_ideas/03_portfall.md) | Multi-Agent + Prolonged | Global Trade | Full economic cascade from one shipping disruption |
| 4 | [Tipping Oracle](deep_dive_ideas/04_tipping_oracle.md) | Multi-Agent | Climate Science | AMOC tipping probability updating live as institutions share data |
| 5 | [TruthWeight](deep_dive_ideas/05_truthweight.md) | Adaptive Retrieval | Gen Z / Health | Health claim credibility sidebar appearing WHILE video plays |
| 6 | [ChronoLaw](deep_dive_ideas/06_chronolaw.md) | Prolonged | Legal / Government | Law visibly changing on-screen as rulings injected |
| 7 | [Ghostwriter Forensics](deep_dive_ideas/07_ghostwriter_forensics.md) | Multi-Agent | Education | AI essay missing "thinking scars" that real essays have |
| 8 | [Exodus Mapper](deep_dive_ideas/08_exodus_mapper.md) | Adaptive Retrieval | Humanitarian | Displacement estimate updates every 30 seconds from new sources |
| 9 | [Protocol Darwin](deep_dive_ideas/09_protocol_darwin.md) | Multi-Agent | Open Source | Protocol spec visibly mutating from real agent failures |
| 10 | [Carbon Lie Detector](deep_dive_ideas/10_carbon_lie_detector.md) | Adaptive Retrieval | Climate / Finance | Corporate logo + Greenwash Score 34/100 appearing with satellite evidence |

---

## Stack of Record

All ideas assume this production-grade stack:

```
MongoDB Atlas
├── Vector Search     — hybrid dense+sparse, $rankFusion
├── Change Streams    — real-time event bus
├── Time-Series       — sensor data, AIS, market prices
├── GridFS            — large artifacts (PDFs, videos, CT scans)
└── Checkpointer      — LangGraph persistent agent state

AWS
├── Bedrock           — Claude 3.5/4.x, Nova, Titan Embeddings
├── AgentCore         — managed agent execution
├── Strands           — tool-first agent framework (May 2025)
├── Lambda            — event-driven agent triggers
├── EventBridge       — scheduled sleep-time consolidation
└── S3                — artifact storage with hash verification
```

---

## Idea Anatomy

Every idea in this section follows 8 fields:

| Field | Description |
|-------|-------------|
| **Domain** | Industry vertical |
| **Difficulty** | 1–5 ⭐ |
| **Time Budget** | 24h / 48h / 1 week |
| **Hook** | One sentence that makes a judge lean forward |
| **Concept** | What it actually does (3–5 bullets) |
| **Paper Anchor** | The 2024–2026 paper this is grounded in |
| **MongoDB + AWS Sketch** | Specific service usage (not generic "use cloud") |
| **Demo Script** | 3-step outline with the wow moment identified |

For full templates, see [templates/](templates/).

---

## The Stakes Formula

Every great hackathon demo opens with **one number that makes the audience feel something**:

> *"75–90% of published papers cannot be reproduced. AI just made that worse."*  
> *"1 in 3 Gen Z checks TikTok before their doctor."*  
> *"The Red Sea crisis cost UK supermarkets £340M in 11 weeks."*  
> *"A meme you shared last week was born in 2013 on a neo-Nazi forum."*

Don't open with architecture. Open with stakes. Then show the machine solving it.

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← 09_Competitions](../09_Competitions/README.md) | [🏠 README](../README.md) | [11_Recent_Topics →](../11_Recent_Topics/README.md) |
