# 🔍 Theme 3: All 23 Adaptive Retrieval Ideas

> Each idea features a retrieval system that adapts its strategy per query type and improves from feedback.

---

## Quick-Pick Guide

| Goal | Best Ideas |
|------|-----------|
| Legal / compliance domain | #78, #82, #95 |
| Healthcare research | #80, #89, #96 |
| Finance / regulatory | #90, #82 |
| Solo, 24-hour sprint | #93, #98, #91 |
| Maximum "wow" | Deep Dives: TruthWeight · Carbon Lie Detector · #100 AdaptiveAtlas |
| CV / multimodal skills | #79, #83, #88, #94 |

---

## #78 — PrecedentBrain
**Domain:** Legal | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Standard semantic search returns overruled cases. Lawyers get sanctioned. This system demotes them automatically.

**Concept:**
- Hybrid Atlas vector + BM25 + `$graphLookup` over citation graph
- Reranker tuned on Shepardized "negative treatment" signals (overruled / distinguished / criticized)
- ReasoningBank stores winning query strategies per practice area (e.g., "tort claims: citation-graph first, then semantic")

**Paper Anchor:** [BRIGHT (arXiv:2407.12883)](https://arxiv.org/abs/2407.12883) + HippoRAG PPR

**MongoDB + AWS Sketch:** Citation graph with `$graphLookup`; negative-treatment tags on each case; hybrid index; Voyage rerank-2.5 with legal instruction; Bedrock for query decomposition

**Demo Script:** (1) Enter research question → (2) famous overruled case appears in results → (3) show it demoted live as negative treatment weighting applied → (4) authority heat-bar per result → (5) ReasoningBank: "tort queries: graph-first wins 2.3× vs. vector-only"

**Stretch Goal:** Jurisdiction-specific authority weighting (binding vs. persuasive precedent)

---

## #79 — ColPriorArt
**Domain:** IP / Legal | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Chemistry patent prior art hides in diagrams and Markush structures that text-only search can't see.

**Concept:**
- ColPali page embeddings for figures + SMILES substructure index in MongoDB
- Agent decomposes patent claims and routes each limitation to text, figure, or structure retrieval
- First open multimodal prior-art system with adaptive routing

**Paper Anchor:** [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) + SMILES substructure search

**MongoDB + AWS Sketch:** Page embedding collection (ColPali); SMILES structure index; hybrid routing logic; Bedrock for claim decomposition; Atlas Vector Search for figure matching

**Demo Script:** (1) Drop chemistry patent → (2) claim decomposed into 3 elements → (3) element 1 routes to text, element 2 routes to figures → (4) 2007 paper retrieved via figure match → (5) highlighted figure region matched to claim element

**Stretch Goal:** Markush structure enumeration: expand generic chemical formula to specific instances

---

## #80 — BiomedHive
**Domain:** Drug Discovery | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Drug repurposing requires reasoning across PubMed, arXiv, ClinicalTrials, and bioRxiv simultaneously — with retraction tracking.

**Concept:**
- HippoRAG-style PPR over UMLS/MeSH knowledge graph for multi-hop reasoning
- Corpus-specific rerankers; retracted papers automatically excluded
- DPO pairs generated from researcher feedback; reranker continuously fine-tuned

**Paper Anchor:** [HippoRAG 2 (arXiv:2502.14802)](https://arxiv.org/abs/2502.14802) + retraction tracking

**MongoDB + AWS Sketch:** UMLS/MeSH graph; multi-corpus embeddings with source tags; retraction index; Voyage rerank-2.5; Bedrock for multi-hop reasoning

**Demo Script:** (1) Query "drugs hitting IL-6 approved for non-inflammation" → (2) multi-hop PPR across drug → target → pathway → disease → (3) retracted paper excluded live → (4) PageRank flowing across KG shown → (5) 3 candidate drugs returned with evidence chains

**Stretch Goal:** Confidence propagation: show how retraction of one paper affects downstream confidence scores

---

## #81 — ThreatLens
**Domain:** Cybersecurity | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** CTI analysts spend hours on queries that return noise — this bandit router learns which retrieval mode wins for each threat class.

**Concept:**
- Bandit router across BM25 / vector / graph (MITRE ATT&CK) / web
- Reward = analyst-validated relevance; MongoDB stores trace + reward per query
- Search-R1 RL applied to CTI corpus; bandit arm winning probabilities animate live

**Paper Anchor:** [Search-R1 (arXiv:2503.09516)](https://arxiv.org/abs/2503.09516) + MITRE ATT&CK graph

**MongoDB + AWS Sketch:** CTI corpus (text + IOC feeds); MITRE ATT&CK graph; bandit reward store; Voyage rerank-2.5; Bedrock for threat reasoning

**Demo Script:** (1) Drop IOC → (2) bandit picks vector first → (3) analyst thumbs down → (4) reward updates → (5) graph mode tried → (6) bandit arm probabilities animate → (7) graph consistently wins for this threat class

**Stretch Goal:** Cross-session bandit learning: analyst from yesterday's session trained the model for today's

---

## #82 — ComplianceCompass
**Domain:** Compliance | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** GDPR and CCPA give different answers to the same question — this system knows which answer belongs in which jurisdiction.

**Concept:**
- Jurisdiction-tagged vector indexes in MongoDB
- Agent picks indexes per query and re-weights by recency/version
- HyPA-RAG-style parameter adaptation: different retrieval hyperparameters per jurisdiction

**Paper Anchor:** HyPA-RAG parameter adaptation + Zep bi-temporal

**MongoDB + AWS Sketch:** Per-jurisdiction embedding collections; version metadata with `valid_from`/`valid_to`; Bedrock for legal reasoning; Atlas hybrid search per jurisdiction

**Demo Script:** (1) Same question routed across EU/CA/US → (2) divergent retrieved clauses shown side by side → (3) conflict between GDPR Art. 9 and CCPA highlighted → (4) version-aware: shows which answer applies before/after EU Digital Omnibus

**Stretch Goal:** Compliance drift meter: show how far a company's practice is from current law

---

## #83 — SatelliteMind
**Domain:** Climate / Aerospace | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Earth observation queries need cross-modal fusion — optical + SAR + time-series + text, all in one retrieval.

**Concept:**
- Agent picks modality per query type; `$rankFusion` blends optical + SAR + text scores
- MongoDB ColPali-style patch embeddings + geospatial indexes
- Uncertainty surfaces per result: "SAR confidence 0.82; optical confidence 0.61 due to cloud cover"

**Paper Anchor:** [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) + multi-modal fusion

**MongoDB + AWS Sketch:** Satellite patch embeddings (ColPali); geospatial collection; temporal metadata; Voyage multimodal; Bedrock for result synthesis

**Demo Script:** (1) Query "deforestation in São Felix, last 90 days" → (2) optical partially obscured → (3) agent switches to SAR → (4) results returned with confidence per source → (5) time-slider scrubbing through evidence

**Stretch Goal:** Change detection: highlight specific pixels that changed between two retrievals

---

## #84 — RepoSeer
**Domain:** Developer Tools | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Code search improves from every PR merge — this system tracks which retrieval strategies work for which code patterns.

**Concept:**
- Function + NL + call-graph indexes; agent picks per query class
- Thumbs-up from code review updates strategy weights in ReasoningBank
- "Call-graph hop succeeded" banner shown when graph traversal finds what vector search missed

**Paper Anchor:** [Search-R1 (arXiv:2503.09516)](https://arxiv.org/abs/2503.09516) + CoSIL code search

**MongoDB + AWS Sketch:** Function embeddings; call-graph collection with `$graphLookup`; strategy weights ReasoningBank; Bedrock for code reasoning; GitHub webhook triggers

**Demo Script:** (1) Type vague intent query → (2) vector search returns wrong function → (3) call-graph hop finds correct one → (4) "call-graph hop succeeded" banner → (5) ReasoningBank update: "async handler patterns: graph-first"

**Stretch Goal:** Cross-repo code search with shared (anonymized) strategy weights

---

## #85 — MachineMemoryFMEA
**Domain:** Manufacturing | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** A 7-year-old failure report matches today's vibration signature — but only if you can retrieve across modalities.

**Concept:**
- Query = sensor-time-series window; retrieves matching past failures
- MongoDB time-series + ColPali on CAD pages; time-series → document retrieval is genuinely novel
- Cosine similarity on time-series windows to find similar past failure modes

**Paper Anchor:** [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) + time-series similarity search

**MongoDB + AWS Sketch:** FMEA history time-series; CAD page embeddings (ColPali); sensor pattern index; Bedrock for failure reasoning; Atlas Vector Search for pattern matching

**Demo Script:** (1) Stream synthetic vibration signal → (2) time-series pattern query → (3) matching 2018 incident retrieved → (4) CAD page showing the bearing location highlighted → (5) maintenance recommendation generated

**Stretch Goal:** Failure probability forecasting: how many hours until likely failure based on pattern match?

---

## #86 — MaterialsLens
**Domain:** Materials Science | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Materials queries go both ways — "find me a material with these properties" and "what properties does this structure have?" — this system handles both.

**Concept:**
- Bidirectional retrieval: structure→property and property→structure modes
- Agent infers query direction from the input; switches strategy accordingly
- MongoDB hybrid search over materials embeddings from Citrination + Materials Project + MatBench

**Paper Anchor:** Materials Project API + MatBench benchmark + bidirectional embedding

**MongoDB + AWS Sketch:** Materials embedding collection (structure + property vectors); bidirectional routing logic; Bedrock for materials reasoning; Atlas Vector Search for structure similarity

**Demo Script:** (1) "Find transparent conductor with band gap >3eV" → (2) agent picks property→structure mode → (3) results returned → (4) follow-up: "what are the properties of ITO?" → (5) agent switches to structure→property → (6) properties heatmap

**Stretch Goal:** Synthesis route prediction: given target structure, retrieve synthesis protocols

---

## #87 — ClimateFusion
**Domain:** Climate | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** NOAA, ESA, and ground sensors disagree — this system knows when to trust each and shows its reasoning.

**Concept:**
- Agent fuses time-series + text claims + sensor data with per-source uncertainty
- ReasoningBank stores "per-region, per-season: which source wins"
- Confidence ribbon per source; provenance shown for every number

**Paper Anchor:** [Search-R1 (arXiv:2503.09516)](https://arxiv.org/abs/2503.09516) + uncertainty quantification

**MongoDB + AWS Sketch:** Multi-source time-series; source reliability weights; Bedrock for synthesis reasoning; Atlas Vector Search for similar historical events

**Demo Script:** (1) Query heatwave attribution → (2) NOAA says +2.1°C; ESA says +1.9°C; sensor says +2.4°C → (3) agent weights by uncertainty → (4) confidence ribbon per source shown → (5) ReasoningBank: "Great Plains summer: ground sensor outperforms satellite by 0.3°C MAE"

**Stretch Goal:** Real-time source calibration: update weights as new sensor readings arrive

---

## #88 — ManuscriptVision
**Domain:** Cultural Heritage | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Medieval manuscript retrieval needs handwriting recognition, iconographic matching, and multilingual text — all at once.

**Concept:**
- ColPali variants adapted for handwritten text + iconographic motifs
- Agent picks index per script class (Carolingian, Gothic, Arabic, etc.)
- MongoDB per-fragment embeddings + iconographic graph with `$graphLookup`

**Paper Anchor:** [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) adapted for handwriting

**MongoDB + AWS Sketch:** Fragment embeddings (handwriting-adapted ColPali); iconographic graph; script-class classifier; Bedrock for paleographic reasoning; GridFS for high-res page images

**Demo Script:** (1) Query an iconographic motif (e.g., angel holding scales) → (2) script classifier picks index → (3) 5 matching folios retrieved across manuscripts → (4) motif regions highlighted → (5) iconographic lineage shown via graph

**Stretch Goal:** Cross-manuscript watermark detection for provenance attribution

---

## #89 — GenomeNav
**Domain:** Healthcare Research | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** A human variant query should find the mouse ortholog literature in 2 hops — not require a bioinformatician to write the query.

**Concept:**
- Agent resolves IDs adaptively (gene symbol → UniProt → Ensembl → GO)
- HippoRAG PPR for ortholog hops: human → ortholog → phenotype → pathway
- First open multi-DB ortholog-aware retrieval with adaptive ID resolution

**Paper Anchor:** [HippoRAG 2 (arXiv:2502.14802)](https://arxiv.org/abs/2502.14802)

**MongoDB + AWS Sketch:** UMLS/MeSH graph with ortholog edges; multi-DB embedding collection; ID resolution cache; Bedrock for biological reasoning

**Demo Script:** (1) Query human BRCA2 variant → (2) agent resolves to UniProt ID → (3) PPR hops to mouse Brca2 orthologs → (4) mouse phenotype literature retrieved → (5) cross-species evidence trail shown

**Stretch Goal:** Confidence propagation across ortholog hops: how much evidence transfers from mouse to human?

---

## #90 — EdgarLinguist
**Domain:** Finance | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** "Operating income" means different things in Toyota's Japanese filings vs. BMW's German ones vs. Ford's EDGAR — this system knows.

**Concept:**
- Multilingual hybrid retrieval with XBRL-aware chunking
- Agent rewrites queries per regulatory regime (XBRL tags vs. free text vs. localized tables)
- Voyage multilingual-3 for cross-language embedding; side-by-side localized extracts

**Paper Anchor:** [BRIGHT (arXiv:2407.12883)](https://arxiv.org/abs/2407.12883) + XBRL standards

**MongoDB + AWS Sketch:** XBRL-aware document chunks; multilingual embeddings; Voyage multilingual-3; Bedrock for financial reasoning; Atlas hybrid search per jurisdiction

**Demo Script:** (1) Query revenue metric in English → (2) Toyota and BMW filings retrieved → (3) side-by-side localized tables shown → (4) XBRL tag differences highlighted → (5) unified number after currency and accounting normalization

**Stretch Goal:** Accounting standard reconciliation: IFRS vs. US GAAP treatment of same item

---

## #91 — PulseTrend
**Domain:** Journalism | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Sarcasm and meme-recontextualization make standard semantic search return the wrong content — this system catches it.

**Concept:**
- Sarcasm-tuned reranking with Voyage instruction following
- Temporal decay weighting: a 2022 meme resurging in 2025 gets a "resurrection score"
- Agent learns per-platform retrieval style (TikTok vs. Reddit vs. Telegram)

**Paper Anchor:** Temporal decay + sarcasm detection + [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) for meme frames

**MongoDB + AWS Sketch:** Meme frame embeddings (ColPali); temporal metadata with resurrection scoring; platform-specific reranking weights; Atlas time-series for virality

**Demo Script:** (1) Query a meme topic → (2) temporal pulse visualization shows current + historical spikes → (3) resurrection score highlights a 2022 meme that came back → (4) sarcasm flag catches ironic use of serious imagery → (5) platform-specific retrieval note

**Stretch Goal:** Cross-platform propagation speed: how fast does a meme travel from Telegram to TikTok?

---

## #92 — AsylumLens
**Domain:** Humanitarian | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Asylum seekers deserve access to country-of-origin information in their own language — not just the languages powerful governments speak.

**Concept:**
- Voyage multilingual-3 bridges low-resource Pashto/Tigrinya/Dari queries to English corpus
- Small-model rewriters translate query intent without losing legal nuance
- MongoDB per-jurisdiction precedent; voice input supported via LiveKit

**Paper Anchor:** Voyage multilingual-3 + UNHCR COI documentation standards

**MongoDB + AWS Sketch:** Multilingual COI corpus; per-jurisdiction precedent collection; LiveKit voice input; Bedrock multilingual for reasoning; Atlas Vector Search with language metadata

**Demo Script:** (1) Voice query in Tigrinya → (2) transcribed + query rewritten → (3) UK Home Office COI report extracts retrieved → (4) translated back to Tigrinya → (5) live voice-to-citations chain shown

**Stretch Goal:** Confidence scoring: how recent is the COI evidence? (2024 report vs. 2019 report)

---

## #93 — SoundFile
**Domain:** Creative Tools | **Difficulty:** ⭐⭐ | **Time Budget:** 24h
**Hook:** Hum a melody and find the sample — with a copyright-risk rating attached.

**Concept:**
- CLAP/OmniEmbed audio embeddings stored in MongoDB
- Agent picks between melody / timbre / lyrical retrieval modes per query
- Copyright-risk traffic light per result (Creative Commons / licensed / all-rights-reserved)

**Paper Anchor:** CLAP contrastive audio-language pretraining

**MongoDB + AWS Sketch:** CLAP audio embeddings; copyright metadata; Bedrock for query intent classification; Atlas Vector Search for audio similarity

**Demo Script:** (1) Hum a melody live → (2) agent classifies as melody-mode query → (3) 3 candidate tracks returned → (4) copyright-risk traffic light shown → (5) one track flagged as high-risk

**Stretch Goal:** Stem separation: isolate the specific element (bass, drums, melody) that matched

---

## #94 — CADLens
**Domain:** Manufacturing | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Engineers waste hours searching for existing CAD components — because no search system understands 3D geometry.

**Concept:**
- PointNet geometric embeddings + semantic tags + manufacturing constraint index
- Agent picks retrieval mode per query (function → semantic; shape → geometric; manufacturability → constraint)
- First open multimodal CAD retrieval; 3D preview with constraints overlay

**Paper Anchor:** PointNet geometric embeddings + [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) for drawing pages

**MongoDB + AWS Sketch:** 3D mesh embeddings (PointNet); assembly graph; manufacturing constraint collection; Atlas Vector Search; Bedrock for engineering reasoning

**Demo Script:** (1) Sketch a bracket → (2) "Find similar, manufacturable with CNC" → (3) agent picks geometric + constraint mode → (4) 5 candidates returned → (5) 3D preview rotating with tolerance overlay

**Stretch Goal:** Design-for-assembly scoring: rank candidates by ease of assembly with existing components

---

## #95 — EvidenceTimeline
**Domain:** Legal / Journalism | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** "The exchange happened at 2:14 PM" — retrieving the exact 2-second video segment with timestamp precision.

**Concept:**
- Scene + audio + transcript joint embeddings (OmniEmbed-Nemotron class)
- MongoDB per-frame + per-scene embeddings; temporal localization to the second
- Scrubber jumps directly to the localized segment; multi-witness cross-corroboration

**Paper Anchor:** OmniEmbed-Nemotron + MultiVENT 2.0 benchmark

**MongoDB + AWS Sketch:** Frame embeddings (ColPali); audio embeddings (CLAP); transcript embeddings; temporal index; Bedrock for cross-modal reasoning; S3 for video storage

**Demo Script:** (1) Query "person handing envelope at courthouse entrance" → (2) 4 videos searched → (3) precise 2-second clip returned with timestamp → (4) scrubber jumps to segment → (5) cross-corroboration: second camera confirms

**Stretch Goal:** Chain-of-custody documentation: cryptographic hash at time of retrieval for legal admissibility

---

## #96 — RadioAtlas
**Domain:** Healthcare Research | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Radiology cohort research requires matching DICOM images, radiology reports, and EHR data — no existing tool does all three.

**Concept:**
- Tri-modal retrieval: DICOM (ColPali) + radiology report (text) + FHIR (structured)
- De-identification baked into the indexing pipeline (Bedrock PII detection)
- Cross-modal evidence row per case: image, report, and EHR fields aligned

**Paper Anchor:** [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) + FHIR standard

**MongoDB + AWS Sketch:** DICOM page embeddings (ColPali); FHIR-shaped documents; report text embeddings; de-identification pipeline; Bedrock for radiology reasoning

**Demo Script:** (1) Query lesion morphology for cohort building → (2) tri-modal search fires → (3) 5 cases returned with image + report + EHR alignment → (4) de-identification verified → (5) cohort statistics auto-generated

**Stretch Goal:** Cohort drift detection: show when a new patient changes cohort composition

---

## #97 — P&IDLens
**Domain:** Manufacturing | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** "All relief valves downstream of pump P-103" — a query no text search engine can answer but every process engineer needs.

**Concept:**
- ColPali on P&ID pages + symbol-graph index capturing instrument relationships
- Agent reasons about symbol topology: "downstream of" requires graph traversal, not semantic similarity
- Symbol-graph-aware reranking: penalize results that are upstream of the target

**Paper Anchor:** [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) + process engineering graph schemas

**MongoDB + AWS Sketch:** P&ID page embeddings (ColPali); instrument relationship graph; symbol taxonomy; Bedrock for topology reasoning; Atlas `$graphLookup` for downstream traversal

**Demo Script:** (1) Query "all relief valves downstream of P-103" → (2) ColPali finds relevant pages → (3) graph traversal identifies downstream instruments → (4) schematic with auto-highlighted nodes shown → (5) 4 relief valves found; 1 upstream excluded

**Stretch Goal:** Design change impact: "if P-103 changes rating, which downstream instruments need re-certification?"

---

## #98 — SportsScout
**Domain:** Media | **Difficulty:** ⭐⭐ | **Time Budget:** 24h
**Hook:** "Show me every clutch-moment turnover from this player's rookie season" — a query that requires player ReID across thousands of hours of footage.

**Concept:**
- Player ReID embeddings evolve over seasons as player appearance changes
- Cross-league retrieval with consistent player identity across teams
- Action localization + commentary embeddings for semantic moment search

**Paper Anchor:** Sports ReID literature + CLAP audio for commentary

**MongoDB + AWS Sketch:** Player ReID embedding history (temporal); action classification embeddings; commentary CLAP embeddings; Atlas Vector Search; Bedrock for sports reasoning

**Demo Script:** (1) Query "rookie season turnovers in clutch situations" → (2) player identified across footage → (3) compilation returned with timestamps → (4) cross-team consistency shown → (5) career trajectory visualization

**Stretch Goal:** Opposition scouting: "how does this player perform against left-handed defenders?"

---

## #99 — PhishLens
**Domain:** Cybersecurity | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Phishing kits live in screenshots, code, and dark-web chatter simultaneously — this system searches all three.

**Concept:**
- Adaptive retrieval picks index by query mode: screenshot → ColPali; code → AST embeddings; chat → text
- DPO pairs from analyst feedback persist and fine-tune the reranker continuously
- "Kit family" cluster visualization: all variants of a phishing kit shown together

**Paper Anchor:** [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) + [Search-R1 (arXiv:2503.09516)](https://arxiv.org/abs/2503.09516)

**MongoDB + AWS Sketch:** Phishing screenshot embeddings (ColPali); code AST embeddings; dark-web text embeddings; DPO pairs; Voyage rerank-2.5; Bedrock for threat analysis

**Demo Script:** (1) Drop phishing screenshot → (2) agent picks ColPali mode → (3) 4 related dark-web posts retrieved → (4) live kit family cluster → (5) analyst thumbs-up → (6) reranker DPO pair mined → (7) nDCG improvement shown

**Stretch Goal:** Kit aging analysis: how old is this phishing kit? Has it been updated recently?

---

## #100 — AdaptiveAtlas
**Domain:** Open Source | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** The first retrieval system that tells you its own BRIGHT score — and improves it while you watch.

**Concept:**
- Embedded live BRIGHT benchmark harness: samples queries, computes nDCG@10, mines hard negatives
- DPO pairs from failures → Voyage AI rerank-2.5 fine-tuned continuously in MongoDB Atlas
- nDCG@10 climbing live as the audience watches — the demo IS the proof

**Paper Anchor:** [BRIGHT (arXiv:2407.12883)](https://arxiv.org/abs/2407.12883) + [Search-R1 (arXiv:2503.09516)](https://arxiv.org/abs/2503.09516) + Rank-DistiLLM (arXiv:2405.07920)

**MongoDB + AWS Sketch:** Document corpus; DPO training pairs collection; benchmark query set; Voyage rerank-2.5 fine-tuning via API; Atlas Vector Search; Bedrock for reasoning

**Demo Script:** (1) Drop fresh corpus → (2) system bootstraps benchmark with 50 auto-generated queries → (3) baseline nDCG@10: 0.41 → (4) hard negatives mined → (5) reranker fine-tuned → (6) nDCG@10: 0.67 — climbing live as audience watches → (7) "It just got smarter from that one interaction."

**Stretch Goal:** Corpus-agnostic bootstrapping: works on any domain without domain-specific fine-tuning seed data

---

*[← Theme 3 README](README.md) | [🏠 10_Hackathons](../README.md) | [Deep Dives →](../deep_dive_ideas/README.md)*
