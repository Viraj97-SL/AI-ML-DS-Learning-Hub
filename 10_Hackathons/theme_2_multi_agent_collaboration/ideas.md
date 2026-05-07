# 🤝 Theme 2: All 34 Multi-Agent Collaboration Ideas

> Each idea deploys multiple specialized agents coordinating via A2A, shared MongoDB blackboard, or EVINCE debate.

---

## Quick-Pick Guide

| Goal | Best Ideas |
|------|-----------|
| Healthcare research domain | #34, #54, #61, #72 |
| Finance / supply chain | #35, #53, #57, #59 |
| Social impact / humanitarian | #45, #49, #51, #56 |
| Solo, 24-hour sprint | #49, #63, #74 |
| Maximum "wow" demo | Deep Dives: Viral Autopsy · Portfall · Tipping Oracle |
| Developer tools / open source | #52, #70, #71 |

---

## #34 — TumorBoardSim
**Domain:** Healthcare Research | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** First open multi-agent system that replicates a virtual molecular tumor board — radiology, pathology, genomics, and oncology agents debating to consensus.

**Concept:**
- Specialist agents (rad, path, genomics, oncology) each own a MongoDB collection
- Exchange findings via Change Stream pub/sub; debate via EVINCE entropy-governed protocol
- Consensus artifact stored with confidence weights and full provenance

**Paper Anchor:** RadCouncil + MAM (ACL Findings 2025) + EVINCE debate

**MongoDB + AWS Sketch:** Per-specialist collections; EVINCE consensus collection; Bedrock for each specialist's reasoning; EventBridge for debate round triggers

**Demo Script:** (1) Drop synthetic case → (2) agents each post findings → (3) EVINCE debate fires on staging disagreement → (4) agents converge → (5) confidence-weighted consensus

**Stretch Goal:** Confidence-weighted disagreement visualization showing which specialist drove the final decision

---

## #35 — Bullwhip
**Domain:** Supply Chain | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Implements Jannelli IJPR 2025's 3-tier escalating consensus — the first open-source implementation of this pattern.

**Concept:**
- Buyer/supplier agents negotiate restocking with private cost information
- A2A-style protocol over MongoDB blackboard; game-theoretic deviation detection
- Toggle a tier-3 supplier into "deceptive" mode and watch bullwhip amplification

**Paper Anchor:** Jannelli IJPR 2025 escalating consensus tiers + game theory for LLM agents (arXiv:2512.07462)

**MongoDB + AWS Sketch:** Order ledger blackboard; private cost collections per agent; Bedrock for negotiation reasoning; EventBridge for demand signal propagation

**Demo Script:** (1) Run 3-tier chain normally → (2) toggle tier-3 to deceptive → (3) show bullwhip-coefficient meter spiking → (4) Jannelli escalation kicks in, dampens oscillation

**Stretch Goal:** Real-time bullwhip coefficient vs. baseline comparison

---

## #36 — EndPoint
**Domain:** Drug Discovery | **Difficulty:** ⭐⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** End-to-end drug discovery pipeline — target to trial design — in a single multi-agent system.

**Concept:**
- 6 specialist agents: target-ID, lead-opt, ADMET, docking, synthesis, trial-design
- Coordinate on MongoDB blackboard via A2A; failure-distillation agent learns from synthesis flops
- Fuses DrugAgent open-source code with synthesis-route + trial-design agents

**Paper Anchor:** DrugAgent + Prompt-to-Pill + A2A Protocol

**MongoDB + AWS Sketch:** Molecule graph collection; synthesis route graph; ADMET prediction cache; Bedrock + SageMaker for molecular reasoning; S3 for docking output storage

**Demo Script:** (1) Pick a disease target → (2) 6 agents run in parallel → (3) ADMET flags toxicity → (4) lead-opt compensates with structural modification → (5) candidate molecule proposed in 2 minutes

**Stretch Goal:** Live 3D molecular structure rotating as agents add functional groups

---

## #37 — NewsTrust
**Domain:** Journalism | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Open-source Bellingcat-style investigative stack — source-verifier, FOIA-tracker, image-forensics, and writer agents working together.

**Concept:**
- Self-hosted on AWS; each agent has a private MongoDB collection
- Vector search peer-discovery: "who in this swarm can verify this type of claim?"
- Confidence ladder rises as agents independently corroborate evidence

**Paper Anchor:** IOHunter (Minici et al. 2025) + MAD-Sherlock (arXiv:2410.20140)

**MongoDB + AWS Sketch:** Evidence collection per agent; vector search capability registry; Bedrock for reasoning; S3 for evidence artifacts

**Demo Script:** (1) Drop a synthetic deepfake claim → (2) agents triangulate primary source + EXIF → (3) confidence ladder rises as each agent adds corroboration → (4) final verdict with evidence chain

**Stretch Goal:** Cross-claim entity resolution: two claims share a source, agent surfaces the connection

---

## #38 — RedBlueLoop
**Domain:** Cybersecurity | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Continuous red/blue team drill — both sides' ReasoningBanks grow from each encounter.

**Concept:**
- Blue (CyberSleuth) and red (HexStrike-style) agents share a CTI knowledge graph
- A2A negotiation under Meta's "Agents Rule of Two" guardrails
- ReasoningBank grows for both sides — the red agent gets smarter and so does the blue

**Paper Anchor:** A2A Protocol + ReasoningBank + CTI threat intelligence patterns

**MongoDB + AWS Sketch:** Shared CTI graph; separate private ReasoningBanks per side; Bedrock for attack/defense reasoning; MITRE ATT&CK as background knowledge graph

**Demo Script:** (1) 90-second drill → (2) red finds a vuln → (3) blue patches → (4) MTTR meter shown → (5) replay reveals which prompt-injection trick blue caught the *second* time (from ReasoningBank)

**Stretch Goal:** Red agent attempting to learn blue's detection patterns and evade them

---

## #39 — PolicyLab
**Domain:** Government / Policy | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** CBO-style multi-agent simulation of proposed legislation — economic, epidemiological, and legal agents debate with explicit utility functions.

**Concept:**
- Specialist agents debate with declared utility functions (not hidden objectives)
- MongoDB blackboard logs disagreements with evidence citations
- Constitutional-AI judge (DIKE-ERIS pair) adjudicates contested claims
- Sankey diagram of "which evidence influenced which conclusion"

**Paper Anchor:** Edward Chang DIKE-ERIS constitutional AI framework + EVINCE debate

**MongoDB + AWS Sketch:** Policy debate blackboard; utility function declarations; Bedrock for domain reasoning; Lambda for debate round orchestration

**Demo Script:** (1) Drop a sugar-tax bill → (2) epi agent projects health effects → (3) economic agent contests → (4) DIKE adjudicates → (5) Sankey of evidence influence appears

**Stretch Goal:** "What if" slider: change one assumption and watch all projections re-derive

---

## #40 — CivicSwarm
**Domain:** Urban Planning | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Traffic + power + emergency agents coordinate during a heatwave — no cross-agency coordination tool does this today.

**Concept:**
- A2A protocol across mock city agencies (traffic, utilities, emergency services)
- MongoDB geospatial + Change Streams as the city-wide substrate
- Heatwave event triggers cascading reallocations across all agencies simultaneously

**Paper Anchor:** arXiv:2502.16131 (multi-agent urban coordination)

**MongoDB + AWS Sketch:** Geospatial city resources collection; Change Streams for threshold alerts; Bedrock for resource allocation reasoning; Atlas Vector Search for nearest-facility queries

**Demo Script:** (1) Trigger a heatwave → (2) power demand exceeds threshold → (3) traffic agent reroutes buses to cooling centers → (4) emergency agent pre-positions resources → (5) live city map heat-pulse

**Stretch Goal:** Counterfactual: run same heatwave without agent coordination and show the difference

---

## #41 — MOFForge
**Domain:** Materials Science | **Difficulty:** ⭐⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** First open multi-agent MOF discovery system with a failure-distillation agent that learns from synthesis flops (Inizan et al. 2025, arXiv:2504.14110).

**Concept:**
- Planner + literature + DFT + lab-robot + failure-distillation agents
- MongoDB skill library and shared vector search index of MOF structures
- Failure-distillation agent is the novel contribution: no existing system learns from synthesis failures

**Paper Anchor:** [Inizan et al. 2025 (arXiv:2504.14110)](https://arxiv.org/abs/2504.14110)

**MongoDB + AWS Sketch:** MOF structure embeddings; synthesis attempt log; failure pattern ReasoningBank; Bedrock + SageMaker for DFT approximation; S3 for structure files

**Demo Script:** (1) Pick gas-separation target → (2) agents propose 5 candidates → (3) 2 fail synthesis → (4) failure-distillation agent updates strategy → (5) 3D MOF cells morph as binding energy reported

**Stretch Goal:** Pareto front of MOF candidates by synthesis yield × separation efficiency

---

## #42 — CourtBench
**Domain:** Legal | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Full courtroom simulation with a citation-verifier MCP server that kills hallucinated case law in real time.

**Concept:**
- Lawyer + prosecutor + judge + clerk agents debate; MongoDB temporal case-file graph
- Citation-verifier MCP server checks every cited case against a real case law database
- HEC Paris study found 486 AI-hallucinated citations in court filings — this catches them live

**Paper Anchor:** AgentCourt + J1-EVAL + HEC Paris fake citation study + MCP

**MongoDB + AWS Sketch:** Case-file temporal graph; citation verification cache; Bedrock for legal reasoning; MCP server for case law lookup

**Demo Script:** (1) 90-second motion-to-dismiss argument → (2) lawyer cites a case → (3) verifier flags it as hallucinated live → (4) judge ruling probability meter updates → (5) corrected argument submitted

**Stretch Goal:** Jurisdiction-specific citation rules (federal vs. state vs. UK law)

---

## #43 — ReviewBoard
**Domain:** Scientific Publishing | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Federated peer review across publishers — reviewer agents travel with papers without leaking IP.

**Concept:**
- A2A across publisher domains; MongoDB RBAC so Nature agents can't read Elsevier private state
- Reviewer agents carry only their expertise profile, not raw institutional data
- Auditable review trail with redactions; first federated peer review with privacy by construction

**Paper Anchor:** A2A Protocol + MongoDB RBAC patterns

**MongoDB + AWS Sketch:** Per-publisher private collections with RBAC; shared review outcome collection; Bedrock for reviewer reasoning; A2A message bus on EventBridge

**Demo Script:** (1) Submit paper to 2 journal agents → (2) each reviews with private context → (3) compatible reviews rendered → (4) auditable trail with redactions shown → (5) cross-journal consensus on major revisions

**Stretch Goal:** Reviewer reputation scoring based on past review accuracy

---

## #44 — CarbonNetwork
**Domain:** Climate | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** 83% of firms can't access supplier emissions — this A2A network makes Scope 3 crawlable.

**Concept:**
- Each supplier hosts an MCP server returning verified Scope 3 with confidence scores
- Buyer agents aggregate via A2A; MongoDB `$graphLookup` for allocation graph
- Confidence-weighted carbon ledger: high-confidence suppliers weight more

**Paper Anchor:** A2A Protocol + MCP + Scope 3.1 standard

**MongoDB + AWS Sketch:** Carbon allocation graph; supplier confidence scores; `$graphLookup` for supply chain traversal; Bedrock for emission factor reasoning

**Demo Script:** (1) 5 simulated suppliers come online → (2) MCP servers return emissions with confidence → (3) buyer rolls up footprint → (4) confidence-weighted total shown with CI bands

**Stretch Goal:** Scope 3 hot-spot identification: which supplier contributes most uncertainty

---

## #45 — DisasterCluster
**Domain:** Humanitarian | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Operationalizes UN OCHA cluster system as an A2A protocol — logistics, health, shelter, and WASH agents coordinate without mandate overlap.

**Concept:**
- Cross-org A2A under explicit conflict-of-mandate negotiation (each cluster declares scope)
- MongoDB shared situation-awareness blackboard
- Resource allocation without duplication: each resource tagged with which cluster "owns" it

**Paper Anchor:** A2A Protocol + UN OCHA cluster coordination standards

**MongoDB + AWS Sketch:** Shared situation blackboard; mandate declaration collection; Bedrock for coordination reasoning; geospatial for resource positioning

**Demo Script:** (1) Trigger a flood event → (2) 4 cluster agents come online → (3) mandate negotiation → (4) resources allocated without overlap → (5) map showing each cluster's responsibility surface

**Stretch Goal:** Overlap detection: flag when two clusters are about to respond to the same location

---

## #46 — EVoteSim
**Domain:** Government / Journalism | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Election misinformation rapid response in hours, not days — before the claim reaches 47M views.

**Concept:**
- Claim-verifier + image-forensics + counter-narrative agents
- Sparse-topology debate (arXiv:2406.11776) for token efficiency; MongoDB claim graph
- Bi-temporal validity: evidence that was true in 2020 but superseded in 2024 is flagged

**Paper Anchor:** [arXiv:2406.11776](https://arxiv.org/abs/2406.11776) + Zep bi-temporal

**MongoDB + AWS Sketch:** Claim graph with bi-temporal validity; forensics results collection; Bedrock for debate; Lambda for claim ingestion

**Demo Script:** (1) Drop synthetic deepfake claim → (2) swarm produces labeled response in <90s → (3) time-to-rebuttal counter shown → (4) claim graph shows bi-temporal evidence state

**Stretch Goal:** Platform-specific counter-narrative formatting (Reel script vs. WhatsApp vs. Reddit)

---

## #47 — RANGuardian
**Domain:** Manufacturing / Telecom | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Open-sources the Deutsche Telekom RAN Guardian pattern for tier-2 carriers — at 1/10th the cost using small models.

**Concept:**
- Cell-tower + planning + ticketing agents; cost-aware specialization with Phi/Gemma for cheap steps
- MongoDB time-series + Change Streams for KPI monitoring
- ReasoningBank for incident classification across tower types

**Paper Anchor:** Deutsche Telekom RAN Guardian + small LLM specialization patterns

**MongoDB + AWS Sketch:** Cell KPI time-series; incident ReasoningBank; Bedrock (Claude Haiku) for cheap steps; Lambda for threshold-triggered agent invocation

**Demo Script:** (1) Inject sleeping-cell incident → (2) agents diagnose in 60s → (3) ticket created → (4) live KPI per cell sector shown → (5) show cost comparison vs. large model

**Stretch Goal:** Proactive anomaly prediction from KPI trend analysis

---

## #48 — AcademicShoals
**Domain:** Scientific Research | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Pre-publication audit agents that catch p-hacking before the paper goes live — not after retraction.

**Concept:**
- Data-cleaner + reanalyzer + statistician agents debate paper figures
- ReasoningBank of common p-hacking patterns (unusual df, post-hoc exclusions, interaction effects)
- MongoDB blackboard with full citation lineage; statistician flags errors during debate

**Paper Anchor:** MechEvalAgent (arXiv:2602.18458) + statistical analysis debate patterns

**MongoDB + AWS Sketch:** Audit debate blackboard; p-hacking pattern ReasoningBank; Bedrock for statistical reasoning; S3 for paper figure storage

**Demo Script:** (1) Drop synthetic paper with planted error → (2) statistician agent flags unusual df → (3) debate fires → (4) concerns ledger grows with provenance → (5) "Reproducibility risk: HIGH" issued

**Stretch Goal:** Run continuously on new arXiv submissions via RSS feed Change Stream

---

## #49 — CivicJustice
**Domain:** Legal / Social Impact | **Difficulty:** ⭐⭐ | **Time Budget:** 24h
**Hook:** Eviction defense disparate-representation gap — this agent fields cases overnight and escalates high-risk ones to human attorneys.

**Concept:**
- Tenant-side agent handles intake → motion drafting → court prep
- A2A handoff to human attorney for high-risk cases (clear escalation logic)
- Win-probability gauge updating per filed motion; MongoDB RBAC for case confidentiality

**Paper Anchor:** A2A Protocol + legal AI frameworks

**MongoDB + AWS Sketch:** Case file collection with RBAC; motion template library; Bedrock for legal reasoning; Lambda for court-deadline monitoring

**Demo Script:** (1) Walk a tenant scenario from intake → (2) agent drafts notice of appearance → (3) win-probability gauge updates → (4) high-risk flag triggers A2A handoff → (5) attorney briefed with case summary

**Stretch Goal:** Court-date calendar integration with automated reminder triggers

---

## #50 — SilkRoute
**Domain:** Logistics | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** HS code misclassification is the top customs cost (Avalara) — this swarm gets it right the first time.

**Concept:**
- Customs-agent per country + product-agent per shipment via A2A
- MongoDB bi-temporal tariff graph (tariff rates change weekly post-2025 trade actions)
- Live landed-cost comparison across 3 routes with provenance to gazette entries

**Paper Anchor:** A2A Protocol + bi-temporal tariff management

**MongoDB + AWS Sketch:** Bi-temporal tariff collection by jurisdiction; shipment classification collection; Bedrock for HS code reasoning; Change Streams for tariff update ingestion

**Demo Script:** (1) Ship synthetic product → (2) customs agents for 3 routes respond → (3) landed costs compared → (4) lowest-cost route identified → (5) live tariff diff between routes shown

**Stretch Goal:** Duty drawback opportunity detection: identify overpaid duties eligible for refund

---

## #51 — RefugeeVoice
**Domain:** Humanitarian | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Voice-first multilingual humanitarian intake in Pashto, Tigrinya, and Dari — languages most tools don't support.

**Concept:**
- LiveKit voice + small-model agents for low-resource language transcription
- Triage agent → specialist (legal / medical / family-reunification) via A2A
- MongoDB per-claimant case file with consent flags; on-device inference for privacy

**Paper Anchor:** A2A Protocol + multilingual small LLMs (Phi-4 / Gemma 2 multilingual)

**MongoDB + AWS Sketch:** Per-claimant case file; consent flags; LiveKit for voice; Bedrock (multilingual) for reasoning; Atlas Vector Search for precedent cases

**Demo Script:** (1) Spoken Tigrinya intake → (2) real-time transcription → (3) triage routes to legal specialist → (4) UNHCR form auto-populated → (5) live transcription + role-handoff visualization

**Stretch Goal:** Cross-claimant family-reunification graph: identify related cases automatically

---

## #52 — CodeForge
**Domain:** Developer Tools | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Open-source maintainer swarm at GitHub-org scale — triage + repro + bisect + reviewer + release-notes agents.

**Concept:**
- A2A across repos in a GitHub organization; shared skill library of issue reproductions
- MongoDB per-issue state + ReasoningBank for issue classification
- Org-wide maintainer pool: the right specialist agent picks up each issue type

**Paper Anchor:** A2A Protocol + Magentic-One orchestration

**MongoDB + AWS Sketch:** Issue state collection; repro skill library; ReasoningBank for issue classification; Bedrock for code reasoning; Lambda for GitHub webhook triggers

**Demo Script:** (1) File 5 issues across 3 repos → (2) swarm triages → (3) repro agent reproduces → (4) 3 PRs generated in 90s → (5) live PR-merge feed

**Stretch Goal:** Issue clustering: detect when 5 different reporters are hitting the same bug

---

## #53 — ProcureNet
**Domain:** Finance | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Enterprise procurement with first-proposal-bias mitigation — the buyer agent catches when vendor agents are exploiting anchoring effects.

**Concept:**
- Buyer + security + legal + finance agents negotiate with 3 vendor agents via A2A
- Magentic-Marketplace substrate with anti-bias guardrails
- Bias-flag count shown live: vendor #2 tried anchoring, buyer caught it

**Paper Anchor:** Magentic-Marketplace + game theory for LLM agents (arXiv:2512.07462)

**MongoDB + AWS Sketch:** Negotiation ledger; vendor proposal history; bias detection collection; Bedrock for negotiation reasoning; EventBridge for round management

**Demo Script:** (1) RFQ for SaaS → (2) 3 vendor agents bid → (3) buyer detects first-proposal-bias attempt from Vendor 2 → (4) bias-flag count shown → (5) fair negotiation completes

**Stretch Goal:** Historical vendor behavior profiling from past negotiations

---

## #54 — OpsDoctor
**Domain:** Healthcare Research | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Hospital ops research swarm — bed + OR + supply-chain agents — with strict "research only, no patient advice" framing.

**Concept:**
- Specialist research agents (queueing-theorist, scheduler, supply) on MongoDB blackboard
- MongoDB time-series for bed turnover; ReasoningBank for surge patterns
- Clearly scoped: operational throughput research, not clinical decisions

**Paper Anchor:** Magentic-One orchestration + hospital operations research

**MongoDB + AWS Sketch:** Bed occupancy time-series; OR schedule collection; supply forecasting; Bedrock for operational reasoning; Atlas Vector Search for historical surge patterns

**Demo Script:** (1) Replay a 12-hour ER shift → (2) inject bus accident → (3) bed/OR/supply agents reallocate → (4) counterfactual throughput delta shown

**Stretch Goal:** Simulation of staffing models: which shift pattern minimizes wait time?

---

## #55 — AdLaundry
**Domain:** Media | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Brand-safety verification at bid time — before the ad runs next to synthetic AI-generated misinformation.

**Concept:**
- Buyer + seller + brand-safety + measurement agents via A2A bidding arena
- Adversarial brand-safety verifier scrutinizes Veo3/Sora2 synthetic context at bid time
- MongoDB immutable bid log; each blocked bid has a reason code

**Paper Anchor:** Magentic-Marketplace + A2A Protocol + synthetic content detection

**MongoDB + AWS Sketch:** Immutable bid log; brand-safety evaluation cache; Bedrock for content analysis; Lambda for bid-stream processing

**Demo Script:** (1) Run 10 bids → (2) safety agent catches synthetic-context risk on bid 7 → (3) bid blocked with reason → (4) bid-stream visualization with live block decisions

**Stretch Goal:** Brand safety score that updates as new synthetic content patterns emerge

---

## #56 — WildSwarm
**Domain:** Conservation / Urban | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Wildlife corridor planning via A2A — ecologist, transport, community, and farmer agents negotiate with explicit utility functions.

**Concept:**
- Public-goods-game framing (arXiv:2512.07462): each agent has a declared utility function
- Pareto front of corridor plans with stakeholder satisfaction scores
- MongoDB geospatial as the common planning substrate

**Paper Anchor:** [arXiv:2512.07462](https://arxiv.org/abs/2512.07462) public goods game framing

**MongoDB + AWS Sketch:** Geospatial corridor options; stakeholder utility declarations; Bedrock for negotiation; Atlas geospatial queries for land-use compatibility

**Demo Script:** (1) Negotiate a 3-mile corridor in 90s → (2) farmer agent vetoes a route → (3) Pareto front updates → (4) final plan shown with stakeholder satisfaction scores

**Stretch Goal:** Corridor viability simulation: model animal movement through proposed corridor

---

## #57 — FederatedFraud
**Domain:** Finance | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Bank-to-bank AML coordination without exposing customer PII — privacy by construction.

**Concept:**
- EPEAgent-style minimal disclosure: agents share only hashed entity IDs and risk scores
- MongoDB RBAC + A2A protocol-level token scopes prevent raw data leakage
- Privacy-budget meter shown live: how many "bits" of information have been shared

**Paper Anchor:** A2A Protocol + EPEAgent privacy-preserving patterns + MongoDB RBAC

**MongoDB + AWS Sketch:** Per-bank private collections (strict RBAC); shared risk-score collection; Bedrock for AML reasoning; privacy-budget tracker

**Demo Script:** (1) Mock money-laundering ring across 2 banks → (2) agents coordinate → (3) ring flagged without exposing customer IDs → (4) privacy-budget meter shown → (5) audit trail demonstrates minimal disclosure

**Stretch Goal:** Cross-border AML coordination: add a third bank in a different jurisdiction

---

## #58 — SpacecraftCouncil
**Domain:** Aerospace | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Cross-operator ground-station market — the first open implementation of constellation contact-window negotiation.

**Concept:**
- Per-spacecraft agents bid for ground-station slots via A2A with reputation scores
- MongoDB scheduling collection; operators can't see each other's private mission data
- Sky map shows claim arrows as spacecraft negotiate

**Paper Anchor:** A2A Protocol + multi-agent auction mechanisms

**MongoDB + AWS Sketch:** Scheduling collection; spacecraft state (private per operator); reputation scores; Bedrock for bid strategy; Lambda for contact window triggers

**Demo Script:** (1) 3 satellites compete for 2 slots → (2) A2A negotiation shown → (3) reputation scores affect outcome → (4) sky map with claim arrows → (5) conflict resolution demonstrated

**Stretch Goal:** Emergency priority override: spacecraft declares anomaly and preempts all slots

---

## #59 — FundamentalsBoard
**Domain:** Finance | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Portfolio committee simulation — quant, fundamental, risk, compliance, and ESG agents debate with conflict-of-interest declarations.

**Concept:**
- Each agent reads private feeds; A2A commits decisions via Saga
- MongoDB bi-temporal portfolio graph; conflict-of-interest declarations required before voting
- Anti-correlated agent disagreement visualization: show when quant and fundamental agree vs. diverge

**Paper Anchor:** FinCon multi-agent framework + A2A + Zep bi-temporal

**MongoDB + AWS Sketch:** Private feed collections per agent; portfolio bi-temporal graph; Saga for position changes; Bedrock for financial reasoning

**Demo Script:** (1) "Should we buy X?" debate in 90s → (2) ESG flags concern → (3) quant overrides with momentum data → (4) compliance declares no-conflict → (5) anti-correlated disagreement visualization

**Stretch Goal:** Post-mortem: replay the debate 6 months later with actual outcomes

---

## #60 — CivicMix
**Domain:** Manufacturing | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Concrete is 8% of global CO₂ — this multi-agent optimizer cuts it without sacrificing structural integrity.

**Concept:**
- Mix-designer + supplier + structural-engineer agents negotiate per pour
- MongoDB pour-history time-series; live MPa / kgCO₂e tradeoff curve
- A2A negotiation results in a mix that meets strength requirements at minimum carbon

**Paper Anchor:** A2A Protocol + Caidio-style concrete optimization

**MongoDB + AWS Sketch:** Pour history time-series; mix recipe collection; structural requirements; Bedrock for materials reasoning; Atlas Vector Search for similar pour history

**Demo Script:** (1) Specify slab pour → (2) mix-designer proposes → (3) structural-engineer approves (or rejects) → (4) live MPa / kgCO₂e curve → (5) final mix at Pareto frontier

**Stretch Goal:** Cumulative CO₂ savings counter across a construction project

---

## #61 — RecallSwarm
**Domain:** Healthcare Regulatory | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Cross-stakeholder recall coordination — manufacturer, FDA, and provider agents negotiate scope via A2A.

**Concept:**
- Constitutional adjudicator resolves disagreements about recall scope and urgency
- Bi-temporal evidence ledger: when did each party know what?
- Patients-protected counter as the human-stakes metric

**Paper Anchor:** A2A Protocol + bi-temporal evidence + constitutional AI adjudication

**MongoDB + AWS Sketch:** Bi-temporal evidence ledger; recall scope collection; Bedrock for regulatory reasoning; Lambda for escalation triggers

**Demo Script:** (1) Trigger Class II recall → (2) 3 org agents come online → (3) scope disagreement → (4) adjudicator rules → (5) patients-protected counter shown

**Stretch Goal:** Cascade analysis: which downstream products contain the recalled component?

---

## #62 — NewsroomQuorum
**Domain:** Journalism | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Cross-newsroom claim verification — Reuters, AP, and Bellingcat-style agents reach quorum without leaking competitive intelligence.

**Concept:**
- Federated A2A claim-verifier; privacy slices in MongoDB prevent raw data leakage
- Attribution preserved: each newsroom gets credit for its contribution
- "Quorum reached" stamp appears when 2+ newsrooms independently verify

**Paper Anchor:** A2A Protocol + federated evidence frameworks

**MongoDB + AWS Sketch:** Per-newsroom private evidence collections; shared quorum collection; Bedrock for verification reasoning; privacy slice enforcement via RBAC

**Demo Script:** (1) Drop viral claim → (2) 3 newsroom agents investigate → (3) 2 converge on same verdict → (4) "Quorum Reached" stamp appears → (5) attribution preserved in final output

**Stretch Goal:** Disagreement surfacing: when newsrooms diverge, show the specific evidence gap

---

## #63 — CropParliament
**Domain:** Agriculture | **Difficulty:** ⭐⭐ | **Time Budget:** 24h
**Hook:** Voice-first agricultural extension in Swahili, Hausa, and Tagalog — most farmers don't use apps.

**Concept:**
- Voice triage agent → pest specialist / agronomy / market-price agents via A2A
- MongoDB per-farmer case file with multi-season history
- Operationalizes Hiywotu 2025's trust gap solutions: farmers trust voice more than text

**Paper Anchor:** Hiywotu 2025 AI trust in agriculture + small multilingual LLMs

**MongoDB + AWS Sketch:** Per-farmer case file; multi-season planting history; LiveKit voice; Bedrock multilingual; Atlas Vector Search for similar pest cases

**Demo Script:** (1) Farmer voice question in Swahili → (2) triage routes → (3) pest specialist responds with citations → (4) live audio + cross-agent reasoning visualization

**Stretch Goal:** Community learning: when 10 farmers report the same pest, issue a region-wide alert

---

## #64 — FermiAgent
**Domain:** Scientific Research | **Difficulty:** ⭐⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Lean theorem prover + Mathlib librarian + tactic-selector agents — the first open multi-agent formal math system.

**Concept:**
- A2A around a Lean MCP server; tactic agents specialize by proof technique
- MongoDB skill library with per-tactic success counters updated after each proof attempt
- Lean kernel green-checks at the end of the demo — provably correct

**Paper Anchor:** Numina-Lean-Agent + A2A Protocol + MCP for Lean integration

**MongoDB + AWS Sketch:** Tactic skill library; proof attempt history; Bedrock for high-level reasoning; Lean MCP server; Lambda for proof compilation

**Demo Script:** (1) State an AMC-tier theorem → (2) tactic agents debate approach → (3) skill library consulted → (4) proof assembled → (5) Lean kernel confirms ✓

**Stretch Goal:** Tactic success rate visualization: which proof techniques work for which problem classes

---

## #65 — SubgridParliament
**Domain:** Climate / Energy | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Cross-ISO agents negotiate dynamic line ratings — unlocking hidden grid capacity without new transmission.

**Concept:**
- Per-utility agents share GET capacity offers via A2A; MongoDB time-series for live conductor data
- Cross-ISO market created purely through agent protocol — no regulatory change needed
- Live MW saved counter on a heat surge event

**Paper Anchor:** A2A Protocol + dynamic line rating standards + ISO market design

**MongoDB + AWS Sketch:** Conductor capacity time-series; cross-ISO offer collection; Bedrock for grid reasoning; Change Streams for demand-surge triggers

**Demo Script:** (1) Inject heat surge in Kansas → (2) demand exceeds static rating → (3) cross-ISO agents transact → (4) dynamic rating unlocks 340 MW → (5) live MW saved counter

**Stretch Goal:** Wildfire risk integration: reduce line ratings in fire-prone corridors automatically

---

## #66 — ChartedCourse
**Domain:** Logistics | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Vessel, port, insurer, and cargo-owner agents all in the loop during a shipping disruption — not just the vessel.

**Concept:**
- A2A negotiation across all stakeholders simultaneously
- MongoDB geospatial + maritime AIS time-series; insurer-in-the-loop is the novel contribution
- Vessel paths re-flow on a live map as each agent commits to a decision

**Paper Anchor:** A2A Protocol + Jannelli supply chain consensus + maritime operations

**MongoDB + AWS Sketch:** Vessel AIS time-series; port berth schedules; insurance policy collection; Bedrock for multi-party reasoning; geospatial for route optimization

**Demo Script:** (1) Close a shipping lane → (2) vessel, port, insurer, cargo-owner all respond → (3) insurer prices war-exclusion in real time → (4) vessels re-route → (5) live map animation

**Stretch Goal:** Multi-vessel convoy coordination during the reroute

---

## #67 — PlanetScope
**Domain:** Climate | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Different climate institutions see different pieces of the picture — this A2A network shares only uncertainty bounds, not raw data.

**Concept:**
- Atmosphere + ocean + ice + permafrost agents share uncertainty-quantified summaries only
- MongoDB reproducibility lineage with bi-temporal validity; cross-institution synthesis
- Lineage graph: which institution's data drove which conclusion

**Paper Anchor:** CMIP6 coupled modeling + A2A + Zep temporal KG + Ditlevsen & Ditlevsen Science 2023

**MongoDB + AWS Sketch:** Per-institution private data (RBAC); shared uncertainty summary collection; bi-temporal synthesis graph; Bedrock for Bayesian synthesis

**Demo Script:** (1) 4 institution agents go online → (2) each shares uncertainty bounds → (3) synthesis builds tipping-risk register → (4) lineage traversal shown → (5) AMOC risk: 0.68 with provenance

**Stretch Goal:** "What if we shared raw data?" counterfactual showing marginal uncertainty reduction

---

## #68 — CivicCheck
**Domain:** Civic Tech | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h solo
**Hook:** Every city council vote triggers a cascade of citizen, journalist, and policy-analyst agents that cross-check against campaign finance, prior votes, and public sentiment.

**Concept:**
- VoteWatcher agent indexes council minutes; FinanceTrace agent maps donations; SentimentPulse agent monitors social signals
- A2A handshake: VoteWatcher triggers downstream agents when a vote deviates from pledged positions
- MongoDB change streams fan out to all agents in < 100 ms

**Paper Anchor:** Magentic-One orchestration + A2A protocol + MongoDB Change Streams

**MongoDB + AWS Sketch:** Council minutes collection; campaign finance time-series; social sentiment rolling window; Bedrock for contradiction detection; EventBridge for vote-event fan-out

**Demo Script:** (1) New vote detected → (2) agents cross-check finance + sentiment + history → (3) contradiction score highlighted → (4) journalist agent drafts a paragraph → (5) audit trail shown

**Stretch Goal:** Predict how a councillor will vote given their finance profile + issue type

---

## #69 — AcademicMatch
**Domain:** Research Infrastructure | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Researcher skill profiles, open grant calls, and collaboration networks are siloed — this multi-agent network surfaces the best researcher–grant fit in real time.

**Concept:**
- ResearcherProfile agent maintains semantic skill vectors; GrantScan agent indexes NIH/Wellcome/ERC calls; CollabGraph agent maps co-author networks
- A2A: GrantScan notifies ResearcherProfile when a new call appears; CollabGraph suggests team compositions
- MongoDB Atlas hybrid search matches researcher skills against grant requirements

**Paper Anchor:** A2A protocol + MongoDB hybrid search ($rankFusion) + Voyage rerank-2.5

**MongoDB + AWS Sketch:** Researcher embeddings (Voyage multilingual-3); grant call collection; co-author graph (edges collection); Bedrock for fit-score narrative; Lambda for nightly grant scrape

**Demo Script:** (1) New NSF call indexed → (2) hybrid search finds matching researchers → (3) CollabGraph suggests team with diversity score → (4) Bedrock drafts specific-aim outline → (5) researcher confirms via A2A accept message

**Stretch Goal:** Simulate multi-institution team formation with IP constraint checking

---

## #70 — AgentMarket
**Domain:** AI Infrastructure | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 3 days
**Hook:** An open marketplace where specialized AI agents advertise capabilities and bid for sub-tasks — self-organizing labor market for AI work.

**Concept:**
- TaskAuctioneer agent decomposes work and publishes bids; SpecialistAgent registry advertises latency + cost + capability; MonitorAgent tracks completion and escrow
- A2A protocol carries capability cards and bid messages; MongoDB stores agent registry, reputation scores, and escrow ledger
- UCB1 bandit selects agent when multiple candidates have equal capability scores

**Paper Anchor:** A2A protocol + Magentic-One + MongoDB Change Streams + BanditRouter pattern

**MongoDB + AWS Sketch:** Agent registry collection; bid/escrow collection; task decomposition tree; Bedrock for task valuation; Lambda for bid timeout enforcement

**Demo Script:** (1) Complex task submitted → (2) auctioneer decomposes → (3) agents bid → (4) winning agents show real-time progress → (5) escrow released on verification → (6) reputation updated

**Stretch Goal:** Malicious agent detection via outcome vs. promised capability divergence

---

## #71 — RFC-Lab
**Domain:** Protocol Engineering | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 3 days
**Hook:** Simulate five decades of IETF debate in seconds — adversarial agents reproduce why TCP, TLS, and QUIC were designed the way they were.

**Concept:**
- Proposer, Skeptic, SecurityAudit, ImplementationReality, and Consensus agents debate each protocol design choice
- EVINCE convergence: debate halts when cosine distance < 0.05 across agent position vectors
- MongoDB stores full debate transcript as a citable knowledge graph

**Paper Anchor:** EVINCE multi-agent debate + SagaLLM compensation + A2A protocol + MongoDB graphLookup

**MongoDB + AWS Sketch:** RFC corpus collection; debate transcript collection with bi-temporal stamps; position-vector time-series; Bedrock for adversarial agent reasoning; Atlas Search for RFC cross-references

**Demo Script:** (1) Load TCP RFC 793 → (2) five agents debate sliding-window design → (3) EVINCE convergence shown with entropy curve → (4) debate replay with citation links → (5) generate a "what-if QUIC came first" counterfactual

**Stretch Goal:** Use agent debate output to auto-generate amendment proposals for an existing RFC

---

## #72 — ICUTriage
**Domain:** Critical Care | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 3–5 days
**Hook:** ICU beds are scarce — a multi-agent system that continuously re-scores patient acuity and negotiates bed assignments across hospital floors.

**Concept:**
- PatientMonitor agent reads live vitals stream; AcuityScorer agent runs SOFA/APACHE-II; BedAllocator agent manages cross-floor negotiation; EthicsAudit agent flags equity anomalies
- A2A handshake: PatientMonitor pushes deterioration alerts; BedAllocator pulls from AcuityScorer before any transfer
- Bedrock Guardrails enforce PHI stripping in all inter-agent messages

**Paper Anchor:** Magentic-One orchestration + A2A protocol + MongoDB time-series + Bedrock Guardrails

**MongoDB + AWS Sketch:** Patient vitals time-series; bed-state change stream; triage decision log; Bedrock Guardrails for PII; Lambda for vitals ingestion; Step Functions for transfer workflow

**Demo Script:** (1) Three patients deteriorate simultaneously → (2) AcuityScorer re-ranks in < 2s → (3) BedAllocator negotiates via A2A → (4) EthicsAudit flags demographic skew → (5) transfer chain shown with full audit

**Stretch Goal:** Simulate mass-casualty event: 20 simultaneous arrivals, live reallocation

---

## #73 — ExchangeSpeak
**Domain:** Finance / Market Microstructure | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Central bank signals, analyst notes, and social momentum are processed by specialized agents who debate a consensus price direction before any trade is placed.

**Concept:**
- MacroPulse agent monitors Fed/ECB statements; SellSide agent aggregates analyst upgrades/downgrades; MomentumAgent tracks order-book imbalance; RiskGate agent enforces drawdown limits
- A2A: each agent publishes a signed "view" message; SynthesizerAgent aggregates with EVINCE
- MongoDB stores signed view messages as an immutable audit chain

**Paper Anchor:** EVINCE convergence + A2A protocol + MongoDB Change Streams + Voyage rerank-2.5

**MongoDB + AWS Sketch:** Fed statement collection; analyst-note embeddings; order-book time-series; immutable view-message log; Bedrock for macro-signal interpretation; Atlas Stream Processing for tick data

**Demo Script:** (1) Surprise FOMC rate cut → (2) MacroPulse fires alert → (3) all agents update views → (4) EVINCE synthesis produces directional signal → (5) RiskGate approves/blocks with limit check

**Stretch Goal:** Run the same event through a 2008-calibrated vs. 2024-calibrated agent ensemble and compare divergence

---

## #74 — CityReview
**Domain:** Urban Planning | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** A new zoning proposal triggers five stakeholder agents — residents, developers, transit planners, environmentalists, and the city council — who negotiate in real time.

**Concept:**
- Each stakeholder agent has a persistent values vector (stored in MongoDB); negotiation uses EVINCE debate with weighted votes
- SagaLLM compensation: if a proposal passes then is reversed by legal review, previous commitments are rolled back
- Full negotiation history stored as a citable record with A2A message provenance

**Paper Anchor:** EVINCE + SagaLLM compensation + A2A protocol + MongoDB bi-temporal schema

**MongoDB + AWS Sketch:** Stakeholder values collection; proposal versions with bi-temporal validity; negotiation transcript; Bedrock for stakeholder reasoning; EventBridge for proposal lifecycle events

**Demo Script:** (1) New mixed-use zoning proposal submitted → (2) five agents debate → (3) EVINCE shows entropy decay → (4) transit planner raises a blocking concern → (5) SagaLLM rollback demonstrated → (6) revised proposal reaches consensus

**Stretch Goal:** Replay a real historical zoning dispute and compare agent consensus outcome to actual vote

---

## #75 — MarsControl
**Domain:** Space Operations | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 3–5 days
**Hook:** 14-minute light-speed delay to Mars means ground control cannot intervene in real time — a persistent multi-agent system on the surface manages its own contingencies.

**Concept:**
- PowerManager, LifeSupport, ScienceScheduler, and HazardResponse agents run autonomously in a LangGraph loop with MongoDB durable state
- A2A: agents negotiate resource allocation when solar power drops; SagaLLM handles mission-abort compensation (undo science experiments if power fails)
- Ground control sends intent updates that agents interpret asynchronously

**Paper Anchor:** ReasoningBank + SagaLLM + LangGraph MongoDB checkpointer + A2A protocol

**MongoDB + AWS Sketch:** Mission state collection (LangGraph checkpointer); resource-allocation log; hazard-event change stream; Bedrock for contingency reasoning; EventBridge for ground-control command relay

**Demo Script:** (1) Solar storm reduces power 40% → (2) agents negotiate autonomously → (3) science experiment paused via SagaLLM rollback → (4) ground command arrives 14 min later → (5) agents reconcile asynchronously → (6) full timeline shown

**Stretch Goal:** Multi-rover coordination: two rovers discover conflicting sample priority

---

## #76 — HeritageGuild
**Domain:** Cultural Heritage | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Digitized manuscripts, oral history audio, and archaeological databases are managed by specialist agents who negotiate attribution and cross-language provenance.

**Concept:**
- ManuscriptReader agent uses ColPali for handwritten page embeddings; OralHistory agent transcribes and embeds audio; ArchaeologySite agent maintains geospatial artifacts
- A2A: cross-agent citation when one agent's finding corroborates another's; collaborative lineage graph in MongoDB
- HippoRAG PPR traversal finds multi-hop connections (artifact → trade route → manuscript reference → oral legend)

**Paper Anchor:** ColPali + HippoRAG 2 PPR + A2A protocol + Voyage multilingual-3

**MongoDB + AWS Sketch:** Manuscript page embeddings (ColPali); audio transcript embeddings; geospatial artifact collection; provenance graph; Bedrock for cross-cultural interpretation; Voyage multilingual-3 for 15+ languages

**Demo Script:** (1) Upload a Latin manuscript page → (2) ColPali indexes without OCR → (3) HippoRAG PPR finds 3-hop connection to oral legend → (4) A2A citation links the two corpora → (5) provenance trail displayed

**Stretch Goal:** Auto-generate an academic citation graph from multi-corpus discovery

---

## #77 — AgriCoOp
**Domain:** Agriculture | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** A cooperative of small farmers uses multi-agent negotiation to optimize shared irrigation, logistics, and market timing without exposing individual farm data.

**Concept:**
- SoilAgent per farm holds private sensor data; MarketAgent monitors commodity futures; WaterAgent manages shared irrigation rights; LogisticsAgent optimizes collective transport
- A2A: agents share aggregate signals only (privacy-preserving negotiation); SagaLLM compensation if a cooperative deal falls through
- MongoDB RBAC ensures each farm's data is visible only to its own agent; shared collection holds anonymized aggregates

**Paper Anchor:** A2A protocol + SagaLLM + MongoDB RBAC + Atlas Stream Processing

**MongoDB + AWS Sketch:** Per-farm private collections (RBAC); shared anonymous aggregate; water-rights ledger; futures price time-series; Bedrock for optimal harvest/irrigation scheduling; Lambda for daily market data pull

**Demo Script:** (1) Drought forecast arrives → (2) WaterAgent initiates rationing negotiation → (3) A2A messages carry only anonymized field-stress scores → (4) MarketAgent advises early harvest for 3 farms → (5) LogisticsAgent builds shared truck route → (6) SagaLLM rollback if one farm backs out

**Stretch Goal:** Simulate a 50-farm cooperative across three crop types with conflicting harvest windows

---

## Navigation

| Previous | Home | Next |
|----------|------|------|
| [← Theme 2 README](README.md) | [🏠 10_Hackathons](../README.md) | [Theme 3 Ideas →](../theme_3_adaptive_retrieval/ideas.md) |

*[← Theme 2 README](README.md) | [🏠 10_Hackathons](../README.md) | [Theme 3 →](../theme_3_adaptive_retrieval/README.md)*