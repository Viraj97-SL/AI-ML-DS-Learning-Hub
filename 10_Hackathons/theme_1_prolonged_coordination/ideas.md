# 🧠 Theme 1: All 33 Prolonged Coordination Ideas

> Each idea is grounded in a 2024–2026 paper and buildable on MongoDB Atlas + AWS.  
> Format: Hook → Concept → Paper → Stack → Demo → Stretch.

---

## Quick-Pick Guide

| Goal | Best Ideas |
|------|-----------|
| Win with healthcare domain | #2, #3, #12, #17, #23 |
| Win with finance domain | #8, #24, #26, #27, #32 |
| Solo, 24-hour sprint | #10, #21, #29, #30 |
| Maximum "wow" demo | #5, #6, #33 — or the [Deep Dives](../deep_dive_ideas/) |
| First-time agentic build | #10, #21, #29 |

---

## #1 — ProsecuteIQ
**Domain:** Legal / IP | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** The first agent that can be *put down for 18 months* and resume mid-patent prosecution.

**Concept:**
- Tracks office actions, claim amendments, IDS updates, and continuation strategy across 4-year cycles
- Bi-temporal claim graph in MongoDB: `valid_from`/`valid_to` on every claim limitation
- SagaLLM compensation lets the agent roll back claim amendments after examiner rejections
- ReasoningBank entries learned from prior successful responses to §103 rejections

**Paper Anchor:** [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382) + [Zep (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)

**MongoDB + AWS Sketch:** Bi-temporal claim graph collection; LangGraph MongoDB checkpointer; Bedrock Claude for reasoning; EventBridge triggers weekly docket scan

**Demo Script:** (1) Restore checkpoint from "8 months ago" → (2) inject fresh §103 office action → (3) agent diffs claim graph, identifies affected limitations, drafts amended claims and rebuttal

**Stretch Goal:** Time-travel slider replaying entire prosecution with forked strategy branches

---

## #2 — NemoRecall
**Domain:** Healthcare / Regulatory | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** FDA was late on 73.9% of recall terminations (GAO-26-107619) — this agent closes that gap.

**Concept:**
- Durable agent watches FAERS, MAUDE, and supplier change streams; opens a case per device
- Walks §806/§7 recall workflow with checkpointed milestones in MongoDB
- SagaLLM compensation for retraction or expansion of recalls

**Paper Anchor:** [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382)

**MongoDB + AWS Sketch:** Change streams on FAERS feed; Saga ledger collection; Lambda-triggered agent steps; Bedrock for signal classification

**Demo Script:** (1) Fast-forward 60 simulated days of FAERS events → (2) crash the agent → (3) restart from checkpointer → (4) show agent still meets 24-hour Form 3500A clock

**Stretch Goal:** Live "what would have been late" counter showing FDA's miss rate vs. agent's

---

## #3 — SiteAct
**Domain:** Healthcare Research | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Trial site activation grew 7 months between 2020–2024 — one agent persists across the entire activation.

**Concept:**
- Per-site agent maintains state across IRB submission, contract negotiation, regulatory binder, SIV
- Protocol amendments trigger surgical replans, not full restarts (SagaLLM compensation)
- MongoDB tracks which sub-tasks are invalidated by an amendment and only redoes those

**Paper Anchor:** [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382)

**MongoDB + AWS Sketch:** Saga ledger per site; amendment diff service on Lambda; Bedrock for contract language analysis

**Demo Script:** (1) Agent mid-activation → (2) drop a protocol amendment PDF → (3) agent identifies 4 invalidated sub-tasks, compensates them, resumes remaining 9

**Stretch Goal:** Per-site Gantt chart rebuilding in real time with compensation arrows shown

---

## #4 — QueueClear
**Domain:** Climate / Energy | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** 1,400 GW are stuck in interconnection queues; only 13% of 2000–2019 applications ever built.

**Concept:**
- Long-horizon agent files cluster-study readiness packets, monitors PJM/CAISO/MISO portals via browser-use
- Replans network upgrades when queue positions shift or moratoriums are announced
- MongoDB time-series tracks queue-depth deltas; checkpointer resumes after multi-year pauses

**Paper Anchor:** [LangGraph Durable Execution](https://langchain-ai.github.io/langgraph/) + VIGIL supervisor

**MongoDB + AWS Sketch:** Time-series for queue depth; browser-use Lambda for portal scraping; Bedrock for filing language; EventBridge for weekly queue scan

**Demo Script:** (1) Compress 3 simulated years into 90 seconds → (2) inject a PJM Order 2023 rule change → (3) agent detects impact, replans affected network upgrades without restarting

**Stretch Goal:** Live MW counter of "capacity delivered to grid because of agent"

---

## #5 — CampaignChem
**Domain:** Materials Science | **Difficulty:** ⭐⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Self-driving labs crash on instrument drift — this agent detects calibration failures and resumes without human intervention.

**Concept:**
- Bayesian-optimization agent runs multi-week polymer-electrolyte campaigns
- MongoDB time-series captures every NMR/UPLC reading; reference-vector distance detects drift
- ReasoningBank distills "retry vs. abandon" decisions from prior failed batches

**Paper Anchor:** [ReasoningBank (arXiv:2504.09762)](https://arxiv.org/abs/2504.09762)

**MongoDB + AWS Sketch:** Time-series for sensor readings; Vector Search for reference-vector drift detection; Bedrock for Bayesian reasoning; SageMaker for surrogate model updates

**Demo Script:** (1) Running campaign → (2) inject calibration drift event → (3) agent detects via reference-vector distance, requests recalibration, resumes → (4) show ReasoningBank entry: "NMR drift detection → request recalibration, retry 2h later. Success rate 78%."

**Stretch Goal:** Live Pareto front of synthesis attempts updating as the audience watches

---

## #6 — APT-Hunter
**Domain:** Cybersecurity | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Median APT dwell time is months — this agent holds a hypothesis open for 30 days and survives shift handovers.

**Concept:**
- Multi-day blue-team agent maintains a persistent hypothesis graph in MongoDB (provenance + evidence weights)
- Hypothesis scores re-rank as new events arrive via change streams
- VIGIL reflective supervisor catches self-contradictions before they propagate

**Paper Anchor:** VIGIL Reflective Supervisor + [ProvAgent (arXiv:2603.09358)](https://arxiv.org/abs/2603.09358)

**MongoDB + AWS Sketch:** Hypothesis graph with provenance; Change Streams on SIEM feed; Bedrock for evidence reasoning; Lambda for shift-handover summaries

**Demo Script:** (1) Replay 3-week APT lab dataset in 90s → (2) show hypothesis re-ranking after hour-47 lateral-movement event → (3) VIGIL catches a false attribution, retracts it

**Stretch Goal:** "Theory tree" visualization that grows and prunes across simulated days with full provenance

---

## #7 — AuditLong
**Domain:** Compliance | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** A continuous agent replaces point-in-time compliance scans — it runs 365 days a year.

**Concept:**
- Daily evidence collection across AWS accounts; drift detection against in-force control set
- Compensable Saga steps when controls regress; `$graphLookup` audit trail lineage
- MongoDB stores full audit evidence with bi-temporal validity

**Paper Anchor:** [Zep Temporal KG (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)

**MongoDB + AWS Sketch:** Evidence collection per AWS service; Saga ledger for control regressions; `$graphLookup` for lineage; Bedrock for control-drift classification

**Demo Script:** (1) Kill agent → (2) re-deploy 30 simulated days later → (3) inject IAM policy violating CC6.1 → (4) agent detects, files ticket, shows bi-temporal audit log

**Stretch Goal:** "Days alive" compliance counter and audit log replaying as a timeline visualization

---

## #8 — MarketWatch13F
**Domain:** Finance | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** The new SEC quarterly filing cadence (Feb/May/Aug/Nov 2026) breaks every legacy compliance tool — except this one.

**Concept:**
- Durable filing agent maintains a holdings ledger as a bi-temporal MongoDB graph
- Cross-checks broker statements via change streams; detects position discrepancies
- Auto-prepares Form SHO ahead of Jan 2, 2028 effective date

**Paper Anchor:** [Zep Temporal KG (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)

**MongoDB + AWS Sketch:** Bi-temporal holdings collection; Change Streams on broker feed; Bedrock for filing language; Lambda for quarterly filing trigger

**Demo Script:** (1) Roll clock from Feb 17 to May 15 → (2) agent re-derives positions → (3) show one false-positive disclosure caught (position that appears held but was cleared)

**Stretch Goal:** Show the discrepancy that would have been filed by a stateless tool — and wasn't

---

## #9 — EvergreenIDE
**Domain:** Developer Tools | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** A background migration agent that survives for weeks — not just the session — and rebases its plan when upstream commits land.

**Concept:**
- Tackles repo migrations (e.g., Pydantic v1→v2 across 10k LOC) as a persistent background task
- LangGraph checkpointer + skill library in MongoDB; sleep-time compute compresses repo knowledge
- On new upstream commit: agent rebases its migration plan rather than restarting

**Paper Anchor:** [ReasoningBank (arXiv:2504.09762)](https://arxiv.org/abs/2504.09762)

**MongoDB + AWS Sketch:** Skill library with per-pattern success rates; LangGraph MongoDB checkpointer; Lambda for CI-triggered agent resume; Bedrock for code reasoning

**Demo Script:** (1) Show checkpoint from "yesterday" → (2) push a fresh upstream commit → (3) agent rebases its plan → (4) show CI failure → (5) ReasoningBank update: "avoid mutable default argument pattern in Pydantic v2 models"

**Stretch Goal:** Replay of CI failure → ReasoningBank update → re-execution that succeeds

---

## #10 — GrantOrbit
**Domain:** Government / Research Admin | **Difficulty:** ⭐⭐ | **Time Budget:** 24h
**Hook:** NIH/NSF grants span years — this agent owns the entire lifecycle from prospecting through Year-3 progress reports.

**Concept:**
- Per-award docket with drawdown schedule, indirect-cost negotiation, no-cost extension requests
- MongoDB schema mirrors RPPR sections; compensation when a PI changes institutions mid-grant
- Saga pattern handles the complex compensation logic for mid-grant institution changes

**Paper Anchor:** [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382)

**MongoDB + AWS Sketch:** Docket collection with RPPR section mapping; Bedrock for NIH language compliance; Lambda for milestone triggers

**Demo Script:** (1) Time-skip from notice-of-award to month-14 progress report → (2) inject PI institution change → (3) agent identifies compensation steps (transfer agreements, budget revisions)

**Stretch Goal:** Counter of "indirect dollars recovered" via the negotiation memory

---

## #11 — TelescopeNight
**Domain:** Space / Astronomy | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Gravitational-wave EM follow-up demands night-to-night memory — this agent persists across weather closures.

**Concept:**
- Autonomous multi-night follow-up agent prioritizes targets, handles weather closures
- ReasoningBank stores "why we observed X last Tuesday" and adapts tonight's queue accordingly
- MongoDB time-series tracks per-object observation history

**Paper Anchor:** [ReasoningBank (arXiv:2504.09762)](https://arxiv.org/abs/2504.09762)

**MongoDB + AWS Sketch:** Observation history time-series; target priority queue; Bedrock for multi-criteria target ranking; EventBridge for nightly planning triggers

**Demo Script:** (1) Inject a kilonova alert mid-run → (2) agent re-ranks queue, suspends low-priority targets → (3) show resumed observation after simulated rain closure

**Stretch Goal:** Sky map heat-replay across 7 simulated nights in 30 seconds

---

## #12 — BiobankShepherd
**Domain:** Healthcare Research | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Multi-year biobank participants drift, die, or revoke consent — this agent catches every edge case automatically.

**Concept:**
- Tracks consent renewals, re-contact rules, biospecimen QC across multi-year cohorts
- Withdrawal event triggers a biospecimen quarantine Saga with cascade to derivative datasets
- MIRIX episodic memory: every consent event logged with full temporal context

**Paper Anchor:** MIRIX multi-memory architecture + [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382)

**MongoDB + AWS Sketch:** Consent event time-series; participant temporal graph; Lambda for consent-expiry triggers; Bedrock for re-contact language

**Demo Script:** (1) Time-jump 14 months → (2) inject withdrawal event → (3) biospecimen quarantine Saga fires → (4) show cascade through 9 derivative datasets

**Stretch Goal:** Visualization of how one revoked consent cascades through dependent research outputs

---

## #13 — ClaimsMarathon
**Domain:** Insurance | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Allianz handles 1-day spoilage claims; bodily injury still takes 60+ days — this agent changes that.

**Concept:**
- Durable claim agent assembles medical records, computes reserve adjustments, schedules subrogation
- MongoDB Saga ledger tracks every compensable step; ColPali embeddings on medical record pages
- Reserve adjustments trigger downstream steps via Change Streams

**Paper Anchor:** [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382) + ColPali for document understanding

**MongoDB + AWS Sketch:** Claim Saga ledger; medical record GridFS + ColPali embeddings; Bedrock for reserve reasoning; Lambda for IME scheduling

**Demo Script:** (1) Crash mid-claim → (2) restart from checkpoint → (3) agent picks up at IME-scheduling step → (4) show live reserve-adjustment chart vs. industry baseline

**Stretch Goal:** Subrogation opportunity detection as the agent recovers additional dollars

---

## #14 — Reef-Witness
**Domain:** Climate / Conservation | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Coral bleaching models need multi-season context — this agent accumulates hypotheses across La Niña/El Niño cycles.

**Concept:**
- Ingests acoustic + satellite + diver-image streams; maintains a hypothesis ledger in MongoDB
- Sleep-time compute refines bleaching-onset model overnight via MIRIX episodic consolidation
- OmniEmbed-Nemotron for multimodal fusion of acoustic, visual, and satellite embeddings

**Paper Anchor:** MIRIX multi-memory architecture

**MongoDB + AWS Sketch:** Hypothesis ledger with confidence scores; multimodal embeddings in Atlas Vector Search; EventBridge for nightly model updates; Bedrock for hypothesis reasoning

**Demo Script:** (1) Replay 18 months of bleaching data → (2) inject unexpected La Niña event → (3) agent revises bleaching-onset model → (4) show hypothesis before/after

**Stretch Goal:** Reef-state heatmap evolving with hypothesis tags across seasons

---

## #15 — NEPA-Reviewer
**Domain:** Government / Permitting | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Federal NEPA reviews average 4.5 years — this agent tracks every public comment via Change Streams.

**Concept:**
- Per-project agent shepherds scoping → DEIS → FEIS → ROD across years
- Public comments ingested as a Change Stream event bus; response-to-comment matrices auto-generated
- Bi-temporal validity tracks document versions and supersessions

**Paper Anchor:** [Zep Temporal KG (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)

**MongoDB + AWS Sketch:** Comment collection with Change Streams; response matrix generation via Bedrock; document version graph; Lambda for comment clustering

**Demo Script:** (1) Inject 50 simulated public comments mid-DEIS → (2) agent clusters by topic → (3) drafts response matrix → (4) show comment-cluster constellation updating live

**Stretch Goal:** Response quality scoring against past FEIS approval rates

---

## #16 — CrystalCampaign
**Domain:** Scientific Research | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Cryo-EM campaigns crash on beamline glitches and lose all progress — this agent survives.

**Concept:**
- Owns the queue at a synchrotron; MongoDB checkpoints every diffraction frame
- ReasoningBank learns "which crystal morphologies are worth retrying after partial collection"
- Beamline recovery: agent resumes mid-collection without re-mounting after hardware restart

**Paper Anchor:** [ReasoningBank (arXiv:2504.09762)](https://arxiv.org/abs/2504.09762)

**MongoDB + AWS Sketch:** Frame collection time-series; ReasoningBank for retry strategy; Bedrock for structure quality assessment; Lambda for beamline event handling

**Demo Script:** (1) Running collection → (2) inject beamline crash → (3) agent resumes mid-collection from checkpoint → (4) show ReasoningBank update: "tetragonal crystal form: always retry after crash, 89% success rate"

**Stretch Goal:** Live electron-density map building over compressed 24 simulated hours

---

## #17 — SubmissionShepherd
**Domain:** Healthcare Regulatory | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** 67% of 510(k) submissions get Additional Information Requests — this agent learns from them.

**Concept:**
- Reconciles cited predicates across a 1,000-page eSTAR submission; ColPali embeddings on figures
- ReasoningBank learns from prior AIRs across a sponsor's portfolio
- Survives 142-day review cycle with full LangGraph checkpointing

**Paper Anchor:** [ReasoningBank (arXiv:2504.09762)](https://arxiv.org/abs/2504.09762) + ColPali

**MongoDB + AWS Sketch:** eSTAR section embeddings in GridFS + ColPali; AIR history ReasoningBank; Bedrock for predicate comparison; Lambda for CDRH portal monitoring

**Demo Script:** (1) Drop an AIR letter → (2) agent diffs claims against original predicate → (3) drafts response with cross-walk table → (4) show "Days saved" counter vs. CDRH average

**Stretch Goal:** Portfolio-level AIR pattern detection across all a sponsor's submissions

---

## #18 — WildfireWatch
**Domain:** Climate / Disaster | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** The 2025 LA fires showed that wildfire ops need agents that survive shift handovers — not just dashboards.

**Concept:**
- Persistent agent fuses MODIS/VIIRS satellite + RAWS sensor + IC voice transcripts via MIRIX memory
- Sleep-time compute refines fuel-model priors overnight; Change Streams trigger alerts
- Multi-shift survival: agent summarizes handover context automatically for incoming crew

**Paper Anchor:** MIRIX multi-memory architecture

**MongoDB + AWS Sketch:** Time-series for sensor streams; Change Streams for threshold alerts; MIRIX episodic consolidation via EventBridge; Bedrock for IC transcript analysis

**Demo Script:** (1) Replay 72-hour 2025 LA fire dataset → (2) inject sensor outage → (3) agent recovers from checkpoint → (4) show fuel-model update from overnight consolidation

**Stretch Goal:** Heatmap re-prioritizing across simulated nights with "what the agent learned while you slept"

---

## #19 — PreFlightOrbit
**Domain:** Aerospace | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** 500+ LEO operators can't staff 1:1 controllers — this per-spacecraft agent maintains hypotheses across orbit gaps.

**Concept:**
- Per-spacecraft agent monitors telemetry, persists anomaly hypotheses across communication passes
- VIGIL supervisor validates hypotheses before acting; stale hypotheses invalidated after orbit gaps
- MongoDB time-series for telemetry + hypothesis graph with temporal decay

**Paper Anchor:** VIGIL Reflective Supervisor + OPS-SAT-AD benchmark

**MongoDB + AWS Sketch:** Telemetry time-series; hypothesis graph with `valid_from`/`valid_to`; Bedrock for anomaly reasoning; Lambda for ground-pass triggers

**Demo Script:** (1) Inject battery brown-out anomaly → (2) show hypothesis forming → (3) orbit gap → (4) show stale hypothesis invalidated post-gap → (5) fresh hypothesis formed from new telemetry

**Stretch Goal:** 3D Earth view with the spacecraft "thinking" as it transits the dark side

---

## #20 — CodexPapyrus
**Domain:** Cultural Heritage | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Vesuvius Challenge–style CT scan decipherment is a months-long human/AI collaboration — this agent holds the thread.

**Concept:**
- Long-running agent iterates segmentation → ink-detection → glyph-recognition across weeks
- ReasoningBank learns from human verification on each fragment; MongoDB stores sub-mm patch embeddings
- Human-in-the-loop: when human challenges a glyph reading, agent updates strategy for similar fragments

**Paper Anchor:** [ReasoningBank (arXiv:2504.09762)](https://arxiv.org/abs/2504.09762)

**MongoDB + AWS Sketch:** Patch embeddings in GridFS + Atlas Vector Search; human verification queue; Bedrock for OCR reasoning; Lambda for nightly batch processing

**Demo Script:** (1) Show 2 weeks of decoding compressed → (2) human challenges a glyph → (3) agent updates strategy → (4) show new Greek words appearing live

**Stretch Goal:** Confidence-weighted transcript with uncertainty bands per glyph

---

## #21 — HarvestChain
**Domain:** Agriculture | **Difficulty:** ⭐⭐ | **Time Budget:** 24h
**Hook:** Smallholder farmers lose seasons because no AI remembers their field's soil and pest history.

**Concept:**
- Per-plot agent learns across an entire growing season; fuses sensor + satellite + voice notes
- MongoDB time-series for sensor readings; small-model inference for Swahili/Hausa on-device
- Sleep-time compute consolidates each night; farmer thumbs-up updates pest-detection strategy

**Paper Anchor:** MIRIX multi-memory architecture + small LLM (Phi-4 / Gemma 2)

**MongoDB + AWS Sketch:** Plot time-series; farmer voice note transcription via Bedrock; sleep consolidation via EventBridge; ReasoningBank for per-crop pest strategies

**Demo Script:** (1) Compress 4 simulated months → (2) pest detection → (3) farmer provides feedback → (4) show ReasoningBank update improving next-week prediction

**Stretch Goal:** Per-field "memory diary" the farmer can scroll through their season

---

## #22 — PolicyMarathon
**Domain:** Legal / Compliance | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** EU AI Act, Colorado ECDIS, NAIC bulletins — this agent tracks law as a living graph, not a static document.

**Concept:**
- Bi-temporal regulation-state graph with `valid_from`/`valid_to` at the section level
- Auto-emits diffs to client compliance dossiers when source text changes
- Zep temporal KG: "what was the legal position on Article 9 on March 15, 2025?"

**Paper Anchor:** [Zep Temporal KG (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)

**MongoDB + AWS Sketch:** Bi-temporal regulation graph; Change Streams for law-update events; Bedrock for diff generation; Lambda for dossier update triggers

**Demo Script:** (1) Replay EU Digital Omnibus delay vote → (2) agent re-baselines all affected dossiers → (3) show before/after compliance status per jurisdiction

**Stretch Goal:** Live "compliance drift" gauge per jurisdiction showing distance from current law

---

## #23 — MAUDEHunter
**Domain:** Healthcare Research | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 48h
**Hook:** PSURs take years and miss signals — this agent runs Bayesian signal detection continuously.

**Concept:**
- Continuous agent maps adverse-event signals, runs disproportionality analyses on rolling windows
- Long-horizon Bayesian updating (not stateless RAG); hypothesis evolution stored in MongoDB
- Crosses signal-of-disproportionate-reporting threshold → auto-drafts PSUR section

**Paper Anchor:** [ReasoningBank (arXiv:2504.09762)](https://arxiv.org/abs/2504.09762) + pharmacovigilance signal detection literature

**MongoDB + AWS Sketch:** AE event time-series; signal hypothesis graph; Bedrock for MedDRA mapping; Lambda for weekly rolling-window analysis

**Demo Script:** (1) Inject a fake cluster of adverse events → (2) agent crosses PRR threshold → (3) PSUR section auto-drafted → (4) show signal scores re-ranking across simulated weeks

**Stretch Goal:** Cross-product signal detection: same mechanism, different devices

---

## #24 — MA-Diligence
**Domain:** Finance | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** M&A diligence and the 100-day post-merger integration have never been owned by a single agent — until now.

**Concept:**
- Durable agent persists from data-room access through Day-60 integration milestones
- MongoDB stores per-thread checkpoints and a synergy-realization tracker
- Saga pattern handles compensation when expected synergies are revised post-close

**Paper Anchor:** [Zep Temporal KG (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956) + [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382)

**MongoDB + AWS Sketch:** Per-diligence-thread checkpoints; synergy tracker collection; Bedrock for financial document analysis; Lambda for milestone triggers

**Demo Script:** (1) Time-jump from LOI to Day-60 integration → (2) new ERP data arrives → (3) synergy dashboard re-derived automatically → (4) show "burn-up" chart with provenance pop-ups

**Stretch Goal:** Synergy variance analysis: what was promised vs. what was delivered, with causal attribution

---

## #25 — StableYear
**Domain:** Healthcare Research | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Chronic-disease cohort studies run for years — this research agent (no clinical advice) maintains continuous feature engineering.

**Concept:**
- Per-patient research agent with MIRIX episodic memory of labs, meds, visits
- Cohort membership re-derived dynamically when new patients are added (no full rescan)
- FHIR-shaped MongoDB docs; Change Streams trigger cohort updates
- Strict research-only framing: feature engineering and cohort selection, not clinical decisions

**Paper Anchor:** MIRIX multi-memory architecture + FHIR standards

**MongoDB + AWS Sketch:** FHIR-shaped patient documents; Atlas Vector Search for phenotype similarity; Bedrock for clinical NLP; Change Streams for cohort membership

**Demo Script:** (1) Add a new patient mid-run → (2) cohort membership re-derived → (3) per-cohort survival curve updating in real time → (4) show which new patient changed which cohort

**Stretch Goal:** Counterfactual cohort analysis: "what would the survival curve look like without this subgroup?"

---

## #26 — AuditAgentTax
**Domain:** Finance / Government | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Corporate tax audits with IDRs and appeals span 6–18 months — this agent owns the entire case.

**Concept:**
- Durable agent owns IDR lifecycle, drafts responses, tracks reserve positions
- Saga handles compensation when adjustments are reversed at appeal
- MongoDB stores full audit trail with `$graphLookup` across adjustments and appeals

**Paper Anchor:** [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382)

**MongoDB + AWS Sketch:** IDR collection with Saga ledger; reserve position time-series; Bedrock for tax law reasoning; Lambda for IDR deadline triggers

**Demo Script:** (1) Show Day-1 checkpoint → (2) jump to Day-90 IDR resolution → (3) inject appeal reversal → (4) Saga compensation fires → (5) show live reserve-volatility chart

**Stretch Goal:** Cross-jurisdiction tax position comparison across entities

---

## #27 — PortfolioGardener
**Domain:** Finance | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Buy-side analysts revisit investment theses for years — current copilots forget everything between sessions.

**Concept:**
- Per-thesis agent persists a "view-of-the-world" document in MongoDB; updated nightly
- Sleep-time compute processes new filings + earnings transcripts + news overnight
- Bi-temporal thesis graph: "what we believed in March vs. now" with provenance diffs

**Paper Anchor:** [Zep Temporal KG (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)

**MongoDB + AWS Sketch:** Bi-temporal thesis collection; earnings transcript embeddings; Bedrock for financial reasoning; EventBridge for nightly updates

**Demo Script:** (1) Show Nvidia thesis from January → (2) fast-forward through key 2025–2026 events → (3) show thesis evolution with "what changed last week" diff → (4) bi-temporal slider replay

**Stretch Goal:** Cross-thesis correlation detection: "this thesis depends on these 3 macro assumptions"

---

## #28 — ProcessLoom
**Domain:** Manufacturing / Industrial | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** Back-office automation (dunning, vendor onboarding) needs agents that run for weeks without babysitting.

**Concept:**
- Multi-week BPO automation with idempotent steps and Temporal-on-MongoDB durable execution
- VIGIL reflective sibling supervisor catches self-miscorrelations
- Side-by-side demo: "agent with VIGIL" vs. "agent without" — one fails silently, one catches it

**Paper Anchor:** VIGIL Reflective Supervisor + durable execution patterns

**MongoDB + AWS Sketch:** Process step ledger; VIGIL supervisor collection; Bedrock for business logic reasoning; Lambda for step execution

**Demo Script:** (1) Run dunning process → (2) crash agent → (3) restart: VIGIL supervisor detects and blocks a double-send that would have gone out → (4) show why idempotency key prevented it

**Stretch Goal:** Process efficiency metrics: time-per-step trending over weeks

---

## #29 — PaperShepherd
**Domain:** Scientific Publishing | **Difficulty:** ⭐⭐ | **Time Budget:** 24h
**Hook:** Multi-round peer review spans months — this agent maintains a "promise ledger" across every reviewer round.

**Concept:**
- Author-side agent persists across review rounds; maintains response-to-reviewers document
- "Promise ledger": every commitment tracked to verification (did we add that control experiment?)
- Runs supplementary analyses via tool calls when reviewers request new data

**Paper Anchor:** [ReasoningBank (arXiv:2504.09762)](https://arxiv.org/abs/2504.09762)

**MongoDB + AWS Sketch:** Promise ledger collection; reviewer comment embeddings; Bedrock for response drafting; Lambda for round-change triggers

**Demo Script:** (1) Drop in Round-2 reviews → (2) agent diffs against Round-1 promises → (3) identifies 3 unkept promises → (4) schedules supplementary analyses

**Stretch Goal:** Automatic detection of "round 2 introduces contradictions with round 1 commitments"

---

## #30 — PenguinPipe
**Domain:** Conservation | **Difficulty:** ⭐⭐ | **Time Budget:** 24h
**Hook:** Trail-cam classification needs per-individual memory across moult events — this agent maintains penguin IDs season to season.

**Concept:**
- Per-individual ColPali embeddings updated as penguins moult and re-ID becomes challenging
- Sleep-time compute self-corrects mis-IDs using colony-wide behavioral consistency
- 3-month behavioral hypothesis updating via MIRIX episodic memory

**Paper Anchor:** MIRIX multi-memory + ColPali embeddings

**MongoDB + AWS Sketch:** Per-individual embedding collection; behavioral graph; Bedrock for behavioral analysis; EventBridge for nightly re-ID consolidation

**Demo Script:** (1) Feed 3 simulated months of trail-cam data → (2) moult event causes mis-ID → (3) agent self-corrects using behavioral consistency → (4) show per-penguin life story timeline

**Stretch Goal:** Flock-level behavioral anomaly detection across seasons

---

## #31 — RFISprint
**Domain:** Construction | **Difficulty:** ⭐⭐⭐ | **Time Budget:** 48h
**Hook:** 5.2-day median RFI response time causes 37% of construction schedule overruns — this agent closes in hours.

**Concept:**
- Durable agent maintains the live submittal log per project across 18 months
- ColPali-embedded spec pages + Procore-style Change Streams for RFI intake
- SagaLLM compensates 14 dependent submittals when a structural change order arrives

**Paper Anchor:** [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382) + ColPali

**MongoDB + AWS Sketch:** Submittal log with Saga ledger; spec embeddings in GridFS + ColPali; Bedrock for spec interpretation; Change Streams for RFI intake

**Demo Script:** (1) Drop in a structural change order → (2) agent identifies 14 dependent submittals → (3) Saga compensation fires for each → (4) show critical-path Gantt re-flowing live

**Stretch Goal:** Automatic escalation routing: which RFIs need architect vs. engineer vs. owner decision

---

## #32 — AdverseShield
**Domain:** Insurance | **Difficulty:** ⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** Subrogation recovery is the largest unbuilt insurance ROI category — this agent hunts opportunities automatically.

**Concept:**
- Long-running agent graph-walks claims-ledger entities to identify subrogation opportunities
- Temporal-style retries for demand letters via Saga pattern
- MongoDB `$graphLookup` traverses liability chains across multiple claims

**Paper Anchor:** [SagaLLM (arXiv:2312.05382)](https://arxiv.org/abs/2312.05382)

**MongoDB + AWS Sketch:** Claims graph with `$graphLookup`; demand letter Saga ledger; Bedrock for liability reasoning; Lambda for demand letter scheduling

**Demo Script:** (1) Time-jump 90 days → (2) show "recovered $" counter → (3) show demand-letter call-graph of who the agent contacted on Day 47 and why

**Stretch Goal:** Liability chain visualization showing multi-party subrogation paths

---

## #33 — ProvAgentX
**Domain:** Cybersecurity | **Difficulty:** ⭐⭐⭐⭐⭐ | **Time Budget:** 1 week
**Hook:** The most sophisticated APT investigation tool ever open-sourced — fusing ProvAgent, VIGIL, and ReasoningBank.

**Concept:**
- Provenance-graph agent retains identity-behavior bindings across a multi-day campaign
- VIGIL supervisor catches self-miscorrelations before they propagate into false attributions
- ReasoningBank distills "what IOCs are reliable signals vs. noise" from past investigations

**Paper Anchor:** [ProvAgent (arXiv:2603.09358)](https://arxiv.org/abs/2603.09358) + VIGIL + [ReasoningBank (arXiv:2504.09762)](https://arxiv.org/abs/2504.09762)

**MongoDB + AWS Sketch:** Provenance hypergraph; VIGIL supervisor collection; ReasoningBank for IOC reliability; Change Streams on SIEM feed; Bedrock for attribution reasoning

**Demo Script:** (1) Replay a 5-day kill chain in 90s → (2) show one false positive → (3) VIGIL supervisor invalidates it with evidence → (4) ReasoningBank update: "LSASS dump alone is insufficient for attribution — requires lateral movement confirmation"

**Stretch Goal:** "Why we now believe" replay over the kill chain timeline with full provenance

---

*[← Theme 1 README](README.md) | [🏠 10_Hackathons](../README.md) | [Theme 2 →](../theme_2_multi_agent_collaboration/README.md)*
