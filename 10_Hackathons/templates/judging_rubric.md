# ⚖️ Hackathon Judging Rubric

> Self-score before demo day. If you score below 3 on any criterion, fix it — don't hope the judges miss it.

---

## Standard Weights

| Criterion | Weight | What Judges Actually Look For |
|-----------|--------|-------------------------------|
| **Innovation / Creativity** | 35% | Is this genuinely new? Does it use a 2024–2026 paper as a foundation? Would a domain expert say "I've never seen that before"? |
| **Technical Depth** | 25% | Is the architecture non-trivial? Is there a real algorithm, not just API calls? Does the MongoDB+AWS usage justify the stack? |
| **Demo Polish** | 25% | Does it work live? Is there a wow moment? Would someone share a 10-second clip of this? |
| **Business / Social Impact** | 15% | Is there a named, real industry pain point? Is there a credible path to value if this were productionized? |

---

## Scoring Guide (1–5 per criterion)

### Innovation / Creativity (35%)

| Score | Description |
|-------|-------------|
| 5 | Grounded in a specific 2024–2026 paper; solves a problem no existing tool handles; judges say "I didn't know this was possible" |
| 4 | Clear novelty; good paper anchor; incremental improvement over state-of-art is well-argued |
| 3 | Interesting combination of existing tools; novelty is incremental but defensible |
| 2 | Familiar pattern (chatbot + RAG + CRUD); no research grounding |
| 1 | Clone of existing product with no differentiation |

### Technical Depth (25%)

| Score | Description |
|-------|-------------|
| 5 | Multi-agent coordination with non-trivial state management; MongoDB used for its specific capabilities (not just as a key-value store); real algorithm present |
| 4 | Solid architecture; clear separation of concerns; retrieval or memory design is thoughtful |
| 3 | Works end-to-end; some design choices explained; stack makes sense |
| 2 | Overly simple pipeline; one API call + one database; no meaningful architecture |
| 1 | Broken, incomplete, or couldn't explain the architecture when asked |

### Demo Polish (25%)

| Score | Description |
|-------|-------------|
| 5 | Live demo works flawlessly; there is a moment that makes the audience gasp; demo tells a story not just shows features |
| 4 | Demo works; clear narrative; at least one impressive visual moment |
| 3 | Demo works; some awkward pauses; output is visible and understandable |
| 2 | Partial demo; some things broken; fell back to slides for key parts |
| 1 | Slides only; live demo crashed and wasn't recovered |

### Business / Social Impact (15%)

| Score | Description |
|-------|-------------|
| 5 | Named industry pain point with quantified cost; clear stakeholder; credible monetization or deployment path |
| 4 | Real problem; plausible impact; some quantification |
| 3 | Problem is real but impact is vague |
| 2 | Toy problem or unclear who benefits |
| 1 | No apparent real-world use case |

---

## Self-Scoring Checklist

### Innovation
- [ ] I can name the specific paper this is grounded in (with arXiv ID)
- [ ] I can name one existing product that tried this and failed, and explain why mine is different
- [ ] A domain expert would find this novel (not just an ML engineer)

### Technical Depth
- [ ] MongoDB is used for a specific capability (Change Streams / Vector Search / Time-Series / GridFS), not just document storage
- [ ] There is at least one non-trivial algorithm (not just prompt + LLM call)
- [ ] I can explain the architecture in one sentence without using the word "pipeline"
- [ ] The agent can fail and recover (checkpointing / compensation demonstrated)

### Demo Polish
- [ ] The demo has been run at least 5 times successfully end-to-end
- [ ] I can identify the exact 5-second moment that is the "wow"
- [ ] I have a fallback if live APIs fail (pre-seeded MongoDB data)
- [ ] Font sizes are readable from 5 meters away

### Business / Social Impact
- [ ] I can state the cost of the problem in dollars, lives, or time
- [ ] I can name a real organization that has this exact problem today
- [ ] I can describe the "before" state (without my system) and "after" state clearly

---

## Anti-Patterns to Avoid

| Anti-Pattern | Fix |
|-------------|-----|
| Generic RAG chatbot over PDFs | Add adaptive retrieval strategy switching; add domain-specific reranking |
| "Our agent uses GPT-4 to..." | Name the specific model capability you're exploiting, not just "GPT-4" |
| No paper anchor | Find the arXiv paper for your core mechanism before building |
| MongoDB used as simple document store | Use Change Streams, Vector Search, or Time-Series — otherwise use Postgres |
| Demo is just slides with fake screenshots | Build a minimal working prototype; demo the actual system |
| "We would also add feature X, Y, Z..." | Ship what you said; don't spec out unbuilt features during the demo |
| No recovery from failure | Implement LangGraph checkpointing; crash + restart is one of the most impressive demo moves |

---

## The Memorable Demo Principle

> A demo that makes one judge say *"I need to show this to my team"* scores higher than a demo that technically checks every box but surprises no one.

Ask yourself: **What is the one 5-second moment in my demo that someone would screenshot and share?**

If you don't have an answer, go find one before demo day.
