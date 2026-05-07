# 🎬 90-Second Demo Script Template

> The demo is not the product. The demo is the argument that the product solves a real problem.

---

## Structure

| Segment | Duration | Goal |
|---------|----------|------|
| **Hook** | 0–10s | One number or question that makes the audience feel something |
| **Problem** | 10–25s | Why existing tools fail (name the specific failure) |
| **Live Demo** | 25–65s | The moment that makes the audience gasp |
| **How It Works** | 65–80s | One clear architectural insight (not a slide) |
| **Ask / CTA** | 80–90s | What you want from the judges |

---

## The Hook (0–10s)

**Formula**: `[Shocking stat] + [Who suffers] + [Why nothing fixes it yet]`

**Examples that work:**
- *"FDA was late on 73.9% of device recall terminations. Three million patients were potentially exposed. The reason: no agent has ever been built to last longer than a single session."*
- *"1 in 3 Gen Z checks TikTok before their doctor. 1 in 5 health videos on TikTok contains misinformation. Current fact-checkers live on websites no one visits."*
- *"The Red Sea crisis cost UK supermarkets £340M. No procurement AI predicted it because none of them think in 18-month horizons."*

**Anti-patterns:**
- ❌ *"Hi, we built an AI agent that..."* — starts with you, not the problem
- ❌ *"Today I'm going to show you..."* — wastes 8 seconds
- ❌ Any sentence starting with "So basically..."

---

## The Problem (10–25s)

Name the **specific mechanism of failure** in existing tools:
- *"Existing tools are stateless. They forget everything the moment the session ends."*
- *"Every retrieval system picks ONE strategy (vector OR keyword) and can't switch."*
- *"They answer the question. They don't track the mutation."*

Then: *"What you're about to see is the first system that [solves specific mechanism]."*

---

## The Live Demo (25–65s)

This is 40 seconds. Every second counts.

**The 3 signature moves** (use at least 2):

### Move A: Crash + Restart
1. Show agent running in state X
2. Kill the process (dramatically)
3. Restart — agent picks up exactly where it left off from MongoDB checkpoint
4. *"It didn't lose a single byte."*

### Move B: Live Event Stream
1. Stream a new document/event into a MongoDB change stream panel
2. Agent reacts in real time (rerank, re-plan, new output appears)
3. *"That just happened. The agent saw it. Here's what it did."*

### Move C: Learning Signal
1. Show system state BEFORE (ReasoningBank empty / nDCG = 0.31)
2. Run one interaction cycle
3. Show state AFTER (new entry / nDCG = 0.68)
4. *"It just got smarter. From this one interaction. Watch what happens next time."*

---

## How It Works (65–80s)

ONE sentence architecture. Not a diagram. Not a list.

**Formula**: `[Input] → [Key insight / mechanism] → [Output that matters]`

**Examples:**
- *"Every mutation of the claim gets a ColPali embedding in MongoDB — so when we see the Arabic dub, we already know where it came from."*
- *"The retrieval router is a bandit: it tries vector search first, gets a quality signal from the analyst's click, and shifts probability toward the winning strategy for that query type."*
- *"The Saga pattern in MongoDB means every compensating action is logged — so when the regulator reverses a ruling, every downstream conclusion gets automatically flagged for review."*

---

## The Ask (80–90s)

What do you want the judges to imagine?

**Formula**: `[What just happened] + [What it means at scale] + [The one thing you want judges to feel]`

*"You just watched a claim mutate across 11 nodes, in 4 languages, over 36 simulated hours — and the counter-narrative was ready for each platform before the next mutation formed. At scale, this is the immune system the information ecosystem doesn't have yet."*

---

## Demo Environment Checklist

Before the demo, verify:
- [ ] MongoDB Atlas connection string tested in the last 30 minutes
- [ ] Demo dataset pre-loaded (don't rely on live API calls for the main sequence)
- [ ] Bedrock model availability checked (use us-east-1 — highest availability)
- [ ] Change stream terminal window open and visible
- [ ] ReasoningBank "before" state screenshot ready as fallback
- [ ] Backup: screen recording of the demo in case live fails
- [ ] Font size 18+ on code panels (judges at the back of the room)

---

## What NOT to Do

| Anti-Pattern | Why It Kills Your Demo |
|--------------|----------------------|
| Showing your code | Judges aren't reading code in 90 seconds |
| Demoing on a slide | Slides are not demos |
| "It would also do X, Y, Z..." | Scope creep signals you didn't finish anything |
| Explaining what RAG is | Assume technical judges; don't teach basics |
| Live API calls for critical path | Rate limits and latency will embarrass you |
| Ending with "...and that's it!" | End with stakes re-anchored, not with a shrug |
