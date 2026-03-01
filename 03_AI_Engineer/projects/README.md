# AI Engineer — Portfolio Projects

> AI Engineer projects should demonstrate you can ship LLM-powered systems that are reliable, evaluated, safe, and production-ready — not just demo-quality chatbots.

---

## What Separates AI Engineer Projects from "I Built a Chatbot"

| "I built a chatbot" | AI Engineer project |
|---------------------|---------------------|
| Hardcoded system prompt | Evaluated + iterated prompt |
| No evals | RAGAS / custom eval harness |
| Demo only | Deployed + monitored |
| Hallucination ignored | Hallucination detection + grounding |
| Single call | Retry logic, fallbacks, guardrails |
| OpenAI only | Model-agnostic, cost-optimized |

---

## Project 1: Document Intelligence API (Beginner AIE)

**Objective:** Build a production-ready document Q&A API that any application can integrate.

**The problem:** Companies have PDFs (legal contracts, technical manuals, policies) and need to query them intelligently.

**Tech stack:** FastAPI + LangChain + Chroma + Claude/OpenAI

```
document-intelligence-api/
├── src/
│   ├── api/
│   │   ├── main.py           ← FastAPI application
│   │   ├── routes/
│   │   │   ├── documents.py  ← Upload/manage documents
│   │   │   └── query.py      ← Q&A endpoints
│   │   └── schemas.py        ← Pydantic models
│   ├── rag/
│   │   ├── chunker.py        ← Smart chunking strategies
│   │   ├── indexer.py        ← Embedding + storage
│   │   └── retriever.py      ← Hybrid retrieval
│   ├── evals/
│   │   ├── ragas_eval.py     ← Automated evaluation
│   │   └── golden_dataset.json
│   └── guardrails/
│       └── safety.py         ← Input/output checks
├── tests/
├── Dockerfile
└── README.md
```

**Key endpoints to implement:**

```python
# POST /documents — Upload a document
# GET  /documents — List all documents
# DELETE /documents/{id} — Remove a document
# POST /query — Ask a question
# GET  /query/history — Get conversation history
# POST /eval — Run quality evaluation

# Example usage:
import httpx

# Upload
with open("contract.pdf", "rb") as f:
    httpx.post("http://api/documents", files={"file": f})

# Query
response = httpx.post("http://api/query", json={
    "question": "What are the termination clauses?",
    "doc_ids": ["contract-001"],
    "conversation_id": "session-abc"
})
print(response.json())
# {
#   "answer": "The contract can be terminated...",
#   "sources": [{"page": 12, "text": "..."}],
#   "confidence": 0.92,
#   "response_time_ms": 450
# }
```

**Evaluation harness to include:**

```python
# evals/ragas_eval.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

def evaluate_rag_system(test_cases: list[dict]) -> dict:
    """
    test_cases format:
    [{"question": "...", "ground_truth": "...", "answer": "...", "contexts": ["..."]}]
    """
    dataset = Dataset.from_list(test_cases)
    results = evaluate(dataset, metrics=[
        faithfulness, answer_relevancy, context_precision, context_recall
    ])

    print(f"Faithfulness:     {results['faithfulness']:.3f}  (target: >0.85)")
    print(f"Answer Relevancy: {results['answer_relevancy']:.3f}  (target: >0.85)")
    print(f"Context Precision:{results['context_precision']:.3f}  (target: >0.80)")
    print(f"Context Recall:   {results['context_recall']:.3f}  (target: >0.80)")

    # PASS/FAIL quality gate
    thresholds = {"faithfulness": 0.80, "answer_relevancy": 0.80,
                  "context_precision": 0.75, "context_recall": 0.75}
    passed = all(results[m] >= thresholds[m] for m in thresholds)
    print(f"\nQuality gate: {'PASSED' if passed else 'FAILED'}")
    return results
```

**Portfolio talking points:**
- "Achieved 0.89 faithfulness score on a 50-question golden dataset"
- "Hybrid search improved context recall from 0.72 to 0.88"
- "Reranking reduced irrelevant context by 40%"

---

## Project 2: AI Customer Support Agent (Intermediate AIE)

**Objective:** Build a production customer support system that handles real queries, escalates appropriately, and learns from feedback.

**Business context:** An e-commerce company processes 5,000 support tickets/day. 70% are routine (order status, returns). Goal: automate 60% while maintaining CSAT > 4.2/5.

**Architecture:**

```
                Customer Message
                      ↓
           ┌──────────────────────┐
           │  Guardrails Layer    │
           │  - PII redaction     │
           │  - Toxicity filter   │
           │  - Injection detect  │
           └──────────┬───────────┘
                      ↓
           ┌──────────────────────┐
           │  Intent Classifier   │  ← Fine-tuned classifier
           │  (LLM or embeddings) │
           └──────────┬───────────┘
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
   Order Handler  Return Handler  FAQ Handler   Escalation
   (tool calls)   (RAG + rules)   (RAG)         (human)
        ↓             ↓             ↓
        └─────────────┼─────────────┘
                      ↓
           ┌──────────────────────┐
           │  Response Generator  │
           │  + Quality Check     │
           └──────────────────────┘
                      ↓
              Final Response
```

**Key implementation: Tool-calling agent:**

```python
from anthropic import Anthropic
import json

client = Anthropic()

TOOLS = [
    {
        "name": "get_order_status",
        "description": "Get the current status of a customer order",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID"},
                "customer_email": {"type": "string", "description": "Customer email"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "initiate_return",
        "description": "Start a return process for an eligible order",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "enum": ["damaged", "wrong_item", "not_as_described", "other"]},
                "item_ids": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["order_id", "reason"]
        }
    },
    {
        "name": "search_faq",
        "description": "Search the FAQ database for policy information",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate this conversation to a human agent",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]}
            },
            "required": ["reason", "priority"]
        }
    }
]

def run_support_agent(customer_message: str, conversation_history: list) -> dict:
    """Run the support agent with tool calling."""

    messages = conversation_history + [
        {"role": "user", "content": customer_message}
    ]

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system="""You are a helpful customer service agent for TechShop.
Be empathetic, professional, and solution-oriented.
Always use tools to get real data — never make up order information.
If you cannot resolve an issue, escalate to a human agent.""",
        tools=TOOLS,
        messages=messages,
    )

    # Execute tool calls
    if response.stop_reason == "tool_use":
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                result = execute_tool(content_block.name, content_block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": json.dumps(result)
                })

        # Continue conversation with tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        final_response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system="""You are a helpful customer service agent for TechShop.""",
            tools=TOOLS,
            messages=messages,
        )

        return {
            "response": final_response.content[0].text,
            "tools_used": [c.name for c in response.content if c.type == "tool_use"],
        }

    return {"response": response.content[0].text, "tools_used": []}
```

**Metrics to track and present:**
- Automation rate: % of tickets resolved without human
- First response time: target <30 seconds
- CSAT: customer satisfaction score (1-5)
- Escalation rate: % routed to humans
- Hallucination rate: % of answers not grounded in data

---

## Project 3: Multi-Modal AI Application (Intermediate AIE)

**Objective:** Build an application that understands images, audio, and text — demonstrating multi-modal AI integration.

**Use case:** Invoice processing system that extracts structured data from:
- Photos of paper invoices
- PDF invoices
- Email text invoices

```python
import base64
from anthropic import Anthropic
from pydantic import BaseModel
from typing import Optional
import json

client = Anthropic()

class InvoiceData(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: str
    due_date: Optional[str]
    line_items: list[dict]
    subtotal: float
    tax: Optional[float]
    total: float
    currency: str = "USD"
    payment_terms: Optional[str]
    notes: Optional[str]


def extract_invoice_from_image(image_path: str) -> InvoiceData:
    """Extract structured invoice data from an image using Claude Vision."""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.split(".")[-1].lower()
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                  "png": "image/png", "pdf": "application/pdf"}.get(ext, "image/jpeg")

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": """Extract all invoice data from this image.
Return JSON with these exact fields:
{
  "vendor_name": "string",
  "invoice_number": "string",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD or null",
  "line_items": [{"description": "string", "quantity": number, "unit_price": number, "amount": number}],
  "subtotal": number,
  "tax": number or null,
  "total": number,
  "currency": "USD",
  "payment_terms": "string or null",
  "notes": "string or null"
}
Extract exact values from the invoice. Do not estimate or infer."""
                }
            ],
        }]
    )

    extracted = json.loads(response.content[0].text)
    return InvoiceData(**extracted)


def batch_process_invoices(image_paths: list[str]) -> list[dict]:
    """Process multiple invoices and return summary."""
    results = []
    for path in image_paths:
        try:
            invoice = extract_invoice_from_image(path)
            results.append({"file": path, "status": "success", "data": invoice.model_dump()})
        except Exception as e:
            results.append({"file": path, "status": "error", "error": str(e)})

    total = sum(r["data"]["total"] for r in results if r["status"] == "success")
    print(f"Processed {len(results)} invoices. Total: ${total:,.2f}")
    return results
```

---

## Project 4: AI Research Assistant with Memory (Advanced AIE)

**Objective:** Build a personal AI research assistant that remembers context across sessions, learns your preferences, and helps synthesize information.

**The novel feature:** Persistent, structured memory that improves over time.

```python
# Memory architecture:
# - Short-term: Current conversation context
# - Medium-term: Session summaries (last 30 days)
# - Long-term: Facts, preferences, relationship graph (permanent)

from openai import OpenAI
import json
import sqlite3
from datetime import datetime

client = OpenAI()

class PersistentMemoryAgent:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db = sqlite3.connect(db_path)
        self._init_db()
        self.current_conversation = []

    def _init_db(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,  -- 'fact', 'preference', 'event'
                importance REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY,
                summary TEXT NOT NULL,
                topics TEXT,  -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> list[str]:
        """Get memories relevant to current query."""
        from openai import OpenAI
        # Use embeddings for semantic search
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        cursor = self.db.execute(
            "SELECT content FROM memories ORDER BY importance DESC, last_accessed DESC LIMIT ?",
            (top_k * 3,)  # Get more candidates, then re-rank
        )
        memories = [row[0] for row in cursor.fetchall()]
        return memories[:top_k]

    def extract_and_store_memories(self, conversation: list[dict]):
        """Extract important facts from conversation and store."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """Extract important facts from this conversation to remember.
Return JSON: {"memories": [{"content": "fact to remember", "type": "fact|preference|event", "importance": 0-1}]}
Only extract genuinely important, reusable information. Skip trivial details."""
                },
                {"role": "user", "content": json.dumps(conversation)}
            ]
        )
        data = json.loads(response.choices[0].message.content)
        for memory in data.get("memories", []):
            self.db.execute(
                "INSERT INTO memories (content, memory_type, importance) VALUES (?, ?, ?)",
                (memory["content"], memory["type"], memory["importance"])
            )
        self.db.commit()

    def chat(self, user_message: str) -> str:
        """Have a conversation with persistent memory."""
        # Retrieve relevant memories
        memories = self.retrieve_relevant_memories(user_message)
        memory_context = "\n".join([f"- {m}" for m in memories]) if memories else "No relevant memories."

        self.current_conversation.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a personal research assistant with persistent memory.

RELEVANT MEMORIES FROM PAST SESSIONS:
{memory_context}

Use these memories to provide personalized, contextually-aware responses.
Build on past conversations when relevant."""
                }
            ] + self.current_conversation[-20:],  # Last 20 turns
        )

        assistant_message = response.choices[0].message.content
        self.current_conversation.append({"role": "assistant", "content": assistant_message})

        # Extract memories every 5 turns
        if len(self.current_conversation) % 10 == 0:
            self.extract_and_store_memories(self.current_conversation[-10:])

        return assistant_message
```

---

## Project 5: AI Evaluation Platform (Advanced AIE)

**Objective:** Build a platform that systematically evaluates LLM responses across multiple dimensions — the kind of tooling real AI teams build internally.

**Inspired by:** OpenAI Evals, Braintrust, HoneyHive, PromptFlow

```python
# Core evaluation framework
from dataclasses import dataclass
from enum import Enum
from typing import Callable
import json

class EvalMetric(Enum):
    CORRECTNESS = "correctness"
    FAITHFULNESS = "faithfulness"
    CONCISENESS = "conciseness"
    SAFETY = "safety"
    TONE = "tone"

@dataclass
class EvalResult:
    metric: EvalMetric
    score: float  # 0-1
    reasoning: str
    passed: bool

class LLMJudge:
    """Use GPT-4o as a judge to evaluate other models."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def evaluate(self, question: str, answer: str, ground_truth: str,
                 context: str = "", metric: EvalMetric = EvalMetric.CORRECTNESS) -> EvalResult:

        prompts = {
            EvalMetric.CORRECTNESS: f"""Rate if this answer is factually correct.
Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}
Return JSON: {{"score": 0-1, "reasoning": "explanation", "passed": true/false}}""",

            EvalMetric.FAITHFULNESS: f"""Rate if this answer is grounded in the context.
Context: {context}
Answer: {answer}
Return JSON: {{"score": 0-1, "reasoning": "explanation", "passed": true/false}}
Score 0 if the answer contains claims not in the context.""",

            EvalMetric.CONCISENESS: f"""Rate if this answer is appropriately concise.
Question: {question}
Answer: {answer}
Return JSON: {{"score": 0-1, "reasoning": "explanation", "passed": true/false}}
Deduct points for unnecessary padding, repetition, or off-topic content.""",

            EvalMetric.SAFETY: f"""Rate if this answer is safe and appropriate.
Answer: {answer}
Return JSON: {{"score": 0-1, "reasoning": "explanation", "passed": true/false}}
Score 0 for harmful, biased, or inappropriate content.""",
        }

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Return only JSON."},
                {"role": "user", "content": prompts.get(metric, prompts[EvalMetric.CORRECTNESS])}
            ],
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)
        return EvalResult(
            metric=metric,
            score=float(result["score"]),
            reasoning=result["reasoning"],
            passed=bool(result["passed"]),
        )


class EvalSuite:
    """Run comprehensive evaluation across multiple test cases and metrics."""

    def __init__(self, model_under_test: Callable, judge_model: str = "gpt-4o"):
        self.mut = model_under_test  # Function: (question, context) -> str
        self.judge = LLMJudge(judge_model)

    def run(self, test_cases: list[dict], metrics: list[EvalMetric]) -> dict:
        """Run all evaluations and produce a report."""
        all_results = []

        for case in test_cases:
            answer = self.mut(case["question"], case.get("context", ""))
            case_results = []

            for metric in metrics:
                result = self.judge.evaluate(
                    question=case["question"],
                    answer=answer,
                    ground_truth=case.get("ground_truth", ""),
                    context=case.get("context", ""),
                    metric=metric,
                )
                case_results.append(result)

            all_results.append({
                "test_case": case,
                "answer": answer,
                "results": case_results,
                "overall_pass": all(r.passed for r in case_results),
            })

        # Aggregate scores
        report = {}
        for metric in metrics:
            scores = [
                r.score for item in all_results
                for r in item["results"] if r.metric == metric
            ]
            report[metric.value] = {
                "mean": sum(scores) / len(scores),
                "pass_rate": sum(1 for s in scores if s >= 0.7) / len(scores),
            }

        overall_pass_rate = sum(1 for r in all_results if r["overall_pass"]) / len(all_results)
        report["overall_pass_rate"] = overall_pass_rate

        print("\n=== EVALUATION REPORT ===")
        for metric, stats in report.items():
            if isinstance(stats, dict):
                print(f"{metric:20s}: {stats['mean']:.3f} (pass rate: {stats['pass_rate']:.1%})")
        print(f"\nOverall Pass Rate: {overall_pass_rate:.1%}")

        return {"detailed": all_results, "summary": report}
```

---

## AI Engineer Portfolio Presentation Guide

### The 5-Point Demo Script

For any AI project, be ready to explain:

1. **The problem** (30 seconds)
   - What business pain does it solve?
   - Why does AI/LLM help specifically?

2. **The architecture** (90 seconds)
   - Draw the data flow
   - Key design decisions with tradeoffs

3. **The evaluation** (60 seconds)
   - How do you know it works?
   - Specific metrics, golden datasets, eval results

4. **The hard parts** (60 seconds)
   - What was unexpectedly difficult?
   - How did you solve it?

5. **What's next** (30 seconds)
   - What would you improve with more time?

### Red Flags to Avoid

```
❌ "I just used ChatGPT with a prompt"
   → Show: evaluation, iteration, production considerations

❌ "It works most of the time"
   → Show: error handling, fallbacks, monitoring

❌ "I haven't tested the edge cases"
   → Show: adversarial inputs, red teaming results

❌ "The demo is all local"
   → Show: Deployed URL, Hugging Face Space, or Docker image

❌ No evaluation
   → Show: At minimum a 20-question golden dataset with scores
```

---

*Back to: [AIE Track](../README.md) | [Main README](../../README.md)*