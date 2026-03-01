# AI Engineer Interview Preparation

> AI Engineer interviews are a mix of LLM system design, coding, and practical AI judgment. The field is new enough that interviewers value demonstrated experience more than textbook answers.

---

## AIE Interview Structure (Varies More Than Other Roles)

```
Round 1: Recruiter Screen (30 min) → Background, experience with LLMs
Round 2: Technical Screen (45-60 min) → Build a simple LLM app live
Round 3: AI System Design (60 min) → Design an AI-powered feature
Round 4: Prompt Engineering Challenge (45 min) → Improve a broken prompt
Round 5: Coding + LLM Integration (60 min) → Build a RAG pipeline
Round 6: Behavioral (45 min) → Ship fast, iterate, judgment under uncertainty
```

---

## LLM Fundamentals Q&A

**Q1: Explain how transformers work at a high level.**

> A transformer processes input tokens through layers of **self-attention** and **feed-forward** networks.
>
> Self-attention lets each token attend to every other token, computing relevance scores. For token i attending to token j: score = Q_i · K_j / √d_k. These scores are softmaxed and used to weight the value vectors V_j, producing a context-aware representation.
>
> Positional encodings are added because transformers have no inherent sense of sequence order. The FFN after attention transforms each token's representation independently.
>
> GPT-style models use **causal (masked) self-attention** so each token only attends to previous tokens — enabling autoregressive generation.

**Q2: What is the difference between GPT and BERT?**

| Aspect | GPT (Decoder-only) | BERT (Encoder-only) |
|--------|-------------------|---------------------|
| Architecture | Causal masked attention | Bidirectional attention |
| Training objective | Next token prediction | Masked language modeling + NSP |
| Good for | Generation, completion, chat | Classification, embeddings, Q&A |
| Examples | GPT-4, Claude, Llama, Mistral | BERT, RoBERTa, DistilBERT |
| Context | Only sees left context | Sees full context both directions |

**Q3: What is RAG and why is it better than pure fine-tuning for knowledge-intensive tasks?**

> **RAG (Retrieval-Augmented Generation)** retrieves relevant documents at query time and injects them into the LLM's context.
>
> **Advantages over fine-tuning:**
> - **Updatable:** Add new documents without retraining
> - **Verifiable:** Sources can be cited and verified
> - **Cost-effective:** No GPU training required
> - **Factual grounding:** Reduces hallucination by grounding in retrieved text
>
> **When fine-tuning is better:**
> - Teaching new behaviors, style, or format (not facts)
> - Domain-specific language patterns that differ from pre-training
> - When latency is critical (no retrieval step)
> - Teaching the model to use specialized tools/APIs

**Q4: What is the difference between these temperature settings and when would you use each?**

```
temperature=0:   Deterministic, always picks highest probability token
                 Use for: Classification, structured extraction, math, code generation

temperature=0.3: Slightly creative, mostly stable
                 Use for: Summarization, formal writing, technical Q&A

temperature=0.7: Balanced creativity (ChatGPT default)
                 Use for: General conversation, content ideas, brainstorming

temperature=1.0: More diverse, matches training distribution
                 Use for: Creative writing, story generation

temperature>1.5: Very random, often incoherent
                 Use for: Almost never in production
```

**Q5: Your RAG system has poor retrieval quality. How do you debug and fix it?**

```
Diagnosis checklist:
1. Is the embedding model good for your domain?
   → Test: embed query + top doc, measure cosine similarity
   → Fix: Try domain-specific embedding models

2. Are chunks the right size?
   → Too small: lose context
   → Too large: dilute relevance signal
   → Fix: Try 500/1000/1500 chars, measure RAGAS context_precision

3. Is the query well-formed?
   → Conversational queries ≠ document language
   → Fix: HyDE (generate hypothetical document), query expansion

4. Is it a recall problem (right doc not retrieved)?
   → Test: Is correct doc in top-100 results?
   → Fix: Increase k, use hybrid search (BM25 + semantic)

5. Is it a precision problem (noisy docs in top-k)?
   → Test: Are top-5 docs actually relevant?
   → Fix: Add reranking cross-encoder

6. Are documents pre-processed correctly?
   → Check: headers removed? tables extracted? PDF artifacts?
   → Fix: Better document parsing (pypdf2 → pymupdf or docling)
```

---

## Prompt Engineering Challenge

**Q6: The following prompt is producing inconsistent, often wrong outputs. Fix it.**

**Broken prompt:**
```
You are a helpful assistant. Look at this customer complaint and tell me what to do.

Complaint: "I ordered the blue shirt 3 weeks ago and it still hasn't arrived.
The tracking says it's been in Memphis for 10 days. I'm very frustrated!"
```

**Problems:** No output format, no role definition, ambiguous "tell me what to do", no tone guidance.

**Fixed prompt:**
```
You are a customer service specialist for an e-commerce company.
Your tone is empathetic, professional, and solution-oriented.

Analyze the following customer complaint and provide a structured response.

COMPLAINT:
{complaint_text}

Respond with EXACTLY this JSON format:
{{
  "urgency": "high | medium | low",
  "issue_category": "shipping | product | billing | account | other",
  "sentiment": "angry | frustrated | neutral | happy",
  "recommended_actions": [
    "Action 1 (specific and actionable)",
    "Action 2"
  ],
  "draft_response_to_customer": "Your empathetic, professional reply here",
  "escalate_to_human": true or false,
  "escalation_reason": "Reason if escalate_to_human is true, else null"
}}
```

**Test this with:**
```python
from openai import OpenAI
import json

client = OpenAI()

complaint = "I ordered the blue shirt 3 weeks ago and it still hasn't arrived. The tracking says it's been in Memphis for 10 days. I'm very frustrated!"

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "You are a customer service specialist. Return JSON only."},
        {"role": "user", "content": f"""
        Analyze this complaint and respond with JSON:

        Complaint: {complaint}

        JSON format:
        {{
          "urgency": "high | medium | low",
          "issue_category": "shipping | product | billing | account",
          "recommended_actions": ["action 1", "action 2"],
          "draft_response": "empathetic reply",
          "escalate": true/false
        }}
        """}
    ],
    temperature=0
)
print(json.dumps(json.loads(response.choices[0].message.content), indent=2))
```

---

## Live Coding: Build a RAG Pipeline

**Typical ask:** "Build a document Q&A system in 30-45 minutes."

**Template to have memorized:**

```python
# The minimal RAG pipeline — know this cold
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Embed documents
def build_index(texts: list[str], persist_dir: str = "./chroma_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents(texts)
    vectorstore = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=persist_dir
    )
    return vectorstore

# 2. Query with context
def answer_question(question: str, vectorstore, k: int = 4) -> str:
    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Generate answer
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context. "
             "If unsure, say 'I don't know based on the available documents.'"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content

# Usage
texts = [
    "Python was created by Guido van Rossum and first released in 1991...",
    "NumPy is the fundamental package for scientific computing in Python...",
]
vs = build_index(texts)
print(answer_question("Who created Python?", vs))
```

---

## AI System Design Questions

**Q7: Design a customer support AI agent for an e-commerce platform.**

```
Requirements:
- Handle 10K queries/day
- Answer order status, returns, product questions
- Escalate complex issues to humans
- Should not make things up (no hallucination)

Architecture:

User message
      ↓
Intent Classifier (classify: order_status | return | product_info | other)
      ↓
Router → dispatches to appropriate handler:

  ORDER STATUS HANDLER:
    - Extract order_id from message (regex or LLM)
    - Fetch order from Order DB via tool call
    - Generate response using actual order data

  RETURN HANDLER:
    - Check return policy (RAG over policy documents)
    - Check if order is eligible (rule: within 30 days)
    - Generate step-by-step return instructions

  PRODUCT INFO HANDLER:
    - Retrieve product info from catalog (RAG)
    - Answer questions about specs, compatibility, etc.

  ESCALATION HANDLER:
    - Triggered when: intent=other, sentiment=very_angry, or
      any handler responds with low confidence
    - Log full conversation for human agent

Anti-hallucination measures:
  - Ground all responses in retrieved data
  - For order status: ALWAYS use tool output, never generate status
  - Confidence scoring: if LLM unsure → escalate
  - Return policy: quote exact policy text, not LLM-generated summary
```

**Q8: How would you evaluate an LLM system before shipping it?**

```
1. Offline Evaluation (before deployment)

   a. Unit tests — test specific behaviors:
      - "Never reveal competitor names" → test 50 adversarial prompts
      - "Always respond in user's language" → test 10 languages
      - "Return JSON only when asked" → test format consistency

   b. RAGAS metrics (for RAG systems):
      - Faithfulness > 0.85  (answer grounded in context)
      - Context Precision > 0.80  (retrieved chunks relevant)
      - Answer Relevancy > 0.85  (answer relevant to question)

   c. Golden dataset evaluation:
      - 100-500 manually curated Q&A pairs
      - Use GPT-4 as judge (LLM-as-judge pattern)
      - Track regression across model updates

2. Online Evaluation (after deployment)

   a. A/B testing: new vs old model on 5% of traffic
   b. User feedback signals: thumbs up/down, corrections, escalations
   c. Latency and cost monitoring (token counts, API costs)
   d. Safety monitoring: log and review flagged outputs
   e. Hallucination detection: cross-reference with source documents

3. Human Evaluation (for high-stakes changes)

   - Blind preference testing (humans rate A vs B without knowing which is which)
   - Red teaming: try to make the model fail
   - Domain expert review (medical, legal, financial)
```

---

## Behavioral Questions for AI Engineers

- Tell me about an AI feature you shipped. What was the hardest part?
- How do you handle hallucinations in production AI systems?
- Describe a time a prompt worked in testing but failed in production.
- How do you balance AI speed of innovation with safety and reliability?
- Tell me about a time you had to explain an AI system failure to a stakeholder.

---

## Key Topics to Study Deeply

| Topic | Resources |
|-------|-----------|
| Transformer architecture | Karpathy's "Neural Networks: Zero to Hero" (YouTube) |
| Prompt engineering | DAIR.AI Prompt Engineering Guide (GitHub) |
| RAG techniques | LlamaIndex and LangChain docs + "Advanced RAG" blog posts |
| LLM evaluation | RAGAS paper + DeepEval docs |
| LLM fine-tuning | Hugging Face PEFT docs + Sebastian Raschka's blog |
| AI safety & alignment | Anthropic research blog + AI Safety Fundamentals course |

---

*Back to: [Interview Prep](../) | [AIE Track](../../03_AI_Engineer/) | [Main README](../../README.md)*
