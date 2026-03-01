# AI Engineer — Beginner Phase

**Goal:** Call LLM APIs confidently, master prompt engineering, and ship your first AI-powered application.

**Duration:** 1–2 months at 8–12 hrs/week
**Prerequisites:** Python functions, classes, basic HTTP/APIs
**Good news:** You can build real, useful AI apps within your FIRST week.

---

## Curriculum Overview

```
Week 1    → How LLMs Work (intuition, not math)
Week 2    → LLM APIs (OpenAI + Anthropic)
Week 3    → Prompt Engineering (the most underrated skill)
Week 4    → Building a Chatbot with Memory
Week 5–6  → Basic RAG (chat with your own documents)
Week 7–8  → Ship It: Build and Deploy Your First AI App
```

---

## Week 1: How LLMs Work

### What is a Large Language Model?

An LLM is a neural network trained to predict the next token (roughly a word-piece) in a sequence. Through this simple objective, on massive amounts of text, these models develop remarkably broad capabilities.

**The key numbers to know:**
```
GPT-4o:         ~200 billion parameters
Claude Sonnet:  ~70 billion parameters (estimate)
Llama 3 70B:    70 billion parameters (open-source!)
GPT-3:          175 billion parameters (2020 breakthrough)
BERT:           340 million parameters (2018)
```

### Key Concepts Every AI Engineer Must Understand

#### Tokens
LLMs don't read words — they read **tokens** (sub-word pieces):
- "ChatGPT" → ["Chat", "GPT"] — 2 tokens
- "unbelievable" → ["un", "bel", "iev", "able"] — 4 tokens
- Rule of thumb: 1 token ≈ 0.75 words, or 100 tokens ≈ 75 words

**Why tokens matter for you:**
- Pricing: APIs charge per token
- Limits: Models have a context window (token limit)
- Performance: Fewer tokens = faster response

#### Context Window
The context window is how much text the model can "see" at once:
```
GPT-3.5-turbo:  16K tokens  (~12,000 words)
GPT-4o:         128K tokens (~96,000 words) — most books fit!
Claude Sonnet:  200K tokens (~150,000 words)
Gemini 1.5 Pro: 1M tokens   (~750,000 words)
```

**Implication:** For RAG systems, you need to fit retrieved documents + question + conversation history within this limit.

#### Temperature
Controls randomness/creativity in outputs:
```
temperature = 0.0  → Deterministic, always same output (best for factual tasks)
temperature = 0.7  → Balanced creativity (default for most use cases)
temperature = 1.0  → More creative, more variable
temperature = 2.0  → Very random, often incoherent (avoid)
```

#### Inference vs Training
- **Training:** Teaching the model (done by OpenAI/Anthropic, costs millions)
- **Inference:** Using the model to generate a response (what you do via API)
- **Fine-tuning:** Adapting a pre-trained model to your specific task

---

## Week 2: LLM APIs

### Setup
```bash
pip install openai anthropic python-dotenv
```

```python
# .env file (NEVER commit this!)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### OpenAI API

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================
# Basic completion
# ============================================
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful data science tutor."},
        {"role": "user", "content": "Explain overfitting in simple terms."}
    ],
    temperature=0.7,
    max_tokens=500,
)

print(response.choices[0].message.content)
print(f"\nTokens used: {response.usage.total_tokens}")
print(f"Cost estimate: ${response.usage.total_tokens * 0.000005:.6f}")

# ============================================
# Streaming (word-by-word output like ChatGPT)
# ============================================
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku about machine learning."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Newline at the end

# ============================================
# Structured output with JSON mode
# ============================================
import json

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": """
        Analyze this product review and return JSON with keys:
        sentiment (positive/negative/neutral),
        score (1-10),
        key_themes (list of strings),
        summary (one sentence)

        Review: "The new MacBook Pro is incredibly fast, but the price is steep.
        Battery life is excellent — lasted all day at a conference.
        The keyboard feels great but I wish the ports were on both sides."
        """}
    ]
)
analysis = json.loads(response.choices[0].message.content)
print(json.dumps(analysis, indent=2))

# ============================================
# Vision: Analyze an image
# ============================================
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image? Describe in detail."},
            {"type": "image_url", "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
            }}
        ]
    }]
)
print(response.choices[0].message.content)
```

### Anthropic Claude API

```python
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Basic message
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are an expert Python programmer. Give concise, practical advice.",
    messages=[
        {"role": "user", "content": "What are the top 5 Python performance tips?"}
    ]
)
print(message.content[0].text)

# Streaming
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=500,
    messages=[{"role": "user", "content": "Explain RAG in 5 sentences."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Multi-turn conversation
conversation = []

def chat(user_message, system="You are a helpful assistant."):
    conversation.append({"role": "user", "content": user_message})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=conversation
    )
    assistant_message = response.content[0].text
    conversation.append({"role": "assistant", "content": assistant_message})
    return assistant_message

# Example multi-turn
print(chat("Hi, I want to learn about neural networks."))
print(chat("Can you explain backpropagation?"))
print(chat("What's a good project to practice?"))
```

### API Cost Management

```python
# Calculate cost before sending expensive requests
def estimate_tokens(text):
    """Rough estimate: 1 token ≈ 4 characters"""
    return len(text) // 4

def estimate_cost(input_text, output_tokens=500, model="gpt-4o"):
    """Estimate API cost in USD."""
    costs = {
        "gpt-4o": {"input": 0.005, "output": 0.015},       # per 1K tokens
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
        "claude-haiku": {"input": 0.00025, "output": 0.00125},
    }
    if model not in costs:
        return "Unknown model"

    input_tokens = estimate_tokens(input_text)
    input_cost = (input_tokens / 1000) * costs[model]["input"]
    output_cost = (output_tokens / 1000) * costs[model]["output"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": f"${input_cost:.6f}",
        "output_cost": f"${output_cost:.6f}",
        "total_cost": f"${input_cost + output_cost:.6f}"
    }

print(estimate_cost("Explain quantum computing in detail" * 100, model="gpt-4o"))
```

---

## Week 3: Prompt Engineering

Prompt engineering is the skill of writing instructions that get LLMs to do what you want, reliably. It's more important than it sounds.

### The Fundamental Principle
> LLMs are autocomplete engines trained on human text. The best prompts are the ones that most resemble text that a knowledgeable human would write before the desired answer.

### Technique 1: Zero-Shot Prompting
```python
# Bad prompt — ambiguous
prompt = "Summarize this."

# Good prompt — specific and clear
prompt = """
Summarize the following research paper abstract in 3 bullet points.
Each bullet should be one sentence. Focus on: method, results, and implications.

Abstract:
{abstract_text}
"""
```

### Technique 2: Few-Shot Prompting
```python
few_shot_prompt = """
Classify the sentiment of customer reviews as: POSITIVE, NEGATIVE, or NEUTRAL.

Examples:
Review: "Best laptop I've ever owned! Fast, beautiful screen, amazing battery."
Sentiment: POSITIVE

Review: "It stopped working after 2 weeks. Support was unhelpful."
Sentiment: NEGATIVE

Review: "It's okay. Does what I need but nothing special."
Sentiment: NEUTRAL

Now classify:
Review: "Shipping was slower than expected but the product quality is excellent."
Sentiment:"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": few_shot_prompt}],
    temperature=0,  # Zero temp for classification tasks
    max_tokens=20
)
print(response.choices[0].message.content)  # → POSITIVE
```

### Technique 3: Chain-of-Thought (CoT)
```python
# Standard prompting (often fails on math/logic)
bad_prompt = "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?"

# Chain-of-thought prompting (works much better)
cot_prompt = """
A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball.
How much does the ball cost?

Let's think step by step:
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": cot_prompt}],
    temperature=0
)
print(response.choices[0].message.content)
# → Ball = $0.05, Bat = $1.05, Total = $1.10 ✓
```

### Technique 4: System Prompts (Persona + Constraints)
```python
system_prompt = """
You are an expert Python code reviewer working at a top tech company.

Your role:
- Review code for correctness, efficiency, and Pythonic style
- Point out security vulnerabilities
- Suggest improvements with clear explanations
- Always provide corrected/improved code

Format your response as:
## Issues Found
[List issues with severity: CRITICAL/HIGH/MEDIUM/LOW]

## Improved Code
[Corrected code with comments]

## Summary
[2-3 sentence overall assessment]

Be direct and specific. Avoid vague feedback.
"""

code_to_review = """
def get_user_data(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    result = db.execute(query)
    password = result["password"]
    return {"id": user_id, "password": password}
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Review this code:\n\n```python\n{code_to_review}\n```"}
    ]
)
print(response.choices[0].message.content)
```

### Technique 5: Structured Output with Pydantic

```python
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Literal
import json

client = OpenAI()

class JobRequirement(BaseModel):
    skill: str
    importance: Literal["must-have", "nice-to-have"]
    years_experience: int | None

class JobAnalysis(BaseModel):
    job_title: str
    company_type: Literal["startup", "scale-up", "enterprise", "unknown"]
    seniority: Literal["junior", "mid", "senior", "staff", "principal"]
    required_skills: List[JobRequirement]
    estimated_salary_range: str
    red_flags: List[str]
    green_flags: List[str]

job_posting = """
Senior ML Engineer — AI-First Fintech Startup

We're looking for a senior ML engineer to join our small but mighty team.
You'll own model training, deployment, and monitoring pipelines.

Requirements:
- 5+ years Python
- PyTorch or TensorFlow (must)
- MLflow or similar (nice to have)
- Kubernetes experience (must)
- Competitive salary: $160k-$200k base + equity

Perks: unlimited PTO, flexible remote, weekly team lunches
"""

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": f"Analyze job postings. Return JSON matching this schema: {JobAnalysis.model_json_schema()}"},
        {"role": "user", "content": f"Analyze this job posting:\n\n{job_posting}"}
    ]
)

analysis = JobAnalysis.model_validate_json(response.choices[0].message.content)
print(f"Title: {analysis.job_title}")
print(f"Level: {analysis.seniority} at {analysis.company_type}")
print(f"Salary: {analysis.estimated_salary_range}")
print(f"Must-have skills: {[s.skill for s in analysis.required_skills if s.importance == 'must-have']}")
```

---

## Week 4: Chatbot with Memory

```python
from openai import OpenAI
from datetime import datetime
from typing import List, Dict
import json

client = OpenAI()

class ConversationMemory:
    """Manages conversation history with token budget."""

    def __init__(self, max_messages: int = 20, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self.messages: List[Dict] = []
        self.max_messages = max_messages

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def _trim_history(self):
        """Keep only the most recent messages to stay within context."""
        if len(self.messages) > self.max_messages:
            # Keep the first 2 messages (important context) and last N
            self.messages = self.messages[:2] + self.messages[-(self.max_messages-2):]

    def get_messages(self) -> List[Dict]:
        return self.messages

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump({"system": self.system_prompt, "messages": self.messages}, f)

    def load(self, filepath: str):
        with open(filepath) as f:
            data = json.load(f)
            self.system_prompt = data.get("system", "")
            self.messages = data.get("messages", [])


class AIAssistant:
    """A configurable AI assistant with persistent memory."""

    def __init__(self, name: str = "Assistant", model: str = "gpt-4o-mini",
                 system_prompt: str = None):
        self.name = name
        self.model = model
        self.memory = ConversationMemory(
            system_prompt=system_prompt or f"You are {name}, a helpful AI assistant."
        )

    def chat(self, user_input: str) -> str:
        self.memory.add_user_message(user_input)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.memory.system_prompt},
                *self.memory.get_messages()
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        reply = response.choices[0].message.content
        self.memory.add_assistant_message(reply)
        return reply

    def run_cli(self):
        """Run an interactive CLI chat session."""
        print(f"\n{'='*50}")
        print(f"  {self.name} — Type 'quit' to exit, 'clear' to reset")
        print(f"{'='*50}\n")

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print(f"\n{self.name}: Goodbye! 👋")
                break
            if user_input.lower() == "clear":
                self.memory = ConversationMemory(system_prompt=self.memory.system_prompt)
                print(f"{self.name}: Conversation cleared!")
                continue

            response = self.chat(user_input)
            print(f"\n{self.name}: {response}\n")


# Create a specialized assistant
ds_tutor = AIAssistant(
    name="DataSciBot",
    model="gpt-4o-mini",  # Cheaper for practice
    system_prompt="""You are DataSciBot, an expert data science tutor.
    Your teaching style:
    - Use concrete, real-world examples
    - Include short code snippets when helpful
    - Ask follow-up questions to check understanding
    - Celebrate when learners grasp concepts
    - Break complex topics into digestible pieces
    - Reference specific Python libraries and their documentation
    """
)

# Test with a few messages
print(ds_tutor.chat("Hi! I'm learning data science. Where should I start?"))
print(ds_tutor.chat("What's the difference between supervised and unsupervised learning?"))
print(ds_tutor.chat("Can you give me a simple example of each?"))

# Or run interactively:
# ds_tutor.run_cli()
```

---

## Week 5–6: Basic RAG — Chat with Your Documents

RAG (Retrieval-Augmented Generation) lets you ask LLMs questions about your own documents, without fine-tuning.

```
Document(s)
    ↓ Split into chunks
Chunks → Embedding model → Vectors
    ↓ Store in vector database
                                    ← User query
                                    ← Query embedding
                                    ← Search vector DB
                                    ← Top-k relevant chunks
[Query + Context chunks] → LLM → Answer
```

```python
# Install dependencies
# pip install langchain langchain-openai langchain-community chromadb pypdf

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# ============================================
# Step 1: Load documents
# ============================================
# Option A: Load a PDF
# loader = PyPDFLoader("your_document.pdf")

# Option B: Load text files
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs/", glob="**/*.txt")

# Option C: Load from URL
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader([
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
])
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# ============================================
# Step 2: Split into chunks
# ============================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Characters per chunk
    chunk_overlap=200,   # Overlap to preserve context across chunks
    separators=["\n\n", "\n", ".", "!", "?", ",", " "],
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
print(f"Sample chunk: {chunks[0].page_content[:200]}...")

# ============================================
# Step 3: Embed and store in vector database
# ============================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Persist to disk so you don't re-embed every time
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # Saves to disk
)
print("Vector store created!")

# Load existing vector store (next time you run)
# vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embeddings
# )

# ============================================
# Step 4: Build the RAG chain
# ============================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Custom prompt to prevent hallucinations
prompt_template = """You are a helpful assistant. Answer the question based ONLY
on the following context. If the context doesn't contain the answer, say
"I don't have enough information to answer that based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),  # Top 4 chunks
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,   # See where answers come from!
)

# ============================================
# Step 5: Query your documents
# ============================================
def ask(question: str):
    result = qa_chain.invoke({"query": question})
    print(f"\n❓ Question: {question}")
    print(f"\n💬 Answer: {result['result']}")
    print(f"\n📚 Sources:")
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get("source", "Unknown")
        print(f"  {i}. {source}: {doc.page_content[:100]}...")

ask("What is machine learning?")
ask("What are the main types of machine learning?")
ask("What is backpropagation?")
```

---

## Week 7–8: Ship Your First AI App

### Build a Streamlit AI App

```python
# app.py — Run with: streamlit run app.py
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Study Buddy")
st.markdown("*Your personalized data science tutor*")

# ============================================
# Sidebar — Configuration
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7)
    subject = st.selectbox("Subject", [
        "Data Science", "Machine Learning", "Python",
        "Statistics", "SQL", "AI Engineering"
    ])
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ============================================
# Chat Interface
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new input
if prompt := st.chat_input("Ask me anything about data science..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"""
                    You are an expert {subject} tutor.
                    Use concrete examples, code snippets when relevant,
                    and ask follow-up questions to check understanding.
                    Keep responses focused and actionable.
                    """},
                    *[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages]
                ],
                temperature=temperature,
                stream=True,
            )
            response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
```

### Deploy to Streamlit Cloud (Free)

1. Push your app to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `OPENAI_API_KEY` in Secrets
5. Click Deploy — live in 2 minutes!

---

## Beginner Phase Skills Checklist

- [ ] Understand tokens, context window, temperature conceptually
- [ ] Can call OpenAI API and get a response
- [ ] Can call Anthropic Claude API
- [ ] Know how to use system prompts effectively
- [ ] Have written zero-shot, few-shot, and chain-of-thought prompts
- [ ] Have built a multi-turn chatbot with conversation history
- [ ] Have built a basic RAG system (documents → embeddings → query)
- [ ] Have deployed at least one AI app (Streamlit Cloud, HF Spaces, etc.)
- [ ] Understand cost management (tokens × price)

**Next:** [Intermediate Phase →](../intermediate/)
