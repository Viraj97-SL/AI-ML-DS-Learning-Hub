# AI Engineer — Intermediate Phase

**Goal:** Build production-quality RAG systems, design AI agents, and deploy AI features to real users.

**Duration:** 2–3 months at 10–15 hrs/week
**Prerequisites:** AIE Beginner Phase complete

---

## Curriculum Overview

```
Week 1–2   → Advanced RAG (chunking, hybrid search, reranking)
Week 3–4   → LangChain deep dive (chains, LCEL, retrievers)
Week 5–6   → Vector Databases (Chroma, Pinecone, pgvector)
Week 7–8   → AI Agents (ReAct, tool use, function calling)
Week 9–10  → LlamaIndex for complex document QA
Week 11–12 → AI Evaluation (RAGAS, LangSmith, custom evals)
Week 13–14 → Production AI (rate limiting, observability, cost control)
```

---

## Week 1–2: Advanced RAG

Basic RAG (chunk → embed → retrieve → generate) works for simple cases. Production RAG requires much more.

### The RAG Quality Problem

```
Naive RAG failure modes:
1. Wrong chunks retrieved (bad embedding similarity)
2. Retrieved chunks miss context (chunk too small)
3. Retrieved chunks include noise (chunk too large)
4. Query-chunk mismatch (user asks question, doc states fact)
5. Multi-hop reasoning required (answer spans multiple docs)
6. Temporal issues (stale documents returned)
```

### Advanced Chunking Strategies

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# ── Strategy 1: Recursive Character Splitting ────────────────
# Best general-purpose strategy
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
)

# ── Strategy 2: Markdown-Aware Splitting ─────────────────────
# Use this for documentation, wikis, technical content
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ],
    strip_headers=False,
)

sample_markdown = """
# Introduction to Machine Learning

## Supervised Learning
Supervised learning uses labeled training data...

### Classification
Classification predicts discrete categories...

### Regression
Regression predicts continuous values...

## Unsupervised Learning
Unsupervised learning finds patterns without labels...
"""

md_chunks = md_splitter.split_text(sample_markdown)
for chunk in md_chunks:
    print(f"H1: {chunk.metadata.get('H1', 'N/A')}")
    print(f"H2: {chunk.metadata.get('H2', 'N/A')}")
    print(f"Content: {chunk.page_content[:100]}\n")

# ── Strategy 3: Semantic Chunking ───────────────────────────
# Splits on semantic similarity — best quality, slowest
embeddings = OpenAIEmbeddings()
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90  # Split when similarity drops below 90th percentile
)
# chunks = semantic_splitter.split_text(long_document)

# ── Strategy 4: Parent Document Retriever ───────────────────
# Retrieve small chunks for precision, return parent for context
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma

# Small chunks for embedding (precise retrieval)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# Large chunks for context (full context returned to LLM)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

vectorstore = Chroma(embedding_function=embeddings)
store = InMemoryStore()  # In prod: use Redis or PostgreSQL

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
# parent_retriever.add_documents(documents)
# docs = parent_retriever.invoke("your query")  # Returns parent chunks!
```

### Hybrid Search (Keyword + Semantic)

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Assume documents are loaded
# documents = loader.load()

# ── Dense retriever (semantic) ────────────────────────────────
vectorstore = Chroma.from_documents(documents, embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ── Sparse retriever (keyword/BM25) ─────────────────────────
# BM25 is great for exact matches: product codes, names, acronyms
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# ── Hybrid: combine both ─────────────────────────────────────
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]  # 40% keyword, 60% semantic
)

results = hybrid_retriever.invoke("What are the activation functions in neural networks?")
print(f"Retrieved {len(results)} documents")
for doc in results:
    print(f"  - {doc.page_content[:100]}...")
```

### Reranking for Better Precision

```python
# pip install flashrank
from flashrank import Ranker, RerankRequest
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# Step 1: Retrieve many candidates (top-20 is fine, precision matters less)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Step 2: Rerank to top-5 using a cross-encoder model (much more accurate than dot product)
compressor = FlashrankRerank(top_n=5)  # Returns top-5 most relevant

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

query = "How does attention mechanism work in transformers?"
docs = reranking_retriever.invoke(query)

print(f"After reranking, top {len(docs)} documents:")
for i, doc in enumerate(docs, 1):
    print(f"\n{i}. Score: {doc.metadata.get('relevance_score', 'N/A')}")
    print(f"   {doc.page_content[:200]}...")
```

### HyDE — Hypothetical Document Embeddings

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser

# The problem: user asks a QUESTION, but documents contain ANSWERS/FACTS
# These have different embedding representations → bad retrieval

# HyDE solution: generate a hypothetical document that ANSWERS the question,
# then use THAT as the search query (its embedding is closer to real documents)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

hyde_prompt = ChatPromptTemplate.from_template("""
Write a short, factual paragraph that would appear in a technical document
and would directly answer this question:

Question: {question}

Write only the paragraph, no preamble:
""")

def hyde_retriever(question: str, vectorstore, k: int = 4):
    """Retrieve documents using HyDE (Hypothetical Document Embeddings)."""
    # Generate hypothetical document
    chain = hyde_prompt | llm | StrOutputParser()
    hypothetical_doc = chain.invoke({"question": question})

    print(f"Hypothetical document:\n{hypothetical_doc}\n")

    # Use hypothetical doc as query (its embedding is closer to real docs)
    results = vectorstore.similarity_search(hypothetical_doc, k=k)
    return results

# Compare standard vs HyDE retrieval
query = "What are the main limitations of Transformer models?"
standard_results = vectorstore.similarity_search(query, k=4)
hyde_results = hyde_retriever(query, vectorstore, k=4)
```

---

## Week 3–4: LangChain Deep Dive (LCEL)

LCEL (LangChain Expression Language) is the modern way to compose AI components.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

# ── Basic LCEL Chain ─────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Be concise and specific."),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"role": "Python expert", "input": "What is a decorator?"})

# ── RAG Chain with LCEL ──────────────────────────────────────
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. Be specific and cite relevant parts.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n---\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    ])

rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How does backpropagation work?")
print(answer)

# ── Conversational RAG with Memory ──────────────────────────
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Step 1: Make retriever context-aware (use chat history to reformulate query)
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question which might reference
    the chat history, formulate a standalone question which can be understood without
    the chat history. Do NOT answer the question, just reformulate it if needed."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# Step 2: Create the QA chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the user's question based on the context below.
    Be accurate, helpful, and cite specific information.\n\n{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# Step 3: Combine into conversational RAG
conversational_rag = create_retrieval_chain(history_aware_retriever, qa_chain)

# Use it with conversation history
chat_history = []

def chat(question: str) -> str:
    result = conversational_rag.invoke({
        "input": question,
        "chat_history": chat_history
    })
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result["answer"]))
    return result["answer"]

print(chat("What is machine learning?"))
print(chat("Can you give me examples of that?"))  # "that" refers to ML from context
print(chat("What's the difference from traditional programming?"))
```

---

## Week 7–8: AI Agents

Agents let LLMs take actions — search the web, run code, call APIs, read files.

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from pydantic import BaseModel, Field
import datetime
import requests

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ── Define Custom Tools ───────────────────────────────────────
@tool
def get_current_datetime() -> str:
    """Returns the current date and time. Use this to answer time-related questions."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression safely.
    Input should be a valid Python math expression.
    Example: "2 ** 10 + 500" or "import math; math.sqrt(144)"
    """
    try:
        # Safe eval for math only
        import math
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather for")

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In production, use a real weather API
    # Here we simulate it
    mock_data = {
        "london": "Cloudy, 12°C, 80% humidity",
        "new york": "Sunny, 22°C, 45% humidity",
        "tokyo": "Partly cloudy, 18°C, 60% humidity",
    }
    return mock_data.get(city.lower(), f"Weather data not available for {city}")


# ── Standard Tools ────────────────────────────────────────────
search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))
python_repl = PythonREPLTool()

tools = [
    get_current_datetime,
    calculate,
    get_weather,
    Tool(name="web_search", func=search_tool.run,
         description="Search the web for current information on any topic."),
    Tool(name="wikipedia", func=wiki_tool.run,
         description="Look up factual information from Wikipedia."),
    Tool(name="python_repl", func=python_repl.run,
         description="Execute Python code. Use for calculations, data analysis, or any computation."),
]

# ── Create Agent ──────────────────────────────────────────────
# Pull the ReAct prompt from LangChain hub
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,         # Shows ReAct thinking process
    max_iterations=10,    # Prevent infinite loops
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# ── Run Agent Tasks ───────────────────────────────────────────
def run_agent(task: str):
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print("="*60)
    result = agent_executor.invoke({"input": task})
    print(f"\nFINAL ANSWER: {result['output']}")
    return result

# Test various tasks
run_agent("What time is it right now?")
run_agent("What is the square root of 2024 times pi?")
run_agent("Search the web and tell me about the latest developments in AI agents in 2024.")
run_agent("Write and run Python code to generate the first 10 Fibonacci numbers.")
```

### Function Calling (OpenAI)

```python
import json
from openai import OpenAI

client = OpenAI()

# Define tools in OpenAI's function calling format
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products in our catalog. Use this to find product details, prices, and availability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the product"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "books", "home", "all"],
                        "description": "Product category to filter by"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price in USD"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place an order for a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1},
                    "shipping_address": {"type": "string"}
                },
                "required": ["product_id", "quantity", "shipping_address"]
            }
        }
    }
]

# Simulate tool execution
def execute_tool(name: str, args: dict) -> str:
    if name == "search_products":
        return json.dumps([
            {"id": "P001", "name": "MacBook Pro 14\"", "price": 1999, "stock": 5},
            {"id": "P002", "name": "MacBook Air M2", "price": 1099, "stock": 12},
        ])
    elif name == "place_order":
        return json.dumps({"order_id": "ORD-12345", "status": "confirmed",
                           "estimated_delivery": "2024-03-15"})
    return "Tool not found"


def agent_with_function_calling(user_query: str):
    messages = [{"role": "user", "content": user_query}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        # If no tool call → done
        if response.choices[0].finish_reason == "stop":
            return message.content

        # Execute tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"🔧 Calling: {func_name}({func_args})")
                result = execute_tool(func_name, func_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

result = agent_with_function_calling("Find me a laptop under $1500 and order the cheapest one to 123 Main St, NYC.")
print(result)
```

---

## Week 11–12: AI Evaluation (RAGAS)

```python
# pip install ragas datasets
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
)
from datasets import Dataset

# Build test dataset
questions = [
    "What is the capital of France?",
    "How does backpropagation work?",
    "What are the main types of neural networks?",
]

ground_truths = [
    ["Paris is the capital of France."],
    ["Backpropagation computes gradients using the chain rule to train neural networks."],
    ["Main types include CNNs, RNNs, Transformers, and GANs."],
]

# Get answers from your RAG system
answers = []
contexts = []
for q in questions:
    # Your RAG chain here
    result = rag_chain.invoke(q)  # Your chain from earlier
    retrieved = retriever.invoke(q)
    answers.append(result)
    contexts.append([doc.page_content for doc in retrieved])

# Build evaluation dataset
eval_data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}
dataset = Dataset.from_dict(eval_data)

# Run RAGAS evaluation
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,        # Is the answer grounded in the context? (0-1)
        answer_relevancy,    # Is the answer relevant to the question? (0-1)
        context_recall,      # Does context cover the ground truth? (0-1)
        context_precision,   # Are retrieved chunks actually relevant? (0-1)
        answer_correctness,  # Is the answer factually correct? (0-1)
    ]
)

print(results.to_pandas())
print(f"\nOverall RAG Quality:")
print(f"  Faithfulness:      {results['faithfulness']:.3f}")
print(f"  Answer Relevancy:  {results['answer_relevancy']:.3f}")
print(f"  Context Recall:    {results['context_recall']:.3f}")
print(f"  Context Precision: {results['context_precision']:.3f}")
```

---

## Intermediate Phase Skills Checklist

- [ ] Implemented at least 3 different chunking strategies and compared them
- [ ] Built a hybrid search RAG (BM25 + semantic)
- [ ] Implemented reranking with a cross-encoder
- [ ] Built a conversational RAG with multi-turn memory
- [ ] Built an AI agent with at least 4 custom tools
- [ ] Used function calling to make an LLM take structured actions
- [ ] Evaluated a RAG system with RAGAS metrics
- [ ] Understand the difference between faithfulness and answer_relevancy

**Next:** [Advanced Phase →](../advanced/)