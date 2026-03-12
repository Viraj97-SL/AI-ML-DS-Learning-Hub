# Build a Q&A Bot with Document Upload

> **Difficulty:** Intermediate | **Time:** 1-2 days | **Track:** AI Engineer

## What You'll Build
A fully functional Q&A chatbot that accepts PDF uploads, indexes their contents into a vector store, and answers natural-language questions with grounded, citation-backed responses using a retrieval-augmented generation (RAG) pipeline.

## Learning Objectives
- Build a RAG pipeline from document ingestion to answer generation
- Chunk, embed, and index documents with FAISS
- Design a multi-turn conversational interface with memory
- Expose the chatbot through an interactive Gradio UI
- Evaluate answer quality and citation accuracy

## Prerequisites
- Python and basic LangChain familiarity
- An OpenAI API key (or compatible endpoint)
- Understanding of embeddings and vector similarity

## Tech Stack
- `langchain` / `langchain-community`: RAG pipeline orchestration
- `openai`: LLM and embedding calls
- `faiss-cpu`: local vector similarity search
- `pypdf`: PDF parsing and text extraction
- `gradio`: interactive web UI
- `tiktoken`: token counting and chunk size estimation

## Step-by-Step Guide

### Step 1: Install Dependencies and Configure
```python
# pip install langchain langchain-community openai faiss-cpu pypdf gradio tiktoken

import os
from pathlib import Path
from typing import Optional

# Set your API key
os.environ["OPENAI_API_KEY"] = "sk-..."   # or load from .env

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

print("All imports successful.")
print(f"LangChain ready | Model: gpt-4o-mini | Embeddings: text-embedding-3-small")
```

### Step 2: Document Ingestion and Chunking
```python
from langchain.schema import Document

def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(pdf_path)
    raw_pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_pages)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["source_file"] = Path(pdf_path).name

    print(f"Loaded '{Path(pdf_path).name}': {len(raw_pages)} pages → {len(chunks)} chunks")
    return chunks

# Demo: create a mock PDF text file for testing
import tempfile, textwrap
sample_text = textwrap.dedent("""
    Attention Is All You Need
    Vaswani et al., 2017

    Abstract
    The dominant sequence transduction models are based on complex recurrent or convolutional
    neural networks. We propose a new simple network architecture, the Transformer, based
    solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

    1. Introduction
    Recurrent neural networks, long short-term memory and gated recurrent networks have been
    firmly established as state-of-the-art approaches in sequence modeling and transduction
    problems such as language modeling and machine translation.
""")
with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
    f.write(sample_text)
    tmp_path = f.name
print(f"Demo file written to: {tmp_path}")
```

### Step 3: Build the FAISS Vector Store
```python
def build_vector_store(
    chunks: list[Document],
    embedding_model: str = "text-embedding-3-small",
) -> FAISS:
    """Embed document chunks and index them in FAISS."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store built: {vector_store.index.ntotal} vectors indexed")
    return vector_store

def add_documents_to_store(
    vector_store: FAISS,
    new_chunks: list[Document],
) -> FAISS:
    """Incrementally add new documents without rebuilding the store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store.add_documents(new_chunks)
    print(f"Store updated: {vector_store.index.ntotal} total vectors")
    return vector_store

def save_and_load_store(vector_store: FAISS, path: str = "./faiss_index") -> FAISS:
    """Persist the vector store to disk and reload it."""
    vector_store.save_local(path)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    reloaded = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print(f"Store saved to '{path}' and reloaded ({reloaded.index.ntotal} vectors)")
    return reloaded
```

### Step 4: Assemble the Conversational RAG Chain
```python
def build_qa_chain(
    vector_store: FAISS,
    model_name: str = "gpt-4o-mini",
    top_k: int = 4,
) -> ConversationalRetrievalChain:
    """Create a conversational RAG chain with memory."""
    llm = ChatOpenAI(model=model_name, temperature=0)

    retriever = vector_store.as_retriever(
        search_type="mmr",              # Maximal Marginal Relevance for diversity
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    print(f"RAG chain ready | LLM: {model_name} | top-k: {top_k} | search: MMR")
    return chain

def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """Ask a question and return the answer with source citations."""
    result = chain.invoke({"question": question})
    answer = result["answer"]
    sources = [
        f"{doc.metadata.get('source_file', 'unknown')}, p.{doc.metadata.get('page', '?')}"
        for doc in result.get("source_documents", [])
    ]
    return {"answer": answer, "sources": list(set(sources))}
```

### Step 5: Build the Gradio Interface
```python
import gradio as gr

def build_gradio_app(qa_chain: ConversationalRetrievalChain):
    """Wrap the RAG chain in an interactive Gradio chat UI."""

    def upload_and_index(file) -> str:
        nonlocal qa_chain
        chunks = load_and_chunk_pdf(file.name)
        store = build_vector_store(chunks)
        qa_chain = build_qa_chain(store)
        return f"Indexed {len(chunks)} chunks from '{Path(file.name).name}'. Ask me anything!"

    def respond(message: str, history: list) -> str:
        if not message.strip():
            return "Please enter a question."
        result = ask(qa_chain, message)
        response = result["answer"]
        if result["sources"]:
            response += f"\n\n**Sources:** {', '.join(result['sources'])}"
        return response

    with gr.Blocks(title="PDF Q&A Bot") as demo:
        gr.Markdown("## PDF Q&A Bot\nUpload a PDF, then ask questions about its content.")
        with gr.Row():
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            status_box = gr.Textbox(label="Status", interactive=False)
        file_input.upload(upload_and_index, inputs=file_input, outputs=status_box)
        gr.ChatInterface(respond)

    return demo

# Launch: demo = build_gradio_app(qa_chain); demo.launch()
print("Gradio app defined. Call build_gradio_app(chain).launch() to start the UI.")
```

## Expected Output
- A running Gradio web app accepting PDF file uploads
- Chunked, embedded, and indexed document content in a FAISS vector store
- Multi-turn conversational Q&A with accurate, grounded answers
- Source citations (page number and file name) appended to each response
- Persistent vector store that survives restarts via `save_local` / `load_local`

## Stretch Goals
- [ ] **Multi-document support:** Allow uploading multiple PDFs simultaneously; display a sidebar listing all indexed files and their chunk counts
- [ ] **Answer grounding score:** After each answer, compute the cosine similarity between the answer embedding and the retrieved chunks, and display a confidence percentage to the user
- [ ] **Streaming responses:** Switch the Gradio interface to streaming mode using `ChatOpenAI(streaming=True)` and LangChain's `StreamingStdOutCallbackHandler` so users see tokens arrive in real time

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`
