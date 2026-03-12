# Autonomous Research Agent

> **Difficulty:** Advanced | **Time:** 2-3 days | **Track:** AI Engineer

## What You'll Build
An autonomous multi-step research agent that accepts a research topic, queries the web using Tavily search, synthesizes findings across multiple sources, and generates a well-structured markdown report — complete with citations, section headings, and an executive summary — through a Streamlit interface.

## Learning Objectives
- Design a plan-then-execute agentic loop with LangChain
- Integrate real-time web search via the Tavily API
- Build a sub-agent workflow (planner → researcher → writer)
- Structure and cite multi-source LLM outputs reliably
- Deploy the agent as an interactive Streamlit app with live progress

## Prerequisites
- LangChain agent and tool fundamentals
- A Tavily API key (free tier: 1000 searches/month)
- An OpenAI API key

## Tech Stack
- `langchain` / `langchain-community`: agent orchestration
- `tavily-python`: real-time web search API
- `openai`: GPT-4o for reasoning and writing
- `streamlit`: interactive research dashboard with live status
- `markdown2` / `pdfkit`: optional export to PDF or HTML

## Step-by-Step Guide

### Step 1: Configure Tools and API Clients
```python
# pip install langchain langchain-community openai tavily-python streamlit markdown2

import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool

os.environ["OPENAI_API_KEY"]  = "sk-..."
os.environ["TAVILY_API_KEY"]  = "tvly-..."

# Tavily search tool — returns top-k results with URL, title, and content snippet
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",          # deep crawl for richer content
    include_answer=True,
    include_raw_content=False,
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

print("Tools configured: Tavily search + GPT-4o")
print(f"Tavily search depth: advanced | max results per query: 5")
```

### Step 2: Research Planner Sub-Agent
```python
from langchain_core.prompts import ChatPromptTemplate

PLANNER_PROMPT = """You are a research planner. Given a research topic, generate a structured research plan.

Topic: {topic}

Output a JSON object with:
- "title": report title
- "sections": list of 4-6 section names (e.g., "Overview", "Key Trends", "Challenges")
- "search_queries": list of 6-10 specific web search queries that together will cover all sections
- "estimated_words": target word count for the final report (500-2000)

Return only valid JSON."""

def plan_research(topic: str) -> dict:
    """Generate a structured research plan for a given topic."""
    import json, re
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    chain = prompt | llm
    response = chain.invoke({"topic": topic}).content
    # Extract JSON block even if surrounded by markdown
    match = re.search(r'\{.*\}', response, re.DOTALL)
    plan = json.loads(match.group()) if match else {}
    print(f"Research plan: {len(plan.get('sections', []))} sections, "
          f"{len(plan.get('search_queries', []))} queries")
    return plan

# Example output:
example_plan = plan_research("The impact of large language models on software engineering")
print("Sections:", example_plan.get("sections", []))
```

### Step 3: Web Research Executor
```python
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class SearchResult:
    query: str
    url: str
    title: str
    content: str
    relevance_score: float = 0.0

@dataclass
class ResearchContext:
    topic: str
    plan: dict
    raw_results: list[SearchResult] = field(default_factory=list)
    synthesized_notes: dict[str, str] = field(default_factory=dict)

def execute_research(plan: dict, topic: str, delay_s: float = 0.5) -> ResearchContext:
    """Run all planned search queries and collect results."""
    ctx = ResearchContext(topic=topic, plan=plan)
    queries = plan.get("search_queries", [])

    for i, query in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] Searching: {query}")
        try:
            results = search_tool.invoke(query)
            for r in results:
                ctx.raw_results.append(SearchResult(
                    query=query,
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    content=r.get("content", ""),
                ))
            time.sleep(delay_s)   # respect rate limits
        except Exception as e:
            print(f"    Search error: {e}")

    print(f"Research complete: {len(ctx.raw_results)} results from {len(queries)} queries")
    return ctx

def deduplicate_results(ctx: ResearchContext) -> ResearchContext:
    """Remove duplicate URLs and overly similar content snippets."""
    seen_urls: set[str] = set()
    unique: list[SearchResult] = []
    for r in ctx.raw_results:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            unique.append(r)
    ctx.raw_results = unique
    print(f"After dedup: {len(ctx.raw_results)} unique sources")
    return ctx
```

### Step 4: Section Synthesis and Report Writing
```python
SYNTHESIS_PROMPT = """You are a research analyst. Synthesize the following search results into
a clear, informative section for a research report.

Section: {section_name}
Topic: {topic}

Search results:
{search_results}

Write 2-4 paragraphs that are factual, neutral, and cite sources inline as [Source: URL].
Do not hallucinate. If data is missing, say so."""

REPORT_PROMPT = """You are a technical writer. Assemble the following research notes into a
complete, well-formatted markdown research report.

Title: {title}
Sections provided: {section_names}
Notes per section:
{notes}

Format:
# {title}
**Executive Summary** (3-5 sentences)
## [Section 1]
...
## References
- numbered list of all cited URLs"""

def synthesize_section(
    section_name: str,
    topic: str,
    results: list[SearchResult],
    top_k: int = 6,
) -> str:
    """Use the LLM to synthesize raw search results into a report section."""
    snippets = "\n\n".join(
        f"[{r.title}]({r.url})\n{r.content[:600]}"
        for r in results[:top_k]
    )
    prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
    chain = prompt | llm
    return chain.invoke({
        "section_name": section_name,
        "topic": topic,
        "search_results": snippets,
    }).content

def write_full_report(ctx: ResearchContext) -> str:
    """Synthesize each section then assemble the final report."""
    sections = ctx.plan.get("sections", [])
    notes = {}

    for section in sections:
        print(f"  Writing section: {section}")
        relevant = [r for r in ctx.raw_results if section.lower() in r.query.lower()] or ctx.raw_results
        notes[section] = synthesize_section(section, ctx.topic, relevant)

    notes_text = "\n\n".join(f"### {k}\n{v}" for k, v in notes.items())
    prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)
    chain = prompt | llm
    report = chain.invoke({
        "title": ctx.plan.get("title", ctx.topic),
        "section_names": ", ".join(sections),
        "notes": notes_text,
    }).content
    return report
```

### Step 5: Streamlit Research Dashboard
```python
# research_app.py — run with: streamlit run research_app.py
import streamlit as st

def run_streamlit_app():
    st.set_page_config(page_title="Research Agent", layout="wide")
    st.title("Autonomous Research Agent")
    st.caption("Powered by GPT-4o + Tavily real-time search")

    topic = st.text_input("Research topic", placeholder="e.g., 'State of RAG in 2025'")
    depth = st.selectbox("Report depth", ["Brief (3 sections)", "Standard (5 sections)", "Deep (7 sections)"])
    run_btn = st.button("Start Research", type="primary")

    if run_btn and topic.strip():
        status = st.status("Planning research...", expanded=True)

        with status:
            st.write("Step 1/3: Generating research plan...")
            plan = plan_research(topic)
            st.write(f"Plan ready: {len(plan.get('sections', []))} sections, "
                     f"{len(plan.get('search_queries', []))} queries")

            st.write("Step 2/3: Executing web searches...")
            ctx = execute_research(plan, topic)
            ctx = deduplicate_results(ctx)
            st.write(f"Found {len(ctx.raw_results)} unique sources")

            st.write("Step 3/3: Writing report...")
            report = write_full_report(ctx)
            status.update(label="Research complete!", state="complete")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(report)
        with col2:
            st.download_button("Download Report (.md)", report, file_name="research_report.md")
            st.metric("Sources Used", len(ctx.raw_results))
            st.metric("Sections Written", len(ctx.plan.get("sections", [])))
            with st.expander("Raw Sources"):
                for r in ctx.raw_results[:10]:
                    st.write(f"- [{r.title}]({r.url})")

# run_streamlit_app()
print("Streamlit app defined. Run: streamlit run research_app.py")
```

## Expected Output
- A Streamlit dashboard with a topic input and live three-step progress display
- A structured markdown research report with executive summary, 4-6 sections, and inline citations
- A download button for the final report as a `.md` file
- Source metadata panel listing all URLs used in the research
- Console logs showing each search query and synthesis step

## Stretch Goals
- [ ] **Multi-agent debate:** Spawn two researcher agents with opposing viewpoints (e.g., "LLMs will replace programmers" vs. "LLMs will augment programmers"), have each research independently, then use a third "moderator" agent to synthesize both perspectives into a balanced report
- [ ] **Iterative deep-dive:** After the first report is generated, parse it to identify knowledge gaps and automatically generate follow-up search queries to fill them, then produce a revised version — repeat until no new material is found
- [ ] **PDF export with citations:** Export the finished report to a styled PDF using `pdfkit` + `weasyprint`, with a proper bibliography section formatted in APA or MLA style

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`