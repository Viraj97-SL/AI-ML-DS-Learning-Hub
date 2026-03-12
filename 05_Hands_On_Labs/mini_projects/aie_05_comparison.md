# LLM Benchmark & Comparison Tool

> **Difficulty:** Advanced | **Time:** 2-3 days | **Track:** AI Engineer

## What You'll Build
A systematic benchmarking framework that evaluates and compares multiple LLMs (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) across standardized tasks — including reasoning, coding, summarization, and factual recall — and visualizes the results in an interactive Plotly dashboard.

## Learning Objectives
- Design a provider-agnostic LLM evaluation harness
- Measure latency, cost, token counts, and quality scores side-by-side
- Implement automated scoring rubrics (exact match, LLM-as-judge, ROUGE)
- Visualize multi-dimensional benchmark results with Plotly
- Export reproducible benchmark reports to HTML and CSV

## Prerequisites
- API keys for OpenAI, Anthropic, and Google Gemini
- Familiarity with async Python and pandas
- Basic understanding of NLP evaluation metrics

## Tech Stack
- `openai`: GPT-4o API client
- `anthropic`: Claude 3.5 Sonnet API client
- `google-generativeai`: Gemini 1.5 Pro API client
- `pandas`: results aggregation and analysis
- `plotly`: interactive benchmark visualizations
- `rouge-score`: ROUGE-L for summarization evaluation
- `asyncio` / `aiohttp`: concurrent API calls for faster benchmarking

## Step-by-Step Guide

### Step 1: Provider-Agnostic LLM Client
```python
# pip install openai anthropic google-generativeai pandas plotly rouge-score

import os, time, asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable
import openai
import anthropic
import google.generativeai as genai

@dataclass
class LLMResponse:
    provider: str
    model: str
    prompt: str
    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str] = None

# Cost per 1M tokens (input / output) — update as pricing changes
PRICING = {
    "gpt-4o":                     (5.00,  15.00),
    "claude-3-5-sonnet-20241022": (3.00,  15.00),
    "gemini-1.5-pro":             (3.50,  10.50),
}

def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING:
        return 0.0
    in_price, out_price = PRICING[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000

print("Providers: OpenAI GPT-4o | Anthropic Claude 3.5 Sonnet | Google Gemini 1.5 Pro")
print("Metrics tracked: latency, input/output tokens, cost, response quality")
```

### Step 2: Provider Clients with Uniform Interface
```python
def call_openai(prompt: str, model: str = "gpt-4o", max_tokens: int = 512) -> LLMResponse:
    """Call OpenAI and record timing and token usage."""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        latency = (time.perf_counter() - start) * 1000
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
        return LLMResponse(
            provider="OpenAI", model=model, prompt=prompt,
            response=resp.choices[0].message.content,
            latency_ms=latency, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=compute_cost(model, in_tok, out_tok),
        )
    except Exception as e:
        return LLMResponse("OpenAI", model, prompt, "", 0, 0, 0, 0, error=str(e))

def call_claude(prompt: str, model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 512) -> LLMResponse:
    """Call Anthropic Claude and record timing and token usage."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    start = time.perf_counter()
    try:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - start) * 1000
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        return LLMResponse(
            provider="Anthropic", model=model, prompt=prompt,
            response=resp.content[0].text,
            latency_ms=latency, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=compute_cost(model, in_tok, out_tok),
        )
    except Exception as e:
        return LLMResponse("Anthropic", model, prompt, "", 0, 0, 0, 0, error=str(e))

def call_gemini(prompt: str, model: str = "gemini-1.5-pro", max_tokens: int = 512) -> LLMResponse:
    """Call Google Gemini and record timing and token usage."""
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    start = time.perf_counter()
    try:
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
        latency = (time.perf_counter() - start) * 1000
        in_tok = resp.usage_metadata.prompt_token_count
        out_tok = resp.usage_metadata.candidates_token_count
        return LLMResponse(
            provider="Google", model=model, prompt=prompt,
            response=resp.text,
            latency_ms=latency, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=compute_cost(model, in_tok, out_tok),
        )
    except Exception as e:
        return LLMResponse("Google", model, prompt, "", 0, 0, 0, 0, error=str(e))

print("Three provider clients defined with uniform LLMResponse output.")
```

### Step 3: Benchmark Task Suite
```python
from dataclasses import dataclass
from typing import Any

@dataclass
class BenchmarkTask:
    task_id: str
    category: str       # reasoning, coding, summarization, factual, safety
    prompt: str
    reference_answer: Optional[str] = None
    scorer: Optional[Callable] = None   # fn(response, reference) -> float 0-1

def exact_match_score(response: str, reference: str) -> float:
    return 1.0 if reference.strip().lower() in response.strip().lower() else 0.0

def rouge_l_score(response: str, reference: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, response)["rougeL"].fmeasure

BENCHMARK_TASKS = [
    BenchmarkTask("reason_01", "reasoning",
        "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
        reference_answer="5 cents", scorer=exact_match_score),
    BenchmarkTask("reason_02", "reasoning",
        "If all Bloops are Razzles, and all Razzles are Lazzles, are all Bloops definitely Lazzles?",
        reference_answer="yes", scorer=exact_match_score),
    BenchmarkTask("code_01", "coding",
        "Write a Python function to check if a string is a palindrome. Include edge cases.",
        reference_answer="def is_palindrome", scorer=exact_match_score),
    BenchmarkTask("summ_01", "summarization",
        "Summarize in one sentence: The transformer architecture introduced self-attention, "
        "allowing models to attend to all positions simultaneously. This replaced recurrent "
        "layers, enabling massive parallelization and superior performance on NLP tasks.",
        reference_answer="transformer self-attention parallelization", scorer=rouge_l_score),
    BenchmarkTask("fact_01", "factual",
        "What year was the original Transformer paper 'Attention Is All You Need' published?",
        reference_answer="2017", scorer=exact_match_score),
]

print(f"Benchmark suite: {len(BENCHMARK_TASKS)} tasks across "
      f"{len(set(t.category for t in BENCHMARK_TASKS))} categories")
```

### Step 4: Run Benchmarks and Collect Results
```python
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PROVIDERS = [
    ("gpt-4o",                     call_openai),
    ("claude-3-5-sonnet-20241022", call_claude),
    ("gemini-1.5-pro",             call_gemini),
]

def run_benchmark(
    tasks: list[BenchmarkTask],
    providers: list[tuple[str, Callable]],
    max_workers: int = 6,
) -> pd.DataFrame:
    """Run all tasks across all providers concurrently."""
    jobs = [(task, model, fn) for task in tasks for model, fn in providers]
    records = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fn, task.prompt): (task, model)
            for task, model, fn in jobs
        }
        for future in as_completed(futures):
            task, model = futures[future]
            resp: LLMResponse = future.result()
            score = None
            if resp.error is None and task.scorer and task.reference_answer:
                score = task.scorer(resp.response, task.reference_answer)

            records.append({
                "task_id": task.task_id, "category": task.category,
                "provider": resp.provider, "model": resp.model,
                "latency_ms": resp.latency_ms, "cost_usd": resp.cost_usd,
                "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                "quality_score": score, "error": resp.error,
            })
            print(f"  {resp.provider:<12} | {task.task_id:<10} | "
                  f"latency={resp.latency_ms:.0f}ms | score={score:.2f}" if score else
                  f"  {resp.provider:<12} | {task.task_id:<10} | ERROR")

    return pd.DataFrame(records)

# df = run_benchmark(BENCHMARK_TASKS, PROVIDERS)
# Simulated results for offline demo:
import numpy as np; np.random.seed(42)
df = pd.DataFrame({
    "provider": ["OpenAI","Anthropic","Google"] * 5,
    "category": ["reasoning"]*3 + ["coding"]*3 + ["summarization"]*3 + ["factual"]*3 + ["reasoning"]*3,
    "latency_ms": np.random.uniform(400, 2500, 15),
    "cost_usd": np.random.uniform(0.0001, 0.002, 15),
    "quality_score": np.random.uniform(0.5, 1.0, 15),
})
print(df.groupby("provider")[["latency_ms","cost_usd","quality_score"]].mean().round(3))
```

### Step 5: Visualize Results with Plotly
```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_benchmark_dashboard(df: pd.DataFrame) -> go.Figure:
    """Create a multi-panel interactive benchmark dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Quality Score by Provider & Category",
            "Latency Distribution (ms)",
            "Cost per Request (USD)",
            "Quality vs Latency Trade-off",
        ],
    )

    colors = {"OpenAI": "#10a37f", "Anthropic": "#d4a017", "Google": "#4285f4"}
    providers = df["provider"].unique()

    # Panel 1: Grouped bar — quality by category
    for provider in providers:
        sub = df[df["provider"] == provider]
        avg = sub.groupby("category")["quality_score"].mean()
        fig.add_trace(go.Bar(name=provider, x=avg.index, y=avg.values,
                             marker_color=colors[provider]), row=1, col=1)

    # Panel 2: Box plot — latency distribution
    for provider in providers:
        sub = df[df["provider"] == provider]
        fig.add_trace(go.Box(name=provider, y=sub["latency_ms"],
                             marker_color=colors[provider], showlegend=False), row=1, col=2)

    # Panel 3: Bar — mean cost
    avg_cost = df.groupby("provider")["cost_usd"].mean()
    fig.add_trace(go.Bar(x=avg_cost.index, y=avg_cost.values * 1000,
                         marker_color=[colors[p] for p in avg_cost.index],
                         showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="Cost (milliUSD)", row=2, col=1)

    # Panel 4: Scatter — quality vs latency
    for provider in providers:
        sub = df[df["provider"] == provider]
        fig.add_trace(go.Scatter(
            name=provider, x=sub["latency_ms"], y=sub["quality_score"],
            mode="markers", marker=dict(size=10, color=colors[provider]),
            showlegend=False,
        ), row=2, col=2)

    fig.update_layout(title="LLM Benchmark Dashboard", height=800, barmode="group")
    fig.write_html("benchmark_report.html")
    print("Dashboard saved to benchmark_report.html")
    return fig

dashboard = plot_benchmark_dashboard(df)
print("Plotly dashboard created. Open benchmark_report.html to explore.")
```

## Expected Output
- A pandas DataFrame with one row per (task, provider) combination
- Metrics captured: latency, token counts, cost per request, quality score
- Interactive Plotly dashboard (HTML) with four panels: quality by category, latency distribution, cost comparison, quality-vs-latency scatter
- CSV export of full results for reproducibility
- Summary table showing which model wins each category

## Stretch Goals
- [ ] **LLM-as-judge scoring:** For tasks without a reference answer (creative writing, open-ended Q&A), use GPT-4o itself as a judge: send both responses and ask it to rate each 1-10 with justification, then aggregate scores across multiple judge calls to reduce variance
- [ ] **Streaming throughput test:** Measure tokens-per-second for streaming responses by timing the first token latency vs. full response time; create a dedicated streaming benchmark category and add it to the dashboard
- [ ] **Regression tracking:** Save each benchmark run to a timestamped JSONL file, then add a fifth dashboard panel that plots quality and cost trends over time so you can detect when a provider silently degrades

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`