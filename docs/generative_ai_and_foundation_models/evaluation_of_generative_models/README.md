# Evaluation of Generative Models

> A comprehensive guide to evaluating LLMs and generative AI systems — covering automatic metrics, LLM-as-judge, RAG evaluation with RAGAS, benchmark suites, and building reliable eval pipelines.

---

## Overview

Evaluating generative AI models is fundamentally harder than evaluating discriminative models. There is no single correct output for "Write a summary of this article" — multiple valid responses exist. Traditional metrics like accuracy don't apply. Yet evaluation is critical: without it, you cannot tell if a new model version is better, if a prompt change improved quality, or if your RAG system is hallucinating.

Modern LLM evaluation combines three approaches:
1. **Automatic metrics** — fast, cheap, scalable but imperfect proxies
2. **LLM-as-judge** — use a capable LLM to evaluate another LLM's outputs
3. **Human evaluation** — gold standard but expensive and slow

AI engineers must build evaluation pipelines before shipping features. "Vibe checks" (manually reading 20 outputs) don't scale and introduce confirmation bias. Treat evals like unit tests: write them first, run them on every change, and monitor them in production.

---

## Key Concepts

### Why LLM Evaluation Is Hard

- **No single ground truth**: Many valid outputs exist for most prompts
- **Multidimensional quality**: Accuracy, fluency, coherence, helpfulness, safety — these trade off against each other
- **Context dependency**: Quality depends on the use case (customer service vs. code generation have different standards)
- **Adversarial failure modes**: Models that are fluent and confident while being wrong (hallucination)
- **Reference-free scenarios**: For many tasks (creative writing, brainstorming), no reference answer exists

### Core Evaluation Dimensions

| Dimension | Question | How to Measure |
|-----------|----------|----------------|
| **Faithfulness** | Does the output contain only facts from the source? | NLI models, LLM-as-judge |
| **Relevance** | Does the output address the question asked? | LLM-as-judge, embedding similarity |
| **Accuracy** | Is the output factually correct? | Human eval, knowledge graph lookup |
| **Coherence** | Is the output logically consistent? | LLM-as-judge |
| **Safety** | Does the output avoid harmful content? | Classifiers (LlamaGuard, Perspective API) |
| **Groundedness** | Is each claim supported by retrieved context? | RAGAS faithfulness |

---

## Automatic Metrics

### Text Similarity Metrics

**BLEU (Bilingual Evaluation Understudy)**: n-gram precision between generated and reference text. Designed for translation; poor for open-ended generation.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: n-gram recall and F1. Used for summarization. ROUGE-L measures longest common subsequence.

**BERTScore**: Uses BERT embeddings to compare semantic similarity. More robust than n-gram metrics for paraphrases.

```python
from evaluate import load
import numpy as np

# ROUGE
rouge = load("rouge")
predictions = ["The cat sat on the mat and looked outside."]
references = ["The cat was sitting on the mat, gazing out the window."]

scores = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1: {scores['rouge1']:.3f}")
print(f"ROUGE-2: {scores['rouge2']:.3f}")
print(f"ROUGE-L: {scores['rougeL']:.3f}")

# BERTScore
bertscore = load("bertscore")
results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en",
    model_type="distilbert-base-uncased",
)
print(f"BERTScore F1: {np.mean(results['f1']):.3f}")
```

**Perplexity**: Measures how well a language model predicts a text. Lower = more confident. Useful for comparing generation quality on in-domain text.

---

## LLM-as-Judge

Use a powerful LLM (GPT-4, Claude Opus) to evaluate outputs from a less capable model. This scales evaluation without human annotators and can assess nuanced quality.

```python
from anthropic import Anthropic
import json

client = Anthropic()

JUDGE_PROMPT = """You are an expert evaluator for AI assistant responses.

You will be given:
- A user question
- An AI assistant's response
- The source context (if applicable)

Evaluate the response on these dimensions (score 1-5):
1. Accuracy: Is the information factually correct?
2. Relevance: Does the response address what was asked?
3. Faithfulness: Is every claim supported by the provided context?
4. Clarity: Is the response clear and well-organized?

Return a JSON object with scores and one-sentence justifications.

Question: {question}
Context: {context}
Response: {response}

Return ONLY valid JSON in this format:
{{"accuracy": {{"score": 4, "reason": "..."}},
 "relevance": {{"score": 5, "reason": "..."}},
 "faithfulness": {{"score": 3, "reason": "..."}},
 "clarity": {{"score": 4, "reason": "..."}},
 "overall": 4}}
"""

def judge_response(question: str, context: str, response: str) -> dict:
    """Use Claude as a judge to evaluate an LLM response."""
    prompt = JUDGE_PROMPT.format(
        question=question, context=context, response=response
    )
    result = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(result.content[0].text)

# Example usage
scores = judge_response(
    question="What is the capital of France?",
    context="France is a country in Western Europe. Its capital city is Paris.",
    response="The capital of France is Paris, known for the Eiffel Tower.",
)
print(scores)
```

### Pitfalls of LLM-as-Judge
- **Position bias**: Judges prefer the first option when comparing two responses
- **Verbosity bias**: Judges often prefer longer, more elaborate responses
- **Self-enhancement bias**: GPT-4 may prefer GPT-4 outputs
- **Mitigation**: Use randomized ordering, calibrate with human labels, use multiple judges

---

## RAG Evaluation with RAGAS

RAGAS (RAG Assessment) is the leading framework for evaluating Retrieval-Augmented Generation systems end-to-end.

### RAGAS Metrics

| Metric | Measures | Range |
|--------|---------|-------|
| **Faithfulness** | Are claims in the answer supported by the retrieved context? | 0–1 |
| **Answer Relevancy** | Does the answer address the question? | 0–1 |
| **Context Precision** | Are retrieved chunks actually relevant to the question? | 0–1 |
| **Context Recall** | Is all necessary information present in retrieved context? | 0–1 |

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is the transformer architecture?",
        "How does RAG work?",
    ],
    "answer": [
        "The transformer uses self-attention mechanisms to process sequences in parallel.",
        "RAG retrieves relevant documents and uses them as context for generation.",
    ],
    "contexts": [
        ["The Transformer model uses self-attention to relate positions in a sequence..."],
        ["RAG (Retrieval-Augmented Generation) combines information retrieval with generation..."],
    ],
    "ground_truth": [
        "Transformers use self-attention to process input sequences.",
        "RAG retrieves documents then generates answers conditioned on them.",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Run RAGAS evaluation
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(results.to_pandas())
print(f"\nFaithfulness:     {results['faithfulness']:.3f}")
print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")
print(f"Context Precision:{results['context_precision']:.3f}")
print(f"Context Recall:   {results['context_recall']:.3f}")
```

---

## Benchmarks

| Benchmark | Tests | Use Case |
|-----------|-------|---------|
| **MMLU** | 57-subject multiple choice | General knowledge |
| **HumanEval** | Python coding problems | Code generation |
| **HellaSwag** | Commonsense completion | Language understanding |
| **TruthfulQA** | Truthfulness on tricky questions | Hallucination |
| **MATH** | Math word problems | Reasoning |
| **GSM8K** | Grade school math | Arithmetic reasoning |
| **MT-Bench** | Multi-turn conversation | Chat quality |
| **LMSYS Chatbot Arena** | Human preference head-to-head | Real-world preference |
| **SWE-bench** | Real GitHub issues | Software engineering |

---

## Building an Eval Pipeline

```python
import pandas as pd
from dataclasses import dataclass
from typing import Callable

@dataclass
class EvalCase:
    id: str
    prompt: str
    reference: str | None = None
    metadata: dict = None

@dataclass
class EvalResult:
    case_id: str
    model_output: str
    scores: dict
    passed: bool

class EvalPipeline:
    """Minimal eval pipeline for LLM applications."""

    def __init__(self, generate_fn: Callable, metrics: list):
        self.generate = generate_fn
        self.metrics = metrics

    def run(self, cases: list[EvalCase]) -> pd.DataFrame:
        results = []
        for case in cases:
            output = self.generate(case.prompt)
            scores = {}
            for metric in self.metrics:
                scores[metric.__name__] = metric(
                    prompt=case.prompt,
                    output=output,
                    reference=case.reference,
                )
            results.append(EvalResult(
                case_id=case.id,
                model_output=output,
                scores=scores,
                passed=all(v >= 0.7 for v in scores.values()),
            ))

        df = pd.DataFrame([
            {"id": r.case_id, "output": r.model_output, "passed": r.passed, **r.scores}
            for r in results
        ])
        print(f"Pass rate: {df['passed'].mean():.1%}")
        return df
```

---

## Tools & Libraries

| Tool | Purpose | Notes |
|------|---------|-------|
| **RAGAS** | RAG evaluation | Best-in-class for RAG systems |
| **promptfoo** | Prompt/model testing | CI-friendly, YAML config |
| **LangSmith** | LLM tracing + evaluation | LangChain ecosystem |
| **Braintrust** | Eval platform | Great for human + auto evals |
| **Weights & Biases (W&B)** | Experiment tracking + evals | Integrates with most frameworks |
| **HuggingFace Evaluate** | Automatic metrics (BLEU, ROUGE) | Standard metric library |
| **lm-evaluation-harness** | Benchmark suite runner | Run MMLU, HellaSwag, etc. locally |
| **LlamaGuard** | Safety classification | Meta's content safety model |

---

## Resources

### Courses & Guides
- [RAGAS Documentation](https://docs.ragas.io/) — Official RAGAS framework docs with tutorials
- [Hugging Face: Evaluation on the Hub](https://huggingface.co/docs/evaluate/index) — Comprehensive metric library docs
- [Chip Huyen: Evaluating LLMs](https://huyenchip.com/2023/01/24/llm-evaluations.html) — Excellent practical overview

### Key Papers
- [RAGAS: Automated Evaluation of RAG Pipelines](https://arxiv.org/abs/2309.15217) — Es et al. 2023
- [Judging LLM-as-a-Judge with MT-Bench](https://arxiv.org/abs/2306.05685) — Zheng et al. 2023
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) — Lin et al. 2021

---

## Projects & Exercises

**Project 1 — RAG Eval Suite**
Build a RAG system over a 50-document corpus on a topic you know well. Create 20 test questions with reference answers. Run RAGAS (faithfulness, answer relevancy, context precision, context recall). Identify the weakest dimension and improve it (better chunking, different embedding model, reranker). Re-run and compare.

**Project 2 — LLM-as-Judge vs Human Eval**
Generate 50 responses with GPT-3.5 to creative writing prompts. Have an LLM judge score them 1-5 on quality. Have 3 humans score the same set. Calculate Spearman correlation between LLM scores and average human scores. Document where LLM-as-judge agrees and disagrees with humans.

**Project 3 — Regression Testing Pipeline**
Build a CI evaluation pipeline using GitHub Actions: when a prompt template changes, automatically run 30 test cases, compute LLM-as-judge scores, and fail the PR if average score drops more than 5% vs. baseline. Use promptfoo or a custom script.

---

## Related Topics
- [AI Evaluation Notebook →](../../03_AI_Engineer/intermediate/06_ai_evaluation.ipynb)
- [Advanced Evals Notebook →](../../03_AI_Engineer/advanced/06_advanced_evals.ipynb)
- [Production AI Systems →](../../03_AI_Engineer/intermediate/07_production_ai.ipynb)
- [Natural Language Processing →](../natural_language_processing/README.md)
