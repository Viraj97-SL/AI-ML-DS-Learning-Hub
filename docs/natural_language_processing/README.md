# Natural Language Processing

> A comprehensive guide to NLP for data scientists and AI engineers — from text preprocessing and classical methods through transformer fine-tuning and production deployment.

---

## Overview

Natural Language Processing (NLP) is the branch of AI that gives computers the ability to understand, interpret, and generate human language. It bridges linguistics, statistics, and deep learning to process unstructured text — which makes up an estimated 80% of all enterprise data.

NLP powers search engines, recommendation systems, chatbots, automated document processing, sentiment analysis, machine translation, and the LLMs (large language models) that underpin modern AI products. For data scientists, NLP skills unlock an entirely new class of problems; for AI engineers, NLP is the core discipline.

The field underwent a paradigm shift in 2017–2018 with the introduction of the Transformer architecture and BERT. Today, the dominant approach is fine-tuning large pretrained models (BERT, RoBERTa, T5, LLaMA) rather than training from scratch.

---

## Key Concepts

### Text Preprocessing Pipeline
1. **Lowercasing** — normalize case (task-dependent)
2. **Tokenization** — split text into tokens (words, subwords, characters)
3. **Stop word removal** — remove high-frequency but low-information words ("the", "is")
4. **Stemming / Lemmatization** — reduce words to base forms ("running" → "run")
5. **Noise removal** — strip HTML, URLs, special characters

### Tokenization Strategies

| Strategy | How | Vocab Size | Used In |
|----------|-----|-----------|---------|
| Word-level | Split on whitespace | Large (100k+) | Classic NLP |
| Character-level | Split into chars | Tiny (100) | CharCNN |
| BPE (Byte-Pair Encoding) | Merge frequent pairs | 30k–50k | GPT, RoBERTa |
| WordPiece | Maximize likelihood | 30k | BERT |
| SentencePiece | Language-agnostic BPE | Configurable | T5, LLaMA |

### Word Representations
- **One-hot encoding**: Sparse, no semantic meaning
- **TF-IDF**: Statistical importance weighting — great for retrieval and classification
- **Word2Vec (Skip-gram / CBOW)**: Dense vectors, captures semantic similarity
- **GloVe**: Global co-occurrence statistics, often outperforms Word2Vec
- **FastText**: Subword embeddings — handles misspellings and rare words well
- **Contextual embeddings (BERT etc.)**: Same word → different vector depending on context

### The Transformer Architecture
The Transformer (Vaswani et al., 2017) replaced RNNs as the dominant architecture. Key components:
- **Multi-head self-attention**: Each token attends to all other tokens simultaneously
- **Positional encoding**: Injects sequence order information
- **Feed-forward sublayers**: Per-position MLP after attention
- **Layer normalization + residual connections**: Training stability

**Encoder-only** (BERT, RoBERTa): Best for classification, NER, QA
**Decoder-only** (GPT, LLaMA): Best for text generation
**Encoder-decoder** (T5, BART): Best for translation, summarization

---

## Learning Path

### Beginner
1. Text preprocessing with Python (regex, NLTK, spaCy)
2. TF-IDF vectorization + scikit-learn classifiers for sentiment analysis
3. Word2Vec/GloVe embeddings with Gensim
4. Named entity recognition with spaCy

### Intermediate
5. HuggingFace `transformers` library — loading and running BERT/GPT models
6. Fine-tuning BERT for text classification on a custom dataset
7. Token classification: Named Entity Recognition with BERT
8. Extractive QA with BERT (`BertForQuestionAnswering`)
9. Text generation with GPT-2

### Advanced
10. Sequence-to-sequence models: T5/BART for summarization and translation
11. Parameter-efficient fine-tuning: LoRA, adapters
12. Retrieval-Augmented Generation (RAG) pipelines
13. LLM inference optimization: quantization (GPTQ, AWQ), speculative decoding
14. Building NLP evaluation pipelines

---

## Code Examples

### Text Classification with BERT (HuggingFace)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load and tokenize dataset
dataset = load_dataset("imdb")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()
```

### TF-IDF + Logistic Regression (Fast Baseline)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Build pipeline
nlp_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # log normalization
        min_df=2,                # ignore rare terms
        strip_accents="unicode",
    )),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)),
])

# Fit and evaluate
nlp_pipeline.fit(X_train, y_train)
y_pred = nlp_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# TF-IDF baselines are surprisingly strong — always run before BERT!
```

### Named Entity Recognition with spaCy

```python
import spacy

nlp = spacy.load("en_core_web_trf")  # transformer-based model

text = """
Apple CEO Tim Cook announced a new partnership with OpenAI in San Francisco
on March 15, 2026. The deal is worth $1.5 billion over three years.
"""

doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(f"{ent.text:30s} {ent.label_:10s} {spacy.explain(ent.label_)}")

# Dependency parsing
for token in doc[:10]:
    print(f"{token.text:15s} {token.dep_:10s} → {token.head.text}")
```

---

## Tools & Libraries

| Library | Purpose | When to Use |
|---------|---------|-------------|
| **spaCy** | Industrial NLP pipeline | NER, POS tagging, parsing, production |
| **NLTK** | Classic NLP algorithms | Learning, tokenization, WordNet |
| **HuggingFace Transformers** | BERT, GPT, T5, LLaMA | Everything transformer-based |
| **HuggingFace Datasets** | Benchmark datasets | Training and evaluation |
| **Gensim** | Word2Vec, FastText, LDA | Word embeddings, topic modeling |
| **scikit-learn** | TF-IDF, vectorizers, classifiers | Fast baselines |
| **sentence-transformers** | Semantic similarity, embeddings | Search, clustering, RAG |
| **LangChain / LlamaIndex** | LLM application frameworks | RAG, agents, chains |
| **BERTopic** | Neural topic modeling | Unsupervised topic discovery |
| **Presidio** | PII detection and anonymization | Privacy, GDPR compliance |

---

## Resources

### Articles & Guides
- [Natural Language Processing (NLP): A Complete Beginner’s Guide](https://www.bettermindlabs.org/post/natural-language-processing-nlp-a-complete-beginner-s-guide) — High-level overview of what NLP is and why it matters.
- [Natural Language Processing (NLP) [A Complete Guide]](https://www.deeplearning.ai/resources/natural-language-processing) — A guide from deeplearning.ai covering concepts and applications.
- [Master Natural Language Processing in 2025](https://www.analyticsvidhya.com/blog/2022/01/master-natural-language-processing-in-2022-with-best-resources) — A curated list of resources from Analytics Vidhya.

### Courses & Tutorials
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) — Best academic NLP course; free lecture videos
- [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) — Hands-on, free, covers Transformers library end-to-end
- [fast.ai NLP](https://course.fast.ai/) — Practical, top-down approach to NLP
- [Natural Language Processing (NLP) Full Course – Beginner to Advanced](https://www.youtube.com/watch?v=Rj-OtK2n5jU) — Comprehensive YouTube course by Edureka covering NLP with Python.
- [Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer](https://www.youtube.com/watch?v=sVcwVQRHIc8) — A practical, code-first tutorial on building RAG systems.
- [Natural Language Processing with spaCy & Python - Course for Beginners](https://www.youtube.com/watch?v=dIUTsFT2MeQ) — A freeCodeCamp course focused on the spaCy library.
- [My Top Picks: 5 Free NLP Courses I’d Recommend for 2025](https://www.kdnuggets.com/top-picks-5-free-nlp-courses-recommend-2025) — A curated list of free courses from KDnuggets.
- [Rycolab Intro to NLP (Fall 2025)](https://rycolab.io/classes/intro-nlp-f25) — University course materials from ETH Zurich.

### Books
- *Speech and Language Processing* — Jurafsky & Martin — [Free PDF](https://web.stanford.edu/~jurafsky/slp3/) — Definitive NLP textbook
- *Natural Language Processing with Transformers* — Tunstall, von Werra, Wolf (O'Reilly) — HuggingFace team's book
- *Practical Natural Language Processing* — Vajjala et al. (O'Reilly) — Production-focused
- *Natural Language Processing with Python* — Bird, Klein, & Loper (O'Reilly) — The classic "NLTK book," excellent for foundational concepts.

### Key Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. 2017 — The Transformer
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al. 2018
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al. 2020
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. 2021
- [CoinRAG: Contextualized Information Nugget KV Cache Reuse for Long-Context RAG](https://arxiv.org/abs/2608.07458v1) — Liu et al. 2026 — Research on optimizing RAG performance.
- [CreativeInstruct: Scalably Teaching LLMs to Balance Quality, Creativity, and Diversity](https://arxiv.org/abs/2608.07460v1) — Zhou et al. 2026 — A paper on improving LLM instruction tuning for creative tasks.

---

## Projects & Exercises

For more ideas, check out [10 Must-Try NLP Projects That'll Get You Noticed!](https://www.youtube.com/watch?v=YUjZPR3L8SA) (YouTube).

**Project 1 — Sentiment Analysis API**
Fine-tune DistilBERT on a product review dataset (Amazon or Yelp). Wrap it in a FastAPI app that returns `{sentiment, confidence, explanation}`. Compare against a TF-IDF + Logistic Regression baseline. Document when the simpler model is good enough.
*For a guided walkthrough, see [Python Sentiment Analysis Project with NLTK and 🤗 Transformers](https://www.youtube.com/watch?v=QpzMWQvxXWk).*

**Project 2 — Semantic Search Engine**
Build a search engine over a corpus of 1000+ documents using `sentence-transformers`. Encode all documents, store embeddings in ChromaDB, and implement a search endpoint that returns top-5 semantically similar documents. Add keyword highlighting.

**Project 3 — Document Summarizer + Q&A**
Use T5 or BART for abstractive summarization of long articles. Then implement extractive Q&A over the same documents using BERT. Package both as a single Streamlit app where users can paste an article, get a summary, and ask questions about it.

**Project 4 — Movie Recommendation System**
Build a content-based movie recommender using TF-IDF on movie plot summaries. Create a FastAPI endpoint that takes a movie title and returns the top 5 most similar movies. See this [end-to-end tutorial](https://www.youtube.com/watch?v=IoL9FzHvL3I) for an example.

---

## Related Topics
- [AI Engineer Track →](../../03_AI_Engineer/README.md) — LLM APIs, RAG, agents
- [NLP Fundamentals Notebook →](../../01_Data_Scientist/advanced/02_nlp_fundamentals.ipynb) — Hands-on NLP with code
- [Evaluating Generative Models →](../generative_ai_and_foundation_models/evaluation_of_generative_models/README.md) — How to evaluate LLM outputs
