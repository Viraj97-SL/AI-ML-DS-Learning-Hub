# Community Challenges

> Monthly challenges designed to sharpen your skills through friendly competition and community learning.

Each challenge is open for the entire month. Fork the repo, build your solution, and open a PR — your work will be visible to everyone in the community.

---

## Upcoming Challenges

| # | Challenge | Difficulty | Track | Status |
|---|-----------|------------|-------|--------|
| 01 | [Titanic Survival Prediction](#challenge-01-titanic-survival-prediction) | Beginner | Data Scientist | Coming Soon |
| 02 | [Build a RAG Pipeline from Scratch](#challenge-02-build-a-rag-pipeline-from-scratch) | Intermediate | AI Engineer | Coming Soon |
| 03 | [Deploy an ML Model with FastAPI + Docker](#challenge-03-deploy-an-ml-model-with-fastapi--docker) | Intermediate | ML Engineer | Coming Soon |

---

## Challenge Details

### Challenge 01: Titanic Survival Prediction

**Track:** Data Scientist | **Difficulty:** Beginner

**Goal:** Build a machine learning model to predict passenger survival on the Titanic dataset.

**Requirements:**
- Perform exploratory data analysis (EDA) with at least 5 visualizations
- Engineer at least 3 new features from the raw data
- Train and compare at least 3 classification models (e.g., Logistic Regression, Random Forest, XGBoost)
- Report accuracy, precision, recall, and F1 score for each model
- Write a brief summary of your findings (what features mattered most and why)

**Bonus:** Use SHAP values to explain your best model's predictions.

---

### Challenge 02: Build a RAG Pipeline from Scratch

**Track:** AI Engineer | **Difficulty:** Intermediate

**Goal:** Build a Retrieval-Augmented Generation (RAG) system without using a high-level framework like LangChain — implement the retrieval loop yourself.

**Requirements:**
- Use any embedding model (e.g., `sentence-transformers`, OpenAI embeddings) to embed a document corpus
- Implement cosine similarity search manually (no vector DB required, though you may use one)
- Build a question-answering interface over the corpus using an LLM of your choice
- Benchmark retrieval quality on at least 10 test questions (precision@k)
- Document any design decisions in your notebook

**Bonus:** Compare two chunking strategies (fixed-size vs. sentence-based) and report which performs better.

---

### Challenge 03: Deploy an ML Model with FastAPI + Docker

**Track:** ML Engineer | **Difficulty:** Intermediate

**Goal:** Take a trained sklearn model from a Jupyter notebook all the way to a containerised REST API.

**Requirements:**
- Train any classification or regression model on a public dataset
- Wrap the model in a FastAPI application with a `/predict` endpoint
- Write a `Dockerfile` that builds and runs the service
- Include a `docker-compose.yml` for one-command startup
- Provide a `README.md` inside your solution folder with setup and usage instructions

**Bonus:** Add a `/health` endpoint and a basic smoke test using `pytest` + `httpx`.

---

## How to Submit Your Solution

1. **Fork** this repository to your own GitHub account.
2. Create a new folder under `05_Hands_On_Labs/challenges/submissions/` named `challenge_XX_your-github-username/` (e.g., `challenge_01_viraj97-sl/`).
3. Add your notebook or code files inside that folder.
4. Include a short `README.md` in your submission folder describing your approach and results.
5. **Open a Pull Request** against the `master` branch of this repo with the title format: `[Challenge XX] Your Name — Brief Description`.
6. In the PR description, briefly explain your approach and link to any live demos if applicable.

> Note: All submitted solutions will be visible to the public. By submitting, you agree to share your work under the MIT license.

---

*Back to: [Hands-On Labs](../README.md) | [Main README](../../README.md)*