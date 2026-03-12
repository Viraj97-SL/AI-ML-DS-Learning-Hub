# Contributing to the AI/ML/DS Learning Hub

First of all — **thank you** for wanting to contribute! This repo exists because of community contributions. Whether you're fixing a typo, adding a resource, or creating a whole new section, every contribution counts.

---

## Ways to Contribute

| Type | Examples |
|------|---------|
| Fix errors | Typos, broken links, incorrect information |
| Add resources | Books, courses, tools, papers you found valuable |
| Improve explanations | Make a concept clearer or more accurate |
| Add exercises | New hands-on problems or notebooks |
| Translate | Help make this accessible in more languages |
| Create projects | Add a new portfolio project guide |
| Update content | New tools, salary data, course updates |
| Review PRs | Help review other contributions |

---

## Before You Start

1. **Check existing issues** — your idea might already be in progress
2. **Open an issue first** for large contributions — discuss before spending hours writing
3. **Keep it accurate** — verify information before adding it
4. **Keep it practical** — this repo serves learners, not recruiters

---

## How to Contribute (Step-by-Step)

### 1. Fork the Repository
Click "Fork" in the top right of the GitHub page.

### 2. Clone Your Fork
```bash
git clone https://github.com/YOUR-USERNAME/ai-ml-ds-learning-hub.git
cd ai-ml-ds-learning-hub
```

### 3. Create a Branch
```bash
# Use a descriptive branch name
git checkout -b add-pytorch-tutorial
git checkout -b fix-typo-in-salary-guide
git checkout -b update-langchain-resources
```

### 4. Make Your Changes
- Edit or create files following the style guide below
- Test any code you add (notebooks should run cleanly)
- Update the relevant README if needed

### 5. Commit with a Clear Message
```bash
git add .
git commit -m "Add: PyTorch beginner tutorial notebook

- Covers tensors, autograd, and a simple neural network
- Includes exercises with solutions in comments
- Tested on Colab T4 GPU"
```

### 6. Push and Create a Pull Request
```bash
git push origin your-branch-name
```
Then go to GitHub and click "New Pull Request".

---

## Style Guide

### Markdown Files
- Use `#` for H1 (only one per file — the title)
- Use `##` for major sections, `###` for subsections
- Tables for comparisons (use the pipe format)
- Code blocks with language specified (` ```python`, ` ```bash`, ` ```sql`)
- Links: use descriptive text, not "click here"
- Emoji: use sparingly, only where it genuinely helps
- Keep lines under 120 characters where practical

### Notebooks (.ipynb)
- Each notebook should be runnable top-to-bottom without errors
- Add a header cell with: Title, Description, Prerequisites, Estimated time
- Use markdown cells to explain what and why, not just how
- Include "Try it yourself" challenge cells at the end of each section
- Comment your code
- Test on Google Colab before submitting
- Clear all output before committing (to keep diffs clean)

### Code Quality
- Python: follow PEP 8 (use `black` formatter)
- Use type hints for function signatures
- Include docstrings for functions and classes
- Functions should do one thing
- Prefer readability over cleverness

### Resources (books, courses, tools)
When adding resources to a list:
- Verify it's still active/available
- Include: title, author/creator, free or paid, difficulty level
- Add a brief note on what makes it worth including
- Don't add resources you haven't actually used or vetted

---

## Content Standards

### Accuracy
- Cite sources where appropriate (link to papers, documentation)
- Don't state things as fact without basis
- Note if something is your opinion vs. consensus
- Salary figures: note the source and date

### Neutrality
- Don't prefer one tool over another without explaining why
- Acknowledge trade-offs and alternatives
- Don't promote paid products over equivalent free ones without clear justification

### Inclusivity
- Use accessible language (explain jargon when first used)
- Write for global audience (avoid idioms, local references)
- Credit original authors and sources

---

## What We Won't Accept

- Plagiarized content (without attribution)
- Content that primarily markets/sells a commercial product
- Content containing incorrect or misleading information
- Off-topic content (must relate to DS, MLE, or AIE)
- Notebooks that don't run
- Broken links (without an alternative)

---

## Issue Templates

When opening an issue, please use the appropriate template:

- **Bug Report:** Something is wrong (broken link, incorrect info)
- **Content Request:** Ask for content to be added
- **Question:** Ask a question about content (use GitHub Discussions instead)
- **Enhancement:** Suggest an improvement

---

## Code of Conduct

By participating, you agree to:
- Be respectful and constructive
- Welcome all experience levels
- Give and receive feedback gracefully
- Prioritize what's best for learners

Harassment, discrimination, or disrespectful behavior will result in removal.

---

## Recognition

All contributors are recognized in our Contributors section. Significant contributors are highlighted in section-specific READMEs.

Thank you for making this better for everyone!

---

*Questions? Open a [GitHub Discussion](../../discussions) or an Issue.*
