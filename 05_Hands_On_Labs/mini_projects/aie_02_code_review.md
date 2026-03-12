# Automated Code Review Agent

> **Difficulty:** Advanced | **Time:** 2-3 days | **Track:** AI Engineer

## What You'll Build
An agentic system that integrates with the GitHub API to fetch open pull requests, analyze changed code using a large language model, and post structured review comments — including security flags, style suggestions, and logic issues — directly back to the PR.

## Learning Objectives
- Build a multi-step LangChain agent with tool use
- Integrate with the GitHub REST API (diff fetching, comment posting)
- Prompt engineer for structured, actionable code review output
- Parse and route agent outputs back to the correct PR line numbers
- Handle rate limiting, large diffs, and multi-file PRs gracefully

## Prerequisites
- LangChain agent fundamentals
- A GitHub personal access token with `repo` scope
- An OpenAI or Anthropic API key

## Tech Stack
- `langchain` / `langchain-community`: agent framework and LLM orchestration
- `openai` or `anthropic`: LLM backend (GPT-4o or Claude 3.5 Sonnet)
- `PyGithub`: GitHub API client for PR data and comment posting
- `pygments`: syntax highlighting for diff rendering in CLI output
- `pydantic`: structured output validation for review comments

## Step-by-Step Guide

### Step 1: GitHub API Client Setup
```python
# pip install langchain openai anthropic PyGithub pygments pydantic

import os
from typing import Optional
from github import Github, PullRequest, Repository

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]      # personal access token
gh = Github(GITHUB_TOKEN)

def get_pr_diff(repo_name: str, pr_number: int) -> dict:
    """Fetch PR metadata and changed files with their diffs."""
    repo: Repository = gh.get_repo(repo_name)
    pr: PullRequest = repo.get_pull(pr_number)
    files = pr.get_files()

    diff_data = {
        "pr_title": pr.title,
        "pr_body": pr.body or "",
        "base_branch": pr.base.ref,
        "head_branch": pr.head.ref,
        "files": []
    }

    for f in files:
        diff_data["files"].append({
            "filename": f.filename,
            "status": f.status,          # added, modified, removed
            "additions": f.additions,
            "deletions": f.deletions,
            "patch": f.patch or "",      # unified diff
        })

    print(f"PR #{pr_number}: '{pr.title}' | {len(diff_data['files'])} files changed")
    return diff_data
```

### Step 2: Define Review Comment Schema
```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal

class Severity(str, Enum):
    CRITICAL = "critical"    # security / correctness bug
    WARNING  = "warning"     # potential issue, should address
    INFO     = "info"        # style / best practice suggestion

class ReviewComment(BaseModel):
    filename: str
    line_number: Optional[int] = None   # None = file-level comment
    severity: Severity
    category: Literal["security", "correctness", "performance", "style", "maintainability"]
    message: str = Field(..., min_length=10)
    suggestion: Optional[str] = None    # concrete fix suggestion

class PRReview(BaseModel):
    summary: str
    overall_verdict: Literal["approve", "request_changes", "comment"]
    comments: list[ReviewComment]
    security_issues_found: bool
    estimated_review_minutes: int

def format_diff_for_llm(diff_data: dict, max_tokens_per_file: int = 2000) -> str:
    """Summarize diff into a prompt-friendly string with token budget per file."""
    parts = [f"PR: {diff_data['pr_title']}\n"]
    for f in diff_data["files"]:
        patch = f["patch"]
        if len(patch) > max_tokens_per_file * 4:   # rough char estimate
            patch = patch[:max_tokens_per_file * 4] + "\n... [truncated]"
        parts.append(
            f"\n--- {f['filename']} ({f['status']}, +{f['additions']} -{f['deletions']}) ---\n{patch}"
        )
    return "\n".join(parts)

print("Review schemas defined. Pydantic models ready for structured LLM output.")
```

### Step 3: Build the Code Review LangChain Tools
```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

@tool
def analyze_security(code_diff: str) -> str:
    """Scan a code diff for security vulnerabilities: injection, hardcoded secrets, insecure deserialization."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = f"""You are a security engineer. Review this diff for OWASP Top 10 vulnerabilities,
hardcoded credentials, SQL injection, XSS, and insecure configurations.
List each issue as: [CRITICAL/WARNING] filename:line - description

Diff:
{code_diff[:4000]}
"""
    return llm.invoke(prompt).content

@tool
def analyze_code_quality(code_diff: str) -> str:
    """Review code for correctness, performance, and maintainability issues."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = f"""You are a senior engineer reviewing a PR. Identify:
- Logic bugs or off-by-one errors
- Missing error handling
- Performance anti-patterns (N+1 queries, unnecessary loops)
- Unclear naming or missing docstrings
Format: [WARNING/INFO] filename:line - description + suggested fix

Diff:
{code_diff[:4000]}
"""
    return llm.invoke(prompt).content

@tool
def generate_review_summary(security_findings: str, quality_findings: str, pr_title: str) -> str:
    """Combine security and quality findings into a structured PR review summary."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = f"""Combine these findings into a concise PR review for '{pr_title}'.
Write a 2-3 sentence summary, then list action items by priority.

Security findings:
{security_findings}

Quality findings:
{quality_findings}
"""
    return llm.invoke(prompt).content
```

### Step 4: Orchestrate the Agent
```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

tools = [analyze_security, analyze_code_quality, generate_review_summary]

SYSTEM_PROMPT = """You are an expert code reviewer. Given a pull request diff:
1. First call analyze_security to find security issues
2. Then call analyze_code_quality for logic and style issues
3. Finally call generate_review_summary to produce the final review
Always complete all three steps before responding."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)

def review_pull_request(repo_name: str, pr_number: int) -> str:
    """Run the full code review agent on a pull request."""
    diff_data = get_pr_diff(repo_name, pr_number)
    diff_text = format_diff_for_llm(diff_data)
    result = agent_executor.invoke({"input": f"Review this PR diff:\n{diff_text}"})
    return result["output"]

print("Code review agent assembled and ready.")
```

### Step 5: Post Review Comments Back to GitHub
```python
from github import GithubException

def post_review_to_github(
    repo_name: str,
    pr_number: int,
    review_text: str,
    verdict: str = "comment",
) -> str:
    """Post the AI-generated review as a GitHub PR review."""
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    gh_event = {
        "approve": "APPROVE",
        "request_changes": "REQUEST_CHANGES",
        "comment": "COMMENT",
    }.get(verdict, "COMMENT")

    try:
        review = pr.create_review(body=review_text, event=gh_event)
        url = f"https://github.com/{repo_name}/pull/{pr_number}#pullrequestreview-{review.id}"
        print(f"Review posted: {url}")
        return url
    except GithubException as e:
        print(f"GitHub API error: {e.status} — {e.data}")
        raise

# Full pipeline example:
# diff_data = get_pr_diff("owner/repo", 42)
# review = review_pull_request("owner/repo", 42)
# post_review_to_github("owner/repo", 42, review, verdict="request_changes")
print("GitHub posting function ready.")
```

## Expected Output
- Agent that fetches any public/private PR diff via the GitHub API
- Structured review covering security (OWASP), correctness, style, and performance
- Review comments posted directly to the PR with the correct severity labels
- Verbose agent trace showing each tool call and reasoning step
- CLI output with colorized diff and comment annotations via Pygments

## Stretch Goals
- [ ] **Inline line comments:** Use `pr.create_review_comment()` to post comments at exact line numbers in the diff instead of a single review body, making it look like a real human reviewer
- [ ] **GitHub Actions integration:** Create a workflow YAML that triggers this agent on every `pull_request` event, runs the review, and adds a `ai-reviewed` label to the PR
- [ ] **Learning from feedback:** Log each review to a JSONL file; when a developer resolves or dismisses a comment, record it as negative feedback and use it to fine-tune future review prompts via few-shot examples

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`