# AI-Powered CLI Assistant

> **Difficulty:** Intermediate | **Time:** 1-2 days | **Track:** AI Engineer

## What You'll Build
A polished command-line tool that uses an LLM to help users understand terminal commands, debug shell errors, generate scripts, explain code snippets, and execute suggested commands — all without leaving the terminal.

## Learning Objectives
- Structure a production-quality CLI with Click and Rich
- Integrate the Claude or OpenAI API for conversational assistance
- Safely handle subprocess execution with user confirmation prompts
- Maintain a persistent conversation history across CLI invocations
- Package the tool as an installable Python package with `pyproject.toml`

## Prerequisites
- Comfortable with Python and the command line
- An OpenAI or Anthropic API key
- Basic understanding of subprocess and file I/O

## Tech Stack
- `anthropic` or `openai`: LLM API client
- `click`: CLI framework for commands, options, and argument parsing
- `rich`: beautiful terminal output with syntax highlighting and panels
- `subprocess`: safe shell command execution with timeout and streaming
- `typer` (optional): alternative to Click with automatic type inference

## Step-by-Step Guide

### Step 1: Project Setup and Core Imports
```python
# pip install anthropic openai click rich

# File structure:
# ai_cli/
#   __init__.py
#   cli.py          ← main Click group
#   llm.py          ← LLM client abstraction
#   history.py      ← conversation persistence
#   executor.py     ← safe subprocess runner
# pyproject.toml

import os
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

console = Console()

HISTORY_FILE = Path.home() / ".ai_cli_history.json"
MAX_HISTORY  = 20       # keep last N turns in context

def load_history() -> list[dict]:
    """Load conversation history from disk."""
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())[-MAX_HISTORY:]
    return []

def save_history(history: list[dict]) -> None:
    """Persist conversation history to disk."""
    existing = load_history()
    updated = (existing + history)[-MAX_HISTORY:]
    HISTORY_FILE.write_text(json.dumps(updated, indent=2))

print(f"History file: {HISTORY_FILE}")
print(f"Max retained turns: {MAX_HISTORY}")
```

### Step 2: LLM Client with Streaming
```python
import anthropic
from anthropic import Anthropic

SYSTEM_PROMPT = """You are an expert DevOps and software engineering assistant embedded in a CLI.
When the user asks about a command or error:
1. Explain what it means clearly and concisely
2. If suggesting a shell command, wrap it in a ```bash block
3. If suggesting a fix, provide the corrected code in a fenced code block
4. If the request involves file paths, use safe relative paths
5. Never suggest commands that delete data without explicit warning

Be brief and practical. Prefer examples over long explanations."""

def ask_claude(
    user_message: str,
    history: list[dict],
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 1024,
) -> str:
    """Send a message to Claude with conversation history. Returns full response."""
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = history + [{"role": "user", "content": user_message}]

    response_text = ""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            response_text += text
            console.print(text, end="", highlight=False)   # stream to terminal
        console.print()   # newline after stream ends

    return response_text

print("LLM client configured with streaming output.")
```

### Step 3: Safe Command Executor
```python
import subprocess
import shlex
from rich.prompt import Confirm

DANGEROUS_PATTERNS = ["rm -rf", "dd if=", "mkfs", ":(){:|:&};:", "chmod 777"]

def is_dangerous(command: str) -> tuple[bool, str]:
    """Check if a command matches known destructive patterns."""
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return True, pattern
    return False, ""

def execute_command(
    command: str,
    timeout_s: int = 30,
    require_confirm: bool = True,
) -> tuple[int, str, str]:
    """Execute a shell command safely with optional confirmation prompt.

    Returns (returncode, stdout, stderr).
    """
    dangerous, pattern = is_dangerous(command)
    if dangerous:
        console.print(f"[bold red]DANGER:[/] Command matches pattern '{pattern}'. Refusing to execute.")
        return -1, "", f"Blocked: dangerous pattern '{pattern}'"

    if require_confirm:
        console.print(Panel(Syntax(command, "bash", theme="monokai"), title="Command to Execute"))
        if not Confirm.ask("[yellow]Execute this command?[/]"):
            return -1, "", "User declined"

    try:
        result = subprocess.run(
            shlex.split(command),
            capture_output=True, text=True,
            timeout=timeout_s,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timed out after {timeout_s}s"
    except FileNotFoundError as e:
        return -1, "", f"Command not found: {e}"

print("Safe executor ready. Dangerous patterns blocked:", DANGEROUS_PATTERNS[:3], "...")
```

### Step 4: Click CLI Commands
```python
import click
import re

@click.group()
def cli():
    """AI-powered CLI assistant. Ask questions, debug errors, generate scripts."""
    pass

@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--run", is_flag=True, help="Auto-run any suggested command (with confirmation).")
def ask(question: tuple[str, ...], run: bool):
    """Ask the AI a question about any terminal topic.

    Example: ai ask 'how do I find files larger than 100MB?'
    """
    query = " ".join(question)
    history = load_history()

    console.rule("[bold cyan]AI CLI Assistant[/]")
    console.print(f"[dim]You:[/] {query}\n")

    response = ask_claude(query, history)
    save_history([
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ])

    if run:
        # Extract first bash code block from response
        match = re.search(r"```bash\n(.*?)```", response, re.DOTALL)
        if match:
            cmd = match.group(1).strip()
            returncode, stdout, stderr = execute_command(cmd)
            if stdout:
                console.print(Panel(stdout, title="Output", border_style="green"))
            if stderr:
                console.print(Panel(stderr, title="Stderr", border_style="red"))

@cli.command()
@click.argument("error_text")
def debug(error_text: str):
    """Explain and fix a terminal error message.

    Example: ai debug 'ModuleNotFoundError: No module named pandas'
    """
    prompt = f"I got this error. Explain what it means and how to fix it:\n\n{error_text}"
    history = load_history()
    response = ask_claude(prompt, history)
    save_history([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ])

@cli.command()
@click.argument("description")
@click.option("--lang", default="bash", show_default=True, help="Script language (bash, python, zsh).")
def generate(description: str, lang: str):
    """Generate a script from a natural language description.

    Example: ai generate 'backup all .py files to ~/backups with today date' --lang bash
    """
    prompt = f"Write a {lang} script that: {description}\nInclude comments and error handling."
    history = load_history()
    response = ask_claude(prompt, history)
    save_history([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ])

@cli.command()
def history():
    """Show recent conversation history."""
    hist = load_history()
    if not hist:
        console.print("[dim]No history yet.[/]")
        return
    for i, msg in enumerate(hist):
        role_color = "cyan" if msg["role"] == "user" else "green"
        console.print(f"[{role_color}]{msg['role'].capitalize()}:[/] {msg['content'][:120]}...")

if __name__ == "__main__":
    cli()

print("CLI commands defined: ask, debug, generate, history")
print("Usage: python cli.py ask 'how do I list open ports?'")
```

### Step 5: Package as an Installable Tool
```python
# pyproject.toml content (write to disk for real packaging)
PYPROJECT_TOML = """
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "ai-cli"
version = "0.1.0"
description = "AI-powered terminal assistant"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.30",
    "openai>=1.30",
    "click>=8.1",
    "rich>=13.0",
]

[project.scripts]
ai = "ai_cli.cli:cli"
"""

from pathlib import Path
Path("pyproject.toml").write_text(PYPROJECT_TOML.strip())
print("pyproject.toml written.")
print()
print("Install locally with: pip install -e .")
print("Then use from anywhere: ai ask 'explain git rebase'")
print("                        ai debug 'Permission denied (publickey)'")
print("                        ai generate 'rotate logs older than 7 days' --lang bash")
```

## Expected Output
- A working `ai` CLI command installable via `pip install -e .`
- `ai ask`: natural-language Q&A with streaming output and optional command execution
- `ai debug`: paste any error and get a clear explanation with fix steps
- `ai generate`: generate bash/Python scripts from plain-English descriptions
- `ai history`: colorized recent conversation log
- All commands use Rich panels, syntax highlighting, and confirmation prompts

## Stretch Goals
- [ ] **Context-aware mode:** Before answering, the tool reads the current directory's `README.md`, `pyproject.toml`, or `package.json` and injects them as context so the AI knows the project it's helping with
- [ ] **Shell integration (magic mode):** Add a `--magic` flag that wraps the entire session in a pseudo-PTY, intercepts failed commands (non-zero exit codes), and automatically sends the error to the AI for diagnosis without requiring any extra typing
- [ ] **Plugin system:** Design a `~/.ai_cli_plugins/` directory where users can drop Python files that register new `@cli.command()` entries (e.g., a `ai docker` plugin or an `ai git` plugin) loaded dynamically at startup

## Share Your Work
Post your solution in GitHub Discussions with the tag `#mini-project`