"""
Add "Open in Colab" badge to the first cell of every Jupyter notebook.

Usage:
    python scripts/add_colab_badges.py

Run from the repo root. Skips notebooks that already have a Colab badge.
"""

import json
import pathlib
import uuid

REPO = "viraj97-sl/ai-ml-ds-learning-hub"
BRANCH = "master"
COLAB_BASE = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}"
BADGE_IMG = "https://colab.research.google.com/assets/colab-badge.svg"

# Directories to skip entirely
SKIP_DIRS = {".git", ".venv", "venv", "site", "node_modules", "__pycache__"}


def make_badge_cell(nb_path_relative: str) -> dict:
    """Return a Jupyter markdown cell dict with the Colab badge."""
    url = f"{COLAB_BASE}/{nb_path_relative}"
    source = (
        f'<a href="{url}" target="_parent">'
        f'<img src="{BADGE_IMG}" alt="Open In Colab"/></a>'
    )
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [source],
    }


def already_has_badge(nb: dict) -> bool:
    """Return True if the notebook already contains a Colab badge."""
    if not nb.get("cells"):
        return False
    first = nb["cells"][0]
    source = "".join(first.get("source", []))
    return "colab.research.google.com" in source


def process_notebook(path: pathlib.Path, repo_root: pathlib.Path) -> bool:
    """Add badge to notebook if missing. Returns True if modified."""
    try:
        text = path.read_text(encoding="utf-8")
        nb = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  SKIP (parse error): {path.relative_to(repo_root)} — {e}")
        return False

    if already_has_badge(nb):
        return False

    # Build relative path with forward slashes for the URL
    rel = path.relative_to(repo_root).as_posix()
    badge_cell = make_badge_cell(rel)
    nb["cells"].insert(0, badge_cell)

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return True


def main():
    repo_root = pathlib.Path(__file__).parent.parent.resolve()
    notebooks = [
        p for p in repo_root.rglob("*.ipynb")
        if not any(part in SKIP_DIRS for part in p.parts)
        and ".ipynb_checkpoints" not in str(p)
    ]

    notebooks.sort()
    modified = 0
    skipped = 0

    print(f"Found {len(notebooks)} notebooks in {repo_root}\n")

    for nb_path in notebooks:
        rel = nb_path.relative_to(repo_root)
        if process_notebook(nb_path, repo_root):
            print(f"  + badge added: {rel}")
            modified += 1
        else:
            skipped += 1

    print(f"\nDone. {modified} notebooks updated, {skipped} already had badge.")


if __name__ == "__main__":
    main()
