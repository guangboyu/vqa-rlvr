"""The public tree carries no tool attribution; the judge's pinned model ids are the exception."""

import subprocess
from pathlib import Path

import pytest

from vqar.judge import AUDIT_MODEL, MODEL

ROOT = Path(__file__).resolve().parent.parent
# Spelled backwards so this file does not trip its own check.
BANNED = tuple(word[::-1] for word in ("edualc", "xedoc"))
# The judge's and the auditor's pinned model ids are data, not attribution.
ALLOWED_PATH = "vqar/judge.py"
ALLOWED_SNIPPETS = (f'MODEL = "{MODEL}"', f'AUDIT_MODEL = "{AUDIT_MODEL}"')


def tracked_files() -> list[str]:
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [path for path in listing.stdout.decode().split("\0") if path]


def banned_words(path: str) -> set[str]:
    text = path
    file = ROOT / path
    if file.is_file() and not file.is_symlink():
        content = file.read_bytes().decode("utf-8", errors="ignore")
        if path == ALLOWED_PATH:
            for snippet in ALLOWED_SNIPPETS:
                content = content.replace(snippet, "")
        text += "\n" + content
    return {word for word in BANNED if word in text.lower()}


def test_tracked_files_name_no_tools():
    offenders = {path: words for path in tracked_files() if (words := banned_words(path))}
    assert not offenders, offenders


def test_allowlisted_snippets_are_still_in_the_judge_module():
    # A stale allowlist would silently stop guarding the judge module.
    text = (ROOT / ALLOWED_PATH).read_text()
    assert all(snippet in text for snippet in ALLOWED_SNIPPETS)
