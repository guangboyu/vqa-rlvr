"""The public tree carries no tool attribution; the judge's pinned model id is the one exception."""

import subprocess
from pathlib import Path

import pytest

from vqar.judge import MODEL

ROOT = Path(__file__).resolve().parent.parent
# Spelled backwards so this file does not trip its own check.
BANNED = tuple(word[::-1] for word in ("edualc", "xedoc"))
# The judge study's pinned model id is data, not attribution.
ALLOWED = {"vqar/judge.py": (f'MODEL = "{MODEL}"',)}


def tracked_files() -> list[str]:
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [path for path in listing.stdout.decode().split("\0") if path]


def banned_words(path: str) -> list[str]:
    found = [word for word in BANNED if word in path.lower()]
    file = ROOT / path
    if file.is_file() and not file.is_symlink():
        text = file.read_bytes().decode("utf-8", errors="ignore")
        for allowed in ALLOWED.get(path, ()):
            text = text.replace(allowed, "")
        found += [word for word in BANNED if word in text.lower()]
    return found


def test_tracked_files_name_no_tools():
    offenders = {path: words for path in tracked_files() if (words := banned_words(path))}
    assert not offenders, offenders


def test_allowlist_matches_the_judge_module():
    # A stale allowlist would silently stop guarding the judge module.
    for path, snippets in ALLOWED.items():
        text = (ROOT / path).read_text()
        assert all(snippet in text for snippet in snippets), path
