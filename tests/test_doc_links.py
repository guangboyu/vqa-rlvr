"""Relative links and anchors in the public docs resolve."""

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")),
         *sorted((ROOT / "docs" / "findings").glob("*.md")),
         *sorted((ROOT / "docs" / "harness").glob("*.md"))]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    heading = re.sub(r"<!--.*?-->|`", "", heading).strip().lower()
    return re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")


def outside_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def anchors(path: Path) -> set[str]:
    return {slug(match) for match in re.findall(r"^#{1,6} (.+)$", outside_code(path.read_text()),
                                                re.MULTILINE)}


def broken_links(page: Path) -> list[str]:
    broken = []
    for target in LINK.findall(outside_code(page.read_text())):
        if re.match(r"[a-z]+:", target):
            continue
        path, _, anchor = target.partition("#")
        resolved = Path(os.path.normpath(page.parent / path)) if path else page
        if not resolved.exists() or (anchor and resolved.suffix == ".md"
                                     and anchor not in anchors(resolved)):
            broken.append(target)
    return broken


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(ROOT)))
def test_links_resolve(page):
    assert not broken_links(page)
