"""Every number quoted outside a generated table comes from results/numbers.json."""

import json

import pytest

from scripts.report_page import NUMBERS_PATH, QUOTED_NUMBER, fill_numbers, numbered_pages, numbers


def public_pages():
    return [page for page in numbered_pages() if page.parent.name != "interview"]


def test_fill_refreshes_the_value_between_the_tags():
    text = "from <!--n:a-->1<!--/n--> to <!--n:b.c-->2<!--/n-->"
    assert fill_numbers(text, {"a": "7", "b.c": "8"}) == ("from <!--n:a-->7<!--/n--> to "
                                                         "<!--n:b.c-->8<!--/n-->")


def test_a_quoted_number_needs_a_known_key():
    with pytest.raises(KeyError):
        fill_numbers("<!--n:missing-->1<!--/n-->", {})


def test_committed_numbers_match_the_records():
    assert json.loads(NUMBERS_PATH.read_text()) == numbers(), "run python -m scripts.report_page"


def test_quoted_numbers_match_the_numbers_file():
    values = json.loads(NUMBERS_PATH.read_text())
    for page in public_pages():
        text = page.read_text()
        assert fill_numbers(text, values) == text, f"{page.name} drifted"


def test_the_readme_quotes_its_numbers():
    readme = next(page for page in public_pages() if page.name == "README.md")
    assert len(QUOTED_NUMBER.findall(readme.read_text())) >= 15
