"""The rubric documents quote the prompts the code actually sends."""

from pathlib import Path

from scripts.audit_judge import PROMPT as AUDIT_PROMPT
from vqar.judge import PROMPT as JUDGE_PROMPT

RUBRIC = Path(__file__).resolve().parent.parent / "docs" / "harness" / "rubrics" / "judge.md"


def test_the_judge_rubric_quotes_both_prompts():
    text = RUBRIC.read_text()
    assert JUDGE_PROMPT in text
    assert AUDIT_PROMPT in text
