import pytest

from scripts.summarize_comparison import paired_intervention, score_variants


def test_intervention_requires_the_same_question_and_reference():
    original = [{"qid": "1", "answers": ["cat"], "prediction": "cat", "score": 1.0}]
    changed = [{"qid": "1", "answers": ["dog"], "prediction": "dog", "score": 1.0}]
    with pytest.raises(ValueError, match="references differ"):
        paired_intervention(original, changed)


def test_case_diagnostic_keeps_the_official_score_unchanged():
    rows = [{"prediction": "Yes", "answers": ["yes"] * 10, "score": 0.0}]
    result = score_variants("vqav2", rows)
    assert result == {
        "normalized": 1.0, "lowercase_official": 1.0, "normalized_changed_questions": 1,
    }
    assert rows[0]["score"] == 0.0
