import pytest
from datasets import Dataset

from vqar.data import PROMPT_REASONING, PROMPT_SHORT, _clean_cauldron_answer, _vqav2_assignments


def rows_with_qa_counts(counts):
    """Fake cauldron rows: each row i has counts[i] QA pairs (no images needed)."""
    return Dataset.from_list(
        [{"texts": [{"user": f"q{i}-{j}\nSuffix.", "assistant": "A."} for j in range(n)]}
         for i, n in enumerate(counts)]
    )


class TestCauldronAnswer:
    def test_strips_period_and_lowercases(self):
        assert _clean_cauldron_answer("Net.") == "net"

    def test_keeps_decimals_intact(self):
        assert _clean_cauldron_answer("1.50") == "1.50"


class TestAssignments:
    def test_quotas_filled_exactly(self):
        rows = rows_with_qa_counts([3, 3, 3, 3])
        out = _vqav2_assignments(rows, n_sft=4, n_rl=3)
        assert len(out["sft"]) == 4
        assert len(out["rl"]) == 3

    def test_image_disjoint(self):
        rows = rows_with_qa_counts([3] * 10)
        out = _vqav2_assignments(rows, n_sft=7, n_rl=5)
        sft_rows = {i for i, _ in out["sft"]}
        rl_rows = {i for i, _ in out["rl"]}
        assert sft_rows.isdisjoint(rl_rows)

    def test_image_never_split_across_buckets(self):
        # SFT quota fills mid-row: that row's remaining QAs must be skipped, not
        # given to RL.
        rows = rows_with_qa_counts([2, 3, 2, 2])
        out = _vqav2_assignments(rows, n_sft=3, n_rl=2)
        assert (1, 0) in out["sft"] and (1, 1) not in out["sft"]
        assert all(i != 1 for i, _ in out["rl"])

    def test_raises_when_source_exhausted(self):
        rows = rows_with_qa_counts([2, 2])
        with pytest.raises(ValueError, match="exhausted"):
            _vqav2_assignments(rows, n_sft=3, n_rl=3)


class TestTemplates:
    def test_short_template_formats(self):
        assert "single word" in PROMPT_SHORT.format(question="Q?")

    def test_reasoning_template_has_tags(self):
        rendered = PROMPT_REASONING.format(question="Q?")
        assert "<think>" in rendered and "<answer>" in rendered
