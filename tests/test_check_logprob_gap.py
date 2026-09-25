"""The log-prob gap check splits |delta logp| by answer form and by sampled-token probability."""

import math

import pytest

from scripts.check_logprob_gap import (
    batches,
    delta_stats,
    first_mismatch,
    format_report,
    is_ratio,
    is_spaced,
    probability_bin,
    summarize,
)

VOCABULARY = {}


def sample(text, tokens, vllm_probabilities, deltas, mismatch=None):
    """A scored response whose trainer log-probs sit `deltas` above vLLM's."""
    vllm = [math.log(p) for p in vllm_probabilities]
    return {"text": text, "tokens": tokens,
            "completion_ids": [VOCABULARY.setdefault(t, len(VOCABULARY)) for t in tokens],
            "vllm_logprobs": vllm,
            "train_logprobs": [v + d for v, d in zip(vllm, deltas, strict=True)],
            "prompt_mismatch": mismatch}


@pytest.mark.parametrize("text, spaced", [
    ("<think>x</think><answer> green </answer>", True),
    ("<answer> green</answer>", True),
    ("<answer>green </answer>", True),
    ("<answer>\ngreen\n</answer>", True),
    ("<answer>green</answer>", False),
    ("the answer is green", False),
])
def test_spaced_means_whitespace_inside_the_tags(text, spaced):
    assert is_spaced(text) is spaced


def test_probability_bins_by_the_sampled_tokens_vllm_probability():
    assert probability_bin(math.log(0.95)) == "p>0.9"
    assert probability_bin(math.log(0.6)) == "0.5-0.9"
    assert probability_bin(math.log(0.2)) == "0.1-0.5"
    assert probability_bin(math.log(0.01)) == "p<0.1"
    assert probability_bin(-1e9) == "p<0.1"


def test_first_mismatch_counts_length_differences():
    assert first_mismatch([1, 2, 3], [1, 2, 3]) is None
    assert first_mismatch([1, 2, 3], [1, 9, 3]) == 1
    assert first_mismatch([1, 2], [1, 2, 3]) == 2


def test_is_ratio_is_the_exponentiated_summed_difference():
    assert is_ratio([-1.0, -2.0], [-1.1, -1.8]) == pytest.approx(math.exp(0.1))
    assert is_ratio([-1000.0], [0.0]) == math.exp(700.0)  # capped, not an overflow


def test_delta_stats_on_empty_and_filled_lists():
    assert delta_stats([])["mean"] is None
    stats = delta_stats([0.1, 0.3, 0.2, 0.4])
    assert stats["tokens"] == 4 and stats["mean"] == pytest.approx(0.25)
    assert stats["max"] == 0.4 and stats["p50"] == 0.3


def test_summary_splits_by_form_probability_and_token():
    samples = [
        sample("<answer>green</answer>", ["<answer>", "green", "</answer>"],
               [0.99, 0.95, 0.99], [0.01, 0.01, -0.01]),
        sample("<answer> green </answer>", ["<answer>", " green", " </answer>"],
               [0.99, 0.3, 0.05], [0.01, -0.2, -0.5], mismatch=4),
    ]
    summary = summarize(samples, min_count=1)
    assert summary["responses_by_form"] == {"spaced": 1, "unspaced": 1}
    assert summary["prompt_id_mismatches"] == 1
    assert summary["overall"]["tokens"] == 6
    assert summary["by_form"]["unspaced"]["mean"] == pytest.approx(0.01)
    assert summary["by_form"]["spaced"]["mean"] == pytest.approx((0.01 + 0.2 + 0.5) / 3)
    assert summary["by_probability"]["p>0.9"]["tokens"] == 4
    assert summary["by_probability"]["p<0.1"]["mean"] == pytest.approx(0.5)
    assert summary["by_form_and_probability"]["unspaced"]["p<0.1"]["tokens"] == 0
    assert summary["by_form_and_probability"]["spaced"]["0.1-0.5"]["mean"] == pytest.approx(0.2)
    top = summary["top_tokens"]["rows"]
    assert top[0]["token"] == " </answer>" and top[0]["mean"] == pytest.approx(0.5)
    assert top[0]["share_of_total"] == pytest.approx(0.5 / 0.74)
    assert summary["is_ratio"]["unspaced"]["mean"] == pytest.approx(math.exp(0.01))
    assert summary["is_ratio"]["spaced"]["masked_share"] == 0.0


def test_top_tokens_need_the_minimum_count():
    samples = [sample("<answer>a</answer>", ["a", "b", "b"], [0.5, 0.9, 0.9], [0.9, 0.1, 0.1])]
    rows = summarize(samples, min_count=2)["top_tokens"]["rows"]
    assert [row["token"] for row in rows] == ["b"] and rows[0]["count"] == 2


def test_ratios_above_the_training_cap_count_as_masked():
    samples = [sample("<answer>a</answer>", ["a"], [0.1], [2.0]),
               sample("<answer>a</answer>", ["a"], [0.1], [0.0])]
    assert summarize(samples)["is_ratio"]["all"]["masked_share"] == 0.5


def test_unscored_responses_are_counted_but_not_aggregated():
    unscored = sample("<answer>a</answer>", ["a"], [0.5], [0.0], mismatch=3)
    unscored["train_logprobs"] = None
    summary = summarize([unscored, sample("<answer>a</answer>", ["a"], [0.5], [0.2])])
    assert summary["unscored"] == 1 and summary["prompt_id_mismatches"] == 1
    assert summary["overall"]["tokens"] == 1
    assert summary["responses_by_form"] == {"spaced": 0, "unspaced": 1}


def test_report_prints_every_section():
    samples = [sample("<answer> a </answer>", ["a"], [0.5], [0.1])]
    report = format_report(summarize(samples, min_count=1))
    for part in ("1 spaced", "mismatches", "spaced, vLLM 0.5-0.9", "'a'", "masked"):
        assert part in report


def test_batches_keep_order_and_a_short_tail():
    assert batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_an_unmerged_adapter_is_recorded_and_served_at_its_rank(tmp_path):
    from argparse import Namespace

    from scripts.check_logprob_gap import adapter_rank, settings
    from vqar.config import GRPO_PRESETS

    (tmp_path / "adapter_config.json").write_text('{"r": 16, "lora_alpha": 32}')
    assert adapter_rank(str(tmp_path)) == 16
    args = Namespace(revision=None, adapter=str(tmp_path), questions=4, samples=8,
                     questions_per_batch=4, seed=0, unmerged_adapter=False)
    merged = settings(args, GRPO_PRESETS["grpo_2b_main"], "base")
    assert "vllm_adapter" not in merged  # sample files from merged runs still match
    args.unmerged_adapter = True
    assert settings(args, GRPO_PRESETS["grpo_2b_main"], "base") == {
        **merged, "vllm_adapter": "unmerged"}
