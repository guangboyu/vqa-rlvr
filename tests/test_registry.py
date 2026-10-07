from dataclasses import replace

import pytest

from vqar import metrics
from vqar.registry import (
    DATASETS,
    LADDER,
    PROMPTS,
    ROLES,
    SOURCE_REVISIONS,
    datasets_with_role,
    score,
)


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_every_dataset_declares_roles_scorers_and_a_ladder_position(name):
    spec = DATASETS[name]
    assert spec.subsets and set(spec.subsets) <= set(ROLES)
    assert spec.ladder in LADDER
    assert (callable(spec.official) or (spec.official is None and callable(spec.project)))
    assert callable(spec.normalized)
    assert set(spec.sources) <= set(SOURCE_REVISIONS)


def test_ood_dataset_cannot_declare_a_training_role():
    textvqa = DATASETS["textvqa"]
    assert textvqa.ladder == "far-ood" and set(textvqa.subsets) == {"test"}
    with pytest.raises(ValueError, match="OOD dataset cannot have a training role"):
        replace(textvqa, subsets={"rl": "textvqa_rl", "test": "textvqa_eval"})


def test_official_splits_must_name_declared_roles():
    with pytest.raises(ValueError, match="official splits"):
        replace(DATASETS["textvqa"], official_splits=frozenset({"dev"}))


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="unknown roles"):
        replace(DATASETS["clevr"], subsets={"train": "clevr_rl"})


def test_role_listing_follows_registry_order():
    assert datasets_with_role("test") == ["vqav2", "gqa", "clevr", "textvqa", "chartqa",
                                          "clevrcomp", "vizwiz", "realworldqa"]
    assert datasets_with_role("sft") == ["vqav2", "gqa", "chartqa", "clevrcomp"]
    assert datasets_with_role("sft", optional=False) == ["vqav2", "gqa"]
    assert datasets_with_role("dev", optional=False) == ["vqav2", "gqa", "clevr"]


class TestScoreDispatch:
    def test_vqav2_normalized_uses_vqa_accuracy(self):
        assert score("vqav2", "cat", ["cat"] * 10, "normalized") == 1.0

    def test_gqa_normalized_uses_exact_match(self):
        assert score("gqa", "left", ["left"], "normalized") == 1.0

    def test_clevr_normalized_maps_number_words(self):
        assert score("clevr", "four", ["4"], "normalized") == 1.0

    def test_official_keeps_the_authors_casing_rule(self):
        answers = ["yes"] * 10
        assert score("vqav2", "Yes", answers, "official") == 0.0
        assert score("vqav2", "Yes", answers, "normalized") == 1.0

    def test_gqa_official_does_not_strip_articles(self):
        assert score("gqa", "the frisbee", ["frisbee"], "official") == 0.0
        assert score("gqa", "the frisbee", ["frisbee"], "normalized") == 1.0

    def test_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown scorer variant"):
            score("gqa", "left", ["left"], "lenient")

    def test_official_vqa_scorers_need_all_annotators(self):
        for name in ("vqav2", "textvqa"):
            assert DATASETS[name].official_roles == {"test"}
            with pytest.raises(ValueError, match="10 annotator"):
                score(name, "yes", ["yes"], "official")


class TestSingleReferenceFallback:
    """Single-gold inputs must not be zeroed by leave-one-out (the historical reward bug).

    The VQAv2 training pools store exactly one gold answer; the 10-annotator
    leave-one-out metric returns 0.0 for any prediction there, which silently
    removed the correctness reward on the VQAv2 half of the July GRPO mixes.
    """

    def test_single_gold_exact_pred_scores_one(self):
        assert score("vqav2", "yes", ["yes"], "normalized") == 1.0

    def test_single_gold_wrong_pred_scores_zero(self):
        assert score("vqav2", "no", ["yes"], "normalized") == 0.0

    def test_single_gold_normalized_match(self):
        assert score("vqav2", "The black one.", ["black 1"], "normalized") == 1.0

    def test_ten_golds_still_leave_one_out(self):
        # 3 of 10 annotators said "cat": leave-one-out partial credit, not EM.
        golds = ["cat"] * 3 + ["dog"] * 7
        assert 0.0 < score("vqav2", "cat", golds, "normalized") < 1.0

    def test_textvqa_single_gold_fallback(self):
        assert score("textvqa", "42", ["42"], "normalized") == 1.0


def test_exact_string_match_is_raw():
    assert metrics.exact_string_match("Left", ["left"]) == 0.0
    assert metrics.exact_string_match("left", ["left", "right"]) == 1.0


class TestPrompts:
    def test_headline_templates_are_short_and_reasoning(self):
        headline = [name for name, prompt in PROMPTS.items() if not prompt.diagnostic]
        assert headline == ["short", "reasoning"]

    def test_short_template_formats(self):
        assert "single word" in PROMPTS["short"].text.format(question="Q?")

    def test_reasoning_template_has_tags(self):
        rendered = PROMPTS["reasoning"].text.format(question="Q?")
        assert rendered.startswith("Q?\n") and "<think>" in rendered and "<answer>" in rendered
