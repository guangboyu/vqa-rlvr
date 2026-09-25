from vqar.metrics import (
    exact_match,
    normalize_answer,
    official_vqa_accuracy,
    textvqa_accuracy,
    vqa_accuracy,
)
from vqar.registry import score


class TestNormalizeAnswer:
    def test_lowercase_and_period(self):
        assert normalize_answer("Yes.") == "yes"

    def test_number_words(self):
        assert normalize_answer("two") == "2"
        assert normalize_answer("none") == "0"

    def test_articles_removed(self):
        assert normalize_answer("a dog") == "dog"
        assert normalize_answer("the red car") == "red car"

    def test_contractions(self):
        assert normalize_answer("isnt") == "isn't"
        assert normalize_answer("dont") == "don't"

    def test_decimal_points_kept(self):
        assert normalize_answer("1.50") == "1.50"

    def test_hyphen_split(self):
        # No adjacent space: official rule replaces punctuation with a space.
        assert normalize_answer("black-and-white") == "black and white"

    def test_spelling_variants_not_unified(self):
        # Motivating case for the LLM judge: EM treats these as different.
        assert normalize_answer("gray") != normalize_answer("grey")


class TestVqaAccuracy:
    def test_unanimous_match(self):
        assert vqa_accuracy("yes", ["yes"] * 10) == 1.0

    def test_unanimous_miss(self):
        assert vqa_accuracy("no", ["yes"] * 10) == 0.0

    def test_partial_credit_three_of_ten(self):
        # 3 of 10 said "cat": each "cat" annotator sees 2 other matches (2/3),
        # each "dog" annotator sees 3 (capped at 1.0). Mean = (3*2/3 + 7*1)/10 = 0.9.
        answers = ["cat"] * 3 + ["dog"] * 7
        assert abs(vqa_accuracy("cat", answers) - 0.9) < 1e-9

    def test_partial_credit_exact_value(self):
        # 2 of 10 annotators agree with the prediction.
        answers = ["red"] * 2 + ["blue"] * 8
        # "red" annotators see 1 other match (1/3); "blue" annotators see 2 (2/3).
        expected = (2 * (1 / 3) + 8 * (2 / 3)) / 10
        assert abs(vqa_accuracy("red", answers) - expected) < 1e-9

    def test_normalization_applied_when_annotators_disagree(self):
        answers = ["two"] * 9 + ["three"]
        assert vqa_accuracy("2", answers) > 0.9

    def test_normalization_applied_even_when_unanimous(self):
        # Deliberate deviation from the 2016 code, matching lmms-eval: "Yes" must
        # match a unanimous ["yes"]*10 (capitalized zero-shot answers are not errors).
        assert vqa_accuracy("2", ["two"] * 10) == 1.0
        assert vqa_accuracy("Yes", ["yes"] * 10) == 1.0


class TestExactMatch:
    def test_simple(self):
        assert exact_match("Yes", ["yes"]) == 1.0

    def test_number_word(self):
        assert exact_match("four", ["4"]) == 1.0

    def test_miss(self):
        assert exact_match("4 people", ["4"]) == 0.0  # judge-worthy miss


class TestSingleReferenceFallback:
    """One reference must not be zeroed by leave-one-out (the July training-reward defect).

    vqav2_rl stores one answer per question; the ten-annotator leave-one-out metric returns
    0.0 for any prediction there, which silently removed the correctness reward from the
    VQAv2 half of every July GRPO mix. The normalized scorer, which defines the reward,
    falls back to exact match.
    """

    def test_single_reference_exact_prediction_scores_one(self):
        assert score("vqav2", "yes", ["yes"], "normalized") == 1.0

    def test_single_reference_wrong_prediction_scores_zero(self):
        assert score("vqav2", "no", ["yes"], "normalized") == 0.0

    def test_single_reference_normalized_match(self):
        assert score("vqav2", "The black one.", ["black 1"], "normalized") == 1.0

    def test_ten_references_keep_partial_credit(self):
        # 3 of 10 annotators said "cat": leave-one-out partial credit, not exact match.
        answers = ["cat"] * 3 + ["dog"] * 7
        assert 0.0 < score("vqav2", "cat", answers, "normalized") < 1.0

    def test_textvqa_single_reference_fallback(self):
        assert score("textvqa", "42", ["42"], "normalized") == 1.0

    def test_the_reward_pays_a_single_reference_answer(self):
        from vqar.rewards import correctness_reward

        rewards = correctness_reward(prompts=None, completions=["<answer>yes</answer>"],
                                     answers=[["yes"]], dataset=["vqav2"],
                                     rollout_prompt=["reasoning"])
        assert rewards == [1.0]


def test_official_vqa_keeps_unanimous_case_rule_separate():
    answers = ["yes"] * 10
    assert official_vqa_accuracy("Yes", answers) == 0.0
    assert vqa_accuracy("Yes", answers) == 1.0


def test_official_vqa_normalizes_when_annotators_disagree():
    assert official_vqa_accuracy("Two.", ["2"] * 3 + ["3"] * 7) == 0.9


def test_textvqa_uses_evalai_possessive_tokenization():
    assert textvqa_accuracy("dog's", ["dog 's"] * 10) == 1.0
