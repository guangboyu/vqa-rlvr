from vqar.rewards import correctness_reward, format_reward, list_aware_correctness_reward

GOOD = (
    "Let me count the objects step by step. There are three cubes on the left and "
    "one metallic sphere behind them, so the total number of items is four.\n"
    "<answer>4</answer>"
)


def completion(text):
    return [{"role": "assistant", "content": text}]


def test_the_list_aware_reward_pays_reordered_chartqa_lists_and_leaves_other_data_alone():
    kwargs = dict(prompts=None, rollout_prompt=["reasoning"] * 2,
                  completions=[completion("<answer>Niger, Gambia</answer>")] * 2,
                  answers=[["[Gambia, Niger]"], ["[Gambia, Niger]"]], dataset=["chartqa", "gqa"])
    assert correctness_reward(**kwargs) == [0.0, 0.0]
    assert list_aware_correctness_reward(**kwargs) == [1.0, 0.0]


class TestCorrectness:
    def test_the_reasoning_template_pays_only_a_tagged_answer(self):
        r = correctness_reward(
            prompts=None, completions=[completion("4"), completion("<answer>4</answer>")],
            answers=[["4"], ["4"]], dataset=["clevr"] * 2, rollout_prompt=["reasoning"] * 2,
        )
        assert r == [0.0, 1.0]

    def test_the_short_template_scores_the_whole_response(self):
        # The short template never asks for tags, so tags would make every reward zero.
        r = correctness_reward(
            prompts=None, completions=[completion(" Four. "), completion("It is red.")],
            answers=[["4"], ["red"]], dataset=["clevr", "gqa"], rollout_prompt=["short"] * 2,
        )
        assert r == [1.0, 0.0]

    def test_clevr_numeric_match(self):
        r = correctness_reward(
            prompts=None,
            completions=[completion(GOOD)],
            answers=[["4"]],
            dataset=["clevr"],
            rollout_prompt=["reasoning"],
        )
        assert r == [1.0]

    def test_vqav2_partial_credit(self):
        r = correctness_reward(
            prompts=None,
            completions=[completion("<answer>red</answer>")],
            answers=[["red"] * 2 + ["blue"] * 8],
            dataset=["vqav2"],
            rollout_prompt=["reasoning"],
        )
        assert 0.0 < r[0] < 1.0

    def test_miss(self):
        r = correctness_reward(
            prompts=None, completions=[completion("<answer>7</answer>")],
            answers=[["4"]], dataset=["clevr"], rollout_prompt=["reasoning"],
        )
        assert r == [0.0]


class TestFormat:
    def test_well_formed(self):
        assert format_reward(prompts=None, completions=[completion(GOOD)]) == [1.0]

    def test_bare_answer_without_reasoning(self):
        assert format_reward(
            prompts=None, completions=[completion("<answer>4</answer>")]
        ) == [0.0]

    def test_trailing_text_rejected(self):
        bad = GOOD + " Also, some rambling."
        assert format_reward(prompts=None, completions=[completion(bad)]) == [0.0]

    def test_answer_dump_rejected(self):
        bad = f"{'I think it could be several values. ' * 5}<answer>{'4 or maybe 5 ' * 20}</answer>"
        assert format_reward(prompts=None, completions=[completion(bad)]) == [0.0]

    def test_multiple_answer_tags_rejected(self):
        bad = f"{'Reasoning about the scene here. ' * 5}<answer>3</answer><answer>4</answer>"
        assert format_reward(prompts=None, completions=[completion(bad)]) == [0.0]



def test_official_correctness_needs_the_option_letter_on_multiple_choice():
    from vqar.rewards import correctness_reward, official_correctness_reward

    kwargs = dict(prompts=[None, None], answers=[["B", "kidney"], ["B", "kidney"]],
                  dataset=["deeptumorx", "deeptumorx"], rollout_prompt=["short", "short"])
    completions = ["B", "kidney"]
    assert correctness_reward(completions=completions, **kwargs) == [1.0, 1.0]
    assert official_correctness_reward(completions=completions, **kwargs) == [1.0, 0.0]
