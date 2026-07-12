from types import SimpleNamespace

from vqar.judge import Judge
from vqar.rewards import correctness_reward, format_reward, make_judge_reward

GOOD = (
    "Let me count the objects step by step. There are three cubes on the left and "
    "one metallic sphere behind them, so the total number of items is four.\n"
    "<answer>4</answer>"
)


def completion(text):
    return [{"role": "assistant", "content": text}]


class TestCorrectness:
    def test_clevr_numeric_match(self):
        r = correctness_reward(
            prompts=None,
            completions=[completion(GOOD)],
            answers=[["4"]],
            dataset=["clevr"],
        )
        assert r == [1.0]

    def test_vqav2_partial_credit(self):
        r = correctness_reward(
            prompts=None,
            completions=[completion("<answer>red</answer>")],
            answers=[["red"] * 2 + ["blue"] * 8],
            dataset=["vqav2"],
        )
        assert 0.0 < r[0] < 1.0

    def test_miss(self):
        r = correctness_reward(
            prompts=None, completions=[completion("<answer>7</answer>")],
            answers=[["4"]], dataset=["clevr"],
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


class TestJudgeReward:
    def _judge(self, tmp_path, text="yes"):
        client = SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=text)],
                    usage=SimpleNamespace(input_tokens=100, output_tokens=2),
                )
            )
        )
        return Judge(cache_path=tmp_path / "j.db", budget_usd=10.0, client=client)

    def test_rescues_em_miss(self, tmp_path):
        fn = make_judge_reward(self._judge(tmp_path, "yes"))
        r = fn(
            prompts=None,
            completions=[completion("<answer>grey</answer>")],
            answers=[["gray"]],
            dataset=["gqa"],
            question=["What color is the cat?"],
        )
        assert r == [1.0]

    def test_em_match_skips_judge(self, tmp_path):
        judge = self._judge(tmp_path)
        calls = []
        judge.judge = lambda *a: calls.append(a) or True
        fn = make_judge_reward(judge)
        r = fn(
            prompts=None,
            completions=[completion("<answer>4</answer>")],
            answers=[["4"]],
            dataset=["clevr"],
            question=["How many?"],
        )
        assert r == [1.0] and not calls

    def test_budget_exhaustion_degrades_gracefully(self, tmp_path):
        judge = self._judge(tmp_path)
        judge.budget_usd = 0.0  # every call raises
        fn = make_judge_reward(judge)
        r = fn(
            prompts=None,
            completions=[completion("<answer>grey</answer>")] * 2,
            answers=[["gray"]] * 2,
            dataset=["gqa"] * 2,
            question=["Q"] * 2,
        )
        assert r == [0.0, 0.0]  # fell back to EM without crashing
