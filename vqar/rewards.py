"""Reward functions for GRPO (RLVR). TRL calling convention:

    fn(prompts=..., completions=..., completion_ids=..., **dataset_columns) -> list[float]

Extra columns of the RL dataset (question, answers, dataset, rollout_prompt) arrive as
kwargs, each a list aligned with completions. Rewards return raw signals in [0, 1];
relative weighting (e.g. format 0.2) is set via GRPOConfig.reward_weights, so ablating a
reward is a config change, not a code change.
"""

from vqar.answers import reward_answer
from vqar.metrics import chartqa_list_accuracy
from vqar.registry import PROMPTS, score

# The format reward wants reasoning followed by one short, terminal <answer> tag.
# Deliberately NOT literal <think> tags: Qwen3-VL writes its chain of thought as plain prose
# and never samples literal think tags, so a think-tag reward is unearnable (flat 0 over 400
# GRPO steps from BOTH the base and SFT policies). Reward the behavior, not magic tokens.
_MIN_REASONING_CHARS = 100


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]  # conversational format


def correctness_reward(prompts, completions, answers, dataset, rollout_prompt,
                       **kwargs) -> list[float]:
    """The dataset's normalized score of the answer read by the rollout prompt's rule."""
    out = []
    for completion, golds, ds, template in zip(completions, answers, dataset, rollout_prompt,
                                               strict=True):
        pred = reward_answer(_text(completion), PROMPTS[template].answer_rule)
        out.append(0.0 if pred is None else score(ds, pred, golds, "normalized"))
    return out


def list_aware_correctness_reward(prompts, completions, answers, dataset, rollout_prompt,
                                  **kwargs) -> list[float]:
    """correctness_reward, except ChartQA answers also count with Unicode dashes and with
    list items in any order (the normalized scorer marks those right answers 0)."""
    out = []
    for completion, golds, ds, template in zip(completions, answers, dataset, rollout_prompt,
                                               strict=True):
        pred = reward_answer(_text(completion), PROMPTS[template].answer_rule)
        if pred is None:
            out.append(0.0)
        elif ds == "chartqa":
            out.append(chartqa_list_accuracy(pred, golds))
        else:
            out.append(score(ds, pred, golds, "normalized"))
    return out


def _well_formed(text: str) -> bool:
    text = text.strip()
    if text.count("<answer>") != 1 or not text.endswith("</answer>"):
        return False  # exactly one answer tag, nothing after it
    reasoning, _, tail = text.partition("<answer>")
    answer = tail.removesuffix("</answer>").strip()
    # Substantive reasoning first; short clean answer (no tag-stuffing or dumping).
    return (
        len(reasoning.strip()) >= _MIN_REASONING_CHARS
        and 1 <= len(answer) <= 80
        and "<" not in answer
    )


def format_reward(prompts, completions, **kwargs) -> list[float]:
    """1.0 for reasoning prose followed by one short, terminal <answer> tag."""
    return [1.0 if _well_formed(_text(completion)) else 0.0 for completion in completions]


REWARDS = {
    "correctness": correctness_reward,
    "correctness_lists": list_aware_correctness_reward,
    "format": format_reward,
}
