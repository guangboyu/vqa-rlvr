"""Reward functions for GRPO (RLVR). TRL calling convention:

    fn(prompts=..., completions=..., completion_ids=..., **dataset_columns) -> list[float]

Extra columns of the RL dataset (question, answers, dataset) arrive as kwargs, each a
list aligned with completions. Rewards return raw 0/1 signals; relative weighting
(e.g. format 0.2) is set via GRPOConfig.reward_weights, so ablating a reward is a
config change, not a code change.
"""

import re
import warnings

from vqar.judge import Judge, JudgeBudgetExceeded
from vqar.metrics import score

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Reasoning followed by one short, terminal <answer> tag. Deliberately NOT literal
# <think> tags: Qwen3-VL writes its chain of thought as plain prose and never samples
# literal think tags, so a think-tag reward is unearnable (flat 0 over 400 GRPO steps
# from BOTH the base and SFT policies). Reward the behavior, not magic tokens.
_MIN_REASONING_CHARS = 100


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]  # conversational format


def _strict_answer(text: str) -> str | None:
    """RL-side extraction is STRICT: no <answer> tag, no reward. The eval-side
    fallback (score the raw text) let the CLEVR sandbox policy drift to bare
    digits — RL happily discards formats that aren't load-bearing for reward."""
    matches = _ANSWER_TAG.findall(text)
    return matches[-1].strip() if matches else None


def correctness_reward(prompts, completions, answers, dataset, **kwargs) -> list[float]:
    """Verifiable per-dataset match: official VQA accuracy / normalized EM (in [0, 1])."""
    out = []
    for completion, golds, ds in zip(completions, answers, dataset, strict=True):
        pred = _strict_answer(_text(completion))
        out.append(0.0 if pred is None else score(ds, pred, golds))
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


def make_judge_reward(judge: Judge):
    """Correctness with LLM-judge rescue: rule-based match first; on failure, Claude
    judges semantic equivalence. Degrades to plain correctness if the budget cap hits.
    """
    state = {"budget_exhausted": False}

    def judge_reward(prompts, completions, answers, dataset, question, **kwargs) -> list[float]:
        out = []
        for completion, golds, ds, q in zip(completions, answers, dataset, question, strict=True):
            pred = _strict_answer(_text(completion))
            if pred is None:
                out.append(0.0)
                continue
            base = score(ds, pred, golds)
            if base > 0 or state["budget_exhausted"] or not pred:
                out.append(base)
                continue
            try:
                out.append(1.0 if judge.judge(q, golds, pred) else 0.0)
            except JudgeBudgetExceeded:
                state["budget_exhausted"] = True
                warnings.warn(
                    "judge budget exhausted; falling back to rule-based rewards",
                    stacklevel=2,
                )
                out.append(base)
        return out

    return judge_reward


REWARDS = {
    "correctness": correctness_reward,
    "format": format_reward,
}
