"""Reward functions for GRPO (RLVR). TRL calling convention:

    fn(prompts=..., completions=..., completion_ids=..., **dataset_columns) -> list[float]

Extra columns of the RL dataset (question, answers, dataset) arrive as kwargs, each a
list aligned with completions. Rewards return raw 0/1 signals; relative weighting
(e.g. format 0.2) is set via GRPOConfig.reward_weights, so ablating a reward is a
config change, not a code change.
"""

import re
import warnings

from vqar.inference import extract_answer
from vqar.judge import Judge, JudgeBudgetExceeded
from vqar.metrics import score

# <think> reasoning </think> then a short final answer, nothing after it.
_FORMAT = re.compile(r"^\s*<think>.+?</think>\s*<answer>[^<>]{1,80}</answer>\s*$", re.DOTALL)


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]  # conversational format


def correctness_reward(prompts, completions, answers, dataset, **kwargs) -> list[float]:
    """Verifiable per-dataset match: official VQA accuracy / normalized EM (in [0, 1])."""
    return [
        score(ds, extract_answer(_text(completion)), golds)
        for completion, golds, ds in zip(completions, answers, dataset, strict=True)
    ]


def format_reward(prompts, completions, **kwargs) -> list[float]:
    """1.0 for a well-formed <think>...</think><answer>...</answer> completion.

    The answer slot is length-capped so dumping candidate answers into it scores 0.
    """
    return [1.0 if _FORMAT.match(_text(completion)) else 0.0 for completion in completions]


def make_judge_reward(judge: Judge):
    """Correctness with LLM-judge rescue: rule-based match first; on failure, Claude
    judges semantic equivalence. Degrades to plain correctness if the budget cap hits.
    """
    state = {"budget_exhausted": False}

    def judge_reward(prompts, completions, answers, dataset, question, **kwargs) -> list[float]:
        out = []
        for completion, golds, ds, q in zip(completions, answers, dataset, question, strict=True):
            pred = extract_answer(_text(completion))
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
