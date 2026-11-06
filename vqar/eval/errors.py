"""Error analysis from saved predictions: flips between two arms, an error taxonomy, and
answer-form checks. `scripts/error_analysis.py` applies them to every declared arm; every
definition it uses is here.

Right: a question's score is at least RIGHT_THRESHOLD. On exact-match datasets that is a
match; on VQAv2 and TextVQA (ten annotators) it means at least two annotators gave the
answer (accuracy 0.6), while one annotator alone (0.3) counts as wrong.

The answer-form checks look for ways to game an answer checker (hedging, several answers,
dumping text), so they read the answer the RL correctness reward reads: under the reasoning
prompt the last <answer> tag, and a response without one is never flagged (the taxonomy
counts it); under the short prompt the whole response.

The taxonomy and the answer-form checks are heuristics for triage, not scores: they read
the answer text with simple rules and can misfile, for example, a synonym as wrong content
or a fraction such as "1/2" as a hedge.
"""

import re

from vqar.answers import reward_answer
from vqar.registry import PROMPTS

RIGHT_THRESHOLD = 0.5

# Why a response scored wrong, checked in this order; the first that applies wins.
ERROR_KINDS = {
    "scorer_only": "wrong under the official scorer, right under the normalized one "
                   "(casing or punctuation)",
    "truncated": "cut off at the token budget",
    "no_answer_tag": "no closing </answer> tag, under a prompt that asks for one",
    "partial": "the lowercased answer and a reference contain one another as text, e.g. a "
               "minority annotator's answer or 'tan' for 'light tan'",
    "wrong_content": "anything else",
}

# Answer forms a few random samples cannot hit by chance.
OPEN_FORMS = ("number", "text")

HEDGE = re.compile(r"\bor\b|/")
LONG_ANSWER_WORDS = 6


def hedged(answer: str | None, raw: str) -> bool:
    """The answer contains the word 'or' or a slash, e.g. 'red or blue', 'red/blue'."""
    return answer is not None and bool(HEDGE.search(answer.lower()))


def multiple_answer_tags(answer: str | None, raw: str) -> bool:
    """The response opens more than one <answer> tag."""
    return raw.count("<answer>") > 1


def long_answer(answer: str | None, raw: str) -> bool:
    """The answer has more than six words."""
    return answer is not None and len(answer.split()) > LONG_ANSWER_WORDS


ANSWER_FORM_CHECKS = {check.__name__: check
                      for check in (hedged, multiple_answer_tags, long_answer)}


def is_right(score: float) -> bool:
    return score >= RIGHT_THRESHOLD


def flip_counts(reference: dict[str, float], candidate: dict[str, float]) -> dict:
    """Paired by question, how the verdict moved from the reference arm to the candidate:
    wrong_to_right (fixed), right_to_wrong (broken), and the questions that kept theirs."""
    if reference.keys() != candidate.keys():
        raise ValueError("Flips need the same questions in both records.")
    counts = {"n": len(reference), "wrong_to_right": 0, "right_to_wrong": 0,
              "right_to_right": 0, "wrong_to_wrong": 0}
    for qid, score in reference.items():
        before, after = is_right(score), is_right(candidate[qid])
        counts[f"{'right' if before else 'wrong'}_to_{'right' if after else 'wrong'}"] += 1
    return counts


def error_kind(line: dict, prompt: str) -> str | None:
    """The ERROR_KINDS entry of one prediction line, or None when it is right.

    Right means right under the official scorer, or under the normalized one where the
    official scorer cannot score the references.
    """
    official, normalized = line["scores"]["official"], line["scores"]["normalized"]
    if is_right(normalized if official is None else official):
        return None
    if is_right(normalized):
        return "scorer_only"
    if line["finish_reason"] == "length":
        return "truncated"
    if PROMPTS[prompt].answer_rule == "tagged" and "</answer>" not in line["raw"]:
        return "no_answer_tag"
    answer = line["prediction"].lower().strip()
    references = [reference.lower().strip() for reference in line["answers"]]
    if answer and any(answer in reference or reference in answer for reference in references):
        return "partial"
    return "wrong_content"


def error_taxonomy(lines: list[dict], prompt: str) -> dict:
    """The share of all predictions in each error kind, and the share that is right."""
    kinds = [error_kind(line, prompt) for line in lines]
    shares = {kind: kinds.count(kind) / len(kinds) for kind in ERROR_KINDS}
    return {"right": kinds.count(None) / len(kinds), **shares}


def answer_form(lines: list[dict], prompt: str) -> dict:
    """The share of all responses each answer-form check flags."""
    rule = PROMPTS[prompt].answer_rule
    readings = [(reward_answer(line["raw"], rule), line["raw"]) for line in lines]
    return {name: sum(check(*reading) for reading in readings) / len(lines)
            for name, check in ANSWER_FORM_CHECKS.items()}


def definitions() -> dict:
    """Every rule above in words, for the JSON output."""
    return {"right": f"score >= {RIGHT_THRESHOLD}", "error_kinds": ERROR_KINDS,
            "answer_form": {name: check.__doc__ for name, check in ANSWER_FORM_CHECKS.items()},
            "answer_form_reads": "the answer the RL correctness reward reads under the "
                                 "prompt's answer rule (vqar.answers.reward_answer)"}
