"""How an answer is read out of a response: one lenient rule for evaluation, and for the RL
reward the rule its rollout prompt template declares.

- Evaluation: the last <answer> tag, else the whole trimmed response, for every template.
- Reward, "tagged" templates (reasoning asks for tags): the last <answer> tag, or no
  answer at all. RL drops any format that earns nothing, so a lenient reward rule let a
  CLEVR sandbox policy drift to bare digits.
- Reward, "plain" templates (short asks for one word or phrase): the whole trimmed
  response, since that template never asks for tags.
"""

import re

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ANSWER_RULES = ("tagged", "plain")
YES_NO = {"yes", "no", "true", "false"}
NUMBER = re.compile(r"[-+]?[\d,]*\.?\d+%?")


def reference_form(answers: list[str]) -> str:
    """The most common reference's form: yes/no, choice, number or text.

    `answer_type` means something different in each dataset; this one tag does not. A choice
    reference is an uppercase option letter followed by at least one more entry, which is how
    multiple-choice loaders store it. A capital letter alone is an open answer.
    """
    if len(answers) > 1 and re.fullmatch(r"[A-Z]", answers[0].strip()):
        return "choice"
    cleaned = [answer.strip().rstrip(".").lower() for answer in answers]
    common = max(cleaned, key=cleaned.count) if cleaned else ""
    if common in YES_NO:
        return "yes/no"
    return "number" if NUMBER.fullmatch(common) else "text"


def extract_answer(text: str) -> str:
    """Evaluation's reading: the content of the last <answer> tag, or the stripped text."""
    matches = ANSWER_TAG.findall(text)
    return matches[-1].strip() if matches else text.strip()


def reward_answer(text: str, rule: str) -> str | None:
    """The reward's reading under an answer rule; None means no answer was given."""
    if rule == "plain":
        return text.strip()
    if rule == "tagged":
        matches = ANSWER_TAG.findall(text)
        return matches[-1].strip() if matches else None
    raise ValueError(f"Unknown answer rule {rule!r}; expected one of {ANSWER_RULES}")
