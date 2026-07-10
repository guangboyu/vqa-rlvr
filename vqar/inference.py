"""Generation helpers and answer extraction.

Answer extraction is shared by eval and RL rewards: models trained with the
`reasoning` template answer inside <answer></answer> tags; models using the
`short` template answer directly. Both are scored identically after extraction.
"""

import re

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str) -> str:
    """Return the content of the last <answer> tag, or the stripped full text."""
    matches = _ANSWER_TAG.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()
