import copy

import pytest

from train_opd import with_reference_answer


@pytest.mark.parametrize("content", ["Question?", [{"type": "text", "text": "Question?"}]])
def test_gold_context_is_added_only_to_teacher_copy(content):
    prompt = [{"role": "user", "content": content}]
    original = copy.deepcopy(prompt)
    teacher = with_reference_answer(prompt, "cat")
    assert prompt == original
    assert "Reference answer: cat" in str(teacher)
    assert teacher is not prompt
