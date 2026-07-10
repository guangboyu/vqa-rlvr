from vqar.inference import extract_answer


def test_plain_text_passthrough():
    assert extract_answer("  blue  ") == "blue"


def test_answer_tag():
    assert extract_answer("<think>hmm</think><answer>4</answer>") == "4"


def test_last_tag_wins():
    assert extract_answer("<answer>3</answer> wait <answer>4</answer>") == "4"


def test_multiline_answer():
    assert extract_answer("<answer>\nred\n</answer>") == "red"


def test_unclosed_tag_falls_back_to_full_text():
    text = "<answer>4"
    assert extract_answer(text) == "<answer>4"
