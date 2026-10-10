import pytest
from datasets import Dataset

from vqar.data import _clean_cauldron_answer, _vqav2_assignments


def rows_with_qa_counts(counts):
    """Fake cauldron rows: each row i has counts[i] QA pairs (no images needed)."""
    return Dataset.from_list(
        [{"texts": [{"user": f"q{i}-{j}\nSuffix.", "assistant": "A."} for j in range(n)]}
         for i, n in enumerate(counts)]
    )


class TestCauldronAnswer:
    def test_strips_period_and_lowercases(self):
        assert _clean_cauldron_answer("Net.") == "net"

    def test_keeps_decimals_intact(self):
        assert _clean_cauldron_answer("1.50") == "1.50"


class TestAssignments:
    def test_quotas_filled_exactly(self):
        rows = rows_with_qa_counts([3, 3, 3, 3])
        out = _vqav2_assignments(rows, n_sft=4, n_rl=3)
        assert len(out["sft"]) == 4
        assert len(out["rl"]) == 3

    def test_image_disjoint(self):
        rows = rows_with_qa_counts([3] * 10)
        out = _vqav2_assignments(rows, n_sft=7, n_rl=5)
        sft_rows = {i for i, _ in out["sft"]}
        rl_rows = {i for i, _ in out["rl"]}
        assert sft_rows.isdisjoint(rl_rows)

    def test_image_never_split_across_buckets(self):
        # SFT quota fills mid-row: that row's remaining QAs must be skipped, not
        # given to RL.
        rows = rows_with_qa_counts([2, 3, 2, 2])
        out = _vqav2_assignments(rows, n_sft=3, n_rl=2)
        assert (1, 0) in out["sft"] and (1, 1) not in out["sft"]
        assert all(i != 1 for i, _ in out["rl"])

    def test_raises_when_source_exhausted(self):
        rows = rows_with_qa_counts([2, 2])
        with pytest.raises(ValueError, match="exhausted"):
            _vqav2_assignments(rows, n_sft=3, n_rl=3)



def test_cauldron_clevr_questions_drop_the_style_instruction_and_keep_rgb_images():
    import io

    from PIL import Image as PILImage

    from vqar.data import _clevr_questions

    buffer = io.BytesIO()
    PILImage.new("RGBA", (4, 4), (10, 20, 30, 255)).save(buffer, format="PNG")
    row = {"images": [{"bytes": buffer.getvalue(), "path": None}],
           "texts": [{"user": "Are there more cubes than spheres?\nAnswer briefly.",
                      "assistant": "Yes.", "source": "clevr"},
                     {"user": "How many cylinders are there?\nBe short.", "assistant": "3.",
                      "source": "clevr"},
                     {"user": "What material is the cube?\nOne word.", "assistant": "Rubber.",
                      "source": "clevr"}]}
    questions = _clevr_questions(row, 0, 7, [0, 1, 2])
    assert [q["question"] for q in questions] == [
        "Are there more cubes than spheres?", "How many cylinders are there?",
        "What material is the cube?"]
    assert [q["answers"] for q in questions] == [["yes"], ["3"], ["rubber"]]
    assert [q["answer_type"] for q in questions] == ["yes/no", "number", "attribute"]
    assert questions[1]["qid"] == "0-7-1"
    with PILImage.open(io.BytesIO(questions[0]["image"])) as image:
        assert image.mode == "RGB"


def test_chartqa_answer_types():
    from vqar.data import _chartqa_answer_type

    assert _chartqa_answer_type("12.5") == "number"
    assert _chartqa_answer_type("1,234") == "number"
    assert _chartqa_answer_type("Blue") == "text"


def test_kvasir_atomic_rows_show_the_labels_own_question():
    from vqar.data import _kvasir_atomic

    rows = [{"img_id": "a", "complexity": 1, "question_class": ["instrument_location"],
             "original": '[{"q": "Where in the image is the instrument?", "a": "none"}]'},
            {"img_id": "a", "complexity": 2, "question_class": ["x", "y"],
             "original": '[{"q": "q1", "a": "1"}, {"q": "q2", "a": "2"}]'}]
    (record,) = _kvasir_atomic(rows, "train")
    assert record["question"] == "Where in the image is the instrument?"
    assert record["answers"] == ["none"] and record["answer_type"] == "instrument_location"
    assert record["qid"] == "train-c1-0"


def test_omnimed_items_become_lettered_choices_or_are_dropped():
    from vqar.data import _omnimed_record

    row = {"question_id": "X_1", "question": "Which organ?", "gt_answer": "Liver",
           "option_A": "Kidney", "option_B": "Liver", "option_C": "Lung", "option_D": None,
           "image_path": "Images/X/1.png", "question_type": "Anatomy Identification"}
    record = _omnimed_record(row)
    assert record["answers"] == ["B", "Liver"]
    assert record["question"].splitlines()[1:] == [
        "A. Kidney", "B. Liver", "C. Lung", "Answer with the letter of the correct option."]
    assert record["member"] == "OmniMedVQA/Images/X/1.png"
    assert _omnimed_record({**row, "gt_answer": "Heart"}) is None


def test_re_split_keeps_every_question_of_an_image_together():
    from vqar.data import _split_by_image

    records = [{"key": f"img{i // 3}", "qid": str(i)} for i in range(300)]
    splits = _split_by_image(records, seed=0, held_out=0.1)
    assert [len({r["key"] for r in splits[s]}) for s in ("train", "val", "test")] == [80, 10, 10]
    owners = {}
    for name, part in splits.items():
        for record in part:
            assert owners.setdefault(record["key"], name) == name
    assert _split_by_image(records, seed=0, held_out=0.1) == splits
