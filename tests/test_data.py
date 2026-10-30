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


def test_held_out_collections_are_whole_seeded_and_leave_each_modality_in_training():
    from vqar.data import held_out_collections

    sizes = {"RadImageNet": 600, "a1": 40, "a2": 30, "a3": 20, "b1": 30, "b2": 20,
             "c1": 50, "d1": 10, "d2": 10}
    modality = {"RadImageNet": "mr", "a1": "a", "a2": "a", "a3": "a", "b1": "b", "b2": "b",
                "c1": "c", "d1": "d", "d2": "d"}
    families = (("a1", "d1"),)
    held = held_out_collections(sizes, modality, seed=0, share=0.1, families=families,
                                keep=("RadImageNet",))
    assert held == held_out_collections(sizes, modality, seed=0, share=0.1,
                                        families=families, keep=("RadImageNet",))
    assert "RadImageNet" not in held and "c1" not in held  # c's only collection stays
    assert ("a1" in held) == ("d1" in held)  # a family moves as one unit
    assert sum(sizes[name] for name in held) >= 0.1 * sum(sizes.values())
    assert len({modality[name] for name in held}) >= 2
    for label in ("a", "b", "d"):
        assert any(modality[n] == label and n not in held for n in sizes)


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


def test_rexvqa_rows_show_question_and_options_but_never_the_report():
    from vqar.data import _rexvqa_answers, _rexvqa_question

    row = {"question": " What device is visible? ", "correct_answer": "B",
           "options": ["A. Endotracheal tube", "B. Right internal jugular catheter",
                       "C. Feeding tube", "D. Chest tube"],
           "Findings": "secret report text", "Impression": "more report text"}
    question = _rexvqa_question(row)
    assert question.splitlines() == ["What device is visible?", "A. Endotracheal tube",
                                     "B. Right internal jugular catheter", "C. Feeding tube",
                                     "D. Chest tube",
                                     "Answer with the letter of the correct option."]
    assert "report" not in question
    assert _rexvqa_answers(row) == ["B", "Right internal jugular catheter"]
    assert _rexvqa_answers({**row, "correct_answer": "E"}) is None


def _xray(path, size=(4, 3)):
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.arange(12, dtype=np.uint16).reshape(3, 4) * 5000, mode="I;16").save(path)
    return path


def test_rexvqa_keeps_one_frontal_view_per_study_and_only_extracted_patients(tmp_path):
    import io

    from PIL import Image

    from vqar.data import _cxr_rgb_png, rexvqa_records

    root = tmp_path / "png"
    lateral = "../deid_png/P1/A1/studies/S1/series/x/instances/lat.png"
    ap = "../deid_png/P1/A1/studies/S1/series/x/instances/ap.png"
    pa = "../deid_png/P1/A1/studies/S1/series/x/instances/pa.png"
    for relative in (lateral, ap, pa):
        _xray(root / relative.removeprefix("../"))
    only_lateral = "../deid_png/P2/A2/studies/S2/series/x/instances/lat.png"
    _xray(root / only_lateral.removeprefix("../"))
    meta = {"s1": {"PatientID": "P1", "ImagePath": [lateral, ap, pa],
                   "ImageViewPosition": ["LATERAL", "AP", "PA"]},
            "s2": {"PatientID": "P2", "ImagePath": [only_lateral],
                   "ImageViewPosition": ["LATERAL"]},
            "s3": {"PatientID": "P3", "ImagePath": [pa], "ImageViewPosition": ["PA"]}}
    row = {"question": "q", "options": ["A. a", "B. b"], "correct_answer": "A",
           "task_name": "Presence Assessment"}
    questions = {"q2": {**row, "study_id": "s1", "PatientID": "P1"},
                 "q1": {**row, "study_id": "s1", "PatientID": "P1"},
                 "q3": {**row, "study_id": "s2", "PatientID": "P2"},
                 "q4": {**row, "study_id": "s3", "PatientID": "P3"}}
    records = rexvqa_records(questions, meta, root, patients={"P1", "P2"})
    assert [r["qid"] for r in records] == ["q1", "q2"]  # P2 has no frontal view, P3 no images
    assert records[0]["file"].endswith("pa.png") and records[0]["answers"] == ["A", "a"]
    with Image.open(io.BytesIO(_cxr_rgb_png(records[0]["file"]))) as image:
        assert image.mode == "RGB"
        assert image.getextrema() == ((0, 255), (0, 255), (0, 255))



def test_deeptumor_options_go_one_per_line_and_answers_hold_letter_and_text():
    from vqar.data import deeptumor_record

    record = deeptumor_record("q1", "BDMAP_1", "How large is the liver, expressed as a volume? "
                              "A: 1293.7 B: 1154.4 C: 1436.0 D: 986.6", "A", "1293.7",
                              "measurement", "scan.png")
    assert record["question"].splitlines() == [
        "How large is the liver, expressed as a volume?", "A. 1293.7", "B. 1154.4", "C. 1436.0",
        "D. 986.6", "Answer with the letter of the correct option."]
    assert record["answers"] == ["A", "1293.7"]
    assert record["patient"] == "BDMAP_1"


def test_deeptumor_drops_multi_answer_single_option_and_mismatched_rows():
    from vqar.data import deeptumor_record

    question = "Which organs have lesions? A: liver B: kidney C: pancreas"
    assert deeptumor_record("q", "p", question, "A,B", "liver", "recognition", "f") is None
    assert deeptumor_record("q", "p", "HU difference? A: 1.5", "A", "1.5", "measurement",
                            "f") is None
    assert deeptumor_record("q", "p", question, "B", "liver", "recognition", "f") is None
    assert deeptumor_record("q", "p", question, "D", "liver", "recognition", "f") is None


def test_deeptumor_dev_patients_are_a_stable_seeded_share():
    from vqar.data import deeptumor_dev_patient

    patients = [f"BDMAP_{i:08d}" for i in range(4000)]
    chosen = [p for p in patients if deeptumor_dev_patient(p, 3, seed=0)]
    assert 80 < len(chosen) < 160
    assert chosen == [p for p in patients if deeptumor_dev_patient(p, 3, seed=0)]
    assert chosen != [p for p in patients if deeptumor_dev_patient(p, 3, seed=1)]


def test_deeptumor_ladder_puts_the_truth_at_every_rank_with_one_format():
    import random
    from collections import Counter

    from vqar.data import deeptumor_ladder

    ranks = Counter()
    for seed in range(400):
        options = deeptumor_ladder("1293.7", 4, "organ volume measurement", random.Random(seed))
        assert len(set(options)) == 4 and all(len(o.split(".")[1]) == 1 for o in options)
        values = [float(o) for o in options]
        assert values == sorted(values)
        ranks[values.index(1293.7)] += 1
    assert set(ranks) == {0, 1, 2, 3} and min(ranks.values()) > 70
    counts = deeptumor_ladder("1", 4, "lesion counting", random.Random(0))
    assert counts in (["0", "1", "2", "3"], ["1", "2", "3", "4"])


def test_deeptumor_ladder_refuses_values_out_of_range_and_signed_values_go_additive():
    import random

    from vqar.data import deeptumor_ladder

    ladders = [deeptumor_ladder("99.5", 4, "tumor burden percentage", random.Random(s))
               for s in range(50)]
    assert any(ladder is None for ladder in ladders)  # ranks below the top would pass 100
    assert all(max(map(float, ladder)) <= 100 for ladder in ladders if ladder)
    signed = deeptumor_ladder("-4.2", 4, "tumor organ HU difference", random.Random(3))
    values = [float(v) for v in signed]
    gaps = {round(b - a, 1) for a, b in zip(values, values[1:], strict=False)}
    assert len(gaps) <= 2 and 4.5 <= min(gaps) <= 15.5  # one HU step, up to rounding


def test_deeptumor_debias_shuffles_and_keeps_the_answer_consistent():
    from vqar.data import deeptumor_debias, deeptumor_record

    record = deeptumor_record("q7", "p", "Clustering? A: Highly clustered B: Somewhat clustered "
                              "C: Widely distributed", "A", "Highly clustered",
                              "visual reasoning", "f", "liver lesion clustering")
    out = deeptumor_debias(record, seed=0)
    letter, text = out["answers"]
    assert sorted(out["options"]) == sorted(record["options"]) and text == "Highly clustered"
    assert out["options"][ord(letter) - ord("A")] == text
    assert f"{letter}. {text}" in out["question"]
    assert out == deeptumor_debias(record, seed=0)


def test_deeptumor_balance_equalizes_discrete_answers_and_passes_continuous_values():
    from collections import Counter

    from vqar.data import deeptumor_balance

    def rec(qid, subtype, answer, options):
        return {"qid": qid, "subtype": subtype, "answers": ["A", answer], "options": options}

    records = ([rec(f"y{i}", "colon lesion existence", "Yes", ["Yes", "No"]) for i in range(30)]
               + [rec(f"n{i}", "colon lesion existence", "No", ["No", "Yes"]) for i in range(170)]
               + [rec(f"r{i}", "rare grade", "Rare", ["Rare", "Common"]) for i in range(3)]
               + [rec(f"c{i}", "rare grade", "Common", ["Common", "Rare"]) for i in range(40)]
               + [rec(f"v{i}", "organ volume measurement", str(100 + i), ["1", "2", "3", "4"])
                  for i in range(25)])
    kept = deeptumor_balance(records, min_per_answer=10, seed=0)
    by = Counter((r["subtype"], r["answers"][1]) for r in kept)
    assert by[("colon lesion existence", "Yes")] == by[("colon lesion existence", "No")] == 30
    assert not any(subtype == "rare grade" for subtype, _ in by)  # one answer left: dropped
    assert sum(1 for r in kept if r["subtype"] == "organ volume measurement") == 25
