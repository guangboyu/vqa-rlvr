"""Public benchmarks: scorers on each benchmark's own rules, loader rows from tiny source rows,
clean and published roles, and how their records reach the tables. No raw data is read."""

import base64
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset
from fixtures import png, save_subset, write_splits
from PIL import Image as PILImage

from vqar import data
from vqar.benchmark_metrics import (
    charxiv_accuracy,
    docvqa_anls,
    is_choice,
    levenshtein,
    mathvista_accuracy,
    mmmu_accuracy,
    mmstar_accuracy,
    normalized_choice_accuracy,
    normalized_mixed_accuracy,
    numbers_in,
    ocrbench_accuracy,
    option_accuracy,
    pope_accuracy,
    pope_reading,
    pope_summary,
    read_number,
    yes_no_accuracy,
)
from vqar.eval import ModelRef, Protocol, run_eval
from vqar.eval.examples import select_rows, to_example
from vqar.eval.protocol import protocol_identity
from vqar.eval.record import RunRecord, record_files
from vqar.eval.report import is_current, load_cells
from vqar.registry import CATEGORIES, DATASETS, LADDER
from vqar.splits import (
    build_split,
    clean_rows,
    extra_image_keys,
    gray_key,
    near_duplicate_index,
    near_duplicate_pools,
    near_duplicate_rows,
    near_key,
    overlap_report,
    pixel_key,
    question_key,
    question_stem,
    read_split,
    role_keys,
    split_path,
    subset_gray_keys,
    validate_split,
)

BENCHMARKS = {"mmstar": "General VQA", "mathvista": "Reasoning", "mmmu": "Reasoning",
              "charxiv": "Document-chart", "ocrbench": "Document-chart",
              "docvqa": "Document-chart", "vstar": "Perception", "hrbench4k": "Perception",
              "pope": "Hallucination", "hallusionbench": "Hallucination"}
PROJECT_ONLY = {"mathvista", "charxiv", "vstar", "hrbench4k", "hallusionbench"}
CITIES = data.choice_answers("C", ["London", "Rome", "Paris", "Oslo"])


class TestMmstar:
    @pytest.mark.parametrize("prediction, expected", [
        ("C", 1), ("c.", 1), ("(C) Paris", 1), ("Option C", 1), ("The answer is C.", 1),
        ("A", 0), ("I think C", 0), ("", 0), ("Paris", 0),
        ("cats", 1),  # the authors read only the first character
    ])
    def test_the_letter_must_open_the_response(self, prediction, expected):
        assert mmstar_accuracy(prediction, CITIES) == expected


class TestOptionReader:
    @pytest.mark.parametrize("prediction, expected", [
        ("C", 1), ("(C)", 1), ("C. Paris", 1), ("The answer is C.", 1), ("Paris", 1),
        ("It is in Paris.", 1), ("A", 0), ("Oslo", 0), ("Madrid", 0), ("", 0),
        ("A or C", 0),  # two letters and no option text: nothing can be read
        ("c", 0),  # letters are read in capitals only, as in can_infer
    ])
    def test_one_letter_named_or_one_option_contained(self, prediction, expected):
        assert option_accuracy(prediction, CITIES) == expected

    def test_either_copy_of_a_repeated_option_counts(self):
        repeated = data.choice_answers("C", ["9", "12", "18", "18"])  # MathVista pid 781
        assert option_accuracy("C", repeated) == option_accuracy("D", repeated) == 1
        assert option_accuracy("B", repeated) == 0


class TestNormalizedChoice:
    @pytest.mark.parametrize("prediction, expected", [
        ("c", 1), ("(C) Paris", 1), ("option c", 1), ("paris.", 1), ("A", 0), ("", 0),
        ("Rome", 0),  # another option's text never matches
        ("I think", 0),  # I is not an option of this row
    ])
    def test_rows_read_their_own_letters_and_the_reference_text_only(self, prediction, expected):
        assert normalized_choice_accuracy(prediction, CITIES) == expected

    def test_letters_past_e(self):
        nine = data.choice_answers("I", [f"t{i}" for i in range(9)])
        assert normalized_choice_accuracy("I", nine) == 1
        assert normalized_choice_accuracy("F", nine) == 0

    def test_open_rows_use_normalized_exact_match(self):
        assert normalized_mixed_accuracy("The answer.", ["answer"]) == 1
        assert normalized_mixed_accuracy("Paris", CITIES) == 1


def test_numbers_read_in_the_notations_benchmarks_use():
    assert read_number("40,000") == 40000
    assert read_number("1e+06") == 1e6
    assert read_number("10^-1") == pytest.approx(0.1)
    assert read_number("1/64") == 1 / 64
    assert read_number("94%") == 94
    assert read_number("$5.") == 5
    assert read_number("1/0") is None and read_number("Yes") is None and read_number("nan") is None
    assert numbers_in("5 cm^2, then 2-3 and x2") == [5, 2, 3]


class TestMathvista:
    @pytest.mark.parametrize("prediction, reference, expected", [
        ("14", "14", 1), ("14 cm", "14", 1), ("The area is 14 cm^2.", "14", 1),
        ("14.7", "14", 1),  # the authors truncate: str(int(float(x)))
        ("15", "14", 0), ("no idea", "14", 0), ("1,000", "1000", 1),
        ("3.14159", "3.14", 1), ("3.1", "3.14", 0), ("about 3.14", "3.14", 1),
        ("0", "0.0", 1), ("x = -4.3", "-4.3", 1), ("4.3", "-4.3", 0),
        ("[2014, 2016]", "[2014, 2016]", 1), ("2014, 2016", "[2014, 2016]", 1),
        ("[2016, 2014]", "[2014, 2016]", 0),
    ])
    def test_free_form_by_the_references_type(self, prediction, reference, expected):
        assert mathvista_accuracy(prediction, [reference]) == expected

    def test_multiple_choice_reads_the_option(self):
        assert mathvista_accuracy("(C) Paris", CITIES) == 1
        assert mathvista_accuracy("Rome", CITIES) == 0


class TestMmmu:
    PRICES = data.choice_answers("B", ["$6", "$7", "$8", "$9"])

    @pytest.mark.parametrize("prediction, expected", [
        ("(B)", 1), ("B", 1), ("B.", 1), ("The answer is (B).", 1), ("(A) or (B)", 1),
        ("(B) then (A)", 0),  # the last one named wins
        ("the correct value here is about $7 per unit", 1),  # text is read past five tokens
        ("the correct value here is about $6 per unit", 0),
    ])
    def test_multiple_choice_parser(self, prediction, expected):
        assert mmmu_accuracy(prediction, self.PRICES) == expected

    def test_an_unparsable_choice_is_a_guess_that_replays_identically(self):
        scores = [mmmu_accuracy(f"unclear {i}", self.PRICES) for i in range(400)]
        assert scores == [mmmu_accuracy(f"unclear {i}", self.PRICES) for i in range(400)]
        assert 0.15 < sum(scores) / len(scores) < 0.35  # four options: a quarter on average

    @pytest.mark.parametrize("prediction, references, expected", [
        ("so the answer is 1.06.", ["1.06"], 1), ("1.0600", ["1.06"], 1), ("1.1", ["1.06"], 0),
        ("x = 5", ["5"], 1), ("The process is called transformation", ["Transformation"], 1),
        ("translation", ["Transformation"], 0), ("The answer is C", ["C"], 1),
        ("clearly", ["C"], 0),  # a one-character reference must stand alone
        ("It is Florida", ["Tampa", "Florida"], 1), ("Georgia", ["Tampa", "Florida"], 0),
    ])
    def test_open_questions(self, prediction, references, expected):
        assert mmmu_accuracy(prediction, references) == expected

    def test_an_open_reference_that_is_a_letter_is_not_a_choice(self):
        assert not is_choice(["C"]) and not is_choice(["Tampa", "Florida"])
        assert is_choice(self.PRICES)


class TestCharxiv:
    @pytest.mark.parametrize("prediction, reference, expected", [
        ("40000", "40,000", 1), ("The value is 40,000.", "40,000", 1), ("4e4", "40,000", 1),
        ("39000", "40,000", 0),  # no percentage tolerance: the authors grade numbers as exact
        ("0.1", "10^-1", 1), ("10^-1", "10^-1", 1), ("10^-2", "10^-1", 0),
        ("1000000", "1e+06", 1), ("85.5", "85.5%", 1), ("85.5%", "85.5%", 1), ("85", "85.5%", 0),
        ("Yes", "94", 0),
    ])
    def test_numbers_compare_by_value(self, prediction, reference, expected):
        assert charxiv_accuracy(prediction, [reference]) == expected

    @pytest.mark.parametrize("prediction, reference, expected", [
        ("joint-cnn", "Joint-CNN", 1), ("Joint CNN", "Joint-CNN", 1), ("b", "(b)", 1),
        ("(c)", "(b)", 0), ("a", "(a)", 1), ("(b)", "(a)", 0), ("", "(a)", 0),
        ("SRLSTM, MomentumLSTM", "MomentumLSTM, SRLSTM", 1), ("40, 30", "30, 40", 1),
        ("Accelerated DATs", "Accelerated DAT", 1),  # 90% of the longer string
        ("round robin combo approach", "ROUND-ROBIN-COMBO", 0),  # below 90%
    ])
    def test_text_compares_normalized(self, prediction, reference, expected):
        assert charxiv_accuracy(prediction, [reference]) == expected


class TestOcrbench:
    FORMULA = ["x > - \\frac { 5 } { 2 }\n"]

    @pytest.mark.parametrize("prediction, references, expected", [
        ("The word is centre.", ["CENTRE"], 1), ("cent", ["CENTRE"], 0), ("", ["CENTRE"], 0),
        ("new york city", ["NY", "New York"], 1),
        ("x>-\\frac{5}{2}", FORMULA, 1),
        ("X>-\\frac{5}{2}", FORMULA, 0),  # formulas keep case
        ("x > - \\frac {5} {2} is it", FORMULA, 1),
    ])
    def test_a_reference_is_a_substring_of_the_response(self, prediction, references, expected):
        assert ocrbench_accuracy(prediction, references) == expected


class TestAnls:
    def test_levenshtein(self):
        assert levenshtein("kitten", "sitting") == 3 and levenshtein("", "abc") == 3

    @pytest.mark.parametrize("prediction, references, expected", [
        ("0.28", ["0.28"], 1.0), ("0.29", ["0.28"], 0.75), ("wrong", ["0.28"], 0.0),
        ("1", ["one", "1"], 1.0), ("  HELLO   World ", ["hello world"], pytest.approx(1 - 0)),
        ("abxx", ["abcd"], 0.5),  # at the threshold
        ("axxx", ["abcd"], 0.0),  # below it
    ])
    def test_best_reference_above_the_threshold(self, prediction, references, expected):
        assert docvqa_anls(prediction, references) == expected


class TestPope:
    @pytest.mark.parametrize("prediction, expected", [
        ("Yes", "yes"), ("No", "no"), ("no.", "no"), ("There is not a dog", "no"),
        ("Yes, there is no dog", "no"), ("Yes. No.", "yes"),
        ("", "yes"), ("NO", "yes"),  # the authors' rule reads these as yes
    ])
    def test_the_authors_reading(self, prediction, expected):
        assert pope_reading(prediction) == expected
        assert pope_accuracy(prediction, [expected]) == 1

    def test_summary_takes_yes_as_the_positive_class(self):
        lines = [{"prediction": said, "answers": [gold]} for said, gold in
                 (("Yes", "yes"), ("Yes", "yes"), ("No", "yes"), ("Yes", "no"), ("No", "no"))]
        summary = pope_summary(lines)["yes_no"]
        assert summary["precision"] == pytest.approx(2 / 3)
        assert summary["recall"] == pytest.approx(2 / 3)
        assert summary["f1"] == pytest.approx(2 / 3)
        assert summary["yes_ratio"] == pytest.approx(3 / 5)


@pytest.mark.parametrize("prediction, reference, expected", [
    ("Yes.", "yes", 1), ("No, it is not", "no", 1), ("no", "yes", 0), ("Unsure", "yes", 0),
    ("", "no", 0), ("yesterday", "yes", 0), ("Nope", "no", 0),
])
def test_hallusionbench_reads_the_first_yes_or_no(prediction, reference, expected):
    assert yes_no_accuracy(prediction, [reference]) == expected


class TestRegistry:
    @pytest.mark.parametrize("name", sorted(BENCHMARKS))
    def test_a_benchmark_is_test_only_with_a_declared_primary_scorer(self, name):
        spec = DATASETS[name]
        assert set(spec.subsets) == {"test", "published"}
        assert spec.subsets["test"] == spec.subsets["published"] == f"{name}_eval"
        assert spec.ladder == "benchmark" and not spec.require_split
        assert spec.category == BENCHMARKS[name]
        assert spec.sizes["rows"] > 0
        assert callable(spec.normalized)
        primary = spec.metric_metadata["primary"]
        assert primary == ("project" if name in PROJECT_ONLY else "official")
        assert spec.metric_metadata[primary]["source"]
        assert (spec.official is None) == (name in PROJECT_ONLY)
        assert spec.official_roles == (frozenset() if name in PROJECT_ONLY
                                       else {"test", "published"})

    def test_every_dataset_has_a_category(self):
        assert {spec.category for spec in DATASETS.values()} == set(CATEGORIES)
        medical = {name for name, spec in DATASETS.items() if spec.category == "Medical"}
        assert {"medthink", "rexvqa", "m3dvqa"} <= medical and "chartqa" not in medical
        assert DATASETS["chartqa"].category == "Document-chart"
        with pytest.raises(ValueError, match="unknown category"):
            replace(DATASETS["vqav2"], category="Vision")
        assert "benchmark" in LADDER

    def test_category_is_outside_a_protocols_identity(self, monkeypatch):
        protocol = Protocol("vqav2")
        before = protocol_identity(protocol, {"subsets": {}})
        assert "category" not in json.dumps(before)
        spec = DATASETS["vqav2"]
        monkeypatch.setitem(DATASETS, "vqav2", replace(spec, category="Medical"))
        after = protocol_identity(protocol, {"subsets": {}})
        assert Protocol.id_of(after) == Protocol.id_of(before)

    def test_committed_records_keep_their_protocol_ids(self):
        """Each current record's identity rebuilt from today's registry hashes as it did."""
        root = Path(__file__).resolve().parent.parent / "results" / "records"
        records = [RunRecord.load(path) for path in record_files(root)]
        current = [record for record in records if not record.legacy]
        assert current, "the repository's records are missing"
        for record in current:
            stored = record.protocol
            protocol = Protocol(stored["dataset"], stored["prompt"], stored["role"],
                                stored["tier"], stored["image_mode"], stored["max_tokens"],
                                stored["temperature"], stored["samples"], stored.get("seed", 0),
                                stored["max_pixels"], screening=True)
            rebuilt = protocol_identity(protocol, stored["data"])
            assert Protocol.id_of(rebuilt) == record.protocol_id, record.path


class TestLoaderRows:
    """Each loader's row function on a source row of the shape its parquet or JSON holds."""

    def test_mmstar_reads_both_option_shapes(self):
        assert data.mmstar_options("Q?\nOptions: A: x, y., B: z, C: nan, D: nan") == [
            "x, y.", "z", "nan", "nan"]
        assert data.mmstar_options("Hint: h\nQuestion: Q?\nChoices:\n(A) 1\n(B) 2\n(C) 3") == [
            "1", "2", "3"]
        assert data.mmstar_options("Hint: h\nQuestion: Q?\n(A) 54\n(B) 52") == ["54", "52"]
        with pytest.raises(ValueError, match="unreadable options"):
            data.mmstar_options("Q? Choose one.")

    def test_mmstar_row_adds_the_instruction_only_where_missing(self):
        row = {"index": 7, "question": "Q?\nOptions: A: x, B: y", "answer": "B", "category": "math",
               "image": png(1)}
        built = data.mmstar_row(row, 0)
        assert built["qid"] == "7" and built["answers"] == ["B", "y", "x", "y"]
        assert built["question"].endswith(data.OPTION_INSTRUCTION)
        question = "Hint: give the correct option letter\nQuestion: Q?\n(A) x\n(B) y"
        assert data.mmstar_row({**row, "question": question}, 0)["question"] == question

    def test_mathvista_row_maps_the_answer_to_its_letter_and_keeps_the_authors_query(self):
        row = {"pid": "3", "decoded_image": {"bytes": png(1)}, "choices": ["135", "140"],
               "precision": None, "answer": "140", "question_type": "multi_choice",
               "answer_type": "text", "query": "Hint: h\nQuestion: q\nChoices:\n(A) 135\n(B) 140"}
        built = data.mathvista_row(row, 0)
        assert built["answers"] == ["B", "140", "135", "140"]
        assert built["answer_type"] == "multi_choice" and built["question"] == row["query"]

    def test_mathvista_row_refuses_a_precision_the_reference_does_not_show(self):
        row = {"pid": "1", "decoded_image": {"bytes": png(1)}, "choices": None, "precision": 2.0,
               "answer": "1.2", "question_type": "free_form", "answer_type": "float", "query": "q"}
        with pytest.raises(ValueError, match="precision"):
            data.mathvista_row(row, 0)

    def test_mmmu_row_keeps_every_stored_image_and_parses_list_answers(self):
        images = {f"image_{i}": None for i in range(1, 8)}
        images.update(image_1={"bytes": png(1)}, image_2={"bytes": png(2)})
        choice = {"id": "v1", "question": "<image 1> Q?", "options": "['x', 'y']", "answer": "B",
                  "question_type": "multiple-choice", **images}
        built = data.mmmu_row(choice, 0)
        assert len(built["images"]) == 2 and built["image"] == built["images"][0]
        assert built["answers"] == ["B", "y", "x", "y"]
        assert "(A) x\n(B) y" in built["question"]
        opened = {**choice, "question_type": "open", "options": "[]",
                  "answer": "['Tampa', 'Florida']"}
        assert data.mmmu_row(opened, 0)["answers"] == ["Tampa", "Florida"]
        assert data.mmmu_row({**opened, "answer": "C"}, 0)["answers"] == ["C"]

    def test_charxiv_row_uses_the_reasoning_question(self):
        row = {"image": {"bytes": png(1)}, "figure_path": "images/12.jpg", "reasoning_q": "Q?",
               "reasoning_a": "94", "reasoning_a_type": 3}
        built = data.charxiv_row(row, 0)
        assert built["qid"] == "12" and built["answers"] == ["94"]
        assert built["answer_type"] == "number-in-chart"

    def test_ocrbench_row_guards_the_newline_its_scorer_reads(self):
        row = {"dataset": "HME100k", "question": "Q", "question_type": "formula",
               "answer": ["x\n"], "image": {"bytes": png(1)}}
        assert data.ocrbench_row(row, 3)["qid"] == "test-3"
        with pytest.raises(ValueError, match="newline"):
            data.ocrbench_row({**row, "dataset": "IIIT5K"}, 3)

    def test_docvqa_row_keeps_all_annotator_answers(self):
        row = {"questionId": "49153", "question": "Q?", "question_types": ["table/list", "form"],
               "image": {"bytes": png(1)}, "answers": ["a", "b"]}
        built = data.docvqa_row(row, 0)
        assert built["qid"] == "49153" and built["answers"] == ["a", "b"]
        assert built["answer_type"] == "table/list"

    def test_vstar_row_swaps_the_closing_instruction(self):
        text = ("What is it?\n(A) rubber\n(B) cotton\n"
                "Answer with the option's letter from the given choices directly.")
        row = {"text": text, "category": "direct_attributes", "question_id": "0", "label": "B"}
        built = data.vstar_row(row, png(1))
        assert built["answers"] == ["B", "cotton", "rubber", "cotton"]
        assert built["question"] == ("What is it?\n(A) rubber\n(B) cotton\n"
                                     f"{data.OPTION_INSTRUCTION}")

    def test_hrbench_row_decodes_the_image_without_resizing(self):
        buffer = io.BytesIO()
        PILImage.new("RGB", (64, 48), (9, 9, 9)).save(buffer, "JPEG")
        row = {"index": 5, "answer": "C", "question": "Q?", "A": "a", "B": "b", "C": "c", "D": "d",
               "category": "single", "image": base64.b64encode(buffer.getvalue()).decode()}
        built = data.hrbench_row(row, 0)
        assert built["image"]["bytes"] == buffer.getvalue()
        assert built["answers"] == ["C", "c", "a", "b", "c", "d"]
        assert "A. a\nB. b\nC. c\nD. d" in built["question"]

    def test_pope_row(self):
        row = {"id": "6000", "question": "Is there a dog?", "answer": "yes", "category": "random",
               "image": {"bytes": png(1)}}
        built = data.pope_row(row, 0)
        assert (built["qid"], built["answers"], built["answer_type"]) == ("6000", ["yes"], "random")

    def test_hallusion_row_keeps_text_only_questions_without_an_image(self):
        row = {"category": "VS", "subcategory": "map", "set_id": "1", "figure_id": "0",
               "question_id": "2", "question": "Is it?", "gt_answer": "0", "filename": None}
        placeholder = data.text_only_placeholder()
        alone = data.hallusion_row(row, None, placeholder)
        assert alone["qid"] == "VS-map-1-0-2" and alone["answers"] == ["no"]
        assert alone["images"] == [] and alone["image"]["bytes"] == placeholder
        assert alone["answer_type"] == "VS map text-only"
        shown = data.hallusion_row({**row, "gt_answer": "1"}, png(1), placeholder)
        assert shown["answers"] == ["yes"] and len(shown["images"]) == 1
        assert shown["answer_type"] == "VS map"


class TestQuestionKey:
    @pytest.mark.parametrize("question", [
        "How many?",
        "How many?\nOptions: A: 1, B: 2\nAnswer with the letter of the correct option.",
        "Hint: Please answer.\nQuestion: How many?\nChoices:\n(A) 1\n(B) 2",
        "How many?\n(A) 1\n(B) 2\nAnswer with the option's letter from the given choices directly.",
        "How many?\nA. 1\nB. 2",
        "<image 1> How many?",
    ])
    def test_wrapping_does_not_hide_a_repeated_question(self, question):
        assert question_stem(question).strip() == "How many?"

    def test_the_key_compares_normalized_question_and_reference(self):
        assert question_key("How many DOGS?", ["2"]) == question_key("how many dogs", ["two"])
        assert question_key("How many?", ["2"]) != question_key("How many?", ["3"])
        # A yes/no reference has no key, and a capital letter alone is an open answer.
        assert question_key("Is it red?", ["Yes."]) is None
        assert question_key("Which?", ["C"]) == "which\nc"
        # A choice row compares its option's text; ten annotators compare their common answer.
        assert question_key("Q?", ["B", "dog", "cat", "dog"]) == question_key("Q?", ["dog"])
        assert question_key("Q?", ["cat", "dog", "dog"]) == question_key("Q?", ["dog"])


def test_clean_rows_counts_each_check_and_their_overlap():
    indices, counts = clean_rows([{"a"}, {"b"}, {"c"}, {"d"}], ["q", "seen", "seen", "q"],
                                 {"b", "d"}, {"seen"})
    assert indices == [0]
    assert counts == {"published": 4, "test": 1, "image_overlap": 2, "question_overlap": 2,
                      "both": 1, "no_image": 0, "near_duplicate_lower_bound": 0}


def test_clean_rows_removes_text_only_rows_and_near_copies():
    indices, counts = clean_rows([{"a"}, {"b"}, {"c"}, {"d"}], [None, None, None, "seen"],
                                 set(), {"seen"}, no_image={0}, near_copies={1, 3})
    assert indices == [2]
    # Row 3 is a near copy too, but the question check already removed it.
    assert counts == {"published": 4, "test": 1, "image_overlap": 0, "question_overlap": 1,
                      "both": 0, "no_image": 1, "near_duplicate_lower_bound": 1}


def benchmark_rows(rows):
    return [(f"p{i}", image, question, answers, "") for i, (image, question, answers)
            in enumerate(rows)]


@pytest.fixture
def trained(tmp_path):
    """VQAv2 subsets to train on: images 1-6, each with its own question, plus one VQAv2 test."""
    save_subset(tmp_path, "vqav2_sft", "vqav2", [(f"s{i}", png(i), f"Train {i}?", [f"a{i}"], "")
                                                 for i in range(1, 5)])
    save_subset(tmp_path, "vqav2_rl", "vqav2", [(f"r{i}", png(i), f"Train {i}?", [f"a{i}"], "")
                                                for i in range(5, 7)])
    save_subset(tmp_path, "vqav2_eval", "vqav2", [("e0", png(7), "Held?", ["x"] * 10, "")])
    write_splits(tmp_path, ["vqav2"])
    return tmp_path


def build_benchmark(data_dir, name="pope"):
    others = [read_split(data_dir, "vqav2")] if split_path(data_dir, "vqav2").exists() else []
    split = build_split(name, data_dir, others, seed=0, dev_target=1)
    split_path(data_dir, name).parent.mkdir(parents=True, exist_ok=True)
    split_path(data_dir, name).write_text(json.dumps(split))
    return split


def overlapping_benchmark(data_dir):
    """Row 0 is clean; 1 shares an image with training (shade 5, in the RL pool or carved dev),
    2 repeats a training question and answer on a new image, 3 repeats only the question."""
    save_subset(data_dir, "pope_eval", "pope", benchmark_rows([
        (png(100), "Is it clean?", ["yes"]),
        (png(5), "A new question?", ["yes"]),
        (png(101), "Train 6?", ["a6"]),
        (png(102), "Train 6?", ["other"]),
    ]))
    return build_benchmark(data_dir)


class TestCleanAndPublished:
    def test_published_keeps_every_row_and_test_only_clean_ones(self, trained):
        split = overlapping_benchmark(trained)
        assert split["roles"]["published"][0]["indices"] == [0, 1, 2, 3]
        assert split["roles"]["test"][0]["indices"] == [0, 3]
        assert split["clean_rows"] == {"published": 4, "test": 2, "image_overlap": 1,
                                       "question_overlap": 1, "both": 0, "no_image": 0,
                                       "near_duplicate_lower_bound": 0}

    def test_a_yes_no_repeat_is_not_a_leak_and_other_forms_are(self, tmp_path):
        save_subset(tmp_path, "vqav2_sft", "vqav2", [
            (f"s{i}", png(i), question, answers, "") for i, (question, answers) in enumerate([
                ("Is there a chair in the image?", ["no"]), ("What color is the car?", ["red"]),
                ("How many dogs?", ["2"]), ("Train?", ["a"]), ("Train 2?", ["b"])])])
        save_subset(tmp_path, "vqav2_rl", "vqav2", [("r0", png(10), "Train 3?", ["c"], "")])
        save_subset(tmp_path, "vqav2_eval", "vqav2", [("e0", png(11), "Held?", ["x"] * 10, "")])
        write_splits(tmp_path, ["vqav2"])
        save_subset(tmp_path, "pope_eval", "pope", benchmark_rows([
            (png(100), "Is there a chair in the image?", ["no"]),
            (png(101), "What color is the car?", ["red"]),
            (png(102), "How many dogs?", ["2"]),
        ]))
        vqav2 = read_split(tmp_path, "vqav2")
        sft = vqav2["roles"]["sft"][0]["indices"]
        split = build_benchmark(tmp_path)
        expected = [0] + [row for row, source in ((1, 1), (2, 2)) if source not in sft]
        assert split["roles"]["test"][0]["indices"] == expected

    def test_published_is_exempt_from_the_overlap_check_and_test_is_not(self, trained):
        split = overlapping_benchmark(trained)
        training = read_split(trained, "vqav2")
        assert overlap_report([training, split])["held_out_images"]
        tampered = json.loads(json.dumps(split))
        tampered["roles"]["test"][0]["indices"] = [0, 1]
        with pytest.raises(ValueError, match="held-out images appear in training pools"):
            overlap_report([training, tampered])

    def test_held_out_key_sets_leave_published_out(self, trained):
        split = overlapping_benchmark(trained)
        published = role_keys([split], {"published"})
        assert role_keys([split], {"dev", "test"}) < published

    def test_a_later_training_pool_loses_what_a_benchmarks_clean_rows_hold(self, tmp_path):
        save_subset(tmp_path, "pope_eval", "pope", benchmark_rows([
            (png(100), "What is it?", ["clean"]),
        ]))
        pope = build_benchmark(tmp_path)
        save_subset(tmp_path, "vqav2_sft", "vqav2", [
            ("s0", png(1), "Train 1?", ["a"], ""),
            ("s1", png(100), "Train 2?", ["b"], ""),  # the benchmark's image
            ("s2", png(2), "What is it", ["clean"], ""),  # the benchmark's question and answer
            ("s3", png(3), "What is it?", ["dirty"], ""),  # the question alone stays
            ("s4", png(4), "Train 4?", ["c"], ""),
        ])
        save_subset(tmp_path, "vqav2_rl", "vqav2", [("r0", png(5), "What is it?", ["clean"], "")])
        save_subset(tmp_path, "vqav2_eval", "vqav2", [("e0", png(6), "Held?", ["x"] * 10, "")])
        vqav2 = build_split("vqav2", tmp_path, [pope], seed=0, dev_target=1)
        sft = set(vqav2["roles"]["sft"][0]["indices"])
        dev = set(vqav2["roles"]["dev"][0]["indices"])
        assert not sft & {1, 2} and 1 not in dev
        assert 3 in sft | dev
        rl_pool = next(part for part in vqav2["roles"]["rl"] if part["subset"] == "vqav2_rl")
        assert rl_pool["indices"] == []

    def test_a_datasets_test_role_without_published_keeps_the_image_rule_alone(self, trained):
        """Rebuilding an existing dataset must not change the rows of a recorded table."""
        save_subset(trained, "textvqa_eval", "textvqa", [
            ("t0", png(103), "Train 6?", ["a6"] * 10, ""),
        ])
        textvqa = build_split("textvqa", trained, [read_split(trained, "vqav2")], seed=0,
                              dev_target=1)
        assert textvqa["roles"]["test"][0]["indices"] == [0]
        assert "clean_rows" not in textvqa and "published" not in textvqa["roles"]

    def test_a_test_only_dataset_records_no_dev_selection(self, trained):
        assert "dev_selection" not in overlapping_benchmark(trained)

    def test_a_missing_training_subset_stops_the_question_check(self, trained):
        save_subset(trained, "pope_eval", "pope", benchmark_rows([(png(100), "Q?", ["yes"])]))
        shutil.rmtree(trained / "vqav2_rl")  # the split file records it; the rows are gone
        with pytest.raises(FileNotFoundError):
            build_split("pope", trained, [read_split(trained, "vqav2")], seed=0, dev_target=1)

    def test_validate_refuses_a_test_row_outside_published(self, trained):
        split = overlapping_benchmark(trained)
        split["roles"]["published"][0]["indices"] = [0]
        with pytest.raises(ValueError, match="clean rows must be published rows"):
            validate_split(split)


def multi_image_subset(data_dir, name, dataset, rows):
    """Rows of (qid, images); `image` holds the first or a placeholder, as the loaders write."""
    placeholder = data.text_only_placeholder()
    records = [{"dataset": dataset, "qid": qid,
                "image": {"bytes": images[0] if images else placeholder, "path": None},
                "images": [{"bytes": image, "path": None} for image in images],
                "question": f"Is {qid} right?", "answers": ["yes"], "answer_type": ""}
               for qid, images in rows]
    Dataset.from_list(records, features=data.MULTI_IMAGE_FEATURES).save_to_disk(
        str(data_dir / name))


def test_every_shown_image_counts_toward_overlap(trained):
    multi_image_subset(trained, "mmmu_eval", "mmmu", [
        ("clean", [png(110), png(111)]),
        ("second-image-trained", [png(112), png(5)]),
    ])
    split = build_benchmark(trained, "mmmu")
    assert split["roles"]["test"][0]["indices"] == [0]
    assert split["clean_rows"]["image_overlap"] == 1
    assert set(split["subsets"]["mmmu_eval"]["extra_image_keys"]) == {"0", "1"}
    # The later images count as held out, so a future training pool cannot show them either.
    from datasets import load_from_disk

    rows = load_from_disk(str(trained / "mmmu_eval"))
    assert set(extra_image_keys(rows)["0"]) <= role_keys([split], {"test"})
    assert not set(extra_image_keys(rows)["1"]) & role_keys([split], {"test"})


def test_reading_images_for_the_checks_leaves_the_subset_untouched(tmp_path):
    """A cast to an undecoded image list would write a cache beside another dataset's rows."""
    multi_image_subset(tmp_path, "mmmu_eval", "mmmu", [("two", [png(150), png(151)])])
    before = sorted(path.name for path in (tmp_path / "mmmu_eval").iterdir())
    build_benchmark(tmp_path, "mmmu")
    assert len(subset_gray_keys(tmp_path, "mmmu_eval")[0]) == 2
    assert sorted(path.name for path in (tmp_path / "mmmu_eval").iterdir()) == before


def encoded(image: PILImage.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def pattern(seed: int) -> PILImage.Image:
    """A smooth random gray pattern: enough structure for a perceptual hash to tell apart."""
    cells = np.random.default_rng(seed).integers(0, 255, (8, 8), dtype=np.uint8)
    return PILImage.fromarray(cells).resize((128, 128), PILImage.BILINEAR).convert("RGB")


class TestNearDuplicatePass:
    def pools(self, data_dir, image):
        save_subset(data_dir, "vqav2_sft", "vqav2", [
            (f"s{i}", png(i), f"Train {i}?", [f"a{i}"], "") for i in range(1, 5)])
        save_subset(data_dir, "vqav2_rl", "vqav2", [("r0", encoded(image), "Train 5?", ["a5"], "")])
        save_subset(data_dir, "vqav2_eval", "vqav2", [("e0", png(6), "Held?", ["x"] * 10, "")])
        write_splits(data_dir, ["vqav2"])

    def test_a_resized_copy_leaves_test_and_stays_in_published(self, tmp_path):
        original = pattern(1)
        self.pools(tmp_path, original)
        save_subset(tmp_path, "pope_eval", "pope", benchmark_rows([
            (encoded(original.resize((96, 96), PILImage.BILINEAR)), "Copy?", ["yes"]),
            (encoded(pattern(2)), "Other?", ["yes"]),
        ]))
        split = build_benchmark(tmp_path)
        assert pixel_key(encoded(original)) not in role_keys([split], {"published"})
        assert split["roles"]["test"][0]["indices"] == [1]
        assert split["roles"]["published"][0]["indices"] == [0, 1]
        assert split["clean_rows"]["near_duplicate_lower_bound"] == 1
        assert split["clean_rows"]["image_overlap"] == 0
        assert split["near_duplicate_pass"]["pools"] == ["vqav2_rl", "vqav2_sft"]

    def test_a_later_image_of_a_multi_image_row_counts(self, tmp_path):
        original = pattern(3)
        self.pools(tmp_path, original)
        multi_image_subset(tmp_path, "mmmu_eval", "mmmu", [
            ("copy-second", [encoded(pattern(4)),
                             encoded(original.resize((96, 96), PILImage.BILINEAR))]),
            ("clean", [encoded(pattern(5))]),
        ])
        split = build_benchmark(tmp_path, "mmmu")
        assert split["clean_rows"]["near_duplicate_lower_bound"] == 1
        assert split["roles"]["test"][0]["indices"] == [1]

    def test_a_hash_collision_with_a_different_picture_is_not_a_copy(self):
        ours = bytes([100] * 4096)
        assert near_duplicate_rows([(("abc", ours),)], {"abc": {bytes([100] * 4096)}}) == {0}
        assert near_duplicate_rows([(("abc", ours),)], {"abc": {bytes([120] * 4096)}}) == set()
        assert near_duplicate_rows([(("abc", ours),)], {"xyz": {ours}}) == set()

    def test_a_near_uniform_image_has_no_key(self):
        assert near_key(png(5)) is None
        assert near_key(encoded(pattern(6))) is not None

    def test_ct_pools_are_skipped_without_reading_them(self, tmp_path):
        ct = {"dataset": "deeptumor", "subsets": {},
              "roles": {"sft": [{"subset": "never_materialized", "indices": [0]}]}}
        assert near_duplicate_index([ct], tmp_path) == {}
        assert near_duplicate_pools([ct]) == []


class TestColorModeCopies:
    """A diagram and its grayscale copy: different RGB pixels, the same gray pixels."""

    COLORED = PILImage.new("RGB", (4, 4), (10, 200, 30))

    def copies(self):
        return encoded(self.COLORED), encoded(self.COLORED.convert("L"))

    def test_only_the_gray_key_matches_them(self):
        colored, gray = self.copies()
        assert pixel_key(colored) != pixel_key(gray)
        assert gray_key(colored) == gray_key(gray)
        assert gray_key(colored) != pixel_key(colored)

    def test_a_later_training_pool_loses_the_gray_copy_of_a_clean_row(self, tmp_path):
        colored, gray = self.copies()
        save_subset(tmp_path, "pope_eval", "pope", benchmark_rows([(colored, "Clean?", ["yes"])]))
        pope = build_benchmark(tmp_path)
        save_subset(tmp_path, "vqav2_sft", "vqav2", [
            (f"s{i}", png(i), f"Train {i}?", ["a"], "") for i in range(1, 5)
        ] + [("copy", gray, "Train copy?", ["a"], "")])
        save_subset(tmp_path, "vqav2_rl", "vqav2", [("r0", gray, "Train copy 2?", ["a"], "")])
        save_subset(tmp_path, "vqav2_eval", "vqav2", [("e0", png(6), "Held?", ["x"] * 10, "")])
        vqav2 = build_split("vqav2", tmp_path, [pope], seed=0, dev_target=1)
        assert 4 not in vqav2["roles"]["sft"][0]["indices"]
        rl_pool = next(part for part in vqav2["roles"]["rl"] if part["subset"] == "vqav2_rl")
        assert rl_pool["indices"] == []



class Echo:
    """Answers every question with a fixed reply and keeps what it was shown."""

    def __init__(self, reply="Yes"):
        self.reply = reply
        self.shown = []

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        self.shown += conversations
        return [[{"text": self.reply, "output_tokens": 3, "finish_reason": "stop"}] * samples
                for _ in conversations]


@pytest.fixture
def model(tmp_path):
    weights = tmp_path / "model"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"w")
    return ModelRef(str(weights), label="arm")


class TestReachingTheTables:
    def test_a_benchmark_record_reaches_load_cells_and_an_ood_cohort_does_not(self, trained, model,
                                                                              tmp_path):
        overlapping_benchmark(trained)
        save_subset(trained, "vizwiz_eval", "vizwiz", [("v0", png(120), "Q?", ["yes"] * 10, "")])
        write_splits(trained, ["vizwiz"])
        records = tmp_path / "records"
        run_eval(model, Protocol("pope"), Echo(), data_dir=trained, records_dir=records)
        run_eval(model, Protocol("pope", role="published"), Echo(), data_dir=trained,
                 records_dir=records)
        run_eval(model, Protocol("vizwiz"), Echo(), data_dir=trained, records_dir=records)

        cells = load_cells(records, trained)
        assert {(cell.dataset, cell.role) for cell in cells} == {("pope", "test"),
                                                                 ("pope", "published")}
        for cell, record in cells.items():
            assert is_current(record, trained)
            expected = 2 if cell.role == "test" else 4
            assert record.metrics["n"] == expected
            assert record.metrics["yes_no"]["yes_ratio"] == 1.0

    def test_test_and_published_are_different_protocols_over_one_subset(self, trained):
        overlapping_benchmark(trained)
        clean = select_rows(Protocol("pope"), trained)
        everything = select_rows(Protocol("pope", role="published"), trained)
        assert len(clean[0]) == 2 and len(everything[0]) == 4
        assert clean[1]["subsets"] == everything[1]["subsets"]
        assert clean[1]["selection_sha256"] != everything[1]["selection_sha256"]

    def test_a_project_only_benchmark_records_no_official_score(self, trained, model, tmp_path):
        save_subset(trained, "vstar_eval", "vstar", [
            ("v0", png(130), "Q?\n(A) x\n(B) y", data.choice_answers("B", ["x", "y"]), "kind"),
        ])
        build_benchmark(trained, "vstar")
        record = run_eval(model, Protocol("vstar"), Echo("B"), data_dir=trained,
                          records_dir=tmp_path / "records")
        assert record.metrics["official"] is None
        assert record.metrics["project"]["overall"] == 1.0
        assert record.metrics["normalized"]["overall"] == 1.0

    def test_text_only_rows_show_no_image_and_multi_image_rows_show_all(self, tmp_path, model):
        multi_image_subset(tmp_path, "hallusionbench_eval", "hallusionbench", [
            ("with-image", [png(140)]), ("text-only", []),
        ])
        build_benchmark(tmp_path, "hallusionbench")
        generator = Echo()
        run_eval(model, Protocol("hallusionbench", role="published"), generator, data_dir=tmp_path,
                 records_dir=tmp_path / "records")
        shown = [sum(part["type"] != "text" for part in conversation[0]["content"])
                 for conversation in generator.shown]
        assert shown == [1, 0]
        clean = select_rows(Protocol("hallusionbench"), tmp_path)[0]
        assert list(clean["qid"]) == ["with-image"]
        assert read_split(tmp_path, "hallusionbench")["clean_rows"]["no_image"] == 1

        multi_image_subset(tmp_path, "mmmu_eval", "mmmu", [("two", [png(141), png(142)])])
        build_benchmark(tmp_path, "mmmu")
        generator = Echo()
        run_eval(model, Protocol("mmmu"), generator, data_dir=tmp_path,
                 records_dir=tmp_path / "records")
        assert sum(part["type"] != "text" for part in generator.shown[0][0]["content"]) == 2


def test_to_example_reads_an_empty_image_list_as_no_image():
    row = {"dataset": "hallusionbench", "qid": "q", "question": "Q?", "answers": ["yes"],
           "answer_type": "", "image": {"bytes": b"placeholder", "path": None}, "images": []}
    assert to_example(row).images == []
    single = {key: value for key, value in row.items() if key != "images"}
    assert to_example(single).images == [b"placeholder"]
