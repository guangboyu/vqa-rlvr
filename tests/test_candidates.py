"""Training candidates: loader rows from tiny source rows, the 0/1 reward scorers on references
copied from each source, and how overlap takes rows out of a new pool. No raw data is read."""

import io
import json

import numpy as np
import pytest
from fixtures import png, save_subset
from PIL import Image as PILImage

from vqar import data
from vqar.answers import reference_form
from vqar.benchmark_metrics import (
    anls_match,
    math_answer_accuracy,
    math_form,
    math_value,
    plain_number,
)
from vqar.eval.screening import DECLARED, failed_gates
from vqar.registry import DATASETS, score
from vqar.rewards import correctness_reward
from vqar.splits import (
    build_split,
    gray_key,
    overlap_report,
    pixel_key,
    read_split,
    role_keys,
    split_path,
)

CANDIDATES = ("geoqa", "mmk12", "virl39k", "docvqat", "infovqa")


def encoded(image: PILImage.Image, format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format)
    return buffer.getvalue()


class TestRegistry:
    @pytest.mark.parametrize("name", CANDIDATES)
    def test_each_candidate_trains_and_holds_out_dev(self, name):
        spec = DATASETS[name]
        assert set(spec.subsets) == {"sft", "rl", "dev"}
        assert spec.subsets["sft"] == spec.subsets["rl"]
        assert spec.ladder == "in-domain" and not spec.require_split and spec.max_images == 1

    def test_categories_and_the_benchmark_stays_a_separate_dataset(self):
        assert {DATASETS[name].category for name in ("geoqa", "mmk12", "virl39k")} == {"Reasoning"}
        assert {DATASETS[name].category for name in ("docvqat", "infovqa")} == {"Document-chart"}
        assert set(DATASETS["docvqa"].subsets) == {"test", "published"}
        assert DATASETS["docvqat"].subsets["dev"] == "docvqat_train"

    def test_infovqa_alone_has_a_released_held_out_split(self):
        assert DATASETS["infovqa"].subsets["dev"] == "infovqa_val"
        assert all(DATASETS[name].subsets["dev"] == DATASETS[name].subsets["sft"]
                   for name in CANDIDATES if name != "infovqa")

    def test_a_licence_nobody_confirmed_fails_the_gate_until_someone_does(self):
        assert [name for name in CANDIDATES if not DECLARED[name].license_usable] == [
            "geoqa", "docvqat", "infovqa"]
        assert "license" in failed_gates(5_000, DECLARED["geoqa"], {})
        assert "license" not in failed_gates(5_000, DECLARED["mmk12"], {})
        assert all("unverified" in DECLARED[name].license
                   for name in ("geoqa", "docvqat", "infovqa"))


class TestImages:
    def test_transparent_strokes_stay_visible_on_white(self):
        layer = PILImage.new("RGBA", (8, 8), (0, 0, 0, 0))
        layer.putpixel((2, 2), (0, 0, 0, 255))
        flat = PILImage.open(io.BytesIO(data.on_white(encoded(layer))))
        assert flat.mode == "RGB"
        assert flat.getpixel((0, 0)) == (255, 255, 255) and flat.getpixel((2, 2)) == (0, 0, 0)
        assert encoded(layer.convert("RGB")) != encoded(flat)  # dropping alpha would be all black

    def test_an_opaque_image_keeps_the_pixels_dropping_alpha_gives(self):
        layer = PILImage.new("RGBA", (4, 4), (9, 80, 200, 255))
        flat = PILImage.open(io.BytesIO(data.on_white(encoded(layer))))
        assert flat.tobytes() == layer.convert("RGB").tobytes()

    @pytest.mark.parametrize("mode", ["RGB", "L"])
    def test_rgb_and_gray_bytes_stay_as_released(self, mode):
        payload = encoded(PILImage.new(mode, (4, 4)))
        assert data.on_white(payload) == payload

    def test_gray_with_a_transparent_colour_is_flattened(self):
        image = PILImage.new("L", (4, 4), 0)
        buffer = io.BytesIO()
        image.save(buffer, "PNG", transparency=0)
        flat = PILImage.open(io.BytesIO(data.on_white(buffer.getvalue())))
        assert flat.getpixel((0, 0)) == (255, 255, 255)


class TestGeoqaRows:
    @staticmethod
    def row(solution: str) -> dict:
        return {"image": {"bytes": png(3), "path": "1000.png"},
                "problem": "In the given diagram, what is the measure of angle 2?",
                "solution": solution}

    @pytest.mark.parametrize("solution, answers, form", [
        ("<answer> 145° </answer>", ["145", "145°"], "number"),
        ("<answer> 40度 </answer>", ["40", "40度"], "number"),
        ("<answer> 10cm </answer>", ["10", "10cm"], "number"),
        ("<answer> 8 </answer>", ["8"], "number"),
        ("<answer> \\frac{3}{4} </answer>", ["\\frac{3}{4}"], "text"),
        ("<answer> 2√{3} </answer>", ["2√{3}"], "text"),
        ("<answer> notenoughinformation </answer>", ["notenoughinformation"], "text"),
    ])
    def test_references_hold_the_bare_number_then_the_released_text(self, solution, answers, form):
        row = data.geoqa_row(self.row(solution), 7)
        assert row["answers"] == answers and reference_form(row["answers"]) == form
        assert row["qid"] == "train-7" and row["dataset"] == "geoqa" and row["answer_type"] == ""

    def test_an_empty_answer_is_dropped(self):
        assert data.geoqa_row(self.row("<answer>  </answer>"), 0) is None
        assert data.geoqa_drop(self.row("<answer>  </answer>")) == "empty answer"


class TestMmk12Rows:
    @staticmethod
    def row(answer: str) -> dict:
        return {"id": "cc7f", "question": "This geometric shape is ___.", "answer": answer,
                "subject": "math", "image": {"bytes": png(3), "path": "x.png"}}

    @pytest.mark.parametrize("answer, answers", [
        ("$$60^{\\circ}$$", ["60", "$$60^{\\circ}$$"]),
        ("$$\\quantity{12}{cm}$$", ["12", "$$\\quantity{12}{cm}$$"]),
        ("3", ["3"]),
        ("$$x>2$$", ["$$x>2$$"]),
    ])
    def test_references(self, answer, answers):
        row = data.mmk12_row(self.row(answer), 0)
        assert row["answers"] == answers and row["qid"] == "cc7f" and row["answer_type"] == "math"

    @pytest.mark.parametrize("answer", [
        "$$y=\\left\\lbrace\\begin{align}& 10t,0\\leqslant t\\cr&2\\end{align}\\right .$$",
        "x" * 81,
    ])
    def test_answers_no_rule_can_match_are_dropped(self, answer):
        assert data.mmk12_row(self.row(answer), 0) is None
        assert data.mmk12_drop(self.row(answer)) == "answer longer than a final answer"


class TestVirlRows:
    @staticmethod
    def row(question: str, answer: str, images=("images/x-0.png",), source="Processed") -> dict:
        return {"qid": "Processed-1", "question": question, "answer": answer, "category": "Spatial",
                "source": source, "image": list(images)}

    @pytest.mark.parametrize("released, cleaned", [
        ("<image>\nIs the dotted line a line?", "Is the dotted line a line?"),
        ("<image>/nBased on the given images, answer the following question.\n\nFind x.",
         "Find x."),
        ("The below problem is with the following images:\n<image>\nHow many?", "How many?"),
        ("How long?\nThe above problem is with the following images:\n<image>", "How long?"),
        ("Find $\\\\sqrt{2}$.\\nA. 1\\nB. 2", "Find $\\sqrt{2}$.\nA. 1\nB. 2"),
        ("Is $a\\neq b$?", "Is $a\\neq b$?"),
    ])
    def test_wrappers_leave_the_question(self, released, cleaned):
        assert data.virl_question(released) == cleaned

    @pytest.mark.parametrize("question, options", [
        ("Find the measure.\nChoices:\n(A) 46\n(B) 92\n(C) 110\n(D) 156",
         ["46", "92", "110", "156"]),
        ("Lines m and n, then ( )\nChoices:\nA. 73°\nB. 83°", ["73°", "83°"]),
        ("How long?\nChoices:\nA. 17 seconds B. 18 seconds C. 19 seconds",
         ["17 seconds", "18 seconds", "19 seconds"]),
        ("Which? Choices are (A) bucket, (B) crab, (C) boat.", ["bucket", "crab", "boat"]),
        ("如图?\n选项：\n(A) 2\n(B) 4", ["2", "4"]),
        ("At point A. Then which?\nA) one\nB) two\nAnswer with the option's letter.",
         ["one", "two"]),
        ("Which label is it?\nA\nC\nB\nPlease answer based on the options.", []),
        ("Pick one.\nChoices:\nA. B. C. D.", []),  # options drawn in the image
        ("The side is ___.", []),
    ])
    def test_written_options(self, question, options):
        assert data.written_options(question) == options

    def test_a_letter_takes_the_choice_form_when_its_options_read(self):
        answers = data.virl_answers("Find.\n(A) 46\n(B) 92", "B")
        assert answers == ["B", "92", "46", "92"] and reference_form(answers) == "choice"

    def test_a_letter_without_options_stays_an_open_answer(self):
        answers = data.virl_answers("Which label?\nA\nC\nB\nPlease answer.", "C")
        assert answers == ["C"] and reference_form(answers) == "text"

    def test_a_blank_option_list_cannot_pay_any_letter(self):
        answers = data.virl_answers("Pick one.\nChoices:\nA. B. C. D.", "C")
        assert math_answer_accuracy("C", answers) == 1
        assert math_answer_accuracy("A", answers) == 0

    def test_row_schema(self):
        row = data.virl_row(self.row("<image>\nFind x.", "\\boxed{60^\\circ}"), png(4))
        assert row["qid"] == "Processed-1" and row["question"] == "Find x."
        assert row["answer_type"] == "Spatial"
        assert row["answers"] == ["60", "60^\\circ"] and row["image"]["bytes"] == png(4)

    @pytest.mark.parametrize("row, reason", [
        (row("q", "\\boxed{1}", images=("a", "b")), "several images (training shows one)"),
        (row("q", "\\boxed{1}", source="DeepScaleR"),
         "blank placeholder image (DeepScaleR, text-only math)"),
        (row("Then $\\angle P = \\boxed{120}^\\circ$.", "\\boxed{120}"),
         "question states its own answer"),
        (row("q", "\\boxed{" + "x" * 81 + "}"), "answer longer than a final answer"),
        (row("q", "\\boxed{}"), "empty answer"),
        (row("q", "\\boxed{1}"), None),
    ])
    def test_drops(self, row, reason):
        assert data.virl_drop(row) == reason


class TestSpanRows:
    def test_docvqat_keeps_every_answer_and_its_first_type(self):
        row = data.docvqat_row({"questionId": "338", "question": "Contact person?",
                                "question_types": ["handwritten", "form"],
                                "image": {"bytes": png(5), "path": "x.png"},
                                "answers": ["P. Carter", "p. carter"]}, 0)
        assert row["qid"] == "338" and row["dataset"] == "docvqat"
        assert row["answers"] == ["P. Carter", "p. carter"] and row["answer_type"] == "handwritten"

    def test_infovqa_train_has_no_answer_type_and_validation_does(self):
        base = {"questionId": "65718", "question": "Which fonts?", "answers": ["serif fonts"],
                "image": {"bytes": png(5), "path": "x.jpeg"}}
        assert data.infovqa_row({**base, "answer_type": None}, 0)["answer_type"] == ""
        typed = data.infovqa_row({**base, "answer_type": ["multi-span"]}, 0)
        assert typed["answer_type"] == "multi-span"

    @pytest.mark.parametrize("answers", [None, [], ["  "]])
    def test_a_row_with_no_answer_is_dropped(self, answers):
        row = {"questionId": "1", "question": "q", "question_types": [], "answers": answers,
               "image": {"bytes": png(5), "path": "x"}}
        assert data.docvqat_row(row, 0) is None
        assert data.span_drop(row) == "empty answer"

    def test_a_blank_answer_among_others_leaves(self):
        row = {"questionId": "1", "question": "q", "question_types": None, "answers": ["a", " "],
               "image": {"bytes": png(5), "path": "x"}}
        assert data.docvqat_row(row, 0)["answers"] == ["a"]


class TestMathReward:
    """References are copied from the sources; predictions are what a model might write."""

    @pytest.mark.parametrize("prediction, reference, expected", [
        # GEOQA
        ("145", "145°", 1), ("145 degrees", "145°", 1), ("145^{\\circ}", "145°", 1),
        ("35°", "145°", 0),
        ("40", "40度", 1), ("40°", "40度", 1), ("4.5 m", "4.5米", 1), ("100π", "100π平方米", 1),
        ("3/4", "\\frac{3}{4}", 1), ("0.75", "\\frac{3}{4}", 1), ("4/3", "\\frac{3}{4}", 0),
        ("2\\sqrt{3}", "2√{3}", 1), ("2√3", "2√{3}", 1), ("3.46", "2√{3}", 0),
        ("24/7", "\\frac{24}{7}米", 1), ("not enough information", "notenoughinformation", 1),
        ("8.8", "8.80", 1), ("10 m", "10cm", 1),  # units are stripped, not compared
        # MMK12
        ("60", "$$60^{ \\circ }$$", 1), ("20", "$20{}^\\circ$", 1),
        ("22.5", "${{22.5}^{\\circ }}$", 1),
        ("0.75", "$$\\number{0.75}$$", 1), ("12 cm", "$$\\quantity{12}{cm}$$", 1),
        ("-1/2", "$$-\\dfrac{1}{2}$$", 1),
        ("$\\frac{5\\sqrt{3}}{3}$", "$$\\dfrac{5\\sqrt{3}}{3}$$", 1),
        ("36-12√3", "$$36-12\\sqrt 3$$", 1), ("x>2", "$$x>2$$", 1), ("x<2", "$$x>2$$", 0),
        ("(3/2, 3/2)", "$$\\left (\\dfrac{3}{2},\\dfrac{3}{2} \\right )$$", 1),
        ("15", "∠AOD=15°", 1), ("35", "7×5=35", 1), ("x = 5", "5", 1),
        ("2.5", "2\\frac{1}{2}", 1),  # a mixed number, not a product
        ("8", "8 (m³)", 1), ("1234", "1,234", 1), ("60%", "60", 1), ("35", "35º", 1),
        ("7", "$$07$$", 1),
        ("90", "$$\\angle ABC=90^{\\circ}$$", 1), ("20", "20cm^2^", 1),
        # An equation to state: its right side is no answer.
        ("13", "(x-2)^2+(y+3)^2=13", 0), ("1", "$$\\dfrac{x^{2}}{8}+\\dfrac{y^{2}}{2}=1$$", 0),
        ("90", "a+b=90", 0), ("(x-2)^2+(y+3)^2=13", "(x-2)^2+(y+3)^2=13", 1),
        ("c", "a-c", 0), ("c", "a > c", 0),  # an article to VQA normalization, a variable here
        # ViRL39K
        ("12π", "\\( 12\\pi \\) inches", 1), ("12π inches", "\\( 12\\pi \\) inches", 1),
        ("32", "$32 \\mathrm{~m}$", 1), ("9/16", "\\dfrac{9}{16}", 1), ("Yes", "Yes", 1),
        ("yes.", "Yes", 1),
        ("Durban", "Durban", 1), ("the Durban", "Durban", 1), ("Cape Town", "Durban", 0),
        ("4n-3", "4n - 3", 1), ("4n", "4n - 3", 0), ("two", "2", 1),
        # Answers that must not pay
        ("", "5", 0), ("   ", "5", 0), ("?", "a", 0), ("5", "x>=5", 0), ("x>=5", "5", 0),
        ("5", "", 0),
    ])
    def test_references_from_the_sources(self, prediction, reference, expected):
        assert math_answer_accuracy(prediction, [reference]) == expected

    def test_any_stored_reference_counts(self):
        assert math_answer_accuracy("145°", ["145", "145°"]) == 1
        assert math_answer_accuracy("145", ["145", "145°"]) == 1

    @pytest.mark.parametrize("prediction, expected", [
        ("B", 1), ("(B)", 1), ("B. 92", 1), ("92", 1), ("92°", 1), ("A", 0), ("46", 0), ("", 0),
    ])
    def test_options_read_by_letter_then_text(self, prediction, expected):
        assert math_answer_accuracy(prediction, ["B", "92", "46", "92", "110"]) == expected

    @pytest.mark.parametrize("prediction, expected", [
        ("A cyclin is unable to release", 1), ("a driver is present", 0), ("B", 0), ("C", 1),
        ("B cells are present", 0),
    ])
    def test_an_options_text_is_not_read_as_a_leading_letter(self, prediction, expected):
        answers = ["C", "A cyclin is unable to release", "a driver is present",
                   "B cells are present", "A cyclin is unable to release"]
        assert math_answer_accuracy(prediction, answers) == expected

    @pytest.mark.parametrize("answers", [
        ["B", "C", "A and B", "C", "D", "B"],  # options that are letters, from ViRL39K
        ["B", "-a-2b-c", "-a-c", "-a-2b-c", "a-2b+c"],
    ])
    def test_a_bare_letter_is_that_letter(self, answers):
        assert math_answer_accuracy("B", answers) == 1
        assert math_answer_accuracy("A", answers) == 0

    @pytest.mark.parametrize("hostile", [
        "__import__('os').system('true')", "9**9**9**9", "(" * 500, "1/0", "sqrt(-1)", "1" * 400,
        "pi.real", "lambda: 1",
    ])
    def test_the_evaluator_reads_only_arithmetic(self, hostile):
        assert math_value(math_form(hostile)) is None

    @pytest.mark.parametrize("text, number", [
        ("145°", "145"), ("$$60^{\\circ}$$", "60"), ("10cm", "10"), ("40度", "40"),
        ("4.5米", "4.5"),
        ("1,234", "1234"), ("8", "8"), ("\\frac{3}{4}", None), ("2√{3}", None), ("x>2", None),
        ("∠AOD=15°", "15"), ("x=-3", "-3"), ("7×5=35", "35"), ("a+b=90", None),
        ("x(14-2x)=20", None),
    ])
    def test_plain_number(self, text, number):
        assert plain_number(text) == number


class TestAnlsReward:
    @pytest.mark.parametrize("prediction, answers, expected", [
        # DocVQA
        ("P. Carter", ["P. Carter", "p. carter"], 1), ("p carter", ["P. Carter", "p. carter"], 1),
        ("J. Smith", ["P. Carter", "p. carter"], 0),
        ("Vanderbilt University", ["Vanderbilt university", "vanderbilt university"], 1),
        ("$29.25", ["29.25", "$29.25"], 1), ("8:30", ["8.30", "8:30"], 1),
        ("1/8/93", ["1/8/93"], 1),
        # InfographicVQA
        ("1.5 billion people", ["1.5 billion", "1.5 billion people"], 1),
        ("serif font", ["serif fonts"], 1), ("serif", ["serif fonts"], 0),  # 6 of 11 letters off
        ("sans serif", ["serif fonts"], 0), ("", ["serif fonts"], 0),
        ("   ", ["serif fonts"], 0),
    ])
    def test_real_references(self, prediction, answers, expected):
        assert anls_match(prediction, answers) == expected

    def test_the_threshold_is_inclusive_and_the_reward_is_whole(self):
        assert anls_match("ab", ["ac"]) == 1  # distance 1 of 2: exactly 0.5, the authors keep it
        assert anls_match("abc", ["ade"]) == 0  # 2 of 3

    def test_a_number_needs_the_same_number(self):
        assert anls_match("2020", ["2021"]) == 0
        assert anls_match("12", ["13"]) == 0
        assert anls_match("2021", ["2021"]) == 1
        assert anls_match("1,250", ["1250"]) == 1
        assert anls_match("2021", ["2021", "the year"]) == 1
        assert anls_match("the yeat", ["2021", "the year"]) == 1


class TestRewardPath:
    def reward(self, dataset, response, answers, prompt="reasoning"):
        return correctness_reward(None, [response], [answers], [dataset], [prompt])[0]

    @pytest.mark.parametrize("dataset", ["geoqa", "mmk12", "virl39k"])
    def test_math_sources_pay_the_tagged_answer(self, dataset):
        assert self.reward(dataset, "x <answer>145 degrees</answer>", ["145", "145°"]) == 1.0
        assert self.reward(dataset, "x <answer>35</answer>", ["145", "145°"]) == 0.0
        assert self.reward(dataset, "145", ["145", "145°"]) == 0.0  # no tag, no answer

    @pytest.mark.parametrize("dataset", ["docvqat", "infovqa"])
    def test_document_sources_pay_one_or_nothing(self, dataset):
        assert self.reward(dataset, "<answer>Carter</answer>", ["P. Carter"]) == 1.0
        assert self.reward(dataset, "<answer>Smith</answer>", ["P. Carter"]) == 0.0

    def test_the_official_column_keeps_anls_credit(self):
        assert score("docvqat", "Carter", ["P. Carter"], "official") == pytest.approx(1 - 3 / 9)
        assert score("docvqat", "Carter", ["P. Carter"], "normalized") == 1.0
        assert DATASETS["geoqa"].official is None
        assert score("geoqa", "145", ["145°"], "project") == 1.0


def write(data_dir, split):
    path = split_path(data_dir, split["dataset"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))


def benchmark_rows(rows):
    return [(f"p{i}", image, question, answers, "") for i, (image, question, answers)
            in enumerate(rows)]


def role_indices(split, role):
    return [i for part in split["roles"][role] for i in part["indices"]]


def diagram_copies() -> tuple[bytes, bytes]:
    """A colored diagram and its grayscale copy: different RGB pixels, the same gray pixels."""
    gradient = PILImage.linear_gradient("L").resize((16, 16))
    colored = PILImage.merge("RGB", (gradient, gradient.rotate(90), gradient.rotate(180)))
    return encoded(colored), encoded(colored.convert("L"))


@pytest.fixture
def benchmarks(tmp_path):
    """MathVista's clean rows: a colored diagram and a question; ChartQA's held-out images."""
    save_subset(tmp_path, "mathvista_eval", "mathvista", benchmark_rows([
        (diagram_copies()[0], "What is the area of the circle?", ["25"]),
        (png(120), "Find angle B.", ["60"]),
    ]))
    write(tmp_path, build_split("mathvista", tmp_path, [], seed=0, dev_target=1))
    save_subset(tmp_path, "chartqa_train", "chartqa", [("t", png(130), "Chart train?", ["1"], "")])
    save_subset(tmp_path, "chartqa_val", "chartqa", [("v", png(131), "Chart val?", ["1"], "")])
    save_subset(tmp_path, "chartqa_test", "chartqa", [("e", png(132), "Chart test?", ["1"], "")])
    write(tmp_path, build_split("chartqa", tmp_path, [read_split(tmp_path, "mathvista")], seed=0))
    return tmp_path


def build_geoqa(data_dir, rows, dev_target=1):
    save_subset(data_dir, "geoqa_train", "geoqa", rows)
    others = [read_split(data_dir, name) for name in ("mathvista", "chartqa")]
    split = build_split("geoqa", data_dir, others, seed=0, dev_target=dev_target)
    write(data_dir, split)
    return split


class TestOverlap:
    def pool(self):
        return [
            ("own0", png(1), "Own 0?", ["1"], ""),
            ("own1", png(2), "Own 1?", ["2"], ""),
            ("own2", png(3), "Own 2?", ["3"], ""),
            ("exact", png(120), "Another angle?", ["70"], ""),  # MathVista's image
            ("gray", diagram_copies()[1], "A gray copy?", ["5"], ""),  # MathVista's, in gray
            ("repeat", png(4), "What is the area of the circle?", ["25", "25°"], ""),
            ("val", png(131), "Copy of a dev image?", ["9"], ""),  # ChartQA dev
            ("test", png(132), "Copy of a test image?", ["9"], ""),  # ChartQA test
            ("train", png(130), "A trained ChartQA image?", ["9"], ""),  # allowed: training only
            ("words", png(5), "What is the area of the circle?", ["7"], ""),  # another answer
        ]

    def test_rows_leave_both_training_roles_and_the_benchmarks_stay_put(self, benchmarks):
        before = {name: split_path(benchmarks, name).read_bytes()
                  for name in ("mathvista", "chartqa")}
        split = build_geoqa(benchmarks, self.pool())
        qids = [row[0] for row in self.pool()]
        training = {qids[i] for i in role_indices(split, "sft")}
        carved = {qids[i] for i in role_indices(split, "dev")}
        assert not {"exact", "gray", "repeat", "val", "test"} & (training | carved)
        assert "train" in training and "words" in training | carved
        assert role_indices(split, "rl") == role_indices(split, "sft")
        for name, raw in before.items():
            assert split_path(benchmarks, name).read_bytes() == raw

    def test_dev_is_whole_images_kept_out_of_training(self, benchmarks):
        rows = self.pool() + [("own0-again", png(1), "Own 0 again?", ["4"], "")]
        split = build_geoqa(benchmarks, rows, dev_target=2)
        dev, sft = set(role_indices(split, "dev")), set(role_indices(split, "sft"))
        assert dev and not dev & sft
        keys = split["subsets"]["geoqa_train"]["image_keys"]
        assert not {keys[i] for i in dev} & {keys[i] for i in sft}
        assert split["dev_selection"]["method"] == "whole images from the SFT pool"
        report = overlap_report([read_split(benchmarks, "mathvista"),
                                 read_split(benchmarks, "chartqa"), split])
        assert report["held_out_images"] >= 3

    def test_the_split_records_what_each_dataset_took(self, benchmarks):
        split = build_geoqa(benchmarks, self.pool())
        losses = split["pool_losses"]["geoqa_train"]
        assert losses["source_rows"] == 10 and losses["rows_lost"] == 5
        assert losses["by_dataset"]["mathvista"] == {"image": 1, "gray": 2, "near": 1,
                                                     "question": 1}
        assert losses["by_dataset"]["chartqa"] == {"image": 2, "gray": 0, "near": 0, "question": 0}
        kept = len(role_indices(split, "sft")) + len(role_indices(split, "dev"))
        assert kept == losses["source_rows"] - losses["rows_lost"]

    def test_a_jpeg_copy_of_a_clean_image_leaves_though_no_pixel_key_matches(self, benchmarks):
        cells = np.random.default_rng(1).integers(0, 255, (8, 8), dtype=np.uint8)
        source = PILImage.fromarray(cells).resize((128, 128), PILImage.BILINEAR).convert("RGB")
        buffer = io.BytesIO()
        source.save(buffer, "JPEG", quality=95)
        original, copy = encoded(source), buffer.getvalue()
        assert pixel_key(original) != pixel_key(copy) and gray_key(original) != gray_key(copy)
        save_subset(benchmarks, "mmstar_eval", "mmstar", benchmark_rows([
            (original, "Option text?", data.choice_answers("A", ["red", "blue"]))]))
        write(benchmarks, build_split("mmstar", benchmarks, [], seed=0, dev_target=1))
        save_subset(benchmarks, "geoqa_train", "geoqa", [
            ("copy", copy, "A paraphrase?", ["1"], ""), ("own0", png(1), "Own?", ["1"], ""),
            ("own1", png(2), "Own 1?", ["1"], ""), ("own2", png(3), "Own 2?", ["1"], "")])
        split = build_split("geoqa", benchmarks, [read_split(benchmarks, "mmstar")], seed=0,
                            dev_target=1)
        assert 0 not in role_indices(split, "sft") and 0 not in role_indices(split, "dev")
        assert split["pool_losses"]["geoqa_train"]["by_dataset"]["mmstar"]["near"] == 1

    def test_another_candidates_dev_images_leave_a_later_pool(self, benchmarks):
        geoqa = build_geoqa(benchmarks, self.pool(), dev_target=1)
        held = role_keys([geoqa], {"dev"})
        keys = geoqa["subsets"]["geoqa_train"]["image_keys"]
        shared = next(i for i, key in enumerate(keys) if key in held)
        image = self.pool()[shared][1]
        save_subset(benchmarks, "mmk12_train", "mmk12", [
            ("a", image, "Same image?", ["1"], ""), ("b", png(40), "Other?", ["1"], ""),
            ("c", png(41), "Other 2?", ["1"], "")])
        others = [read_split(benchmarks, name) for name in ("mathvista", "chartqa", "geoqa")]
        split = build_split("mmk12", benchmarks, others, seed=0, dev_target=1)
        assert 0 not in role_indices(split, "sft") and 0 not in role_indices(split, "dev")
        assert split["pool_losses"]["mmk12_train"]["by_dataset"]["geoqa"]["image"] == 1


def test_the_summary_counts_numbers_the_reward_pays_one_digit_off():
    from scripts.candidate_sources import numeric_leniency, one_digit_off

    assert one_digit_off("2021") == "2022" and one_digit_off("$29.25") == "$29.26"
    assert one_digit_off("19") == "10"  # only the last digit changes
    rows = [["2021"], ["$29.25", "29.25"], ["9"], ["Washington", "washington"]]
    assert numeric_leniency(rows) == {"numeric_rows": 3, "pay_one_digit_off": 0}
