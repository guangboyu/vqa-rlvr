"""The experiment matrix's missing-cell check."""

import pytest
from fixtures import every_dataset

from vqar.eval import ModelRef, Protocol, run_eval
from vqar.eval.matrix import Arm, load_matrix, missing_cells


class Echo:
    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        return [[{"text": "2", "output_tokens": 1, "finish_reason": "stop"}]
                for _ in conversations]


@pytest.fixture
def matrix(tmp_path):
    every_dataset(tmp_path / "data")
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.safetensors").write_bytes(name.encode())
    (tmp_path / "experiments.toml").write_text('''
[[arms]]
label = "a"
name = "Arm A"
model = "a"

[[arms]]
label = "b"
name = "Arm B"
model = "b"

[[protocols]]
datasets = ["gqa", "clevr"]
prompts = ["short"]
''')
    return tmp_path


def test_matrix_expands_datasets_prompts_and_image_modes(tmp_path):
    (tmp_path / "experiments.toml").write_text('''
[[arms]]
label = "a"
name = "Arm A"
model = "a"

[[protocols]]
datasets = ["gqa", "clevr"]
prompts = ["short", "reasoning"]
image_modes = ["original", "black"]
tier = "dev"
''')
    arms, protocols = load_matrix(tmp_path / "experiments.toml")
    assert [arm.label for arm in arms] == ["a"]
    assert protocols == [
        Protocol(dataset, prompt=prompt, tier="dev", image_mode=mode)
        for dataset in ("gqa", "clevr") for prompt in ("short", "reasoning")
        for mode in ("original", "black")]


def test_missing_cells_lists_only_what_has_no_record(matrix):
    arms, protocols = load_matrix(matrix / "experiments.toml")
    records = matrix / "records"
    run_eval(ModelRef(str(matrix / "a"), label="a"), Protocol("gqa"), Echo(),
             data_dir=matrix / "data", records_dir=records)
    missing = missing_cells(arms, protocols, records, matrix / "data", matrix)
    assert [(arm.label, p.dataset) for arm, p in missing] == [
        ("a", "clevr"), ("b", "gqa"), ("b", "clevr")]


def test_a_record_of_another_model_under_an_arm_label_is_an_error(matrix):
    arms, protocols = load_matrix(matrix / "experiments.toml")
    records = matrix / "records"
    run_eval(ModelRef(str(matrix / "b"), label="a"), Protocol("gqa"), Echo(),
             data_dir=matrix / "data", records_dir=records)
    with pytest.raises(ValueError, match="records of another model"):
        missing_cells(arms, protocols, records, matrix / "data", matrix)


def test_an_arm_whose_merge_was_removed_keeps_its_records_unverified(matrix):
    gone = Arm("gone", "Removed merge", "checkpoints/removed")
    assert not gone.available(matrix)
    assert Arm("hub", "Hub", "Qwen/Qwen3-VL-2B-Instruct", revision="abc").available(matrix)
    records = matrix / "records"
    run_eval(ModelRef(str(matrix / "b"), label="gone"), Protocol("gqa"), Echo(),
             data_dir=matrix / "data", records_dir=records)
    missing = missing_cells([gone], [Protocol("gqa"), Protocol("clevr")], records,
                            matrix / "data", matrix)
    assert [(arm.label, p.dataset) for arm, p in missing] == [("gone", "clevr")]


def test_comparisons_pair_each_candidate_with_its_reference(tmp_path):
    from vqar.eval.matrix import load_comparisons

    (tmp_path / "experiments.toml").write_text(
        '[[comparisons]]\nreference = "a"\ncandidates = ["b", "c"]\n')
    assert load_comparisons(tmp_path / "experiments.toml") == [("a", "b"), ("a", "c")]
