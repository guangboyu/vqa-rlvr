"""Dataset screening: greedy protocols on a training role, the battery, its measures,
gates, and ranking."""

import pytest
from fixtures import every_dataset

from vqar.eval import ModelRef, Protocol, protocol_identity, run_eval


class Fixed:
    """Every sample of every question gets the same reply."""

    def __init__(self, text):
        self.text = text

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        return [[{"text": self.text, "output_tokens": 2, "finish_reason": "stop"}] * samples
                for _ in conversations]


@pytest.fixture
def setup(tmp_path):
    every_dataset(tmp_path / "data")
    weights = tmp_path / "model"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"w")
    return tmp_path / "data", ModelRef(str(weights), label="base"), tmp_path / "records"


def test_existing_protocol_identities_are_unchanged():
    # Hashes computed before the screening flag existed; records keep these ids.
    data = {"subsets": {"x": "f"}, "selection_sha256": "s"}
    expected = {
        Protocol("vqav2"): "7a2d2a8404ca119a",
        Protocol("gqa", prompt="reasoning", role="dev", tier="dev"): "fe6ce1d17fb8cf15",
        Protocol("clevr", tier="first-128", image_mode="black"): "0e57d42a0b3dbc6d",
        Protocol("gqa", prompt="reasoning", role="rl", tier="dev", max_tokens=640,
                 temperature=1.0, samples=8): "16c2577d3db92428",
    }
    for protocol, prefix in expected.items():
        assert Protocol.id_of(protocol_identity(protocol, data)).startswith(prefix)


def test_the_screening_flag_is_not_part_of_the_identity():
    data = {"subsets": {"x": "f"}, "selection_sha256": "s"}
    identity = protocol_identity(Protocol("gqa", role="rl", screening=True), data)
    assert "screening" not in identity
    assert Protocol("gqa", role="rl", screening=True).slug() == "gqa-rl-full-short"


def test_only_screening_reads_a_training_role_greedily():
    Protocol("gqa", role="rl", screening=True)
    Protocol("vqav2", role="sft", screening=True)
    with pytest.raises(ValueError, match="learnability protocols"):
        Protocol("vqav2", role="sft")
    with pytest.raises(ValueError, match="clevr has no sft role"):
        Protocol("clevr", role="sft", screening=True)


def test_a_greedy_screening_record_stays_out_of_the_learnability_table(setup):
    from vqar.eval.report import learnability_table, load_cells

    data, model, records = setup
    greedy = Protocol("gqa", role="rl", screening=True)
    record = run_eval(model, greedy, Fixed("2"), data_dir=data, records_dir=records)
    assert record.metrics["normalized"]["overall"] == 1.0
    assert "learnability" not in record.metrics
    assert learnability_table(load_cells(records)) == \
        "No learnability pre-check has been recorded yet.\n"
