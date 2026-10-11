"""Dataset screening: greedy protocols on a training role, the battery, its measures,
gates, and ranking."""

import base64
import io
import json
import sys

import pytest
from fixtures import every_dataset
from PIL import Image as PILImage

from vqar.eval import ModelRef, Protocol, RunRecord, protocol_identity, run_eval, screening
from vqar.eval.screening import battery


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


class Screener:
    """Right under the short prompt, wrong under the greedy reasoning prompt, wrong on a
    black image, and right on every other sample."""

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        outputs = []
        for messages in conversations:
            text = next(part["text"] for part in messages[0]["content"]
                        if part["type"] == "text")
            if image_shade(messages) == 0:
                replies = ["0"]
            elif temperature > 0:
                replies = ["<answer>2</answer>", "<answer>3</answer>"]
            elif "single word" in text:
                replies = ["2"]
            else:
                replies = ["<answer>3</answer>"]
            outputs.append([{"text": replies[i % len(replies)], "output_tokens": 2,
                             "finish_reason": "stop"} for i in range(samples)])
        return outputs


def image_shade(messages) -> int:
    url = next(part["image_url"]["url"] for part in messages[0]["content"]
               if part["type"] == "image_url")
    with PILImage.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        return image.convert("L").getpixel((0, 0))


def run_battery(cells, model, data, records):
    for protocol in cells.protocols().values():
        run_eval(model, protocol, Screener(), data_dir=data, records_dir=records)


def test_the_battery_reads_the_training_pool_with_the_learnability_protocol():
    from vqar.rl import ROLLOUT_TEMPERATURE

    cells = battery("gqa")
    assert [p.slug() for p in cells.protocols().values()] == [
        "gqa-rl-dev-short", "gqa-rl-dev-short-black", "gqa-rl-dev-short-swapped",
        "gqa-rl-dev-reasoning", "gqa-rl-dev-direct", "gqa-rl-dev-stepwise",
        "gqa-rl-dev-reasoning-t1k8", "gqa-rl-dev-short-t1k8"]
    assert not cells.validation_only
    assert cells.sampled.temperature == ROLLOUT_TEMPERATURE
    # The same cell as `scripts.learnability --preset grpo_2b_main --tier dev`.
    data = {"subsets": {"x": "f"}, "selection_sha256": "s"}
    assert Protocol.id_of(protocol_identity(cells.sampled, data)).startswith("16c2577d3db92428")


def test_a_dataset_without_a_training_pool_is_screened_only_for_validation():
    assert battery("textvqa") is None
    cells = battery("textvqa", "first-300", validate_on_test=True)
    assert cells.role == "test" and cells.validation_only
    assert cells.short.slug() == "textvqa-test-first-300-short"
    assert battery("clevr", validate_on_test=True).role == "rl"


def test_sampling_measures_from_per_question_counts():
    result = screening.sampling_measures({"a": 0, "b": 4, "c": 8, "d": 1}, 8)
    assert result["pass_at_1"] == pytest.approx((0 + 0.5 + 1 + 0.125) / 4)
    assert result["pass_at_k"] == 0.75
    assert result["rl_headroom"]["delta"] == pytest.approx(100 * (0.75 - 0.40625))
    assert result["rl_headroom"]["low"] <= result["rl_headroom"]["delta"]
    assert result["learnable_share"] == 0.5


def test_parse_rate_follows_the_reward_answer_rule():
    lines = [{"raw": "so <answer>2</answer>"}, {"raw": "2"}, {"raw": "<answer> </answer>"},
             {"raw": "<answer>a</answer> then <answer>b</answer>"}]
    assert screening.parse_rate(lines, "reasoning") == 0.5
    assert screening.parse_rate(lines, "short") == 1.0


def test_gates_and_their_boundaries():
    failed_gates, good = screening.failed_gates, screening.DECLARED["gqa"]
    assert failed_gates(2_000, good, {"parse_rate": 0.95, "short_accuracy": 0.899}) == []
    assert failed_gates(1_999, good, {}) == ["training questions"]
    assert failed_gates(5_000, good, {"parse_rate": 0.949, "short_accuracy": 0.90}) == [
        "not saturated"]
    assert failed_gates(5_000, None, {}) == ["split separable", "license"]
    closed = screening.Declaration("patient", False, "research only", False)
    assert failed_gates(5_000, closed, {}) == ["split separable", "license"]


def test_the_image_gate_holds_for_any_way_of_training():
    failed_gates, good = screening.failed_gates, screening.DECLARED["gqa"]
    passing = {"image_dependence": {"low": 10.0}, "wrong_image_dependence": {"low": 25.0}}
    assert failed_gates(5_000, good, passing) == []
    assert failed_gates(5_000, good, {**passing,
                                      "wrong_image_dependence": {"low": 9.9}}) == [
        "image needed"]


def test_each_rollout_prompt_has_its_own_conditions():
    reasoning = {"parse_rate": 0.95, "matched_payoff": {"high": 0.0}, "learnable_share": 0.25}
    routes = screening.route_failures
    assert routes({}) == {"reasoning": None, "short": None}
    assert routes(reasoning)["reasoning"] == []
    assert routes({**reasoning, "matched_payoff": {"high": -0.1}, "parse_rate": 0.9})[
        "reasoning"] == ["parse rate", "reasoning not worse"]
    assert routes({**reasoning, "learnable_share": 0.249})["reasoning"] == ["learnable"]
    assert routes({"short_learnable_share": 0.25})["short"] == []
    assert routes({"short_learnable_share": 0.249})["short"] == ["learnable"]


def test_a_candidate_passes_on_either_route_and_fails_only_when_both_do():
    status = screening.status_of
    assert status([], {"reasoning": ["reasoning not worse"], "short": []}, True) == "passed"
    assert status([], {"reasoning": [], "short": None}, True) == "passed"
    assert status([], {"reasoning": ["learnable"], "short": ["learnable"]}, True) == "rejected"
    assert status([], {"reasoning": ["parse rate"], "short": None}, False) == "incomplete"
    assert status(["image needed"], {"reasoning": [], "short": []}, True) == "rejected"
    assert status([], {"reasoning": [], "short": []}, False) == "incomplete"


def row(model, dataset, status, payoff=0.0, headroom=0.0):
    return {"model": model, "dataset": dataset, "status": status,
            "matched_payoff": {"delta": payoff}, "rl_headroom": {"delta": headroom}}


def test_ranking_orders_survivors_by_payoff_then_headroom_per_model():
    rows = screening.rank([
        row("m", "rejected-a", "rejected"), row("m", "low", "passed", -5.0, 30.0),
        row("m", "tie-more-headroom", "passed", 2.0, 20.0),
        row("m", "tie-less-headroom", "passed", 2.0, 10.0),
        row("m", "unfinished", "incomplete"), row("n", "other", "passed", -9.0)])
    assert [(r["model"], r["dataset"], r["rank"]) for r in rows] == [
        ("m", "tie-more-headroom", 1), ("m", "tie-less-headroom", 2), ("m", "low", 3),
        ("m", "unfinished", None), ("m", "rejected-a", None), ("n", "other", 1)]


def test_the_battery_end_to_end_from_records(setup, monkeypatch):
    from scripts.screen_datasets import batteries, collect

    data, model, records = setup
    planned = batteries(["gqa", "textvqa", "clevr"], "dev", validate_on_test=True)
    for dataset in ("gqa", "textvqa"):
        run_battery(planned[dataset], model, data, records)
    monkeypatch.setattr(screening, "MIN_POOL_QUESTIONS", 3)
    rows = {r["dataset"]: r for r in collect("base", model, planned, data, records)}

    gqa = rows["gqa"]
    assert gqa["role"] == "rl" and gqa["questions"] == gqa["pool_questions"]
    assert gqa["reasoning_payoff"]["delta"] == -100.0
    assert gqa["image_dependence"]["delta"] == 100.0
    assert (gqa["pass_at_1"], gqa["pass_at_k"], gqa["learnable_share"]) == (0.5, 1.0, 1.0)
    assert gqa["rl_headroom"]["delta"] == 50.0
    assert gqa["parse_rate"] == 1.0
    assert gqa["short_accuracy"] == 1.0
    # A swapped image is another question's picture, which the screener still answers.
    assert gqa["wrong_image_dependence"]["delta"] == 0.0
    assert gqa["matched_payoff"]["delta"] == 0.0
    assert gqa["failed_gates"] == ["not saturated", "image needed"]
    assert gqa["status"] == "rejected"

    textvqa = rows["textvqa"]
    assert textvqa["validation_only"] and textvqa["pool_questions"] == 0
    assert textvqa["reasoning_payoff"]["delta"] < 0
    assert "training questions" in textvqa["failed_gates"]

    clevr = rows["clevr"]
    assert clevr["status"] == "incomplete"
    assert clevr["missing"] == ["short", "black", "swapped", "reasoning", "direct", "stepwise",
                                "sampled", "sampled_short"]
    assert "reasoning_payoff" not in clevr

    table = screening.markdown(screening.rank(list(rows.values())))
    assert "test (validation only)" in table
    assert "rejected (failed: not saturated, image needed)" in table
    assert ("incomplete: missing short, black, swapped, reasoning, direct, stepwise, sampled, "
            "sampled_short") in table


def test_older_learnability_records_are_recounted_from_their_predictions(setup):
    data, model, records = setup
    record = run_eval(model, battery("gqa").sampled, Screener(), data_dir=data,
                      records_dir=records)
    stored = screening.correct_counts(record)
    assert set(stored.values()) == {4}
    learnability = {k: v for k, v in record.metrics["learnability"].items()
                    if k != "correct_samples"}
    older = RunRecord(record.model, record.protocol,
                      {**record.metrics, "learnability": learnability}, record.provenance,
                      path=record.path)
    assert screening.correct_counts(older) == stored


def test_dry_run_lists_the_battery_and_from_records_writes_the_table(setup, tmp_path,
                                                                     monkeypatch, capsys):
    from scripts import screen_datasets

    data, model, records = setup
    matrix = tmp_path / "experiments.toml"
    matrix.write_text(f'[[arms]]\nlabel = "base"\nname = "Base"\nmodel = "{model.path}"\n')
    common = ["screen_datasets", "--arms", "base", "--matrix", str(matrix), "--data-dir",
              str(data), "--records-dir", str(records), "--datasets", "gqa", "textvqa"]
    monkeypatch.setattr(sys, "argv", [*common, "--dry-run"])
    screen_datasets.main()
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("base gqa short: gqa-rl-dev-short: missing, ")
    assert [line.split(": ")[1] for line in printed[:8]] == [
        "gqa-rl-dev-short", "gqa-rl-dev-short-black", "gqa-rl-dev-short-swapped",
        "gqa-rl-dev-reasoning", "gqa-rl-dev-direct", "gqa-rl-dev-stepwise",
        "gqa-rl-dev-reasoning-t1k8", "gqa-rl-dev-short-t1k8"]
    assert printed[8].startswith("base textvqa: no training pool")

    run_battery(battery("gqa"), model, data, records)
    out = tmp_path / "screening"
    monkeypatch.setattr(sys, "argv", [*common, "--from-records", "--out", str(out)])
    screen_datasets.main()
    written = json.loads((out / "screening.json").read_text())
    assert written["gates"]["min_parse_rate"] == 0.95
    assert [(r["dataset"], r["status"]) for r in written["rows"]] == [
        ("gqa", "rejected"), ("textvqa", "rejected")]
    assert written["rows"][1]["failed_gates"] == ["training questions"]
    assert "| base | — | GQA | rl |" in (out / "screening.md").read_text()


def test_writing_one_dataset_updates_its_row_and_keeps_the_others(tmp_path):
    from scripts.screen_datasets import merged

    def row(dataset, status):
        return {"model": "base", "dataset": dataset, "status": status}

    (tmp_path / "screening.json").write_text(json.dumps(
        {"rows": [row("gqa", "rejected"), row("slake", "incomplete")]}))
    rows = merged([row("slake", "passed")], tmp_path)
    assert sorted((r["dataset"], r["status"]) for r in rows) == [
        ("gqa", "rejected"), ("slake", "passed")]
    assert merged([row("gqa", "passed")], tmp_path / "missing") == [row("gqa", "passed")]

