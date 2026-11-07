"""The checker audit: the draw, the rates, the budget guard and the report, with a fake client."""

import json
from types import SimpleNamespace

import pytest
from fixtures import every_dataset, png, save_subset, write_splits

from scripts.audit_checker import (
    INPUT_TOKEN_BOUND,
    allocate,
    audit_source,
    draw_side,
    rate,
    read_responses,
    sampled_record,
    shown_references,
    side_of,
    write_report,
)
from scripts.audit_judge import batch_cost, cost_bound, wilson
from vqar.eval import ModelRef, run_eval
from vqar.eval.screening import battery
from vqar.judge import Judge, JudgeBudgetExceeded


class Generator:
    """Per question, right on odd samples, wrong on even ones, with no answer on the last."""

    def generate(self, conversations, *, max_tokens, temperature, samples, seed):
        replies = ["<answer>9</answer>", "<answer>7</answer>"] * samples
        return [[{"text": "no tags here" if i == samples - 1 else replies[i], "output_tokens": 2,
                  "finish_reason": "stop"} for i in range(samples)] for _ in conversations]


class FakeBatches:
    """An auditor that says yes to one answer and no to every other."""

    def __init__(self, accept="7", fail=0):
        self.created, self.accept, self.fail = [], accept, fail

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id="batch", processing_status="ended")

    def retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def results(self, batch_id):
        for position, request in enumerate(self.created[-1]):
            if position < self.fail:
                yield SimpleNamespace(custom_id=request["custom_id"],
                                      result=SimpleNamespace(type="errored"))
                continue
            text = request["params"]["messages"][0]["content"][1]["text"]
            verdict = "yes" if f"Candidate answer: {self.accept}" in text else "no"
            message = SimpleNamespace(
                stop_reason="end_turn", content=[SimpleNamespace(type="text", text=verdict)],
                usage=SimpleNamespace(input_tokens=1_000, output_tokens=100))
            yield SimpleNamespace(custom_id=request["custom_id"],
                                  result=SimpleNamespace(type="succeeded", message=message))


def judge_with(tmp_path, batches, budget=10.0) -> Judge:
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    return Judge(cache_path=tmp_path / "judge.db", budget_usd=budget, client=client)


@pytest.fixture
def sampled(tmp_path):
    """A GEOQA-like pool whose questions are all worth 7, as a number or as a fraction (text)."""
    data = tmp_path / "data"
    every_dataset(data)
    rows = [(f"q{i}", png(200 + i), f"What is the length {i}?",
             ["7"] if i % 4 else ["\\frac{14}{2}"], "") for i in range(12)]
    save_subset(data, "geoqa_train", "geoqa", rows)
    write_splits(data, ["geoqa"])
    weights = tmp_path / "model"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"w")
    model = ModelRef(str(weights), label="base")
    run_eval(model, battery("geoqa").sampled, Generator(), data_dir=data,
             records_dir=tmp_path / "records")
    return sampled_record("geoqa", model, data, tmp_path / "records"), data


class TestSides:
    @pytest.mark.parametrize("answer, reward, side", [
        ("7", 1.0, "right"), ("9", 0.0, "wrong"), ("9", 0.3, "partial"), (None, 0.0, "no answer"),
        ("", 0.0, "no answer"),
    ])
    def test_side_of(self, answer, reward, side):
        assert side_of(answer, reward) == side

    def test_responses_read_as_the_reward_reads_them(self):
        lines = [{"qid": "q", "sample": i, "question": "Q?", "raw": raw, "answers": ["7", "7°"]}
                 for i, raw in enumerate(["<answer>7</answer>", "x <answer>9</answer>", "7"])]
        rows = read_responses(lines, "geoqa", "reasoning")
        assert [row["side"] for row in rows] == ["right", "wrong", "no answer"]
        assert [row["reward"] for row in rows] == [1.0, 0.0, 0.0]
        assert rows[0]["prediction"] == "7" and rows[0]["form"] == "number"

    def test_the_auditor_sees_one_option_not_the_whole_list(self):
        assert shown_references(["B", "92", "46", "92", "110"]) == ["B", "92"]
        assert shown_references(["P. Carter", "p. carter", "Carter"]) == ["P. Carter", "p. carter",
                                                                          "Carter"]


class TestDraw:
    def test_allocation_is_proportional_and_reaches_small_forms(self):
        assert allocate({"number": 300, "text": 150, "choice": 50, "yes/no": 2}, 50) == {
            "number": 29, "text": 15, "choice": 5, "yes/no": 1}

    def test_allocation_takes_all_when_there_are_few(self):
        assert allocate({"number": 3, "text": 2}, 50) == {"number": 3, "text": 2}

    def test_allocation_with_fewer_draws_than_forms(self):
        assert sum(allocate({"a": 5, "b": 4, "c": 3}, 2).values()) == 2

    def responses(self, side="right"):
        forms = ["number"] * 80 + ["text"] * 15 + ["choice"] * 5
        return [{"dataset": "geoqa", "qid": f"q{i}", "sample": sample, "side": side, "form": form}
                for i, form in enumerate(forms) for sample in range(3)]

    def test_one_response_per_question_stratified_and_seeded(self):
        drawn = draw_side(self.responses(), "right", 50, seed=0)
        assert len(drawn) == 50 and len({row["qid"] for row in drawn}) == 50
        forms = [row["form"] for row in drawn]
        assert [forms.count(form) for form in ("number", "text", "choice")] == [40, 7, 3]
        assert drawn == draw_side(self.responses(), "right", 50, seed=0)
        assert drawn != draw_side(self.responses(), "right", 50, seed=1)

    def test_only_the_named_side_is_drawn(self):
        assert draw_side(self.responses("wrong"), "right", 50, seed=0) == []


class TestRates:
    def test_wilson_matches_the_published_formula(self):
        low, high = wilson(5, 50)
        assert low == pytest.approx(0.0435, abs=1e-3) and high == pytest.approx(0.2136, abs=1e-3)
        assert wilson(0, 0) is None
        assert wilson(0, 50)[0] == pytest.approx(0.0, abs=1e-12) and wilson(0, 50)[1] > 0.05

    def test_a_rate_counts_only_audited_rows_and_overrules(self):
        rows = [{"side": "wrong", "auditor": True}, {"side": "wrong", "auditor": False},
                {"side": "wrong", "auditor": False}, {"side": "wrong"}]
        result = rate(rows)
        assert (result["drawn"], result["audited"], result["overruled"]) == (4, 3, 1)
        assert result["rate"] == pytest.approx(1 / 3)
        assert result["interval"] == wilson(1, 3)

    def test_nothing_audited_has_no_rate(self):
        assert rate([{"side": "right"}]) == {"drawn": 1, "audited": 0, "overruled": 0, "rate": None,
                                             "interval": None}

    def test_the_cost_bound_uses_the_auditors_batch_price(self):
        assert batch_cost(1_000, 100) == pytest.approx((4_000 + 2_000) / 1e6 / 2)
        assert cost_bound(100) == pytest.approx(100 * (1_700 * 4 + 300 * 20) / 1e6 / 2)


class TestAudit:
    def test_rates_examples_cost_and_the_record_read(self, sampled, tmp_path):
        record, data = sampled
        batches = FakeBatches(accept="7")  # the auditor accepts 7 and rejects 9
        row = audit_source(record, judge_with(tmp_path, batches), per_side=6, data_dir=data)
        assert row["record"] == record.path.name and row["protocol_id"] == record.protocol_id
        # One image of twelve is carved for dev. Of eight samples, 3 are right, 4 wrong, 1 untagged.
        assert row["responses"] == {"right": 33, "wrong": 44, "partial": 0, "no answer": 11}
        assert row["false_reject"]["audited"] == 6 and row["false_accept"]["audited"] == 6
        assert row["false_reject"]["overruled"] == 0
        assert row["false_accept"]["overruled"] == 0
        assert row["cost_usd"] == pytest.approx(12 * batch_cost(1_000, 100))
        assert row["disagreements"] == {"right": [], "wrong": []}
        assert {item["side"] for item in row["items"]} == {"right", "wrong"}

    def test_an_auditor_that_disagrees_gives_rates_and_examples(self, sampled, tmp_path):
        record, data = sampled
        row = audit_source(record, judge_with(tmp_path, FakeBatches(accept="9")), per_side=6,
                           data_dir=data)
        assert row["false_reject"]["rate"] == 1.0 and row["false_accept"]["rate"] == 1.0
        assert row["false_reject"]["interval"] == wilson(6, 6)
        example = row["disagreements"]["wrong"][0]
        assert example["prediction"] == "9" and example["auditor"] is True
        assert set(row["by_form"]["wrong"]) == {"number", "text"}

    def test_the_image_question_references_and_answer_reach_the_auditor(self, sampled, tmp_path):
        record, data = sampled
        batches = FakeBatches()
        audit_source(record, judge_with(tmp_path, batches), per_side=2, data_dir=data)
        request = batches.created[0][0]["params"]
        image, text = request["messages"][0]["content"]
        assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
        assert "What is the length" in text["text"] and "Reference answers" in text["text"]
        assert "Candidate answer: " in text["text"]

    def test_the_bound_prints_before_any_call_and_a_dry_run_calls_nothing(self, sampled, tmp_path,
                                                                           capsys):
        record, data = sampled
        batches = FakeBatches()
        assert audit_source(record, judge_with(tmp_path, batches), per_side=6, data_dir=data,
                            dry_run=True) is None
        assert batches.created == []
        printed = capsys.readouterr().out
        bound = cost_bound(12, INPUT_TOKEN_BOUND)
        assert f"cost at most about ${bound:.2f}" in printed and "cap $10.00" in printed

    def test_a_batch_past_the_cap_is_refused(self, sampled, tmp_path):
        record, data = sampled
        batches = FakeBatches()
        with pytest.raises(JudgeBudgetExceeded):
            audit_source(record, judge_with(tmp_path, batches, budget=1e-9), per_side=6,
                         data_dir=data)
        assert batches.created == []

    def test_spend_joins_the_ledger_and_a_repeat_costs_nothing(self, sampled, tmp_path):
        record, data = sampled
        batches = FakeBatches()
        judge = judge_with(tmp_path, batches)
        first = audit_source(record, judge, per_side=6, data_dir=data)
        assert judge.spend() == pytest.approx(first["cost_usd"])
        second = audit_source(record, judge, per_side=6, data_dir=data)
        assert len(batches.created) == 1
        assert second["false_accept"] == first["false_accept"]
        assert second["cost_usd"] == pytest.approx(first["cost_usd"])

    def test_a_failed_request_stays_unaudited(self, sampled, tmp_path):
        record, data = sampled
        row = audit_source(record, judge_with(tmp_path, FakeBatches(fail=2)), per_side=6,
                           data_dir=data)
        drawn = row["false_accept"]["drawn"] + row["false_reject"]["drawn"]
        audited = row["false_accept"]["audited"] + row["false_reject"]["audited"]
        assert drawn - audited == 2

    def test_a_missing_record_says_how_to_make_it(self, tmp_path, capsys):
        data = tmp_path / "data"
        every_dataset(data)
        save_subset(data, "geoqa_train", "geoqa", [(f"q{i}", png(200 + i), "Q?", ["7"], "")
                                                   for i in range(4)])
        write_splits(data, ["geoqa"])
        weights = tmp_path / "model"
        weights.mkdir()
        (weights / "model.safetensors").write_bytes(b"w")
        model = ModelRef(str(weights), label="base")
        assert sampled_record("geoqa", model, data, tmp_path / "records") is None
        assert "scripts.screen_datasets" in capsys.readouterr().out


class TestReport:
    def test_one_source_updates_only_its_row(self, tmp_path):
        output = tmp_path / "checker-audit.json"
        write_report({"geoqa": {"cost_usd": 1.0}}, 50, 0, output)
        write_report({"mmk12": {"cost_usd": 2.0}}, 50, 0, output)
        write_report({"geoqa": {"cost_usd": 3.0}}, 50, 0, output)
        report = json.loads(output.read_text())
        assert report["sources"] == {"geoqa": {"cost_usd": 3.0}, "mmk12": {"cost_usd": 2.0}}
        assert report["per_side"] == 50 and "false_reject" in report and "prompt" in report
