from types import SimpleNamespace

import pytest

from vqar.judge import Judge, JudgeBudgetExceeded


def fake_message(text="yes", input_tokens=100, output_tokens=2):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeClient:
    def __init__(self, text="yes"):
        self.calls = 0
        self._text = text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        return fake_message(self._text)


def make_judge(tmp_path, client, budget=10.0):
    return Judge(cache_path=tmp_path / "judge.db", budget_usd=budget, client=client)


def test_yes_verdict(tmp_path):
    judge = make_judge(tmp_path, FakeClient("yes"))
    assert judge.judge("What color?", ["gray"], "grey") is True


def test_no_verdict(tmp_path):
    judge = make_judge(tmp_path, FakeClient("no"))
    assert judge.judge("What color?", ["gray"], "blue") is False


def test_cache_prevents_second_api_call(tmp_path):
    client = FakeClient("yes")
    judge = make_judge(tmp_path, client)
    judge.judge("Q", ["a"], "b")
    judge.judge("Q", ["a"], "b")
    assert client.calls == 1


def test_cache_key_uses_normalized_prediction(tmp_path):
    client = FakeClient("yes")
    judge = make_judge(tmp_path, client)
    judge.judge("Q", ["gray"], "Grey.")
    judge.judge("Q", ["gray"], "grey")  # normalizes to the same key
    assert client.calls == 1


def test_cache_persists_across_instances(tmp_path):
    client = FakeClient("yes")
    make_judge(tmp_path, client).judge("Q", ["a"], "b")
    fresh = make_judge(tmp_path, FakeClient("no"))
    assert fresh.judge("Q", ["a"], "b") is True  # served from disk cache


def test_budget_cap_raises(tmp_path):
    client = FakeClient("yes")
    judge = make_judge(tmp_path, client, budget=0.0)
    with pytest.raises(JudgeBudgetExceeded):
        judge.judge("Q", ["a"], "b")


def test_spend_accumulates(tmp_path):
    judge = make_judge(tmp_path, FakeClient("yes"))
    judge.judge("Q1", ["a"], "b")
    judge.judge("Q2", ["a"], "b")
    # 2 calls x (100 in x $1 + 2 out x $5) / 1M
    assert judge.spend() == pytest.approx(2 * (100 * 1.0 + 2 * 5.0) / 1e6)


def test_judge_many_uses_cache_and_batch(tmp_path):
    sync_client = FakeClient("yes")
    judge = make_judge(tmp_path, sync_client)
    judge.judge("Q1", ["a"], "p1")  # pre-cache one item

    class FakeBatches:
        def __init__(self):
            self.created = None

        def create(self, requests):
            self.created = requests
            return SimpleNamespace(id="batch_1", processing_status="ended")

        def retrieve(self, batch_id):
            return SimpleNamespace(id=batch_id, processing_status="ended")

        def results(self, batch_id):
            for req in self.created:
                yield SimpleNamespace(
                    custom_id=req["custom_id"],
                    result=SimpleNamespace(type="succeeded", message=fake_message("no")),
                )

    batch_client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches()))
    judge._client = batch_client
    verdicts = judge.judge_many([("Q1", ["a"], "p1"), ("Q2", ["a"], "p2")])
    assert verdicts == [True, False]  # first from cache, second from batch
    assert len(batch_client.messages.batches.created) == 1  # only the uncached item sent
