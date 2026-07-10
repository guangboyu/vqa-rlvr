"""LLM judge for answer equivalence, used where rule-based matching fails.

Claude Haiku judges whether a predicted answer is semantically equivalent to the
gold answer(s) ("grey" vs "gray", "4 people" vs "4"). Called only on rule-based
mismatches, so volume stays small. Every verdict is cached in sqlite keyed by
(model, question, golds, normalized prediction) — RL rollouts repeat wrong
answers constantly, so the hit rate is high. Cumulative spend is persisted and a
hard budget cap raises JudgeBudgetExceeded rather than overspending.

Two call paths:
- judge(): synchronous, cache-first — used inside the GRPO reward function.
- judge_many(): Message Batches API (50% price) — used for bulk eval scoring.
"""

import hashlib
import os
import sqlite3
import time
from pathlib import Path

from vqar.metrics import normalize_answer

MODEL = "claude-haiku-4-5-20251001"  # pinned for reproducibility
PRICE_IN, PRICE_OUT = 1.0, 5.0  # USD per MTok
DEFAULT_BUDGET_USD = float(os.environ.get("JUDGE_BUDGET_USD", "10.0"))

PROMPT = """Question: {question}
Gold answer(s): {golds}
Candidate answer: {prediction}

Does the candidate answer mean the same thing as one of the gold answers, in the \
context of the question? Ignore differences in casing, phrasing, or specificity that \
do not change the meaning. Reply with exactly one word: yes or no."""


class JudgeBudgetExceeded(RuntimeError):
    pass


class Judge:
    def __init__(
        self,
        cache_path: str | Path = ".judge_cache/judge.db",
        budget_usd: float = DEFAULT_BUDGET_USD,
        client=None,
    ):
        self.budget_usd = budget_usd
        self._client = client  # injectable for tests; lazily constructed otherwise
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS judgments ("
            "key TEXT PRIMARY KEY, verdict INTEGER, cost_usd REAL, created REAL)"
        )
        self._db.commit()

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def spend(self) -> float:
        """Total USD spent across all sessions (persisted in the cache DB)."""
        return self._db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM judgments").fetchone()[0]

    def _key(self, question: str, golds: list[str], prediction: str) -> str:
        raw = "|".join([MODEL, question, *sorted(golds), normalize_answer(prediction)])
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cached(self, key: str) -> bool | None:
        row = self._db.execute("SELECT verdict FROM judgments WHERE key = ?", (key,)).fetchone()
        return None if row is None else bool(row[0])

    def _store(self, key: str, verdict: bool, cost_usd: float) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO judgments VALUES (?, ?, ?, ?)",
            (key, int(verdict), cost_usd, time.time()),
        )
        self._db.commit()

    def _check_budget(self) -> None:
        if self.spend() >= self.budget_usd:
            raise JudgeBudgetExceeded(
                f"judge spend ${self.spend():.2f} reached the ${self.budget_usd:.2f} cap"
            )

    def _params(self, question: str, golds: list[str], prediction: str) -> dict:
        return {
            "model": MODEL,
            "max_tokens": 8,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        question=question, golds="; ".join(golds), prediction=prediction
                    ),
                }
            ],
        }

    @staticmethod
    def _cost(usage, batch: bool = False) -> float:
        cost = (usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT) / 1e6
        return cost * 0.5 if batch else cost

    @staticmethod
    def _verdict(message) -> bool:
        text = next((b.text for b in message.content if b.type == "text"), "")
        return text.strip().lower().startswith("yes")

    def judge(self, question: str, golds: list[str], prediction: str) -> bool:
        """Synchronous single judgment (RL reward path). Cache-first, budget-capped."""
        key = self._key(question, golds, prediction)
        cached = self._cached(key)
        if cached is not None:
            return cached
        self._check_budget()
        message = self.client.messages.create(**self._params(question, golds, prediction))
        verdict = self._verdict(message)
        self._store(key, verdict, self._cost(message.usage))
        return verdict

    def judge_many(
        self, items: list[tuple[str, list[str], str]], poll_seconds: float = 30.0
    ) -> list[bool]:
        """Bulk judgments via the Message Batches API (50% price). Eval path.

        items: (question, gold_answers, prediction) triples. Order-preserving.
        """
        keys = [self._key(q, g, p) for q, g, p in items]
        verdicts: dict[str, bool] = {k: v for k in keys if (v := self._cached(k)) is not None}
        todo = {}  # key -> item, deduplicated
        for key, item in zip(keys, items, strict=True):
            if key not in verdicts and key not in todo:
                todo[key] = item
        if todo:
            self._check_budget()
            batch = self.client.messages.batches.create(
                requests=[
                    {"custom_id": key[:60], "params": self._params(*item)}
                    for key, item in todo.items()
                ]
            )
            while True:
                batch = self.client.messages.batches.retrieve(batch.id)
                if batch.processing_status == "ended":
                    break
                time.sleep(poll_seconds)
            short_to_key = {key[:60]: key for key in todo}
            for result in self.client.messages.batches.results(batch.id):
                if result.result.type != "succeeded":
                    continue  # errored/expired items simply stay unjudged (verdict False)
                key = short_to_key[result.custom_id]
                message = result.result.message
                verdict = self._verdict(message)
                self._store(key, verdict, self._cost(message.usage, batch=True))
                verdicts[key] = verdict
        return [verdicts.get(k, False) for k in keys]
