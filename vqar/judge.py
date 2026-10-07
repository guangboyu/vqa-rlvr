"""LLM judge for answer equivalence, used where rule-based matching fails.

A small hosted model (pinned in MODEL) judges whether a predicted answer is
semantically equivalent to the gold answer(s) ("grey" vs "gray", "4 people" vs "4").
Called only on rule-based mismatches, so volume stays small. Every verdict is cached
in sqlite keyed by (model, question, golds, normalized prediction), so repeated wrong
answers cost nothing. Cumulative spend is persisted and a hard budget cap raises
JudgeBudgetExceeded rather than overspending.

Two call paths:
- judge(): synchronous, cache-first.
- judge_many(): Message Batches API (50% price), used to rescore saved records. Items the
  API fails to judge come back as None, never as a silent "no".
"""

import hashlib
import os
import sqlite3
import time
from pathlib import Path

from vqar.metrics import normalize_answer

MODEL = "claude-haiku-4-5-20251001"  # pinned for reproducibility
# A stronger model that sees the image audits a sample of verdicts (scripts/audit_judge.py).
AUDIT_MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 1.0, 5.0  # USD per MTok
BATCH_DISCOUNT = 0.5
BATCH_CHUNK = 5_000  # requests per batch; a finished chunk is cached before the next is sent
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

            # Fallback: read the key from the project .env when the shell has none.
            if not os.environ.get("ANTHROPIC_API_KEY"):
                env_file = Path(__file__).parent.parent / ".env"
                if env_file.exists():
                    for line in env_file.read_text().splitlines():
                        key, _, value = line.partition("=")
                        if key.strip() == "ANTHROPIC_API_KEY" and value.strip():
                            os.environ["ANTHROPIC_API_KEY"] = value.strip()
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

    def _check_budget(self, projected_usd: float = 0.0) -> None:
        """Refuse a call whose projected cost would take total spend past the cap."""
        if self.spend() + projected_usd > self.budget_usd:
            raise JudgeBudgetExceeded(
                f"judge spend ${self.spend():.2f} + projected ${projected_usd:.2f} would pass "
                f"the ${self.budget_usd:.2f} cap"
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
        return cost * BATCH_DISCOUNT if batch else cost

    @staticmethod
    def estimate_batch_cost(items) -> float:
        """A deliberately high estimate: ~3 characters per token plus 20 tokens of framing."""
        tokens_in = sum(len(PROMPT) + len(q) + len("; ".join(g)) + len(p) for q, g, p in items) / 3
        tokens_in += 20 * len(items)
        return (tokens_in * PRICE_IN + 4 * len(items) * PRICE_OUT) / 1e6 * BATCH_DISCOUNT

    @staticmethod
    def _verdict(message) -> bool:
        text = next((b.text for b in message.content if b.type == "text"), "")
        return text.strip().lower().startswith("yes")

    def cached_verdict(self, question: str, golds: list[str], prediction: str) -> bool | None:
        """The stored verdict, or None when this item has not been judged; never calls the API."""
        return self._cached(self._key(question, golds, prediction))

    def uncached(self, items: list[tuple[str, list[str], str]]) -> list[tuple]:
        """The distinct items a call would still send to the API."""
        unique = {self._key(*item): item for item in items}
        return [item for key, item in unique.items() if self._cached(key) is None]

    def judge(self, question: str, golds: list[str], prediction: str) -> bool:
        """Synchronous single judgment (RL reward path). Cache-first, budget-capped."""
        key = self._key(question, golds, prediction)
        cached = self._cached(key)
        if cached is not None:
            return cached
        self._check_budget(self.estimate_batch_cost([(question, golds, prediction)])
                           / BATCH_DISCOUNT)
        message = self.client.messages.create(**self._params(question, golds, prediction))
        verdict = self._verdict(message)
        self._store(key, verdict, self._cost(message.usage))
        return verdict

    def judge_many(
        self, items: list[tuple[str, list[str], str]], poll_seconds: float = 30.0
    ) -> list[bool | None]:
        """Bulk judgments via the Message Batches API (50% price). Eval path.

        items: (question, gold_answers, prediction) triples. Order-preserving. Uncached items
        go out together in batches of at most BATCH_CHUNK; items whose request failed are
        retried once, and any still unjudged come back as None.
        """
        keys = [self._key(q, g, p) for q, g, p in items]
        verdicts: dict[str, bool] = {k: v for k in keys if (v := self._cached(k)) is not None}
        todo = {}  # key -> item, deduplicated
        for key, item in zip(keys, items, strict=True):
            if key not in verdicts and key not in todo:
                todo[key] = item
        for _attempt in range(2):
            pending = [key for key in todo if key not in verdicts]
            chunks = [{key: todo[key] for key in pending[start:start + BATCH_CHUNK]}
                      for start in range(0, len(pending), BATCH_CHUNK)]
            self._check_budget(sum(self.estimate_batch_cost(c.values()) for c in chunks))
            submitted = [(self._submit(chunk), chunk) for chunk in chunks]
            for batch_id, chunk in submitted:
                verdicts.update(self._collect(batch_id, chunk, poll_seconds))
        return [verdicts.get(k) for k in keys]

    def _submit(self, chunk: dict) -> str:
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": key[:60], "params": self._params(*item)}
                      for key, item in chunk.items()]
        )
        print(f"judge: submitted batch {batch.id} ({len(chunk)} requests)", flush=True)
        return batch.id

    def _collect(self, batch_id: str, chunk: dict, poll_seconds: float) -> dict[str, bool]:
        """Wait for one batch, then cache and return the verdicts that succeeded."""
        while self.client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(poll_seconds)
        short_to_key = {key[:60]: key for key in chunk}
        verdicts = {}
        for result in self.client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                continue  # errored, canceled or expired: left for the retry
            key = short_to_key[result.custom_id]
            message = result.result.message
            verdicts[key] = self._verdict(message)
            self._store(key, verdicts[key], self._cost(message.usage, batch=True))
        return verdicts
