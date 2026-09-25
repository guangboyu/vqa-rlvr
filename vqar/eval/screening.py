"""Dataset screening: choose training datasets by measurement, not by reputation.

Each candidate gets one fixed battery of four protocols on the same questions, drawn from
its training pool (the rl role, else the sft role; never dev or test rows):

- short and reasoning prompts, greedy: the reasoning payoff, accuracy(reasoning) −
  accuracy(short);
- short prompt with black images: the image dependence, accuracy(image) − accuracy(black);
- reasoning prompt, 8 samples at T = 1 with the GRPO completion budget (the protocol of
  scripts.learnability): pass@1, pass@8, the RL headroom pass@8 − pass@1, and the
  learnable share (1–7 of 8 correct), all under the GRPO correctness reward.

Accuracies use the normalized scorer, the one every role can score. Differences carry
paired 95% bootstrap intervals over questions. The gates are fixed here, before any
candidate is measured; survivors are ranked by reasoning payoff, then RL headroom.
"""

import json
from dataclasses import asdict, dataclass

from vqar.answers import reward_answer
from vqar.config import GRPO_PRESETS
from vqar.eval.paired import paired_bootstrap, question_scores
from vqar.eval.protocol import Protocol
from vqar.eval.record import RunRecord
from vqar.eval.runner import sampled_metrics
from vqar.registry import DATASETS, PROMPTS, TRAINING_ROLES
from vqar.splits import read_split

TIER = "dev"  # up to 500 pool questions; the 2B learnability pre-check used the same tier
SAMPLES = 8
TEMPERATURE = 1.0  # vqar.rl.ROLLOUT_TEMPERATURE, literal so screening loads without the trainer
SAMPLED_MAX_TOKENS = GRPO_PRESETS["grpo_2b_main"].max_completion_length
VARIANT = "normalized"

# Hard gates, fixed before any candidate is measured.
MIN_POOL_QUESTIONS = 2_000  # rows in the training pool role
MIN_PARSE_RATE = 0.95  # sampled reasoning responses the reward's answer rule can read
MAX_SHORT_ACCURACY = 0.90  # at or above it, the dataset is saturated
RANKING = "passed candidates by reasoning payoff, then RL headroom, both descending"


@dataclass(frozen=True)
class Declaration:
    """What screening cannot measure about a dataset, declared by hand."""

    split_unit: str  # what dev and test rows are held out by: "image" or "patient"
    split_separable: bool
    license: str
    license_usable: bool


# Registered datasets' held-out images are also checked by vqar.splits before training.
DECLARED = {
    "vqav2": Declaration("image", True, "annotations CC BY 4.0; COCO images, Flickr terms",
                         True),
    "gqa": Declaration("image", True, "CC BY 4.0 (Visual Genome images)", True),
    "clevr": Declaration("image", True, "CC BY 4.0", True),
    "textvqa": Declaration("image", True, "annotations CC BY 4.0; Open Images CC BY 2.0",
                           True),
}


@dataclass(frozen=True)
class Battery:
    """One candidate's four protocols, all on the same questions."""

    short: Protocol
    reasoning: Protocol
    black: Protocol
    sampled: Protocol

    @property
    def role(self) -> str:
        return self.short.role

    @property
    def validation_only(self) -> bool:
        """Screened on test rows to check the module, never to choose the dataset."""
        return self.role not in TRAINING_ROLES

    def protocols(self) -> dict[str, Protocol]:
        """In run order: the black-image cell pairs with the short cell's predictions."""
        return {"short": self.short, "black": self.black, "reasoning": self.reasoning,
                "sampled": self.sampled}


def pool_role(dataset: str) -> str | None:
    """The training role screening reads: rl, else sft, else none."""
    return next((role for role in ("rl", "sft") if role in DATASETS[dataset].subsets), None)


def battery(dataset: str, tier: str = TIER, validate_on_test: bool = False) -> Battery | None:
    """The battery on the dataset's training pool; None when it has none.

    With `validate_on_test`, a dataset without a training pool is screened on its test role
    instead, to check the module against a known answer. It still fails the pool gate.
    """
    role = pool_role(dataset) or ("test" if validate_on_test else None)
    if role is None:
        return None
    greedy = {"dataset": dataset, "role": role, "tier": tier,
              "screening": role in TRAINING_ROLES}
    return Battery(
        short=Protocol(prompt="short", **greedy),
        reasoning=Protocol(prompt="reasoning", **greedy),
        black=Protocol(prompt="short", image_mode="black", **greedy),
        sampled=Protocol(dataset, prompt="reasoning", role=role, tier=tier,
                         max_tokens=SAMPLED_MAX_TOKENS, temperature=TEMPERATURE,
                         samples=SAMPLES),
    )


def pool_questions(dataset: str, data_dir) -> int:
    """Rows in the dataset's training pool role, from its split file; 0 without one."""
    role = pool_role(dataset)
    if role is None:
        return 0
    return sum(len(part["indices"]) for part in read_split(data_dir, dataset)["roles"][role])


def read_lines(record: RunRecord) -> list[dict]:
    return [json.loads(line) for line in record.predictions_path().read_text().splitlines()]


def correct_counts(record: RunRecord) -> dict[str, int]:
    """Per question, the samples that earned the GRPO correctness reward.

    Newer records store the counts; older ones are recounted from their predictions by the
    same function that stores them.
    """
    stored = record.metrics["learnability"].get("correct_samples")
    if stored is not None:
        return stored
    lines = read_lines(record)
    protocol = record.protocol
    return sampled_metrics(lines, protocol["dataset"],
                           protocol["prompt"])["learnability"]["correct_samples"]


def sampling_measures(counts: dict[str, int], samples: int) -> dict:
    """pass@1 (mean share of correct samples), pass@k, their paired difference (the RL
    headroom), and the learnable share (neither all nor no samples correct)."""
    pass_at_1 = {qid: n / samples for qid, n in counts.items()}
    pass_at_k = {qid: float(n > 0) for qid, n in counts.items()}
    return {
        "samples": samples,
        "pass_at_1": sum(pass_at_1.values()) / len(counts),
        "pass_at_k": sum(pass_at_k.values()) / len(counts),
        "rl_headroom": paired_bootstrap(pass_at_1, pass_at_k),
        "learnable_share": sum(0 < n < samples for n in counts.values()) / len(counts),
    }


def parse_rate(lines: list[dict], prompt: str) -> float:
    """Share of responses from which the reward's answer rule reads a non-empty answer."""
    rule = PROMPTS[prompt].answer_rule
    return sum(bool(reward_answer(line["raw"], rule)) for line in lines) / len(lines)


def scores(record: RunRecord) -> dict[str, float]:
    return question_scores(record.predictions_path(), VARIANT)


def measure(records: dict[str, RunRecord]) -> dict:
    """Every measure whose records exist; the four cells must share their questions."""
    by_question = {name: scores(record) for name, record in records.items()
                   if name != "sampled"}
    result = {}
    if "sampled" in records:
        sampled = records["sampled"]
        counts = correct_counts(sampled)
        by_question["sampled"] = counts
        result.update(sampling_measures(counts, sampled.protocol["samples"]))
        result["parse_rate"] = parse_rate(read_lines(sampled), sampled.protocol["prompt"])
    if len({frozenset(questions) for questions in by_question.values()}) > 1:
        raise ValueError("The battery's records scored different questions.")
    if by_question:
        result["questions"] = len(next(iter(by_question.values())))
    for name in ("short", "reasoning"):
        if name in by_question:
            result[f"{name}_accuracy"] = mean(by_question[name])
    if {"short", "reasoning"} <= by_question.keys():
        result["reasoning_payoff"] = paired_bootstrap(by_question["short"],
                                                      by_question["reasoning"])
    if {"short", "black"} <= by_question.keys():
        result["image_dependence"] = paired_bootstrap(by_question["black"],
                                                      by_question["short"])
    return result


def mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


def failed_gates(pool_size: int, declaration: Declaration | None, measures: dict) -> list[str]:
    """The gates a candidate fails. A measure not yet taken fails nothing."""
    parse = measures.get("parse_rate")
    short = measures.get("short_accuracy")
    checks = {
        "training questions": pool_size >= MIN_POOL_QUESTIONS,
        "split separable": declaration is not None and declaration.split_separable,
        "license": declaration is not None and declaration.license_usable,
        "parse rate": parse is None or parse >= MIN_PARSE_RATE,
        "not saturated": short is None or short < MAX_SHORT_ACCURACY,
    }
    return [name for name, passed in checks.items() if not passed]


def screen(model: str, dataset: str, cells: Battery | None,
           records: dict[str, RunRecord | None], pool_size: int) -> dict:
    """One candidate's row: its measures, the gates it fails, and its status."""
    found = {name: record for name, record in records.items() if record is not None}
    missing = [name for name, record in records.items() if record is None]
    declaration = DECLARED.get(dataset)
    measures = measure(found)
    failed = failed_gates(pool_size, declaration, measures)
    status = "rejected" if failed else "incomplete" if missing or cells is None else "passed"
    return {
        "model": model, "dataset": dataset, "display_name": DATASETS[dataset].display_name,
        "role": cells.role if cells else None,
        "validation_only": cells.validation_only if cells else False,
        "tier": cells.short.tier if cells else None,
        "pool_questions": pool_size,
        "declared": asdict(declaration) if declaration else None,
        **measures,
        "failed_gates": failed, "status": status, "missing": missing,
        "records": {name: str(record.path) for name, record in found.items()},
    }


def rank(rows: list[dict]) -> list[dict]:
    """Per model: passed rows numbered in ranking order, then incomplete, then rejected."""
    def order(row):
        if row["status"] == "passed":
            return (row["model"], 0, -row["reasoning_payoff"]["delta"],
                    -row["rl_headroom"]["delta"], row["dataset"])
        return (row["model"], 1 if row["status"] == "incomplete" else 2, 0, 0, row["dataset"])

    ranked, position = [], {}
    for row in sorted(rows, key=order):
        number = None
        if row["status"] == "passed":
            number = position[row["model"]] = position.get(row["model"], 0) + 1
        ranked.append({**row, "rank": number})
    return ranked


def summary() -> dict:
    """The gates and ranking rule, written beside every table."""
    return {"tier": TIER, "samples": SAMPLES, "temperature": TEMPERATURE,
            "sampled_max_tokens": SAMPLED_MAX_TOKENS, "scorer": VARIANT,
            "gates": {"min_pool_questions": MIN_POOL_QUESTIONS,
                      "min_parse_rate": MIN_PARSE_RATE,
                      "max_short_accuracy": MAX_SHORT_ACCURACY,
                      "declared": ["split separable", "license"]},
            "ranking": RANKING}


def interval(result: dict | None) -> str:
    if result is None:
        return "—"
    return f"{result['delta']:+.1f} [{result['low']:+.1f}, {result['high']:+.1f}]"


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def markdown(rows: list[dict]) -> str:
    """The ranked table, preceded by the gates and ranking rule it was built under."""
    lines = [
        "Screening battery on each candidate's training pool (normalized scorer; "
        "differences in points with paired 95% bootstrap intervals).",
        "",
        f"Gates: at least {MIN_POOL_QUESTIONS:,} training questions; split separable and "
        f"license usable (declared); parse rate ≥ {100 * MIN_PARSE_RATE:.0f}% of sampled "
        f"reasoning responses; short-prompt accuracy < {100 * MAX_SHORT_ACCURACY:.0f}%. "
        f"Ranking: {RANKING}.",
        "",
        "| Model | Rank | Dataset | Pool | Questions | Reasoning payoff | Short (%) | "
        "pass@1 → pass@k (%) | RL headroom | Learnable (%) | Image dependence | "
        "Parse (%) | Status |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        pool = row["role"] or "none"
        if row["validation_only"]:
            pool += " (validation only)"
        passes = ("—" if row.get("pass_at_1") is None else
                  f"{percent(row['pass_at_1'])} → {percent(row['pass_at_k'])}")
        status = row["status"]
        if row["failed_gates"]:
            status += ": " + ", ".join(row["failed_gates"])
        elif row["missing"]:
            status += ": missing " + ", ".join(row["missing"])
        lines.append("| " + " | ".join([
            row["model"], str(row["rank"] or "—"), row["display_name"], pool,
            str(row.get("questions", "—")), interval(row.get("reasoning_payoff")),
            percent(row.get("short_accuracy")), passes, interval(row.get("rl_headroom")),
            percent(row.get("learnable_share")), interval(row.get("image_dependence")),
            percent(row.get("parse_rate")), status]) + " |")
    return "\n".join(lines) + "\n"
