"""Dataset screening: choose training datasets by measurement, not by reputation.

Each candidate gets one fixed battery of protocols on the same questions, drawn from its
training pool (the rl role, else the sft role; never dev or test rows):

- short and reasoning prompts, greedy: the reasoning payoff, accuracy(reasoning) −
  accuracy(short);
- direct and stepwise prompts, greedy: the matched payoff, accuracy(stepwise) −
  accuracy(direct). Both ask for the same tagged final answer and differ only in whether
  the model reasons first, so answer format cannot explain the difference;
- short prompt with black and with swapped images: the image dependence,
  accuracy(image) − accuracy(black) and accuracy(image) − accuracy(swapped);
- reasoning prompt, 8 samples at T = 1 with the GRPO completion budget (the protocol of
  scripts.learnability): pass@1, pass@8, the RL headroom pass@8 − pass@1, and the
  learnable share (1–7 of 8 correct), all under the GRPO correctness reward;
- short prompt, 8 samples at T = 1: the same measures for short-answer RL.

Accuracies use the normalized scorer, the one every role can score. Differences carry
paired 95% bootstrap intervals over questions. The gates are fixed here, before any
candidate is measured. Some gates hold for any RL (enough questions, a separable split, a
usable license, not saturated, the image needed); the rest decide which way of training can
work: reasoning-prompt RL needs readable reasoning answers, a learning signal and reasoning
no worse than a direct answer, short-answer RL needs a learning signal. A candidate passes
when the common gates hold and at least one route is viable (since 2026-10-10: RL counts
when it beats the starting model or SFT, with or without reasoning). Survivors are ranked by
matched payoff, then RL headroom.
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
# Added 2026-10-06, before any reasoning candidate was measured (Part 2 plan): the general
# datasets passed the gates above although reasoning hurt them, so reasoning value, image
# use and reward variation are now gates too.
MIN_MATCHED_PAYOFF_HIGH = 0.0  # stepwise must not be worse than direct: upper bound >= 0
MIN_IMAGE_DROP_LOW = 0.10  # black and swapped images each cost at least 10 points (lower bound)
MIN_LEARNABLE_SHARE = 0.25  # questions with 1-7 of 8 samples right
RANKING = "passed candidates by matched payoff, then RL headroom, both descending"
ROUTES = ("reasoning", "short")  # the rollout prompts RL can train under


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
    # Reasoning candidates, declared 2026-10-06 (licenses for research use; owner to confirm
    # before any release).
    "chartqa": Declaration("image", True, "GPL-3.0 (ChartQA release); charts from Statista, "
                           "Pew, OWID and OECD", True),
    "clevrcomp": Declaration("image", True, "CC BY 4.0", True),
    # No patient IDs are released; cases are the unit, and no image file is shared between
    # the official train and test cases (byte hashes, up to eight images per case).
    "medthink": Declaration("case", True, "CC BY-NC-SA 4.0 (research use)", True),
    # Medical candidates, declared 2026-10-10; the owner confirmed non-commercial use. No
    # patient IDs are released for any of them (see the registry on the re-splits).
    "kvasirx1": Declaration("image", True, "CC BY-NC 4.0 (images from HyperKvasir and "
                            "Kvasir-Instrument)", True),
    "omnimedvqa": Declaration("image", True, "open-access subset; each source dataset's own "
                              "license (research use)", True),
    # Whole source collections held out (the strictest unit available without patient IDs).
    "omnimedx": Declaration("source collection", True, "open-access subset; each source "
                            "dataset's own license (research use)", True),
    "slake": Declaration("image", True, "CC BY 4.0", True),
    "pathvqa": Declaration("image", True, "MIT (release); textbook and PEIR images, research "
                           "use", True),
    # Declared 2026-10-11: official patient-level splits; the owner accepted the
    # non-commercial agreement, which also forbids redistributing the data.
    "rexvqa": Declaration("patient", True, "ReXGradient-160K Non-Commercial Data Access and "
                          "Use Agreement (research use, no redistribution)", True),
    # Declared 2026-10-29: benchmark patients held out from training, dev by training patient.
    "deeptumor": Declaration("patient", True, "annotations CC BY 4.0; CT data from AbdomenAtlas "
                             "3.0 Mini, CC BY-NC-SA 4.0 (non-commercial)", True),
}


@dataclass(frozen=True)
class Battery:
    """One candidate's four protocols, all on the same questions."""

    short: Protocol
    reasoning: Protocol
    black: Protocol
    sampled: Protocol
    swapped: Protocol
    direct: Protocol
    stepwise: Protocol
    first: Protocol | None = None  # multi-image datasets: the first image only
    none: Protocol | None = None  # multi-image datasets: no image at all
    sampled_short: Protocol | None = None  # short-answer RL's learnability

    @property
    def role(self) -> str:
        return self.short.role

    @property
    def validation_only(self) -> bool:
        """Screened on test rows to check the module, never to choose the dataset."""
        return self.role not in TRAINING_ROLES

    def protocols(self) -> dict[str, Protocol]:
        """In run order: the image interventions pair with the short cell's predictions."""
        cells = {"short": self.short, "black": self.black, "swapped": self.swapped,
                 "reasoning": self.reasoning, "direct": self.direct, "stepwise": self.stepwise,
                 "sampled": self.sampled}
        cells.update({name: protocol for name, protocol in (
            ("sampled_short", self.sampled_short), ("first", self.first),
            ("none", self.none)) if protocol})
        return cells


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
    multi = DATASETS[dataset].max_images > 1
    return Battery(
        short=Protocol(prompt="short", **greedy),
        reasoning=Protocol(prompt="reasoning", **greedy),
        black=Protocol(prompt="short", image_mode="black", **greedy),
        swapped=Protocol(prompt="short", image_mode="swapped", **greedy),
        direct=Protocol(prompt="direct", **greedy),
        stepwise=Protocol(prompt="stepwise", **greedy),
        first=Protocol(prompt="short", image_mode="first", **greedy) if multi else None,
        none=Protocol(prompt="short", image_mode="none", **greedy) if multi else None,
        sampled=Protocol(dataset, prompt="reasoning", role=role, tier=tier,
                         max_tokens=SAMPLED_MAX_TOKENS, temperature=TEMPERATURE,
                         samples=SAMPLES),
        sampled_short=Protocol(dataset, prompt="short", role=role, tier=tier,
                               temperature=TEMPERATURE, samples=SAMPLES),
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
                   if not name.startswith("sampled")}
    result = {}
    if "sampled" in records:
        sampled = records["sampled"]
        counts = correct_counts(sampled)
        by_question["sampled"] = counts
        result.update(sampling_measures(counts, sampled.protocol["samples"]))
        result["parse_rate"] = parse_rate(read_lines(sampled), sampled.protocol["prompt"])
    if "sampled_short" in records:
        sampled = records["sampled_short"]
        counts = correct_counts(sampled)
        by_question["sampled_short"] = counts
        result.update({f"short_{name}": value for name, value in
                       sampling_measures(counts, sampled.protocol["samples"]).items()})
    if len({frozenset(questions) for questions in by_question.values()}) > 1:
        raise ValueError("The battery's records scored different questions.")
    if by_question:
        result["questions"] = len(next(iter(by_question.values())))
    for name in ("short", "reasoning", "direct", "stepwise"):
        if name in by_question:
            result[f"{name}_accuracy"] = mean(by_question[name])
    for name in ("reasoning", "direct", "stepwise"):
        if name in records:
            metrics = records[name].metrics
            result[f"{name}_behavior"] = {key: metrics[key] for key in (
                "mean_output_tokens", "truncated_fraction", "answer_tag_fraction")}
    pairs = {"reasoning_payoff": ("short", "reasoning"),
             "matched_payoff": ("direct", "stepwise"),
             "image_dependence": ("black", "short"),
             "wrong_image_dependence": ("swapped", "short"),
             "multi_image_gain": ("first", "short"),
             "text_only_drop": ("none", "short")}
    for measure_name, (reference, candidate) in pairs.items():
        if {reference, candidate} <= by_question.keys():
            result[measure_name] = paired_bootstrap(by_question[reference],
                                                    by_question[candidate])
    return result


def mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


def failed_gates(pool_size: int, declaration: Declaration | None, measures: dict) -> list[str]:
    """The gates any way of training needs. A measure not yet taken fails nothing."""
    short = measures.get("short_accuracy")
    drops = [measures.get(name) for name in ("image_dependence", "wrong_image_dependence")]
    checks = {
        "training questions": pool_size >= MIN_POOL_QUESTIONS,
        "split separable": declaration is not None and declaration.split_separable,
        "license": declaration is not None and declaration.license_usable,
        "not saturated": short is None or short < MAX_SHORT_ACCURACY,
        "image needed": all(drop is None or drop["low"] >= 100 * MIN_IMAGE_DROP_LOW
                            for drop in drops),
    }
    return [name for name, passed in checks.items() if not passed]


def route_failures(measures: dict) -> dict[str, list[str] | None]:
    """Per rollout prompt, the conditions RL under it fails; None until they are measured."""
    parse, learnable = measures.get("parse_rate"), measures.get("learnable_share")
    matched = measures.get("matched_payoff")
    reasoning = None
    if None not in (parse, learnable, matched):
        reasoning = [name for name, passed in (
            ("parse rate", parse >= MIN_PARSE_RATE),
            ("reasoning not worse", matched["high"] >= 100 * MIN_MATCHED_PAYOFF_HIGH),
            ("learnable", learnable >= MIN_LEARNABLE_SHARE)) if not passed]
    short_learnable = measures.get("short_learnable_share")
    short = None if short_learnable is None else (
        [] if short_learnable >= MIN_LEARNABLE_SHARE else ["learnable"])
    return {"reasoning": reasoning, "short": short}


def status_of(failed: list[str], routes: dict, complete: bool) -> str:
    """Rejected when a common gate fails or every measured route fails and none is left
    unmeasured; passed when complete and some route is viable; otherwise incomplete."""
    if failed:
        return "rejected"
    if any(found == [] for found in routes.values()):
        return "passed" if complete else "incomplete"
    if all(found is not None for found in routes.values()):
        return "rejected"
    return "incomplete"


def screen(model: str, dataset: str, cells: Battery | None,
           records: dict[str, RunRecord | None], pool_size: int) -> dict:
    """One candidate's row: its measures, the gates it fails, and its status."""
    found = {name: record for name, record in records.items() if record is not None}
    missing = [name for name, record in records.items() if record is None]
    declaration = DECLARED.get(dataset)
    measures = measure(found)
    failed = failed_gates(pool_size, declaration, measures)
    routes = route_failures(measures)
    status = status_of(failed, routes, complete=not missing and cells is not None)
    return {
        "model": model, "dataset": dataset, "display_name": DATASETS[dataset].display_name,
        "role": cells.role if cells else None,
        "validation_only": cells.validation_only if cells else False,
        "tier": cells.short.tier if cells else None,
        "pool_questions": pool_size,
        "declared": asdict(declaration) if declaration else None,
        **measures,
        "failed_gates": failed, "routes": routes, "status": status, "missing": missing,
        "records": {name: str(record.path) for name, record in found.items()},
    }


def rank(rows: list[dict]) -> list[dict]:
    """Per model: passed rows numbered in ranking order, then incomplete, then rejected."""
    def order(row):
        if row["status"] == "passed":
            headroom = max(row.get(name, {"delta": 0.0})["delta"]
                           for name in ("rl_headroom", "short_rl_headroom"))
            return (row["model"], 0, -row["matched_payoff"]["delta"], -headroom,
                    row["dataset"])
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
                      "min_matched_payoff_high": MIN_MATCHED_PAYOFF_HIGH,
                      "min_image_drop_low": MIN_IMAGE_DROP_LOW,
                      "min_learnable_share": MIN_LEARNABLE_SHARE,
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
        f"Gates for any RL: at least {MIN_POOL_QUESTIONS:,} training questions; split "
        "separable and license usable (declared); short-prompt accuracy < "
        f"{100 * MAX_SHORT_ACCURACY:.0f}%; black and swapped images each cost ≥ "
        f"{100 * MIN_IMAGE_DROP_LOW:.0f} points (lower bound). Routes: reasoning-prompt RL "
        f"needs parse rate ≥ {100 * MIN_PARSE_RATE:.0f}% of sampled reasoning responses, "
        "stepwise not worse than direct (upper bound ≥ 0) and a learnable share ≥ "
        f"{100 * MIN_LEARNABLE_SHARE:.0f}%; short-answer RL needs a short-prompt learnable "
        f"share ≥ {100 * MIN_LEARNABLE_SHARE:.0f}%. A candidate passes with the common gates "
        f"and at least one viable route. Ranking: {RANKING}.",
        "",
        "| Model | Rank | Dataset | Pool | Questions | Short (%) | Reasoning payoff | "
        "Matched payoff | pass@1 → pass@k (%) | RL headroom | Learnable (%) | "
        "Image: black | Image: swapped | Parse (%) | Short learnable (%) | Routes | Status |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        pool = row["role"] or "none"
        if row["validation_only"]:
            pool += " (validation only)"
        passes = ("—" if row.get("pass_at_1") is None else
                  f"{percent(row['pass_at_1'])} → {percent(row['pass_at_k'])}")
        status = row["status"]
        routes = row.get("routes") or {}
        viable = ", ".join(name for name in ROUTES if routes.get(name) == []) or (
            "none" if all(routes.get(name) is not None for name in ROUTES) else "—")
        if row["failed_gates"]:
            status += " (failed: " + ", ".join(row["failed_gates"]) + ")"
        elif row["missing"]:
            status += ": missing " + ", ".join(row["missing"])
        lines.append("| " + " | ".join([
            row["model"], str(row["rank"] or "—"), row["display_name"], pool,
            str(row.get("questions", "—")), percent(row.get("short_accuracy")),
            interval(row.get("reasoning_payoff")), interval(row.get("matched_payoff")),
            passes, interval(row.get("rl_headroom")), percent(row.get("learnable_share")),
            interval(row.get("image_dependence")), interval(row.get("wrong_image_dependence")),
            percent(row.get("parse_rate")), percent(row.get("short_learnable_share")),
            viable, status]) + " |")
    return "\n".join(lines) + "\n"
