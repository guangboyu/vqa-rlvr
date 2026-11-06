"""Beyond accuracy: how far each model moved, how much it relies on the image, how well its
confidence tracks its accuracy. Reads the records and KL files the declared sets produced.

    python -m scripts.beyond_accuracy        # writes results/beyond-accuracy.json

Per [[beyond_accuracy]] set in experiments.toml and per model (its starting model included):
- accuracy: the greedy original-image cell, normalized scorer (the training reward's scorer);
- kl: forward KL from the starting model on the task's questions (scripts.kl_to_base);
- reliance: accuracy with a black image and its change from the starting model's, and the
  accuracy lost with a black and with a swapped image (original − changed), paired 95%
  intervals;
- calibration, from the first 8 samples at temperature 1 of each question: the majority answer
  (normalized text) and the share of samples that give it as the confidence; the majority
  answer's accuracy, the expected calibration error over 10 equal-width bins, and the accuracy
  on the half and on the 80% of questions the model is most confident about.
"""

import json
import tomllib
from collections import Counter, defaultdict

from vqar.eval.paired import paired_bootstrap, question_scores
from vqar.eval.record import RunRecord, record_files
from vqar.eval.report import ROOT
from vqar.eval.runner import RECORDS_DIR
from vqar.metrics import normalize_answer

OUTPUT = ROOT / "results" / "beyond-accuracy.json"
VARIANT = "normalized"
SAMPLES = 8
BINS = 10


def find(records: list[RunRecord], label: str, dataset: str, prompt: str, tier: str,
         image_mode: str = "original", sampled: bool = False) -> RunRecord | None:
    """The record of one cell: greedy, or sampled at T = 1 with at least SAMPLES samples."""
    for record in records:
        p = record.protocol
        if (record.model.get("label") != label or record.model.get("step") is not None
                or p["dataset"] != dataset or p["role"] != "test" or p["tier"] != tier
                or p["prompt"] != prompt or p["image_mode"] != image_mode):
            continue
        if sampled and p["temperature"] == 1.0 and p["samples"] >= SAMPLES:
            return record
        if not sampled and p["samples"] == 1 and p["temperature"] == 0.0:
            return record
    return None


def majority(lines: list[dict]) -> dict[str, tuple[float, float]]:
    """Per question: (confidence, correctness) of the majority answer of the first samples."""
    by_question = defaultdict(list)
    for line in lines:
        if line["sample"] < SAMPLES:
            by_question[line["qid"]].append(line)
    result = {}
    for qid, samples in by_question.items():
        keys = [normalize_answer(s["prediction"]) for s in samples]
        answer, count = Counter(keys).most_common(1)[0]
        score = next(s["scores"][VARIANT] for s, key in zip(samples, keys, strict=True)
                     if key == answer)
        result[qid] = (count / len(samples), score)
    return result


def calibration_error(pairs: list[tuple[float, float]], bins: int = BINS) -> float:
    """Expected calibration error: the gap between mean confidence and mean correctness,
    weighted by how many questions fall in each equal-width confidence bin."""
    grouped = defaultdict(list)
    for confidence, correct in pairs:
        grouped[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    return sum(len(group) / len(pairs) * abs(sum(c for c, _ in group) / len(group)
                                             - sum(k for _, k in group) / len(group))
               for group in grouped.values())


def selective_accuracy(pairs: list[tuple[float, float]], coverage: float) -> float:
    """Accuracy on the share `coverage` of questions with the highest confidence."""
    ranked = sorted(pairs, key=lambda pair: -pair[0])
    kept = ranked[:max(1, round(coverage * len(ranked)))]
    return sum(correct for _, correct in kept) / len(kept)


def calibration(record: RunRecord) -> dict:
    lines = [json.loads(line) for line in record.predictions_path().read_text().splitlines()]
    pairs = list(majority(lines).values())
    return {"samples": SAMPLES, "questions": len(pairs),
            "majority_accuracy": 100 * sum(k for _, k in pairs) / len(pairs),
            "mean_confidence": 100 * sum(c for c, _ in pairs) / len(pairs),
            "calibration_error": 100 * calibration_error(pairs),
            "selective_50": 100 * selective_accuracy(pairs, 0.5),
            "selective_80": 100 * selective_accuracy(pairs, 0.8)}


def reliance(original: RunRecord | None, changed: RunRecord | None) -> dict | None:
    """original − changed with a paired interval: the accuracy lost to a changed image, or,
    given two black-image cells, how much more one model answers without the image."""
    if original is None or changed is None:
        return None
    return paired_bootstrap(question_scores(changed.predictions_path(), VARIANT),
                            question_scores(original.predictions_path(), VARIANT))


def kl(spec: dict, label: str) -> dict | None:
    path = ROOT / "results" / "kl" / f"{spec['dataset']}-{spec['prompt']}-{spec['base']}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["models"].get(label)


def measure_set(spec: dict, records: list[RunRecord]) -> dict:
    rows = []
    start_black = find(records, spec["base"], spec["dataset"], spec["prompt"], spec["tier"],
                       image_mode="black")
    for label in [spec["base"], *spec["models"]]:
        args = (records, label, spec["dataset"], spec["prompt"], spec["tier"])
        original = find(*args)
        if original is None:
            rows.append({"model": label, "missing": True})
            continue
        sampled = find(*args, sampled=True)
        black = find(*args, image_mode="black")
        rows.append({
            "model": label,
            "accuracy": 100 * original.metrics[VARIANT]["overall"],
            "output_tokens": original.metrics["mean_output_tokens"],
            "kl": kl(spec, label),
            "black_accuracy": 100 * black.metrics[VARIANT]["overall"] if black else None,
            "black_gain": reliance(black, start_black),
            "black_drop": reliance(original, black),
            "swapped_drop": reliance(original, find(*args, image_mode="swapped")),
            "calibration": calibration(sampled) if sampled else None,
        })
    return {**spec, "scorer": VARIANT, "rows": rows}


def main():
    declared = tomllib.loads((ROOT / "experiments.toml").read_text())["beyond_accuracy"]
    records = [RunRecord.load(path) for path in record_files(RECORDS_DIR)]
    sets = [measure_set(spec, records) for spec in declared]
    OUTPUT.write_text(json.dumps({"sets": sets}, indent=1) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
