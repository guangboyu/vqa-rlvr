"""run_eval: generate, score with both scorer variants, summarize, cache, and resume."""

import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from vqar.eval.examples import image_donors, select_rows, shown_images, to_example
from vqar.eval.protocol import Protocol, protocol_identity, slug
from vqar.eval.record import ModelRef, RunRecord, model_id, record_path
from vqar.inference import build_messages, extract_answer
from vqar.metrics import normalize_answer
from vqar.registry import DATASETS, PROMPTS, SCORER_VARIANTS

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
RECORDS_DIR = ROOT / "results" / "records"
SOURCES = ("vqar/eval/__init__.py", "vqar/eval/examples.py", "vqar/eval/protocol.py",
           "vqar/eval/record.py", "vqar/eval/runner.py", "vqar/inference.py",
           "vqar/config.py", "vqar/metrics.py", "vqar/registry.py", "vqar/splits.py",
           "uv.lock")


def run_eval(model: ModelRef, protocol: Protocol, generator, *, data_dir: Path = DATA_DIR,
             records_dir: Path = RECORDS_DIR, batch_size: int = 128) -> RunRecord:
    """Evaluate one (model, protocol) cell, or return its cached record.

    `generator.generate(conversations, max_tokens=, temperature=, samples=, seed=)` returns,
    per conversation, `samples` dicts with text, output_tokens, and finish_reason. Batches
    are saved as they finish, so an interrupted run resumes where it stopped.
    """
    rows, model_identity, identity, path = resolve(model, protocol, data_dir, records_dir)
    if path.exists():
        cached = RunRecord.load(path)
        if (cached.model_id, cached.protocol_id) != (model_id(model_identity),
                                                     Protocol.id_of(identity)):
            raise ValueError(f"{path} holds a different cell; its short ids collide")
        return cached

    if protocol.image_mode != "original":
        # Interventions are reported against the same questions with original images.
        original = original_lines(model, protocol, data_dir, records_dir)
        if original is None:
            raise ValueError("Evaluate the original-image cell first; its saved predictions "
                             f"pair with this {protocol.image_mode}-image cell.")

    progress = path.with_suffix(".progress")
    qids = list(rows["qid"])
    lines = load_progress(progress, {"model": model_identity, "protocol": identity}, qids,
                          protocol.samples)
    donors = image_donors(rows, protocol.image_mode)
    template = PROMPTS[protocol.prompt].text
    done = len({line["qid"] for line in lines})
    for start in range(done, len(rows), batch_size):
        indices = range(start, min(start + batch_size, len(rows)))
        examples = [to_example(rows[i]) for i in indices]
        sources = [to_example(rows[donors[i]]) for i in indices]
        conversations = [
            build_messages(e.question, shown_images(e, s, protocol.image_mode), template)
            for e, s in zip(examples, sources, strict=True)
        ]
        outputs = generator.generate(conversations, max_tokens=protocol.budget,
                                     temperature=protocol.temperature, samples=protocol.samples,
                                     seed=protocol.seed if protocol.sampled else None)
        if any(len(samples) != protocol.samples for samples in outputs):
            raise ValueError(f"The generator must return {protocol.samples} samples per question")
        batch = [prediction_line(example, source.qid, index, output, protocol.role)
                 for example, source, samples in zip(examples, sources, outputs, strict=True)
                 for index, output in enumerate(samples)]
        atomic_json(progress / f"batch-{start:08d}.json", batch)
        lines.extend(batch)
        print(f"{protocol.slug()}: saved {start + len(examples)}/{len(rows)}", flush=True)

    metrics = summarize(lines, protocol.dataset)
    if protocol.image_mode != "original":
        metrics["paired_with_original"] = paired_with_original(lines, original)
    record = RunRecord(model_identity, identity, metrics, provenance())
    saved = record.save(path, lines)
    shutil.rmtree(progress, ignore_errors=True)
    return saved


def original_lines(model, protocol, data_dir, records_dir) -> list[dict] | None:
    """Predictions of the same cell with original images, when they were saved."""
    _, _, _, path = resolve(model, replace(protocol, image_mode="original"), data_dir,
                            records_dir)
    sidecar = path.with_name(path.stem + ".predictions.jsonl")
    if not sidecar.exists():
        return None
    return [json.loads(line) for line in sidecar.read_text().splitlines()]


def paired_with_original(changed: list[dict], original: list[dict]) -> dict:
    """How an image intervention moved scores and answers on the same questions.

    Scores keep the original image's references, so they measure agreement with those
    references, not accuracy on the changed image. Only first samples are compared.
    """
    by_qid = {line["qid"]: line for line in original if line["sample"] == 0}
    pairs = [(by_qid[line["qid"]], line) for line in changed if line["sample"] == 0]
    variants = [v for v in SCORER_VARIANTS if pairs[0][0]["scores"][v] is not None]
    return {
        "n": len(pairs),
        "original": {v: sum(o["scores"][v] for o, _ in pairs) / len(pairs) for v in variants},
        "normalized_answer_changed_fraction": sum(
            normalize_answer(o["prediction"]) != normalize_answer(c["prediction"])
            for o, c in pairs
        ) / len(pairs),
    }


@dataclass(frozen=True)
class Cell:
    """Where a (model, protocol) cell's record lives, and whether it already exists."""

    path: Path
    questions: int
    model_id: str
    protocol_id: str

    @property
    def cached(self) -> bool:
        return self.path.exists()


def find_cell(model: ModelRef, protocol: Protocol, *, data_dir: Path = DATA_DIR,
              records_dir: Path = RECORDS_DIR) -> Cell:
    """Resolve a cell without generating anything (for dry runs and missing-cell checks)."""
    rows, model_identity, identity, path = resolve(model, protocol, data_dir, records_dir)
    return Cell(path, len(rows), model_id(model_identity), Protocol.id_of(identity))


def resolve(model, protocol, data_dir, records_dir):
    rows, data = select_rows(protocol, Path(data_dir))
    identity = protocol_identity(protocol, data)
    model_identity = model.identity()
    return rows, model_identity, identity, record_path(Path(records_dir), model_identity,
                                                         slug(identity), identity)


def prediction_line(example, image_qid: str, sample: int, output: dict, role: str) -> dict:
    spec = DATASETS[example.dataset]
    prediction = extract_answer(output["text"])
    # The official scorer is undefined where the authors' evaluator cannot score the
    # references, e.g. one VQAv2 reference where the evaluator needs ten.
    scores = {variant: None if variant == "official" and role not in spec.official_roles
              else spec.scorer(variant)(prediction, example.answers)
              for variant in SCORER_VARIANTS}
    return {"qid": example.qid, "sample": sample, "question": example.question,
            "raw": output["text"], "prediction": prediction, "scores": scores,
            "answers": example.answers, "answer_type": example.answer_type,
            "image_qid": image_qid, "output_tokens": output["output_tokens"],
            "finish_reason": output["finish_reason"]}


def summarize(lines: list[dict], dataset: str) -> dict:
    """Both scorer variants overall and per answer type, plus behavior metrics.

    With several samples per question, accuracies average over samples (avg@k). Two casing
    diagnostics show how much of the gap between the variants is capitalization.
    """
    if not lines:
        raise ValueError("Evaluation subset is empty.")
    metrics = {"n": len({line["qid"] for line in lines})}
    for variant in SCORER_VARIANTS:
        scores = [line["scores"][variant] for line in lines]
        if any(score is None for score in scores):
            metrics[variant] = None
            continue
        by_type = defaultdict(list)
        for line, score in zip(lines, scores, strict=True):
            by_type[line["answer_type"] or "all"].append(score)
        metrics[variant] = {
            "overall": sum(scores) / len(scores),
            "by_answer_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
        }
    count = len(lines)
    metrics.update({
        "mean_output_tokens": sum(line["output_tokens"] for line in lines) / count,
        "truncated_fraction": sum(line["finish_reason"] == "length" for line in lines) / count,
        "answer_tag_fraction": sum("</answer>" in line["raw"] for line in lines) / count,
        "empty_answer_fraction": sum(not line["prediction"] for line in lines) / count,
    })
    if metrics["official"] is not None:
        official = DATASETS[dataset].official
        lowercased = [official(line["prediction"].lower(), line["answers"]) for line in lines]
        metrics["official_on_lowercased_answers"] = sum(lowercased) / count
        metrics["answers_scored_differently"] = sum(
            not math.isclose(line["scores"]["official"], line["scores"]["normalized"],
                             abs_tol=1e-12)
            for line in lines
        )
    if len(lines) > metrics["n"]:
        metrics.update(sampled_metrics(lines, dataset))
    return metrics


def sampled_metrics(lines: list[dict], dataset: str) -> dict:
    """pass@k per scorer variant, and learnability under the training reward.

    pass@k is the share of questions with at least one fully correct sample. Learnability
    scores each sample with the GRPO correctness reward (the tagged answer, normalized
    scorer), so it counts the prompts whose sample group would carry a learning signal:
    those whose pass rate lies strictly between 0 and 1.
    """
    from vqar.rewards import correctness_reward

    by_question = defaultdict(list)
    for line in lines:
        by_question[line["qid"]].append(line)
    k = len(next(iter(by_question.values())))
    result = {"samples_per_question": k, "pass_at_k": {}}
    for variant in SCORER_VARIANTS:
        if lines[0]["scores"][variant] is None:
            result["pass_at_k"][variant] = None
            continue
        result["pass_at_k"][variant] = sum(
            max(line["scores"][variant] for line in group) >= 1.0 - 1e-12
            for group in by_question.values()
        ) / len(by_question)
    pass_rates = []
    for group in by_question.values():
        rewards = correctness_reward(
            prompts=None, completions=[line["raw"] for line in group],
            answers=[line["answers"] for line in group], dataset=[dataset] * len(group))
        pass_rates.append(sum(reward >= 1.0 - 1e-12 for reward in rewards))
    histogram = {f"{hits}/{k}": pass_rates.count(hits) for hits in range(k + 1)}
    result["learnability"] = {
        "rule": "GRPO correctness reward: normalized score of the tagged answer",
        "correct_samples_histogram": histogram,
        "fraction_between_0_and_1": sum(0 < hits < k for hits in pass_rates) / len(pass_rates),
    }
    return result


def load_progress(directory: Path, identity: dict, qids: list[str], samples: int) -> list[dict]:
    """Accept only complete, contiguous batches saved by this exact cell."""
    state = directory / "identity.json"
    if state.exists():
        if json.loads(state.read_text()) != json.loads(json.dumps(identity)):
            raise ValueError(f"Saved progress belongs to another evaluation: {directory}")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Saved progress has no identity; remove it to start over: {directory}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(state, identity)
    lines, done = [], 0
    for path in sorted(directory.glob("batch-*.json")):
        if path.name != f"batch-{done:08d}.json":
            raise ValueError(f"Missing or overlapping evaluation batch: {path}")
        batch = json.loads(path.read_text())
        batch_qids = qids[done:done + len(batch) // samples]
        expected = [(qid, sample) for qid in batch_qids for sample in range(samples)]
        if not batch or [(line["qid"], line["sample"]) for line in batch] != expected:
            raise ValueError(f"Prediction IDs do not match the evaluation subset: {path}")
        lines.extend(batch)
        done += len(batch_qids)
    return lines


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def provenance() -> dict:
    """Where a record came from. Recorded for reference, never part of any identity."""
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip()

    versions = {}
    for name in ("torch", "transformers", "vllm", "datasets"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "code_version": git("rev-parse", "HEAD") or None,
        "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCES if (ROOT / name).is_file()},
        "versions": versions,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }
