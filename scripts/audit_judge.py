"""Audit the LLM judge with a stronger model that sees the image.

    python -m scripts.audit_judge --dry-run    # draw the sample, estimate the cost
    python -m scripts.audit_judge              # writes results/judge-audit.json

The judge (vqar/judge.py) runs on a small model and never sees the image. This audit draws a
seeded, stratified sample of its verdicts on the headline records' exact-match misses: misses
it accepted, by dataset and answer length, and misses it rejected, by dataset. Opus 5.5
(vqar.judge.AUDIT_MODEL), shown the image, the question and the references, says whether each
answer is correct.
Per stratum, the share the auditor accepts is the judge's precision (accepted misses) or the
share of correct answers it missed (rejected misses). scripts.judge_records turns these rates
into audit-corrected accuracies.
"""

import argparse
import base64
import io
import json
import random
import time

from datasets import Image as ImageFeature
from PIL import Image

from scripts.judge_records import MATRIX, item, misses, read_cell, stratum
from vqar.eval.matrix import declared_records, load_matrix
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR, RECORDS_DIR
from vqar.judge import AUDIT_MODEL, Judge
from vqar.splits import load_role

OUTPUT = ROOT / "results" / "judge-audit.json"
DATASETS = ("vqav2", "gqa", "textvqa")
PER_STRATUM = 50
MAX_SIDE = 1024
PROMPT = """You are checking an automatic grader of visual question answering.
Question: {question}
Reference answers from human annotators (they sometimes disagree): {golds}
Candidate answer: {prediction}

Looking at the image, is the candidate a correct final answer to the question, one a careful \
human grader would accept? Accept synonyms, paraphrases, singular or plural, and capitalization \
differences. Do not accept an answer that is wrong for the image, only partly right, or \
unfinished reasoning instead of a final answer. Reply with exactly one word: yes or no."""


def candidates(judge: Judge) -> dict[str, dict[tuple, dict]]:
    """Distinct judged misses of the headline cells, grouped by stratum."""
    arms, protocols = load_matrix(ROOT / "experiments.toml", MATRIX)
    records = declared_records(arms, protocols, RECORDS_DIR, DATA_DIR, ROOT)
    strata = {}
    for (_, protocol), record in records.items():
        if protocol.dataset not in DATASETS:
            continue
        for row in misses(read_cell(record)):
            verdict = judge.cached_verdict(*item(row))
            if verdict is None:
                continue
            key = (protocol.dataset, row["qid"], row["prediction"])
            strata.setdefault(stratum(protocol.dataset, verdict, row["prediction"]), {})[key] = {
                "dataset": protocol.dataset, "qid": row["qid"], "question": row["question"],
                "answers": row["answers"], "prediction": row["prediction"],
                "judge_accepted": verdict}
    return strata


def draw(strata: dict, seed: int = 0) -> list[dict]:
    sample = []
    for name in sorted(strata):
        rows = [strata[name][key] for key in sorted(strata[name])]
        chosen = random.Random(f"{seed}|{name}").sample(rows, min(PER_STRATUM, len(rows)))
        sample += [{**row, "stratum": name, "stratum_size": len(rows)} for row in chosen]
    return sample


def image_payloads(sample: list[dict]) -> dict[tuple[str, str], tuple[str, str]]:
    """(dataset, qid) -> (media type, base64 data), images shrunk to at most MAX_SIDE pixels."""
    wanted = {(row["dataset"], row["qid"]) for row in sample}
    payloads = {}
    for dataset in DATASETS:
        rows = load_role(DATA_DIR, [dataset], "test", seed=None)
        rows = rows.cast_column("image", ImageFeature(decode=False))
        for qid, image in zip(rows["qid"], rows["image"], strict=True):
            if (dataset, qid) in wanted:
                raw = image["bytes"] or open(image["path"], "rb").read()
                payloads[dataset, qid] = encode(raw)
    return payloads


def encode(raw: bytes) -> tuple[str, str]:
    with Image.open(io.BytesIO(raw)) as picture:
        picture = picture.convert("RGB")
        picture.thumbnail((MAX_SIDE, MAX_SIDE))
        buffer = io.BytesIO()
        picture.save(buffer, "JPEG", quality=90)
    return "image/jpeg", base64.standard_b64encode(buffer.getvalue()).decode()


def request(index: int, row: dict, image: tuple[str, str]) -> dict:
    text = PROMPT.format(question=row["question"], golds="; ".join(row["answers"]),
                         prediction=row["prediction"])
    return {"custom_id": f"audit-{index}", "params": {
        "model": AUDIT_MODEL, "max_tokens": 2000, "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": image[0],
                                         "data": image[1]}},
            {"type": "text", "text": text}]}]}}


def run_batch(client, requests: list[dict], poll_seconds: float = 30.0) -> dict[str, dict]:
    batch = client.messages.batches.create(requests=requests)
    print(f"audit: submitted batch {batch.id} ({len(requests)} requests)", flush=True)
    while client.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(poll_seconds)
    answers = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded" or result.result.message.stop_reason == "refusal":
            continue
        message = result.result.message
        text = next((block.text for block in message.content if block.type == "text"), "")
        answers[result.custom_id] = {"verdict": text.strip().lower().startswith("yes"),
                                     "input_tokens": message.usage.input_tokens,
                                     "output_tokens": message.usage.output_tokens}
    return answers


def summarize(sample: list[dict]) -> dict[str, dict]:
    """Per stratum: how often the auditor accepts, with a Wilson 95% interval."""
    rates = {}
    for name in sorted({row["stratum"] for row in sample}):
        rows = [row for row in sample if row["stratum"] == name and "auditor" in row]
        accepted = sum(row["auditor"] for row in rows)
        rates[name] = {"audited": len(rows), "auditor_accepts": accepted,
                       "rate": accepted / len(rows) if rows else None,
                       "interval": wilson(accepted, len(rows)),
                       "stratum_size": rows[0]["stratum_size"] if rows else 0}
    return rates


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return [centre - half, centre + half]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    judge = Judge(cache_path=ROOT / ".judge_cache" / "judge.db")
    sample = draw(candidates(judge))
    images = image_payloads(sample)
    requests = [request(i, row, images[row["dataset"], row["qid"]])
                for i, row in enumerate(sample)]
    # Upper bound at batch prices: a 1024-pixel image is at most ~1,400 tokens, plus the text,
    # and low effort keeps thinking and answer near 150 output tokens.
    bound = len(requests) * (1_700 * 4 + 300 * 20) / 1e6 / 2
    print(f"{len(requests)} audit items; cost at most about ${bound:.2f}")
    if args.dry_run:
        return
    answers = run_batch(judge.client, requests)
    for i, row in enumerate(sample):
        if f"audit-{i}" in answers:
            row["auditor"] = answers[f"audit-{i}"]["verdict"]
    tokens_in = sum(a["input_tokens"] for a in answers.values())
    tokens_out = sum(a["output_tokens"] for a in answers.values())
    report = {"auditor": "Opus 5.5, pinned as vqar.judge.AUDIT_MODEL", "effort": "low",
              "per_stratum": PER_STRATUM,
              "prompt": PROMPT, "cost_usd": (tokens_in * 4 + tokens_out * 20) / 1e6 / 2,
              "rates": summarize(sample),
              "items": [{key: value for key, value in row.items() if key != "answers"}
                        for row in sample]}
    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUTPUT}: {len(answers)} of {len(sample)} audited, ${report['cost_usd']:.2f}")


if __name__ == "__main__":
    main()
