"""Parity check: the same model scored by lmms-eval and by this harness, question by question.

    python -m scripts.compare_lmms_eval --samples-dir <lmms-eval output dir> \
        --model-label base2 --output results/parity-lmms-eval.json

Pairs lmms-eval's logged GQA and TextVQA responses with this harness's short-prompt test
records of the same model, then splits any gap in accuracy into two parts: different
responses (generation) and different scoring of the same responses (scorer).
"""

import argparse
import json
from pathlib import Path

from vqar.eval.record import RunRecord, record_files
from vqar.eval.runner import RECORDS_DIR
from vqar.registry import score


def lmms_samples(samples_dir: Path, task: str) -> list[dict]:
    (path,) = sorted(samples_dir.rglob(f"*_samples_{task}.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def our_lines(label: str, dataset: str) -> list[dict]:
    for path in record_files(RECORDS_DIR):
        record = RunRecord.load(path)
        p = record.protocol
        if (record.model["label"], p["dataset"], p["role"], p["tier"], p["prompt"],
                p["image_mode"], p["max_pixels"]) == (label, dataset, "test", "full", "short",
                                                      "original", 786432):
            lines = [json.loads(x) for x in record.predictions_path().read_text().splitlines()]
            return lines
    raise FileNotFoundError(f"No full short-prompt {dataset} test record for {label}")


def pair(dataset: str, samples: list[dict], lines: list[dict]) -> list[tuple[dict, dict]]:
    """GQA samples follow the testdev order the harness also keeps; TextVQA carries its IDs."""
    if dataset == "textvqa":
        by_qid = {line["qid"]: line for line in lines}
        return [(s, by_qid[str(s["submission"]["question_id"])]) for s in samples]
    if len(samples) != len(lines):
        raise ValueError("GQA question counts differ")
    pairs = list(zip(samples, lines, strict=True))
    if any(not s["input"].startswith(line["question"]) for s, line in pairs):
        raise ValueError("GQA questions are not in the same order")
    return pairs


def compare(dataset: str, pairs: list[tuple[dict, dict]], lmms_metric: float) -> dict:
    n = len(pairs)
    same = sum(s["filtered_resps"].strip() == line["raw"].strip() for s, line in pairs)
    ours = {v: sum(line["scores"][v] for _, line in pairs) / n for v in ("official", "normalized")}
    # lmms-eval's responses under this harness's scorers: the gap left is generation only.
    rescored = {v: sum(score(dataset, s["filtered_resps"].strip(), line["answers"], v)
                       for s, line in pairs) / n for v in ("official", "normalized")}
    return {"dataset": dataset, "n": n, "lmms_eval_score": lmms_metric,
            "harness_official": ours["official"], "harness_normalized": ours["normalized"],
            "lmms_responses_under_harness_official": rescored["official"],
            "lmms_responses_under_harness_normalized": rescored["normalized"],
            "identical_responses": same}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="base2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (results,) = sorted(args.samples_dir.rglob("*_results.json"))
    metrics = json.loads(results.read_text())["results"]
    rows = []
    for task, dataset in (("gqa", "gqa"), ("textvqa_val", "textvqa")):
        pairs = pair(dataset, lmms_samples(args.samples_dir, task),
                     our_lines(args.model_label, dataset))
        rows.append(compare(dataset, pairs, metrics[task]["exact_match,none"]))
        print(json.dumps(rows[-1]))
    config = json.loads(results.read_text()).get("config", {})
    args.output.write_text(json.dumps({"lmms_eval": {"model_args": config.get("model_args")},
                                       "model": args.model_label, "comparisons": rows},
                                      indent=1) + "\n")


if __name__ == "__main__":
    main()
