"""Judge-vs-EM agreement study: how many rule-based misses are metric false negatives?

Samples EM-zero predictions (plus EM-positive controls) from selected runs, asks the
Haiku judge for semantic-equivalence verdicts via the Batches API, and reports the
rescue rate and judge-corrected accuracy per run. Costs pennies; capped at $10 overall.

Usage:
    uv run python scripts/judge_study.py --runs grpo_2b_main_sft_reasoning-vqav2 \
        grpo_2b_main_sft_reasoning-gqa --sample 400
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk

from vqar.judge import Judge

ROOT = Path(__file__).parent.parent


def study(run_id: str, sample: int, judge: Judge, rng: random.Random) -> dict:
    preds_text = (ROOT / "results/preds" / f"{run_id}.jsonl").read_text()
    preds = [json.loads(line) for line in preds_text.splitlines()]
    run = json.loads((ROOT / "results/runs" / f"{run_id}.json").read_text())
    subset = load_from_disk(str(ROOT / "data" / run["eval"]["subset"]))
    question_of = dict(zip(subset["qid"], subset["question"], strict=True))

    misses = [r for r in preds if r["score"] == 0 and r["prediction"].strip()]
    hits = [r for r in preds if r["score"] > 0]
    miss_sample = rng.sample(misses, min(sample, len(misses)))
    control_sample = rng.sample(hits, min(sample // 4, len(hits)))

    sampled = miss_sample + control_sample
    items = [(question_of[r["qid"]], r["answers"], r["prediction"]) for r in sampled]
    verdicts = judge.judge_many(items)
    miss_verdicts = verdicts[: len(miss_sample)]
    control_verdicts = verdicts[len(miss_sample) :]

    rescue_rate = sum(miss_verdicts) / len(miss_verdicts) if miss_verdicts else 0.0
    control_agreement = sum(control_verdicts) / len(control_verdicts) if control_verdicts else 1.0
    em_accuracy = run["metrics"]["overall"]
    miss_share = len(misses) / len(preds)
    corrected = em_accuracy + miss_share * rescue_rate
    return {
        "run_id": run_id,
        "em_accuracy": round(em_accuracy, 4),
        "sampled_misses": len(miss_sample),
        "judge_rescue_rate": round(rescue_rate, 4),
        "judge_corrected_accuracy": round(corrected, 4),
        "control_agreement_on_em_hits": round(control_agreement, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--sample", type=int, default=400)
    args = parser.parse_args()

    judge = Judge()
    rng = random.Random(0)
    results = [study(run_id, args.sample, judge, rng) for run_id in args.runs]
    out = ROOT / "results" / "judge_study.json"
    payload = {"results": results, "judge_spend_usd": round(judge.spend(), 4)}
    out.write_text(json.dumps(payload, indent=2))
    for r in results:
        print(r)
    print(f"total judge spend: ${judge.spend():.4f} -> {out}")


if __name__ == "__main__":
    main()
