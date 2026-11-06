"""Reasoning traces for SFT: one correct sampled response per question.

    python -m scripts.learnability --label base2 --datasets chartqa --tier full \
        --records-dir runs/cards/traces/records
    python -m scripts.build_traces RECORD.json data/traces/chartqa-base2.jsonl

The source is a released model's own samples under the reasoning prompt, scored by the same reward
GRPO uses, so the targets hold the information RL has: the model's samples and the checker.
Questions with no correct sample are left out.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from vqar.eval.record import RunRecord
from vqar.rewards import correctness_reward

SEED = 42


def correct_samples(lines: list[dict], dataset: str, prompt: str) -> dict[str, list[str]]:
    """question id -> its sampled responses the training reward marks right."""
    rewards = correctness_reward(prompts=None, completions=[line["raw"] for line in lines],
                                 answers=[line["answers"] for line in lines],
                                 dataset=[dataset] * len(lines),
                                 rollout_prompt=[prompt] * len(lines))
    correct = defaultdict(list)
    for line, reward in zip(lines, rewards, strict=True):
        correct[line["qid"]] += [line["raw"]] if reward == 1 else []
    return dict(correct)


def choose_traces(correct: dict[str, list[str]], seed: int = SEED) -> dict[str, str]:
    """One response per question that has any right, seeded and independent of file order."""
    rng = random.Random(seed)
    return {qid: rng.choice(correct[qid]) for qid in sorted(correct) if correct[qid]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", type=Path, help="a sampled record of the rl role")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = RunRecord.load(args.record)
    protocol = record.protocol
    if protocol["role"] != "rl" or protocol["samples"] < 2:
        raise SystemExit("Traces come from a sampled record of the rl role.")
    correct = correct_samples(record.predictions(), protocol["dataset"], protocol["prompt"])
    traces = choose_traces(correct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps({"qid": qid, "target": target}) + "\n"
                                   for qid, target in traces.items()))
    questions = record.metrics["n"]
    print(f"{len(traces)} of {questions} questions have a correct sample "
          f"({100 * len(traces) / questions:.1f}%) -> {args.output}")


if __name__ == "__main__":
    main()
