"""Summarize actual GRPO prompt exposure and reward variation from saved draws.

Usage: python -m scripts.summarize_training runs/base runs/sft-rl --output exposure.json
This uses completed updates, dropping duplicate attempts after checkpoint resume.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def completed_draws(path, steps, group_size):
    latest = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        step = record["step_before_update"]
        if step < steps:
            latest[step] = record
    if sorted(latest) != list(range(steps)):
        raise ValueError(f"{path}: missing draws for completed updates")
    groups = []
    for step, record in sorted(latest.items()):
        rows = list(zip(record["dataset"], record["qid"], record["scores"], strict=True))
        if len(rows) % group_size:
            raise ValueError(f"{path}: incomplete rollout group at step {step}")
        for start in range(0, len(rows), group_size):
            group = rows[start:start + group_size]
            identities = {(dataset, qid) for dataset, qid, _ in group}
            if len(identities) != 1:
                raise ValueError(f"{path}: mixed question IDs in one rollout group")
            dataset, qid = next(iter(identities))
            groups.append((step, dataset, qid, [score for _, _, score in group]))
    return groups


def group_summary(groups):
    scores = [score for _, _, _, values in groups for score in values]
    return {
        "prompt_groups": len(groups),
        "unique_questions": len({(dataset, qid) for _, dataset, qid, _ in groups}),
        "responses": len(scores),
        "mean_reward": sum(scores) / len(scores) if scores else None,
        "all_zero_reward_groups": sum(all(score == 0 for score in v) for _, _, _, v in groups),
        "all_full_reward_groups": sum(all(score == 1 for score in v) for _, _, _, v in groups),
        "variable_reward_fraction": (
            sum(max(v) - min(v) > 1e-12 for _, _, _, v in groups) / len(groups)
            if groups else None
        ),
    }


def summarize_run(directory, sft_questions, manifest_sha):
    config = json.loads((directory / "checkpoints/run_config.json").read_text())
    summary = json.loads((directory / "checkpoints/run_summary.json").read_text())
    if not summary["completed"]:
        raise ValueError(f"{directory}: training is not complete")
    if config["data"]["manifest_sha256"] != manifest_sha:
        raise ValueError(f"{directory}: split manifest differs from --data-dir")
    path = directory / "prompt_draws.jsonl"
    groups = completed_draws(path, summary["steps"], config["preset"]["num_generations"])
    buckets = defaultdict(list)
    for group in groups:
        _, dataset, qid, _ = group
        membership = "sft_train" if (dataset, qid) in sft_questions else "rl_only"
        buckets[f"{dataset}/{membership}"].append(group)
    order = [(step, dataset, qid) for step, dataset, qid, _ in groups]
    return {
        "completed_updates": summary["steps"],
        "updates_without_reward_variation": summary["steps"] - len({
            step for step, _, _, scores in groups if max(scores) - min(scores) > 1e-12
        }),
        "prompt_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        "draw_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "overall": group_summary(groups),
        "by_training_membership": {key: group_summary(value) for key, value in buckets.items()},
        "gpu": summary.get("gpu"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from vqar.splits import load_training_subsets, split_dev

    manifest_path = args.data_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sft = load_training_subsets(args.data_dir, manifest["sft_dev"]["subsets"],
                                manifest["sft_dev"]["seed"])
    train, _ = split_dev(sft)
    sft_questions = set(zip(train["dataset"], train["qid"], strict=True))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runs = {str(path): summarize_run(path, sft_questions, manifest_sha) for path in args.runs}
    report = {
        "split_manifest_sha256": manifest_sha,
        "runs": runs,
        "same_prompt_order": (
            len({r["prompt_order_sha256"] for r in runs.values()}) == 1 if len(runs) > 1 else None
        ),
        "interpretation": (
            "Membership describes the SFT training set; the base-to-RL arm did not receive SFT. "
            "Reward variation describes sampled groups, not held-out accuracy. "
            "Zero rewards can reflect incorrect answers or missing required answer tags."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
