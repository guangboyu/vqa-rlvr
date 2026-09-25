"""Summarize actual GRPO prompt exposure, reward variation, and training diagnostics.

Usage: python -m scripts.summarize_training runs/base runs/sft-rl --output exposure.json
This uses completed updates, dropping duplicate attempts after checkpoint resume.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from vqar.training import trained_draws, training_diagnostics


def completed_draws(path, steps, group_size):
    """The prompt groups each completed update trained on (kept groups, for refills)."""
    trained = {step: rows for step, rows in trained_draws(path).items() if step < steps}
    if sorted(trained) != list(range(steps)):
        raise ValueError(f"{path}: missing draws for completed updates")
    groups = []
    for step, record in sorted(trained.items()):
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


def same_data(recorded: dict, splits: dict[str, dict], split_sha: dict[str, str]) -> bool:
    """A run matches the current split files directly, or, for runs that predate them,
    through the manifest the files were migrated from with their roles still unedited."""
    if "manifest_sha256" not in recorded:
        return all(split_sha.get(name) == sha for name, sha in recorded["split_sha256"].items())
    from vqar.splits import roles_sha256

    migrated = [split for split in splits.values() if "migrated_from" in split]
    return bool(migrated) and all(
        split["migrated_from"]["sha256"] == recorded["manifest_sha256"]
        and split["migrated_from"]["roles_sha256"] == roles_sha256(split)
        for split in migrated
    )


def summarize_run(directory, sft_questions, splits, split_sha):
    config = json.loads((directory / "checkpoints/run_config.json").read_text())
    summary = json.loads((directory / "checkpoints/run_summary.json").read_text())
    if not summary["completed"]:
        raise ValueError(f"{directory}: training is not complete")
    if not same_data(config["data"], splits, split_sha):
        raise ValueError(f"{directory}: trained on different split files than --data-dir")
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
        "diagnostics": training_diagnostics(summary["logs"], path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from vqar.registry import DATASETS, datasets_with_role
    from vqar.splits import load_role, read_split, split_sha256

    splits = {name: read_split(args.data_dir, name) for name in DATASETS}
    split_sha = {name: split_sha256(args.data_dir, name) for name in DATASETS}
    train = load_role(args.data_dir, datasets_with_role("sft"), "sft", seed=None)
    sft_questions = set(zip(train["dataset"], train["qid"], strict=True))
    runs = {str(path): summarize_run(path, sft_questions, splits, split_sha)
            for path in args.runs}
    report = {
        "split_sha256": split_sha,
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
