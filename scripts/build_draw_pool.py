"""Declare the exact distinct questions retained by an existing RL training run.

    python -m scripts.build_draw_pool --name short2-actual4000 \
        --draws runs/part1-general-20260925/local/f-short2-base/prompt_draws.jsonl \
        --source-run-config checkpoints/merged/p1_f_short2_base/run_config.json \
        --expected-updates 1000 --expected-questions 4000

The pool retains draw order and original labels without copying images. SFT shuffles
the source roles with its usual seed and filters them to these questions; this Dataset
order precedes the Trainer's sampler. Question exposure matches, objectives/compute differ.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from vqar.pools import VERSION, load_pool, pool_path
from vqar.pools import provenance as pool_provenance
from vqar.splits import provenance as split_provenance
from vqar.splits import read_split
from vqar.training import ROOT, trained_draws


def ordered_sha256(records) -> str:
    """Canonical JSON lines: order matters, whitespace and dictionary insertion do not."""
    payload = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                      for record in records)
    return hashlib.sha256(payload.encode()).hexdigest()


def draw_keys(draws: Path, updates: int, questions: int,
              groups_per_update: int, generations: int) -> list[tuple[str, str]]:
    """Use the last attempt of a resumed update; reject refill or ambiguous grouping."""
    for line in draws.read_text().splitlines():
        record = json.loads(line)
        if "round" in record or "kept" in record:
            raise ValueError("Refill draws require a separate exposure design")
        if (type(record["step_before_update"]) is not int
                or record["step_before_update"] < 0):
            raise ValueError("Draw update IDs must be nonnegative integers")
        if not len(record["dataset"]) == len(record["qid"]) == len(record["scores"]):
            raise ValueError("Draw dataset, qid, and scores columns differ in length")
    attempts = trained_draws(draws)
    if list(attempts) != list(range(updates)):
        raise ValueError(f"Expected consecutive updates 0..{updates - 1}, got {len(attempts)}")
    keys = []
    for step, rows in attempts.items():
        counts = Counter(zip(rows["dataset"], rows["qid"], strict=True))
        if len(counts) != groups_per_update or set(counts.values()) != {generations}:
            raise ValueError(f"Update {step}: expected {groups_per_update} distinct groups "
                             f"of {generations} responses, got {dict(counts)}")
        keys.extend(counts)
    if len(keys) != questions or len(set(keys)) != questions:
        raise ValueError(f"Expected {questions} draws and distinct questions; "
                         f"got {len(keys)} draws / {len(set(keys))} distinct")
    return keys


def build_draw_pool(data_dir: Path, name: str, draws: Path, source_run_config: Path,
                    expected_updates: int, expected_questions: int,
                    groups_per_update: int = 4, generations: int = 8) -> dict:
    """Resolve identities through checked RL roles and retain row/label provenance."""
    source = json.loads(source_run_config.read_text())
    preset = source["preset"]
    datasets, seed = preset["datasets"], preset["seed"]
    if preset["max_steps"] != expected_updates or preset["num_generations"] != generations:
        raise ValueError("Expected draw geometry differs from the source recipe")
    current_data = split_provenance(data_dir, datasets)
    if source["data"] != current_data:
        raise ValueError("Current split files differ from the source run")
    source_pool = preset.get("pool")
    if source.get("pool") != pool_provenance(data_dir, source_pool):
        raise ValueError("Current source pool differs from the source run")
    keys = draw_keys(draws, expected_updates, expected_questions, groups_per_update, generations)
    wanted = set(keys)
    rows = load_pool(data_dir, datasets, seed, source_pool)
    columns = ["dataset", "qid", "source_subset", "source_row", "image_key", "answers"]
    identities = rows.select_columns(columns)
    all_keys = list(zip(identities["dataset"], identities["qid"], strict=True))
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("The source RL roles repeat question IDs")
    outside = wanted - set(all_keys)
    if outside:
        raise ValueError(f"Draw question is not in the source RL roles: {sorted(outside)[0]}")
    sft_coordinates = {
        (part["subset"], index)
        for dataset in datasets
        for part in read_split(data_dir, dataset)["roles"].get("sft", [])
        for index in part["indices"]
    }
    selected = identities.select([i for i, key in enumerate(all_keys) if key in wanted])
    by_key = {}
    for row in selected:
        if not row["answers"] or not isinstance(row["answers"][0], str):
            raise ValueError(f"{row['dataset']} {row['qid']} has no SFT answer target")
        row["in_full_sft"] = (row["source_subset"], row["source_row"]) in sft_coordinates
        by_key[row["dataset"], row["qid"]] = row
    in_sft = sum(row["in_full_sft"] for row in by_key.values())
    training_keys = [{"dataset": row["dataset"], "qid": row["qid"]} for row in selected]
    draw_order = [{"dataset": d, "qid": q} for d, q in keys]
    return {
        "version": VERSION, "name": name, "datasets": sorted(datasets),
        "source_data": current_data,
        "selection": {
            "keep": "all distinct retained questions, last attempt per update",
            "draws": {"path": str(draws), "sha256": hashlib.sha256(draws.read_bytes()).hexdigest(),
                      "ordered_questions_sha256": ordered_sha256(draw_order)},
            "source_run": {"path": str(source_run_config),
                           "sha256": hashlib.sha256(source_run_config.read_bytes()).hexdigest(),
                           "preset": preset, "policy_model": source["policy_model"]},
            "updates": expected_updates, "groups_per_update": groups_per_update,
            "responses_per_group": generations,
            "label_policy": "original source answers[0], as in the SFT recipe",
            "loaded_dataset_order": {
                "seed": seed, "rule": "seeded source-role shuffle, then member filter",
                "ordered_questions_sha256": ordered_sha256(training_keys)},
            "sft_sampler": "Trainer applies its seeded random sampler after loading; "
                           "the loaded Dataset order is not the optimizer-consumption order",
        },
        "exposure": {"source_rl_pool_questions": len(rows),
                     "full_sft_questions": len(sft_coordinates), "selected_questions": len(keys),
                     "selected_in_full_sft": in_sft, "selected_rl_only": len(keys) - in_sft,
                     "source_retained_responses": len(keys) * generations},
        "members": [by_key[key] for key in keys],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--draws", required=True, type=Path)
    parser.add_argument("--source-run-config", required=True, type=Path)
    parser.add_argument("--expected-updates", required=True, type=int)
    parser.add_argument("--expected-questions", required=True, type=int)
    parser.add_argument("--groups-per-update", default=4, type=int)
    parser.add_argument("--generations", default=8, type=int)
    parser.add_argument("--data-dir", default=ROOT / "data", type=Path)
    args = parser.parse_args()
    for key in ("expected_updates", "expected_questions", "groups_per_update", "generations"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    path = pool_path(args.data_dir, args.name)
    if path.exists():
        raise SystemExit(f"{path} exists; a declared pool is never rewritten")
    pool = build_draw_pool(args.data_dir, args.name, args.draws, args.source_run_config,
                           args.expected_updates, args.expected_questions,
                           args.groups_per_update, args.generations)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, indent=1) + "\n")
    print(json.dumps({"pool": str(path), **pool["exposure"],
                      "provenance": pool_provenance(args.data_dir, args.name)}, indent=2))


if __name__ == "__main__":
    main()
