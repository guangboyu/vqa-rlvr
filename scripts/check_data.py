"""Build or validate every dataset's split file, then check held-out images globally.

    python -m scripts.check_data

Existing split files are validated against the data on disk and never rewritten: each
subset must hold the rows its split file records (compared by content when the dataset
fingerprint differs). A registered dataset without a split file gets a new one that leaves
the others unchanged. The overlap report lands next to the split files.

`--migrate data/split_manifest.json` converted the single heldout-images-v2 manifest into
per-dataset files once, keeping every row of every role.
"""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_from_disk

from vqar.registry import DATASETS
from vqar.splits import (
    VERSION,
    build_split,
    overlap_report,
    read_split,
    roles_sha256,
    split_path,
    verify_subset,
)


def retained(entry: dict, name: str, keep=lambda key: True) -> dict:
    """The manifest's retained rows of one subset, optionally filtered by image key."""
    keys = entry["image_keys"]
    return {"subset": name, "indices": [i for i in entry["retained_indices"] if keep(keys[i])]}


def migrate(manifest: dict, manifest_sha256: str) -> dict[str, dict]:
    """Per-dataset split files from the heldout-images-v2 manifest, rows unchanged."""
    if manifest["version"] != "heldout-images-v2":
        raise ValueError("Only the heldout-images-v2 manifest can be migrated.")
    dev_keys = set(manifest["sft_dev"]["image_keys"])
    entries = manifest["subsets"]
    splits = {}
    for dataset, spec in DATASETS.items():
        subsets = {
            name: {key: entries[name][key]
                   for key in ("n", "fingerprint", "qids_sha256", "image_keys")}
            for name in dict.fromkeys(spec.subsets.values())
        }
        sft, rl, dev, test = (spec.subsets.get(role) for role in ("sft", "rl", "dev", "test"))
        # VQAv2 and GQA carve dev from their SFT pools; CLEVR's dev is its own val slice.
        carved = sft is not None and dev == sft
        roles = {}
        if sft:
            roles["sft"] = [retained(entries[sft], sft, lambda key: key not in dev_keys)]
        if rl:
            roles["rl"] = manifest["rl_views"][rl]
        if dev:
            roles["dev"] = [retained(entries[dev], dev, lambda key: key in dev_keys)
                            if carved else retained(entries[dev], dev)]
        if test:
            roles["test"] = [retained(entries[test], test)]
        split = {"version": VERSION, "dataset": dataset, "ladder": spec.ladder,
                 "official_splits": sorted(spec.official_splits), "sources": dict(spec.sources),
                 "subsets": subsets, "roles": roles}
        # Lets runs trained on the manifest be matched to these files, while any later
        # edit to the roles breaks the match.
        split["migrated_from"] = {"version": manifest["version"], "sha256": manifest_sha256,
                                  "roles_sha256": roles_sha256(split)}
        if carved:
            sft_dev = manifest["sft_dev"]
            split["dev_selection"] = {
                "seed": sft_dev["seed"], "target": sft_dev["target"],
                "method": "whole images drawn jointly from the VQAv2 and GQA SFT pools",
            }
        splits[dataset] = split
    return splits


def write_split(path: Path, split: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(split, separators=(",", ":")) + "\n")
    temporary.replace(path)


def check_subsets(data_dir: Path, split: dict) -> None:
    for name, entry in split["subsets"].items():
        verify_subset(data_dir, name, entry, load_from_disk(str(data_dir / name)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--migrate", type=Path, help="heldout-images-v2 manifest to convert")
    args = parser.parse_args()
    if args.migrate:
        raw = args.migrate.read_bytes()
        for dataset, split in migrate(json.loads(raw), hashlib.sha256(raw).hexdigest()).items():
            path = split_path(args.data_dir, dataset)
            if path.exists():
                raise SystemExit(f"{path} exists; migration never overwrites a split file")
            write_split(path, split)

    splits = {}
    for dataset in DATASETS:
        if split_path(args.data_dir, dataset).exists():
            splits[dataset] = read_split(args.data_dir, dataset)
            check_subsets(args.data_dir, splits[dataset])
    for dataset in DATASETS:
        if dataset not in splits:
            splits[dataset] = build_split(dataset, args.data_dir, list(splits.values()))
            write_split(split_path(args.data_dir, dataset), splits[dataset])
            print(f"{dataset}: new split file", flush=True)
    for dataset, split in splits.items():
        sizes = {role: sum(len(p["indices"]) for p in parts)
                 for role, parts in split["roles"].items()}
        print(f"{dataset} ({split['ladder']}): {sizes}", flush=True)
    report = overlap_report(list(splits.values()))
    path = args.data_dir / "splits" / "overlap-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"no held-out image in any training pool; report: {path}")


if __name__ == "__main__":
    main()
