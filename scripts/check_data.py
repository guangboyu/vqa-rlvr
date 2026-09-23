"""Create safe training views without overwriting data: python -m scripts.check_data."""

import argparse
import json
from itertools import combinations
from pathlib import Path

from datasets import load_from_disk

from vqar.config import SFT_PRESETS
from vqar.registry import DATASETS
from vqar.splits import VERSION, image_keys, qid_hash, retained_indices, training_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="write a separate manifest for a new view")
    args = parser.parse_args()
    source = json.loads((args.data_dir / "manifest.json").read_text())
    keys, entries = {}, {}
    for name in sorted(source["subsets"]):
        dataset = load_from_disk(str(args.data_dir / name))
        keys[name] = image_keys(dataset)
        entries[name] = dict(
            n=len(dataset),
            fingerprint=dataset._fingerprint,
            qids_sha256=qid_hash(dataset),
            image_keys=keys[name],
        )
        print(
            f"{name}: {len(dataset)} questions, {len(set(keys[name]))} unique RGB images",
            flush=True,
        )
    kept = retained_indices(keys)
    overlaps = {
        f"{a}/{b}": len(set(keys[a]) & set(keys[b]))
        for a, b in combinations(keys, 2)
        if set(keys[a]) & set(keys[b])
    }
    for name, indices in kept.items():
        entries[name]["retained_indices"] = indices
        print(f"{name}: retain {len(indices)}/{len(keys[name])}", flush=True)
    # The frozen dev split is carved from every registered SFT pool.
    sft_subsets = [spec.subsets["sft"] for spec in DATASETS.values()
                   if spec.subsets.get("sft") in keys]
    views = training_views(keys, kept, sft_subsets, seed=SFT_PRESETS["sft_2b"].seed)
    for name, parts in views["rl_views"].items():
        print(f"{name}: shared pool {sum(len(part['indices']) for part in parts)}", flush=True)
    result = dict(version=VERSION, subsets=entries, overlaps=overlaps, **views)
    path = args.output or args.data_dir / "split_manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    temporary.replace(path)
    print(f"overlapping images: {overlaps}\nmanifest: {path}")


if __name__ == "__main__":
    main()
