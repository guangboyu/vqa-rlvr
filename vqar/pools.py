"""RL pools: the questions an RL run draws its prompts from.

A run draws from its datasets' whole rl roles unless it names a declared pool: a subset of
those roles saved as data/pools/<name>.json together with how it was chosen. Loading a
pool checks every member against the rl roles, which the global held-out check already
keeps free of dev and test images, so a pool cannot leak held-out questions.
"""

import hashlib
import json
from pathlib import Path

from vqar.splits import load_role

VERSION = "rl-pool-v1"


def pool_path(data_dir: Path, name: str) -> Path:
    return Path(data_dir) / "pools" / f"{name}.json"


def read_pool(data_dir: Path, name: str) -> dict:
    pool = json.loads(pool_path(data_dir, name).read_text())
    if pool.get("version") != VERSION:
        raise ValueError(f"{name}: unsupported pool file {pool.get('version')}")
    return pool


def load_pool(data_dir: Path, datasets, seed: int, pool: str | None = None):
    """The rl-role rows a run draws from, shuffled with `seed`; a named pool keeps only its
    members, in that same order."""
    rows = load_role(data_dir, datasets, "rl", seed)
    if pool is None:
        return rows
    declared = read_pool(data_dir, pool)
    if sorted(declared["datasets"]) != sorted(datasets):
        raise ValueError(f"{pool} was selected from {declared['datasets']}, not {list(datasets)}")
    members = {(m["dataset"], m["qid"]) for m in declared["members"]}
    keys = list(zip(rows["dataset"], rows["qid"], strict=True))
    if len(set(keys)) != len(keys):
        raise ValueError("The rl roles repeat a question ID; pool members would be ambiguous.")
    outside = members - set(keys)
    if outside:
        raise ValueError(f"{len(outside)} members of {pool} are not in the rl roles, "
                         f"e.g. {sorted(outside)[0]}")
    return rows.select([i for i, key in enumerate(keys) if key in members])


def provenance(data_dir: Path, pool: str | None) -> dict | None:
    """Recorded with every run that trains on a pool: which file defined it."""
    if pool is None:
        return None
    path = pool_path(data_dir, pool)
    return {"name": pool, "version": VERSION,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def filtered_pool(name: str, records: list, low: int, high: int) -> dict:
    """A pool of the questions whose correct-sample count under the training reward lies in
    [low, high], taken from learnability records (one per dataset)."""
    members, sources = [], []
    for record in records:
        learnability = record.metrics["learnability"]
        dataset = record.protocol["dataset"]
        members += [{"dataset": dataset, "qid": qid, "correct_samples": hits}
                    for qid, hits in learnability["correct_samples"].items()
                    if low <= hits <= high]
        sources.append({"dataset": dataset, "record": str(record.path),
                        "model": record.model, "protocol_id": record.protocol_id,
                        "tier": record.protocol["tier"],
                        "samples_per_question": record.metrics["samples_per_question"],
                        "scored": record.metrics["n"], "rule": learnability["rule"]})
    return {"version": VERSION, "name": name,
            "datasets": sorted({s["dataset"] for s in sources}),
            "selection": {"keep": f"{low} <= correct samples <= {high}", "records": sources},
            "members": members}
