"""RL pools: the questions an RL run draws its prompts from.

A run draws from its datasets' whole rl roles unless it names a declared pool: a subset of
those roles saved as data/pools/<name>.json together with how it was chosen. Loading a
pool checks every member against the rl roles, which the global held-out check already
keeps free of dev and test images, so a pool cannot leak held-out questions.

A run that refills dead groups (DAPO dynamic sampling) draws its replacement questions
from the same pool through a RefillStream.
"""

import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from vqar.splits import load_role
from vqar.splits import provenance as split_provenance

VERSION = "rl-pool-v1"
REFILL_VERSION = "refill-stream-v1"


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
    if not members or len(members) != len(declared["members"]):
        raise ValueError(f"{pool}: pool members must be nonempty and distinct")
    if ("source_data" in declared
            and declared["source_data"] != split_provenance(data_dir, datasets)):
        raise ValueError(f"{pool}: split files differ from its source training run")
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


def filtered_pool(name: str, records: list, low: int, high: int,
                  limit: int | None = None) -> dict:
    """A pool of the questions whose correct-sample count under the training reward lies in
    [low, high], taken from learnability records (one per dataset). `limit` keeps only the
    first kept questions in record order, e.g. a small set for an overfitting check."""
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
    selection = {"keep": f"{low} <= correct samples <= {high}", "records": sources}
    if limit is not None:
        members = members[:limit]
        selection["first"] = limit
    return {"version": VERSION, "name": name,
            "datasets": sorted({s["dataset"] for s in sources}),
            "selection": selection, "members": members}


def union_pool(name: str, data_dir: Path, pools: list[str], roles: list[str]) -> dict:
    """A pool of other declared pools' members plus whole rl roles, e.g. the questions two
    single-dataset runs drew, so one run can train on exactly their union."""
    members, parts = [], []
    for pool in pools:
        declared = read_pool(data_dir, pool)
        members += [{"dataset": m["dataset"], "qid": m["qid"]} for m in declared["members"]]
        parts.append({"pool": pool, "members": len(declared["members"]),
                      "sha256": hashlib.sha256(pool_path(data_dir, pool).read_bytes()).hexdigest()})
    for dataset in roles:
        qids = load_role(data_dir, [dataset], "rl", seed=None)["qid"]
        members += [{"dataset": dataset, "qid": qid} for qid in qids]
        parts.append({"role": f"{dataset} rl", "members": len(qids)})
    if len({(m["dataset"], m["qid"]) for m in members}) != len(members):
        raise ValueError(f"{name}: the parts overlap")
    datasets = sorted({m["dataset"] for m in members})
    return {"version": VERSION, "name": name, "datasets": datasets,
            "source_data": split_provenance(data_dir, datasets),
            "selection": {"union": parts}, "members": members}


@dataclass(frozen=True)
class RefillStream:
    """Replacement questions for dead groups, drawn from the run's own pool.

    Each update has its own stream: a generator seeded with the run seed and the update
    number visits the pool's rows in a fixed order, never twice in one update, and skips the
    questions the update already drew. It depends on nothing else, so a resumed update
    redraws the same replacements and checkpoints need no extra state. The stream is
    separate from the main sampler's order (a torch permutation of the same rows), so a
    refill does not pull the next updates' questions forward; a question can still appear
    both as a refill and later in the main order.
    """

    keys: tuple[tuple[str, str], ...]  # (dataset, qid) of each pool row, in the pool's order
    seed: int

    @classmethod
    def over(cls, rows, seed: int) -> "RefillStream":
        """The stream over a loaded pool; reads only its ID columns, never its images."""
        ids = rows.with_format(None).select_columns(["dataset", "qid"])
        return cls(tuple(zip(ids["dataset"], ids["qid"], strict=True)), seed)

    def draws(self, update: int, exclude: Iterable[tuple[str, str]] = ()) -> Iterator[int]:
        """Row indices of update `update`'s replacements, in draw order."""
        rng = random.Random(f"refill:{self.seed}:{update}")  # str seeds hash stably
        skip, visited = set(exclude), set()
        while len(visited) < len(self.keys):
            index = rng.randrange(len(self.keys))
            if index in visited:
                continue
            visited.add(index)
            if self.keys[index] not in skip:
                yield index

    def provenance(self) -> dict:
        """Recorded with a refilling run: enough to replay every update's draw order."""
        return {"version": REFILL_VERSION, "seed": self.seed, "pool_rows": len(self.keys),
                "order": "random.Random('refill:<seed>:<update>'), no repeats in an update"}
