"""The experiment matrix: arms × protocols declared in experiments.toml, and its cells.

A cell is present when a record under the arm's label carries exactly the protocol
identity the matrix declares today. Records are also checked against the arm's declared
model, so a mislabeled record cannot fill another arm's row.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from vqar.eval.examples import select_rows
from vqar.eval.protocol import Protocol, protocol_identity
from vqar.eval.record import ModelRef, RunRecord, model_id, record_files


@dataclass(frozen=True)
class Arm:
    label: str
    name: str
    model: str
    revision: str | None = None
    gpu_memory_utilization: float = 0.8

    def available(self, root: Path) -> bool:
        """Hub models always are; a local merge only while it is on disk."""
        return self.revision is not None or (root / self.model).is_dir()

    def ref(self, root: Path) -> ModelRef:
        path = self.model if self.revision is not None else str(root / self.model)
        return ModelRef(path, self.revision, self.label)


def load_matrix(path: Path) -> tuple[list[Arm], list[Protocol]]:
    """The declared arms and the protocols every arm is evaluated under."""
    matrix = tomllib.loads(Path(path).read_text())
    protocols = [
        Protocol(dataset, prompt=prompt, role=entry.get("role", "test"),
                 tier=entry.get("tier", "full"), image_mode=image_mode)
        for entry in matrix.get("protocols", [])
        for dataset in entry["datasets"]
        for prompt in entry["prompts"]
        for image_mode in entry.get("image_modes", ["original"])
    ]
    return [Arm(**arm) for arm in matrix["arms"]], protocols


def missing_cells(arms: list[Arm], protocols: list[Protocol], records_dir: Path,
                  data_dir: Path, root: Path) -> list[tuple[Arm, Protocol]]:
    """Every declared (arm, protocol) cell without a record; raises on a mislabeled record."""
    records = [RunRecord.load(path) for path in record_files(records_dir)]
    expected = {protocol: Protocol.id_of(protocol_identity(protocol,
                                                           select_rows(protocol, data_dir)[1]))
                for protocol in protocols}
    missing = []
    for arm in arms:
        mine = [r for r in records if r.model["label"] == arm.label and r.model["step"] is None]
        if mine and arm.available(root):
            declared = model_id(arm.ref(root).identity())
            strangers = [str(r.path) for r in mine if r.model_id != declared]
            if strangers:
                raise ValueError(f"{arm.label}: records of another model: {strangers}")
        present = {r.protocol_id for r in mine}
        missing += [(arm, p) for p in protocols if expected[p] not in present]
    return missing
