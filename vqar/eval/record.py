"""Model identity and the RunRecord: one JSON per (model, protocol) plus a predictions sidecar."""

import functools
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from vqar.eval.protocol import Protocol

SCHEMA = "run-record/1"


@dataclass(frozen=True)
class ModelRef:
    """A model to evaluate: a local directory or a Hub id, plus optional training context."""

    path: str
    revision: str | None = None
    label: str | None = None  # names the records folder; defaults to the directory name
    run: str | None = None  # training run that produced it, when known
    step: int | None = None  # training step of that checkpoint, when known
    adapter: str | None = None  # LoRA adapter directory served on top of `path`

    def identity(self) -> dict:
        local = Path(self.path)
        if local.is_dir():
            weights = {"sha256": weights_sha256(local), "config_sha256": config_sha256(local)}
        else:
            weights = {"hub": self.path, "revision": hub_revision(self.path, self.revision)}
        if self.adapter:
            weights = {"base": weights, "adapter_sha256": adapter_sha256(Path(self.adapter))}
        identity = {"source": self.path, "label": self.label or local.name, "run": self.run,
                    "step": self.step, "weights": weights}
        if self.adapter:
            identity["adapter"] = self.adapter
        return identity

    def pinned(self) -> "ModelRef":
        """This model with a Hub revision resolved to its commit, so generation loads
        exactly the weights its records name."""
        weights = self.identity()["weights"]
        return replace(self, revision=weights.get("base", weights).get("revision"))


def weights_sha256(directory: Path) -> str:
    """One hash over every safetensors shard, so a re-merge or retrain is a new model."""
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {directory}")
    # Hashing gigabytes is slow; reuse the result while no shard has changed on disk.
    signature = tuple((str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns) for f in files)
    return _hash_shards(signature)


@functools.lru_cache(maxsize=32)
def _hash_shards(signature: tuple) -> str:
    files = [Path(name) for name, _, _ in signature]
    digest = hashlib.sha256()
    for file in files:
        shard = hashlib.sha256()
        with file.open("rb") as stream:
            while chunk := stream.read(1 << 24):
                shard.update(chunk)
        digest.update(f"{file.name}:{shard.hexdigest()}\n".encode())
    return digest.hexdigest()


# Files besides the weights that change what the model generates.
INFERENCE_FILES = ("config.json", "generation_config.json", "chat_template.jinja",
                   "chat_template.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                   "preprocessor_config.json", "processor_config.json")


def config_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in INFERENCE_FILES:
        if (directory / name).is_file():
            digest.update(f"{name}:{hashlib.sha256((directory / name).read_bytes()).hexdigest()}\n"
                          .encode())
    return digest.hexdigest()


def adapter_sha256(directory: Path) -> str:
    """A LoRA adapter's identity: its weights and its configuration (rank, targets)."""
    digest = hashlib.sha256()
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        digest.update(f"{name}:{hashlib.sha256((directory / name).read_bytes()).hexdigest()}\n"
                      .encode())
    return digest.hexdigest()


def hub_revision(repo_id: str, revision: str | None) -> str:
    """Resolve a Hub model to its commit, from the local cache when possible."""
    from huggingface_hub import HfApi, snapshot_download

    try:
        return Path(snapshot_download(repo_id, revision=revision, local_files_only=True)).name
    except Exception:  # not cached: ask the Hub
        return HfApi().model_info(repo_id, revision=revision).sha


def model_id(model: dict) -> str:
    return Protocol.id_of(model["weights"])


@dataclass(frozen=True)
class RunRecord:
    model: dict
    protocol: dict
    metrics: dict
    provenance: dict
    legacy: bool = False
    path: Path | None = field(default=None, compare=False)

    @property
    def model_id(self) -> str:
        return model_id(self.model)

    @property
    def protocol_id(self) -> str:
        return Protocol.id_of(self.protocol)

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "legacy": self.legacy, "model_id": self.model_id,
                "protocol_id": self.protocol_id, "model": self.model, "protocol": self.protocol,
                "metrics": self.metrics, "provenance": self.provenance}

    def predictions_path(self) -> Path:
        return self.path.with_name(self.path.stem + ".predictions.jsonl")

    def save(self, path: Path, predictions: list[dict]) -> "RunRecord":
        path.parent.mkdir(parents=True, exist_ok=True)
        record = RunRecord(self.model, self.protocol, self.metrics, self.provenance,
                           self.legacy, path)
        sidecar = record.predictions_path()
        temporary = sidecar.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n"
                                     for line in predictions))
        temporary.replace(sidecar)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
        return RunRecord.load(path)

    @classmethod
    def load(cls, path: Path) -> "RunRecord":
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a {SCHEMA} file")
        record = cls(raw["model"], raw["protocol"], raw["metrics"], raw["provenance"],
                     raw["legacy"], Path(path))
        if (record.model_id, record.protocol_id) != (raw["model_id"], raw["protocol_id"]):
            raise ValueError(f"{path} was edited: its identities no longer match its contents")
        return record


def record_path(records_dir: Path, model: dict, slug: str, identity: dict) -> Path:
    """Records live under <label>-<model id>/<protocol slug>-<protocol id>.json.

    An existing folder for the same model id is reused whatever its label, so a cache hit
    does not depend on how the model was named.
    """
    short_model = model_id(model)[:10]
    folders = sorted(records_dir.glob(f"*-{short_model}")) if records_dir.exists() else []
    folder = folders[0] if folders else records_dir / f"{model['label']}-{short_model}"
    return folder / f"{slug}-{Protocol.id_of(identity)[:10]}.json"
