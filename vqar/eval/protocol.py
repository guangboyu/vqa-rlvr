"""Evaluation protocols and their identity."""

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

from vqar import metrics
from vqar.config import MAX_PIXELS, RL_IMAGE_SIZE
from vqar.registry import DATASETS, PROMPTS, TRAINING_ROLES

# canvas shows the image as RL training does (vqar.data.to_uniform_canvas); first keeps only
# the first image of a multi-image question, none shows no image at all.
IMAGE_MODES = ("original", "black", "swapped", "canvas", "first", "none")
# Interventions change what the image shows; they pair with the original-image cell.
INTERVENTIONS = ("black", "swapped", "first", "none")
# Training roles are read only by the learnability pre-check and the screening battery.
EVAL_ROLES = ("dev", "test", "rl", "sft")
EXTRACTION_RULE = "last_answer_tag_or_full_text"
# The corrected campaign's rule; retention reports it beside mean output tokens.
ANSWER_TAG_RULE = "raw text contains </answer>"


@dataclass(frozen=True)
class Protocol:
    """What makes two numbers comparable. `max_tokens=None` takes the prompt's budget.

    `screening` lets a greedy protocol read a training role for the dataset screening
    battery (vqar.eval.screening). It changes what may be evaluated, never the numbers, so
    it is not part of the identity.
    """

    TIERS: ClassVar[dict[str, int | None]] = {"smoke": 32, "dev": 500, "full": None}

    dataset: str
    prompt: str = "short"
    role: str = "test"
    tier: str = "full"
    image_mode: str = "original"
    max_tokens: int | None = None
    temperature: float = 0.0  # 0 is greedy decoding
    samples: int = 1
    seed: int = 0  # used only when sampling
    max_pixels: int = MAX_PIXELS  # image budget the engine resizes to; the model's own is 16.8 MP
    screening: bool = False

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset: {self.dataset}")
        spec = DATASETS[self.dataset]
        if self.role not in EVAL_ROLES:
            raise ValueError(f"evaluation role must be one of {EVAL_ROLES}")
        if self.role in TRAINING_ROLES and self.samples < 2 and not self.screening:
            raise ValueError(f"the {self.role} role is read only by learnability protocols, "
                             "which sample a group of at least two responses, and by the "
                             "screening battery")
        if self.role not in spec.subsets:
            raise ValueError(f"{self.dataset} has no {self.role} role")
        if self.prompt not in PROMPTS:
            raise ValueError(f"unknown prompt template: {self.prompt}")
        self.tier_size(self.tier)  # rejects an unknown tier
        if self.image_mode not in IMAGE_MODES:
            raise ValueError(f"image mode must be one of {IMAGE_MODES}")
        if self.samples < 1 or (self.max_tokens is not None and self.max_tokens < 1):
            raise ValueError("samples and max_tokens must be positive")
        if self.max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        if self.image_mode == "canvas" and self.max_pixels < RL_IMAGE_SIZE**2:
            raise ValueError("the image budget must hold the RL canvas, or the engine "
                             "would shrink it")
        if self.temperature == 0 and self.samples != 1:
            raise ValueError("greedy decoding draws one sample; set a temperature to sample")

    @classmethod
    def tier_size(cls, tier: str) -> int | None:
        """Questions a tier takes from the role's order: a named tier, or "first-N"."""
        if tier in cls.TIERS:
            return cls.TIERS[tier]
        count = tier.removeprefix("first-")
        if tier.startswith("first-") and count.isdigit() and int(count) > 0:
            return int(count)
        raise ValueError(f"tier must be one of {tuple(cls.TIERS)} or first-N")

    @property
    def budget(self) -> int:
        return self.max_tokens or PROMPTS[self.prompt].max_tokens

    @property
    def sampled(self) -> bool:
        return self.temperature > 0

    def slug(self) -> str:
        return slug({"dataset": self.dataset, "role": self.role, "tier": self.tier,
                     "prompt": self.prompt, "image_mode": self.image_mode,
                     "temperature": self.temperature, "samples": self.samples,
                     "max_pixels": self.max_pixels})

    @staticmethod
    def id_of(identity: dict) -> str:
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def slug(identity: dict) -> str:
    """A readable file-name prefix for a record, e.g. vqav2-test-full-short-black."""
    parts = [identity["dataset"], identity["role"], identity["tier"], identity["prompt"]]
    if identity["image_mode"] != "original":
        parts.append(identity["image_mode"])
    if identity["temperature"] > 0:
        parts.append(f"t{identity['temperature']:g}k{identity['samples']}")
    if identity.get("max_pixels", MAX_PIXELS) != MAX_PIXELS:
        parts.append(f"px{identity['max_pixels']}")
    return "-".join(parts)


def scorer_version(scorer) -> str | None:
    if scorer is None:
        return None
    return f"{scorer.__name__}/v{metrics.SCORING_VERSION}"


def protocol_identity(protocol: Protocol, data: dict) -> dict:
    """The declared fields that define a protocol, plus the identity of the data selected."""
    spec = DATASETS[protocol.dataset]
    identity = {
        "dataset": protocol.dataset,
        "ladder": spec.ladder,  # frozen before training; OOD columns are never tuned on
        "role": protocol.role,
        "tier": protocol.tier,
        "question_limit": Protocol.tier_size(protocol.tier),
        "prompt": protocol.prompt,
        "prompt_text": PROMPTS[protocol.prompt].text,
        "max_tokens": protocol.budget,
        "temperature": protocol.temperature,
        "samples": protocol.samples,
        "image_mode": protocol.image_mode,
        "max_pixels": protocol.max_pixels,  # the image budget the generator resizes to
        "scorers": {"official": scorer_version(spec.official),
                    "normalized": scorer_version(spec.normalized)},
        "extraction": EXTRACTION_RULE,
        "answer_tag_rule": ANSWER_TAG_RULE,
        "data": data,
    }
    if spec.project is not None:
        identity["scorers"]["project"] = scorer_version(spec.project)
    if spec.metric_metadata is not None:
        identity["metric_metadata"] = dict(spec.metric_metadata)
    if protocol.sampled:
        identity["seed"] = protocol.seed
    if protocol.image_mode == "canvas":
        identity["canvas_size"] = RL_IMAGE_SIZE
    return identity
