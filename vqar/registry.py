"""The registries: the only lists of datasets and prompt templates in the project.

A dataset entry declares where the data comes from, which materialized subset plays each
role, how answers are scored, and where the dataset sits on the OOD ladder. Adding a dataset
means adding one entry here plus its loader in `vqar.data`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from vqar import metrics

ROLES = ("sft", "rl", "dev", "test")
TRAINING_ROLES = frozenset({"sft", "rl"})
LADDER = ("in-domain", "near-ood", "far-ood")
SCORER_VARIANTS = ("official", "normalized")

Scorer = Callable[[str, list[str]], float]


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str  # formatted with question=...
    max_tokens: int  # default generation budget for this template
    diagnostic: bool = False  # only for controlled prompt studies, never in headline tables


_ANSWER_INSTRUCTION = (
    "Then write the final answer as a single word or phrase inside <answer> </answer> tags."
)

PROMPTS: dict[str, Prompt] = {
    p.name: p
    for p in [
        Prompt("short", "{question}\nAnswer the question using a single word or phrase.", 64),
        Prompt(
            "reasoning",
            "{question}\n"
            "First think step by step inside <think> </think> tags, then write only the final "
            "answer inside <answer> </answer> tags.",
            512,
        ),
        Prompt(
            "direct",
            "{question}\nOutput an empty <think></think> block. " + _ANSWER_INSTRUCTION,
            512,
            diagnostic=True,
        ),
        Prompt(
            "brief",
            "{question}\nGive at most one sentence of reasoning inside <think> </think> tags. "
            + _ANSWER_INSTRUCTION,
            512,
            diagnostic=True,
        ),
        Prompt(
            "stepwise",
            "{question}\nThink step by step inside <think> </think> tags. " + _ANSWER_INSTRUCTION,
            512,
            diagnostic=True,
        ),
    ]
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    sources: Mapping[str, str]  # Hub repository -> pinned revision
    subsets: Mapping[str, str]  # role -> materialized subset under data/
    official_splits: frozenset[str]  # roles drawn from the dataset's own official split
    ladder: str  # position on the OOD ladder, frozen before training
    answer_schema: str
    official: Scorer  # the dataset authors' evaluator: the primary column
    normalized: Scorer  # unconditional normalization: the companion column and RL reward
    # Roles whose references the official scorer can score. VQAv2 and TextVQA training
    # examples keep one answer, but their official evaluators need all ten annotators.
    official_roles: frozenset[str]
    sizes: Mapping[str, int]  # materialization sizes and seeds, recorded in data/manifest.json
    materialize: Callable[[Mapping[str, int]], dict]  # sizes -> {subset name: Dataset}

    def __post_init__(self):
        if unknown := set(self.subsets) - set(ROLES):
            raise ValueError(f"{self.name}: unknown roles {sorted(unknown)}")
        if self.ladder not in LADDER:
            raise ValueError(f"{self.name}: unknown ladder position {self.ladder!r}")
        if self.ladder != "in-domain" and TRAINING_ROLES & set(self.subsets):
            raise ValueError(f"{self.name}: an OOD dataset cannot have a training role")
        if not self.official_splits <= set(self.subsets):
            raise ValueError(f"{self.name}: official splits must name declared roles")

    def scorer(self, variant: str) -> Scorer:
        if variant not in SCORER_VARIANTS:
            raise ValueError(f"Unknown scorer variant: {variant}")
        return self.official if variant == "official" else self.normalized


def _vqav2(sizes):
    from vqar import data

    sft, rl = data.load_vqav2_train(sizes["sft"], sizes["rl"], sizes["train_seed"])
    return {"vqav2_sft": sft, "vqav2_rl": rl,
            "vqav2_eval": data.load_vqav2_eval(sizes["eval"], sizes["eval_seed"])}


def _gqa(sizes):
    from vqar import data

    sft, rl = data.load_gqa_train(sizes["sft"], sizes["rl"], sizes["train_seed"])
    return {"gqa_sft": sft, "gqa_rl": rl, "gqa_eval": data.load_gqa_eval()}


def _clevr(sizes):
    from vqar import data

    splits = data.load_clevr(sizes["rl"], sizes["val"], sizes["test"], sizes["seed"])
    return {f"clevr_{name}": subset for name, subset in splits.items()}


def _textvqa(sizes):
    from vqar import data

    return {"textvqa_eval": data.load_textvqa_eval(sizes["eval"], sizes["eval_seed"])}


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        DatasetSpec(
            name="vqav2",
            display_name="VQAv2",
            sources={
                "HuggingFaceM4/the_cauldron": "847a98a779b1652d65111daf20c972dfcd333605",
                "lmms-lab/VQAv2": "32665d35052eb4a6d4414851c3c829a72754915a",
            },
            subsets={"sft": "vqav2_sft", "rl": "vqav2_rl", "test": "vqav2_eval"},
            official_splits=frozenset({"sft", "rl", "test"}),
            ladder="in-domain",
            answer_schema="open-ended phrase; ten annotator answers in test, one in training",
            official=metrics.official_vqa_accuracy,
            normalized=metrics.vqa_accuracy,
            official_roles=frozenset({"test"}),
            sizes={"sft": 20_000, "rl": 8_000, "eval": 5_000, "train_seed": 42, "eval_seed": 0},
            materialize=_vqav2,
        ),
        DatasetSpec(
            name="gqa",
            display_name="GQA",
            sources={"lmms-lab/GQA": "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"},
            subsets={"sft": "gqa_sft", "rl": "gqa_rl", "test": "gqa_eval"},
            official_splits=frozenset({"sft", "rl", "test"}),
            ladder="in-domain",
            answer_schema="open-ended phrase; one answer",
            official=metrics.exact_string_match,
            normalized=metrics.exact_match,
            official_roles=frozenset(ROLES),
            # The test role is the full official testdev_balanced split.
            sizes={"sft": 20_000, "rl": 8_000, "train_seed": 42},
            materialize=_gqa,
        ),
        DatasetSpec(
            name="clevr",
            display_name="CLEVR counting",
            sources={"leonardPKU/clevr_cogen_a_train": "d5a8da66aa97c9c19ebe4075b7c7f777fef850a6"},
            subsets={"rl": "clevr_rl", "test": "clevr_test"},
            official_splits=frozenset(),  # slices of one shuffled source split
            ladder="in-domain",
            answer_schema="integer count",
            official=metrics.exact_string_match,
            normalized=metrics.exact_match,
            official_roles=frozenset(ROLES),
            sizes={"rl": 8_000, "val": 3_000, "test": 3_000, "seed": 42},
            materialize=_clevr,
        ),
        DatasetSpec(
            name="textvqa",
            display_name="TextVQA",
            sources={"lmms-lab/textvqa": "9c0699cd19768ac5ab97568f6b3cbac4c0062884"},
            subsets={"test": "textvqa_eval"},
            official_splits=frozenset({"test"}),
            ladder="far-ood",
            answer_schema="open-ended text read from the image; ten annotator answers",
            official=metrics.textvqa_accuracy,
            normalized=metrics.vqa_accuracy,
            official_roles=frozenset({"test"}),
            sizes={"eval": 2_000, "eval_seed": 0},
            materialize=_textvqa,
        ),
    ]
}

SOURCE_REVISIONS = {
    repo: revision for spec in DATASETS.values() for repo, revision in spec.sources.items()
}


def score(dataset: str, prediction: str, answers: list[str], variant: str) -> float:
    """Score one prediction with the named scorer variant of its dataset."""
    return DATASETS[dataset].scorer(variant)(prediction, answers)


def datasets_with_role(role: str) -> list[str]:
    return [name for name, spec in DATASETS.items() if role in spec.subsets]


def pools(role: str, *names: str) -> tuple[str, ...]:
    """The materialized subsets playing `role` for the named datasets, e.g. a recipe's RL pools."""
    missing = [name for name in names if role not in DATASETS[name].subsets]
    if missing:
        raise ValueError(f"{', '.join(missing)} has no {role} role")
    return tuple(DATASETS[name].subsets[role] for name in names)
