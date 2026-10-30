"""The registries: the only lists of datasets and prompt templates in the project.

A dataset entry declares where the data comes from, which materialized subset plays each
role, how answers are scored, and where the dataset sits on the OOD ladder. Adding a dataset
means adding one entry here plus its loader in `vqar.data`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from vqar import metrics
from vqar.ood_metrics import official_vizwiz_accuracy, project_case_whitespace_em

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
    answer_rule: str = "tagged"  # how the RL reward reads its answer (vqar.answers)

    def __post_init__(self):
        from vqar.answers import ANSWER_RULES

        if self.answer_rule not in ANSWER_RULES:
            raise ValueError(f"{self.name}: unknown answer rule {self.answer_rule!r}")


_ANSWER_INSTRUCTION = (
    "Then write the final answer as a single word or phrase inside <answer> </answer> tags."
)

PROMPTS: dict[str, Prompt] = {
    p.name: p
    for p in [
        Prompt("short", "{question}\nAnswer the question using a single word or phrase.", 64,
               answer_rule="plain"),
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
    # role -> the materialized subset under data/ it draws from. A dev role drawn from the
    # SFT subset holds out whole images from it; the split files record the exact rows.
    subsets: Mapping[str, str]
    official_splits: frozenset[str]  # roles drawn from the dataset's own official split
    ladder: str  # position on the OOD ladder, frozen before training
    answer_schema: str
    official: Scorer | None  # None when no author evaluator has been verified
    normalized: Scorer  # unconditional normalization: the companion column and RL reward
    # Roles whose references the official scorer can score. VQAv2 and TextVQA training
    # examples keep one answer, but their official evaluators need all ten annotators.
    official_roles: frozenset[str]
    sizes: Mapping[str, int]  # materialization sizes and seeds, recorded in data/manifest.json
    materialize: Callable[[Mapping[str, int]], dict]  # sizes -> {subset name: Dataset}
    project: Scorer | None = None
    metric_metadata: Mapping | None = None
    require_split: bool = True  # optional OOD data is not required to run the tutorial
    max_images: int = 1  # images shown with one question; the engine must accept that many
    # False when the licence forbids publishing per-question results: records stay local
    # (.gitignore) and generated tables leave the dataset out, so they match a fresh checkout.
    public_records: bool = True

    def __post_init__(self):
        if unknown := set(self.subsets) - set(ROLES):
            raise ValueError(f"{self.name}: unknown roles {sorted(unknown)}")
        if self.ladder not in LADDER:
            raise ValueError(f"{self.name}: unknown ladder position {self.ladder!r}")
        if self.ladder != "in-domain" and TRAINING_ROLES & set(self.subsets):
            raise ValueError(f"{self.name}: an OOD dataset cannot have a training role")
        if not self.official_splits <= set(self.subsets):
            raise ValueError(f"{self.name}: official splits must name declared roles")
        if self.official is None and self.official_roles:
            raise ValueError(f"{self.name}: unavailable official scorer cannot declare roles")
        if self.project is not None and not self.metric_metadata:
            raise ValueError(f"{self.name}: a project scorer needs declared metric metadata")

    @property
    def scorer_variants(self) -> tuple[str, ...]:
        return SCORER_VARIANTS + (("project",) if self.project is not None else ())

    def scorer(self, variant: str) -> Scorer:
        if variant not in self.scorer_variants:
            raise ValueError(f"Unknown scorer variant: {variant}")
        scorer = {"official": self.official, "normalized": self.normalized,
                  "project": self.project}[variant]
        if scorer is None:
            raise ValueError(f"{self.name}: no verified {variant} scorer")
        return scorer


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


def _chartqa(sizes):
    from vqar import data

    return {f"chartqa_{name}": subset for name, subset in data.load_chartqa().items()}


def _clevrcomp(sizes):
    from vqar import data

    splits = data.load_clevr_compositional(sizes["train_shard"], sizes["test_shard"],
                                           sizes["per_image"], sizes["seed"])
    return {f"clevrcomp_{name}": subset for name, subset in splits.items()}


def _medthink(sizes):
    from vqar import data

    return {f"medthink_{name}": subset
            for name, subset in data.load_medthink(sizes["max_images"]).items()}


def _kvasirx1(sizes):
    from vqar import data

    splits = data.load_kvasir_x1(sizes["seed"], sizes["held_out_percent"] / 100)
    return {f"kvasirx1_{name}": subset for name, subset in splits.items()}


def _omnimedvqa(sizes):
    from vqar import data

    splits = data.load_omnimedvqa(sizes["seed"], sizes["held_out_percent"] / 100)
    return {f"omnimedvqa_{name}": subset for name, subset in splits.items()}


def _omnimedx(sizes):
    from vqar import data

    splits = data.load_omnimedx(sizes["seed"], sizes["test_percent"] / 100)
    return {f"omnimedx_{name}": subset for name, subset in splits.items()}


def _slake(sizes):
    from vqar import data

    return {f"slake_{name}": subset for name, subset in data.load_slake().items()}


def _pathvqa(sizes):
    from vqar import data

    return {f"pathvqa_{name}": subset for name, subset in data.load_pathvqa().items()}


def _rexvqa(sizes):
    from vqar import data

    return {f"rexvqa_{name}": subset for name, subset in data.load_rexvqa().items()}


def _deeptumor(sizes):
    from vqar import data

    splits = data.load_deeptumor(sizes["train"], sizes["dev"], sizes["dev_patient_percent"],
                                 sizes["seed"])
    return {f"deeptumor_{name}": subset for name, subset in splits.items()}


def _deeptumorx(sizes):
    from vqar import data

    splits = data.load_deeptumorx(sizes["train"], sizes["dev"], sizes["dev_patient_percent"],
                                  sizes["seed"], sizes["min_test_per_answer"],
                                  sizes["min_train_per_answer"])
    return {f"deeptumorx_{name}": subset for name, subset in splits.items()}


def _prepared_ood(sizes):
    raise ValueError("Use python -m scripts.prepare_ood with pinned local source files")


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
            subsets={"sft": "vqav2_sft", "rl": "vqav2_rl", "dev": "vqav2_sft",
                     "test": "vqav2_eval"},
            official_splits=frozenset({"sft", "rl", "dev", "test"}),
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
            subsets={"sft": "gqa_sft", "rl": "gqa_rl", "dev": "gqa_sft", "test": "gqa_eval"},
            official_splits=frozenset({"sft", "rl", "dev", "test"}),
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
            subsets={"rl": "clevr_rl", "dev": "clevr_val", "test": "clevr_test"},
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
            # The whole validation split (5,000 questions) in a seeded order; its first
            # 2,000 rows are the sample the corrected campaign evaluated.
            sizes={"eval": 5_000, "eval_seed": 0},
            materialize=_textvqa,
        ),
        # Reasoning candidates (Part 2), screened before any training. Optional like the OOD
        # cohorts: the tutorial does not need them and the headline tables leave them out.
        DatasetSpec(
            name="chartqa", display_name="ChartQA (human)",
            sources={"HuggingFaceM4/ChartQA": "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"},
            subsets={"sft": "chartqa_train", "rl": "chartqa_train", "dev": "chartqa_val",
                     "test": "chartqa_test"},
            official_splits=frozenset({"sft", "rl", "dev", "test"}),
            ladder="in-domain", answer_schema="number or short text read or computed from "
            "a chart; one human-written reference",
            official=metrics.chartqa_relaxed_accuracy,
            normalized=metrics.chartqa_normalized_accuracy,
            official_roles=frozenset(ROLES), sizes={}, materialize=_chartqa,
            require_split=False,
        ),
        DatasetSpec(
            name="clevrcomp", display_name="CLEVR compositional",
            sources={"HuggingFaceM4/the_cauldron": "847a98a779b1652d65111daf20c972dfcd333605"},
            subsets={"sft": "clevrcomp_train", "rl": "clevrcomp_train",
                     "dev": "clevrcomp_train", "test": "clevrcomp_test"},
            official_splits=frozenset(),  # re-split by image from CLEVR's training questions
            ladder="in-domain", answer_schema="yes/no, a count, or an attribute word",
            # CLEVR answers are single lowercase words; the raw string must match, as for
            # CLEVR counting. The normalized scorer forgives case and number words.
            official=metrics.exact_string_match, normalized=metrics.exact_match,
            official_roles=frozenset(ROLES),
            sizes={"train_shard": 0, "test_shard": 1, "per_image": 5, "seed": 0},
            materialize=_clevrcomp, require_split=False,
        ),
        # Medical candidate: multi-image radiology cases with five candidate diagnoses.
        # Evaluation shows up to eight images per case; training is not supported yet.
        DatasetSpec(
            name="medthink", display_name="MedThinkVQA",
            sources={"bio-nlp-umass/MedThinkVQA": "5555b80e167c4796efab92fe709fb05914077dc0"},
            subsets={"sft": "medthink_train", "rl": "medthink_train", "dev": "medthink_train",
                     "test": "medthink_test"},
            official_splits=frozenset({"sft", "rl", "test"}),  # dev carved from official train
            ladder="in-domain", answer_schema="option letter A-E; references hold the letter "
            "and the option text",
            official=metrics.choice_letter_accuracy, normalized=metrics.choice_accuracy,
            official_roles=frozenset(ROLES), sizes={"max_images": 8},
            materialize=_medthink, require_split=False, max_images=8,
        ),
        # Medical candidates (2026-10-10), screened for RL viability before any training.
        # Kvasir-VQA-x1's released train and test splits share most images, and OmniMedVQA has
        # no training split, so both are re-split here by image content (10% of images each
        # for test and dev, seed 0). Neither releases patient IDs: frames of one procedure or
        # slices of one scan can land on both sides, so held-out scores may run optimistic.
        DatasetSpec(
            name="kvasirx1", display_name="Kvasir-VQA-x1 (atomic)",
            sources={"SimulaMet/Kvasir-VQA-x1": "61e41148c3214bc5140ad0ab4c28520a512e2a73"},
            subsets={"sft": "kvasirx1_train", "rl": "kvasirx1_train", "dev": "kvasirx1_val",
                     "test": "kvasirx1_test"},
            official_splits=frozenset(),
            ladder="in-domain", answer_schema="label(s) from a closed vocabulary of about 60 "
            "GI-endoscopy terms; several values joined by '; '",
            official=metrics.label_set_match, normalized=metrics.normalized_label_set_match,
            official_roles=frozenset(ROLES), sizes={"seed": 0, "held_out_percent": 10},
            materialize=_kvasirx1, require_split=False,
        ),
        DatasetSpec(
            name="omnimedvqa", display_name="OmniMedVQA (open access)",
            sources={"foreverbeliever/OmniMedVQA": "1ba51c28fc0773bdf7efb8396e5bcfd4227e22da"},
            subsets={"sft": "omnimedvqa_train", "rl": "omnimedvqa_train",
                     "dev": "omnimedvqa_val", "test": "omnimedvqa_test"},
            official_splits=frozenset(),
            ladder="in-domain", answer_schema="option letter A-D (two to four options); "
            "references hold the letter and the option text",
            official=metrics.choice_letter_accuracy, normalized=metrics.choice_accuracy,
            official_roles=frozenset(ROLES), sizes={"seed": 0, "held_out_percent": 10},
            materialize=_omnimedvqa, require_split=False,
        ),
        # OmniMedVQA again, with whole source collections held out for test (2026-10-11): the
        # image-level re-split above lets SFT reach 99.7%, since test images come from the
        # collections it trained on. Test collections total about 10% of the questions,
        # drawn round-robin over imaging modalities with seed 0 (vqar.data.
        # held_out_collections); RadImageNet stays in training, and collections that share
        # images move together. Every open-access image already sits in one of omnimedvqa's
        # roles and no held-out image may enter any training pool, so the split file keeps
        # only the test rows omnimedvqa also tests on (about a tenth), training drops
        # omnimedvqa's dev and test images, and there is no dev role.
        DatasetSpec(
            name="omnimedx", display_name="OmniMedVQA (held-out collections)",
            sources={"foreverbeliever/OmniMedVQA": "1ba51c28fc0773bdf7efb8396e5bcfd4227e22da"},
            subsets={"sft": "omnimedx_train", "rl": "omnimedx_train", "test": "omnimedx_test"},
            official_splits=frozenset(),
            ladder="in-domain", answer_schema="option letter A-D (two to four options); "
            "references hold the letter and the option text",
            official=metrics.choice_letter_accuracy, normalized=metrics.choice_accuracy,
            official_roles=frozenset(ROLES), sizes={"seed": 0, "test_percent": 10},
            materialize=_omnimedx, require_split=False,
        ),
        DatasetSpec(
            name="slake", display_name="SLAKE (English)",
            sources={"BoKelvin/SLAKE": "a9083ce6c34ac3ffb17671a605962924d8a8f9e9"},
            subsets={"sft": "slake_train", "rl": "slake_train", "dev": "slake_validation",
                     "test": "slake_test"},
            official_splits=frozenset({"sft", "rl", "dev", "test"}),
            ladder="in-domain", answer_schema="yes/no or a short phrase; one reference",
            official=metrics.casefold_exact_match, normalized=metrics.exact_match,
            official_roles=frozenset(ROLES), sizes={}, materialize=_slake,
            require_split=False,
        ),
        # A negative control: published RL models score as well with blank or mismatched
        # PathVQA images, so its questions are answerable from text.
        DatasetSpec(
            name="pathvqa", display_name="PathVQA",
            sources={"flaviagiammarino/path-vqa": "1685832883334b5bb5beaf4e4b333fdeecaa4ad9"},
            subsets={"sft": "pathvqa_train", "rl": "pathvqa_train", "dev": "pathvqa_validation",
                     "test": "pathvqa_test"},
            official_splits=frozenset({"sft", "rl", "dev", "test"}),
            ladder="in-domain", answer_schema="yes/no (half) or an open phrase from a figure "
            "caption; one reference",
            official=metrics.casefold_exact_match, normalized=metrics.exact_match,
            official_roles=frozenset(ROLES), sizes={}, materialize=_pathvqa,
            require_split=False,
        ),
        # Chest X-rays (2026-10-11): ReXVQA's multiple-choice questions on ReXGradient-160K
        # studies, on the official patient-level splits. Only the first of the ten image
        # archive parts is used (about a tenth of each split's patients); each question shows
        # one frontal view of its study (the first PA image, else the first AP image; studies
        # with no frontal image are dropped), 16-bit PNGs stretched to 8 bits per image. The
        # licence forbids redistributing the data or any compilation of it: the split file,
        # pools and per-question records of this dataset stay out of git (.gitignore).
        DatasetSpec(
            name="rexvqa", display_name="ReXVQA (frontal view)",
            sources={"rajpurkarlab/ReXVQA": "b1adadb963bb16634f2e07c1c224a2e83fc8558a",
                     "rajpurkarlab/ReXGradient-160K":
                     "b9277dd929190931f5a8b6ae19f3e3c8aae2f1f7"},
            subsets={"sft": "rexvqa_train", "rl": "rexvqa_train", "dev": "rexvqa_valid",
                     "test": "rexvqa_test"},
            official_splits=frozenset({"sft", "rl", "dev", "test"}),
            ladder="in-domain", answer_schema="option letter A-D; references hold the letter "
            "and the option text",
            official=metrics.choice_letter_accuracy, normalized=metrics.choice_accuracy,
            official_roles=frozenset(ROLES), sizes={"archive_parts": 1},
            materialize=_rexvqa, require_split=False, public_records=False,
        ),
        # Abdominal CT (2026-10-29): DeepTumorVQA v2's multiple-choice questions (two to four
        # options; recognition, measurement, visual and medical reasoning). The released 2D
        # input for a training scan is one montage of five axial slices across the volume, so
        # every question here, test included, shows that whole-volume montage. Test is the
        # benchmark (991 patients); training draws 40,000 questions from the 428,050 released
        # ones after dropping the 68 benchmark patients that recur in training, questions with
        # several correct letters or a single option, and a 3% share of training patients that
        # becomes dev. Annotations CC BY 4.0; CT data from AbdomenAtlas 3.0 Mini, CC BY-NC-SA.
        DatasetSpec(
            name="deeptumor", display_name="DeepTumorVQA (whole-volume montage)",
            sources={"tumor-vqa/DeepTumorVQA_2.0": "a8b9db5133fda02567476dcb49498ec51f6aed37"},
            subsets={"sft": "deeptumor_train", "rl": "deeptumor_train", "dev": "deeptumor_dev",
                     "test": "deeptumor_test"},
            official_splits=frozenset({"test"}),
            ladder="in-domain", answer_schema="option letter A-D (two to four options); "
            "references hold the letter and the option text",
            official=metrics.choice_letter_accuracy, normalized=metrics.choice_accuracy,
            official_roles=frozenset(ROLES),
            sizes={"train": 40_000, "dev": 500, "dev_patient_percent": 3, "seed": 0},
            materialize=_deeptumor, require_split=False,
        ),
        # The same questions with their option shortcuts removed (2026-10-30; vqar.data
        # deeptumor_debias, deeptumor_balance): continuous values get options with the true value
        # at a uniformly random rank, counts a window of consecutive integers, every option list
        # is shuffled, and each subtype with discrete answers keeps equally many questions per
        # answer (answers with fewer than 10 test or 50 training questions dropped). Test keeps
        # 6,607 of the benchmark's questions; the montages are stored as released (grayscale).
        DatasetSpec(
            name="deeptumorx", display_name="DeepTumorVQA (debiased options)",
            sources={"tumor-vqa/DeepTumorVQA_2.0": "a8b9db5133fda02567476dcb49498ec51f6aed37"},
            subsets={"sft": "deeptumorx_train", "rl": "deeptumorx_train",
                     "dev": "deeptumorx_dev", "test": "deeptumorx_test"},
            official_splits=frozenset(),
            ladder="in-domain", answer_schema="option letter A-D (two to four options); "
            "references hold the letter and the option text",
            official=metrics.choice_letter_accuracy, normalized=metrics.choice_accuracy,
            official_roles=frozenset(ROLES),
            sizes={"train": 40_000, "dev": 500, "dev_patient_percent": 3, "seed": 0,
                   "min_test_per_answer": 10, "min_train_per_answer": 50},
            materialize=_deeptumorx, require_split=False,
        ),
        DatasetSpec(
            name="vizwiz", display_name="VizWiz",
            sources={"https://vizwiz.cs.colorado.edu/VizWiz_all_answers/VQA_test.json":
                     "fa113f0e90230ab4690bb41c4d74e40f7087cc7c192dd8363fa4d6bad339b7aa",
                     "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip":
                     "4d03f8c21823a3e4d1f501b1234e7380eb7cf2429ff0eb0281741ff1cc44f22a",
                     "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/API.zip":
                     "76de4354ca2fbc57dd2369b579dc9bb5684e41d6a135b36b324a5ba25b7b9ca2"},
            subsets={"test": "vizwiz_eval"}, official_splits=frozenset({"test"}),
            ladder="far-ood", answer_schema="open-ended phrase; ten raw annotator answers",
            official=official_vizwiz_accuracy, normalized=metrics.vqa_accuracy,
            official_roles=frozenset({"test"}), sizes={"eval": 8000},
            materialize=_prepared_ood, require_split=False,
            metric_metadata={"primary": "official", "official": {
                "name": "native VizWiz ten-annotator accuracy",
                "native_source_sha256":
                    "d26ef7ac2db06351444c3531f8d95fd9c454423fbcb9fd5dd984f12281146694",
                "rule": "normalize prediction with native contractions, compare raw references"}},
        ),
        DatasetSpec(
            name="realworldqa", display_name="RealWorldQA",
            sources={"xai-org/RealworldQA": "17e7f75e092e47169732462ea3cdfebe911105dd"},
            subsets={"test": "realworldqa_eval"}, official_splits=frozenset({"test"}),
            ladder="far-ood", answer_schema="original question/options; one raw answer",
            official=None, normalized=project_case_whitespace_em,
            official_roles=frozenset(), sizes={"eval": 765},
            materialize=_prepared_ood, project=project_case_whitespace_em,
            metric_metadata={
                "primary": "project", "project": {"name": "case/whitespace exact match",
                    "source": "project-defined; no native evaluator claimed",
                    "rule": "casefold and collapse whitespace; blank predictions fail; "
                            "no punctuation/article/number normalization or option guessing"},
                "normalized": {"same_as": "project", "name": "case/whitespace exact match"}},
            require_split=False,
        ),
    ]
}

SOURCE_REVISIONS = {
    repo: revision for spec in DATASETS.values() for repo, revision in spec.sources.items()
}


def score(dataset: str, prediction: str, answers: list[str], variant: str) -> float:
    """Score one prediction with the named scorer variant of its dataset."""
    return DATASETS[dataset].scorer(variant)(prediction, answers)


def check_trainable(datasets) -> None:
    """Training reads one image per question; refuse multi-image datasets until it reads more."""
    if multi := [name for name in datasets if DATASETS[name].max_images > 1]:
        raise ValueError(f"{multi}: training does not support multi-image questions yet")


def datasets_with_role(role: str, optional: bool = True) -> list[str]:
    """Datasets declaring the role, in registry order. `optional=False` keeps only the
    headline datasets every checkout prepares, leaving out OOD cohorts and candidates."""
    return [name for name, spec in DATASETS.items()
            if role in spec.subsets and (optional or spec.require_split)]
