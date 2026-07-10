"""Dataset loaders: four sources → one unified schema, materialized by prepare_data.py.

Unified example: {dataset, qid, image, question, answers: list[str], answer_type}.
VQAv2 eval keeps all 10 annotator answers (required by the official metric);
single-answer datasets store a one-element list.

Train subsets for SFT and RL are sampled in a single deterministic pass so they are
image-disjoint by construction. Eval splits are officially disjoint from train
(VQAv2: train2014 vs val2014 images; GQA: train_balanced vs testdev_balanced;
CLEVR: held-out shards of the R1-V set never used for training).
"""

from collections.abc import Iterable, Iterator

from datasets import Dataset, Features, Image, Sequence, Value, load_dataset

from vqar.inference import extract_answer

PROMPT_SHORT = "{question}\nAnswer the question using a single word or phrase."
PROMPT_REASONING = (
    "{question}\n"
    "First think step by step inside <think> </think> tags, then write only the final "
    "answer inside <answer> </answer> tags."
)

FEATURES = Features(
    {
        "dataset": Value("string"),
        "qid": Value("string"),
        "image": Image(),
        "question": Value("string"),
        "answers": Sequence(Value("string")),
        "answer_type": Value("string"),
    }
)


def _dataset_from(generator: Iterable[dict]) -> Dataset:
    return Dataset.from_generator(lambda: iter(generator), features=FEATURES)


def _clean_cauldron_answer(text: str) -> str:
    """Cauldron reformatted VQAv2 answers as 'Net.' — restore the dataset's lowercase form."""
    return text.strip().removesuffix(".").lower()


def load_vqav2_train(n_sft: int, n_rl: int, seed: int) -> tuple[Dataset, Dataset]:
    """Flatten the_cauldron/vqav2 (multi-QA per image) into SFT and RL QA subsets.

    Walks shuffled image-rows, filling the SFT quota first and the RL quota second,
    so the two subsets never share an image.
    """
    rows = load_dataset("HuggingFaceM4/the_cauldron", "vqav2", split="train").shuffle(seed=seed)

    def emit() -> Iterator[tuple[str, dict]]:
        sft = rl = 0
        for i, row in enumerate(rows):
            bucket = "sft" if sft < n_sft else "rl"
            for j, qa in enumerate(row["texts"]):
                if bucket == "sft" and sft >= n_sft:
                    break  # don't split one image across buckets
                if bucket == "rl" and rl >= n_rl:
                    return
                question = qa["user"].split("\n")[0].strip()
                example = {
                    "dataset": "vqav2",
                    "qid": f"cauldron-{i}-{j}",
                    "image": row["images"][0],
                    "question": question,
                    "answers": [_clean_cauldron_answer(qa["assistant"])],
                    "answer_type": "",
                }
                yield bucket, example
                if bucket == "sft":
                    sft += 1
                else:
                    rl += 1

    pairs = list(emit())
    sft_ds = _dataset_from([ex for b, ex in pairs if b == "sft"])
    rl_ds = _dataset_from([ex for b, ex in pairs if b == "rl"])
    return sft_ds, rl_ds


def load_vqav2_eval(n: int, seed: int) -> Dataset:
    rows = load_dataset("lmms-lab/VQAv2", split="validation").shuffle(seed=seed).select(range(n))

    def emit() -> Iterator[dict]:
        for row in rows:
            yield {
                "dataset": "vqav2",
                "qid": str(row["question_id"]),
                "image": row["image"],
                "question": row["question"],
                "answers": [a["answer"] for a in row["answers"]],
                "answer_type": row["answer_type"],
            }

    return _dataset_from(emit())


def _gqa_examples(instructions: Iterable[dict], images: Dataset) -> Iterator[dict]:
    image_index = {img_id: i for i, img_id in enumerate(images["id"])}
    for row in instructions:
        yield {
            "dataset": "gqa",
            "qid": row["id"],
            "image": images[image_index[row["imageId"]]]["image"],
            "question": row["question"],
            "answers": [row["answer"]],
            "answer_type": row["types"]["structural"],
        }


def load_gqa_train(n_sft: int, n_rl: int, seed: int) -> tuple[Dataset, Dataset]:
    """Sample GQA balanced-train QAs: SFT quota first, then RL from unused images only."""
    instructions = load_dataset(
        "lmms-lab/GQA", "train_balanced_instructions", split="train"
    ).shuffle(seed=seed)
    images = load_dataset("lmms-lab/GQA", "train_balanced_images", split="train")

    sft_rows, rl_rows, sft_images = [], [], set()
    for row in instructions:
        if len(sft_rows) < n_sft:
            sft_rows.append(row)
            sft_images.add(row["imageId"])
        elif len(rl_rows) < n_rl:
            if row["imageId"] not in sft_images:
                rl_rows.append(row)
        else:
            break
    return (
        _dataset_from(_gqa_examples(sft_rows, images)),
        _dataset_from(_gqa_examples(rl_rows, images)),
    )


def load_gqa_eval() -> Dataset:
    """Full official testdev_balanced split (12,578 questions)."""
    instructions = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    images = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    return _dataset_from(_gqa_examples(instructions, images))


def load_clevr(n_rl: int, n_val: int, n_test: int, seed: int) -> dict[str, Dataset]:
    """Split the R1-V CLEVR-CoGenT counting set into held-out test/val and an RL pool."""
    rows = load_dataset("leonardPKU/clevr_cogen_a_train", split="train").shuffle(seed=seed)

    def emit(start: int, count: int, tag: str) -> Iterator[dict]:
        for i in range(start, start + count):
            row = rows[i]
            yield {
                "dataset": "clevr",
                "qid": f"{tag}-{i}",
                "image": row["image"].convert("RGB"),  # source images are RGBA
                "question": row["problem"],
                "answers": [extract_answer(row["solution"])],
                "answer_type": "number",
            }

    return {
        "test": _dataset_from(emit(0, n_test, "test")),
        "val": _dataset_from(emit(n_test, n_val, "val")),
        "rl": _dataset_from(emit(n_test + n_val, n_rl, "rl")),
    }


def load_textvqa_eval(n: int, seed: int) -> Dataset:
    rows = load_dataset("lmms-lab/textvqa", split="validation").shuffle(seed=seed).select(range(n))

    def emit() -> Iterator[dict]:
        for row in rows:
            yield {
                "dataset": "textvqa",
                "qid": str(row["question_id"]),
                "image": row["image"],
                "question": row["question"],
                "answers": row["answers"],
                "answer_type": "",
            }

    return _dataset_from(emit())
