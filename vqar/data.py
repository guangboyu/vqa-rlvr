"""Dataset loaders: four sources → one unified schema, materialized by prepare_data.py.

Unified example: {dataset, qid, image, question, answers: list[str], answer_type}.
VQAv2 eval keeps all 10 annotator answers (required by the official metric);
single-answer datasets store a one-element list.

Memory notes (30GB-RAM WSL2 box): images are read with decode=False and passed
through as bytes — never decoded, never held in a list. Only CLEVR is decoded,
one image at a time, because its RGBA sources need RGB conversion.

Train subsets for SFT and RL are sampled in a single deterministic assignment pass
so they are image-disjoint by construction. Eval splits are officially disjoint
from train (VQAv2: train2014 vs val2014 images; GQA: train_balanced vs
testdev_balanced; CLEVR: held-out shards of the R1-V set never used for training).
"""

from collections.abc import Iterator

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


def _build(gen, **gen_kwargs) -> Dataset:
    return Dataset.from_generator(gen, features=FEATURES, gen_kwargs=gen_kwargs)


def _clean_cauldron_answer(text: str) -> str:
    """Cauldron reformatted VQAv2 answers as 'Net.' — restore the dataset's lowercase form."""
    return text.strip().removesuffix(".").lower()


def _vqav2_assignments(rows: Dataset, n_sft: int, n_rl: int) -> dict[str, list[tuple[int, int]]]:
    """Walk shuffled image-rows and assign (row, qa) pairs: SFT quota first, then RL.

    An image's QAs are never split across buckets, so the subsets are image-disjoint.
    """
    texts = rows.select_columns(["texts"])  # never touches the image column
    out: dict[str, list[tuple[int, int]]] = {"sft": [], "rl": []}
    for i, row in enumerate(texts):
        bucket = "sft" if len(out["sft"]) < n_sft else "rl"
        quota = n_sft if bucket == "sft" else n_rl
        for j in range(len(row["texts"])):
            if len(out[bucket]) >= quota:
                break
            out[bucket].append((i, j))
        if len(out["rl"]) >= n_rl:
            return out
    raise ValueError(f"source exhausted before filling quotas (sft={n_sft}, rl={n_rl})")


def _vqav2_generate(rows: Dataset, assignments: list[tuple[int, int]]) -> Iterator[dict]:
    for i, j in assignments:
        row = rows[i]
        qa = row["texts"][j]
        yield {
            "dataset": "vqav2",
            "qid": f"cauldron-{i}-{j}",
            "image": row["images"][0],  # {bytes, path} passthrough, not decoded
            "question": qa["user"].split("\n")[0].strip(),
            "answers": [_clean_cauldron_answer(qa["assistant"])],
            "answer_type": "",
        }


def load_vqav2_train(n_sft: int, n_rl: int, seed: int) -> tuple[Dataset, Dataset]:
    """Flatten the_cauldron/vqav2 (multi-QA per image) into SFT and RL QA subsets."""
    rows = load_dataset("HuggingFaceM4/the_cauldron", "vqav2", split="train").shuffle(seed=seed)
    rows = rows.cast_column("images", Sequence(Image(decode=False)))
    assignments = _vqav2_assignments(rows, n_sft, n_rl)
    return (
        _build(_vqav2_generate, rows=rows, assignments=assignments["sft"]),
        _build(_vqav2_generate, rows=rows, assignments=assignments["rl"]),
    )


def _vqav2_eval_generate(rows: Dataset) -> Iterator[dict]:
    for row in rows:
        yield {
            "dataset": "vqav2",
            "qid": str(row["question_id"]),
            "image": row["image"],
            "question": row["question"],
            "answers": [a["answer"] for a in row["answers"]],
            "answer_type": row["answer_type"],
        }


def load_vqav2_eval(n: int, seed: int) -> Dataset:
    rows = load_dataset("lmms-lab/VQAv2", split="validation").shuffle(seed=seed).select(range(n))
    rows = rows.cast_column("image", Image(decode=False))
    return _build(_vqav2_eval_generate, rows=rows)


def _gqa_image_payloads(images: Dataset, needed: set[str]) -> dict[str, dict]:
    """One sequential pass collecting {bytes, path} payloads for the needed ids.

    Runs OUTSIDE from_generator on purpose: iterating a Dataset inside a builder
    callback falls off the fast formatting path (~1s/row; profiled via faulthandler).
    """
    payloads: dict[str, dict] = {}
    for row in images:
        if row["id"] in needed:
            payloads[row["id"]] = row["image"]
    return payloads


def _gqa_generate(instructions: list[dict], payloads: dict[str, dict]) -> Iterator[dict]:
    for row in instructions:
        yield {
            "dataset": "gqa",
            "qid": row["id"],
            "image": payloads[row["imageId"]],
            "question": row["question"],
            "answers": [row["answer"]],
            "answer_type": row["types"]["structural"],
        }


def _gqa_images(config: str, split: str) -> Dataset:
    images = load_dataset("lmms-lab/GQA", config, split=split)
    return images.cast_column("image", Image(decode=False))


def load_gqa_train(n_sft: int, n_rl: int, seed: int) -> tuple[Dataset, Dataset]:
    """Sample GQA balanced-train QAs: SFT quota first, then RL from unused images only."""
    instructions = load_dataset(
        "lmms-lab/GQA", "train_balanced_instructions", split="train"
    ).shuffle(seed=seed)
    images = _gqa_images("train_balanced_images", "train")

    sft_rows, rl_rows, sft_images = [], [], set()
    for row in instructions:
        if len(sft_rows) < n_sft:
            sft_rows.append(row)
            sft_images.add(row["imageId"])
        elif row["imageId"] not in sft_images:
            rl_rows.append(row)
            if len(rl_rows) >= n_rl:
                break
    needed = {row["imageId"] for row in sft_rows + rl_rows}
    payloads = _gqa_image_payloads(images, needed)
    return (
        _build(_gqa_generate, instructions=sft_rows, payloads=payloads),
        _build(_gqa_generate, instructions=rl_rows, payloads=payloads),
    )


def load_gqa_eval() -> Dataset:
    """Full official testdev_balanced split (12,578 questions)."""
    instructions = list(
        load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    )
    images = _gqa_images("testdev_balanced_images", "testdev")
    payloads = _gqa_image_payloads(images, {row["imageId"] for row in instructions})
    return _build(_gqa_generate, instructions=instructions, payloads=payloads)


def _clevr_generate(rows: Dataset, start: int, count: int, tag: str) -> Iterator[dict]:
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


def load_clevr(n_rl: int, n_val: int, n_test: int, seed: int) -> dict[str, Dataset]:
    """Split the R1-V CLEVR-CoGenT counting set into held-out test/val and an RL pool."""
    rows = load_dataset("leonardPKU/clevr_cogen_a_train", split="train").shuffle(seed=seed)
    return {
        "test": _build(_clevr_generate, rows=rows, start=0, count=n_test, tag="test"),
        "val": _build(_clevr_generate, rows=rows, start=n_test, count=n_val, tag="val"),
        "rl": _build(_clevr_generate, rows=rows, start=n_test + n_val, count=n_rl, tag="rl"),
    }


def _textvqa_generate(rows: Dataset) -> Iterator[dict]:
    for row in rows:
        yield {
            "dataset": "textvqa",
            "qid": str(row["question_id"]),
            "image": row["image"],
            "question": row["question"],
            "answers": row["answers"],
            "answer_type": "",
        }


def load_textvqa_eval(n: int, seed: int) -> Dataset:
    rows = load_dataset("lmms-lab/textvqa", split="validation").shuffle(seed=seed).select(range(n))
    rows = rows.cast_column("image", Image(decode=False))
    return _build(_textvqa_generate, rows=rows)
