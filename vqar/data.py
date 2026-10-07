"""Dataset loaders: sources → one unified schema, materialized by prepare_data.py.

`vqar.registry` declares each dataset (sources, sizes, roles, scorers) and calls these
loaders to materialize its subsets.

Unified example: {dataset, qid, image, question, answers: list[str], answer_type}.
VQAv2 eval keeps all 10 annotator answers (required by the official metric);
single-answer datasets store a one-element list.

Memory notes (30GB-RAM WSL2 box): images are read with decode=False and passed
through as bytes — never decoded, never held in a list. Only CLEVR is decoded,
one image at a time, because its RGBA sources need RGB conversion.

Source-ID assignment alone does not guarantee image disjointness: COCO/VG images
can appear under different IDs or across VQAv2 and GQA. Run scripts.check_data
after materialization; its training views hold dev/evaluation images out while
allowing SFT training examples to be reused in RL.
"""

from collections.abc import Iterator

from datasets import Dataset, Features, Image, Sequence, Value, load_dataset
from PIL import Image as PILImage

from vqar.answers import extract_answer
from vqar.config import MAX_PIXELS, RL_IMAGE_SIZE
from vqar.registry import SOURCE_REVISIONS


def _load_source(repo_id, *args, **kwargs):
    """Pin new materializations; historical downloads may predate these revisions."""
    return load_dataset(repo_id, *args, revision=SOURCE_REVISIONS[repo_id], **kwargs)


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


def cap_pixels(image: PILImage.Image, max_pixels: int = MAX_PIXELS) -> PILImage.Image:
    """Resize down to the project pixel budget (aspect preserved).

    Applied in the TRAINING transforms rather than via processor config: in GRPO
    colocate mode, vLLM preprocesses images with its own default processor, so a
    runtime-patched training processor disagrees with it on image token counts and
    Qwen3-VL's M-RoPE indexing crashes (shape mismatch in get_rope_index). Capping
    the pixels in the data makes every consumer see the same image.
    """
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), PILImage.LANCZOS)


def to_uniform_canvas(image: PILImage.Image) -> PILImage.Image:
    """Resize to fit (aspect preserved) and center on a black RL_IMAGE_SIZE square.

    Every RL prompt image goes through this, so rollouts and training share one
    vision-token budget; the `canvas` evaluation image mode applies it too.
    """
    image = image.convert("RGB")
    w, h = image.size
    scale = RL_IMAGE_SIZE / max(w, h)
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = PILImage.new("RGB", (RL_IMAGE_SIZE, RL_IMAGE_SIZE))
    canvas.paste(image, ((RL_IMAGE_SIZE - image.width) // 2, (RL_IMAGE_SIZE - image.height) // 2))
    return canvas


def _clean_cauldron_answer(text: str) -> str:
    """Cauldron reformatted VQAv2 answers as 'Net.' — restore the dataset's lowercase form."""
    return text.strip().removesuffix(".").lower()


def _vqav2_assignments(rows: Dataset, n_sft: int, n_rl: int) -> dict[str, list[tuple[int, int]]]:
    """Walk shuffled image-rows and assign (row, qa) pairs: SFT quota first, then RL.

    These are storage buckets; checked RL views can also reuse the SFT bucket.
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
    rows = _load_source("HuggingFaceM4/the_cauldron", "vqav2", split="train").shuffle(seed=seed)
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
    rows = _load_source("lmms-lab/VQAv2", split="validation").shuffle(seed=seed).select(range(n))
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
    images = _load_source("lmms-lab/GQA", config, split=split)
    return images.cast_column("image", Image(decode=False))


def load_gqa_train(n_sft: int, n_rl: int, seed: int) -> tuple[Dataset, Dataset]:
    """Sample GQA balanced-train QAs: SFT quota first, then RL from unused images only."""
    instructions = _load_source(
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
        _load_source("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
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
    rows = _load_source("leonardPKU/clevr_cogen_a_train", split="train").shuffle(seed=seed)
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
    rows = _load_source("lmms-lab/textvqa", split="validation").shuffle(seed=seed).select(range(n))
    rows = rows.cast_column("image", Image(decode=False))
    return _build(_textvqa_generate, rows=rows)


def _rgb_png(payload: bytes) -> bytes:
    """The image as RGB PNG bytes; RGB sources pass through unchanged."""
    import io

    with PILImage.open(io.BytesIO(payload)) as image:
        if image.mode == "RGB":
            return payload
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "PNG")
        return buffer.getvalue()


def _from_records(records: list[dict]) -> Dataset:
    """Plain dicts (image bytes included) to the unified schema, off the slow iterator path."""
    def generate(records):
        for record in records:
            yield {**record, "image": {"bytes": record["image"], "path": None}}

    return _build(generate, records=records)


def _parquet_rows(repo_id: str, filename: str, columns: list[str]) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, filename, repo_type="dataset",
                           revision=SOURCE_REVISIONS[repo_id])
    return pq.read_table(path, columns=columns).to_pylist()


CHARTQA = "HuggingFaceM4/ChartQA"
CHARTQA_HUMAN = 0  # human_or_machine: 0 marks the human-written questions


def _chartqa_answer_type(answer: str) -> str:
    from vqar.metrics import _number

    return "number" if _number(answer) is not None else "text"


def load_chartqa() -> dict[str, Dataset]:
    """ChartQA's human-written questions of each official split, in source order.

    The machine-generated half (mostly value lookup) is left out. Question IDs keep the
    official split and the row's position in it.
    """
    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files(CHARTQA, repo_type="dataset",
                                                       revision=SOURCE_REVISIONS[CHARTQA])
                   if f.endswith(".parquet"))
    subsets = {}
    for split, name in (("train", "train"), ("val", "val"), ("test", "test")):
        rows = [row for f in files if f.startswith(f"data/{split}-")
                for row in _parquet_rows(CHARTQA, f, ["image", "query", "label",
                                                      "human_or_machine"])]
        records = [{"dataset": "chartqa", "qid": f"{split}-{i}",
                    "image": _rgb_png(row["image"]["bytes"]),
                    "question": row["query"], "answers": row["label"],
                    "answer_type": _chartqa_answer_type(row["label"][0])}
                   for i, row in enumerate(rows) if row["human_or_machine"] == CHARTQA_HUMAN]
        subsets[name] = _from_records(records)
    return subsets


CAULDRON = "HuggingFaceM4/the_cauldron"


def _clevr_answer_type(answer: str) -> str:
    if answer in ("yes", "no"):
        return "yes/no"
    return "number" if answer.isdigit() else "attribute"


def _cauldron_clevr_shard(index: int) -> list[dict]:
    from huggingface_hub import HfApi

    shards = sorted(f for f in HfApi().list_repo_files(CAULDRON, repo_type="dataset",
                                                        revision=SOURCE_REVISIONS[CAULDRON])
                    if f.startswith("clevr/") and f.endswith(".parquet"))
    return _parquet_rows(CAULDRON, shards[index], ["images", "texts"])


def _clevr_questions(row: dict, shard: int, image: int, turns: list[int]) -> list[dict]:
    """Questions of one CLEVR image; the cauldron appends a style instruction after a newline."""
    payload = _rgb_png(row["images"][0]["bytes"])
    questions = []
    for turn in turns:
        text = row["texts"][turn]
        answer = _clean_cauldron_answer(text["assistant"])
        questions.append({"dataset": "clevrcomp", "qid": f"{shard}-{image}-{turn}",
                          "image": payload, "question": text["user"].split("\n")[0].strip(),
                          "answers": [answer], "answer_type": _clevr_answer_type(answer)})
    return questions


def load_clevr_compositional(train_shard: int, test_shard: int, per_image: int,
                             seed: int) -> dict[str, Dataset]:
    """CLEVR v1.0 questions (filter, count, compare, query; ten per image in the source)
    from two cauldron shards: `per_image` seeded questions of every training-shard image,
    and one seeded question per image of the test shard."""
    import random

    rng = random.Random(seed)
    train = [question for i, row in enumerate(_cauldron_clevr_shard(train_shard))
             for question in _clevr_questions(
                 row, train_shard, i, sorted(rng.sample(range(len(row["texts"])), per_image)))]
    test = [question for i, row in enumerate(_cauldron_clevr_shard(test_shard))
            for question in _clevr_questions(
                row, test_shard, i, [rng.randrange(len(row["texts"]))])]
    return {"train": _from_records(train), "test": _from_records(test)}
