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

import re
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


def upscale_to_budget(image: PILImage.Image, max_pixels: int = MAX_PIXELS) -> PILImage.Image:
    """Resize to the pixel budget with the aspect kept, up or down: small medical images (a
    224x224 MRI slice is 49 vision tokens) give the vision encoder more patches to read."""
    image = image.convert("RGB")
    w, h = image.size
    scale = (max_pixels / (w * h)) ** 0.5
    if abs(scale - 1) < 1e-3:
        return image
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(size, PILImage.BICUBIC if scale > 1 else PILImage.LANCZOS)


def to_uniform_canvas(image: PILImage.Image, size: int = RL_IMAGE_SIZE) -> PILImage.Image:
    """Resize to fit (aspect preserved) and center on a black square of side `size`.

    Every RL prompt image goes through this, so rollouts and training share one
    vision-token budget; the `canvas` evaluation image mode applies it too.
    """
    image = image.convert("RGB")
    w, h = image.size
    scale = size / max(w, h)
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = PILImage.new("RGB", (size, size))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
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


MEDTHINK = "bio-nlp-umass/MedThinkVQA"
# Multi-image rows keep their first image in `image` (split keys, swapped donors) and every
# shown image, the first included, in `images`.
MULTI_IMAGE_FEATURES = Features({**FEATURES, "images": Sequence(Image())})


def _medthink_question(row: dict) -> str:
    """The clinical history and the five candidate diagnoses; the imaging findings, the
    discussion and the image captions are expert information and stay out."""
    options = "\n".join(f"{letter}. {text}" for letter, text in row["options"].items())
    return (f"{(row['CLINICAL_HISTORY'] or '').strip()}\n"
            f"Which is the most likely diagnosis?\n{options}\n"
            "Answer with the letter of the correct option.")


def _medthink_images(row: dict, archive, max_images: int) -> list[bytes]:
    """The case's images in order, at most `max_images`; parquet embeds only some of them."""
    images = []
    for index in range(1, min(row["image_count"], max_images) + 1):
        embedded = row.get(f"image_{index:02d}")
        payload = (embedded["bytes"] if embedded and embedded.get("bytes")
                   else archive.read(row[f"image_{index:02d}_path"]))
        images.append(_rgb_png(payload))
    return images


def load_medthink(max_images: int) -> dict[str, Dataset]:
    """MedThinkVQA's official train and test cases: history, images and five options."""
    import zipfile

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def fetch(filename: str) -> str:
        return hf_hub_download(MEDTHINK, filename, repo_type="dataset",
                               revision=SOURCE_REVISIONS[MEDTHINK])

    archive = zipfile.ZipFile(fetch("images.zip"))
    subsets = {}
    for split in ("train", "test"):
        table = pq.read_table(fetch(f"{split}.parquet"))
        columns = ["title", "CLINICAL_HISTORY", "options", "correct_answer",
                   "correct_answer_text", "image_count", "image_01_modality"]
        columns += [name for name in table.column_names
                    if name.startswith("image_") and name.split("_")[1].isdigit()
                    and int(name.split("_")[1]) <= max_images
                    and (name.count("_") == 1 or name.endswith("_path"))]
        records = []
        for row in table.select(columns).to_pylist():
            images = _medthink_images(row, archive, max_images)
            records.append({
                "dataset": "medthink",
                "qid": row["image_01_path"].split("/")[1],
                "image": {"bytes": images[0], "path": None},
                "images": [{"bytes": image, "path": None} for image in images],
                "question": _medthink_question(row),
                "answers": [row["correct_answer"], row["correct_answer_text"]],
                "answer_type": row["image_01_modality"] or "",
            })
        subsets[split] = Dataset.from_generator(_yield, features=MULTI_IMAGE_FEATURES,
                                                gen_kwargs={"records": records})
    return subsets


def _yield(records):
    yield from records


# Medical candidates (Part 2, 2026-10-10). Images are read one at a time inside the generator,
# from a zip archive or a file, so a large source never sits in memory. Records go in as a
# tuple: datasets treats a list in gen_kwargs as shards and calls the generator once per
# element, which would reopen a 10 GB archive for every question.


def _content_key(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _split_by_image(records: list[dict], seed: int, held_out: float) -> dict[str, list[dict]]:
    """Re-split questions by image: a seeded `held_out` share of images each for test and
    val, the rest train. Records carry `key`, a content hash, so exact copies stay together."""
    import random

    keys = sorted({record["key"] for record in records})
    random.Random(seed).shuffle(keys)
    n = round(len(keys) * held_out)
    split_of = {key: "test" for key in keys[:n]}
    split_of.update({key: "val" for key in keys[n:2 * n]})
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        splits[split_of.get(record["key"], "train")].append(record)
    return splits


def _unified(record: dict, payload: bytes) -> dict:
    row = {name: record[name] for name in ("dataset", "qid", "question", "answers", "answer_type")}
    return {**row, "image": {"bytes": _rgb_png(payload), "path": None}}


def _from_zip(archive: str, records: list[dict]) -> Dataset:
    """Unified rows whose images are zip members (`member`), read lazily."""
    def generate(archive, records):
        import zipfile

        with zipfile.ZipFile(archive) as source:
            for record in records:
                yield _unified(record, source.read(record["member"]))

    return _build(generate, archive=archive, records=tuple(records))


def _from_files(records: list[dict]) -> Dataset:
    """Unified rows whose images are files on disk (`file`), read lazily."""
    def generate(records):
        from pathlib import Path

        for record in records:
            yield _unified(record, Path(record["file"]).read_bytes())

    return _build(generate, records=tuple(records))


KVASIR_X1 = "SimulaMet/Kvasir-VQA-x1"


def _kvasir_atomic(rows: list[dict], split: str) -> list[dict]:
    """The complexity-1 rows as their atomic question and short label (`original`); the
    rephrased question and sentence answer can disagree with that label (a yes/no wording
    over a location label), so the label's own question is shown."""
    import json

    atomic = []
    for index, row in enumerate(rows):
        if row["complexity"] != 1:
            continue
        (pair,) = json.loads(row["original"])
        atomic.append({"dataset": "kvasirx1", "qid": f"{split}-c1-{index}",
                       "img_id": row["img_id"], "question": pair["q"].strip(),
                       "answers": [pair["a"].strip()], "answer_type": row["question_class"][0]})
    return atomic


def load_kvasir_x1(seed: int, held_out: float) -> dict[str, Dataset]:
    """Kvasir-VQA-x1's atomic questions on GI endoscopy images, re-split by image.

    The release's train and test splits share 3,821 of 6,449 images (and nearly every test
    atomic question), so both are pooled, de-duplicated by (image, question), and re-split.
    """
    from pathlib import Path

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(KVASIR_X1, repo_type="dataset",
                                  revision=SOURCE_REVISIONS[KVASIR_X1]))
    records, seen = [], set()
    for split in ("train", "test"):
        rows = pq.read_table(root / "data" / f"{split}-00000-of-00001.parquet",
                             columns=["img_id", "complexity", "original",
                                      "question_class"]).to_pylist()
        for record in _kvasir_atomic(rows, split):
            if (record["img_id"], record["question"]) not in seen:
                seen.add((record["img_id"], record["question"]))
                records.append(record)
    keys = {}
    for record in records:
        record["file"] = str(root / "images" / f"{record['img_id']}.jpg")
        if record["img_id"] not in keys:
            keys[record["img_id"]] = _content_key(Path(record["file"]).read_bytes())
        record["key"] = keys[record["img_id"]]
    return {name: _from_files(part)
            for name, part in _split_by_image(records, seed, held_out).items()}


OMNIMEDVQA = "foreverbeliever/OmniMedVQA"


def _omnimed_question(row: dict, letters: list[str]) -> str:
    options = "\n".join(f"{letter}. {row[f'option_{letter}']}" for letter in letters)
    return f"{row['question']}\n{options}\nAnswer with the letter of the correct option."


def _omnimed_record(row: dict, dataset: str = "omnimedvqa") -> dict | None:
    """One open-access item as a multiple-choice question; None when no option is the answer."""
    letters = [letter for letter in "ABCD" if row.get(f"option_{letter}") is not None]
    correct = [letter for letter in letters if row[f"option_{letter}"] == row["gt_answer"]]
    if len(correct) != 1:
        return None
    return {"dataset": dataset, "qid": row["question_id"],
            "member": f"OmniMedVQA/{row['image_path']}",
            "question": _omnimed_question(row, letters),
            "answers": [correct[0], row["gt_answer"]],
            "answer_type": row["question_type"]}


def load_omnimedvqa(seed: int, held_out: float) -> dict[str, Dataset]:
    """OmniMedVQA's open-access items (the restricted half ships without images), re-split
    by image: the benchmark has no training split."""
    import json
    import zipfile

    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(OMNIMEDVQA, "OmniMedVQA.zip", repo_type="dataset",
                              revision=SOURCE_REVISIONS[OMNIMEDVQA])
    records, keys = [], {}
    with zipfile.ZipFile(archive) as source:
        for name in sorted(n for n in source.namelist()
                           if "/QA_information/Open-access/" in n and n.endswith(".json")):
            for row in json.loads(source.read(name)):
                record = _omnimed_record(row)
                if record is None:
                    continue
                if record["member"] not in keys:
                    keys[record["member"]] = _content_key(source.read(record["member"]))
                record["key"] = keys[record["member"]]
                records.append(record)
    return {name: _from_zip(archive, part)
            for name, part in _split_by_image(records, seed, held_out).items()}


# Source collections of the open-access release that share images, each held out or trained on
# as one unit: the first two groups share decoded pixels exactly (11, 369 and 12 images), the
# COVID chest X-ray group shares resized copies (15 matches of a 16x16 difference hash).
OMNIMED_FAMILIES = (("Adam Challenge", "PALM2019"),
                    ("CoronaHack", "OCT & X-Ray 2017", "Retinal OCT-C8"),
                    ("COVIDx CXR-4", "Chest X-Ray PA", "Covid-19 tianchi", "Covid19 heywhale"))
OMNIMED_IN_TRAINING = ("RadImageNet",)  # about 64% of the questions; never held out


def held_out_collections(sizes: dict[str, int], modality: dict[str, str], seed: int,
                         share: float, families=OMNIMED_FAMILIES,
                         keep=OMNIMED_IN_TRAINING) -> list[str]:
    """Whole source collections for a held-out role, about `share` of all questions.

    Collections that share images (`families`) move as one unit, labelled by the modality of
    its largest collection. Units are drawn round-robin over modalities, each in a seeded
    order, so several modalities are tested, and every modality keeps at least one unit in
    training; the `keep` collections always stay in training.
    """
    import random

    grouped = {name for family in families for name in family}
    units = [tuple(sorted(f for f in family if f in sizes)) for family in families]
    units = [unit for unit in units if unit]
    units += [(name,) for name in sorted(sizes) if name not in grouped]
    units = [unit for unit in units if not set(unit) & set(keep)]
    rng = random.Random(seed)
    by_modality: dict[str, list[tuple]] = {}
    for unit in rng.sample(units, len(units)):
        label = modality[max(unit, key=lambda name: sizes[name])]
        by_modality.setdefault(label, []).append(unit)
    order = rng.sample(sorted(by_modality), len(by_modality))
    target, chosen, count = share * sum(sizes.values()), [], 0
    while count < target:
        progressed = False
        for label in order:
            left = [unit for unit in by_modality[label] if unit not in chosen]
            if len(left) <= 1 or count >= target:
                continue
            chosen.append(left[0])
            count += sum(sizes[name] for name in left[0])
            progressed = True
        if not progressed:
            break
    return sorted(name for unit in chosen for name in unit)


def load_omnimedx(seed: int, test_share: float) -> dict[str, Dataset]:
    """OmniMedVQA's open-access items with whole source collections held out for test, so a
    model cannot score by recognizing collections it trained on (held_out_collections)."""
    import json
    import random
    import zipfile
    from collections import Counter

    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(OMNIMEDVQA, "OmniMedVQA.zip", repo_type="dataset",
                              revision=SOURCE_REVISIONS[OMNIMEDVQA])
    records = []
    with zipfile.ZipFile(archive) as source:
        for name in sorted(n for n in source.namelist()
                           if "/QA_information/Open-access/" in n and n.endswith(".json")):
            for row in json.loads(source.read(name)):
                record = _omnimed_record(row, "omnimedx")
                if record is not None:
                    records.append({**record, "collection": row["dataset"],
                                    "modality": row.get("modality_type")})
    sizes = Counter(record["collection"] for record in records)
    modality = {name: Counter(r["modality"] for r in records if r["collection"] == name
                              and r["modality"]).most_common(1)[0][0] for name in sizes}
    test = set(held_out_collections(sizes, modality, seed, test_share))
    # Rows in a seeded order: collection by collection, one collection's 13-megapixel images
    # (ISIC2020) would fill a saved shard past Arrow's 2 GB limit.
    rng = random.Random(seed)
    parts = {"train": [r for r in records if r["collection"] not in test],
             "test": [r for r in records if r["collection"] in test]}
    return {name: _from_zip(archive, rng.sample(part, len(part))) for name, part in parts.items()}


SLAKE = "BoKelvin/SLAKE"


def load_slake() -> dict[str, Dataset]:
    """SLAKE's English questions of each official split (split by image by its authors)."""
    import json

    from huggingface_hub import hf_hub_download

    def fetch(filename: str) -> str:
        return hf_hub_download(SLAKE, filename, repo_type="dataset",
                               revision=SOURCE_REVISIONS[SLAKE])

    archive = fetch("imgs.zip")
    subsets = {}
    for split in ("train", "validation", "test"):
        rows = json.loads(open(fetch(f"{split}.json")).read())
        records = [{"dataset": "slake", "qid": str(row["qid"]),
                    "member": f"imgs/{row['img_name']}", "question": row["question"].strip(),
                    "answers": [row["answer"].strip()],
                    "answer_type": row["answer_type"].lower()}
                   for row in rows if row["q_lang"] == "en"]
        subsets[split] = _from_zip(archive, records)
    return subsets


PATHVQA = "flaviagiammarino/path-vqa"


def _pathvqa_generate(paths: list[str], split: str) -> Iterator[dict]:
    """Rows straight from the parquet files, one batch at a time (never a Dataset)."""
    import pyarrow.parquet as pq

    index = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                answer = row["answer"].strip()
                yield _unified({"dataset": "pathvqa", "qid": f"{split}-{index}",
                                "question": row["question"].strip(), "answers": [answer],
                                "answer_type": "yes/no" if answer.lower() in ("yes", "no")
                                else "open"}, row["image"]["bytes"])
                index += 1


def load_pathvqa() -> dict[str, Dataset]:
    """PathVQA's official splits: pathology textbook and PEIR images, yes/no and open answers."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(PATHVQA, repo_type="dataset",
                                  revision=SOURCE_REVISIONS[PATHVQA]))
    subsets = {}
    for split in ("train", "validation", "test"):
        paths = sorted(str(path) for path in (root / "data").glob(f"{split}-*.parquet"))
        subsets[split] = _build(_pathvqa_generate, paths=paths, split=split)
    return subsets


REXVQA = "rajpurkarlab/ReXVQA"
REXGRADIENT = "rajpurkarlab/ReXGradient-160K"
# Only the first of ReXGradient's ten archive parts is downloaded and extracted (the licence
# allows only the portion a project needs). The archive is ordered by patient ID, so that part
# holds every image of the patients whose IDs sort first, about a tenth of each split.
REXGRADIENT_PNG = "REXGRADIENT_PNG"  # environment variable: the extracted archive's root
REXGRADIENT_PNG_DEFAULT = "/mnt/f/VQA/raw/rexgradient/png"
FRONTAL_VIEWS = ("PA", "AP")  # in order of preference


def _rexvqa_question(row: dict) -> str:
    """The question and its lettered options; the report text in the row is never shown."""
    options = "\n".join(option.strip() for option in row["options"])
    return f"{row['question'].strip()}\n{options}\nAnswer with the letter of the correct option."


def _rexvqa_answers(row: dict) -> list[str] | None:
    """The correct letter and its option text; None when no option carries that letter."""
    letter = row["correct_answer"].strip().upper()
    for option in row["options"]:
        if option.strip().startswith(f"{letter}."):
            return [letter, option.strip()[len(letter) + 1:].strip()]
    return None


def _frontal_images(meta: dict, root, patients: set[str]) -> dict[str, str]:
    """Per study with an extracted image, one frontal view: the first PA image, else the first
    AP image. Studies with only lateral or unlabeled views are left out."""
    chosen = {}
    for study, entry in meta.items():
        if entry["PatientID"] not in patients:
            continue
        found = {}
        for path, view in zip(entry["ImagePath"], entry["ImageViewPosition"], strict=True):
            file = root / path.removeprefix("../")
            if view in FRONTAL_VIEWS and view not in found and file.is_file():
                found[view] = str(file)
        view = next((view for view in FRONTAL_VIEWS if view in found), None)
        if view:
            chosen[study] = found[view]
    return chosen


def _cxr_rgb_png(path: str) -> bytes:
    """A 16-bit grayscale chest X-ray as 8-bit RGB PNG, stretched between its own minimum and
    maximum (the usual display window when no DICOM window is stored)."""
    import io

    import numpy as np

    with PILImage.open(path) as image:
        pixels = np.asarray(image, dtype=np.float32)
    low, high = float(pixels.min()), float(pixels.max())
    scaled = ((pixels - low) * (255.0 / max(high - low, 1.0))).round().astype(np.uint8)
    buffer = io.BytesIO()
    PILImage.fromarray(scaled).convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


def _rexvqa_generate(records: tuple[dict, ...]) -> Iterator[dict]:
    for record in records:
        row = {name: record[name] for name in ("dataset", "qid", "question", "answers",
                                               "answer_type")}
        yield {**row, "image": {"bytes": _cxr_rgb_png(record["file"]), "path": None}}


def rexvqa_records(questions: dict, meta: dict, root, patients: set[str]) -> list[dict]:
    """One split's questions whose study has an extracted frontal image, in question-ID order."""
    frontal = _frontal_images(meta, root, patients)
    records = []
    for qid in sorted(questions):
        row = questions[qid]
        answers = _rexvqa_answers(row)
        if row["study_id"] in frontal and answers is not None:
            records.append({"dataset": "rexvqa", "qid": qid, "file": frontal[row["study_id"]],
                            "question": _rexvqa_question(row), "answers": answers,
                            "answer_type": row["task_name"], "patient": row["PatientID"]})
    return records


def load_rexvqa() -> dict[str, Dataset]:
    """ReXVQA's multiple-choice chest X-ray questions on the official patient-level splits,
    restricted to studies with a frontal image in the extracted archive part."""
    import json
    import os
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    def fetch(repo: str, filename: str) -> str:
        return hf_hub_download(repo, filename, repo_type="dataset",
                               revision=SOURCE_REVISIONS[repo])

    root = Path(os.environ.get(REXGRADIENT_PNG, REXGRADIENT_PNG_DEFAULT))
    if not (root / "deid_png").is_dir():
        raise FileNotFoundError(f"No extracted ReXGradient images under {root}; set "
                                f"{REXGRADIENT_PNG} to the folder holding deid_png/")
    patients = set(os.listdir(root / "deid_png"))
    splits, owners = {}, {}
    for split in ("train", "valid", "test"):
        meta = json.loads(Path(fetch(REXGRADIENT,
                                     f"metadata/{split}_metadata_view_position.json")).read_text())
        questions = json.loads(Path(fetch(REXVQA, f"metadata/{split}_vqa_data.json")).read_text())
        records = rexvqa_records(questions, meta, root, patients)
        for record in records:
            if owners.setdefault(record["patient"], split) != split:
                raise ValueError(f"Patient {record['patient']} is in {owners[record['patient']]}"
                                 f" and {split}")
        splits[split] = _build(_rexvqa_generate, records=tuple(records))
    return splits



# DeepTumorVQA v2 (2026-10-29): multiple-choice questions on abdominal CT, read from a local
# download of the pinned Hugging Face revision (question files and the 2D montages only).
DEEPTUMOR = "tumor-vqa/DeepTumorVQA_2.0"
DEEPTUMOR_ROOT = "DEEPTUMORVQA_ROOT"  # environment variable: the downloaded dataset folder
DEEPTUMOR_ROOT_DEFAULT = "/mnt/f/VQA/raw/deeptumorvqa"
_DEEPTUMOR_OPTION = re.compile(r"(?:^|(?<=\s))([A-F]):\s")


def _deeptumor_parts(mc_question: str) -> tuple[str, list[str]] | None:
    """The stem and the option texts in letter order, from the released form
    'stem A: x B: y C: z'; None when the letters do not run A, B, ..."""
    parts = _DEEPTUMOR_OPTION.split(mc_question.strip())
    stem, letters, texts = parts[0].strip(), parts[1::2], [t.strip() for t in parts[2::2]]
    if len(letters) < 2 or letters != list("ABCDEF"[:len(letters)]) or not all(texts):
        return None
    return stem, texts


def _deeptumor_prompt(stem: str, texts: list[str]) -> str:
    options = "\n".join(f"{chr(ord('A') + i)}. {text}" for i, text in enumerate(texts))
    return f"{stem}\n{options}\nAnswer with the letter of the correct option."


def deeptumor_question(mc_question: str) -> tuple[str, list[str]] | None:
    """The question with one lettered option per line, and the option texts in letter order;
    None when the released options do not parse."""
    parts = _deeptumor_parts(mc_question)
    return None if parts is None else (_deeptumor_prompt(*parts), parts[1])


def deeptumor_record(qid: str, patient: str, mc_question: str, correct: str, answer: str,
                     answer_type: str, file: str, subtype: str = "") -> dict | None:
    """One question, or None when it has several correct letters, its options do not parse, or
    the released answer is not the text of the correct option."""
    parts = _deeptumor_parts(mc_question)
    letter = correct.strip().upper()
    if parts is None or len(letter) != 1:
        return None
    stem, texts = parts
    index = ord(letter) - ord("A")
    if not 0 <= index < len(texts) or texts[index] != answer.strip():
        return None
    return {"dataset": "deeptumor", "qid": qid, "patient": patient, "file": file,
            "question": _deeptumor_prompt(stem, texts), "answers": [letter, texts[index]],
            "answer_type": answer_type, "subtype": subtype, "stem": stem, "options": texts,
            "correct": index}


def deeptumor_dev_patient(patient: str, percent: int, seed: int) -> bool:
    """A stable, seeded share of training patients held out as dev."""
    import hashlib

    return int(hashlib.sha256(f"{seed}:{patient}".encode()).hexdigest(), 16) % 100 < percent


def _deeptumor_generate(records: tuple[dict, ...]) -> Iterator[dict]:
    from functools import lru_cache
    from pathlib import Path

    @lru_cache(maxsize=1024)
    def image(file: str) -> bytes:
        return _rgb_png(Path(file).read_bytes())

    for record in records:
        row = {name: record[name] for name in ("dataset", "qid", "question", "answers",
                                               "answer_type")}
        yield {**row, "image": {"bytes": image(record["file"]), "path": None}}


def load_deeptumor(train_size: int, dev_size: int, dev_percent: int,
                   seed: int) -> dict[str, Dataset]:
    """The benchmark's test questions (all but four that offer a single option), a seeded sample
    of the training questions, and a dev sample from held-out training patients. Every
    question shows its scan's whole-volume montage (five axial slices), the only 2D view
    released for training scans. Training patients that also appear in the benchmark are
    dropped, and the released validation pool is not used (it reuses benchmark patients)."""
    import csv
    import json
    import os
    import random
    from pathlib import Path

    root = Path(os.environ.get(DEEPTUMOR_ROOT, DEEPTUMOR_ROOT_DEFAULT))
    if not (root / "train" / "images_2d" / "whole").is_dir():
        raise FileNotFoundError(f"No DeepTumorVQA download under {root}; set {DEEPTUMOR_ROOT}")
    test = []
    for line in (root / "benchmark" / "test_qa.jsonl").read_text().splitlines():
        row = json.loads(line)
        test.append(deeptumor_record(
            row["qid"], row["image_id"], row["mc_question"], row["correct_option"], row["answer"],
            row["question_type"],
            str(root / "benchmark" / "images_2d" / "whole" / f"{row['image_id']}.png")))
    if test.count(None) > len(test) // 100:
        raise ValueError(f"{test.count(None)} benchmark questions do not parse")
    test = [record for record in test if record is not None]
    test_patients = {record["patient"] for record in test}
    train, dev = [], []
    with open(root / "train" / "train_qa.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            patient = row["Image ID"]
            if patient in test_patients:
                continue
            record = deeptumor_record(
                f"train_{int(row['Question ID']):06d}", patient, row["multiple-choice question"],
                row["correct option"], row["answer"], row["question type"],
                str(root / "train" / "images_2d" / "whole" / f"{patient}.png"))
            if record is not None:
                (dev if deeptumor_dev_patient(patient, dev_percent, seed) else train).append(record)
    rng = random.Random(seed)
    picked = {"train": rng.sample(train, train_size), "dev": rng.sample(dev, dev_size)}
    splits = {"test": test}
    for name, records in picked.items():
        # Grouped by scan so each montage is encoded once; question order does not matter in
        # a pool, and dev keeps this order.
        splits[name] = sorted(records, key=lambda r: (r["patient"], r["qid"]))
    return {name: _build(_deeptumor_generate, records=tuple(records))
            for name, records in splits.items()}


# DeepTumorVQA with its option shortcuts removed (2026-10-30). The released distractors for a
# continuous value are drawn from [0.7, 1.3] times the true value, so the true value is the
# second or third smallest of four options 88% of the time, and binary and categorical answers
# follow the patients' prevalence (85% "No" for colon lesions). Here every continuous value
# gets a ladder of options with the true value at a uniformly random rank, counts a window of
# consecutive integers, every option list is shuffled, and each subtype with discrete answers
# keeps equally many questions per answer.
DEEPTUMOR_SIGNED = frozenset({"organ HU measurement", "tumor organ HU difference"})
DEEPTUMOR_PERCENT = frozenset({"tumor burden percentage"})


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def deeptumor_continuous(record: dict) -> bool:
    """A question whose options are measured values (not counts)."""
    return "count" not in record["subtype"] and all(map(_is_number, record["options"]))


def deeptumor_ladder(truth: str, k: int, subtype: str, rng) -> list[str] | None:
    """k options around a true value, in ascending order with the truth at a uniformly random
    rank; the spacing is drawn per question and does not depend on that rank, so the option set
    does not reveal it. Counts get k consecutive integers. None when some rank would leave the
    value's range (a negative volume, a percentage over 100), or the values collide."""
    value = float(truth)
    if "count" in subtype:
        if value != int(value):
            return None
        start = rng.randrange(max(0, int(value) - k + 1), int(value) + 1)
        return [str(v) for v in range(start, start + k)]
    rank = rng.randrange(k)
    signed = subtype in DEEPTUMOR_SIGNED or value <= 0
    if signed:  # HU values: additive steps of 5-15 HU, as the released options use
        step = rng.uniform(5, 15)
        values = [value + step * (i - rank) for i in range(k)]
    else:  # positive quantities: geometric steps of 8-15%
        step = rng.uniform(0.08, 0.15)
        values = [value * (1 + step) ** (i - rank) for i in range(k)]
    if (subtype in DEEPTUMOR_PERCENT and max(values) > 100) or (not signed and min(values) <= 0):
        return None
    decimals = len(truth.split(".")[1]) if "." in truth else 0
    for places in (decimals, decimals + 1, decimals + 2):  # one format for every option
        texts = [f"{v:.{places}f}" for v in values]
        texts[rank] = f"{value:.{places}f}"
        if len(set(texts)) == k:
            return texts
    return None


def deeptumor_debias(record: dict, seed: int) -> dict | None:
    """The question with regenerated numeric options and a shuffled option order (seeded per
    question); None when no ladder fits its true value."""
    import random

    rng = random.Random(f"{seed}:{record['qid']}")
    options, truth = list(record["options"]), record["options"][record["correct"]]
    if all(map(_is_number, options)):
        options = deeptumor_ladder(truth, len(options), record["subtype"], rng)
        if options is None:
            return None
        truth = next(text for text in options if float(text) == float(truth))
    rng.shuffle(options)
    index = options.index(truth)
    return {**record, "dataset": "deeptumorx", "options": options, "correct": index,
            "question": _deeptumor_prompt(record["stem"], options),
            "answers": [chr(ord("A") + index), truth]}


def deeptumor_balance(records: list[dict], min_per_answer: int, seed: int) -> list[dict]:
    """Within each subtype with discrete answers (binary, categorical, counts), the same number
    of questions per answer: answers with fewer than `min_per_answer` questions are dropped and
    the others sampled down to the rarest remaining one. Continuous-value questions pass."""
    import random
    from collections import defaultdict

    groups, kept = defaultdict(lambda: defaultdict(list)), []
    for record in records:
        if deeptumor_continuous(record):
            kept.append(record)
        else:
            groups[record["subtype"]][record["answers"][1]].append(record)
    rng = random.Random(seed)
    for subtype in sorted(groups):
        answers = {a: rs for a, rs in groups[subtype].items() if len(rs) >= min_per_answer}
        if len(answers) < 2:
            continue
        size = min(len(rs) for rs in answers.values())
        for answer in sorted(answers):
            kept.extend(rng.sample(sorted(answers[answer], key=lambda r: r["qid"]), size))
    return sorted(kept, key=lambda r: r["qid"])


def _deeptumor_gray_generate(records: tuple[dict, ...]) -> Iterator[dict]:
    """Rows with the released grayscale PNG bytes as they are (a third of the size of RGB)."""
    from functools import lru_cache
    from pathlib import Path

    @lru_cache(maxsize=1024)
    def image(file: str) -> bytes:
        return Path(file).read_bytes()

    for record in records:
        row = {name: record[name] for name in ("dataset", "qid", "question", "answers",
                                               "answer_type")}
        yield {**row, "image": {"bytes": image(record["file"]), "path": None}}


# The volume as many slices (2026-11-01): DeepTumorVQA's CT video holds 150 axial slices of
# 300 x 300 pixels; 32 of them, evenly spaced, tiled 4 x 8 at 256 pixels (2,048 vision tokens),
# against the montage's nine. The video is already lossy, so the grid is stored as JPEG.
DEEPTUMOR_GRID = (4, 8)
DEEPTUMOR_TILE = 256


def deeptumor_volume_grid(video: str, grid: tuple[int, int] = DEEPTUMOR_GRID,
                          tile: int = DEEPTUMOR_TILE) -> bytes:
    """Evenly spaced axial slices of a CT video tiled row by row, as grayscale JPEG bytes."""
    import io

    import cv2
    import numpy as np

    capture, frames = cv2.VideoCapture(video), []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame[:, :, 0])
    capture.release()
    if not frames:
        raise ValueError(f"No frames in {video}")
    rows, columns = grid
    picks = np.linspace(0, len(frames) - 1, rows * columns).round().astype(int)
    tiles = [cv2.resize(frames[i], (tile, tile), interpolation=cv2.INTER_AREA) for i in picks]
    image = np.vstack([np.hstack(tiles[r * columns:(r + 1) * columns]) for r in range(rows)])
    buffer = io.BytesIO()
    PILImage.fromarray(image, mode="L").save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def _deeptumor_grid_generate(records: tuple[dict, ...]) -> Iterator[dict]:
    from functools import lru_cache

    image = lru_cache(maxsize=1024)(deeptumor_volume_grid)
    for record in records:
        row = {name: record[name] for name in ("dataset", "qid", "question", "answers",
                                               "answer_type")}
        yield {**row, "image": {"bytes": image(record["file"]), "path": None}}


def load_deeptumorx(train_size: int, dev_size: int, dev_percent: int, seed: int,
                    min_test: int, min_train: int, view: str = "montage") -> dict[str, Dataset]:
    """DeepTumorVQA's questions with the option shortcuts removed: the same patients, roles and
    whole-volume montages as load_deeptumor, after deeptumor_debias on every question and
    deeptumor_balance within each role (test at `min_test` questions per answer, training and
    dev at `min_train`). `view="grid"` shows each scan as 32 slices of its CT video instead."""
    import csv
    import json
    import os
    import random
    from pathlib import Path

    root = Path(os.environ.get(DEEPTUMOR_ROOT, DEEPTUMOR_ROOT_DEFAULT))
    if not (root / "train" / "images_2d" / "whole").is_dir():
        raise FileNotFoundError(f"No DeepTumorVQA download under {root}; set {DEEPTUMOR_ROOT}")

    def debiased(record):
        if record is None:
            return None
        found = deeptumor_debias(record, seed)
        return None if found is None else {**found, "dataset": dataset}

    def scan(role: str, patient: str) -> str:
        if view == "grid":
            return str(root / role / "videos" / "whole" / f"{patient}.mp4")
        return str(root / role / "images_2d" / "whole" / f"{patient}.png")

    dataset = {"montage": "deeptumorx", "grid": "deeptumorv"}[view]

    test = []
    for line in (root / "benchmark" / "test_qa.jsonl").read_text().splitlines():
        row = json.loads(line)
        test.append(debiased(deeptumor_record(
            row["qid"], row["image_id"], row["mc_question"], row["correct_option"], row["answer"],
            row["question_type"], scan("benchmark", row["image_id"]),
            row["question_subtype"])))
    test_patients = {json.loads(line)["image_id"] for line in
                     (root / "benchmark" / "test_qa.jsonl").read_text().splitlines()}
    test = deeptumor_balance([r for r in test if r is not None], min_test, seed)
    train, dev = [], []
    with open(root / "train" / "train_qa.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            patient = row["Image ID"]
            if patient in test_patients:
                continue
            record = debiased(deeptumor_record(
                f"train_{int(row['Question ID']):06d}", patient, row["multiple-choice question"],
                row["correct option"], row["answer"], row["question type"],
                scan("train", patient), row["question subtype"]))
            if record is not None:
                (dev if deeptumor_dev_patient(patient, dev_percent, seed) else train).append(record)
    train = deeptumor_balance(train, min_train, seed)
    dev = deeptumor_balance(dev, min_test, seed)
    rng = random.Random(seed)
    splits = {"test": test, "train": rng.sample(train, min(train_size, len(train))),
              "dev": rng.sample(dev, min(dev_size, len(dev)))}
    splits = {name: sorted(records, key=lambda r: (r["patient"], r["qid"]))
              for name, records in splits.items()}
    generate = _deeptumor_grid_generate if view == "grid" else _deeptumor_gray_generate
    return {name: _build(generate, records=tuple(records)) for name, records in splits.items()}


def load_deeptumorff(train_size: int, dev_size: int, dev_percent: int,
                     seed: int) -> dict[str, Dataset]:
    """DeepTumorVQA's questions in free form: the released question without options and its
    answer, on the patients, roles and whole-volume montages of load_deeptumor (every benchmark
    question kept; training rows with one correct option)."""
    import csv
    import json
    import os
    import random
    from pathlib import Path

    root = Path(os.environ.get(DEEPTUMOR_ROOT, DEEPTUMOR_ROOT_DEFAULT))
    if not (root / "train" / "images_2d" / "whole").is_dir():
        raise FileNotFoundError(f"No DeepTumorVQA download under {root}; set {DEEPTUMOR_ROOT}")

    def record(qid, patient, question, answer, answer_type, file):
        return {"dataset": "deeptumorff", "qid": qid, "patient": patient, "file": file,
                "question": question.strip(), "answers": [answer.strip()],
                "answer_type": answer_type}

    test = []
    for line in (root / "benchmark" / "test_qa.jsonl").read_text().splitlines():
        row = json.loads(line)
        test.append(record(row["qid"], row["image_id"], row["question"], str(row["answer"]),
                           row["question_type"],
                           str(root / "benchmark" / "images_2d" / "whole" /
                               f"{row['image_id']}.png")))
    test_patients = {r["patient"] for r in test}
    train, dev = [], []
    with open(root / "train" / "train_qa.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            patient = row["Image ID"]
            if (patient in test_patients or len(row["correct option"].strip()) != 1
                    or not row["answer"].strip()):
                continue
            item = record(f"train_{int(row['Question ID']):06d}", patient, row["question"],
                          row["answer"], row["question type"],
                          str(root / "train" / "images_2d" / "whole" / f"{patient}.png"))
            (dev if deeptumor_dev_patient(patient, dev_percent, seed) else train).append(item)
    rng = random.Random(seed)
    splits = {"test": test, "train": rng.sample(train, train_size),
              "dev": rng.sample(dev, dev_size)}
    splits = {name: sorted(records, key=lambda r: (r["patient"], r["qid"]))
              for name, records in splits.items()}
    return {name: _build(_deeptumor_gray_generate, records=tuple(records))
            for name, records in splits.items()}


# M3D-VQA (2026-11-01), questions only: its multiple-choice questions on CT and MRI volumes
# (from Radiopaedia reports), each shown with a black image, to measure what the questions and
# options give away without the volume. A black image cannot tell volumes apart, so each one
# carries a one-pixel tag (two bytes of a hash of the volume path) that lets the held-out
# check keep test volumes out of training; the tag says nothing about the answer.
M3DVQA = "GoodBaiBai88/M3D-VQA"
M3DVQA_ROOT = "M3DVQA_ROOT"  # environment variable: the folder holding the M3D-VQA csv files
M3DVQA_ROOT_DEFAULT = "/mnt/f/VQA/raw/m3dvqa"
M3DVQA_TYPES = {"1": "plane", "2": "phase", "3": "organ", "4": "abnormality", "5": "location"}
M3DVQA_SIDE = 256


def m3dvqa_record(qid: str, row: dict) -> dict | None:
    """One question with its four lettered options; None when the answer letter or text does
    not match an option."""
    letters = "ABCD"
    texts = [row[f"Choice {letter}"].strip() for letter in letters]
    letter = row["Answer Choice"].strip().upper()
    if letter not in letters or not all(texts) or texts[letters.index(letter)] != \
            row["Answer"].strip():
        return None
    return {"dataset": "m3dvqa", "qid": qid, "volume": row["Image Path"],
            "question": _deeptumor_prompt(row["Question"].strip(), texts),
            "answers": [letter, texts[letters.index(letter)]],
            "answer_type": M3DVQA_TYPES.get(row["Question Type"], row["Question Type"]),
            "subtype": M3DVQA_TYPES.get(row["Question Type"], row["Question Type"]),
            "options": texts, "correct": letters.index(letter)}


def _tagged_black_png(volume: str) -> bytes:
    import hashlib
    import io

    tag = hashlib.sha256(volume.encode()).digest()
    image = PILImage.new("L", (M3DVQA_SIDE, M3DVQA_SIDE))
    image.putpixel((0, 0), 1 + tag[0] % 255)
    image.putpixel((1, 0), 1 + tag[1] % 255)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _m3dvqa_generate(records: tuple[dict, ...]) -> Iterator[dict]:
    for record in records:
        row = {name: record[name] for name in ("dataset", "qid", "question", "answers",
                                               "answer_type")}
        yield {**row, "image": {"bytes": _tagged_black_png(record["volume"]), "path": None}}


def load_m3dvqa(train_size: int, dev_size: int, seed: int) -> dict[str, Dataset]:
    """M3D-VQA's released splits (volumes do not overlap): all test questions, seeded samples
    of the training and validation questions."""
    import csv
    import os
    import random
    from pathlib import Path

    root = Path(os.environ.get(M3DVQA_ROOT, M3DVQA_ROOT_DEFAULT))

    def read(split):
        with open(root / f"M3D_VQA_{split}.csv", newline="") as handle:
            records = [m3dvqa_record(f"{split}_{i:06d}", row)
                       for i, row in enumerate(csv.DictReader(handle))]
        return [r for r in records if r is not None]

    rng = random.Random(seed)
    splits = {"test": read("test"), "train": rng.sample(read("train"), train_size),
              "dev": rng.sample(read("val"), dev_size)}
    return {name: _build(_m3dvqa_generate, records=tuple(sorted(records, key=lambda r: r["qid"])))
            for name, records in splits.items()}


# Public benchmarks (Milestone 2, 2026-11-06). Test-only rows in source order. Nothing is resized
# here: the evaluation protocol's pixel budget decides what the model sees. Parquet rows are read
# one batch at a time inside the generator, so a multi-gigabyte source never sits in memory or in
# the build's fingerprint.
OPTION_INSTRUCTION = "Answer with the letter of the correct option."


def _as_released(payload: bytes) -> bytes:
    """RGB and grayscale bytes stay as released: decoding reads grayscale as R = G = B, and an RGB
    PNG would add 70% to DocVQA's scans. Alpha and palettes become RGB, as other harnesses do."""
    import io

    with PILImage.open(io.BytesIO(payload)) as image:
        keep = image.mode in ("RGB", "L")
    return payload if keep else _rgb_png(payload)


def choice_answers(letter: str, options: list[str]) -> list[str]:
    """The reference letter, its option's text, then every option in letter order: the first two
    are the project's multiple-choice convention, the rest feed the rule-based option readers."""
    return [letter, options[ord(letter) - ord("A")], *options]


def _benchmark_row(dataset: str, qid: str, image: bytes, question: str, answers: list[str],
                   answer_type: str) -> dict:
    return {"dataset": dataset, "qid": str(qid),
            "image": {"bytes": _as_released(image), "path": None},
            "question": question, "answers": answers, "answer_type": answer_type}


def _images_row(dataset: str, qid: str, images: list[bytes], placeholder: bytes | None,
                question: str, answers: list[str], answer_type: str) -> dict:
    """A unified row with an `images` list; `image` keeps the first, or a placeholder when there
    are none, so split files can still key the row."""
    shown = [{"bytes": _as_released(image), "path": None} for image in images]
    first = shown[0] if shown else {"bytes": placeholder, "path": None}
    return {"dataset": dataset, "qid": str(qid), "image": first, "images": shown,
            "question": question, "answers": answers, "answer_type": answer_type}


def _snapshot(repo_id: str, patterns: list[str]):
    """The pinned revision's local folder; offline it reads the hub cache without a request."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, repo_type="dataset", allow_patterns=patterns,
                                  revision=SOURCE_REVISIONS[repo_id]))


def _parquet_generate(files: tuple[str, ...], columns: tuple[str, ...], to_row) -> Iterator[dict]:
    import pyarrow.parquet as pq

    index = 0
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=16, columns=list(columns)):
            for row in batch.to_pylist():
                yield to_row(row, index)
                index += 1


def _checked_rows(rows: Dataset, expected: int) -> Dataset:
    """A pinned source must give the rows it was registered with, each under its own ID."""
    if len(rows) != expected or len(set(rows["qid"])) != expected:
        raise ValueError(f"expected {expected} distinct rows, got {len(rows)} "
                         f"with {len(set(rows['qid']))} IDs")
    return rows


def _from_parquet(files, columns, to_row, expected: int, features=FEATURES) -> Dataset:
    gen_kwargs = {"files": tuple(str(file) for file in files), "columns": tuple(columns),
                  "to_row": to_row}
    rows = Dataset.from_generator(_parquet_generate, features=features, gen_kwargs=gen_kwargs)
    return _checked_rows(rows, expected)


_MMSTAR_INLINE = re.compile(r"(?:^|,\s)([A-D]):\s")
_MMSTAR_BLOCK = re.compile(r"(?:^|\n)\(([A-D])\)\s")


def mmstar_options(question: str) -> list[str]:
    """The option texts a question writes out, 'Options: A: x, B: y' or lines of '(A) x'. Some
    rows hold the literal option 'nan', as released."""
    parts = _MMSTAR_BLOCK.split(question)
    if len(parts) == 1:
        parts = _MMSTAR_INLINE.split(question.split("Options:")[-1].strip())
    letters, texts = parts[1::2], parts[2::2]
    if not letters or letters != [chr(ord("A") + i) for i in range(len(letters))]:
        raise ValueError(f"unreadable options: {question[-120:]!r}")
    return [text.strip() for text in texts]


def mmstar_row(row: dict, index: int) -> dict:
    question = row["question"]
    if "option letter" not in question:  # the rows copied from MathVista already say so
        question = f"{question}\n{OPTION_INSTRUCTION}"
    return _benchmark_row("mmstar", row["index"], row["image"], question,
                          choice_answers(row["answer"], mmstar_options(row["question"])),
                          row["category"])


def load_mmstar(expected: int) -> Dataset:
    """MMStar's val split, its only one, with each question as released."""
    root = _snapshot("Lin-Chen/MMStar", ["mmstar.parquet"])
    return _from_parquet([root / "mmstar.parquet"],
                         ["index", "question", "answer", "category", "image"], mmstar_row, expected)


def mathvista_row(row: dict, index: int) -> dict:
    """The authors' `query` is the question: it carries the hint, the unit and the options."""
    answer = row["answer"]
    if row["question_type"] == "multi_choice":
        letter = chr(ord("A") + row["choices"].index(answer))
        answers, kind = choice_answers(letter, row["choices"]), "multi_choice"
    else:
        answers, kind = [answer], row["answer_type"]
    if row["precision"] is not None and len(answer.split(".")[1]) != int(row["precision"]):
        raise ValueError(f"{row['pid']}: the scorer reads precision from the reference's decimals")
    return _benchmark_row("mathvista", row["pid"], row["decoded_image"]["bytes"], row["query"],
                          answers, kind)


def load_mathvista(expected: int) -> Dataset:
    root = _snapshot("AI4Math/MathVista", ["data/testmini-*.parquet"])
    return _from_parquet(sorted((root / "data").glob("testmini-*.parquet")),
                         ["pid", "decoded_image", "choices", "precision", "answer", "question_type",
                          "answer_type", "query"], mathvista_row, expected)


def mmmu_row(row: dict, index: int) -> dict:
    """Every stored image in order, `<image n>` placeholders left in the text as released. Options
    print as the authors' prompt does, and a list-shaped open answer means any of its items."""
    import ast

    images = [row[f"image_{i}"]["bytes"] for i in range(1, 8)
              if row[f"image_{i}"] and row[f"image_{i}"]["bytes"]]
    question, answer = row["question"], row["answer"]
    if row["question_type"] == "multiple-choice":
        options = [str(option) for option in ast.literal_eval(row["options"])]
        lines = "\n".join(f"({chr(ord('A') + i)}) {text}" for i, text in enumerate(options))
        question = f"{question}\n{lines}\n{OPTION_INSTRUCTION}"
        answers = choice_answers(answer, options)
    else:
        answers = ast.literal_eval(answer) if answer.startswith("[") else [answer]
    return _images_row("mmmu", row["id"], images, None, question, answers, row["question_type"])


def load_mmmu(expected: int) -> Dataset:
    """MMMU's validation split; each subject is one parquet file."""
    root = _snapshot("MMMU/MMMU", ["*/validation-*.parquet"])
    columns = ["id", "question", "options", "answer", "question_type",
               *(f"image_{i}" for i in range(1, 8))]
    return _from_parquet(sorted(root.glob("*/validation-*.parquet")), columns, mmmu_row, expected,
                         features=MULTI_IMAGE_FEATURES)


CHARXIV_ANSWER_TYPES = {1: "text-in-chart", 2: "text-in-general", 3: "number-in-chart",
                        4: "number-in-general"}


def charxiv_row(row: dict, index: int) -> dict:
    """One reasoning question per chart; the descriptive questions are left out."""
    qid = row["figure_path"].removeprefix("images/").removesuffix(".jpg")
    return _benchmark_row("charxiv", qid, row["image"]["bytes"], row["reasoning_q"],
                          [row["reasoning_a"]], CHARXIV_ANSWER_TYPES[row["reasoning_a_type"]])


def load_charxiv(expected: int) -> Dataset:
    root = _snapshot("princeton-nlp/CharXiv", ["val.parquet"])
    return _from_parquet([root / "val.parquet"], ["image", "figure_path", "reasoning_q",
                                                  "reasoning_a", "reasoning_a_type"],
                         charxiv_row, expected)


def ocrbench_row(row: dict, index: int) -> dict:
    """References stay as released: the official rule tells formulas by their trailing newline."""
    answers = list(row["answer"])
    if (row["dataset"] == "HME100k") != any(answer.endswith("\n") for answer in answers):
        raise ValueError(f"row {index}: only HME100k references may end in a newline")
    return _benchmark_row("ocrbench", f"test-{index}", row["image"]["bytes"], row["question"],
                          answers, row["question_type"])


def load_ocrbench(expected: int) -> Dataset:
    root = _snapshot("echo840/OCRBench", ["data/test-*.parquet"])
    return _from_parquet(sorted((root / "data").glob("test-*.parquet")),
                         ["dataset", "question", "question_type", "answer", "image"],
                         ocrbench_row, expected)


def docvqa_row(row: dict, index: int) -> dict:
    """A question can have several `question_types`; its answer type is the first."""
    kinds = row["question_types"] or [""]
    return _benchmark_row("docvqa", row["questionId"], row["image"]["bytes"], row["question"],
                          list(row["answers"]), kinds[0])


def load_docvqa(expected: int) -> Dataset:
    root = _snapshot("lmms-lab-encoder/DocVQA", ["DocVQA/validation-*.parquet"])
    return _from_parquet(sorted((root / "DocVQA").glob("validation-*.parquet")),
                         ["questionId", "question", "question_types", "image", "answers"],
                         docvqa_row, expected)


_VSTAR_OPTION = re.compile(r"^\(([A-Z])\)\s*(.*)$", re.MULTILINE)


def vstar_row(row: dict, image: bytes) -> dict:
    """The text carries its options. Its closing instruction ('... directly') would contradict
    the reasoning prompt, so it gives way to the project's option instruction."""
    options = [text.strip() for _, text in _VSTAR_OPTION.findall(row["text"])]
    stem_and_options = row["text"].rsplit("\n", 1)[0]
    return _benchmark_row("vstar", row["question_id"], image,
                          f"{stem_and_options}\n{OPTION_INSTRUCTION}",
                          choice_answers(row["label"], options), row["category"])


def _vstar_generate(root: str, rows: tuple[dict, ...]) -> Iterator[dict]:
    from pathlib import Path

    for row in rows:
        yield vstar_row(row, (Path(root) / row["image"]).read_bytes())


def load_vstar(expected: int) -> Dataset:
    """The rows of test_questions.jsonl; the repository's other image folders are not in it."""
    import json

    root = _snapshot("craigwu/vstar_bench", ["test_questions.jsonl", "direct_attributes/*",
                                             "relative_position/*"])
    lines = (root / "test_questions.jsonl").read_text().splitlines()
    rows = tuple(json.loads(line) for line in lines)
    return _checked_rows(_build(_vstar_generate, root=str(root), rows=rows), expected)


def hrbench_row(row: dict, index: int) -> dict:
    """One rotation of one question: the authors list each question four times with its options
    rotated. The image is the released 4K JPEG, not downscaled."""
    import base64

    options = [row[letter] for letter in "ABCD"]
    lines = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", options, strict=True))
    question = f"{row['question']}\n{lines}\n{OPTION_INSTRUCTION}"
    return _benchmark_row("hrbench4k", row["index"], base64.b64decode(row["image"]), question,
                          choice_answers(row["answer"], options), row["category"])


def load_hrbench4k(expected: int) -> Dataset:
    root = _snapshot("DreamMr/HR-Bench", ["hr_bench_4k.parquet"])
    return _from_parquet([root / "hr_bench_4k.parquet"],
                         ["index", "answer", "question", "A", "B", "C", "D", "category", "image"],
                         hrbench_row, expected)


def pope_row(row: dict, index: int) -> dict:
    return _benchmark_row("pope", row["id"], row["image"]["bytes"], row["question"],
                          [row["answer"]], row["category"])


def load_pope(expected: int) -> Dataset:
    """POPE's adversarial, popular and random sets: 3,000 questions each on the same 500 images."""
    root = _snapshot("lmms-lab-encoder/POPE", ["data/test-*.parquet"])
    return _from_parquet(sorted((root / "data").glob("test-*.parquet")),
                         ["id", "question", "answer", "category", "image"], pope_row, expected)


def text_only_placeholder() -> bytes:
    """Never shown: the split file's key for a question released without an image. No other
    registered source stores a black 16-pixel square, so it cannot match a training image."""
    import io

    buffer = io.BytesIO()
    PILImage.new("RGB", (16, 16)).save(buffer, "PNG")
    return buffer.getvalue()


def hallusion_row(row: dict, image: bytes | None, placeholder: bytes) -> dict:
    """gt_answer 1 is yes, 0 is no. Questions with `visual_input` 0 are text-only controls: they
    keep an empty `images` list, their answer type says so, and only the `published` role holds
    them, since a row without an image has no image gain to report."""
    kind = f"{row['category']} {row['subcategory']}" + ("" if image else " text-only")
    qid = "-".join(row[key] for key in ("category", "subcategory", "set_id", "figure_id",
                                        "question_id"))
    return _images_row("hallusionbench", qid, [image] if image else [], placeholder,
                       row["question"], ["yes" if row["gt_answer"] == "1" else "no"], kind)


def _hallusion_generate(files: tuple[tuple[str, str], ...],
                        rows: tuple[dict, ...]) -> Iterator[dict]:
    from pathlib import Path

    # The JSON writes .png where nine files on disk end in .PNG, which only a case-insensitive
    # file system forgives.
    paths = {name.lower(): Path(path) for name, path in files}
    placeholder = text_only_placeholder()
    for row in rows:
        name = row["filename"]
        image = paths[str(Path(name)).lower()].read_bytes() if name else None
        yield hallusion_row(row, image, placeholder)


def load_hallusionbench(expected: int) -> Dataset:
    """Every row of HallusionBench.json. The README's table (254 questions) and the sample result
    file predate the dataset's expansion; the JSON holds the current benchmark."""
    import json

    root = _snapshot("rayguan/HallusionBench", ["HallusionBench.json", "data/*"])
    rows = tuple(json.loads((root / "HallusionBench.json").read_text()))
    files = tuple((str(path.relative_to(root / "data")), str(path))
                  for path in sorted((root / "data").rglob("*")) if path.is_file())
    built = Dataset.from_generator(_hallusion_generate, features=MULTI_IMAGE_FEATURES,
                                   gen_kwargs={"files": files, "rows": rows})
    return _checked_rows(built, expected)
