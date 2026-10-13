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

