"""Rebuild the demo's recorded gallery from saved predictions: python -m scripts.make_gallery

Needs the corrected campaign's predictions and the VQAv2 test subset; without them the
recorded gallery stays as it is."""

import argparse
import hashlib
import json
from pathlib import Path

from vqar.inference import extract_answer
from vqar.registry import score as dataset_score

ROOT = Path(__file__).resolve().parent.parent


def intervention_example(output: Path):
    """One recorded contrast: changed answers need not be grounded answers.

    Its probe images and responses live in the research archive; without them, the card
    already recorded in the gallery is kept unchanged.
    """
    directory = ROOT / "idea-stage/battery/swap_probes"
    source = ROOT / "idea-stage/battery/audit/raw/battery_Qwen3-VL-2B-Instruct.json"
    if not source.exists():
        recorded = output / "gallery.json"
        cards = json.loads(recorded.read_text()) if recorded.exists() else []
        kept = next((c for c in cards if c["title"] == "Image sensitivity is not enough"), None)
        print("Intervention card: research archive absent; "
              + ("keeping the recorded card." if kept else "no recorded card to keep."))
        return kept
    records = [json.loads(line) for line in (directory / "probes.jsonl").read_text().splitlines()]
    records = [r for r in records if r["qid"] == "201030621"]
    raw = json.loads(source.read_text())["components"]["swap_probes"]["raw"]
    responses = []
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original = next(r for r in records if r["condition"] == "original")
    for record in records:
        image = directory / record["image"]
        name = f"intervention_{record['condition']}{image.suffix}"
        (output / name).write_bytes(image.read_bytes())
        responses.append(
            {
                "label": f"Base 2B, {record['condition']} image",
                "run_id": record["probe_id"],
                "raw": raw[record["probe_id"]],
                "image": f"/examples/{name}",
                "official": None,
                "normalized": None,
                "source_sha256": source_hash,
            }
        )
    return {
        "title": "Image sensitivity is not enough",
        "lesson": "The answer changes with the image, but the model still answers a question "
        "about a boy when shown flowers or a black image. Sensitivity alone does "
        "not establish correct grounding or isolate the image encoder.",
        "question": original["question"],
        "qid": original["qid"],
        "dataset": "GQA",
        "image": f"/examples/intervention_original{Path(original['image']).suffix}",
        "answers": original["answers"],
        "reference_label": "Reference for the original image",
        "responses": responses,
    }


def make_gallery():
    from datasets import Image, load_from_disk

    names = ["zero_shot_2b_reasoning", "sft_2b_reasoning", "grpo_2b_fmt0_fixedreward_reasoning"]
    paths = [ROOT / "results/preds" / f"{name}-vqav2.jsonl" for name in names]
    labels = ["Base 2B", "Short-answer SFT", "Correctness-only GRPO"]
    arms = ["base2", "sft2", "grpo2-base", "grpo2-sft"]
    fresh = [ROOT / "results/baselines/evaluation" / arm / "preds" /
             f"{arm}-reasoning-vqav2.jsonl" for arm in arms]
    current = all(path.exists() for path in fresh)
    if current:
        paths = fresh
        names = [f"{arm}-reasoning" for arm in arms]
        labels = ["Instruct 2B", "Short-answer SFT", "Correctness-only GRPO", "SFT then GRPO"]
    if not all(path.exists() for path in paths):
        print("Gallery unchanged: its saved predictions are unavailable.")
        return
    rows = [{r["qid"]: r for r in map(json.loads, path.read_text().splitlines())} for path in paths]

    def score(row, variant="official"):
        return dataset_score("vqav2", extract_answer(row["raw"]), row["answers"], variant)

    cases = [
        (
            "One answer, two scores",
            "The original VQA scorer skips normalization when annotators agree. "
            "Capitalization can then change the score.",
            lambda a, b, c, d=None: (
                score(a) == 0
                and score(a, "normalized") == 1
                and extract_answer(a["raw"]) in ("Yes", "No")
            ),
            [0],
        ),
        (
            "SFT changes the visible response",
            "Both answers receive full credit here. Short-answer SFT produces less explanation; "
            "this does not prove internal reasoning was erased.",
            lambda a, b, c, d=None: (
                len(a["raw"]) > 300 and len(b["raw"]) < 50 and score(a) == score(b) == 1
            ),
            [0, 1],
        ),
        (
            "A long explanation can be wrong",
            "The base model explains its answer but misses the reference. "
            "The GRPO response is shorter and correct on this question.",
            lambda a, b, c, d=None: (
                300 < len(a["raw"]) < 1200
                and score(a, "normalized") == 0
                and score(c) == 1
                and len(c["raw"]) < 100
            ),
            [0, 2],
        ),
        (
            "Shorter is not always better",
            "This counterexample goes the other way: the longer base response is correct "
            "and the short GRPO response is wrong.",
            lambda a, b, c, d=None: (
                300 < len(a["raw"]) < 1200
                and score(a) == 1
                and score(c, "normalized") == 0
                and extract_answer(c["raw"]).lower() in ("yes", "no")
                and len(c["raw"]) < 100
            ),
            [0, 2],
        ),
    ]
    if current:
        cases.extend([
            (
                "An improvement after SFT then RL",
                "On this question, SFT gives a short wrong answer; subsequent GRPO produces "
                "a longer response with a correct final answer. This example does not establish "
                "that the explanation is faithful or that the same change holds generally.",
                lambda a, b, c, d: (
                    score(b, "normalized") == 0 and score(d) == 1
                    and max(100, 2 * len(b["raw"])) < len(d["raw"]) < 1200
                ),
                [1, 3],
            ),
            (
                "RL can also lose a correct answer",
                "The SFT answer receives full credit here, but the SFT-then-GRPO answer "
                "is wrong under both scoring variants. Aggregate gains can include regressions.",
                lambda a, b, c, d: score(b) == 1 and score(d, "normalized") == 0,
                [1, 3],
            ),
        ])
    dataset = load_from_disk(str(ROOT / "data/vqav2_eval")).cast_column(
        "image", Image(decode=False)
    )
    indices = {qid: i for i, qid in enumerate(dataset["qid"])}
    output = ROOT / "demo/examples"
    output.mkdir(exist_ok=True)
    examples, used = [], set()
    for title, lesson, condition, selected in cases:
        qid = next((
            qid
            for qid in rows[0]
            if qid not in used
            and all(qid in group for group in rows)
            and condition(*(group[qid] for group in rows))
        ), None)
        if qid is None:
            print(f"No saved example meets the condition: {title}")
            continue
        used.add(qid)
        source = dataset[indices[qid]]
        payload = source["image"]["bytes"]
        if payload is None:
            payload = Path(source["image"]["path"]).read_bytes()
        suffix = ".png" if payload.startswith(b"\x89PNG") else ".jpg"
        filename = qid + suffix
        (output / filename).write_bytes(payload)
        responses = [
            dict(
                label=labels[i],
                run_id=names[i],
                raw=rows[i][qid]["raw"],
                official=score(rows[i][qid]),
                normalized=score(rows[i][qid], "normalized"),
                source_sha256=hashlib.sha256(paths[i].read_bytes()).hexdigest(),
            )
            for i in selected
        ]
        examples.append(
            dict(
                title=title,
                lesson=lesson,
                question=source["question"],
                qid=qid,
                image=f"/examples/{filename}",
                answers=source["answers"],
                responses=responses,
                study="Corrected shared-pool comparison" if current else "Historical experiments",
            )
        )
    intervention = intervention_example(output)
    if intervention is not None:
        intervention["study"] = "Historical public-checkpoint image audit"
        examples.append(intervention)
    (output / "gallery.json").write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Gallery: {len(examples)} examples copied from recorded responses")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    make_gallery()


if __name__ == "__main__":
    main()
