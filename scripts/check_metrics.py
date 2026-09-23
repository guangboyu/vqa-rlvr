"""Compare saved predictions with pinned dataset-author accuracy implementations.

Usage: python -m scripts.check_metrics --sources-dir path/to/evaluators
The directory must contain vqav2.py, textvqa.py, and GQA's eval.py. See docs/validation.md.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

from vqar.inference import extract_answer
from vqar.registry import score


def author_evaluators(directory: Path):
    namespace = {}
    # The VQA release uses Python 2 print statements; its scoring code is unchanged.
    source = (directory / "vqav2.py").read_text()
    source = re.sub(r"(?m)^(\s*)print (.+)$", r"\1print(\2)", source)
    exec(compile(source, "official/vqav2.py", "exec"), namespace)
    vqa_class = namespace["VQAEval"]

    module = ast.parse((directory / "textvqa.py").read_text())
    module.body = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name in ("EvalAIAnswerProcessor", "TextVQAAccuracyEvaluator")
    ]
    namespace = {"re": re}
    exec(compile(module, "official/textvqa.py", "exec"), namespace)
    textvqa = namespace["TextVQAAccuracyEvaluator"]()

    # GQA's other metrics require scene graphs, answer choices, and attention maps.
    module = ast.parse((directory / "eval.py").read_text())
    loop = next(
        node
        for node in module.body
        if isinstance(node, ast.For) and ast.unparse(node.target) == "(qid, question)"
    )
    accuracy = ast.Module(body=loop.body[:4], type_ignores=[])
    gqa_code = compile(accuracy, "official/gqa-accuracy.py", "exec")

    def vqav2(records):
        qa = {
            i: {
                "answers": [{"answer_id": j + 1, "answer": a} for j, a in enumerate(r["answers"])],
                "question_type": "all",
                "answer_type": "all",
            }
            for i, r in enumerate(records)
        }
        results = {i: {"answer": r["prediction"]} for i, r in enumerate(records)}
        evaluator = vqa_class(
            SimpleNamespace(qa=qa, getQuesIds=lambda: list(qa)),
            SimpleNamespace(qa=results),
            n=10,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
        return [evaluator.evalQA[i] / 100 for i in range(len(records))]

    def textvqa_scores(records):
        return [
            textvqa.eval_pred_list([{"pred_answer": r["prediction"], "gt_answers": r["answers"]}])
            for r in records
        ]

    def gqa(records):
        scores = []
        for r in records:
            values = {
                "question": {"answer": r["answers"][0]},
                "qid": r["qid"],
                "predictions": {r["qid"]: r["prediction"]},
                "toScore": float,
            }
            exec(gqa_code, values)
            scores.append(values["score"])
        return scores

    # Datasets whose authors published an evaluator we can run; CLEVR has none.
    return {"vqav2": vqav2, "textvqa": textvqa_scores, "gqa": gqa}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-dir", type=Path, required=True)
    parser.add_argument("--preds-dir", type=Path, default=Path("results/preds"))
    parser.add_argument("--output", type=Path, default=Path("results/metric_parity.json"))
    args = parser.parse_args()
    evaluators = author_evaluators(args.sources_dir)
    runs = []
    for path in sorted(args.preds_dir.glob("*.jsonl")):
        dataset = path.stem.rsplit("-", 1)[-1]
        if dataset not in evaluators:
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            record["prediction"] = extract_answer(record["raw"])
        expected = evaluators[dataset](records)
        actual = [score(dataset, r["prediction"], r["answers"], "official") for r in records]
        mismatches = sum(abs(a - b) > 1e-10 for a, b in zip(actual, expected, strict=True))
        runs.append(
            dict(
                run_id=path.stem,
                n=len(records),
                mismatches=mismatches,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        print(f"{path.stem}: {mismatches} mismatches / {len(records)}", flush=True)
    result = {
        "scope": "VQAv2/TextVQA accuracy; GQA accuracy expression only",
        "runs": runs,
        "evaluator_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.sources_dir.glob("*.py")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not runs or any(run["mismatches"] for run in runs):
        raise SystemExit("Scorer verification failed; inspect the report.")


if __name__ == "__main__":
    main()
