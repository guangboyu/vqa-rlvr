"""Render markdown results tables from results/runs/*.json.

README numbers are pasted from this output — never hand-typed.

Usage: uv run python scripts/make_tables.py
"""

import json
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).parent.parent / "results" / "runs"
DATASETS = ["vqav2", "gqa", "clevr", "textvqa"]
HEADERS = {
    "vqav2": "VQAv2-val (VQA acc)",
    "gqa": "GQA-testdev (EM)",
    "clevr": "CLEVR-test (EM)",
    "textvqa": "TextVQA-val (OOD)",
}


def main() -> None:
    rows: dict[str, dict[str, float]] = defaultdict(dict)
    for path in sorted(RUNS.glob("*.json")):
        run = json.loads(path.read_text())
        dataset = run["eval"]["dataset"]
        model_id = run["run_id"].removesuffix(f"-{dataset}")
        rows[model_id][dataset] = run["metrics"]["overall"]

    present = [d for d in DATASETS if any(d in v for v in rows.values())]
    print("| Model | " + " | ".join(HEADERS[d] for d in present) + " |")
    print("|---" * (len(present) + 1) + "|")
    for model_id, scores in rows.items():
        cells = [f"{scores[d] * 100:.1f}" if d in scores else "—" for d in present]
        print(f"| {model_id} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
