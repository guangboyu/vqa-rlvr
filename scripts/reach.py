"""Reach: what Instruct can already sample, against where each method's greedy answers end.

    python -m scripts.reach     # writes results/reach.json from saved sidecars (no GPU)

Reach is Instruct's pass@8 under the row's prompt (8 is GRPO's group size), on open-answer
questions only: eight random picks among four options are right 90% of the time, and a model's
samples are not uniform, so choice and yes/no questions are left out rather than corrected. Both
sides use the same rule, full credit under the normalized scorer, which is also the training
reward. A question counts for a greedy arm when its one answer gets full credit.
"""

import json
from dataclasses import dataclass

from vqar.config import MAX_PIXELS
from vqar.eval.errors import OPEN_FORMS, reference_form
from vqar.eval.paired import paired_bootstrap
from vqar.eval.pass_at_k import FULL_CREDIT, estimate_pass_at_k, load_sampled
from vqar.eval.report import ROOT, Cell, load_cells

OUTPUT = ROOT / "results" / "reach.json"
SCORER = "normalized"
K = 8
RL_MARGIN = 5.0  # RL from Instruct ends at most this far above reach on every open-answer set
SFT_MARGIN = 5.0  # SFT ends at least this far above reach on SLAKE, at both sizes


@dataclass(frozen=True)
class Row:
    label: str
    dataset: str
    prompt: str
    base: str
    tier: str
    samples: int
    rl: str
    sft: str
    sft_rl: str | None = None


ROWS = (
    Row("SLAKE, 2B", "slake", "short", "base2", "full", 8, "slk-short2-base-e1", "slk-sft2",
        "slk-short2-sft-e1"),
    Row("SLAKE, 8B", "slake", "short", "base8", "full", 8, "slk-short8-base-e1", "slk-sft8",
        "slk-short8-sft-e1"),
    Row("ChartQA", "chartqa", "reasoning", "base2", "full", 32, "cqa-grpo2-base-e1", "cqa-sft2",
        "cqa-grpo2-sft-e1"),
    Row("ChartQA", "chartqa", "short", "base2", "full", 32, "cqa-short2-base-e1", "cqa-sft2"),
    *(Row(name, dataset, "reasoning", "base2", "first-1000", 32, "p1-f-grpo2-base", "sft2",
          "p1-f-grpo2-sft")
      for name, dataset in (("VQAv2", "vqav2"), ("GQA", "gqa"), ("TextVQA", "textvqa"))),
    *(Row(name, dataset, "short", "base2", "first-1000", 32, "p1-f-short2-base", "sft2",
          "p1-f-short2-sft")
      for name, dataset in (("VQAv2", "vqav2"), ("GQA", "gqa"), ("TextVQA", "textvqa"))),
)


def reach_by_question(cell, k: int = K) -> dict[str, float]:
    """Instruct's pass@k per open-answer question of a sampled cell."""
    return {qid: estimate_pass_at_k(cell.samples, cell.outcomes[qid][0], k)
            for qid, fields in cell.cohort.items() if reference_form(list(fields[1])) in OPEN_FORMS}


def greedy_by_question(record, qids) -> dict[str, float]:
    """1 where the arm's greedy answer gets full credit, on the given questions."""
    scores = {row["qid"]: row["scores"][SCORER] for row in record.predictions()}
    return {qid: float(scores[qid] >= FULL_CREDIT) for qid in qids}


def mean(values: dict[str, float]) -> float:
    return 100 * sum(values.values()) / len(values)


def measure(row: Row, cells: dict) -> dict:
    sampled = cells[Cell(row.base, row.dataset, "test", row.tier, row.prompt, "original", None,
                         1.0, row.samples, MAX_PIXELS)]
    cell = load_sampled(sampled.path, SCORER)
    reach = reach_by_question(cell)

    def arm(label: str) -> dict[str, float]:
        record = cells[Cell(label, row.dataset, "test", "full", row.prompt, "original", None, 0, 1,
                            MAX_PIXELS)]
        return greedy_by_question(record, reach)

    greedy = {"instruct": arm(row.base), "rl": arm(row.rl), "sft": arm(row.sft)}
    if row.sft_rl:
        greedy["sft_rl"] = arm(row.sft_rl)
    return {"label": row.label, "dataset": row.dataset, "prompt": row.prompt, "k": K,
            "questions": len(cell.outcomes), "open_questions": len(reach), "reach": mean(reach),
            "greedy": {name: mean(values) for name, values in greedy.items()},
            "rl_minus_reach": paired_bootstrap(reach, greedy["rl"]),
            "sft_minus_reach": paired_bootstrap(reach, greedy["sft"])}


def verdict(rows: list[dict]) -> dict:
    """Whether 'RL mostly sharpens, SFT adds' survives the margins fixed before this ran."""
    rl_above = {f"{r['label']} {r['prompt']}": r["greedy"]["rl"] - r["reach"] for r in rows}
    sft_above = {r["label"]: r["greedy"]["sft"] - r["reach"] for r in rows
                 if r["dataset"] == "slake"}
    return {"rl_margin": RL_MARGIN, "sft_margin": SFT_MARGIN,
            "rl_above_reach": rl_above, "sft_above_reach_on_slake": sft_above,
            "holds": (all(value <= RL_MARGIN for value in rl_above.values())
                      and all(value >= SFT_MARGIN for value in sft_above.values()))}


def main():
    cells = load_cells(ROOT / "results" / "records", ROOT / "data")
    rows = [measure(row, cells) for row in ROWS]
    report = {"scorer": SCORER, "k": K, "open_forms": OPEN_FORMS, "rows": rows,
              "verdict": verdict(rows)}
    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}; holds: {report['verdict']['holds']}")
    for r in rows:
        print(f"{r['label']:10s} {r['prompt']:9s} n={r['open_questions']:5d} "
              f"reach={r['reach']:5.1f} "
              + " ".join(f"{name}={value:5.1f}" for name, value in r["greedy"].items()))


if __name__ == "__main__":
    main()
