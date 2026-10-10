"""Rescore the headline records' exact-match misses with the LLM judge (vqar.judge).

    python -m scripts.judge_records --dry-run    # count misses, estimate the cost
    python -m scripts.judge_records              # judge, then write results/judge-final.json
    python -m scripts.judge_records --pilot chartqa  # writes results/judge-chartqa.json

A miss is a non-empty prediction the official scorer gives 0: no match on GQA and CLEVR, no
annotator match on VQAv2 and TextVQA. The judge sees the question, the references and the
extracted answer, never the reasoning text or the image, so it measures answer equivalence,
not visual correctness. Judge-adjusted accuracy gives a rescued miss full credit: exact for
GQA and CLEVR, an upper bound for the ten-annotator datasets. Partial-credit answers are not
judged. A seeded sample of exact hits (score 1) per cell measures how often the judge rejects
an answer the official scorer accepts. Once scripts.audit_judge has measured how often an
image-aware auditor confirms the judge's acceptances (by dataset and answer length), the audited
accuracy counts each accepted miss with that confirmation rate. Misses the judge rejected stay
at 0, although the audit finds some of them correct, so the audited accuracy remains a
conservative estimate.
"""

import argparse
import json
import random
from pathlib import Path

from vqar.eval.matrix import declared_records, load_comparisons, load_matrix, load_pilot
from vqar.eval.paired import paired_bootstrap
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR, RECORDS_DIR
from vqar.judge import Judge

OUTPUT = ROOT / "results" / "judge-final.json"
MATRIX = "headline"
CONTROLS_PER_CELL = 100


def read_cell(record) -> list[dict]:
    """One greedy prediction per question: qid, question, references, answer, official score."""
    rows = [json.loads(line) for line in record.predictions_path().read_text().splitlines()]
    return [{key: row[key] for key in ("qid", "question", "answers", "prediction")}
            | {"official": row["scores"]["official"]} for row in rows]


def is_miss(row: dict) -> bool:
    return row["official"] == 0 and bool(row["prediction"].strip())


def misses(rows: list[dict]) -> list[dict]:
    return [row for row in rows if is_miss(row)]


def control_sample(rows: list[dict], seed: str) -> list[dict]:
    hits = [row for row in rows if row["official"] >= 1.0 - 1e-12]
    return random.Random(seed).sample(hits, min(CONTROLS_PER_CELL, len(hits)))


def item(row: dict) -> tuple[str, list[str], str]:
    return row["question"], row["answers"], row["prediction"]


def key_of(entry: tuple[str, list[str], str]) -> tuple:
    question, answers, prediction = entry
    return question, tuple(answers), prediction


def length_bucket(prediction: str) -> str:
    words = len(prediction.split())
    return "1 word" if words == 1 else "2-5 words" if words <= 5 else "6+ words"


def stratum(dataset: str, accepted: bool, prediction: str) -> str:
    """Audit strata: accepted misses by dataset and answer length, rejected misses by dataset."""
    return (f"{dataset}|accepted|{length_bucket(prediction)}" if accepted
            else f"{dataset}|rejected")


def audit_path(pilot: str | None) -> Path:
    """scripts.audit_judge's rates: the headline audit, or a pilot's own."""
    return ROOT / "results" / (f"judge-audit-{pilot}.json" if pilot else "judge-audit.json")


def audit_rates(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    return {name: rate["rate"] for name, rate in json.loads(path.read_text())["rates"].items()}


def judged_cells(pilot: str | None):
    """The arms, protocols and (reference, candidate) pairs to rescore: a pilot's, or the
    headline matrix's."""
    matrix_path = ROOT / "experiments.toml"
    if pilot:
        return load_pilot(matrix_path, pilot)
    arms, protocols = load_matrix(matrix_path, MATRIX)
    return arms, protocols, load_comparisons(matrix_path, MATRIX)


def scores(scorer: str, rows: list[dict], verdict, dataset: str, rates) -> dict[str, float]:
    """Each question's credit: official; judge-adjusted (an accepted miss gets full credit); or
    audited (an accepted miss gets the rate at which the auditor confirms its stratum)."""
    def credit(row):
        if scorer == "official" or not is_miss(row) or not verdict(row):
            return row["official"]
        if scorer == "adjusted":
            return 1.0
        return rates[stratum(dataset, True, row["prediction"])]
    return {row["qid"]: credit(row) for row in rows}


def scorers_for(dataset: str, rates) -> tuple[str, ...]:
    audited = rates is not None and any(name.startswith(f"{dataset}|") for name in rates)
    return ("official", "adjusted", "audited") if audited else ("official", "adjusted")


def summarize_cell(rows: list[dict], verdict, controls: list[dict], dataset: str,
                   rates) -> dict:
    missed = misses(rows)
    judged = [row for row in missed if verdict(row) is not None]
    rescued = sum(bool(verdict(row)) for row in judged)
    control_verdicts = [verdict(row) for row in controls if verdict(row) is not None]
    accuracy = {scorer: 100 * sum(scores(scorer, rows, verdict, dataset, rates).values())
                / len(rows) for scorer in scorers_for(dataset, rates)}
    return {
        "n": len(rows),
        **accuracy,
        "misses": len(missed),
        "judged": len(judged),
        "unjudged": len(missed) - len(judged),
        "rescued": rescued,
        "rescue_rate": rescued / len(judged) if judged else None,
        "controls": len(control_verdicts),
        "control_agreement": (sum(control_verdicts) / len(control_verdicts)
                              if control_verdicts else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="count and estimate only")
    parser.add_argument("--pilot", default=None,
                        help="rescore a single-dataset pilot's cells ([pilots.NAME])")
    parser.add_argument("--output", default=None,
                        help="default: results/judge-final.json, or judge-<pilot>.json")
    args = parser.parse_args()
    output = args.output or (ROOT / "results" / f"judge-{args.pilot}.json" if args.pilot
                             else OUTPUT)

    arms, protocols, comparisons = judged_cells(args.pilot)
    records = declared_records(arms, protocols, RECORDS_DIR, DATA_DIR, ROOT)
    cells = {cell: read_cell(record) for cell, record in records.items()
             if record.metrics["official"] is not None}
    controls = {cell: control_sample(rows, f"{cell[0]}|{cell[1].slug()}")
                for cell, rows in cells.items()}
    items = [item(row) for rows in cells.values() for row in misses(rows)]
    items += [item(row) for rows in controls.values() for row in rows]

    judge = Judge(cache_path=ROOT / ".judge_cache" / "judge.db")
    pending = judge.uncached(items)
    print(f"{len(cells)} cells, {len(items)} items to judge, {len(pending)} distinct ones not "
          f"cached; projected cost ${judge.estimate_batch_cost(pending):.2f} "
          f"(spent so far ${judge.spend():.2f}, cap ${judge.budget_usd:.2f})")
    if args.dry_run:
        return

    found = dict(zip(map(key_of, items), judge.judge_many(items), strict=True))

    def verdict(row):
        return found[key_of(item(row))]

    rates = audit_rates(audit_path(args.pilot))
    names = {arm.label: arm.name for arm in arms}
    summary = [{"arm": label, "name": names[label], "dataset": protocol.dataset,
                "prompt": protocol.prompt,
                **summarize_cell(rows, verdict, controls[(label, protocol)], protocol.dataset,
                                 rates)}
               for (label, protocol), rows in cells.items()]
    deltas = []
    for reference, candidate in comparisons:
        for protocol in protocols:
            pair = cells.get((reference, protocol)), cells.get((candidate, protocol))
            if None in pair:
                continue
            for scorer in scorers_for(protocol.dataset, rates):
                paired = [scores(scorer, rows, verdict, protocol.dataset, rates)
                          for rows in pair]
                deltas.append({"reference": reference, "candidate": candidate,
                               "dataset": protocol.dataset, "prompt": protocol.prompt,
                               "scorer": scorer, **paired_bootstrap(*paired)})
    report = {
        "judge_model": "Haiku 4.5, pinned as vqar.judge.MODEL",
        "miss": "non-empty prediction with official score 0",
        "adjusted": "rescued misses get full credit (upper bound on VQAv2/TextVQA)",
        "audited": None if rates is None else
                   "each accepted miss counts with the auditor's confirmation rate for its "
                   "dataset and answer length; rejected misses stay at 0; intervals do not "
                   "include the audit's own sampling error",
        "spend_usd": judge.spend(),
        "cells": summary,
        "paired_deltas": deltas,
    }
    with open(output, "w") as handle:
        json.dump(report, handle, indent=1)
        handle.write("\n")
    print(f"wrote {output}: {len(summary)} cells, {len(deltas)} paired deltas, "
          f"total judge spend ${judge.spend():.2f}")


if __name__ == "__main__":
    main()
