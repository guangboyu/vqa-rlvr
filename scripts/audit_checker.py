"""Audit a training source's reward: how often it rejects right answers or accepts wrong ones.

    python -m scripts.audit_checker --arm base2 --sources geoqa mmk12 --dry-run   # draw, estimate
    python -m scripts.audit_checker --arm base2 --sources geoqa mmk12             # pays for a batch

Screening measures how often a reward fires, not whether it is right. This reads a source's sampled
reasoning record from the screening battery (scripts.screen_datasets), on the rl role, and draws up
to 50 responses the reward marks right and 50 it marks wrong: seeded, one per question on each side,
split across reference forms in proportion to their questions. The auditor (vqar.judge.AUDIT_MODEL)
sees the image, the question, the references and the answer the reward read, and says whether the
answer is right.

The false-reject rate is the share of audited reward-wrong answers the auditor accepts; the
false-accept rate is the share of audited reward-right answers it rejects. Each is conditional on
the reward's own mark, with a Wilson 95% interval; the record's right and wrong counts turn them
into shares of all responses. Responses with no readable answer leave the wrong side: the reward
pays nothing for them, but there is no answer to audit. They are counted, as are partial rewards.

The cost bound prints before any call, verdicts and spend go to the judge's ledger (a repeat costs
nothing), and a batch that would pass the budget cap (vqar.judge) is refused.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.audit_judge import (
    PROMPT,
    batch_cost,
    cost_bound,
    image_payloads,
    request,
    run_batch,
    wilson,
)
from vqar.answers import reference_form, reward_answer
from vqar.benchmark_metrics import is_choice
from vqar.eval import RunRecord, find_cell
from vqar.eval.matrix import load_matrix
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR, RECORDS_DIR
from vqar.eval.screening import battery
from vqar.judge import AUDIT_MODEL, Judge
from vqar.registry import PROMPTS
from vqar.rewards import correctness_reward

OUTPUT = ROOT / "results" / "checker-audit.json"
PER_SIDE = 50
EXAMPLES = 5  # disagreements kept per side
MAX_SIDE = 1568  # the API's largest unscaled side, so a document page's small print stays readable
INPUT_TOKEN_BOUND = 3_600  # such an image is at most ~3,300 tokens, plus the text
SIDES = ("right", "wrong")


def side_of(answer: str | None, reward: float) -> str:
    if not answer:
        return "no answer"
    if reward >= 1.0 - 1e-12:
        return "right"
    return "wrong" if reward == 0 else "partial"


def shown_references(answers: list[str]) -> list[str]:
    """What the auditor reads as references: a row with options keeps the letter and its text, not
    the option list stored after them, which would read as every option being accepted."""
    return answers[:2] if is_choice(answers) else answers


def read_responses(lines: list[dict], dataset: str, prompt: str) -> list[dict]:
    """Every sampled response as the reward reads it: the answer, the reward and its side."""
    rule = PROMPTS[prompt].answer_rule
    rewards = correctness_reward(
        prompts=None, completions=[line["raw"] for line in lines],
        answers=[line["answers"] for line in lines], dataset=[dataset] * len(lines),
        rollout_prompt=[prompt] * len(lines))
    responses = []
    for line, reward in zip(lines, rewards, strict=True):
        answer = reward_answer(line["raw"], rule)
        responses.append({"dataset": dataset, "qid": line["qid"], "sample": line["sample"],
                          "question": line["question"], "prediction": answer,
                          "answers": shown_references(line["answers"]), "reward": reward,
                          "form": reference_form(line["answers"]),
                          "side": side_of(answer, reward)})
    return responses


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Up to `total` draws across forms: one per form, largest first, while there is room; each of
    the rest to the largest size / (2 * draws + 1), proportional in whole draws (Sainte-Lague)."""
    counts = dict.fromkeys(sizes, 0)
    for form in sorted(sizes, key=sizes.get, reverse=True)[:total]:
        counts[form] = 1
    for _ in range(min(total, sum(sizes.values())) - sum(counts.values())):
        open_forms = [form for form in sorted(sizes) if counts[form] < sizes[form]]
        form = max(open_forms, key=lambda name: sizes[name] / (2 * counts[name] + 1))
        counts[form] += 1
    return counts


def draw_side(responses: list[dict], side: str, per_side: int, seed: int) -> list[dict]:
    """A seeded sample of one side: one response per question, stratified by reference form."""
    by_form = {}
    for row in sorted(responses, key=lambda row: (row["qid"], row["sample"])):
        if row["side"] == side:
            by_form.setdefault(row["form"], {}).setdefault(row["qid"], []).append(row)
    counts = allocate({form: len(questions) for form, questions in by_form.items()}, per_side)
    chosen = []
    for form in sorted(by_form):
        rng = random.Random(f"{seed}|{responses[0]['dataset']}|{side}|{form}")
        for qid in rng.sample(sorted(by_form[form]), counts[form]):
            chosen.append(rng.choice(by_form[form][qid]))
    return chosen


def ledger_key(row: dict) -> str:
    """One auditor's verdict on one answer to one question, in the judge's ledger."""
    parts = ["checker-audit", AUDIT_MODEL, row["dataset"], row["qid"], row["prediction"]]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def audit_rows(sample: list[dict], judge: Judge, role: str, data_dir: Path) -> None:
    """Fill each row's `auditor` verdict and `cost_usd`: from the ledger, else from one batch. A
    request the API fails leaves its row unaudited, never a silent no."""
    for row in sample:
        if (entry := judge.ledger_entry(ledger_key(row))) is not None:
            row["auditor"], row["cost_usd"] = entry
    pending = [row for row in sample if "auditor" not in row]
    if not pending:
        return
    judge.check_budget(cost_bound(len(pending), INPUT_TOKEN_BOUND))
    images = image_payloads(pending, role, data_dir, MAX_SIDE)
    requests = [request(i, row, images[row["dataset"], row["qid"]])
                for i, row in enumerate(pending)]
    answers = run_batch(judge.client, requests)
    for i, row in enumerate(pending):
        if (answer := answers.get(f"audit-{i}")) is not None:
            row["auditor"] = answer["verdict"]
            row["cost_usd"] = batch_cost(answer["input_tokens"], answer["output_tokens"])
            judge.record_in_ledger(ledger_key(row), row["auditor"], row["cost_usd"])


def disagrees(row: dict) -> bool:
    """The auditor and the reward differ: a right mark rejected, or a wrong mark accepted."""
    return row["auditor"] != (row["side"] == "right")


def rate(rows: list[dict]) -> dict:
    """How often the auditor overrules this side's mark, with a Wilson 95% interval."""
    audited = [row for row in rows if "auditor" in row]
    overruled = sum(disagrees(row) for row in audited)
    return {"drawn": len(rows), "audited": len(audited), "overruled": overruled,
            "rate": overruled / len(audited) if audited else None,
            "interval": wilson(overruled, len(audited))}


def example(row: dict) -> dict:
    return {key: row[key] for key in ("qid", "sample", "form", "question", "answers",
                                      "prediction", "reward", "auditor")}


def disagreement_examples(rows: list[dict]) -> list[dict]:
    return [example(row) for row in rows if "auditor" in row and disagrees(row)][:EXAMPLES]


def summarize(sample: list[dict], responses: list[dict], record: RunRecord) -> dict:
    """One source's row of results/checker-audit.json."""
    sides = {side: [row for row in sample if row["side"] == side] for side in SIDES}
    return {
        "record": record.path.name, "protocol_id": record.protocol_id,
        "model": record.model["label"],
        "responses": {name: sum(row["side"] == name for row in responses)
                      for name in (*SIDES, "partial", "no answer")},
        "false_reject": rate(sides["wrong"]), "false_accept": rate(sides["right"]),
        "by_form": {side: {form: rate([row for row in rows if row["form"] == form])
                           for form in sorted({row["form"] for row in rows})}
                    for side, rows in sides.items()},
        "cost_usd": sum(row.get("cost_usd", 0.0) for row in sample),
        "disagreements": {side: disagreement_examples(rows) for side, rows in sides.items()},
        "items": [{key: row.get(key) for key in ("qid", "sample", "form", "side", "prediction",
                                                 "auditor")} for row in sample],
    }


def audit_source(record: RunRecord, judge: Judge, *, per_side: int = PER_SIDE, seed: int = 0,
                 data_dir: Path = DATA_DIR, dry_run: bool = False) -> dict | None:
    """Draw one source's sample from its sampled record, print the bound, then audit it."""
    protocol = record.protocol
    dataset, prompt = protocol["dataset"], protocol["prompt"]
    responses = read_responses(record.predictions(), dataset, prompt)
    sample = [row for side in SIDES for row in draw_side(responses, side, per_side, seed)]
    pending = sum(judge.ledger_entry(ledger_key(row)) is None for row in sample)
    print(f"{dataset}: {len(sample)} responses drawn, {pending} not yet audited; cost at most "
          f"about ${cost_bound(pending, INPUT_TOKEN_BOUND):.2f} (spent so far "
          f"${judge.spend():.2f}, cap ${judge.budget_usd:.2f})", flush=True)
    if dry_run:
        return None
    audit_rows(sample, judge, protocol["role"], data_dir)
    return summarize(sample, responses, record)


def write_report(rows: dict[str, dict], per_side: int, seed: int, output: Path) -> None:
    """Merge into the existing file, so auditing one source updates only its row."""
    sources = json.loads(output.read_text())["sources"] if output.exists() else {}
    report = {"auditor": "vqar.judge.AUDIT_MODEL", "effort": "low", "per_side": per_side,
              "seed": seed, "prompt": PROMPT,
              "false_reject": "share of audited reward-wrong answers the auditor accepts",
              "false_accept": "share of audited reward-right answers the auditor rejects",
              "sources": {**sources, **rows}}
    output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {output}")


def sampled_record(dataset: str, model, data_dir: Path, records_dir: Path) -> RunRecord | None:
    protocol = battery(dataset).sampled
    if protocol.role != "rl":
        raise ValueError(f"{dataset}: its sampled record reads {protocol.role}, not the rl role")
    cell = find_cell(model, protocol, data_dir=data_dir, records_dir=records_dir)
    if not cell.cached:
        print(f"{dataset}: no sampled record yet, expected at {cell.path}; run "
              "scripts.screen_datasets first", flush=True)
        return None
    return RunRecord.load(cell.path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--arm", default="base2", help="whose sampled records to read")
    parser.add_argument("--dry-run", action="store_true", help="draw and estimate; no API call")
    parser.add_argument("--per-side", type=int, default=PER_SIDE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--matrix", type=Path, default=ROOT / "experiments.toml")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".judge_cache" / "judge.db",
                        help="the judge's verdicts and spend, which the budget cap reads")
    args = parser.parse_args()

    model = {arm.label: arm for arm in load_matrix(args.matrix)[0]}[args.arm].ref(ROOT).pinned()
    judge = Judge(cache_path=args.ledger)
    rows = {}
    for dataset in args.sources:
        record = sampled_record(dataset, model, args.data_dir, args.records_dir)
        if record is None:
            continue
        row = audit_source(record, judge, per_side=args.per_side, seed=args.seed,
                           data_dir=args.data_dir, dry_run=args.dry_run)
        if row is not None:
            rows[dataset] = row
            print(f"{dataset}: false reject {row['false_reject']['rate']}, false accept "
                  f"{row['false_accept']['rate']}, ${row['cost_usd']:.2f}", flush=True)
    if rows:
        write_report(rows, args.per_side, args.seed, args.out)


if __name__ == "__main__":
    main()
