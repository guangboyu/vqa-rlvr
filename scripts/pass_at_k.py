"""pass@k curves and paired before/after differences from saved sampled records.

    python -m scripts.pass_at_k --declared heldout chartqa   # [passk.<name>] in experiments.toml
    python -m scripts.pass_at_k --compare BEFORE.json AFTER.json --output report.json

No inference runs: records come from `evaluate.py --temperature 1 --samples N`. Intervals
resample images (questions on one image are not independent) and cover which questions were
asked, not training seeds or a fresh draw of samples.
"""

import argparse
import json
import tomllib
from pathlib import Path

from vqar.eval.matrix import declared_records, load_matrix
from vqar.eval.pass_at_k import VARIANTS, default_ks, load_sampled, paired_pass_at_k
from vqar.eval.protocol import Protocol
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR, RECORDS_DIR


def declared_output(name: str) -> Path:
    return ROOT / "results" / f"passk-{name}.json"


def declared_pairs(name: str) -> list[tuple[Path, Path]]:
    """(reference, candidate) record paths for every comparison in [passk.<name>]."""
    matrix = ROOT / "experiments.toml"
    config = tomllib.loads(matrix.read_text())["passk"][name]
    protocols = {(dataset, entry["prompt"]): Protocol(
                     dataset, prompt=entry["prompt"], tier=config["tier"],
                     temperature=config["temperature"], samples=config["samples"],
                     seed=config["seed"])
                 for entry in config["comparisons"] for dataset in config["datasets"]}
    arms, _ = load_matrix(matrix)
    records = declared_records(arms, list(protocols.values()), RECORDS_DIR, DATA_DIR, ROOT)
    pairs = []
    for entry in config["comparisons"]:
        for candidate in entry["candidates"]:
            for dataset in config["datasets"]:
                protocol = protocols[(dataset, entry["prompt"])]
                pair = records.get((entry["reference"], protocol)), records.get((candidate,
                                                                                protocol))
                if None in pair:
                    print(f"skipped {candidate} vs {entry['reference']} {protocol.slug()}: "
                          "no record yet")
                    continue
                pairs.append((pair[0].path, pair[1].path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", nargs="*", type=Path)
    parser.add_argument("--compare", nargs=2, action="append", type=Path, default=[],
                        metavar=("REFERENCE", "CANDIDATE"))
    parser.add_argument("--declared", nargs="+", metavar="NAME", default=[],
                        help="compare the pairs of [passk.NAME] in experiments.toml, each into "
                        "results/passk-NAME.json")
    parser.add_argument("--variant", choices=VARIANTS, default="official")
    parser.add_argument("--k", nargs="+", type=int, help="default: 1, 2, 4, ... up to n")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.records and not args.compare and not args.declared:
        parser.error("give records, --compare REFERENCE CANDIDATE, or --declared NAME")
    if (args.records or args.compare) and args.output is None:
        parser.error("--output is required for explicit records")
    jobs = [(declared_output(name), [], declared_pairs(name)) for name in args.declared]
    if args.records or args.compare:
        jobs.append((args.output, args.records, args.compare))
    for output, records, pairs in jobs:
        write_report(output, records, pairs, args, parser)


def write_report(output: Path, records: list[Path], pairs: list, args, parser) -> None:
    paths = dict.fromkeys([*records, *(path for pair in pairs for path in pair)])
    cells = {path: load_sampled(path, args.variant) for path in paths}
    samples = min(cell.samples for cell in cells.values())
    ks = args.k or default_ks(samples)
    if max(ks) > samples or min(ks) < 1:
        parser.error(f"k must lie between 1 and the smallest sample count ({samples})")
    report = {
        "variant": args.variant,
        "estimator": "1 - C(n-c, k) / C(n, k) per question, averaged over questions",
        "success": "full credit under the scorer variant",
        "records": [{"dataset": cell.protocol["dataset"], "prompt": cell.protocol["prompt"],
                     **cell.summary(ks)} for cell in cells.values()],
        "comparisons": [{"reference": cells[reference].model["label"],
                         "candidate": cells[candidate].model["label"],
                         "dataset": cells[reference].protocol["dataset"],
                         "prompt": cells[reference].protocol["prompt"],
                         "deltas": paired_pass_at_k(cells[reference], cells[candidate], ks,
                                                    resamples=args.resamples)}
                        for reference, candidate in pairs],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{len(cells)} records, {len(pairs)} comparisons -> {output}")


if __name__ == "__main__":
    main()
