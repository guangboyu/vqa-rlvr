"""Regenerate every report table from the stored RunRecords.

    python -m scripts.report

Rewrites the marked blocks in README.md and docs/supplementary/results.md and the demo's
results/report.json. A test regenerates them and fails when a committed copy drifts.
"""

import argparse
import json

from vqar.eval.report import MARKED_FILES, ROOT, build, fill


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    blocks, export = build()
    for name in MARKED_FILES:
        path = ROOT / name
        text = path.read_text()
        updated = fill(text, blocks)
        if updated != text:
            path.write_text(updated)
            print(f"updated {name}")
    (ROOT / "results" / "report.json").write_text(json.dumps(export, indent=1) + "\n")
    print("wrote results/report.json")


if __name__ == "__main__":
    main()
