"""Write the corrected campaign's RunRecords into results/records/ (needs saved predictions).

    python -m scripts.migrate_campaign

A one-off migration: the original files under results/baselines/ stay unchanged and the
records it writes are versioned. Rerunning it verifies that the stored records still match.
Weights identities come from the local copies of the merged models. The 8B GRPO merge was
once removed to reclaim disk; rebuilt from its archived adapter, its single shard hashes to
the recorded value. Without the rebuild, its records keep that hash and no config hash.
"""

import argparse
import hashlib
import json
from pathlib import Path

from vqar.eval import ModelRef, RunRecord
from vqar.eval.legacy import corrected_records
from vqar.eval.protocol import slug
from vqar.eval.record import record_path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_COPIES = {
    "sft2": "checkpoints/merged/sharedpool_20260913_sft_2b",
    "grpo2-base": "checkpoints/merged/sharedpool_20260913_grpo_2b_base",
    "grpo2-sft": "checkpoints/merged/sharedpool_20260913_grpo_2b_sft",
    "opd2-self": "checkpoints/merged/sharedpool_20260913_opd_2b_self",
    "grpo8-base": "checkpoints/merged/sharedpool_20260913_grpo_8b_base",
}
# sha256 of the removed merge's single model.safetensors, from its model-sha256.txt.
GRPO8_SHARD = "41dc686c5f7a041eea3bcbbb5f3d639df1278f31b034c144e4554cc299d96ae0"
GRPO8_WEIGHTS = hashlib.sha256(f"model.safetensors:{GRPO8_SHARD}\n".encode()).hexdigest()


def campaign_weights(baselines: Path) -> dict[str, dict]:
    bases = json.loads((baselines / "protocol.json").read_text())["models"]
    weights = {
        "base2": {"hub": bases["2b"]["repository"], "revision": bases["2b"]["revision"]},
        "base8": {"hub": bases["8b"]["repository"], "revision": bases["8b"]["revision"]},
        "grpo8-base": {"sha256": GRPO8_WEIGHTS, "config_sha256": None},
    }
    for arm, path in LOCAL_COPIES.items():
        if (ROOT / path).is_dir():
            weights[arm] = ModelRef(str(ROOT / path)).identity()["weights"]
    if weights["grpo8-base"]["sha256"] != GRPO8_WEIGHTS:
        raise SystemExit("The rebuilt 8B GRPO merge differs from the recorded weights.")
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baselines", type=Path, default=ROOT / "results/baselines")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--records-dir", type=Path, default=ROOT / "results/records")
    args = parser.parse_args()
    written = verified = 0
    for record, lines in corrected_records(args.baselines, args.data_dir,
                                           campaign_weights(args.baselines)):
        path = record_path(args.records_dir, record.model, slug(record.protocol),
                           record.protocol)
        if path.exists():
            stored = RunRecord.load(path)
            if (stored.model, stored.protocol, stored.metrics) != (
                    record.model, json.loads(json.dumps(record.protocol)),
                    json.loads(json.dumps(record.metrics))):
                raise SystemExit(f"{path} differs from a fresh migration")
            verified += 1
            continue
        record.save(path, lines)
        written += 1
    print(f"{written} records written, {verified} verified unchanged -> {args.records_dir}")


if __name__ == "__main__":
    main()
