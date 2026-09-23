"""Evaluate every adapter a training run kept on the dev tier, with both prompts.

    python scripts/evaluate_checkpoints.py checkpoints/<run>     # or <run-dir>/checkpoints

Trainers run this at the end of a completed run (skip with --no-dev-eval). One vLLM engine
loads the starting model and serves each kept adapter through LoRA, so the curve costs no
merges. Records are tagged with the run and step and stay with the run, in
<checkpoints>/dev-records; finished cells are cached, so an interrupted evaluation resumes
where it stopped. To publish a curve in the report, evaluate into results/records with
--records-dir (or copy the run's records there) under a run name used once.
"""

import argparse
import json
import re
from pathlib import Path

from vqar.config import is_large
from vqar.eval import ModelRef, Protocol, run_eval
from vqar.eval.runner import DATA_DIR
from vqar.inference import VLLMGenerator
from vqar.registry import DATASETS


def kept_adapters(output: Path) -> list[tuple[int, Path]]:
    """(step, directory) of every adapter the run kept, in training order."""
    found = []
    for directory in (output / "adapters").glob("step-*"):
        match = re.fullmatch(r"step-(\d+)", directory.name)
        if match and (directory / "adapter_model.safetensors").exists():
            found.append((int(match.group(1)), directory))
    return sorted(found)


def dev_cells(output: Path, config: dict, prompts, tier: str) -> list[tuple[ModelRef, Protocol]]:
    """Every (checkpoint, dev protocol) pair of a run, from its recorded configuration."""
    run = config["training_args"]["run_name"]
    datasets = [d for d in config["preset"]["datasets"] if "dev" in DATASETS[d].subsets]
    return [
        (ModelRef(config["policy_model"], label=run, run=run, step=step,
                  adapter=str(directory)),
         Protocol(dataset, prompt=prompt, role="dev", tier=tier))
        for step, directory in kept_adapters(output)
        for dataset in datasets
        for prompt in prompts
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="the run's checkpoint directory")
    parser.add_argument("--prompts", nargs="+", default=["short", "reasoning"])
    parser.add_argument("--tier", choices=tuple(Protocol.TIERS), default="dev")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--records-dir", type=Path, default=None,
                        help="default: <output>/dev-records, beside the run")
    args = parser.parse_args()
    config = json.loads((args.output / "run_config.json").read_text())
    cells = dev_cells(args.output, config, args.prompts, args.tier)
    records_dir = args.records_dir or args.output / "dev-records"
    if not cells:
        raise SystemExit(f"No kept adapters under {args.output / 'adapters'}")
    rank = json.loads((Path(cells[0][0].adapter) / "adapter_config.json").read_text())["r"]
    engine = VLLMGenerator(config["policy_model"], max_lora_rank=rank,
                           gpu_memory_utilization=0.88 if is_large(config["preset"]["name"])
                           else 0.8)
    for model, protocol in cells:
        record = run_eval(model, protocol, engine.with_adapter(model.adapter),
                          data_dir=args.data_dir, records_dir=records_dir)
        normalized = record.metrics["normalized"]["overall"]
        print(f"step {model.step} {protocol.slug()}: normalized {normalized:.4f}", flush=True)


if __name__ == "__main__":
    main()
