"""Before an RL run: how many pool prompts would give GRPO a learning signal?

    python -m scripts.learnability --preset grpo_2b_main --label base2 --tier dev

Takes a seeded random sample of each dataset's RL pool, draws a group of responses per
prompt from the starting model with the preset's rollout settings and the reasoning prompt,
scores them with the GRPO correctness reward, and reports how many prompts have a pass rate
strictly between 0 and 1. A prompt the model always or never answers correctly gives GRPO
no gradient. Each dataset's result is a RunRecord under a sampled protocol.
"""

import argparse
from pathlib import Path

from vqar.config import GRPO_PRESETS
from vqar.eval import ModelRef, Protocol, run_eval
from vqar.eval.runner import DATA_DIR, RECORDS_DIR
from vqar.inference import VLLMGenerator
from vqar.registry import datasets_with_role
from vqar.rl import ROLLOUT_TEMPERATURE


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(GRPO_PRESETS), default="grpo_2b_main",
                        help="the run to check: its group size and completion budget")
    parser.add_argument("--model", default=None,
                        help="the RL starting model (default: the preset's)")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--label", default=None, help="records folder name")
    parser.add_argument("--datasets", nargs="+", choices=datasets_with_role("rl"),
                        default=datasets_with_role("rl"))
    parser.add_argument("--tier", choices=tuple(Protocol.TIERS), default="smoke",
                        help="how many pool prompts to sample (smoke 32, dev 500)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    args = parser.parse_args()

    preset = GRPO_PRESETS[args.preset]
    model = ModelRef(args.model or preset.model, args.revision, args.label).pinned()
    generator = VLLMGenerator(model.path, model.revision,
                              gpu_memory_utilization=args.gpu_memory_utilization)
    for dataset in args.datasets:
        protocol = Protocol(dataset, prompt="reasoning", role="rl", tier=args.tier,
                            max_tokens=preset.max_completion_length,
                            temperature=ROLLOUT_TEMPERATURE, samples=preset.num_generations)
        record = run_eval(model, protocol, generator, data_dir=args.data_dir,
                          records_dir=args.records_dir)
        learnability = record.metrics["learnability"]
        print(f"{dataset}: {learnability['fraction_between_0_and_1']:.1%} of "
              f"{record.metrics['n']} prompts learnable; correct samples per prompt "
              f"{learnability['correct_samples_histogram']} -> {record.path}")


if __name__ == "__main__":
    main()
