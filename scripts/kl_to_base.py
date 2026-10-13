"""How far training moved a model from its start: forward KL on the task's own questions.

    python -m scripts.kl_to_base --dataset slake --base base2 \
        --models slk-sft2 slk-short2-base-e1 slk-short2-sft-e1

RL's Razor (Shenfeld et al., 2025) finds that forgetting elsewhere tracks the forward KL
KL(pi_0 || pi) = E[x, y ~ pi_0](log pi_0(y|x) - log pi(y|x)) between the starting model pi_0 and
the fine-tuned pi, measured on the new task, and that on-policy RL moves less than SFT. Here
pi_0 samples responses to the first questions of the dataset's test role at temperature 1 under
the given prompt, keeping each sampled token's log-probability; each trained model then scores
the same tokens through vLLM's prompt log-probabilities. Samples and per-model scores stay
under runs/kl/. A response's KL term is the sum over
its tokens (the stop token is not scored); the estimate averages over samples and questions,
with a 95% bootstrap interval over questions. A response whose text does not re-tokenize to
the sampled ids is left out and counted. Writes results/kl/<dataset>-<prompt>-<base>.json.

Stages run in child processes so each engine's GPU memory is released: `sample` (the base)
writes the samples beside the output, `score` adds one model, the default runs everything.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from vqar.eval.examples import select_rows, to_example
from vqar.eval.matrix import load_matrix
from vqar.eval.protocol import Protocol
from vqar.eval.report import ROOT
from vqar.eval.runner import DATA_DIR
from vqar.inference import build_messages
from vqar.registry import PROMPTS

OUTPUT_DIR = ROOT / "results" / "kl"
WORK_DIR = ROOT / "runs" / "kl"  # samples and per-model scores, local
TEMPERATURE = 1.0
SEED = 42


def arms() -> dict:
    every, _ = load_matrix(ROOT / "experiments.toml")
    return {arm.label: arm for arm in every}


def conversations(dataset: str, prompt: str, questions: int) -> tuple[list[str], list[list[dict]]]:
    protocol = Protocol(dataset, prompt=prompt, tier=f"first-{questions}")
    rows, _ = select_rows(protocol, DATA_DIR)
    examples = [to_example(row) for row in rows]
    template = PROMPTS[prompt].text
    return ([e.qid for e in examples],
            [build_messages(e.question, e.images, template) for e in examples])


def engine(label: str):
    from vqar.inference import load_vllm

    arm = arms()[label]
    model = arm.model if arm.revision else str(ROOT / arm.model)
    return load_vllm(model, max_model_len=4096, revision=arm.revision,
                     gpu_memory_utilization=min(arm.gpu_memory_utilization, 0.85))


def sample(args, samples_path: Path) -> None:
    """The base model's responses with the log-probability of every sampled token."""
    from vllm import SamplingParams

    qids, chats = conversations(args.dataset, args.prompt, args.questions)
    budget = args.max_tokens or PROMPTS[args.prompt].max_tokens
    params = SamplingParams(temperature=TEMPERATURE, max_tokens=budget, n=args.samples,
                            seed=SEED, logprobs=0)
    llm = engine(args.base)
    rows = []
    for qid, result in zip(qids, llm.chat(chats, params, use_tqdm=True), strict=True):
        for response in result.outputs:
            rows.append({"qid": qid, "text": response.text,
                         "token_ids": list(response.token_ids),
                         "finish_reason": response.finish_reason,
                         "base_logprobs": [step[token].logprob for step, token in
                                           zip(response.logprobs, response.token_ids,
                                               strict=True)]})
    samples_path.write_text(json.dumps({"base": args.base, "rows": rows}) + "\n")
    print(f"sampled {len(rows)} responses -> {samples_path}")


def score(args, samples_path: Path, label: str) -> dict:
    """One trained model's log-probability of each sampled response, scored as the final
    assistant turn of the same conversation."""
    from vllm import SamplingParams

    drawn = json.loads(samples_path.read_text())["rows"]
    qids, chats = conversations(args.dataset, args.prompt, args.questions)
    chat_of = dict(zip(qids, chats, strict=True))
    scored = [chat_of[row["qid"]] + [{"role": "assistant", "content": row["text"]}]
              for row in drawn]
    params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    llm = engine(label)
    results = llm.chat(scored, params, use_tqdm=True, add_generation_prompt=False,
                       continue_final_message=True)
    terms, mismatched = {}, 0
    for row, result in zip(drawn, results, strict=True):
        # A response that stopped ends with the stop token, which the scored prompt omits.
        stopped = row["finish_reason"] == "stop"
        body = row["token_ids"][:-1] if stopped else row["token_ids"]
        prompt_ids = list(result.prompt_token_ids)
        if not body or prompt_ids[-len(body):] != body:
            mismatched += 1  # the text did not re-tokenize to the sampled ids
            continue
        logps = result.prompt_logprobs[-len(body):]
        model_logprob = sum(step[token].logprob for step, token in zip(logps, body, strict=True))
        base_logprob = sum(row["base_logprobs"][:len(body)])
        terms.setdefault(row["qid"], []).append(base_logprob - model_logprob)
    return {"terms": terms, "mismatched": mismatched}


def bootstrap(per_question: dict[str, list[float]], resamples: int = 10_000) -> dict:
    means = [sum(v) / len(v) for v in per_question.values()]
    rng = random.Random(0)
    draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(resamples))
    return {"kl": sum(means) / len(means), "low": draws[int(0.025 * resamples)],
            "high": draws[int(0.975 * resamples) - 1], "questions": len(means)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt", default="short", choices=sorted(PROMPTS))
    parser.add_argument("--base", required=True, help="the starting model's arm label")
    parser.add_argument("--models", nargs="+", required=True, help="trained arm labels")
    parser.add_argument("--questions", type=int, default=500)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--stage", choices=("all", "sample", "score"), default="all")
    parser.add_argument("--model", default=None, help="score stage: the arm to score")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}-{args.prompt}-{args.base}"
    stem = OUTPUT_DIR / name
    work = WORK_DIR / f"{name}-q{args.questions}-k{args.samples}"  # samples are reused per draw
    samples_path = work.with_suffix(".samples.json")
    if args.stage == "sample":
        sample(args, samples_path)
        return
    if args.stage == "score":
        result = score(args, samples_path, args.model)
        Path(f"{work}.{args.model}.json").write_text(json.dumps(result) + "\n")
        return
    base_args = [sys.executable, "-m", "scripts.kl_to_base", *sys.argv[1:]]
    if not samples_path.exists():
        subprocess.run([*base_args, "--stage", "sample"], check=True)
    report = {"dataset": args.dataset, "prompt": args.prompt, "base": args.base,
              "questions": args.questions, "samples": args.samples,
              "temperature": TEMPERATURE, "estimator": "forward KL, base samples", "models": {}}
    for label in args.models:
        part = Path(f"{work}.{label}.json")
        if not part.exists():
            subprocess.run([*base_args, "--stage", "score", "--model", label], check=True)
        result = json.loads(part.read_text())
        report["models"][label] = {**bootstrap(result["terms"]),
                                   "mismatched": result["mismatched"]}
        print(label, report["models"][label])
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
