# Harness cross-check against lmms-eval

**Purpose**: validate that our custom eval harness produces numbers comparable to the
community-standard harness before any training claims are made.

## Setup

- Model: `Qwen/Qwen3-VL-2B-Instruct`, greedy decoding, zero-shot.
- Task: GQA testdev_balanced (identical underlying population for both harnesses).
- lmms-eval: `0.7.2` (source install @ `047ec52`), task `gqa`, `--limit 1000`
  (first 1000 testdev docs), transformers backend, sdpa attention.
- Ours: `evaluate.py --preset zero_shot_2b --datasets gqa` — vLLM backend, full
  12,578-question testdev, same "answer using a single word or phrase" prompt suffix.

## Result

| Harness | n | GQA accuracy |
|---|---|---|
| lmms-eval | 1000 | 58.7 ± 1.56 |
| ours | 12578 | **59.5** |

Difference: 0.8 points, inside lmms-eval's own sampling CI → **pass** (gate was ±2).

## Caveats

- Different sample sizes (1000 vs full split); the comparison is population-level.
- Backends differ (transformers vs vLLM); both greedy.
- Our VQAv2/TextVQA scoring mirrors lmms-eval's normalization exactly (unconditional
  official-normalization of prediction and gold answers; see `vqar/metrics.py`).
