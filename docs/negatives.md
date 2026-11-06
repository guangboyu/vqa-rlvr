# What did not work

Each entry is a change tried against the recipe and kept out of it, in one form: tried, expected,
saw, why, implies. "Why" says whether it was observed here or is a proposed explanation. "Expected" is
what was written down before the run; where nothing was, it says so. The tables are below the
checklist.

## Checklist

### On-policy distillation

- **Tried:** 1,000 updates where a frozen Instruct 2B, shown the reference answer, scores each of the
  student's tokens.
- **Expected:** a dense token-level signal would teach more than a 0/1 reward on the same samples.
- **Saw:** it keeps the reasoning (141 tokens on VQAv2) and gains 15.4 points there under the reasoning
  prompt, but +6.1 normalized and +0.65 audited. Its 16.4-point GQA gain is mostly casing (+0.8
  normalized). The short prompt barely moves.
- **Why (proposed):** a teacher of the same size knows little the reference doesn't already say.
- **Implies:** stays out of the recipe at this scale; a much stronger teacher is the condition for a retry.

### Filtered pool

- **Tried:** only questions Instruct gets right on 1–7 of 8 samples.
- **Expected:** a gain: only 39% of groups had a gradient at 2B, so keeping the questions that give one should help.
- **Saw:** dead groups fall (0.33 against 0.58 early), the short prompt ends level, and the reasoning prompt
  ends worse on all four sets, by 18.1 points on CLEVR.
- **Why (observed):** it removes the reasoning later and less completely (about 20 tokens against 8), and
  what remains miscounts.
- **Implies:** changing the pool changes behavior as well as signal. Offline filtering stays out of the recipe.

### Refilling dead groups (DAPO's dynamic sampling)

- **Tried:** up to 3 extra sampling rounds replace groups whose answers all score the same.
- **Expected:** a gain: filtered pools go stale as the model improves, and refilling keeps every group live.
- **Saw:** within 0.5 points on every set and prompt except CLEVR under the reasoning prompt (−2.0).
- **Why (proposed):** with this learning rate and pool, signal is not what limits the run.
- **Implies:** stays out of the default. Tested on photos; charts are untested.

### Format reward at 8B

- **Tried:** a 0.2 reward for the answer format added to correctness.
- **Expected:** a measure of what keeping the reasoning costs.
- **Saw:** it keeps the reasoning (about 130 tokens) and costs 9.7, 9.6 and 5.6 points under the reasoning prompt.
- **Why (observed):** the reasoning it keeps is not the reasoning correctness would have kept.
- **Implies:** paying for form buys form. The recipe rewards correctness only.

### Perception loss

- **Tried:** GRPO plus PAPO's loss at weight 0.02, rewarding answers that change when 60% of the image is masked.
- **Expected:** answers depend more on the image.
- **Saw:** 9.6 points below plain GRPO on SLAKE; with a black image it scores 3.7% against Instruct's 25.1%.
- **Why (observed):** it found the cheapest way to depend on the image, replying that the image is missing.
- **Implies:** a reward for image dependence needs its own check for gaming. Lower weights are untested.

### Vision LoRA and upscaling

- **Tried:** LoRA on the vision blocks and merger; images upscaled to the pixel budget.
- **Expected:** both help SLAKE, whose images are small and whose domain the encoder saw little of.
- **Saw:** vision LoRA +0.3 [−0.9, +1.3], upscaling −0.8 for SFT and +0.4 for GRPO, all intervals including zero.
- **Why (observed):** 79% of SLAKE's test questions already come with images 512 pixels or larger.
- **Implies:** resolution and encoder adaptation do not limit SLAKE at 2B.

### Datasets rejected by the screen

- **Tried:** compositional CLEVR, MedThinkVQA, Kvasir-VQA-x1, PathVQA, ReXVQA and DeepTumorVQA.
- **Expected:** compositional CLEVR and MedThinkVQA were reasoning candidates.
- **Saw:** CLEVR is nearly saturated (88.4%) and reasoning costs it 5.6 points; the medical sets lose under 10 points without the image.
- **Why (observed):** these fail gates written before the candidates ran ([screen](harness/evaluation.md#dataset-screening)).
- **Implies:** no training time spent. The screen's own test comes with new sources.

### Transfer between families

- **Tried:** each family's model scored on the other family's test sets.
- **Expected:** nothing written down.
- **Saw:** the habit travels. The chart model keeps reasoning on photos and gains 15–16 points on VQAv2 and GQA
  under the reasoning prompt; the photo model answers in 8–10 tokens on charts and loses 6.7.
- **Why (observed):** under the short prompt no model loses more than 0.6 points, so what moves is the response habit.
- **Implies:** what RL teaches here is mostly how to respond under a prompt.

### One line each

- 16 questions per update for the same questions: 4× fewer steps, −0.8 to −2.3 under the reasoning prompt on photos.
- Learning rate 3×: within 0.9 points on every cell. 10×: costs accuracy everywhere.
- Unmerged rollouts: no cell moves more than 0.52 points; kept because they are on-policy.
- Eight single-setting changes on ChartQA (loss averaging, KL, reward scaling, response cap, list-aware reward, canvas, image geometry): none beats a same-seed repeat, at 300 updates.
- Clip-higher: not run, since with one gradient pass per batch the policy ratio is 1 and the clip never binds.

## Considered, not run

- Down-weighting or removing questions answerable without the image. We chose to measure rather than
  intervene: blind-answerable share and image gain are reported instead. That forcing image use makes a
  model over-rely on it is a concern, not a result.
- An LLM cleaning pass and chosen/rejected pairs: RLVR does not train on pairs, and cleaning waits until
  error labels show label noise is a real share.
- LoRA against full fine-tuning: a new training setup for one question.
- MM-Vet: LLM-graded and outside the table.
- A rollout gradient-mechanism analysis, learned mixture weights, a four-way ratio grid, a plugin framework.

## Training knobs (ChartQA)

Each knob changes one training setting against a fixed baseline: the pilot's Instruct 2B →
GRPO run (300 updates, the frozen recipe). Knobs run on ChartQA rather than on VQAv2 and
GQA because there the responses keep their reasoning and vary in length, so loss and length
settings have something to act on; on the general data a reasoning-prompt run collapses to
8-token answers within 200 updates either way. Each knob trains for the same 300 updates and
its final adapter answers ChartQA's full dev role (958 questions) greedily under both
prompts; the test cells are kept for knobs that get promoted to a full run. The expected
direction was written before each run. A knob reads + or − only when its paired 95%
interval excludes zero and the change is larger than the repeat's, a rerun of the baseline
with the same seed whose difference is run-to-run noise. A 300-update budget favors knobs
that speed early learning, so a + is a candidate for a full-length run, not a recipe change.
Official relaxed accuracy, points; `python -m scripts.knobs summarize chartqa`.

<!-- report:knobs-chartqa -->
| Knob | Change | Expected | Reasoning Δ | Short Δ | Reasoning tokens | Train GPU-h | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| repeat | the baseline again, same seed | the noise floor: GPU kernels make same-seed runs differ | +0.52 [-1.25, +2.40] | +0.31 [-0.21, +0.84] | 198 → 186 | 1.1 | — |
| loss-grpo | average each response's token losses first (loss_type grpo) | 0 or −, longer responses: a long wrong answer is penalized less per token | +0.21 [-1.57, +1.98] | +0.00 [-0.52, +0.52] | 198 → 211 | 1.1 | 0 |
| length-1024 | 1,024-token response cap (4 completions per micro-batch, same 32 per update) | small +, longer responses: 4.7% of pilot responses hit the 640 cap and score 0 | -0.21 [-2.09, +1.67] | +0.21 [-0.31, +0.73] | 198 → 227 | 1.3 | 0 |
| scale-none | advantages not divided by the group's reward spread (Dr. GRPO) | about 0 | -1.46 [-3.34, +0.42] | +0.42 [+0.00, +0.94] | 198 → 213 | 1.1 | 0 |
| kl-0.04 | KL penalty 0.04 to the starting model (original GRPO) | slight − at 300 updates: the penalty slows learning | -0.94 [-2.92, +1.04] | +0.21 [-0.31, +0.84] | 198 → 210 | 1.3 | 0 |
| loss-dr-grpo | token losses divided by a fixed constant (loss_type dr_grpo) | − at 300 updates: dividing by the 640-token cap instead of ~200 written tokens shrinks each step about 3× | -1.15 [-3.13, +0.84] | +0.21 [-0.31, +0.84] | 198 → 211 | 1.1 | 0 |
| reward-lists | list-aware ChartQA reward: Unicode dashes as hyphens, bracketed lists as unordered sets | small +: about 2% of questions have list or dash answers the current reward marks wrong | -0.52 [-2.51, +1.46] | +0.42 [-0.31, +1.15] | 198 → 197 | 1.1 | 0 |
| canvas-896 | 896-pixel training canvas instead of 608 (784 vision tokens instead of 361; 4 completions per micro-batch) | + if the 608 canvas costs accuracy in the resolution check | +1.04 [-0.94, +3.03] | +0.21 [-0.21, +0.63] | 198 → 197 | 1.6 | 0 |
| image-budget | each image kept in shape within the 0.79 MP evaluation budget, no padding (up to 768 vision tokens; 4 completions per micro-batch) | +, about canvas-896's: same pixels on most charts, no padding tokens | +0.10 [-1.77, +1.98] | +0.73 [+0.21, +1.36] | 198 → 195 | 1.2 | 0 |
<!-- /report:knobs-chartqa -->

No knob moves reasoning-prompt accuracy beyond the repeat at this budget; the largest
changes, −1.5 without reward scaling and +1.0 with the 896-pixel canvas, sit inside
intervals about ±2 points wide. The last three knobs came from the overfitting check
([validation](harness/validation.md#part-2-checks-2026-10-08)). The list-aware reward, which
concerns about 2% of questions, moves dev accuracy by −0.5. Training on the evaluation image
geometry instead of the 608 canvas moves it by +0.1, although that canvas costs Instruct 4.3
points when it is the image evaluated: every trained model is evaluated on the original
images, so at 300 updates the training geometry does not limit what RL learns about reading
charts. The same run's short answers rise 0.7 points [+0.2, +1.4], against the repeat's 0.3,
a small difference outside the reasoning-prompt verdict.

## Recipe experiments

Before the final rerun, one change at a time was tested against a baseline run at 2B,
evaluated on the full test cells. Each candidate minus its reference, official scorer,
with a 95% bootstrap interval over questions (10,000 resamples); one training seed per arm.

| Arm | Change against its reference |
|---|---|
| Recipe baseline | Instruct 2B → GRPO: whole rl roles of VQAv2 and GQA in random order, 4 questions × 8 answers per update, learning rate 1e-5, correctness reward, 1,000 updates, merged-weight sampler |
| Filtered pool | pool limited to questions Instruct 2B answers right on 1–7 of 8 samples (4,609 questions from the first 5,000 rl questions of each dataset) |
| Short rollout prompt | rollouts under the short prompt with 64-token completions |
| Short rollout prompt, from SFT | the same, starting from SFT 2B; compared with the baseline and with SFT 2B |
| Refill dead groups | up to 3 extra sampling rounds replace groups whose rewards are all equal |
| 16 questions per update | 16 questions × 8 answers per update for 250 updates (the same 4,000 questions) |
| Learning rate 3e-5, 1e-4 | the learning rate |
| Unmerged-adapter rollouts | the sampler serves the LoRA adapter unmerged instead of bf16 merged weights |
| 8B, correctness + 0.2 format | Instruct 8B with a 0.2 format reward added, 300 updates; compared with Instruct 8B → GRPO (Sep 13) |

<!-- report:recipe -->
| Comparison | Prompt | Scorer | VQAv2 | GQA | CLEVR counting | TextVQA (far-OOD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Filtered pool − Recipe baseline | short | official | +0.11 [-0.23, +0.45] | -0.22 [-0.49, +0.05] | +0.00 [+0.00, +0.00] | -0.14 [-0.49, +0.20] |
| Filtered pool − Recipe baseline | reasoning | official | -1.75 [-2.31, -1.19] | -0.78 [-1.21, -0.34] | -18.10 [-19.53, -16.70] | -1.93 [-2.51, -1.37] |
| Short rollout prompt − Recipe baseline | short | official | +0.53 [+0.11, +0.94] | +0.56 [+0.21, +0.90] | -0.07 [-0.17, +0.00] | +0.13 [-0.23, +0.49] |
| Short rollout prompt − Recipe baseline | reasoning | official | -16.11 [-17.26, -14.94] | -17.14 [-17.99, -16.29] | -12.20 [-13.47, -10.97] | -8.10 [-9.10, -7.07] |
| Short rollout prompt, from SFT − Recipe baseline | short | official | +1.84 [+1.19, +2.50] | +2.15 [+1.63, +2.67] | -0.03 [-0.17, +0.07] | -0.17 [-0.80, +0.46] |
| Short rollout prompt, from SFT − Recipe baseline | reasoning | official | -7.61 [-8.57, -6.67] | -1.88 [-2.54, -1.25] | -13.83 [-15.13, -12.57] | -7.19 [-8.16, -6.24] |
| Short rollout prompt, from SFT − SFT 2B | short | official | +0.06 [-0.40, +0.53] | -0.09 [-0.48, +0.29] | +0.07 [-0.10, +0.23] | +0.57 [+0.10, +1.04] |
| Short rollout prompt, from SFT − SFT 2B | reasoning | official | +2.75 [+1.94, +3.55] | +1.72 [+1.22, +2.21] | -6.80 [-8.20, -5.43] | +3.55 [+2.67, +4.46] |
| Refill dead groups − Recipe baseline | short | official | -0.25 [-0.57, +0.07] | +0.10 [-0.18, +0.40] | +0.20 [-0.07, +0.47] | -0.06 [-0.38, +0.24] |
| Refill dead groups − Recipe baseline | reasoning | official | +0.24 [-0.19, +0.66] | +0.25 [-0.08, +0.56] | -1.97 [-2.47, -1.50] | +0.33 [-0.12, +0.77] |
| 16 questions per update − Recipe baseline | short | official | -0.12 [-0.40, +0.16] | -0.34 [-0.60, -0.08] | -0.83 [-1.17, -0.50] | -0.23 [-0.52, +0.06] |
| 16 questions per update − Recipe baseline | reasoning | official | -1.28 [-1.77, -0.80] | -2.27 [-2.70, -1.83] | +0.57 [+0.30, +0.87] | -0.81 [-1.24, -0.38] |
| Learning rate 3e-5 − Recipe baseline | short | official | +0.27 [-0.16, +0.70] | -0.06 [-0.42, +0.31] | -0.10 [-0.27, +0.03] | -0.07 [-0.49, +0.34] |
| Learning rate 3e-5 − Recipe baseline | reasoning | official | -0.71 [-1.32, -0.11] | -0.56 [-1.07, -0.06] | +0.53 [+0.27, +0.83] | +0.86 [+0.27, +1.44] |
| Learning rate 1e-4 − Recipe baseline | short | official | -0.74 [-1.35, -0.10] | -1.64 [-2.16, -1.12] | -11.07 [-12.20, -9.97] | -2.24 [-2.88, -1.61] |
| Learning rate 1e-4 − Recipe baseline | reasoning | official | -5.56 [-6.47, -4.64] | -4.37 [-5.07, -3.70] | -1.20 [-1.77, -0.67] | -3.79 [-4.76, -2.82] |
| Unmerged-adapter rollouts − Recipe baseline | short | official | -0.19 [-0.48, +0.10] | +0.04 [-0.23, +0.31] | -0.17 [-0.33, -0.03] | +0.19 [-0.08, +0.46] |
| Unmerged-adapter rollouts − Recipe baseline | reasoning | official | +0.25 [-0.14, +0.64] | -0.21 [-0.54, +0.13] | +0.43 [+0.20, +0.67] | +0.52 [+0.10, +0.92] |
| 8B, correctness + 0.2 format − Instruct 8B → GRPO (Sep 13) | short | official | +0.00 [-0.18, +0.19] | -0.02 [-0.17, +0.14] | +0.03 [+0.00, +0.10] | +0.00 [-0.23, +0.24] |
| 8B, correctness + 0.2 format − Instruct 8B → GRPO (Sep 13) | reasoning | official | -9.65 [-10.63, -8.67] | -9.60 [-10.33, -8.86] | -0.70 [-1.33, -0.10] | -5.55 [-6.42, -4.70] |
<!-- /report:recipe -->

Three questions decided the recipe. **Does training on questions with a signal help?**
The filtered pool starts with fewer dead groups (0.33 against 0.58 in the first 100
updates) but ends level on the short prompt and worse on the reasoning prompt: it removes
the reasoning later and less completely (about 20 tokens instead of 8), and the reasoning it
keeps miscounts CLEVR (−18.1). **Do larger batches, refilled groups, or a higher learning
rate help?** Refilling and a 3× learning rate stay within 0.9 points on every dataset
but CLEVR under the reasoning prompt; 16 questions per update at the same number of questions trains 4× fewer
steps and ends worse on the reasoning prompt; a 10× learning rate costs accuracy everywhere
and 11 points of CLEVR counting under the short prompt, the pattern of the 8B SFT at too
high a rate. **Does the sampler matter?** Unmerged-adapter rollouts remove the
sampler–trainer gap ([method](harness/method.md#sampler-and-trained-policy)) without changing any
cell by more than 0.52 points, so the frozen recipe keeps them as the correct on-policy
setup. The rollout prompt is the one change that moves the short prompt: short-prompt GRPO
adds 0.5–0.6 points on VQAv2 and GQA over reasoning-prompt GRPO and keeps the reasoning
behavior of its start; both are part of the final rerun. At 8B, a 0.2 format reward keeps
the reasoning (about 130 tokens) and costs 9.7, 9.6, and 5.6 points under the reasoning
prompt against correctness-only GRPO of the same 300 updates, leaving the short prompt
unchanged.

The July runs, which used a defective VQAv2 reward, are tabulated with the erratum in
[history](history.md).

## Image resolution

The headline cells resize images to at most 0.79 megapixels, which no VQAv2, GQA, or CLEVR
test image exceeds but 826 of the 5,000 TextVQA images do. The same arm evaluated at the
model's own budget (16.8 megapixels) shows what the cap costs, with a paired 95% interval.
For Instruct 2B it costs nothing measurable: the full budget changes short-prompt TextVQA by
−0.1 points and reasoning-prompt TextVQA by −0.7, both intervals including zero. An arm
evaluated on the 608×608 canvas RL trains on appears here too, paired with its
original-image cell. On that canvas the four corrected-campaign 2B arms change VQAv2 and
GQA by at most 1 point, including both GRPO arms that trained on it, while TextVQA loses 5
to 7 points for every arm: text needs pixels, and the evaluation budget does not hide a
gain that RL learned at its training resolution. SLAKE cells evaluated on images upscaled to
the budget belong to the [vision-side options](#vision-side-options-slake).

<!-- report:resolution -->
| Arm | Dataset | Prompt | Scorer | Megapixels | Change |
| --- | --- | --- | --- | ---: | ---: |
| Instruct 2B | CLEVR counting | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -0.97 [-2.37, +0.43] |
| Instruct 2B | CLEVR counting | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.97 [-2.37, +0.43] |
| Instruct 2B | CLEVR counting | short | official | 0.79 → 0.37 (608×608 RL canvas) | +2.13 [+1.60, +2.70] |
| Instruct 2B | CLEVR counting | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +2.13 [+1.60, +2.70] |
| Instruct 2B | GQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.30 [-0.33, +0.91] |
| Instruct 2B | GQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.16 [-0.47, +0.79] |
| Instruct 2B | GQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | +0.02 [-0.42, +0.46] |
| Instruct 2B | GQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.01 [-0.43, +0.45] |
| Instruct 2B | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | +1.23 [-0.85, +3.20] |
| Instruct 2B | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.85, +3.11] |
| Instruct 2B | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | +0.47 [-0.75, +1.70] |
| Instruct 2B | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | +0.47 [-0.75, +1.70] |
| Instruct 2B | TextVQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -5.64 [-6.78, -4.51] |
| Instruct 2B | TextVQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -5.65 [-6.78, -4.52] |
| Instruct 2B | TextVQA | reasoning | official | 0.79 → 16.78 | -0.65 [-1.34, +0.05] |
| Instruct 2B | TextVQA | reasoning | normalized | 0.79 → 16.78 | -0.65 [-1.34, +0.05] |
| Instruct 2B | TextVQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -6.89 [-7.83, -5.96] |
| Instruct 2B | TextVQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.90 [-7.84, -5.97] |
| Instruct 2B | TextVQA | short | official | 0.79 → 16.78 | -0.09 [-0.41, +0.22] |
| Instruct 2B | TextVQA | short | normalized | 0.79 → 16.78 | -0.10 [-0.41, +0.21] |
| Instruct 2B | VQAv2 | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.88 [-0.10, +1.89] |
| Instruct 2B | VQAv2 | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.78 [-0.19, +1.75] |
| Instruct 2B | VQAv2 | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.14 [-0.72, +0.44] |
| Instruct 2B | VQAv2 | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.16 [-0.74, +0.42] |
| Instruct → GRPO (Sep 13) | CLEVR counting | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -1.03 [-2.47, +0.37] |
| Instruct → GRPO (Sep 13) | CLEVR counting | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -1.03 [-2.47, +0.37] |
| Instruct → GRPO (Sep 13) | CLEVR counting | short | official | 0.79 → 0.37 (608×608 RL canvas) | +1.53 [+1.07, +2.03] |
| Instruct → GRPO (Sep 13) | CLEVR counting | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +1.53 [+1.07, +2.03] |
| Instruct → GRPO (Sep 13) | GQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.15 [-0.49, +0.80] |
| Instruct → GRPO (Sep 13) | GQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.23 [-0.37, +0.83] |
| Instruct → GRPO (Sep 13) | GQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.15 [-0.59, +0.29] |
| Instruct → GRPO (Sep 13) | GQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.15 [-0.59, +0.29] |
| Instruct → GRPO (Sep 13) | TextVQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -6.65 [-7.69, -5.62] |
| Instruct → GRPO (Sep 13) | TextVQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.65 [-7.69, -5.62] |
| Instruct → GRPO (Sep 13) | TextVQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -6.78 [-7.73, -5.87] |
| Instruct → GRPO (Sep 13) | TextVQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.79 [-7.73, -5.88] |
| Instruct → GRPO (Sep 13) | VQAv2 | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.95 [+0.07, +1.80] |
| Instruct → GRPO (Sep 13) | VQAv2 | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +1.21 [+0.36, +2.05] |
| Instruct → GRPO (Sep 13) | VQAv2 | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.05 [-0.61, +0.53] |
| Instruct → GRPO (Sep 13) | VQAv2 | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.07 [-0.63, +0.51] |
| SFT → GRPO (Sep 13) | CLEVR counting | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -12.20 [-13.50, -10.90] |
| SFT → GRPO (Sep 13) | CLEVR counting | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -12.20 [-13.50, -10.90] |
| SFT → GRPO (Sep 13) | CLEVR counting | short | official | 0.79 → 0.37 (608×608 RL canvas) | +0.37 [+0.07, +0.70] |
| SFT → GRPO (Sep 13) | CLEVR counting | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.37 [+0.07, +0.70] |
| SFT → GRPO (Sep 13) | GQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.02 [-0.56, +0.60] |
| SFT → GRPO (Sep 13) | GQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.14 [-0.44, +0.70] |
| SFT → GRPO (Sep 13) | GQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.13 [-0.55, +0.29] |
| SFT → GRPO (Sep 13) | GQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.13 [-0.55, +0.29] |
| SFT → GRPO (Sep 13) | TextVQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -6.09 [-7.16, -5.04] |
| SFT → GRPO (Sep 13) | TextVQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.10 [-7.16, -5.04] |
| SFT → GRPO (Sep 13) | TextVQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -6.34 [-7.26, -5.42] |
| SFT → GRPO (Sep 13) | TextVQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.35 [-7.27, -5.43] |
| SFT → GRPO (Sep 13) | VQAv2 | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -0.21 [-1.10, +0.67] |
| SFT → GRPO (Sep 13) | VQAv2 | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.35 [-1.20, +0.48] |
| SFT → GRPO (Sep 13) | VQAv2 | short | official | 0.79 → 0.37 (608×608 RL canvas) | +0.13 [-0.39, +0.63] |
| SFT → GRPO (Sep 13) | VQAv2 | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.13 [-0.39, +0.63] |
| SFT 2B | CLEVR counting | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -6.73 [-8.07, -5.43] |
| SFT 2B | CLEVR counting | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.73 [-8.07, -5.43] |
| SFT 2B | CLEVR counting | short | official | 0.79 → 0.37 (608×608 RL canvas) | +0.37 [+0.07, +0.70] |
| SFT 2B | CLEVR counting | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.37 [+0.07, +0.70] |
| SFT 2B | GQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -0.80 [-1.28, -0.34] |
| SFT 2B | GQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.68 [-1.15, -0.21] |
| SFT 2B | GQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.08 [-0.50, +0.33] |
| SFT 2B | GQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.08 [-0.50, +0.33] |
| SFT 2B | TextVQA | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | -5.25 [-6.40, -4.14] |
| SFT 2B | TextVQA | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | -5.26 [-6.41, -4.15] |
| SFT 2B | TextVQA | short | official | 0.79 → 0.37 (608×608 RL canvas) | -6.31 [-7.24, -5.38] |
| SFT 2B | TextVQA | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -6.32 [-7.25, -5.39] |
| SFT 2B | VQAv2 | reasoning | official | 0.79 → 0.37 (608×608 RL canvas) | +0.06 [-0.73, +0.86] |
| SFT 2B | VQAv2 | reasoning | normalized | 0.79 → 0.37 (608×608 RL canvas) | +0.08 [-0.71, +0.87] |
| SFT 2B | VQAv2 | short | official | 0.79 → 0.37 (608×608 RL canvas) | -0.03 [-0.55, +0.50] |
| SFT 2B | VQAv2 | short | normalized | 0.79 → 0.37 (608×608 RL canvas) | -0.03 [-0.55, +0.50] |
| SFT 2B (SLAKE) | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | -0.66 [-1.98, +0.66] |
| SFT 2B (SLAKE) | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | -0.75 [-2.07, +0.57] |
| SFT 2B (SLAKE) | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | -0.38 [-1.70, +0.94] |
| SFT 2B (SLAKE) | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | -0.38 [-1.70, +0.94] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | -0.85 [-2.26, +0.57] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | -0.85 [-2.26, +0.57] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.09, +2.36] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.09, +2.36] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | +0.94 [-1.13, +3.02] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | +0.85 [-1.23, +2.92] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.09, +2.36] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | +1.13 [-0.09, +2.36] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (English) | reasoning | official | ≤ 0.79 → 0.79 (upscaled) | +2.64 [+0.57, +4.71] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (English) | reasoning | normalized | ≤ 0.79 → 0.79 (upscaled) | +2.54 [+0.47, +4.62] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (English) | short | official | ≤ 0.79 → 0.79 (upscaled) | -0.09 [-1.23, +1.04] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (English) | short | normalized | ≤ 0.79 → 0.79 (upscaled) | +0.00 [-1.13, +1.13] |
<!-- /report:resolution -->

## Vision-side options (SLAKE)

Every model above trains only the language model; the vision encoder and its merger stay
frozen, and SLAKE's images (240 to 1024 pixels square) reach the encoder at 56 to 768 vision
tokens. Three options act on the vision side, each one change against a recorded 2B recipe,
declared in `experiments.toml` before training (`[pilots.slake-vision]`): SFT with LoRA also
on the vision blocks and merger (`--vision-lora`), SFT and short-prompt GRPO on images
upscaled to the pixel budget (`--image-upscale`, `--image-fit upscale`), and short-prompt GRPO
with a perception loss after PAPO (`--perception-coef 0.02`, `vqar/perception_trainer.py`),
which rewards answers that change when 60% of the image's patches are masked.

Expected, as declared: upscaling and vision LoRA help SLAKE (small images, a domain the
encoder saw little), and the perception loss makes answers depend more on the image. None of the
three improves SLAKE. The arms on its test set, and their differences from the recipe each one
changes (official scorer, 95% intervals over questions):

<!-- report:slake-vision-pilot -->
| Arm | Prompt | Accuracy | closed answers | open answers | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Instruct 2B | short | 49.76 (49.76) | 75.96 | 32.87 | 3 |
| Instruct 2B | reasoning | 40.34 (40.53) | 57.69 | 29.15 | 138 |
| SFT 2B (SLAKE) | short | 74.84 (74.84) | 82.93 | 69.61 | 3 |
| SFT 2B (SLAKE) | reasoning | 69.18 (69.27) | 81.73 | 61.09 | 25 |
| SFT 2B + vision LoRA (SLAKE) | short | 75.12 (75.12) | 83.17 | 69.92 | 3 |
| SFT 2B + vision LoRA (SLAKE) | reasoning | 66.07 (66.07) | 80.53 | 56.74 | 46 |
| SFT 2B, upscaled images (SLAKE) | short | 74.08 (74.08) | 81.97 | 68.99 | 3 |
| SFT 2B, upscaled images (SLAKE) | reasoning | 69.65 (69.75) | 82.21 | 61.55 | 21 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | short | 65.60 (65.60) | 80.05 | 56.28 | 2 |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | 52.78 (52.87) | 67.31 | 43.41 | 127 |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | short | 65.98 (66.07) | 81.25 | 56.12 | 2 |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | reasoning | 49.58 (49.67) | 62.50 | 41.24 | 153 |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | short | 55.98 (56.27) | 73.80 | 44.50 | 5 |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | reasoning | 35.16 (35.34) | 43.99 | 29.46 | 121 |
<!-- /report:slake-vision-pilot -->

<!-- report:slake-vision-pilot-deltas -->
| Comparison | Prompt | Official |
| --- | --- | ---: |
| SFT 2B + vision LoRA (SLAKE) − SFT 2B (SLAKE) | short | +0.28 [-0.85, +1.32] |
| SFT 2B + vision LoRA (SLAKE) − SFT 2B (SLAKE) | reasoning | -3.11 [-5.00, -1.23] |
| SFT 2B, upscaled images (SLAKE) − SFT 2B (SLAKE) | short | -0.75 [-1.89, +0.28] |
| SFT 2B, upscaled images (SLAKE) − SFT 2B (SLAKE) | reasoning | +0.47 [-0.75, +1.79] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) − Instruct → short-prompt GRPO (SLAKE, 1 epoch) | short | +0.38 [-1.32, +2.07] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) − Instruct → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | -3.20 [-5.56, -0.85] |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) − Instruct → short-prompt GRPO (SLAKE, 1 epoch) | short | -9.61 [-11.97, -7.26] |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) − Instruct → short-prompt GRPO (SLAKE, 1 epoch) | reasoning | -17.62 [-20.64, -14.61] |
<!-- /report:slake-vision-pilot-deltas -->

Each model against Instruct 2B on SLAKE and on the two unseen medical sets (normalized):

<!-- report:vision-transfer-short -->
| Arm | Trained on | SLAKE (English) | OmniMedVQA (open access) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 2B (SLAKE) | SLAKE | +25.07 [+21.87, +28.18] | -0.29 [-0.99, +0.38] | -1.83 [-2.83, -0.85] |
| SFT 2B + vision LoRA (SLAKE) | SLAKE (+ vision LoRA) | +25.35 [+22.15, +28.56] | +0.79 [+0.10, +1.48] | -2.34 [-3.33, -1.32] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (upscaled) | +24.32 [+21.11, +27.43] | -0.08 [-0.79, +0.61] | -2.51 [-3.56, -1.46] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE | +15.83 [+13.10, +18.47] | +0.03 [-0.34, +0.40] | -0.97 [-1.76, -0.18] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (upscaled) | +16.31 [+13.67, +18.94] | +0.13 [-0.21, +0.47] | -0.57 [-1.15, -0.00] |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | SLAKE (+ perception loss) | +6.50 [+3.49, +9.52] | -2.21 [-2.96, -1.47] | -9.29 [-10.37, -8.17] |
<!-- /report:vision-transfer-short -->

<!-- report:vision-transfer-reasoning -->
| Arm | Trained on | SLAKE (English) | OmniMedVQA (open access) | PathVQA |
| --- | --- | ---: | ---: | ---: |
| SFT 2B (SLAKE) | SLAKE | +28.75 [+25.45, +32.05] | +2.79 [+1.82, +3.80] | -5.31 [-6.38, -4.23] |
| SFT 2B + vision LoRA (SLAKE) | SLAKE (+ vision LoRA) | +25.54 [+22.15, +28.84] | -2.89 [-3.86, -1.87] | -9.19 [-10.26, -8.12] |
| SFT 2B, upscaled images (SLAKE) | SLAKE (upscaled) | +29.22 [+25.92, +32.42] | +1.82 [+0.88, +2.78] | -6.69 [-7.81, -5.60] |
| Instruct → short-prompt GRPO (SLAKE, 1 epoch) | SLAKE | +12.35 [+9.71, +15.08] | +2.09 [+1.23, +2.97] | -0.86 [-1.71, -0.03] |
| Instruct → short-prompt GRPO, upscaled images (SLAKE, 1 epoch) | SLAKE (upscaled) | +9.14 [+6.41, +11.88] | +3.16 [+2.24, +4.11] | +0.54 [-0.27, +1.34] |
| Instruct → short-prompt GRPO + perception loss (SLAKE, 1 epoch) | SLAKE (+ perception loss) | -5.18 [-8.39, -2.07] | -19.62 [-20.75, -18.47] | -22.52 [-23.62, -21.42] |
<!-- /report:vision-transfer-reasoning -->

**Upscaling.** Training on images upscaled to the pixel budget changes short-prompt SLAKE by
−0.8 points for SFT and +0.4 for GRPO, both intervals including zero, and costs GRPO 3.2 points
under the reasoning prompt. Upscaling only at test time does not help either: Instruct, SFT and
GRPO move by −0.4 to +1.1 points (the resolution table), and the upscaled GRPO model gains 2.6
under the reasoning prompt when tested on the upscaled images it trained on, which recovers its
own loss and no more. 79% of SLAKE's test questions come with images 512 pixels square or
larger (256 to 768 vision tokens); the answers do not seem limited by resolution. Upscaled
GRPO took 8.3 A30-hours for its epoch; the recipe takes 2.0 hours on the RTX 4090.

**Vision LoRA.** LoRA on the vision blocks and mergers leaves short-prompt SLAKE where SFT has
it (+0.3 [−0.9, +1.3]) and costs 3.1 points under the reasoning prompt. Off SLAKE it moves
OmniMedVQA up by 1.1 points over SFT under the short prompt and down by 5.7 under the
reasoning prompt, and costs PathVQA 0.5 and 3.9 more than SFT: adapting the encoder to SLAKE
narrows the model rather than sharpening its view.

**Perception loss.** At the weight PAPO uses (0.02), GRPO with the perception loss ends 9.6
points below plain GRPO on SLAKE under the short prompt and 17.6 under the reasoning prompt,
and falls 9.3 points below Instruct on PathVQA (normalized). Its training log shows how: the masked-image KL it maximizes
rose from 0.15 to 0.84 nats per token, and from about update 700 the responses grew from 2.6 to
13 tokens while the training reward fell from 0.6–0.7 to 0.45–0.6. The model found the
cheapest way to make its answer depend on the image: say the image is missing. With a black
image it now answers 3.7% of SLAKE correctly (Instruct 25.1%), replying "I can't see the image,
so I can't answer this question. Please upload a clear image again." or "Cannot be determined,
the image is missing." to most questions; with another question's image it answers as before
(8.6 points lost, against 8.5 for plain GRPO), and on real images "cannot be identified"
became one of its five most common answers (40 of 1,061). It learned to detect a missing image,
not to use the image more, and it moved four times as far from Instruct as plain GRPO (KL 8.3
against 2.1 nats). Our KL is estimated on the sampled tokens of 2-token answers, where PAPO
trained on long reasoning; a smaller weight, PAPO's entropy regularization, or a cap tied to
the answer length are the untested remedies.

One seed per arm; the reliance, distance and calibration of each model are in the
[beyond-accuracy](findings/what-the-gain-is.md#beyond-accuracy) tables. Compute: 16.9 cluster GPU-hours (one free RTX 6000
for the vision-LoRA SFT, free A30s for the rest) and 1.9 on the local RTX 4090 for the smoke
test and every evaluation.

## Transfer between training data

Each model trained on one dataset family was also scored on the other family's test sets,
against Instruct 2B: full test roles, official scorer, 95% intervals over questions
(`[transfer]` in experiments.toml; `python -m scripts.paired_deltas`). Under the reasoning
prompt:

<!-- report:transfer-reasoning -->
| Arm | Trained on | VQAv2 | GQA | TextVQA (far-OOD) | CLEVR counting | ChartQA (human) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Instruct → GRPO (ChartQA, 1 epoch) | ChartQA | +15.21 [+13.99, +16.40] | +15.94 [+15.11, +16.76] | +5.85 [+4.90, +6.80] | +2.67 [+1.27, +4.07] | +10.40 [+8.16, +12.64] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) | ChartQA | +4.83 [+3.79, +5.88] | +6.58 [+5.87, +7.31] | +0.86 [-0.03, +1.80] | +0.53 [-0.67, +1.77] | +5.92 [+3.68, +8.08] |
| Instruct → GRPO | VQAv2 + GQA | +23.22 [+21.95, +24.47] | +24.10 [+23.17, +24.98] | +10.24 [+9.17, +11.29] | +12.13 [+10.90, +13.37] | -6.72 [-9.68, -3.84] |
| Instruct → short-prompt GRPO | VQAv2 + GQA | +6.80 [+5.85, +7.78] | +6.76 [+6.07, +7.43] | +2.60 [+1.75, +3.45] | -0.63 [-1.90, +0.60] | +0.48 [-1.20, +2.16] |
| SFT 2B | VQAv2 + GQA | +12.93 [+11.59, +14.28] | +20.47 [+19.56, +21.36] | -0.89 [-2.05, +0.29] | +4.97 [+3.57, +6.30] | -7.92 [-10.80, -5.12] |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) | VQAv2 + GQA + ChartQA | +23.68 [+22.40, +24.95] | +23.30 [+22.40, +24.19] | +9.99 [+8.93, +11.02] | +8.77 [+7.43, +10.10] | -3.76 [-6.80, -0.80] |
<!-- /report:transfer-reasoning -->

Under the short prompt:

<!-- report:transfer-short -->
| Arm | Trained on | VQAv2 | GQA | TextVQA (far-OOD) | CLEVR counting | ChartQA (human) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Instruct → GRPO (ChartQA, 1 epoch) | ChartQA | -0.25 [-0.57, +0.07] | -0.02 [-0.25, +0.22] | +0.12 [-0.25, +0.49] | -0.60 [-0.90, -0.33] | +0.88 [-0.16, +2.00] |
| Instruct → short-prompt GRPO (ChartQA, 1 epoch) | ChartQA | -0.35 [-0.74, +0.02] | +0.05 [-0.21, +0.31] | +0.16 [-0.31, +0.63] | +1.77 [+1.30, +2.27] | +1.68 [+0.32, +3.04] |
| Instruct → GRPO | VQAv2 + GQA | +0.07 [-0.32, +0.47] | +0.60 [+0.29, +0.91] | +0.49 [+0.05, +0.94] | +1.80 [+1.30, +2.33] | +0.00 [-0.88, +0.88] |
| Instruct → short-prompt GRPO | VQAv2 + GQA | +0.75 [+0.27, +1.23] | +0.89 [+0.48, +1.30] | +0.48 [-0.00, +0.97] | +1.83 [+1.33, +2.37] | +0.56 [-0.40, +1.52] |
| SFT 2B | VQAv2 + GQA | +2.00 [+1.36, +2.67] | +2.89 [+2.34, +3.47] | -0.45 [-1.11, +0.22] | +1.73 [+1.27, +2.20] | -0.16 [-1.60, +1.28] |
| Instruct → GRPO (VQAv2 + GQA + ChartQA) | VQAv2 + GQA + ChartQA | -0.10 [-0.52, +0.30] | +0.41 [+0.08, +0.72] | +0.16 [-0.30, +0.62] | +1.90 [+1.40, +2.43] | +1.20 [+0.00, +2.40] |
<!-- /report:transfer-short -->

What GRPO teaches here is a response habit, and the habit travels. Trained on ChartQA with
reasoning, the model keeps reasoning on photos, shorter and more accurate: 66–83 tokens on
VQAv2, GQA and TextVQA against Instruct's 107–148, and +15.2 points on VQAv2, +15.9 on GQA
and +5.9 on TextVQA under the reasoning prompt. Trained on VQAv2 and GQA, GRPO answers in
8–10 tokens under the reasoning prompt on every test set; the habit pays on photos (+23 to
+24 points on its training datasets) and costs 6.7 points on ChartQA ([−9.7, −3.8]), where
reasoning pays. SFT on the same VQA questions, at 36 tokens on ChartQA, costs 7.9 there
([−10.8, −5.1]). Under the short prompt no model loses more than 0.6 points on the other
family's test sets, so training on one family left the other's short answers intact. Each
model still leads on its own data: under the reasoning prompt the VQA-trained model scores
78.3 on VQAv2 against 70.3 for the ChartQA-trained one, and 63.0 on ChartQA against 80.1.
One model trained on both families at once did not learn when to reason: it kept chart
reasoning for 1,750 updates, then dropped reasoning everywhere
([one model](findings/one-model.md#one-model-on-both-families)).
