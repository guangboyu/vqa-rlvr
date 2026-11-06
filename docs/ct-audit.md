# Do 3D CT benchmarks need the scan?

*A note separate from the [main study](https://guangboyu.github.io/vqa-rlvr/report/), using its harness.*

How much of two 3D CT benchmarks can be answered without looking? Accuracy (%) of 2B models trained
with SFT; "without the image" means black images at test time (and in training, except for
DeepTumorVQA multiple choice):

| Benchmark | Text-only rule | Without the image | With the image | Published fine-tuned model |
| --- | ---: | ---: | ---: | ---: |
| DeepTumorVQA v2, multiple choice | <!--n:au.mc.rule-->60.7<!--/n--> | <!--n:au.mc.blind-->59.0<!--/n--> | <!--n:au.mc.sft-->61.4<!--/n--> | 64.6 |
| DeepTumorVQA v2, free form | <!--n:au.ff.rule-->57.4<!--/n--> | <!--n:au.ff.blind-->57.6<!--/n--> | <!--n:au.ff.sft-->60.3<!--/n--> | 63.4 |
| M3D-VQA, mean over question types | — | <!--n:au.m3d.blind.mean-->70.2<!--/n--> | — | <!--n:au.m3d.pub-->75.8<!--/n--> |

DeepTumorVQA's answer options and answer frequencies give most answers away, and M3D-VQA's correct
option is A in nearly half of its questions. A debiased DeepTumorVQA, built here, brings the text-only
rule to chance (<!--n:au.x.rule-->35.2<!--/n-->% against <!--n:au.x.chance-->33.7<!--/n-->%). On it GRPO and SFT reach the same accuracy
(<!--n:au.x.grpo-->36.4<!--/n-->% and <!--n:au.x.sft-->36.7<!--/n-->%), and the one thing a 2B model learns from 2D views of the scan is whether a
lesion is present.

## Shortcut audit of 3D CT benchmarks

Two 3D CT benchmarks, DeepTumorVQA v2 (abdominal CT, 2D views of each scan) and M3D-VQA (CT and
MRI from Radiopaedia, questions only here), measured for how much of their published accuracy is
reachable without the scan, then DeepTumorVQA rebuilt without its option shortcuts to ask what a
2B model and RL learn from the scan itself. Every run was declared in `experiments.toml` before it
trained; text-only rules come from `python -m scripts.option_priors --dataset <name>` and write
`results/screening/<name>-option-prior.json`. Accuracy (%), one training seed, 2B models trained
on 40,000 questions for one epoch unless noted.

| Benchmark, test set | Text-only rule | SFT, without the image | SFT, with the image | Published fine-tuned |
| --- | ---: | ---: | ---: | ---: |
| DeepTumorVQA, multiple choice (9,996) | 60.7 | 59.0 (the sighted model given black images) | 61.4 (8,000 questions) | 64.6 (Qwen3-VL-4B), 66.3 |
| DeepTumorVQA, free form (10,000) | 57.4 | 57.6 (trained and tested on black images) | 60.3 | 63.4 (Qwen3-VL-4B) |
| M3D-VQA, multiple choice (13,790) | 50.6 | 68.7, mean of types 70.2 (trained and tested on black images) | — | 75.8 (M3D-LaMed, mean of types) |

The DeepTumorVQA rule reads the options: its distractors for a continuous value are drawn around
the true value (0.7 to 1.3 times it), so the true value is the second or third smallest of four
options 88% of the time, and binary and categorical answers follow the patients' prevalence. The
free-form rule answers each subtype and organ with the median (numbers) or most common (text)
training answer, scored with the authors' free-form rules (mean relative accuracy for numbers,
exact match for text). In free form the image adds 2.7 points [+2.1, +3.3] for SFT (sighted
model with real images against the blind model with black ones), almost all of it on lesion
existence (78.9% against 69.8%). The M3D-VQA rule is the most common answer letter per question
type: A is correct in 47.5% of its test questions. By type, the blind M3D model scores 86.4
(plane), 65.0 (phase), 72.4 (organ), 66.0 (abnormality) and 61.0 (location) against M3D-LaMed's
98.80, 79.75, 74.75, 66.65 and 58.94 with the volume: abnormality and location questions are
answered as well without the image.

With the shortcuts removed (`deeptumorx`: options regenerated so the true value sits at a uniformly
random rank, every option list shuffled, discrete answers balanced within each subtype; 6,607 test
questions, text-only rule 35.2% against 33.7% chance), little is left for a 2B model to learn from
2D views of the scan (strict letter scorer; Instruct 34.6% under the lenient scorer, which also
accepts option text):

| Model on debiased DeepTumorVQA | Montage (9 slices) | Black image | 32-slice grid | Black grid |
| --- | ---: | ---: | ---: | ---: |
| SFT trained on the released options | 36.0 | — | — | — |
| SFT, 8,000 debiased questions | 36.7 | 34.4 | 36.6 | 34.3 |
| Instruct → GRPO, 2,000 updates | 36.4 | 33.7 | — | — |
| SFT → GRPO, 2,000 updates | 37.5 | 33.9 | — | — |

The 32-slice grid (evenly spaced axial slices of the released CT video, 2,048 vision tokens)
does no better than the nine-slice montage (36.6% against 36.7%), short of the declared 3-point
margin, so RL trained on the montage. GRPO with a reward for the option letter alone (on the same
8,000-question exposure) matches SFT: −0.3 [−1.6, +1.0] from Instruct and +0.7 [−0.4, +1.8] after
SFT, both inside the noise, so RL does not beat SFT here. Every model's one visual skill is lesion
existence: SFT 60.7%, GRPO 57.1%, SFT → GRPO 60.5%, each about 51% with a black image (chance
50%); measurement stays at chance (24.7–26.2% against 25%). Training reward for both GRPO runs
hovered around 0.3–0.5 against 0.34 for guessing.
