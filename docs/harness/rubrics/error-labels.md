# Error-label rubric

Version 1. Labels say why an answer is wrong. They are diagnosis, never a score: no table number
rests on them without a hand-checked sample beside it.

## What the labeler sees

The image, the question, every reference answer, the full response and the extracted answer. Never
the model's name or training. One answer at a time.

## Rules first

Code labels these before any model reads the answer, and the labeler does not relabel them:
truncated at the token budget, no closing answer tag under a tagged prompt, empty answer, scorer-only
miss (right under the normalized scorer, wrong under the official one).

## Labels, in decision order

Take the first label that fits.

| Label | The answer is wrong because | Example |
|---|---|---|
| `label_noise` | The reference is wrong or ambiguous, and the answer is defensible from the image | Reference "2", the image clearly shows 3 |
| `question_mismatch` | It answers a different question, or the wrong type of answer (a number for yes/no, a letter for an open question, the wrong unit) | "Yes" to "How many?" |
| `prior_answer` | It gives the usual answer for the question, which the image contradicts | "Yellow" for a purple banana |
| `hallucination` | It asserts something absent from the image | Names a dog in a photo with no animal |
| `perception_misread` | It misreads something plainly visible: text, a number, a color, an object | Reads 1984 on a chart label that says 1964 |
| `detail_error` | Right region or object, wrong fine detail: a count off by one, a nearby value, part of the answer | "3 people" when a fourth stands at the edge |
| `reasoning_error` | The reading is right but the computation or logic is wrong | Reads 40 and 25, answers 25 for their difference |
| `knowledge_error` | It needs outside knowledge and gets it wrong | Wrong name for a landmark it does recognize |
| `other` | None of the above; say why in the evidence field | |

## Output

One JSON object per answer:

```json
{"label": "detail_error", "evidence": "counts 3; a fourth person is cut off at the right edge", "confidence": "high"}
```

`confidence` is `high`, `medium` or `low`. Evidence is one sentence about the image, not about the model.

## Before shares are reported

1. Hand-label 50 sampled wrong answers with this rubric, before reading the model's labels.
2. Compare: Cohen's kappa of at least 0.6 overall, and no label with fewer than half its hand cases agreed.
3. If either fails, sharpen the definitions here, raise the version, and label again.
4. Report shares with their kappa and sample size.
