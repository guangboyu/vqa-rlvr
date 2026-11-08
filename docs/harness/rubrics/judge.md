# Judge and audit rubric

Version 1, the rubric behind every LLM-graded number in this repository. The prompts below are copied
from the code that sends them; a test fails if either drifts.

## When the judge runs

Only at milestones, and only on headline cells. Development decisions use the rule-based scorers.

## The judge

Model and prompt: `vqar/judge.py` (`MODEL`, `PROMPT`), text only, temperature 0, one word back.

```
Question: {question}
Gold answer(s): {golds}
Candidate answer: {prediction}

Does the candidate answer mean the same thing as one of the gold answers, in the context of the question? Ignore differences in casing, phrasing, or specificity that do not change the meaning. Reply with exactly one word: yes or no.
```

- **Input:** exact-match misses only, the extracted answer, and every reference. Partial-credit answers are not judged.
- **Controls:** 100 seeded exact-match hits per cell, to measure how often the judge rejects a right answer.
- **Cache:** one verdict per (model, question, sorted references, normalized answer), so a rerun costs nothing.

## The audit

Model and prompt: `scripts/audit_judge.py` (`AUDIT_MODEL`, `PROMPT`), shown the image.

```
You are checking an automatic grader of visual question answering.
Question: {question}
Reference answers from human annotators (they sometimes disagree): {golds}
Candidate answer: {prediction}

Looking at the image, is the candidate a correct final answer to the question, one a careful human grader would accept? Accept synonyms, paraphrases, singular or plural, and capitalization differences. Do not accept an answer that is wrong for the image, only partly right, or unfinished reasoning instead of a final answer. Reply with exactly one word: yes or no.
```

- **Sample:** 50 seeded verdicts per stratum. Strata: dataset × judge verdict, and for acceptances, answer length (1 word, 2–5, 6 or more).
- **Use:** an accepted miss counts at its stratum's confirmation rate (the audited score); rates come with Wilson 95% intervals.

## Scores

| Scorer | Rule |
|---|---|
| Official | The dataset authors' evaluator |
| Adjusted | Official, plus every miss the judge accepts at full credit |
| Audited | Official, plus each accepted miss weighted by its stratum's audit confirmation rate |
