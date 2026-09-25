"""Paired comparisons of two records on the same questions: the difference and its CI.

Both records must have scored the same questions. Each question contributes the mean of
its samples' scores, and the bootstrap resamples questions, so the interval reflects which
questions were asked, not training-seed variation.
"""

import json
from pathlib import Path

RESAMPLES = 10_000
SEED = 0


def question_scores(predictions: Path, variant: str) -> dict[str, float]:
    """Each question's mean score under one scorer variant, from a predictions sidecar."""
    totals: dict[str, list[float]] = {}
    for line in predictions.read_text().splitlines():
        row = json.loads(line)
        totals.setdefault(row["qid"], []).append(row["scores"][variant])
    return {qid: sum(scores) / len(scores) for qid, scores in totals.items()}


def paired_bootstrap(reference: dict[str, float], candidate: dict[str, float],
                     resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """candidate − reference in accuracy points, with a percentile 95% bootstrap interval."""
    import numpy as np

    if reference.keys() != candidate.keys():
        raise ValueError("A paired comparison needs the same questions in both records.")
    qids = sorted(reference)
    differences = np.array([candidate[q] - reference[q] for q in qids])
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 500):  # chunks keep memory small on 12k questions
        stop = min(start + 500, resamples)
        draws = rng.integers(0, len(qids), size=(stop - start, len(qids)))
        means[start:stop] = differences[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"n": len(qids), "delta": 100 * float(differences.mean()),
            "low": 100 * float(low), "high": 100 * float(high),
            "resamples": resamples, "seed": seed}
