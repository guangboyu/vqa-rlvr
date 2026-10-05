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
                     resamples: int = RESAMPLES, seed: int = SEED,
                     image_groups: dict[str, str] | None = None) -> dict:
    """candidate − reference in accuracy points, with a percentile 95% bootstrap interval.

    Optional image groups change the resampling unit; every mean still weights questions.
    """
    import numpy as np

    if reference.keys() != candidate.keys():
        raise ValueError("A paired comparison needs the same questions in both records.")
    qids = sorted(reference)
    differences = np.array([candidate[q] - reference[q] for q in qids])
    if not qids:
        raise ValueError("A paired comparison needs at least one question.")
    group_sizes = None
    if image_groups is not None:
        if any(not key for key in image_groups.values()):
            raise ValueError("Every question needs a nonempty image group ID.")
        if image_groups.keys() != reference.keys():
            raise ValueError("Image groups must cover exactly the compared questions.")
        group_names = sorted(set(image_groups.values()))
        group_indices = {key: i for i, key in enumerate(group_names)}
        assignments = np.array([group_indices[image_groups[q]] for q in qids])
        group_totals = np.bincount(assignments, weights=differences)
        group_sizes = np.bincount(assignments)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 500):  # chunks keep memory small on 12k questions
        stop = min(start + 500, resamples)
        if group_sizes is None:
            draws = rng.integers(0, len(qids), size=(stop - start, len(qids)))
            means[start:stop] = differences[draws].mean(axis=1)
        else:
            draws = rng.integers(0, len(group_sizes), size=(stop - start, len(group_sizes)))
            means[start:stop] = group_totals[draws].sum(axis=1) / group_sizes[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    result = {"n": len(qids), "delta": 100 * float(differences.mean()),
            "low": 100 * float(low), "high": 100 * float(high),
            "resamples": resamples, "seed": seed}
    if group_sizes is not None:
        result.update(image_groups=len(group_sizes), bootstrap_unit="image group",
                      point_estimate_unit="question")
    return result
