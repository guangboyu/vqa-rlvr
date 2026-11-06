"""Unbiased pass@k from sampled records, beside the mean score of the same samples.

With n samples per question of which c are fully correct, pass@k = 1 − C(n−c, k) / C(n, k)
(the estimator of Chen et al., 2021), averaged over questions. It is an oracle quantity: the
chance that at least one of k draws is right when a grader can tell which one. It is not a
deployable best-of-k, and sampled pass@1 is not greedy accuracy. Success means full credit,
so a VQA answer that matches only some annotators adds to the mean score but never to pass@k.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from vqar.eval.paired import paired_bootstrap
from vqar.eval.record import RunRecord

FULL_CREDIT = 1.0 - 1e-12
VARIANTS = ("official", "normalized", "reward")  # reward: the GRPO correctness reward
COHORT_FIELDS = ("question", "answers", "image_qid")


def estimate_pass_at_k(n: int, correct: int, k: int) -> float:
    """Unbiased count estimate 1 − C(n−correct,k)/C(n,k), for 1 <= k <= n."""
    if (any(type(value) is not int for value in (n, correct, k))
            or n < 1 or not 0 <= correct <= n or not 1 <= k <= n):
        raise ValueError("Pass@k requires integer n>=1, 0<=correct<=n and 1<=k<=n")
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def default_ks(n: int) -> list[int]:
    """1, 2, 4, ... up to n, and n itself."""
    ks = [2**i for i in range(n.bit_length()) if 2**i <= n]
    return ks if ks[-1] == n else [*ks, n]


@dataclass(frozen=True)
class SampledCell:
    """One sampled record reduced to per-question outcomes under one scorer variant."""

    path: str
    model: dict
    protocol: dict
    variant: str
    samples: int
    outcomes: dict[str, tuple[int, float]]  # qid -> (fully correct samples, mean score)
    cohort: dict[str, tuple]  # qid -> (question, references, image id)

    def pass_at_k(self, k: int) -> dict[str, float]:
        return {qid: estimate_pass_at_k(self.samples, correct, k)
                for qid, (correct, _) in self.outcomes.items()}

    def mean_scores(self) -> dict[str, float]:
        return {qid: mean for qid, (_, mean) in self.outcomes.items()}

    def summary(self, ks: list[int]) -> dict:
        count = len(self.outcomes)
        return {"record": self.path, "model": self.model["label"], "variant": self.variant,
                "questions": count, "samples": self.samples,
                "mean_score": 100 * sum(self.mean_scores().values()) / count,
                "pass_at_k": {k: 100 * sum(self.pass_at_k(k).values()) / count for k in ks}}


def load_sampled(path: Path, variant: str = "official") -> SampledCell:
    """Read a sampled record; every question needs samples 0..n-1 sharing one cohort entry."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    record = RunRecord.load(Path(path))
    protocol, n = record.protocol, record.protocol["samples"]
    if protocol["temperature"] <= 0 or n < 2:
        raise ValueError(f"{path}: pass@k needs a sampled record (temperature > 0, n >= 2)")
    groups = defaultdict(list)
    for row in record.predictions():
        groups[row["qid"]].append(row)
    outcomes, cohort = {}, {}
    for qid, rows in groups.items():
        if sorted(row["sample"] for row in rows) != list(range(n)):
            raise ValueError(f"{path}: question {qid} lacks samples 0..{n - 1}")
        fields = {tuple(_hashable(row[key]) for key in COHORT_FIELDS) for row in rows}
        if len(fields) != 1:
            raise ValueError(f"{path}: question {qid} changes its question, references or image")
        scores = _scores(rows, variant, protocol)
        outcomes[qid] = (sum(score >= FULL_CREDIT for score in scores), sum(scores) / n)
        cohort[qid] = fields.pop()
    return SampledCell(str(path), record.model, protocol, variant, n, outcomes, cohort)


def _hashable(value):
    return tuple(value) if isinstance(value, list) else value


def _scores(rows: list[dict], variant: str, protocol: dict) -> list[float]:
    if variant == "reward":
        from vqar.rewards import correctness_reward

        return correctness_reward(
            prompts=None, completions=[row["raw"] for row in rows],
            answers=[row["answers"] for row in rows], dataset=[protocol["dataset"]] * len(rows),
            rollout_prompt=[protocol["prompt"]] * len(rows))
    scores = [row["scores"][variant] for row in rows]
    if any(score is None for score in scores):
        raise ValueError(f"the {variant} scorer is not defined for this record")
    return scores


def paired_pass_at_k(reference: SampledCell, candidate: SampledCell, ks: list[int], *,
                     image_groups: dict[str, str] | None = None, resamples: int = 10_000,
                     seed: int = 0) -> list[dict]:
    """Candidate minus reference for the mean score and each pass@k, in points.

    Give each question's image key to resample images, since questions on one image are not
    independent; every estimate still weights questions equally. Sidecars don't store the key
    (their image id is the question's own id outside swapped mode), so the caller supplies it.
    """
    if reference.protocol != candidate.protocol:
        raise ValueError("Paired pass@k needs the same protocol in both records")
    if reference.cohort != candidate.cohort or reference.variant != candidate.variant:
        raise ValueError("Paired pass@k needs identical questions, references and images")
    metrics = {"mean_score": (reference.mean_scores(), candidate.mean_scores())}
    metrics |= {f"pass@{k}": (reference.pass_at_k(k), candidate.pass_at_k(k)) for k in ks}
    return [{"metric": name, **paired_bootstrap(*pair, resamples=resamples, seed=seed,
                                                image_groups=image_groups)}
            for name, pair in metrics.items()]
