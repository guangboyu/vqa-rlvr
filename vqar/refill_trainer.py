"""GRPO that refills dead groups (DAPO dynamic sampling) on TRL 1.8.

A separate module because it imports TRL at load time: train_grpo imports it only when
a run refills, and dry runs and CPU checks without TRL never do.
"""

import json
from itertools import islice
from pathlib import Path

import torch
from trl import GRPOTrainer
from trl.data_utils import prepare_multimodal_messages
from trl.trainer.utils import nanstd

from vqar.pools import RefillStream
from vqar.rl import ScoredGroup, fill_live_groups, refill_draw_records


class RefillGRPOTrainer(GRPOTrainer):
    """GRPO whose generation batches hold as many live groups as refills can find.

    TRL 1.8 samples a generation batch, scores it, and turns it into advantages inside one
    method, `_generate_and_score_completions`. This subclass chooses the batch first: it
    samples and scores candidate groups with TRL's own `_generate` and `_calculate_rewards`,
    replacing dead groups from the refill stream (`vqar.rl.fill_live_groups`). Then it runs
    TRL's method unchanged on the chosen rows, and its overrides of those two methods hand
    back the answers and rewards already sampled for them instead of sampling again. So
    TRL's policy log-probabilities, vLLM importance-sampling correction, advantages, loss
    normalization, and metrics all see exactly the batch trained on, in the usual shape.
    """

    def __init__(self, *args, refill: RefillStream, refill_rounds: int, **kwargs):
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1 or self.tools or self.environment_factories:
            raise ValueError("Refilling dead groups supports one process without tools.")
        if self.multi_objective_aggregation != "sum_then_normalize":
            raise ValueError("Refilling reads live groups from the weighted reward sum.")
        self.refill, self.refill_rounds = refill, refill_rounds
        self.draws_path: Path | None = None
        self._chosen: list[ScoredGroup] | None = None  # the batch TRL's pass trains on

    def record_draws_to(self, path: Path) -> None:
        """Log every round's draws there, with the rows kept (vqar.training reads them)."""
        self.draws_path = path

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)
        size, step = self.num_generations, self.state.global_step
        questions = inputs[::size]
        if any(row["qid"] != inputs[i // size * size]["qid"] for i, row in enumerate(inputs)):
            raise ValueError("A generation batch must hold whole groups of repeated questions.")
        stream = self.refill.draws(step, exclude=[(q["dataset"], q["qid"]) for q in questions])

        def draw(count):
            return [self.train_dataset[index] for index in islice(stream, count)]

        rounds, batch = fill_live_groups(questions, draw, self._score_groups, self.refill_rounds)
        self._record_refill(step, rounds, batch)
        self._chosen = batch
        try:
            rows = [dict(group.question) for group in batch for _ in range(size)]
            return super()._generate_and_score_completions(rows)
        finally:
            self._chosen = None

    def _score_groups(self, questions: list[dict]) -> list[ScoredGroup]:
        """Sample and score a group of answers per question, as TRL's own pass would."""
        size = self.num_generations
        rows = [question for question in questions for _ in range(size)]
        # TRL's prompt preparation for rows with one "image" column.
        prompts = [prepare_multimodal_messages(row["prompt"], images=[row["image"]])
                   for row in rows]
        # The chat rollout sleeps vLLM at level 2, which discards its weights, and TRL syncs
        # them only when global_step changes. Force the sync so every round samples from the
        # current policy, exactly as an update's first generation does.
        self._last_loaded_step = -1
        (prompt_ids, completion_ids, tool_mask, completions, _, logprobs, extra, images,
         _) = super()._generate(prompts)
        if tool_mask is not None or extra or images is not None or logprobs is None:
            raise ValueError("Refilling needs a rollout that returns only token IDs and logprobs.")
        rewards = super()._calculate_rewards(rows, prompts, completions, completion_ids)
        total = (rewards * self.reward_weights.to(rewards.device)).nansum(dim=1)
        total[torch.isnan(rewards).all(dim=1)] = torch.nan  # unscorable, as TRL marks them
        spread = nanstd(total.view(-1, size), dim=1)
        live = ~torch.isclose(spread, torch.zeros_like(spread))  # TRL's zero-std test
        groups = []
        for number, question in enumerate(questions):
            part = slice(number * size, (number + 1) * size)
            groups.append(ScoredGroup(
                question=question, scores=rewards[part, 0].tolist(), live=bool(live[number]),
                rollout={"prompt_ids": prompt_ids[part], "completion_ids": completion_ids[part],
                         "completions": completions[part], "logprobs": logprobs[part],
                         "rewards": rewards[part]}))
        return groups

    def _generate(self, prompts):
        if self._chosen is None:
            return super()._generate(prompts)
        rollouts = [group.rollout for group in self._chosen]

        def joined(name):
            return [item for rollout in rollouts for item in rollout[name]]

        completion_ids = joined("completion_ids")
        # The DAPO loss divides by the batch's completion tokens (TRL's num_items_in_batch).
        tokens = torch.tensor(sum(map(len, completion_ids)), device=self.accelerator.device)
        return (joined("prompt_ids"), completion_ids, None, joined("completions"), tokens,
                joined("logprobs"), {}, None, [])

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        if self._chosen is None:
            return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        return torch.cat([group.rollout["rewards"] for group in self._chosen])

    def _record_refill(self, step: int, rounds: list[list[ScoredGroup]],
                       batch: list[ScoredGroup]) -> None:
        """Per update: groups attempted, live groups kept, extra rounds used."""
        metrics = self._metrics["train"]
        metrics["refill/attempted_groups"].append(sum(map(len, rounds)))
        metrics["refill/live_groups"].append(sum(group.live for group in batch))
        metrics["refill/rounds"].append(len(rounds) - 1)
        if self.draws_path is not None:
            with self.draws_path.open("a") as file:
                for record in refill_draw_records(step, rounds, batch):
                    file.write(json.dumps(record) + "\n")
