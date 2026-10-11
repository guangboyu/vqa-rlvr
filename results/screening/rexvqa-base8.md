Screening battery on each candidate's training pool (normalized scorer; differences in points with paired 95% bootstrap intervals).

Gates for any RL: at least 2,000 training questions; split separable and license usable (declared); short-prompt accuracy < 90%; black and swapped images each cost ≥ 10 points (lower bound). Routes: reasoning-prompt RL needs parse rate ≥ 95% of sampled reasoning responses, stepwise not worse than direct (upper bound ≥ 0) and a learnable share ≥ 25%; short-answer RL needs a short-prompt learnable share ≥ 25%. A candidate passes with the common gates and at least one viable route. Ranking: passed candidates by matched payoff, then RL headroom, both descending.

| Model | Rank | Dataset | Pool | Questions | Short (%) | Reasoning payoff | Matched payoff | pass@1 → pass@k (%) | RL headroom | Learnable (%) | Image: black | Image: swapped | Parse (%) | Short learnable (%) | Routes | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| base8 | — | ReXVQA (frontal view) | rl | 500 | 73.8 | -11.0 [-14.6, -7.6] | -7.2 [-11.2, -3.2] | 62.6 → 82.8 | +20.2 [+17.6, +22.7] | 41.8 | +8.4 [+4.6, +12.2] | +6.4 [+3.2, +9.8] | 91.0 | 17.0 | none | rejected (failed: image needed) |
