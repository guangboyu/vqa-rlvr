Screening battery on each candidate's training pool (normalized scorer; differences in points with paired 95% bootstrap intervals).

Gates: at least 2,000 training questions; split separable and license usable (declared); parse rate ≥ 95% of sampled reasoning responses; short-prompt accuracy < 90%; stepwise not worse than direct (upper bound ≥ 0); black and swapped images each cost ≥ 10 points (lower bound); learnable share ≥ 25%. Ranking: passed candidates by matched payoff, then RL headroom, both descending.

| Model | Rank | Dataset | Pool | Questions | Short (%) | Reasoning payoff | Matched payoff | pass@1 → pass@k (%) | RL headroom | Learnable (%) | Image: black | Image: swapped | Parse (%) | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base8 | — | MedThinkVQA | rl | 500 | 49.6 | -42.4 [-46.8, -37.8] | -34.6 [-38.8, -30.4] | 15.3 → 41.4 | +26.1 [+23.0, +29.2] | 39.2 | +2.6 [-1.0, +6.2] | +6.2 [+2.8, +9.6] | 27.0 | rejected (failed: parse rate, reasoning not worse, image needed) |
