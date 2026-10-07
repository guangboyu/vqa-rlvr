Screening battery on each candidate's training pool (normalized scorer; differences in points with paired 95% bootstrap intervals).

Gates: at least 2,000 training questions; split separable and license usable (declared); parse rate ≥ 95% of sampled reasoning responses; short-prompt accuracy < 90%; stepwise not worse than direct (upper bound ≥ 0); black and swapped images each cost ≥ 10 points (lower bound); learnable share ≥ 25%. Ranking: passed candidates by matched payoff, then RL headroom, both descending.

| Model | Rank | Dataset | Pool | Questions | Short (%) | Reasoning payoff | Matched payoff | pass@1 → pass@k (%) | RL headroom | Learnable (%) | Image: black | Image: swapped | Parse (%) | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base2 | 1 | ChartQA (human) | rl | 500 | 69.0 | +2.0 [-2.0, +6.2] | +0.0 [-3.2, +3.2] | 71.0 → 88.6 | +17.6 [+15.3, +20.0] | 41.2 | +53.2 [+48.6, +58.0] | +54.0 [+49.4, +58.6] | 98.0 | passed |
| base2 | — | CLEVR counting | rl | 500 | 98.0 | -9.8 [-13.0, -7.0] | -3.2 [-6.8, +0.2] | 84.9 → 100.0 | +15.1 [+13.4, +17.0] | 48.4 | +98.0 [+96.8, +99.2] | +86.2 [+83.0, +89.4] | 97.2 | rejected (failed: not saturated) |
| base2 | — | CLEVR compositional | rl | 500 | 88.4 | -15.0 [-19.2, -10.8] | -5.6 [-9.2, -2.0] | 70.9 → 94.8 | +23.9 [+21.5, +26.3] | 62.0 | +55.6 [+50.4, +60.6] | +47.2 [+42.0, +52.2] | 94.2 | rejected (failed: parse rate, reasoning not worse) |
| base2 | — | GQA | rl | 500 | 69.6 | -11.8 [-15.4, -8.2] | -9.2 [-12.4, -6.0] | 56.3 → 73.4 | +17.1 [+14.8, +19.5] | 42.0 | +35.6 [+30.6, +40.6] | +37.6 [+32.6, +42.4] | 97.4 | rejected (failed: reasoning not worse) |
| base2 | — | TextVQA | test (validation only) | 500 | 79.8 | -10.6 [-14.1, -7.1] | -3.3 [-6.1, -0.5] | 60.4 → 77.0 | +16.6 [+14.3, +18.9] | 42.0 | +75.9 [+72.2, +79.5] | +77.1 [+73.6, +80.6] | 96.4 | rejected (failed: training questions, reasoning not worse) |
| base2 | — | VQAv2 | rl | 500 | 73.2 | -12.8 [-16.6, -9.2] | -7.6 [-10.6, -4.6] | 60.9 → 80.2 | +19.3 [+16.9, +21.8] | 47.8 | +44.8 [+39.6, +49.8] | +44.6 [+39.8, +49.4] | 97.7 | rejected (failed: reasoning not worse) |
