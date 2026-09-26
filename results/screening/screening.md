Screening battery on each candidate's training pool (normalized scorer; differences in points with paired 95% bootstrap intervals).

Gates: at least 2,000 training questions; split separable and license usable (declared); parse rate ≥ 95% of sampled reasoning responses; short-prompt accuracy < 90%. Ranking: passed candidates by reasoning payoff, then RL headroom, both descending.

| Model | Rank | Dataset | Pool | Questions | Reasoning payoff | Short (%) | pass@1 → pass@k (%) | RL headroom | Learnable (%) | Image dependence | Parse (%) | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base2 | 1 | GQA | rl | 500 | -11.8 [-15.4, -8.2] | 69.6 | 56.3 → 73.4 | +17.1 [+14.8, +19.5] | 42.0 | +35.6 [+30.6, +40.6] | 97.4 | passed |
| base2 | 2 | VQAv2 | rl | 500 | -12.8 [-16.6, -9.2] | 73.2 | 60.9 → 80.2 | +19.3 [+16.9, +21.8] | 47.8 | +44.8 [+39.6, +49.8] | 97.7 | passed |
| base2 | — | CLEVR counting | rl | 500 | -9.8 [-13.0, -7.0] | 98.0 | 84.9 → 100.0 | +15.1 [+13.4, +17.0] | 48.4 | +98.0 [+96.8, +99.2] | 97.2 | rejected (failed: not saturated) |
| base2 | — | TextVQA | test (validation only) | 500 | -10.6 [-14.1, -7.1] | 79.8 | 60.4 → 77.0 | +16.6 [+14.3, +18.9] | 42.0 | +75.9 [+72.2, +79.5] | 96.4 | rejected (failed: training questions) |
| base8 | 1 | GQA | rl | 500 | -10.4 [-14.0, -7.2] | 70.6 | 60.5 → 74.6 | +14.1 [+11.9, +16.5] | 31.2 | +36.2 [+31.4, +41.0] | 98.8 | passed |
| base8 | 2 | VQAv2 | rl | 500 | -12.0 [-15.4, -8.6] | 77.2 | 64.5 → 79.0 | +14.5 [+12.2, +16.9] | 31.8 | +45.6 [+40.8, +50.4] | 98.6 | passed |
| base8 | — | CLEVR counting | rl | 500 | -3.2 [-4.8, -1.8] | 99.8 | 96.7 → 99.8 | +3.1 [+2.4, +3.9] | 15.2 | +99.8 [+99.4, +100.0] | 99.1 | rejected (failed: not saturated) |
| base8 | — | TextVQA | test (validation only) | 500 | -8.2 [-11.3, -5.2] | 83.9 | 66.1 → 76.6 | +10.5 [+8.6, +12.5] | 26.4 | +77.5 [+73.8, +81.0] | 99.1 | rejected (failed: training questions) |
