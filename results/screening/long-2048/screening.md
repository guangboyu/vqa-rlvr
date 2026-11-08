Screening battery on each candidate's training pool (normalized scorer; differences in points with paired 95% bootstrap intervals).

Gates for any RL: at least 2,000 training questions; split separable and license usable (declared); short-prompt accuracy < 90%; black and swapped images each cost ≥ 10 points (lower bound). Routes: reasoning-prompt RL needs parse rate ≥ 95% of sampled reasoning responses, stepwise not worse than direct (upper bound ≥ 0) and a learnable share ≥ 25%; short-answer RL needs a short-prompt learnable share ≥ 25%. A candidate passes with the common gates and at least one viable route. Ranking: passed candidates by matched payoff, then RL headroom, both descending.

| Model | Rank | Dataset | Pool | Questions | Short (%) | Reasoning payoff | Matched payoff | pass@1 → pass@k (%) | RL headroom | Learnable (%) | Image: black | Image: swapped | Parse (%) | Short learnable (%) | Routes | Status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| base2 | 1 | ViRL39K (one image) | rl | 500 | 28.2 | +17.0 [+11.8, +22.4] | +2.4 [-2.4, +7.2] | 42.2 → 67.4 | +25.2 [+22.4, +28.1] | 48.2 | +8.0 [+4.2, +11.8] | +9.0 [+5.2, +12.8] | 59.7 | 41.0 | short | passed |
| base2 | — | GEOQA (R1V train) | rl | 500 | 16.6 | +20.8 [+16.0, +25.6] | +5.8 [+1.0, +10.6] | 32.7 → 54.2 | +21.5 [+18.8, +24.3] | 39.0 | +3.0 [+0.6, +5.4] | +4.2 [+1.4, +7.0] | 48.7 | 26.2 | short | rejected (failed: image needed) |
| base2 | — | MMK12 (train) | rl | 500 | 14.4 | +16.0 [+11.4, +20.4] | +11.6 [+7.4, +15.8] | 28.3 → 46.4 | +18.1 [+15.6, +20.7] | 33.8 | +2.8 [+0.4, +5.2] | +4.4 [+1.8, +7.2] | 49.2 | 24.4 | none | rejected |
