"""While an RL run trains: the signals that show whether it is healthy, per window of updates.

    python -m scripts.watch_training <run-dir>/metrics.jsonl [more runs...] [--window 100]

Reads the Trainer's logged lines (every `logging_steps` updates) and prints, per window, the
mean of each signal, then flags patterns that usually mean trouble. Training reward is
measured on the questions being trained on; held-out progress comes from the dev curves.
"""

import argparse
import json
from pathlib import Path

# (column, metrics.jsonl key); a column is shown only when its run logs the key.
SIGNALS = (
    ("reward", "reward"),
    ("dead", "frac_reward_zero_std"),
    ("entropy", "entropy"),
    ("length", "completions/mean_length"),
    ("truncated", "completions/clipped_ratio"),
    ("logp_gap", "sampling/sampling_logp_difference/mean"),
    ("is_ratio", "sampling/importance_sampling_ratio/mean"),
    ("clip", "clip_ratio/region_mean"),
    ("grad_norm", "grad_norm"),
    ("lr", "learning_rate"),
    ("s/update", "step_time"),
    ("live/attempted", "refill/live_groups"),
)

# Thresholds for flags: rough rules of thumb, not calibrated limits.
MAX_LOGP_GAP = 0.1  # vLLM and trainer disagree on the sampled tokens
MIN_ENTROPY_SHARE = 0.25  # entropy fell below a quarter of its first window
MAX_DEAD = 0.8  # most groups carry no gradient
MAX_TRUNCATED = 0.05
MAX_REWARD_JUMP = 0.15  # between consecutive windows
MIN_LENGTH_SHARE = 0.25  # responses shrank below a quarter of their first window
MAX_CLIP = 1e-3  # one pass per batch leaves only bf16 noise in the ratio


def logged_lines(path: Path) -> list[dict]:
    """Training log lines (not events), the last line per step when a resume repeated one."""
    by_step = {}
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        if "event" not in entry and "reward" in entry:
            by_step[entry["step"]] = entry
    return [by_step[step] for step in sorted(by_step)]


def windows(lines: list[dict], size: int) -> list[dict]:
    """Mean of each signal over consecutive windows of `size` updates."""
    grouped = {}
    for entry in lines:
        grouped.setdefault((entry["step"] - 1) // size, []).append(entry)
    rows = []
    for index, entries in sorted(grouped.items()):
        row = {"updates": f"{index * size + 1}-{entries[-1]['step']}"}
        for column, key in SIGNALS:
            values = [e[key] for e in entries if e.get(key) is not None]
            if values:
                row[column] = sum(values) / len(values)
        rows.append(row)
    return rows


def flags(rows: list[dict]) -> list[str]:
    found = []
    first_entropy = rows[0].get("entropy") if rows else None
    first_length = rows[0].get("length") if rows else None
    for previous, row in zip([None] + rows, rows, strict=False):
        span = row["updates"]
        if row.get("logp_gap", 0) > MAX_LOGP_GAP:
            found.append(f"{span}: sampler/trainer log-prob gap {row['logp_gap']:.3f}")
        if first_entropy and row.get("entropy", first_entropy) < MIN_ENTROPY_SHARE * first_entropy:
            found.append(f"{span}: entropy collapsed to {row['entropy']:.3f}")
        if first_length and row.get("length", first_length) < MIN_LENGTH_SHARE * first_length:
            found.append(f"{span}: responses shrank to {row['length']:.0f} tokens")
        if row.get("dead", 0) > MAX_DEAD:
            found.append(f"{span}: {row['dead']:.0%} of groups dead")
        if row.get("truncated", 0) > MAX_TRUNCATED:
            found.append(f"{span}: {row['truncated']:.1%} of responses truncated")
        if row.get("clip", 0) > MAX_CLIP:
            found.append(f"{span}: clipping active ({row['clip']:.3f}) with one pass per batch")
        if previous and abs(row["reward"] - previous["reward"]) > MAX_REWARD_JUMP:
            found.append(f"{span}: reward moved {row['reward'] - previous['reward']:+.2f}; "
                         "read samples before trusting it")
    return found


def table(rows: list[dict]) -> str:
    columns = ["updates"] + [c for c, _ in SIGNALS if any(c in row for row in rows)]
    lines = [" ".join(f"{c:>10}" for c in columns)]
    for row in rows:
        cells = [row.get(c) for c in columns]
        lines.append(" ".join(f"{v:>10}" if isinstance(v, str) or v is None else f"{v:>10.4g}"
                              for v in cells))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics.jsonl of each run")
    parser.add_argument("--window", type=int, default=100, help="updates per row")
    args = parser.parse_args()
    for path in args.metrics:
        rows = windows(logged_lines(path), args.window)
        print(f"== {path.parent.name}")
        print(table(rows) if rows else "no logged updates yet")
        for flag in flags(rows):
            print(f"FLAG {flag}")


if __name__ == "__main__":
    main()
