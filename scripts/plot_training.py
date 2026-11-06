"""Figures for reading RL runs: training health per update, and dev learning curves.

    uv run --group plots python -m scripts.plot_training <run-dir> [<run-dir> ...] \
        --out figures/training.png [--curves figures/curves.png] [--window 20]

A run directory holds `metrics.jsonl` (the Trainer's logged signals), optionally
`prompt_draws.jsonl` (every rollout's score, for the dead-group split) and
`checkpoints/dev-records/` (the dev evaluation of each kept adapter). Runs are overlaid,
labelled by directory name. What each panel means and when to worry is in
docs/harness/monitoring.md.
"""

import argparse
from pathlib import Path

from scripts.watch_training import logged_lines
from vqar.training import reward_per_update

# (title, metrics.jsonl key); logged every `logging_steps` updates, averaged over them.
LOGGED_PANELS = (
    ("training reward", "reward"),
    ("entropy (nats/token)", "entropy"),
    ("response length (tokens)", "completions/mean_length"),
    ("sampler-trainer log-prob gap", "sampling/sampling_logp_difference/mean"),
    ("gradient norm (zeros dropped: no live group)", "grad_norm"),
)


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Trailing mean over at most `window` values, so early points are not dropped."""
    means, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        means.append(total / min(index + 1, window))
    return means


def dead_share_series(run: Path) -> dict[str, tuple[list[int], list[float]]]:
    """Per update, the share of groups whose samples were all right or all wrong."""
    draws = run / "prompt_draws.jsonl"
    if not draws.exists():
        return {}
    updates, _ = reward_per_update(draws)
    steps = [u["update"] for u in updates]
    return {kind: (steps, [u[f"{kind}_group_fraction"] for u in updates])
            for kind in ("all_right", "all_wrong")}


def dev_curves(run: Path) -> dict[str, tuple[list[int], list[float], list[int]]]:
    """Normalized dev accuracy and question count per (dataset, prompt), by training step."""
    from vqar.eval.report import load_cells

    records = run / "checkpoints" / "dev-records"
    if not records.exists():
        return {}
    points = {}
    for cell, record in load_cells(records).items():
        if cell.step is not None and cell.samples == 1:
            points.setdefault(f"{cell.dataset} / {cell.prompt}", []).append(
                (cell.step, record.metrics["normalized"]["overall"], record.metrics["n"]))
    return {name: tuple(map(list, zip(*sorted(p), strict=True)))
            for name, p in sorted(points.items())}


def pyplot():
    """matplotlib without a display: figures are only ever written to files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_health(runs: list[tuple[Path, str]], out: Path, window: int) -> None:
    """Six panels of logged signals, one colour per (run directory, legend label)."""
    plt = pyplot()

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    panels = list(zip(axes.flat, LOGGED_PANELS, strict=False))
    for (run, label), color in zip(runs, plt.rcParams["axes.prop_cycle"].by_key()["color"],
                                   strict=False):
        lines = logged_lines(run / "metrics.jsonl")
        for ax, (title, key) in panels:
            # A logged gradient norm of exactly 0 means that update had no live group.
            points = [(e["step"], e[key]) for e in lines
                      if e.get(key) is not None and not (key == "grad_norm" and e[key] == 0)]
            if points:
                steps, values = zip(*points, strict=True)
                ax.plot(steps, values, color=color, alpha=0.25, linewidth=0.8)
                ax.plot(steps, rolling_mean(list(values), max(1, window // 10)),
                        color=color, label=label)
            ax.set_title(title)
        ax = axes.flat[5]
        for kind, style in (("all_right", "-"), ("all_wrong", "--")):
            steps, shares = dead_share_series(run).get(kind, ([], []))
            if steps:
                ax.plot(steps, rolling_mean(shares, window), style, color=color)
        ax.set_title(f"dead groups: all right (solid), all wrong (dashed)\n"
                     f"share per update, {window}-update mean", fontsize=10)
    axes.flat[2].set_yscale("log")
    axes.flat[4].set_yscale("log")
    for ax in axes.flat:
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


def plot_curves(runs: list[tuple[Path, str]], out: Path) -> None:
    plt = pyplot()

    curves = {run: dev_curves(run) for run, _ in runs}
    names = sorted({name for c in curves.values() for name in c})
    if not names:
        print("no dev records yet")
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2),
                             constrained_layout=True, squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        for run, label in runs:
            if name in curves[run]:
                steps, accuracy, counts = curves[run][name]
                # One binomial standard error: dev sets are small, so read steps against it.
                errors = [(a * (1 - a) / n) ** 0.5 for a, n in zip(accuracy, counts,
                                                                    strict=True)]
                ax.errorbar(steps, accuracy, yerr=errors, marker="o", markersize=3,
                            capsize=2, label=label)
                size = counts[0]
        ax.set_title(f"dev: {name} (n={size})")
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("normalized accuracy")
    axes[0][0].legend(fontsize=7)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--out", type=Path, required=True, help="training-health figure")
    parser.add_argument("--curves", type=Path, default=None, help="dev learning-curve figure")
    parser.add_argument("--window", type=int, default=20, help="updates per rolling mean")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="legend names, one per run (default: directory names)")
    args = parser.parse_args()
    labels = args.labels or [run.name for run in args.runs]
    if len(labels) != len(args.runs):
        parser.error("give one label per run")
    runs = list(zip(args.runs, labels, strict=True))
    plot_health(runs, args.out, args.window)
    if args.curves:
        plot_curves(runs, args.curves)


if __name__ == "__main__":
    main()
