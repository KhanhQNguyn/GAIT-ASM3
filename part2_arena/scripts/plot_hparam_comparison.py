"""Overlay the mean-episode-reward curves of several training runs on one
axis, for the report's "hyperparameter exploration, with evidence" section
(report sections 11-12, List of Figures F12 / F13).

No single training run produces this figure -- it is built from the
TensorBoard logs of runs that already happened. By default it finds, for
the given --style, the newest run of each preset (tuned_v1..tuned_v4,
curriculum on) plus the newest curriculum-off tuned_v3 run, and overlays
their rollout/ep_rew_mean curves. Override the set with --runs
"<log-dir-name>=<label>" pairs.

Usage:
    cd part2_arena && python scripts/plot_hparam_comparison.py --style 1
    cd part2_arena && python scripts/plot_hparam_comparison.py --style 2
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_PART2_ARENA_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PART2_ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ARENA_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
FIGURES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures" / "part2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], default=1)
    parser.add_argument("--algo", default="ppo")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help='override: "<log-dir-name>=<legend label>" pairs',
    )
    parser.add_argument("--tag", default="rollout/ep_rew_mean")
    parser.add_argument("--smooth", type=int, default=5, help="moving-average window (points)")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def latest_run(prefix: str) -> pathlib.Path | None:
    """Newest logs/<prefix>_<N>/ directory (highest N), or None."""
    cands = [p for p in LOGS_DIR.glob(f"{prefix}_*") if p.is_dir()]
    cands = [p for p in cands if p.name.rsplit("_", 1)[-1].isdigit()]
    if not cands:
        return None
    return max(cands, key=lambda p: int(p.name.rsplit("_", 1)[-1]))


def default_runs(style: int, algo: str) -> list[tuple[str, str]]:
    plan = [
        (f"style{style}_{algo}_tuned_v1_curriculum", "A  tuned_v1  (net 128, lr 2.5e-4, ent 0.005)"),
        (f"style{style}_{algo}_tuned_v2_curriculum", "B  tuned_v2  (net 256, lr 1e-4, ent 0.02)"),
        (f"style{style}_{algo}_tuned_v3_curriculum", "C  tuned_v3  (tuned_v1 + ent 0.008) -- shipped"),
        (f"style{style}_{algo}_tuned_v4_curriculum", "D  tuned_v4  (tuned_v3 + gamma 0.999)"),
        (f"style{style}_{algo}_tuned_v3", "C  tuned_v3, curriculum OFF"),
    ]
    out = []
    for prefix, label in plan:
        run = latest_run(prefix)
        if run is None:
            print(f"  (skip, no run) {prefix}")
            continue
        out.append((run.name, label))
    return out


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def read_series(log_dir: pathlib.Path, tag: str) -> tuple[list[int], list[float]]:
    acc = EventAccumulator(str(log_dir))
    acc.Reload()
    if tag not in set(acc.Tags().get("scalars", [])):
        raise KeyError(f"tag {tag!r} not in {log_dir}")
    events = acc.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def main() -> None:
    args = parse_args()
    runs = (
        [tuple(pair.split("=", 1)) for pair in args.runs]
        if args.runs
        else default_runs(args.style, args.algo)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for dir_name, label in runs:
        log_dir = LOGS_DIR / dir_name
        if not log_dir.is_dir():
            print(f"skip (missing): {log_dir}")
            continue
        steps, values = read_series(log_dir, args.tag)
        style = "--" if "OFF" in label else "-"
        ax.plot(steps, moving_average(values, args.smooth), style, label=label.strip(), linewidth=1.6)
        print(f"{dir_name}: {len(steps)} pts, {steps[-1]:,} steps, final {args.tag} = {values[-1]:.1f}")

    ax.axvline(300_000, color="grey", linestyle=":", linewidth=1)
    ax.text(300_000, ax.get_ylim()[0], " 300k sweep budget", fontsize=8, color="grey", va="bottom")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean episode reward (rollout/ep_rew_mean)")
    ax.set_title(f"Control Style {args.style} -- hyperparameter configurations compared")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.25)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or f"style{args.style}_hparam_comparison.png"
    out_path = FIGURES_DIR / out
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
