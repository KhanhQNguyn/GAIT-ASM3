"""Plot a single run's training curves (episode reward / episode length /
loss) as one row of panels, for report section 12 (List of Figures F5).

This replaces hand-taking a TensorBoard screenshot: it reads the same
scalars straight from the run's event file, so the figure is reproducible
and matches the visual style of the other part-II figures.

Usage:
    cd part2_arena && python scripts/plot_training_curves.py \\
        --run style1_ppo_tuned_v3_curriculum_24 --out style1_training_curves.png

    # or let it pick the newest run for a (style, algo, preset, curriculum):
    cd part2_arena && python scripts/plot_training_curves.py \\
        --style 1 --config tuned_v3 --curriculum on
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

FRIENDLY = {
    "rollout/ep_rew_mean": "Episode reward",
    "rollout/ep_len_mean": "Episode length (steps)",
    "train/loss": "Training loss",
    "train/value_loss": "Value loss",
    "train/entropy_loss": "Entropy loss",
    "train/explained_variance": "Explained variance",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=None, help="log-dir name under logs/ (e.g. style1_ppo_tuned_v3_curriculum_24)")
    p.add_argument("--style", type=int, choices=[1, 2], default=1)
    p.add_argument("--algo", default="ppo")
    p.add_argument("--config", default="tuned_v3")
    p.add_argument("--curriculum", choices=["on", "off"], default="on")
    p.add_argument("--tags", default="rollout/ep_rew_mean,rollout/ep_len_mean,train/loss")
    p.add_argument("--smooth", type=int, default=5, help="moving-average window (points)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def resolve_run(args: argparse.Namespace) -> pathlib.Path:
    if args.run:
        d = LOGS_DIR / args.run.rstrip("/\\")
        if not d.is_dir():
            raise SystemExit(f"no such run directory: {d}")
        return d
    suffix = "_curriculum" if args.curriculum == "on" else ""
    prefix = f"style{args.style}_{args.algo}_{args.config}{suffix}_"
    cands = [p for p in LOGS_DIR.glob(f"{prefix}*") if p.is_dir()
             and p.name.rsplit("_", 1)[-1].isdigit()]
    if not cands:
        raise SystemExit(f"no run directory matching '{prefix}*' under {LOGS_DIR}")
    return max(cands, key=lambda p: int(p.name.rsplit("_", 1)[-1]))


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def read_series(acc: EventAccumulator, tag: str) -> tuple[list[int], list[float]]:
    events = acc.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def main() -> None:
    args = parse_args()
    run_dir = resolve_run(args)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    have = set(acc.Tags().get("scalars", []))
    tags = [t for t in tags if t in have] or [t for t in tags]
    missing = [t for t in tags if t not in have]
    for t in missing:
        print(f"  (skip, tag not in run) {t}")
    tags = [t for t in tags if t in have]
    if not tags:
        raise SystemExit(f"none of the requested tags are in {run_dir}")

    fig, axes = plt.subplots(1, len(tags), figsize=(4.3 * len(tags), 3.7))
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        steps, values = read_series(acc, tag)
        ax.plot(steps, moving_average(values, args.smooth), linewidth=1.6)
        ax.set_title(FRIENDLY.get(tag, tag), fontsize=11)
        ax.set_xlabel("Timesteps")
        ax.grid(alpha=0.25)
        if tag == "rollout/ep_rew_mean":
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        print(f"  {tag:24s} n={len(values):4d}  last_step={steps[-1]:,}  "
              f"final={values[-1]:.1f}  min={min(values):.1f}  max={max(values):.1f}")

    fig.suptitle(f"Style {args.style} final run  ({run_dir.name})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or f"{run_dir.name}_training_curves.png"
    out_path = FIGURES_DIR / out
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
