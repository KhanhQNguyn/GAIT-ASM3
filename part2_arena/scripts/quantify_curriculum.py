"""Creativity hook (c) quantification: curriculum ramp ON vs OFF, per
control style. Reads the two TensorBoard runs scripts/train.py already
produced (--curriculum on / --curriculum off, same --config otherwise) and
reports whether the ramp measurably speeds up early learning or changes
final performance -- a real number, not an eyeballed curve description
(UPDATES.md Task D.7 / MINH.md Task D.7).

Does NOT train anything itself -- run scripts/train.py for both curriculum
settings first (train.py's model_save_path()/run_name() already name a
curriculum-off run with no suffix and a curriculum-on run with
"_curriculum", so both runs' TensorBoard directories always exist under
<preset> naming once both have been run once each).

Usage:
    python scripts/quantify_curriculum.py --style 1 --config tuned_v1
    python scripts/quantify_curriculum.py --style 2 --config tuned_v1
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

_PART2_ARENA_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PART2_ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ARENA_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], required=True)
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--config", type=str, default="tuned_v1")
    return parser.parse_args()


def read_scalar(run_dir: pathlib.Path, tag: str = "rollout/ep_rew_mean"):
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def find_run_dir(logs_dir: pathlib.Path, run_name: str) -> pathlib.Path:
    """train.py writes each run under logs/<run_name>_<n>/ (SB3's
    tb_log_name auto-increments the trailing _n on every invocation) --
    pick the most recently modified match so re-running a config always
    quantifies the latest attempt.

    A plain `glob(f"{run_name}_*")` is NOT safe here: the curriculum-off
    run_name (e.g. "style1_ppo_tuned_v1") is a strict prefix of the
    curriculum-on run_name ("style1_ppo_tuned_v1_curriculum"), so that glob
    would also match the "on" run's directories and could silently pick
    the wrong one depending on mtime ordering. Anchor with a regex instead
    so only "<run_name>_<digits>" (nothing else) counts as a match.
    """
    pattern = re.compile(rf"^{re.escape(run_name)}_\d+$")
    candidates = sorted(
        (p for p in logs_dir.iterdir() if p.is_dir() and pattern.match(p.name)),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No TensorBoard run directory matching '{run_name}_<n>' under {logs_dir} -- "
            f"run scripts/train.py for this style/config/curriculum setting first."
        )
    return candidates[-1]


def steps_to_threshold(steps: list[int], values: list[float], threshold: float) -> int | None:
    """First timestep at which the smoothed reward first reaches
    `threshold`, or None if it never does within the logged run."""
    for step, value in zip(steps, values):
        if value >= threshold:
            return step
    return None


def main() -> None:
    args = parse_args()
    on_name = f"style{args.style}_{args.algo}_{args.config}_curriculum"
    off_name = f"style{args.style}_{args.algo}_{args.config}"

    on_dir = find_run_dir(LOGS_DIR, on_name)
    off_dir = find_run_dir(LOGS_DIR, off_name)

    on_steps, on_values = read_scalar(on_dir)
    off_steps, off_values = read_scalar(off_dir)

    # A threshold reachable by both runs, used for the "how fast did each
    # get there" comparison -- 80% of the better run's final smoothed
    # reward, so this adapts to whatever reward scale this style/config
    # actually reaches instead of a hardcoded number.
    best_final = max(on_values[-1] if on_values else 0.0, off_values[-1] if off_values else 0.0)
    threshold = 0.8 * best_final

    on_first_hit = steps_to_threshold(on_steps, on_values, threshold)
    off_first_hit = steps_to_threshold(off_steps, off_values, threshold)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(on_steps, on_values, label="curriculum ON")
    ax.plot(off_steps, off_values, label="curriculum OFF")
    if on_first_hit is not None:
        ax.axvline(on_first_hit, color="C0", linestyle="--", alpha=0.5)
    if off_first_hit is not None:
        ax.axvline(off_first_hit, color="C1", linestyle="--", alpha=0.5)
    ax.axhline(
        threshold, color="gray", linestyle=":", alpha=0.5, label=f"{threshold:.1f} threshold"
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean episode reward (rollout/ep_rew_mean)")
    ax.set_title(f"Style {args.style}: curriculum ON vs OFF ({args.config})")
    ax.legend()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"curriculum_ablation_style{args.style}_{args.config}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_lines = [
        f"# Curriculum ablation -- Style {args.style} ({args.config})",
        "",
        f"| Run | Final ep_rew_mean | Steps to reach {threshold:.1f} |",
        "|---|---|---|",
        f"| curriculum ON  | {on_values[-1] if on_values else float('nan'):.2f} | "
        f"{on_first_hit if on_first_hit is not None else 'not reached'} |",
        f"| curriculum OFF | {off_values[-1] if off_values else float('nan'):.2f} | "
        f"{off_first_hit if off_first_hit is not None else 'not reached'} |",
        "",
    ]
    if on_first_hit is not None and off_first_hit is not None:
        delta = off_first_hit - on_first_hit
        verdict = (
            f"Curriculum reached the threshold {delta} timesteps "
            f"{'sooner' if delta > 0 else 'later'} than without it."
            if delta != 0
            else "Curriculum reached the threshold at the same timestep as without it."
        )
    else:
        verdict = (
            "One of the two runs never reached the threshold within its logged "
            "timesteps -- see the raw final rewards above instead of a steps-to-threshold claim."
        )
    summary_lines.append(verdict)

    table_path = FIGURES_DIR / f"curriculum_ablation_style{args.style}_{args.config}.md"
    table_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved plot to: {fig_path}")
    print(f"Saved table to: {table_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
