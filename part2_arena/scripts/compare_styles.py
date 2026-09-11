"""Required rubric item (NOT a creativity bonus): "comparison of the two
control sets, with evidence" (UPDATES.md Sec 3.11 / MINH.md Task D.8).

Runs both FINAL trained models headlessly (no rendering -- this is a data
task, not a video-recording pass) for an equal number of episodes under
matching seeds, and records: average return, average survival time
(steps), average phases reached, and average enemies+spawners destroyed.

Each style is evaluated under BOTH action-selection modes: `deterministic`
(argmax -- can collapse a stochastic policy into a degenerate mode) and
`stochastic` (samples the policy, i.e. how it was trained and how the demo
runs). Reporting only the deterministic pass once made a same-config
re-seed look like a cross-the-board regression; both are reported now.

Enemies/spawners destroyed are derived from info["reward_breakdown"]
(kill_enemy / kill_spawner reward contributions divided by the per-kill
constants) rather than a separate counter, since RewardBreakdown is
already the one place those events are named (arena/rewards.py).

CAVEAT (verified against arena/core_env.py::_resolve_collisions): only
enemies destroyed by a PLAYER PROJECTILE increment "enemies_killed" /
kill_enemy. An enemy that touches the player is also removed from the
world (dealing contact damage once) but is NOT counted here -- so
"avg_enemies_destroyed" measures shot-kills specifically, not every way
an enemy can be removed. A style/policy that clears enemies mostly by
tanking contact hits will show a low number here despite genuinely
engaging enemies; cross-check against avg_return / survival before
concluding a style "never destroys enemies."

The written interpretation of these numbers (e.g. "why style 2
outperforms style 1") is report content, out of scope here -- this script
only produces the real table the interpretation would rest on.

Usage:
    python scripts/compare_styles.py --episodes 20
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_PART2_ARENA_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PART2_ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ARENA_ROOT))

from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

from arena.gym_adapter import ArenaGymEnv  # noqa: E402
from arena.rewards_config import R_KILL_ENEMY, R_KILL_SPAWNER  # noqa: E402

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument(
        "--config",
        type=str,
        default="tuned_v1",
        help="hyperparameter preset both final models were trained with",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        choices=["on", "off"],
        default="on",
        help="which final model variant to compare (both styles' curriculum-on run "
        "by default, since that's the recommended real training configuration)",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def evaluate_style(
    style: int,
    algo: str,
    config: str,
    curriculum: str,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> dict:
    # Seed python/numpy/torch so the stochastic pass's action sampling is
    # reproducible run-to-run (the env layout is already pinned per episode
    # via reset(seed=...)). No effect on the deterministic pass.
    set_random_seed(seed)

    suffix = "_curriculum" if curriculum == "on" else ""
    model_path = MODELS_DIR / f"style{style}_{algo}_{config}{suffix}"
    model_cls = PPO if algo == "ppo" else DQN
    model = model_cls.load(model_path)

    env = ArenaGymEnv(control_style=style)

    returns, survival_steps, phases_reached, enemies_destroyed, spawners_destroyed = (
        [],
        [],
        [],
        [],
        [],
    )
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        terminated = truncated = False
        total_reward, steps = 0.0, 0
        ep_enemies_killed, ep_spawners_killed = 0, 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            rb = info["reward_breakdown"]
            if R_KILL_ENEMY:
                ep_enemies_killed += round(rb.kill_enemy / R_KILL_ENEMY)
            if R_KILL_SPAWNER:
                ep_spawners_killed += round(rb.kill_spawner / R_KILL_SPAWNER)

        final_phase = env.core_env.state.phase if env.core_env.state is not None else 0
        returns.append(total_reward)
        survival_steps.append(steps)
        phases_reached.append(final_phase)
        enemies_destroyed.append(ep_enemies_killed)
        spawners_destroyed.append(ep_spawners_killed)

    env.close()

    n = len(returns)
    return {
        "style": style,
        "sampling": "deterministic" if deterministic else "stochastic",
        "episodes": n,
        "avg_return": sum(returns) / n,
        "avg_survival_steps": sum(survival_steps) / n,
        "avg_phases_reached": sum(phases_reached) / n,
        "avg_enemies_destroyed": sum(enemies_destroyed) / n,
        "avg_spawners_destroyed": sum(spawners_destroyed) / n,
    }


def render_markdown_table(rows: list[dict]) -> str:
    header = (
        "| Style | Sampling | Episodes | Avg return | Avg survival (steps) | "
        "Avg phases reached | Avg enemies destroyed | Avg spawners destroyed |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['style']} | {row['sampling']} | {row['episodes']} | "
            f"{row['avg_return']:.2f} | "
            f"{row['avg_survival_steps']:.1f} | {row['avg_phases_reached']:.2f} | "
            f"{row['avg_enemies_destroyed']:.2f} | {row['avg_spawners_destroyed']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = [
        evaluate_style(
            style, args.algo, args.config, args.curriculum, args.episodes, args.seed, deterministic
        )
        for style in (1, 2)
        for deterministic in (True, False)
    ]
    table = render_markdown_table(rows)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "style1_vs_style2_comparison.md"
    out_path.write_text(
        f"# Style 1 vs Style 2 comparison ({args.algo}, {args.config}, "
        f"curriculum={args.curriculum}, {args.episodes} episodes each, seed={args.seed}; "
        f"deterministic + stochastic action-selection passes)\n\n"
        + table
        + "\n*Enemies destroyed = via player projectile only (see this script's module "
        "docstring) -- an enemy destroyed by touching the player is not counted here.*\n",
        encoding="utf-8",
    )
    print(f"Saved comparison table to: {out_path}")
    print(table)


if __name__ == "__main__":
    main()
