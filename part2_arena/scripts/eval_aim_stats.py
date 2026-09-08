"""Headless aim-quality stats for a trained arena agent (either style).

Complements eval_style1.py / eval_style2.py: those show the agent playing
visually; this one measures HOW WELL IT AIMS without a window, so the
before/after of an aim-training change is quantifiable for the report.

Usage (from part2_arena/):
    python scripts/eval_aim_stats.py --style 2 --config tuned_v3 --curriculum on
    python scripts/eval_aim_stats.py --style 1 --config tuned_v3 --curriculum on

Per-episode metrics:
    shots_fired      player shots actually fired (core_env._shots_fired diff;
                     since 2026-09-08d -- BUG_LESS_ATTACK.md -- NOT a
                     len(projectiles) diff, which dropped same-step-collision
                     shots and inflated mean_alignment ~3x)
    mean_alignment   average graded aim flag at fire time (1.0 = dead-on)
    frac_aligned30   fraction of shots within 30 deg of the enemy bearing
    damage_dealt     total enemy HP removed by projectiles
                     (rb.damage_dealt / R_DAMAGE_DEALT_PER_HP)
    wall_frac        fraction of steps with wall_distance < 120 px
    kills / phase / return
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

_PART2_ARENA_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PART2_ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ARENA_ROOT))
from stable_baselines3 import DQN, PPO  # noqa: E402

from arena.gym_adapter import ArenaGymEnv  # noqa: E402
from arena.rewards_config import (  # noqa: E402
    R_DAMAGE_DEALT_PER_HP,
    R_DAMAGE_TAKEN_PER_HP,
    SHOT_NO_TARGET_RADIUS,
    WALL_PROXIMITY_MARGIN,
)

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], required=True)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="on")
    parser.add_argument("--config", type=str, default="tuned_v3")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=["final", "best"],
        default="final",
        help="'final' = the run-end save; 'best' = EvalCallback's best-by-eval-reward",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        choices=["deterministic", "stochastic"],
        default="deterministic",
        help="PPO is stochastic; argmax can collapse to a degenerate mode "
        "(see eval_style1.py --sampling). stochastic matches training.",
    )
    return parser.parse_args()


def run_episode(env: ArenaGymEnv, model, deterministic: bool) -> dict:
    obs, _info = env.reset()
    player_projectiles = 0
    alignment_sum, aligned30 = 0.0, 0
    damage_dealt, damage_taken, kills = 0.0, 0.0, 0
    wall_steps, steps = 0, 0
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        prev_fired = env.core_env._shots_fired
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        # A shot fired this step iff core_env's real fired-shot counter grew
        # (2026-09-08d fix, BUG_LESS_ATTACK.md). The old len(projectiles)
        # diff silently dropped shots that collided the same step they fired
        # -- exactly the spray shots -- so mean_align read ~3x too high.
        if env.core_env._shots_fired > prev_fired:
            player_projectiles += 1
            alignment_sum += float(env.core_env._shot_toward_enemy_flag)
            if env.core_env._shot_toward_enemy_flag >= math.cos(math.radians(30.0)):
                aligned30 += 1
        # Reward terms are scaled (0.05/HP dealt, -0.7/HP taken): divide back
        # out to report raw HP.
        rb = info["reward_breakdown"]
        damage_dealt += rb.damage_dealt / R_DAMAGE_DEALT_PER_HP
        damage_taken += abs(rb.damage_taken) / R_DAMAGE_TAKEN_PER_HP
        kills += int(rb.kill_enemy > 0)
        ev = env.core_env._last_step_events
        if math.isfinite(ev["wall_distance"]) and ev["wall_distance"] < WALL_PROXIMITY_MARGIN:
            wall_steps += 1
    return {
        "steps": steps,
        "return": total_reward,
        "shots": player_projectiles,
        "mean_align": (alignment_sum / player_projectiles) if player_projectiles else 0.0,
        "frac30": (aligned30 / player_projectiles) if player_projectiles else 0.0,
        "dmg_dealt": damage_dealt,
        "dmg_taken": damage_taken,
        "kills": kills,
        "wall_frac": wall_steps / steps if steps else 0.0,
        "phase": env.core_env.state.phase,
    }


def main() -> None:
    args = parse_args()
    suffix = "_curriculum" if args.curriculum == "on" else ""
    base = MODELS_DIR / f"style{args.style}_{args.algo}_{args.config}{suffix}"
    model_path = base if args.checkpoint == "final" else pathlib.Path(
        str(base) + "_best", "best_model.zip"
    )
    model_cls = PPO if args.algo == "ppo" else DQN
    model = model_cls.load(model_path, device="cpu")

    env = ArenaGymEnv(
        control_style=args.style,
        curriculum_enabled=(args.curriculum == "on"),
    )
    det = args.sampling == "deterministic"
    rows = [run_episode(env, model, deterministic=det) for _ in range(args.episodes)]
    env.close()

    print(f"model={model_path.name}  episodes={args.episodes}  {args.sampling}")
    keys = [
        "steps",
        "return",
        "shots",
        "mean_align",
        "frac30",
        "dmg_dealt",
        "dmg_taken",
        "kills",
        "wall_frac",
        "phase",
    ]
    print("  ".join(f"{k:>10}" for k in keys))
    for r in rows:
        print("  ".join(f"{r[k]:>10.2f}" for k in keys))
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    print("MEAN " + "  ".join(f"{mean[k]:>10.2f}" for k in keys))
    print(
        f"engage_range={SHOT_NO_TARGET_RADIUS:.0f}px  "
        f"align measured against nearest enemy in range (else active spawner)"
    )


if __name__ == "__main__":
    main()
