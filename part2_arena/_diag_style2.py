"""Throwaway diagnostic: what does the style-1 policy actually do?

Runs episodes headless and reports survival, kills, phases, and where the
ship spends its time. Not part of the submission - diagnostic only.
"""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from arena.gym_adapter import ArenaGymEnv


def diag(model_path: pathlib.Path, episodes: int = 3) -> None:
    model = PPO.load(model_path)
    env = ArenaGymEnv(control_style=2)
    print(f"\n=== {model_path.name} ===")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        total_r = 0.0
        kills = 0
        spawner_kills = 0
        phases = 0
        xs, ys = [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            done = term or trunc
            total_r += r
            steps += 1
            rb = info.get("reward_breakdown", {})
            kills += getattr(rb, "kill_enemy", 0.0) > 0
            spawner_kills += getattr(rb, "kill_spawner", 0.0) > 0
            phases += getattr(rb, "phase_progress", 0.0) > 0
            if steps % 50 == 0 or done:
                xs.append(env.core_env.state.player.x)
                ys.append(env.core_env.state.player.y)
        w = env.core_env.arena_width
        h = env.core_env.arena_height
        mean_x = sum(xs) / len(xs) / w
        mean_y = sum(ys) / len(ys) / h
        last = (xs[-1] / w, ys[-1] / h) if xs else (float("nan"), float("nan"))
        cause = "died" if info.get("died") else ("truncated" if info.get("truncated") else "?")
        print(
            f"ep{ep}: steps={steps:4d} cause={cause:9s} R={total_r:8.1f} "
            f"kills={kills:3d} spawnerK={spawner_kills} phases={phases} "
            f"meanPos=({mean_x:.2f},{mean_y:.2f}) lastPos=({last[0]:.2f},{last[1]:.2f})"
        )
    env.close()


if __name__ == "__main__":
    models = ROOT / "models"
    for name in [
        "style2_ppo_tuned_v1.zip",
        "style2_ppo_tuned_v1_best/best_model.zip",
        "style2_ppo_tuned_v1_curriculum.zip",
    ]:
        p = models / name
        if p.exists():
            diag(p)
        else:
            print(f"missing: {p}")

