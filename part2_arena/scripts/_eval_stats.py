"""Shared end-of-run statistics for eval_style1.py / eval_style2.py.

The two eval scripts stay standalone per-control-style (rubric Part II-I),
but both accumulate the same per-episode record and print the same summary
block, so that lives here rather than being duplicated.

`EpisodeRecord` is filled in the eval loop; `summarise()` prints the
aggregate a viewer actually needs to judge a policy (death rate, phase
reached, reward mix) instead of just a list of per-episode lines.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.rewards_config import R_KILL_ENEMY, R_KILL_SPAWNER


@dataclass
class EpisodeRecord:
    steps: int = 0
    total_reward: float = 0.0
    died: bool = False
    truncated: bool = False
    final_phase: int = 0
    # cumulative RewardBreakdown terms over the episode
    r_kill_enemy: float = 0.0
    r_kill_spawner: float = 0.0
    r_phase_progress: float = 0.0
    r_damage_taken: float = 0.0
    r_death: float = 0.0

    def add_step(self, reward_breakdown) -> None:
        rb = reward_breakdown
        self.r_kill_enemy += rb.kill_enemy
        self.r_kill_spawner += rb.kill_spawner
        self.r_phase_progress += rb.phase_progress
        self.r_damage_taken += rb.damage_taken
        self.r_death += rb.death


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def summarise(rows: list[EpisodeRecord], *, style: int, sampling: str, max_steps: int) -> None:
    """Print the aggregate summary for a completed eval run."""
    n = len(rows)
    if n == 0:
        print("(no episodes completed)")
        return
    died = sum(r.died for r in rows)
    survived = sum(r.truncated for r in rows)
    phases = [r.final_phase for r in rows]
    steps = [r.steps for r in rows]
    returns = [r.total_reward for r in rows]

    print(f"\n=== Style {style} summary — {n} episodes ({sampling} sampling) ===")
    print(f"  died / survived-to-cap : {died} / {survived}   (death rate {died / n:.0%})")
    print(
        f"  phase reached          : mean {_mean(phases):.1f}   "
        f"median {_median(phases):g}   min {min(phases)}   max {max(phases)}"
    )
    print(f"  episode length         : mean {_mean(steps):.0f} / {max_steps} steps")
    print(
        f"  episode return         : mean {_mean(returns):.1f}   "
        f"min {min(returns):.1f}   max {max(returns):.1f}"
    )
    enemy_kills = _mean([r.r_kill_enemy for r in rows]) / R_KILL_ENEMY
    spawner_kills = _mean([r.r_kill_spawner for r in rows]) / R_KILL_SPAWNER
    print(f"  kills / ep             : {enemy_kills:.1f} enemies   {spawner_kills:.1f} spawners")
    print(
        f"  reward mix (mean/ep)   : "
        f"phase_progress {_mean([r.r_phase_progress for r in rows]):+.0f}   "
        f"damage_taken {_mean([r.r_damage_taken for r in rows]):+.0f}   "
        f"death {_mean([r.r_death for r in rows]):+.0f}"
    )
