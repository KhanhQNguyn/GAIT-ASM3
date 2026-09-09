"""Train a Stable-Baselines3 agent on the arena, for one control style.

Supports both required deliverables (one model per control style) and the
creativity hooks:
  (a) --algo {ppo,dqn}: train either algorithm, enabling a PPO-vs-DQN
      ablation when run twice with the same --style.
  (c) --curriculum {on,off}: toggles PhaseManager's curriculum ramp,
      enabling a with/without curriculum learning-speed comparison.

Usage:
    python scripts/train.py --style 1 --algo ppo --timesteps 300000
    python scripts/train.py --style 2 --algo dqn --timesteps 300000 --curriculum on
    python scripts/train.py --style 1 --algo ppo --timesteps 100000 --death-penalty -30

  (ablation) --death-penalty: override R_DEATH for this run only, so the
      report can show a real number behind the death-penalty choice rather
      than an opinion (docs/KHANG.md C.3).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

# `python scripts/train.py` only puts this file's own directory (scripts/) on
# sys.path, not part2_arena/ -- so `import arena...` fails unless part2_arena/
# is added explicitly. Do this before importing arena.* below.
_PART2_ARENA_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PART2_ARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART2_ARENA_ROOT))

import numpy as np  # noqa: E402
from callbacks import RewardTermLoggingCallback  # noqa: E402  (sibling: scripts/callbacks.py)
from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

from arena.gym_adapter import ArenaGymEnv  # noqa: E402

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
LOGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=int, choices=[1, 2], required=True)
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], required=True)
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="on")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        type=str,
        default="tuned_v3",
        help="hyperparameter preset name in config/hyperparams.json (e.g. baseline, tuned_v1)",
    )
    parser.add_argument(
        "--death-penalty",
        type=float,
        default=None,
        help=(
            "override R_DEATH for this run (ablation only); defaults to "
            "arena/rewards_config.py's value if omitted. Ablation runs are "
            "named 'ablation_style<N>_...' so they can never collide with a "
            "main run's model file or TensorBoard directory."
        ),
    )
    parser.add_argument(
        "--wall-penalty",
        type=float,
        default=None,
        help=(
            "override R_WALL_PROXIMITY_PER_STEP for this run; defaults to "
            "arena/rewards_config.py's value if omitted. Runs with the "
            "override get a '_wall<N>' name suffix so they can never "
            "collide with a main run's model file or TensorBoard directory."
        ),
    )
    return parser.parse_args()


def _load_hyperparams(algo: str, preset: str) -> dict:
    """Load the [algo][preset] block from config/hyperparams.json. Raises
    KeyError (with a clear message) if the algo/preset combination doesn't
    exist, rather than silently falling back to something else -- a typo'd
    --config should fail loudly, not train with the wrong hyperparameters.
    """
    with open(CONFIG_DIR / "hyperparams.json", "r", encoding="utf-8") as f:
        all_presets = json.load(f)
    try:
        return dict(all_presets[algo][preset])
    except KeyError as exc:
        raise KeyError(
            f"No hyperparameter preset '{preset}' for algo '{algo}' in "
            f"{CONFIG_DIR / 'hyperparams.json'}"
        ) from exc


def build_model(
    algo: str, env, tensorboard_log: str, preset: str = "tuned_v1", seed: int | None = None
):
    """Construct a stable_baselines3.PPO or DQN model with an MLP policy
    and meaningfully tuned hyperparameters loaded from
    config/hyperparams.json's [algo][preset] block -- not hardcoded here,
    so a hyperparameter sweep (report section 4) is just a matter of
    running with different --config values, and every run stays
    reproducible against the exact preset that trained it (see
    hyperparams.json's own "_notes" for why presets are additive, not
    edited in place).
    """
    params = _load_hyperparams(algo, preset)
    policy = params.pop("policy", "MlpPolicy")
    net_arch = params.pop("net_arch", [64, 64])
    policy_kwargs = dict(net_arch=net_arch)

    model_cls = PPO if algo == "ppo" else DQN
    return model_cls(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=1,
        **params,
    )


def run_name(
    style: int,
    algo: str,
    curriculum: str,
    preset: str = "tuned_v1",
    death_penalty: float | None = None,
    wall_penalty: float | None = None,
) -> str:
    """Single source of truth for a run's identity, used for BOTH the model
    filename and the TensorBoard run name so the two can never drift apart.

    Ablation runs (--death-penalty set) get an "ablation_" PREFIX rather
    than only a suffix. A suffix alone would make the ablation's directory
    name a prefix-extension of the main run's, and
    scripts/plot_reward_decomposition.py::find_log_dir locates runs with a
    prefix glob -- it would then happily read an ablation run's scalars
    while reporting the main run's numbers.

    --wall-penalty runs get a "_wall<N>" suffix: these are MAIN runs (the
    suffix keeps them from colliding with the canonical name), and a prefix
    glob would still find them alongside the main run's scalars -- but each
    wall override value lands in its own directory, so runs are never
    conflated.
    """
    curriculum_suffix = "_curriculum" if curriculum == "on" else ""
    if death_penalty is None:
        base = f"style{style}_{algo}_{preset}{curriculum_suffix}"
    else:
        base = (
            f"ablation_style{style}_{algo}_{preset}{curriculum_suffix}"
            f"_death{int(death_penalty)}"
        )
    if wall_penalty is not None:
        base += f"_wall{wall_penalty:g}"
    return base


def model_save_path(
    style: int,
    algo: str,
    curriculum: str,
    preset: str = "tuned_v1",
    death_penalty: float | None = None,
    wall_penalty: float | None = None,
) -> pathlib.Path:
    # Preset is part of the filename so a hyperparameter sweep does not
    # overwrite its own earlier runs; death_penalty likewise keeps an
    # ablation run from clobbering the real one; wall_penalty keeps an
    # override run from clobbering the canonical one.
    return MODELS_DIR / run_name(style, algo, curriculum, preset, death_penalty, wall_penalty)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    curriculum_enabled = args.curriculum == "on"
    # Both the training env and EvalCallback's eval env must use the SAME
    # reward function -- if only one is overridden, "best model by eval
    # reward" is selected against a different objective than the one being
    # trained, and the saved best checkpoint is meaningless.
    reward_overrides: dict[str, float] | None = None
    if args.death_penalty is not None or args.wall_penalty is not None:
        reward_overrides = {}
        if args.death_penalty is not None:
            reward_overrides["R_DEATH"] = args.death_penalty
        if args.wall_penalty is not None:
            reward_overrides["R_WALL_PROXIMITY_PER_STEP"] = args.wall_penalty
    env = Monitor(
        ArenaGymEnv(
            control_style=args.style,
            curriculum_enabled=curriculum_enabled,
            reward_overrides=reward_overrides,
        )
    )
    model = build_model(
        args.algo, env, tensorboard_log=str(LOGS_DIR), preset=args.config, seed=args.seed
    )

    save_path = model_save_path(
        args.style,
        args.algo,
        args.curriculum,
        args.config,
        args.death_penalty,
        args.wall_penalty,
    )
    eval_env = Monitor(
        ArenaGymEnv(
            control_style=args.style,
            curriculum_enabled=curriculum_enabled,
            reward_overrides=reward_overrides,
        )
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path) + "_best",
        eval_freq=max(1000, args.timesteps // 10),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    callback = CallbackList([RewardTermLoggingCallback(), eval_callback])

    # tb_log_name encodes style/algo/preset/curriculum (and the ablation
    # marker) so each run gets its own discoverable TensorBoard subfolder
    # (SB3 auto-appends "_1", "_2", ... on repeat runs) --
    # scripts/compare_ppo_dqn.py, scripts/plot_reward_decomposition.py and
    # scripts/plot_death_penalty_ablation.py locate the right run by this
    # name, so it MUST stay in sync with model_save_path (both come from
    # run_name()).
    tb_log_name = run_name(
        args.style,
        args.algo,
        args.curriculum,
        args.config,
        args.death_penalty,
        args.wall_penalty,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name=tb_log_name)
    model.save(save_path)

    print(f"Saved final model to: {save_path}")
    print(f"Saved best checkpoint (by eval reward) under: {save_path}_best")
    print(f"TensorBoard logs under: {LOGS_DIR} (run `tensorboard --logdir {LOGS_DIR}`)")


if __name__ == "__main__":
    main()
