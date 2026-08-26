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
    parser.add_argument("--curriculum", type=str, choices=["on", "off"], default="off")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        type=str,
        default="tuned_v1",
        help="hyperparameter preset name in config/hyperparams.json (e.g. baseline, tuned_v1)",
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


def model_save_path(
    style: int, algo: str, curriculum: str, preset: str = "tuned_v1"
) -> pathlib.Path:
    suffix = "_curriculum" if curriculum == "on" else ""
    # Preset is part of the filename so a hyperparameter sweep does not
    # overwrite its own earlier runs.
    return MODELS_DIR / f"style{style}_{algo}_{preset}{suffix}"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    curriculum_enabled = args.curriculum == "on"
    env = Monitor(ArenaGymEnv(control_style=args.style, curriculum_enabled=curriculum_enabled))
    model = build_model(
        args.algo, env, tensorboard_log=str(LOGS_DIR), preset=args.config, seed=args.seed
    )

    save_path = model_save_path(args.style, args.algo, args.curriculum, args.config)
    eval_env = Monitor(
        ArenaGymEnv(control_style=args.style, curriculum_enabled=curriculum_enabled)
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

    # tb_log_name encodes style/algo/preset/curriculum so each run gets its
    # own discoverable TensorBoard subfolder (SB3 auto-appends "_1", "_2",
    # ... on repeat runs) -- scripts/compare_ppo_dqn.py and
    # scripts/plot_reward_decomposition.py locate the right run by this name.
    curriculum_suffix = "_curriculum" if curriculum_enabled else ""
    tb_log_name = f"style{args.style}_{args.algo}_{args.config}{curriculum_suffix}"
    model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name=tb_log_name)
    model.save(save_path)

    print(f"Saved final model to: {save_path}")
    print(f"Saved best checkpoint (by eval reward) under: {save_path}_best")
    print(f"TensorBoard logs under: {LOGS_DIR} (run `tensorboard --logdir {LOGS_DIR}`)")


if __name__ == "__main__":
    main()
