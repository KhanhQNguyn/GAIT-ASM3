"""Tests for trainer.load_training_config and its validation."""

import pytest
from trainer import _validate_config, load_training_config


def test_load_training_config_merges_overrides():
    base = load_training_config(0)  # no override -> default block
    assert base["episodes"] == 2000
    merged = load_training_config(2)  # override sets max_steps_per_episode 250
    assert merged["max_steps_per_episode"] == 250
    assert merged["episodes"] == 2000  # default preserved where no override


def test_load_training_config_validates_values():
    _validate_config({"alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.0,
                      "epsilon_end": 0.05, "episodes": 100})  # valid, no raise
    for bad in [
        {"alpha": 0.0, "gamma": 0.95, "epsilon_start": 1.0, "epsilon_end": 0.05, "episodes": 100},
        {"alpha": 1.5, "gamma": 0.95, "epsilon_start": 1.0, "epsilon_end": 0.05, "episodes": 100},
        {"alpha": 0.1, "gamma": 0.0, "epsilon_start": 1.0, "epsilon_end": 0.05, "episodes": 100},
        {"alpha": 0.1, "gamma": 0.95, "epsilon_start": 0.0, "epsilon_end": 0.5, "episodes": 100},
        {"alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.2, "epsilon_end": 0.05, "episodes": 100},
        {"alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.0, "epsilon_end": 0.05, "episodes": 0},
    ]:
        with pytest.raises(ValueError):
            _validate_config(bad)
