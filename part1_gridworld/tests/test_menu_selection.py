"""The menu is the only place the Task-5 intrinsic-reward choice is made, so
MenuSelection must carry it through to main.py -- defaulting OFF, and
round-tripping through __repr__ (used in main.py's training log line).
"""

from src.menu import INTRINSIC_LEVEL_ID, MenuSelection


def test_intrinsic_level_id_is_six():
    # Level 6 is the only level whose layout isolates the intrinsic reward
    # (see config/level6.json _design_note); the menu toggle keys off this.
    assert INTRINSIC_LEVEL_ID == 6


def test_menuselection_defaults_intrinsic_off():
    sel = MenuSelection(level_id=0, algorithm="q_learning", watch_only=False)
    assert sel.use_intrinsic_reward is False


def test_menuselection_carries_intrinsic_flag():
    sel = MenuSelection(
        level_id=6,
        algorithm="q_learning",
        watch_only=False,
        use_intrinsic_reward=True,
    )
    assert sel.use_intrinsic_reward is True
    assert "use_intrinsic_reward=True" in repr(sel)
