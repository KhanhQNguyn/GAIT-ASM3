"""Discrete action definitions for the two required control schemes.
Kept in their own module so core_env.py, gym_adapter.py, train.py, and both
eval scripts all reference the SAME action ordering -- a mismatch here
between training and evaluation would silently produce nonsense play.
"""

from __future__ import annotations

from enum import IntEnum


class ControlStyle1(IntEnum):
    """Rotation + Thrust control scheme."""

    NO_OP = 0
    THRUST_FORWARD = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    SHOOT = 4


class ControlStyle2(IntEnum):
    """Direct directional movement control scheme."""

    NO_OP = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    SHOOT = 5


def action_enum_for_style(style: int) -> type[IntEnum]:
    """Return ControlStyle1 or ControlStyle2 for style in {1, 2}.

    Raises ValueError for anything else so a typo'd --style flag fails
    loudly instead of silently training the wrong action set.
    """
    if style == 1:
        return ControlStyle1
    if style == 2:
        return ControlStyle2
    raise ValueError(f"control style must be 1 or 2, got {style!r}")
