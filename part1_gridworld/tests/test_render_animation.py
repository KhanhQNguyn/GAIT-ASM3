"""Regression tests for the agent-movement animation fix in src/render.py.

Context (see docs/comments in render.py / trainer.py): the agent used to
visually teleport because the training loop called renderer.draw() exactly
once per env.step(), and the renderer's lerp target reset every time the
agent's tile changed -- so the interpolation never got more than one frame
to run before being redirected. These tests pin down the fixed behaviour:

  1. Calling draw() repeatedly with the *same* post-step snapshot animates
     the agent smoothly from its previous pixel position to the new tile's
     center over multiple frames, instead of jumping there immediately.
  2. `agent_animation_complete` correctly reports whether the glide has
     finished, so a caller (trainer.py) knows how many extra frames to draw.
  3. The renderer's speed control changes how many frames a glide takes
     (action/animation pacing), not the render clock's FPS.
  4. A blocked move (env_state["agent_pos"] unchanged) never starts a glide,
     so the agent can never appear to slide through a rock.

These tests only exercise GridWorldRenderer with hand-built state snapshot
dicts (matching GridWorldEnv.get_state_snapshot()'s shape) -- they never
touch GridWorldEnv or the RL algorithms, keeping with render.py's
"pygame rendering only" contract.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pygame = pytest.importorskip("pygame")

from src.render import GridWorldRenderer  # noqa: E402


def _snapshot(agent_pos: tuple[int, int]) -> dict:
    """Minimal valid get_state_snapshot()-shaped dict for a 5x5 empty grid."""
    return {
        "grid_w": 5,
        "grid_h": 5,
        "agent_pos": agent_pos,
        "rocks": [(2, 2)],
        "fire": [],
        "apples": [],
        "key_pos": None,
        "chest_pos": None,
        "monsters": [],
        "has_key": False,
        "chest_open": False,
        "step_count": 0,
        "max_steps": 100,
        "done": False,
    }


@pytest.fixture
def renderer():
    r = GridWorldRenderer(grid_size=(5, 5), caption="test")
    yield r
    r.close()


def test_agent_glides_over_multiple_frames_not_one(renderer):
    """Redrawing the same post-step snapshot must take >1 frame to settle,
    and the agent's pixel position must move partway toward the target
    before it's reached -- i.e. it glides, it doesn't teleport.
    """
    renderer.draw(_snapshot((0, 0)))  # first frame: snaps to (0, 0)
    assert renderer.agent_animation_complete

    start_pixel = renderer._agent_pixel

    moved_snapshot = _snapshot((1, 0))  # one tile to the right
    renderer.draw(moved_snapshot)
    # Immediately after the move is first drawn, the glide should just be
    # starting, not finished in a single frame.
    assert not renderer.agent_animation_complete
    mid_pixel = renderer._agent_pixel
    assert mid_pixel != start_pixel  # it moved off the start...
    target_x, _ = renderer._target_pixel
    assert mid_pixel[0] != target_x  # ...but hasn't snapped straight to the target

    frames_taken = 1
    while not renderer.agent_animation_complete:
        renderer.draw(moved_snapshot)
        frames_taken += 1
        assert frames_taken < 10_000  # safety net against an infinite loop

    assert frames_taken > 1, "agent reached its target tile in a single frame (teleport bug)"
    assert renderer._agent_pixel == renderer._target_pixel


def test_redrawing_same_snapshot_after_completion_does_not_restart_glide(renderer):
    """Once the glide finishes, calling draw() again with the *same*
    snapshot (as happens right before the next action) must not kick off
    another glide -- only a genuine change in agent_pos should do that.
    """
    renderer.draw(_snapshot((0, 0)))
    snap = _snapshot((1, 0))
    renderer.draw(snap)
    while not renderer.agent_animation_complete:
        renderer.draw(snap)
    settled_pixel = renderer._agent_pixel

    renderer.draw(snap)  # same agent_pos again
    assert renderer.agent_animation_complete
    assert renderer._agent_pixel == settled_pixel


def test_blocked_move_never_animates_through_the_blocked_tile(renderer):
    """If env.step() was blocked (agent_pos unchanged because the target
    tile was a rock/edge), the renderer must never start a glide at all --
    the "target" stays the agent's real, current position.
    """
    renderer.draw(_snapshot((1, 2)))  # start next to the rock at (2, 2)
    assert renderer.agent_animation_complete
    pixel_before = renderer._agent_pixel

    # Environment reports the move was blocked: agent_pos did not change.
    renderer.draw(_snapshot((1, 2)))

    assert renderer.agent_animation_complete
    assert renderer._agent_pixel == pixel_before


def test_speed_changes_glide_frame_count_not_render_fps(renderer):
    """Speed multiplier must scale how many frames a glide takes (action
    pacing), and must never be fed into the render clock's tick rate.
    """
    renderer._speed_multiplier = 1.0
    renderer.draw(_snapshot((0, 0)))
    renderer.draw(_snapshot((1, 0)))
    baseline_frames = renderer._lerp_frames

    renderer._speed_multiplier = 4.0
    renderer.draw(_snapshot((0, 0)))  # settle back to (0, 0) at high speed
    while not renderer.agent_animation_complete:
        renderer.draw(_snapshot((0, 0)))
    renderer.draw(_snapshot((1, 0)))  # new glide started at speed=4.0
    fast_frames = renderer._lerp_frames

    assert fast_frames < baseline_frames

    # The render clock is always fixed at 60 FPS regardless of speed: draw()
    # must tick the clock with a plain 60, never 60 * speed_multiplier.
    class _FakeClock:
        def __init__(self, real_clock):
            self._real_clock = real_clock
            self.tick_calls: list = []

        def tick(self, fps=None):
            self.tick_calls.append(fps)
            return self._real_clock.tick(fps) if fps is not None else self._real_clock.tick()

        def get_time(self):
            return self._real_clock.get_time()

    fake_clock = _FakeClock(renderer._clock)
    renderer._clock = fake_clock
    renderer.draw(_snapshot((1, 0)))
    assert fake_clock.tick_calls == [60]