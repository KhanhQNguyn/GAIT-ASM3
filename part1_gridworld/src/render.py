"""Pygame rendering ONLY. This module receives a GridWorldEnv's state and
draws it -- it must never contain game logic, reward computation, or RL
decision-making. Keeping this separate from env.py is what lets env.py stay
pygame-free and unit-testable headlessly.

Creativity additions (Section 4 of MEMBER_A_GRIDWORLD_CORE.md):
  - Smooth agent interpolation: lerp pixel position over LERP_FRAMES frames.
  - HUD overlay: episode number, current epsilon, running return displayed
    in-window so the training run is legible evidence of learning.
"""

from __future__ import annotations

import pygame

TILE_SIZE_PX = 48
COLORS = {
    "background": (20, 20, 35),
    "grid_line": (45, 45, 65),
    "agent": (66, 200, 255),
    "rock": (100, 100, 120),
    "fire": (235, 80, 50),
    "apple": (230, 60, 100),
    "key": (250, 220, 60),
    "chest": (200, 150, 60),
    "monster": (180, 60, 230),
    "hud_bg": (0, 0, 0, 160),
    "hud_text": (220, 220, 240),
    "hud_accent": (66, 200, 255),
}

# Smooth interpolation: agent position lerps over this many render calls
LERP_FRAMES: int = 6


def _tile_center(x: int, y: int) -> tuple[int, int]:
    """Pixel center of tile (x, y)."""
    return (
        x * TILE_SIZE_PX + TILE_SIZE_PX // 2,
        y * TILE_SIZE_PX + TILE_SIZE_PX // 2,
    )


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class GridWorldRenderer:
    """Owns the pygame window/surface and draws one GridWorldEnv frame at a
    time. Does not own the environment or the training loop.

    Takes plain-dict state snapshots (from GridWorldEnv.get_state_snapshot())
    — never the GridWorldEnv object itself, preserving the decoupling.
    """

    def __init__(self, grid_size: tuple[int, int], caption: str = "Gridworld RL"):
        """Initialise pygame and create the window.

        Args:
            grid_size: (width, height) in tiles.
            caption: Window title string.
        """
        self.grid_size = grid_size
        gw, gh = grid_size
        width = gw * TILE_SIZE_PX
        # Extra vertical space for HUD at top
        self._hud_height = 56
        height = gh * TILE_SIZE_PX + self._hud_height

        if not pygame.get_init():
            pygame.init()

        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)

        self._clock = pygame.time.Clock()

        # Font for HUD text
        pygame.font.init()
        try:
            self._font_large = pygame.font.SysFont("Segoe UI", 18, bold=True)
            self._font_small = pygame.font.SysFont("Segoe UI", 14)
        except Exception:
            self._font_large = pygame.font.Font(None, 22)
            self._font_small = pygame.font.Font(None, 18)

        # Smooth interpolation state
        self._agent_pixel: tuple[float, float] | None = None
        self._target_pixel: tuple[float, float] | None = None
        self._lerp_t: float = 1.0  # 1.0 = fully at target

        # HUD data injected by caller (trainer.py calls .set_hud_info())
        self._hud: dict = {
            "episode": 0,
            "epsilon": 1.0,
            "return_": 0.0,
            "step": 0,
        }

    def set_hud_info(
        self,
        episode: int = 0,
        epsilon: float = 1.0,
        return_: float = 0.0,
        step: int = 0,
    ) -> None:
        """Update HUD data shown during training. Called by trainer.py each step.

        Args:
            episode: Current episode number.
            epsilon: Current exploration rate.
            return_: Running episode return so far.
            step: Current step within episode.
        """
        self._hud = {
            "episode": episode,
            "epsilon": epsilon,
            "return_": return_,
            "step": step,
        }

    def draw(self, env_state: dict) -> None:
        """Draw one frame from a plain-data snapshot.

        Args:
            env_state: dict as returned by GridWorldEnv.get_state_snapshot().
                Required keys: grid_w, grid_h, agent_pos, rocks, fire, apples,
                key_pos, chest_pos, monsters, has_key, step_count, max_steps.
        """
        self._screen.fill(COLORS["background"])

        gw: int = env_state["grid_w"]
        gh: int = env_state["grid_h"]
        off_y = self._hud_height  # vertical offset for grid (HUD at top)

        # --- Grid lines ---
        for col in range(gw + 1):
            x = col * TILE_SIZE_PX
            pygame.draw.line(
                self._screen, COLORS["grid_line"],
                (x, off_y), (x, off_y + gh * TILE_SIZE_PX)
            )
        for row in range(gh + 1):
            y = off_y + row * TILE_SIZE_PX
            pygame.draw.line(
                self._screen, COLORS["grid_line"],
                (0, y), (gw * TILE_SIZE_PX, y)
            )

        def _draw_tile(tx: int, ty: int, color: tuple, shape: str = "fill", margin: int = 4) -> None:
            rx = tx * TILE_SIZE_PX + margin
            ry = off_y + ty * TILE_SIZE_PX + margin
            rw = TILE_SIZE_PX - 2 * margin
            rh = TILE_SIZE_PX - 2 * margin
            if shape == "fill":
                pygame.draw.rect(self._screen, color, (rx, ry, rw, rh), border_radius=6)
            elif shape == "circle":
                cx = tx * TILE_SIZE_PX + TILE_SIZE_PX // 2
                cy = off_y + ty * TILE_SIZE_PX + TILE_SIZE_PX // 2
                r = TILE_SIZE_PX // 2 - margin
                pygame.draw.circle(self._screen, color, (cx, cy), r)
            elif shape == "diamond":
                cx = tx * TILE_SIZE_PX + TILE_SIZE_PX // 2
                cy = off_y + ty * TILE_SIZE_PX + TILE_SIZE_PX // 2
                r = TILE_SIZE_PX // 2 - margin
                pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
                pygame.draw.polygon(self._screen, color, pts)

        # --- Rocks ---
        for rx, ry in env_state["rocks"]:
            _draw_tile(rx, ry, COLORS["rock"], shape="fill", margin=2)

        # --- Fire ---
        for fx, fy in env_state["fire"]:
            _draw_tile(fx, fy, COLORS["fire"], shape="fill", margin=3)
            # Inner bright highlight
            _draw_tile(fx, fy, (255, 160, 80), shape="diamond", margin=12)

        # --- Apples ---
        for ax, ay in env_state["apples"]:
            _draw_tile(ax, ay, COLORS["apple"], shape="circle", margin=8)

        # --- Key ---
        if env_state.get("key_pos") is not None:
            kx, ky = env_state["key_pos"]
            _draw_tile(kx, ky, COLORS["key"], shape="diamond", margin=8)

        # --- Chest ---
        if env_state.get("chest_pos") is not None:
            cx, cy = env_state["chest_pos"]
            color = COLORS["chest"] if not env_state.get("chest_open") else (80, 200, 80)
            _draw_tile(cx, cy, color, shape="fill", margin=6)

        # --- Monsters ---
        for mx, my in env_state["monsters"]:
            _draw_tile(mx, my, COLORS["monster"], shape="circle", margin=6)
            # Eyes
            eye_y = off_y + my * TILE_SIZE_PX + TILE_SIZE_PX // 3
            for ex in [mx * TILE_SIZE_PX + TILE_SIZE_PX // 3, mx * TILE_SIZE_PX + 2 * TILE_SIZE_PX // 3]:
                pygame.draw.circle(self._screen, (255, 255, 255), (ex, eye_y), 3)
                pygame.draw.circle(self._screen, (0, 0, 0), (ex + 1, eye_y), 1)

        # --- Agent (smooth interpolation) ---
        ax, ay = env_state["agent_pos"]
        target_px = (
            ax * TILE_SIZE_PX + TILE_SIZE_PX // 2,
            off_y + ay * TILE_SIZE_PX + TILE_SIZE_PX // 2,
        )

        if self._agent_pixel is None:
            # First frame — snap to position
            self._agent_pixel = target_px
            self._target_pixel = target_px
            self._lerp_t = 1.0
        elif target_px != self._target_pixel:
            # New target — start lerp from current
            self._target_pixel = target_px
            self._lerp_t = 0.0

        # Advance lerp
        if self._lerp_t < 1.0:
            self._lerp_t = min(1.0, self._lerp_t + 1.0 / LERP_FRAMES)
        px = _lerp(self._agent_pixel[0], self._target_pixel[0], self._lerp_t)
        py = _lerp(self._agent_pixel[1], self._target_pixel[1], self._lerp_t)
        self._agent_pixel = (px, py)

        r = TILE_SIZE_PX // 2 - 7
        pygame.draw.circle(self._screen, COLORS["agent"], (int(px), int(py)), r)
        # Key indicator on agent when carrying
        if env_state.get("has_key"):
            pygame.draw.circle(self._screen, COLORS["key"], (int(px), int(py)), r // 2)

        # --- HUD panel ---
        self._draw_hud(gw)

        pygame.display.flip()
        self._clock.tick(60)

    def _draw_hud(self, grid_w: int) -> None:
        """Draw the top HUD bar with episode/epsilon/return info."""
        panel_w = grid_w * TILE_SIZE_PX
        # Semi-transparent dark bar
        hud_surf = pygame.Surface((panel_w, self._hud_height), pygame.SRCALPHA)
        hud_surf.fill((10, 10, 20, 210))
        self._screen.blit(hud_surf, (0, 0))

        ep = self._hud["episode"]
        eps = self._hud["epsilon"]
        ret = self._hud["return_"]
        stp = self._hud["step"]

        # Left section: Episode
        ep_label = self._font_small.render("EPISODE", True, (130, 130, 160))
        ep_val = self._font_large.render(str(ep), True, COLORS["hud_accent"])
        self._screen.blit(ep_label, (8, 6))
        self._screen.blit(ep_val, (8, 22))

        # Centre: Epsilon
        eps_label = self._font_small.render("ε (explore)", True, (130, 130, 160))
        eps_val = self._font_large.render(f"{eps:.3f}", True, (255, 200, 80))
        self._screen.blit(eps_label, (140, 6))
        self._screen.blit(eps_val, (140, 22))

        # Right: Return & step
        ret_label = self._font_small.render("Return / Step", True, (130, 130, 160))
        ret_val = self._font_large.render(f"{ret:.1f} / {stp}", True, (100, 230, 120))
        self._screen.blit(ret_label, (280, 6))
        self._screen.blit(ret_val, (280, 22))

    def handle_events(self) -> bool:
        """Pump the pygame event queue.

        Returns:
            False if the window was closed (caller should stop the loop),
            True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def close(self) -> None:
        """Tear down the pygame window."""
        pygame.quit()
