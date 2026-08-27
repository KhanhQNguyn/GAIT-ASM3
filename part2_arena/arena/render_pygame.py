"""Pygame rendering ONLY, for the arena. Mirrors part1_gridworld's render.py
separation: this module reads an ArenaState snapshot and draws it; it must
never mutate game state or compute rewards.

`draw()` takes an optional `extra` dict so evaluation can overlay debugging
info without changing the core render contract (core_env.render() calls
`renderer.draw(state, extra)` where `extra` carries transient hit/kill
effects, the last observation vector, and the last step_events):

    extra = {
        "events":      list[("kill", x, y) | ("player_hit",)],   # this step
        "obs":         np.ndarray | None,     # last observation vector
        "step_events": dict | None,           # last step_events dict
    }
"""

from __future__ import annotations

import math

import pygame

from arena.entities import ArenaState
from arena.obs import OBSERVATION_SPEC

COLORS = {
    "background": (12, 14, 20),
    "player": (66, 200, 245),
    "enemy": (230, 70, 70),
    "spawner": (200, 140, 40),
    "projectile_player": (240, 240, 120),
    "projectile_enemy": (240, 90, 90),
    "health_bar_bg": (60, 60, 60),
    "health_bar_fg": (70, 200, 90),
    "hud_text": (220, 220, 230),
    "kill_flash": (255, 230, 150),
    "damage_tint": (200, 40, 40),
}

_PLAYER_RADIUS = 14.0
_ENEMY_RADIUS = 12.0
_SPAWNER_RADIUS = 18.0
_PROJECTILE_RADIUS = 4.0


class ArenaRenderer:
    """Owns the pygame window and draws one ArenaState frame at a time."""

    def __init__(self, width: int, height: int, caption: str = "Arena"):
        self.width = int(width)
        self.height = int(height)
        pygame.init()
        pygame.display.init()
        try:
            pygame.font.init()
        except Exception:  # pragma: no cover - font subsystem optional
            pass
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 15) if pygame.font.get_init() else None

        self.show_debug = False           # toggled with the 'D' key in handle_events()
        self._prev_player_health: float | None = None
        self._damage_tint_ttl = 0         # frames of red edge-tint remaining
        self._flashes: list[list] = []    # [[x, y, ttl], ...] expanding kill rings

    # ------------------------------------------------------------------ draw
    def draw(self, state: ArenaState, extra: dict | None = None) -> None:
        """Render one frame. `extra` is optional (see module docstring)."""
        extra = extra or {}
        self._ingest_effects(state, extra)

        self.screen.fill(COLORS["background"])

        for sp in state.spawners:
            if not sp.active:
                continue
            self._draw_square(sp.x, sp.y, _SPAWNER_RADIUS, COLORS["spawner"])
            self._draw_health_bar(sp.x, sp.y - _SPAWNER_RADIUS - 8, sp.health, sp.max_health)

        for en in state.enemies:
            pygame.draw.circle(
                self.screen, COLORS["enemy"], (int(en.x), int(en.y)), int(_ENEMY_RADIUS)
            )
            self._draw_health_bar(en.x, en.y - _ENEMY_RADIUS - 7, en.health, en.max_health)

        for pr in state.projectiles:
            key = "projectile_enemy" if pr.owner == "enemy" else "projectile_player"
            pygame.draw.circle(
                self.screen, COLORS[key], (int(pr.x), int(pr.y)), int(_PROJECTILE_RADIUS)
            )

        self._draw_player(state)
        self._draw_flashes()
        self._draw_damage_tint()
        self._draw_hud(state)
        if self.show_debug:
            self._draw_debug(extra)

        pygame.display.flip()

    # --------------------------------------------------------------- effects
    def _ingest_effects(self, state: ArenaState, extra: dict) -> None:
        for ev in extra.get("events", []) or []:
            if ev and ev[0] == "kill" and len(ev) >= 3:
                self._flashes.append([float(ev[1]), float(ev[2]), 10])
            elif ev and ev[0] == "player_hit":
                self._damage_tint_ttl = 8

        # Fallback damage detection if the caller didn't pass explicit events.
        h = state.player.health
        if self._prev_player_health is not None and h < self._prev_player_health - 1e-9:
            self._damage_tint_ttl = max(self._damage_tint_ttl, 8)
        self._prev_player_health = h

    def _draw_flashes(self) -> None:
        for f in self._flashes:
            x, y, ttl = f
            radius = int(6 + (10 - ttl) * 3)
            width = max(1, ttl // 3)
            pygame.draw.circle(self.screen, COLORS["kill_flash"], (int(x), int(y)), radius, width)
            f[2] -= 1
        self._flashes = [f for f in self._flashes if f[2] > 0]

    def _draw_damage_tint(self) -> None:
        if self._damage_tint_ttl <= 0:
            return
        alpha = int(90 * self._damage_tint_ttl / 8)
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        border = 26
        pygame.draw.rect(overlay, (*COLORS["damage_tint"], alpha), overlay.get_rect(), border)
        self.screen.blit(overlay, (0, 0))
        self._damage_tint_ttl -= 1

    # ---------------------------------------------------------------- pieces
    def _draw_player(self, state: ArenaState) -> None:
        p = state.player
        if state.control_style == 1:
            # triangle pointing along orientation (inertial ship)
            pts = []
            for ang_off, dist in ((0.0, 1.4), (2.5, 0.9), (-2.5, 0.9)):
                a = p.orientation + ang_off
                pts.append((p.x + math.cos(a) * _PLAYER_RADIUS * dist,
                            p.y + math.sin(a) * _PLAYER_RADIUS * dist))
            pygame.draw.polygon(self.screen, COLORS["player"], pts)
        else:
            pygame.draw.circle(
                self.screen, COLORS["player"], (int(p.x), int(p.y)), int(_PLAYER_RADIUS)
            )
            # short facing tick so "which way will SHOOT fire" is visible
            tx = p.x + math.cos(p.orientation) * _PLAYER_RADIUS * 1.6
            ty = p.y + math.sin(p.orientation) * _PLAYER_RADIUS * 1.6
            pygame.draw.line(self.screen, COLORS["player"], (p.x, p.y), (tx, ty), 3)
        self._draw_health_bar(p.x, p.y - _PLAYER_RADIUS - 10, p.health, p.max_health, w=42)

    def _draw_square(self, x: float, y: float, r: float, color) -> None:
        rect = pygame.Rect(int(x - r), int(y - r), int(2 * r), int(2 * r))
        pygame.draw.rect(self.screen, color, rect)

    def _draw_health_bar(
        self, cx: float, cy: float, hp: float, max_hp: float, w: int = 26, h: int = 4
    ) -> None:
        if max_hp <= 0:
            return
        frac = max(0.0, min(1.0, hp / max_hp))
        left = int(cx - w / 2)
        top = int(cy)
        pygame.draw.rect(self.screen, COLORS["health_bar_bg"], pygame.Rect(left, top, w, h))
        pygame.draw.rect(
            self.screen, COLORS["health_bar_fg"], pygame.Rect(left, top, int(w * frac), h)
        )

    def _draw_hud(self, state: ArenaState) -> None:
        if self.font is None:
            return
        line = (
            f"phase {state.phase}   hp {int(state.player.health)}/{int(state.player.max_health)}   "
            f"enemies {len(state.enemies)}   step {state.step_count}   style {state.control_style}"
        )
        self.screen.blit(self.font.render(line, True, COLORS["hud_text"]), (8, 8))

    def _draw_debug(self, extra: dict) -> None:
        if self.font is None:
            return
        y = 30
        obs = extra.get("obs")
        if obs is not None:
            self.screen.blit(self.font.render("obs vector:", True, COLORS["hud_text"]), (8, y))
            y += 18
            for (name, _desc), val in zip(OBSERVATION_SPEC, list(obs)):
                txt = f"  {name:<32} {float(val):+.3f}"
                self.screen.blit(self.font.render(txt, True, COLORS["hud_text"]), (8, y))
                y += 15
        se = extra.get("step_events")
        if se:
            y += 8
            self.screen.blit(self.font.render("step_events:", True, COLORS["hud_text"]), (8, y))
            y += 18
            for k, v in se.items():
                txt = f"  {k:<32} {v}"
                self.screen.blit(self.font.render(txt, True, COLORS["hud_text"]), (8, y))
                y += 15

    # ---------------------------------------------------------------- events
    def handle_events(self) -> bool:
        """Pump the pygame event queue; return False if the window was
        closed (so the eval loop can stop), True otherwise. 'D' toggles the
        debug overlay.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
        return True

    def close(self) -> None:
        pygame.quit()
