"""Optional PNG sprite loading for the Part I renderer. Never required --
every call site keeps its original pygame.draw.* shape as a fallback, so the
game behaves identically whether or not part1_gridworld/assets/sprites/
contains any images. Deliberately duplicated (not shared) with
part2_arena/arena/sprites.py so the two parts stay independently runnable.
"""

from __future__ import annotations

import pathlib

import pygame

ASSETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"
_sprite_cache: dict[str, pygame.Surface | None] = {}


def load_sprite(name: str, size: tuple[int, int]) -> pygame.Surface | None:
    """Load + cache a PNG from assets/sprites/<name>.png, scaled to `size`.

    Returns None (never raises) if the file is missing or invalid -- callers
    must fall back to their existing pygame.draw shape in that case.
    """
    cache_key = f"{name}:{size[0]}x{size[1]}"
    if cache_key in _sprite_cache:
        return _sprite_cache[cache_key]
    path = ASSETS_DIR / "sprites" / f"{name}.png"
    try:
        surf = pygame.image.load(str(path)).convert_alpha()
        surf = pygame.transform.smoothscale(surf, size)
    except (FileNotFoundError, pygame.error):
        surf = None
    _sprite_cache[cache_key] = surf
    return surf
