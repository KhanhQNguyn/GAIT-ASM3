"""Optional PNG sprite loading for the Part II renderer. Never required --
every call site keeps its original pygame.draw.* shape as a fallback, so the
game behaves identically whether or not part2_arena/assets/sprites/
contains any images. Deliberately duplicated (not shared) with
part1_gridworld/src/sprites.py so the two parts stay independently runnable.
"""

from __future__ import annotations

import pathlib

import pygame

ASSETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"
_sprite_cache: dict[str, pygame.Surface | None] = {}


def _load_surface(path: pathlib.Path, size: tuple[int, int] | None) -> pygame.Surface | None:
    """Load + optionally scale a PNG. Returns None (never raises) if the
    file is missing or invalid -- callers must fall back to their existing
    pygame.draw shape in that case.
    """
    try:
        surf = pygame.image.load(str(path)).convert_alpha()
        if size is not None:
            surf = pygame.transform.smoothscale(surf, size)
    except (FileNotFoundError, pygame.error):
        return None
    return surf


def load_sprite(name: str, size: tuple[int, int]) -> pygame.Surface | None:
    """Load + cache a PNG from assets/sprites/<name>.png (subfolders OK:
    name may contain '/' e.g. "enemy/enemy_m"), scaled to `size`.
    """
    cache_key = f"{name}:{size[0]}x{size[1]}"
    if cache_key in _sprite_cache:
        return _sprite_cache[cache_key]
    surf = _load_surface(ASSETS_DIR / "sprites" / f"{name}.png", size)
    _sprite_cache[cache_key] = surf
    return surf


def load_image(relpath: str, size: tuple[int, int] | None = None) -> pygame.Surface | None:
    """Load + cache a PNG from assets/<relpath> (e.g. the background), at
    native resolution when `size` is None. Same never-raises contract.
    """
    cache_key = f"img:{relpath}:{size}"
    if cache_key in _sprite_cache:
        return _sprite_cache[cache_key]
    surf = _load_surface(ASSETS_DIR / relpath, size)
    _sprite_cache[cache_key] = surf
    return surf
