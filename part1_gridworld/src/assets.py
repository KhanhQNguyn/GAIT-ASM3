"""Small shared helpers for Part I asset files (fonts).

Keeps the OS-dependent font resolution in one place so render.py, menu.py,
and main.py all load the same bundled Kenney Pixel font and fall back
gracefully when pygame cannot load it (e.g. missing file on a fresh clone).
"""

from __future__ import annotations

import pathlib

import pygame

# part1_gridworld/assets/fonts/Kenney Pixel.ttf  (CC0, see assets/README.md)
_ASSETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"
FONT_PATH = _ASSETS_DIR / "fonts" / "Kenney Pixel.ttf"


def load_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Load the bundled Kenney Pixel font at `size`, falling back to pygame's
    default font if the .ttf cannot be loaded. `bold` is accepted for API
    symmetry but has no effect - the bundled .ttf has no bold face, and
    pygame's default Font(None, size) has no bold variant either.
    """
    try:
        return pygame.font.Font(str(FONT_PATH), size)
    except Exception:
        return pygame.font.Font(None, size)
