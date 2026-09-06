# Part II Assets

## Fonts

- **`fonts/ShareTechMono-Regular.ttf`** — "Share Tech Mono" by Carrois Type Design /
  Ralph du Carrois. Licensed under the SIL Open Font License, Version 1.1 (full text in
  `fonts/OFL.txt`). Source: https://github.com/google/fonts/tree/main/ofl/sharetechmono
  (Google Fonts mirror of the OFL-licensed family). Used by `arena/render_pygame.py` for
  the HUD/debug-overlay text so it renders identically across operating systems instead of
  depending on whichever OS-installed font `pygame.font.SysFont` happens to resolve.

## Sprites

- **`sprites/`** — optional PNG overrides for the arena's `pygame.draw.*` placeholder
  shapes, loaded via `arena/sprites.py::load_sprite(name, size)`. Empty by default (the
  game renders identically with plain shapes); drop in `player.png`, `enemy.png`,
  `spawner.png`, `projectile_player.png`, or `projectile_enemy.png` to replace that one
  shape with no code change. `part1_gridworld/src/sprites.py` is the same loader,
  duplicated rather than shared, for `part1_gridworld/assets/sprites/` (agent, rock,
  fire, apple, key, chest, monster) — the two parts stay independently runnable.
