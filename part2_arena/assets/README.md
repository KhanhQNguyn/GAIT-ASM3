# Part II Assets

## Fonts

- **`fonts/ShareTechMono-Regular.ttf`** — "Share Tech Mono" by Carrois Type Design /
  Ralph du Carrois. Licensed under the SIL Open Font License, Version 1.1 (full text in
  `fonts/OFL.txt`). Source: https://github.com/google/fonts/tree/main/ofl/sharetechmono
  (Google Fonts mirror of the OFL-licensed family). Used by `arena/render_pygame.py` for
  the HUD/debug-overlay text so it renders identically across operating systems instead of
  depending on whichever OS-installed font `pygame.font.SysFont` happens to resolve.

## Sprites (2026-09-08 animation pass)

All sprite PNGs live under **`sprites/`** in the subfolders below and are
loaded through `arena/sprites.py::load_sprite("subdir/name", size)` — the
loader accepts subfolder paths, caches, scales to the requested size, and
NEVER raises. Every draw site in `arena/render_pygame.py` keeps its original
`pygame.draw.*` placeholder shape as a fallback, so deleting any PNG (or all
of them) leaves the game fully playable with plain shapes.

Note: PNG file contains iCCP chunk (embedded color profile) with non-conformant sRGB data — common in PNGs exported by Photoshop/older tools. libpng prints warning, but loads image normally. Zero effect on gameplay or rendering. 

### `sprites/enemy/` — enemy turn animation (5 frames, ORDER MATTERS)

`enemy_l2.png, enemy_l1.png, enemy_m.png, enemy_r1.png, enemy_r2.png`

**All ship and enemy sprites face UP in their source files.** The renderer
rotates every blit so the nose tracks the entity's heading
(`render_pygame.py::_blit_facing`, heading = enemy→player bearing for
enemies, `player.orientation` for the ship). The bank frames below select
WHICH variant is shown and sweep in order on direction flips; the rotation
aligns the chosen frame to the actual heading, so the sweep reads as
banking relative to it.

Frame table in `render_pygame.py::_ENEMY_TURN_FRAMES` is ordered
full-left → full-right: **l2, l1, m, r1, r2**. Each enemy's current frame
walks ONE index per rendered frame toward the travel-direction target
(`cos(angle enemy→player)` decides left/right/neutral), so a direction flip
always sweeps through the middle frame in the exact required order — e.g.
turning from hard-left to right plays `l2 → l1 → m → r1 → r2` — with a
~1-frame-per-index transition (very short, matched to the bots' rapid
motion). New enemies snap straight to their target frame.

### `sprites/explosion/` — hit animation (11 frames)

`explosion_01.png` … `explosion_11.png` — played IN ORDER at the impact
point whenever an enemy is damaged by a projectile (not only on kills;
`core_env._resolve_collisions` emits a render-only `("hit", x, y)` event per
enemy hit). One frame per rendered frame (~0.18 s at 60 fps), removed after
the last frame. Kills additionally keep the expanding ring flash.

### `sprites/ship/` — player health tiers (4 sprites)

`ship_fullhealth.png, ship_slightdamaged.png, ship_damaged.png,
ship_verydamaged.png` — swapped automatically by
`render_pygame.py::_ship_sprite_name` from the health fraction (tier table
below) AND rotated every frame so the nose tracks `player.orientation`
(style 1: free rotation; style 2: current velocity direction):

| health | sprite |
|---|---|
| 100–76 | ship_fullhealth |
| 75–51 | ship_slightdamaged |
| 50–26 | ship_damaged |
| < 25 | ship_verydamaged |

### `sprites/spawner/` — looping animation (9 frames)

`mine_01.png` … `mine_09.png` — looped in order (each frame held 2 rendered
frames). All spawners share the renderer's animation clock, so the loop
stays ordered. The old single `sprites/spawner.png` override is superseded
by this folder.

## Background

**`background/Purple_Nebula_06-1024x1024.png`** — loaded via
`arena/sprites.py::load_image` (a generic `assets/<relpath>` loader with the
same never-raises contract) and cover-cropped (aspect-preserving center
crop) to the window size by `render_pygame.py::_load_background`. Missing
file → the original solid-color background.

## Link

- **Ship/Enemies/Spawners/Explosions** - https://ravenmore.itch.io/space-shooter-assets-space-rage
- **Background** - https://screamingbrainstudios.itch.io/seamless-space-backgrounds
