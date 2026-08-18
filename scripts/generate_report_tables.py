"""Generate report-ready reward tables from the single sources of truth.

Reads reward/mechanic constants from BOTH parts of the project:
  - part1_gridworld/config/rewards_constants.py
  - part2_arena/arena/rewards_config.py

and writes Markdown tables to report/figures/reward_tables.md.

This is the only script that should ever produce the reward tables used in
the report — running it after any change to either rewards_constants module
keeps the report from drifting out of sync with the actual code. Do not
hand-copy reward values into the report.

Usage:
    python scripts/generate_report_tables.py
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "report" / "figures" / "reward_tables.md"


def load_part1_constants() -> dict:
    """Import part1_gridworld/config/rewards_constants.py and return its
    public reward constants as a name -> value dict.

    TODO: import the module (sys.path manipulation or importlib), filter to
    UPPER_CASE constants, return them.
    """
    raise NotImplementedError


def load_part2_constants() -> dict:
    """Import part2_arena/arena/rewards_config.py and return its public
    reward constants as a name -> value dict, plus each constant's
    docstring/justification if available.

    TODO: same approach as load_part1_constants, but also capture the
    one-line justification required for each Part II shaping term.
    """
    raise NotImplementedError


def render_markdown_table(title: str, constants: dict) -> str:
    """Render a name/value(/justification) dict as a Markdown table under
    the given heading.

    TODO: implement formatting.
    """
    raise NotImplementedError


def main() -> None:
    """Load both parts' constants, render tables, write to OUTPUT_PATH.

    TODO: wire load_part1_constants / load_part2_constants /
    render_markdown_table together and write the combined output.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
