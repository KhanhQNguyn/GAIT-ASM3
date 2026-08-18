"""Plotting utilities that turn logger.py's CSV output into the training
curve evidence required by the rubric:
  - Task 4: training curves on levels 4 and 5.
  - Task 5: curves comparing WITH vs. WITHOUT intrinsic reward on level 6,
    plus (in the report, not here) a short written explanation.

Figures are written to report/figures/ so generate_report_tables.py's
sibling plot scripts all land in one place the report template already
points to.
"""

from __future__ import annotations

import pathlib

FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "report" / "figures"


def load_episode_csv(csv_path: str | pathlib.Path):
    """Load an EpisodeLogger CSV into a simple structure (e.g. a dict of
    lists, or a pandas DataFrame if the team adds pandas to requirements.txt).

    TODO: implement.
    """
    raise NotImplementedError


def plot_training_curve(csv_paths: dict[str, str], title: str, output_name: str) -> pathlib.Path:
    """Plot one or more episode-return curves (e.g. {"level4": "...csv",
    "level5": "...csv"} or {"with_intrinsic": "...csv", "without_intrinsic":
    "...csv"}) on the same axes with a rolling-average smoothing line, save
    to FIGURES_DIR / output_name, and return the saved path.

    TODO: implement with matplotlib.
    """
    raise NotImplementedError
