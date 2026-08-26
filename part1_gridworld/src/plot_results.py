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


def plot_death_rate(csv_paths: dict[str, str], title: str, output_name: str) -> pathlib.Path:
    """Plot rolling death-rate vs episode for the monster levels (Task 4).
    The EpisodeLogger CSV already records a `died` bool per episode
    (logger.py FIELDNAMES), so this needs no new logging -- just a rolling
    mean of that column. A falling curve on level4/level5 is the cleanest
    single piece of "the agent learned to avoid monsters" evidence for the
    report (see docs/AUDIT_main.md 6.8).

    TODO: implement with matplotlib (mirror plot_training_curve's structure;
    y-axis 0..1, rolling window e.g. 50 episodes).
    """
    raise NotImplementedError
