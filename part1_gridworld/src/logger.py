"""Lightweight CSV episode-return logger for Part I training runs.

Part I only needs training-curve evidence (levels 4/5, and the intrinsic
on/off comparison for level 6) -- the spec requires TensorBoard for Part II,
not Part I, so this stays a simple CSV + matplotlib pipeline rather than
pulling in a TensorBoard dependency here too.
"""

from __future__ import annotations

import csv
import pathlib


class EpisodeLogger:
    """Appends one row per episode to a CSV file: episode index, total
    return, steps taken, whether the agent died, final epsilon.
    """

    FIELDNAMES = ["episode", "return", "steps", "died", "epsilon"]

    def __init__(self, csv_path: str | pathlib.Path):
        self.csv_path = pathlib.Path(csv_path)
        # TODO: open the file, write the header row via csv.DictWriter.
        raise NotImplementedError

    def log_episode(
        self, episode: int, total_return: float, steps: int, died: bool, epsilon: float
    ) -> None:
        """Append one row.

        TODO: implement.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Flush and close the underlying file.

        TODO: implement.
        """
        raise NotImplementedError
