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
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def log_episode(
        self, episode: int, total_return: float, steps: int, died: bool, epsilon: float
    ) -> None:
        """Append one row.

        TODO: implement.
        """
        self._writer.writerow({
            "episode": episode,
            "return": total_return,
            "steps": steps,
            "died": died,
            "epsilon": epsilon,
        })

    def close(self) -> None:
        """Flush and close the underlying file.

        TODO: implement.
        """
        self._file.flush()
        self._file.close()
