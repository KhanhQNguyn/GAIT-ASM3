"""Tests for src/logger.py -- the CSV episode-return logger."""

import csv

from src.logger import EpisodeLogger


def test_episode_logger_writes_header_and_rows(tmp_path):
    p = tmp_path / "log.csv"
    log = EpisodeLogger(p)
    log.log_episode(0, 1.5, 10, False, 0.9)
    log.log_episode(1, 2.0, 12, True, 0.8)
    log.close()
    text = p.read_text()
    assert text.splitlines()[0] == "episode,return,steps,died,epsilon"
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {"episode": "0", "return": "1.5", "steps": "10", "died": "False", "epsilon": "0.9"}
    assert rows[0] == expected
    assert rows[1]["died"] == "True"
