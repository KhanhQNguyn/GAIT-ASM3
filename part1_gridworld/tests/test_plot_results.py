"""Tests for src/plot_results.py -- CSV loading and training-curve plotting."""

from logger import EpisodeLogger
from plot_results import load_episode_csv, plot_training_curve


def test_load_episode_csv_parses_logger_output(tmp_path):
    p = tmp_path / "e.csv"
    log = EpisodeLogger(p)
    log.log_episode(0, 1.0, 5, False, 0.9)
    log.log_episode(1, 2.0, 7, True, 0.8)
    log.close()
    data = load_episode_csv(p)
    assert data["episode"] == [0, 1]
    assert data["return"] == [1.0, 2.0]
    assert data["died"] == [False, True]


def test_plot_training_curve_writes_figure(tmp_path, monkeypatch):
    monkeypatch.setattr("plot_results.FIGURES_DIR", tmp_path)
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    for p, vals in ((p1, [1, 2, 3]), (p2, [2, 3, 4])):
        log = EpisodeLogger(p)
        for i, v in enumerate(vals):
            log.log_episode(i, v, 5, False, 0.9)
        log.close()
    out = plot_training_curve({"A": str(p1), "B": str(p2)}, "Test", "test_curve.png")
    assert out == tmp_path / "test_curve.png"
    assert out.exists()
