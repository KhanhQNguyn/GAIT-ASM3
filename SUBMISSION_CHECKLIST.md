# Submission Checklist (STRICT — exactly 2 items required)

## 1. Zip file (link to it)

Contents required:
- [ ] All gridworld code (`part1_gridworld/`)
- [ ] Arena simulation code (`part2_arena/arena/`)
- [ ] RL training scripts (`part1_gridworld/src/trainer.py`, `part2_arena/scripts/train.py`, etc.)
- [ ] Saved models (`part2_arena/models/`)
- [ ] TensorBoard logs (`part2_arena/logs/`)

Also: the actual zip file must be uploaded to Canvas per the submission
instructions — a link alone is not sufficient there.

## 2. Report PDF

- [ ] Strict max 10 pages, including images, **no appendix** (anything beyond
      page 10 will not be considered).
- [ ] Description of both environments.
- [ ] Observation design (each feature explained + justified).
- [ ] Reward design (explained + justified, including any shaping).
- [ ] Hyperparameter exploration (tables/plots as evidence).
- [ ] Comparison of the two control sets (curves/screenshots/logs).
- [ ] Evidence of training (logs + screenshots).
- [ ] Originality justification.
- [ ] Student numbers + contribution summary for all team members (see `CONTRIBUTIONS.md`).
- [ ] Link to the video recording demonstration (see `VIDEO_SCRIPT.md`).

## Before you submit

- [ ] Every row in `RUBRIC_MAP.md` is `DONE`.
- [ ] `python scripts/generate_report_tables.py` has been re-run against the
      final code so report tables match the submitted constants exactly.
- [ ] Models in `part2_arena/models/` are the exact ones shown in the video.
- [ ] If submitting late: notify the teaching team so the correct version is
      downloaded. 10% penalty applies per calendar day late.
