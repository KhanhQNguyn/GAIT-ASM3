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

import ast
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "report" / "figures" / "reward_tables.md"

PART1_CONSTANTS_PATH = ROOT / "part1_gridworld" / "config" / "rewards_constants.py"
PART2_CONSTANTS_PATH = ROOT / "part2_arena" / "arena" / "rewards_config.py"


def _load_module_from_path(name: str, path: pathlib.Path):
    """Import a module by file path.

    Used instead of a normal import because neither constants module's
    package is on sys.path when this script runs from the repo root, and
    because importing `arena.rewards_config` the normal way would drag in
    the whole arena package. Both target files are import-free leaf
    modules, so executing them in isolation is safe.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assign_name(node: ast.stmt) -> str | None:
    """Return the assigned name for `X = ...` and `X: T = ...`, else None."""
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def _first_paragraph(text: str) -> str:
    """Collapse the first paragraph of a justification onto one line.

    Several constants carry multi-paragraph rationale (R_APPROACH_NEAREST_ENEMY
    documents its whole required gating/cap shape; REWARD_DEATH cites
    docs/lesson.md). Only the first paragraph belongs in a report table cell
    -- the rest stays discoverable in the source.
    """
    return " ".join(text.strip().split("\n\n", 1)[0].split())


def _escape_cell(text: str) -> str:
    """Escape pipes so a justification containing `|` cannot break the row."""
    return str(text).replace("|", r"\|")


def _docstrings_after_assignment(tree: ast.Module) -> dict[str, str]:
    """Part II's convention: each constant is followed by a docstring
    literal, e.g. `R_KILL_ENEMY: float = 5.0` then a triple-quoted string.
    `ast` sees this as an Expr/Constant node immediately after the
    assignment.
    """
    out: dict[str, str] = {}
    body = tree.body
    for index, node in enumerate(body):
        name = _assign_name(node)
        if name is None or index + 1 >= len(body):
            continue
        following = body[index + 1]
        if (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ):
            out[name] = _first_paragraph(following.value.value)
    return out


def _comments_before_assignment(tree: ast.Module, source_lines: list[str]) -> dict[str, str]:
    """Part I's convention is DIFFERENT (verified against
    rewards_constants.py): a `#`-prefixed comment block ABOVE each
    constant, not a docstring after. Python's tokenizer discards `#`
    comments before `ast` ever sees them, so this walks raw source lines
    upward from the assignment instead.

    A bare `#` line ends the first paragraph -- REWARD_DEATH's comment runs
    to seventeen lines across two paragraphs and only the first belongs in
    a table cell.
    """
    out: dict[str, str] = {}
    for node in tree.body:
        name = _assign_name(node)
        if name is None:
            continue
        collected: list[str] = []
        line_index = node.lineno - 2  # lineno is 1-based; -2 is the line above
        while line_index >= 0:
            stripped = source_lines[line_index].strip()
            if not stripped.startswith("#"):
                break
            collected.insert(0, stripped.lstrip("#").strip())
            line_index -= 1
        if "" in collected:  # bare `#` -> paragraph break
            collected = collected[: collected.index("")]
        joined = " ".join(part for part in collected if part)
        if joined:
            out[name] = joined
    return out


def _load_constants(name: str, path: pathlib.Path, use_docstrings: bool) -> dict:
    module = _load_module_from_path(name, path)
    values = {key: value for key, value in vars(module).items() if key.isupper()}
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    if use_docstrings:
        docstrings = _docstrings_after_assignment(tree)
    else:
        docstrings = _comments_before_assignment(tree, source.splitlines())
    return {"values": values, "docstrings": docstrings}


def load_part1_constants() -> dict:
    """Load part1_gridworld/config/rewards_constants.py.

    Returns {"values": {NAME: value}, "docstrings": {NAME: justification}}.
    Justifications come from `#` comment blocks ABOVE each constant.
    """
    return _load_constants("rewards_constants_p1", PART1_CONSTANTS_PATH, use_docstrings=False)


def load_part2_constants() -> dict:
    """Load part2_arena/arena/rewards_config.py.

    Same return shape as load_part1_constants, but justifications come from
    docstrings AFTER each constant -- Part II's convention.
    """
    return _load_constants("rewards_config_p2", PART2_CONSTANTS_PATH, use_docstrings=True)


def render_markdown_table(title: str, constants: dict) -> str:
    """Render a {"values", "docstrings"} dict as a Markdown table.

    Row order follows source-definition order (vars() preserves insertion
    order), so the table reads in the same sequence as the constants file.
    """
    lines = [
        f"## {title}",
        "",
        "| Constant | Value | Justification |",
        "| --- | --- | --- |",
    ]
    for name, value in constants["values"].items():
        justification = constants["docstrings"].get(name, "")
        lines.append(f"| `{name}` | `{value}` | {_escape_cell(justification)} |")
    return "\n".join(lines) + "\n"


def _assert_every_constant_is_justified(label: str, constants: dict) -> None:
    """The rubric requires a stated reason for every reward value, shaping
    terms included. Fail loudly rather than emit a table with a blank
    column -- a silently blank Justification cell is exactly what cost
    marks on a previous submission (REWARD_KEY).
    """
    missing = [
        name for name in constants["values"] if not constants["docstrings"].get(name, "").strip()
    ]
    if missing:
        raise ValueError(
            f"{label}: no justification found for {missing}. Part I uses `#` comments "
            f"above each constant, Part II uses a docstring after it -- check the "
            f"convention in the source file matches the extractor being used."
        )


def main() -> None:
    """Load both parts' constants, render tables, write to OUTPUT_PATH."""
    part1 = load_part1_constants()
    part2 = load_part2_constants()

    _assert_every_constant_is_justified("Part I", part1)
    _assert_every_constant_is_justified("Part II", part2)

    output = (
        "<!-- GENERATED FILE -- do not edit by hand.\n"
        "     Regenerate with: python scripts/generate_report_tables.py -->\n\n"
        "# Reward Tables\n\n"
        + render_markdown_table("Part I - Gridworld Reward Constants", part1)
        + "\n"
        + render_markdown_table("Part II - Arena Reward Constants", part2)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
