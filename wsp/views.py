"""
LLM-friendly text views of numpy arrays.

Everything here is plain numpy and returns strings. The sandbox's `info()`
prints `summarize_weights`.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np


# -------------------------------------------------------------------------------- tables


def render_table(
    matrix: np.ndarray,
    row_labels: list,
    col_labels: list,
    precision: int = 3,
) -> str:
    """
    Render a 2-D array as right-aligned columns, pandas style.
    """
    cells = [[_format_number(value, precision) for value in row] for row in matrix]

    labels = [str(label) for label in row_labels]
    label_width = max(len(label) for label in labels)

    widths = [
        max(len(str(col_labels[j])), *(len(row[j]) for row in cells))
        for j in range(len(col_labels))
    ]

    header = (
        " " * label_width
        + "  "
        + "  ".join(str(label).rjust(width) for label, width in zip(col_labels, widths))
    )

    lines = [header]
    for label, row in zip(labels, cells):
        values = "  ".join(cell.rjust(width) for cell, width in zip(row, widths))
        lines.append(f"{label.rjust(label_width)}  {values}")

    return "\n".join(lines)


# -------------------------------------------------------------------------------- summaries


def summarize_weights(weights: Mapping[str, np.ndarray]) -> str:
    """
    One line per tensor: name, shape and basic statistics.
    """
    names = list(weights)
    stats = np.array([_statistics(weights[name]) for name in names])
    columns = ["mean", "std", "min", "max", "zeros"]

    shapes = [str(tuple(weights[name].shape)) for name in names]
    labels = [f"{name} {shape}" for name, shape in zip(names, shapes)]

    return render_table(stats, row_labels=labels, col_labels=columns, precision=4)


def render_scores(scores: Mapping[str, Any]) -> str:
    """
    Render a dict of metrics: scalars on one line, arrays as tables.
    """
    blocks = []

    for name, value in scores.items():
        value = np.asarray(value)

        if value.ndim == 0:
            blocks.append(f"{name}: {_format_number(value.item(), 4)}")
        elif value.ndim == 1:
            table = render_table(value.reshape(1, -1), [""], list(range(len(value))), 4)
            blocks.append(f"{name}:\n{table}")
        else:
            table = render_table(
                value, list(range(value.shape[0])), list(range(value.shape[1])), 4
            )
            blocks.append(f"{name}:\n{table}")

    return "\n\n".join(blocks)


# -------------------------------------------------------------------------------- private


def _statistics(array: np.ndarray) -> list[float]:
    """
    Mean, std, min, max and the fraction of exact zeros.
    """
    array = np.asarray(array, dtype=np.float64)
    return [
        array.mean(),
        array.std(),
        array.min(),
        array.max(),
        np.mean(array == 0),
    ]


def _format_number(value: Any, precision: int) -> str:
    """
    Integers as integers. Floats to `precision` places, never as -0.000.
    """
    if isinstance(value, (int, np.integer)):
        return str(value)

    if float(f"{value:.{precision}f}") == 0:
        return f"{0:.{precision}f}"

    return f"{value:.{precision}f}"
