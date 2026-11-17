"""
The system prompt section that describes the agent's sandbox, including the
variables placed in it, so the description always matches what is there.
"""

from typing import Any

import numpy as np


def environment_guide(variables: dict[str, Any], memory_gb: float) -> str:
    """
    `variables` are the experiment's sandbox variables, as placed in the sandbox.
    """
    listed = "\n".join(f"- `{name}`: {_describe(value)}" for name, value in variables.items())

    return f"""\
## Your environment

The `python` tool runs your code in one Python namespace that is kept between
calls, like a notebook. It starts with:

- `W`: the network's weights, a dict of tensor name -> numpy array, as listed
  under "Weights at the start". Changing the arrays in `W` is the only way to
  change the network.
{listed}
- `np`: numpy.
- `info()`: prints every tensor in `W` with its shape and statistics.

This is full Python 3.14 with numpy: loops, functions, comprehensions and the
standard library all work. There is no network, and no machine learning
libraries are installed. Files you save in /workspace are kept for the rest
of the run. /inbox and /harness belong to the harness; there is nothing in
them for you.

The sandbox has {memory_gb:g} GB of memory. A cell that needs more stops the
sandbox, which restarts with the weights of your last commit and the variables
listed above, and loses every variable you created. So size large arrays before
you make them, prefer float32, work in chunks, and save code and results you
would need again in /workspace.

Being able to write a loop does not change the rules above.

Variables you create are kept between calls, so you can build an array in
one call and assign it in the next."""


def _describe(value: Any) -> str:
    """
    e.g. "a dict with `x`: an array of int64, shape (9939, 100); `y`: ..."
    """
    if isinstance(value, dict):
        parts = "; ".join(f"`{key}`: {_describe(item)}" for key, item in value.items())
        return f"a dict with {parts}"

    array = np.asarray(value)
    if array.ndim == 0:
        return f"the number {array.item()}"

    kind = "strings" if array.dtype.kind == "U" else str(array.dtype)
    return f"an array of {kind}, shape {array.shape}"
