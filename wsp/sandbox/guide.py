"""
The system prompt section that describes the agent's sandbox.
"""


def environment_guide(memory_gb: float) -> str:
    return f"""\
## Your environment

This is full Python 3.14 with numpy: loops, functions, comprehensions and the
standard library all work. There is no network, and no machine learning
libraries are installed. Files you save in /workspace are kept for the rest
of the run.

The sandbox has {memory_gb:g} GB of memory. A cell that needs more stops the
sandbox, which restarts from your last commit and loses every variable you
created. So size large arrays before you make them, prefer float32, work in
chunks, and save code and results you would need again in /workspace.

Being able to write a loop does not change the rules above.

Variables you create are kept between calls, so you can build an array in
one call and assign it in the next."""
