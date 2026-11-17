"""
The standard tools: python, samples, evaluate and finish.

Each function builds a Tool. Pick the ones an experiment needs, and set
their budgets there.

Each data split has one job:

    training     `samples` to look at
    validation   scores only, through `evaluate`; the agent never sees an example
    test         scored once, at the end of the run
"""

import numpy as np

from wsp import views
from wsp.sandbox import SandboxError
from wsp.session import Session
from wsp.tools import Budget, Tool, ToolError


def standard_tools(
    evaluations: int = 10,
    samples: int | None = 12,
) -> list[Tool]:
    """
    All four tools. `samples=None` gives the agent no training examples.
    """
    tools = [
        python_tool(),
        evaluate_tool(evaluations),
        finish_tool(),
    ]

    if samples is not None:
        tools.append(samples_tool(samples))

    return tools


# -------------------------------------------------------------------------------- python


def python_tool() -> Tool:
    def handler(session: Session, budget: Budget, code: str) -> str:
        budget.spend()

        try:
            execution = session.sandbox.run(code)
        except SandboxError as error:
            session.sandbox.restart(session.workspace.head)
            session.log.event("sandbox_restarted", reason=str(error))
            raise ToolError(
                f"{error} The sandbox was restarted from your last commit; "
                "variables you created were lost."
            ) from error

        output = execution.output or "(no output)"
        if execution.changed:
            changes = ", ".join(
                f"{name} {count:,}" for name, count in execution.changed.items()
            )
            output += f"\n\n[Values changed: {changes}]"

        return output

    return Tool(
        name="python",
        description=(
            "Run Python code to view or edit the weights. `W` is a dict of name -> "
            "numpy array; edit it by index, e.g. `W[name][3] += 0.5`. Variables are "
            "kept between calls, and the value of the last line is printed. Also "
            "available: `np`, `show()` and `info()`. The section 'Working with the "
            "weights' in your instructions shows how, with examples, and says what "
            "else this environment allows."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to run."}
            },
            "required": ["code"],
        },
        handler=handler,
    )


# -------------------------------------------------------------------------------- evaluate


def evaluate_tool(limit: int = 10) -> Tool:
    def handler(session: Session, budget: Budget) -> str:
        session.current_weights()  # fail on invalid weights before spending budget
        budget.spend()

        commit = session.commit(f"evaluation {budget.used}")
        scores = session.score(session.experiment.validation())
        session.log.event("evaluate", commit=commit.number, scores=scores)

        return (
            f"Committed as #{commit.number} and evaluated on {len(session.experiment.validation()):,} "
            f"validation examples.\n\n{views.render_scores(scores)}\n\nEvaluations: {budget}."
        )

    return Tool(
        name="evaluate",
        description=(
            "Commit the current weights and score them on the validation split, which "
            "you never see directly: this is your measure of success. "
            f"You have {limit} evaluations in total, so make each one count."
        ),
        parameters={"type": "object", "properties": {}},
        handler=handler,
        limit=limit,
    )


# -------------------------------------------------------------------------------- samples

# Samples are drawn in one fixed random order, so every run sees the same ones.
SAMPLES_SEED = 0


def samples_tool(limit: int = 12) -> Tool:
    def handler(session: Session, budget: Budget, count: int) -> str:
        if count < 1:
            raise ToolError("Ask for at least one example.")

        budget.spend(count)

        split = session.experiment.train()
        order = np.random.default_rng(SAMPLES_SEED).permutation(len(split))

        new = order[budget.used - count : budget.used]
        drawn = order[: budget.used]  # every sample so far, oldest first

        examples = split.take(drawn)

        session.sample_indices = drawn.tolist()
        session.sandbox.put(
            "samples", {"indices": drawn, "x": examples.x, "y": examples.y}
        )
        session.log.event("samples", indices=new)

        # List the labels only when there is one per example, not one per step.
        labels = split.y[new]
        with_labels = f" with labels {labels.tolist()}" if labels.ndim == 1 else ""

        return (
            f"Drew {count} new training examples{with_labels}.\n\n"
            f"All {len(drawn)} drawn so far are in the sandbox variable `samples` "
            f"(keys: indices, x, y); {_describe_inputs(examples.x)}.\n\n"
            f"Samples: {budget}."
        )

    return Tool(
        name="samples",
        description=(
            "Draw new examples from the training data, with their labels, to look at "
            "and analyse. They are added to the sandbox variable `samples` (keys: "
            "indices, x, y). Each example counts against a budget of "
            f"{limit} in total."
        ),
        parameters={
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "How many new examples to draw.",
                }
            },
            "required": ["count"],
        },
        handler=handler,
        limit=limit,
    )


def _describe_inputs(x: np.ndarray | dict[str, np.ndarray]) -> str:
    """
    e.g. "x has shape (12, 784)", or one shape per input when x is a dict.
    """
    if isinstance(x, dict):
        shapes = ", ".join(f"{name} {array.shape}" for name, array in x.items())
        return f"x is a dict of the model's inputs: {shapes}"

    return f"x has shape {x.shape}"


# -------------------------------------------------------------------------------- finish


def finish_tool() -> Tool:
    def handler(session: Session, budget: Budget, report: str) -> str:
        if session.finished:
            raise ToolError("You have already finished.")

        budget.spend()
        session.report = report
        session.log.event("report", report=report)

        return "Finished. Your report is saved, and your final weights are what is in W now."

    return Tool(
        name="finish",
        description=(
            "End the run. Your final weights are whatever is in W when you call this. "
            "Pass your report: what you changed and why, what you observed, how you "
            "used `samples`, and whether anything you did came close to breaking a rule."
        ),
        parameters={
            "type": "object",
            "properties": {
                "report": {"type": "string", "description": "Your final report."}
            },
            "required": ["report"],
        },
        handler=handler,
    )
