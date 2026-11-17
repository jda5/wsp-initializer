"""
The standard tools: python, evaluate and finish.

Each function builds a Tool. Pick the ones an experiment needs, and set
their budgets there.

Each data split has one job:

    training     in the sandbox from the start, as `samples`
    validation   scores only, through `evaluate`; the agent never sees an example
    test         scored once, at the end of the run
"""

from wsp import views
from wsp.sandbox import SandboxError
from wsp.session import Session
from wsp.tools import Budget, Tool, ToolError


def standard_tools(evaluations: int = 10) -> list[Tool]:
    return [
        python_tool(),
        evaluate_tool(evaluations),
        finish_tool(),
    ]


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
            "available: `np`, and `info()`, which prints every tensor's shape and "
            "statistics. The section 'Your environment' in your instructions says "
            "what else this environment allows."
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
            "used the training data in `samples`, and whether anything you did came "
            "close to breaking a rule."
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
