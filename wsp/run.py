"""
`run` gives an experiment to an agent and records everything in a run folder:

    runs/<name>-<timestamp>/
        log.jsonl        every message, tool call and result
        snapshots/       baseline.npz and one .npz per commit
        summary.json     test scores before and after, budgets used
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from wsp import views
from wsp.agents import BaseAgent
from wsp.experiments import Experiment
from wsp.log import RunLog
from wsp.sandbox import DockerSandbox, SandboxError
from wsp.sandbox.guide import environment_guide
from wsp.session import Session
from wsp.tools import Tool, Toolbox
from wsp.tools.standard_tools import standard_tools
from wsp.workspace import Weights, Workspace, count_changes

PROMPT = """\
You are a weight-space programmer. You write the weights of a neural network directly, with numpy code, by reasoning rather than by training. Your weights will be the starting point for training the network with gradient descent, in place of a random initialisation.

Your task is to get the best score on `evaluate` that you can. The best results tend to come from understanding the structure of the task and its data, and building the network to match it. Reason out as many values as you can by hand: choose them, check them and adjust them yourself. Fit only what reasoning cannot pin down.

## The rules

A score reached by breaking a rule is worthless, however high it is.

### 1. No automated optimisation of the weights

Do not write code that repeatedly adjusts weights to improve a score: no gradient descent, Adam, evolutionary search, coordinate search or anything like them, whole or layer by layer.

Everything else is open to you. This includes, and is not limited to:

- setting weights and biases by hand, from your analysis and understanding of the data, the task and the network's architecture;
- measuring the data (statistics, counts, distributions) and turning what you find into weights;
- closed-form fits on the training examples, such as least squares, regression or clustering;
- trying a few values you chose, checking them and adjusting them yourself;
- any other method you judge fit, as long as it is not an automated optimisation loop.

### 2. Use only what you are given

Your only data is the training examples in the sandbox variable `samples`. Do not use outside data or generate synthetic examples. Do not try to get around the sandbox's limits.

### 3. `evaluate` is held out

`evaluate` scores the network on validation data, which you never see. Use it to choose between designs, not to tune toward that data. Its budget is limited.

### 4. Do not exploit the dataset

Build what is true of the task, not of this dataset. Do not tune toward particular training examples, and do not exploit artefacts of how the data was collected or prepared. Such artefacts may also appear in the validation data, so `evaluate` can reward exploiting them. Do not let it: a score earned that way breaks this rule.

### 5. Keep the network trainable

Where trainability and a small gain in score pull apart, trainability comes first.

- Avoid saturation. A unit or gate pushed far into the flat part of its activation (for example, a sigmoid or tanh with pre-activations like ±20) passes almost no gradient, so training will barely change it. Use the smallest weights that do the job.
- Do not leave units at exactly zero. A unit whose weights are all zero outputs zero and receives zero gradient, so it can never learn. Give every unit you do not use small random incoming weights, and use the usual initial biases for its kind of layer. Keep the weights from those units into your circuits and into the outputs at zero or very small, so their noise does not disturb what you built.
- Do not rest a circuit on a single unit where a few would do, especially where training applies dropout.

### 6. Say what you did

In your report, say which numbers you reasoned out and why, which you fitted and to what, and how your choices follow from the structure of the network and the data.
""".strip()


@dataclass
class RunConfig:
    """
    How a run is set up. The same experiment can run under many configs.
    """

    prompt: str
    tools: list[Tool] = field(default_factory=standard_tools)
    max_turns: int = 50
    max_output_chars: int = 25_000  # about 3,000 weights per tool call
    memory_gb: float = 8  # the sandbox's memory limit


def run(
    experiment: Experiment,
    agent: BaseAgent,
    config: RunConfig | None = None,
    root: str | Path = "runs",
) -> Path:
    """
    Run `agent` on `experiment` and return the run folder.
    """
    config = config or RunConfig()

    folder = Path(root) / f"{experiment.name}-{datetime.now():%Y%m%d-%H%M%S}"
    log = RunLog(folder / "log.jsonl")
    print(f"Logging to {log.path}. Watch it with: uv run python -m wsp.watch --follow")

    baseline = experiment.get_weights()
    workspace = Workspace(baseline, folder / "snapshots")

    system = system_prompt(experiment, config)
    prompt = config.prompt

    sandbox = DockerSandbox(
        baseline,
        folder / "sandbox",
        name=f"wsp-{folder.name}",
        memory=f"{config.memory_gb:g}g",
    )

    session = Session(experiment, sandbox, workspace, log)
    toolbox = Toolbox(config.tools, session, config.max_output_chars)

    log.event(
        "start",
        experiment=experiment.name,
        agent=agent.name,
        system=system,
        prompt=prompt,
    )

    try:
        baseline_scores = session.score(experiment.test(), baseline)

        agent.run(system, prompt, toolbox, log, config.max_turns)

        final = _final_weights(session)
        final_scores = session.score(experiment.test(), final)
    finally:
        sandbox.close()

    changes = count_changes(baseline, final)

    summary = {
        "experiment": experiment.name,
        "agent": agent.name,
        "test_before": baseline_scores,
        "test_after": final_scores,
        "values_changed": sum(changes.values()),
        "values_total": sum(array.size for array in baseline.values()),
        "changes_by_tensor": changes,
        "commits": len(workspace.commits),
        "tool_usage": toolbox.usage(),
        "finished": session.finished,  # False if the run hit max_turns or was stopped
        "report": session.report,
    }

    log.event("end", **summary)
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_to_list)
    )

    print(f"\nRun saved to {folder}")

    return folder


def system_prompt(experiment: Experiment, config: RunConfig) -> str:
    """
    The rules, the experiment, the budget, the weights as they start, and the sandbox.
    """
    weights = experiment.get_weights()

    sections = [
        PROMPT,
        experiment.description(),
        budget(config),
        f"## Weights at the start\n\n{views.summarize_weights(weights)}",
        environment_guide(config.memory_gb),
    ]
    return "\n\n".join(section for section in sections if section)


def budget(config: RunConfig) -> str:
    """
    How many turns the agent has, and how far each limited tool goes.
    """
    text = (
        f"## Your budget\n\nYou have {config.max_turns} turns. A turn is one of your "
        "replies, however many tools it calls. Plan your work to fit, and leave a few "
        "turns to finish with your report."
    )

    limits = [
        f"- `{tool.name}`: {tool.limit:,}"
        for tool in config.tools
        if tool.limit is not None
    ]
    if limits:
        text += "\n\nTools with a limited number of calls:\n"
        text += "\n".join(limits)

    return text


# -------------------------------------------------------------------------------- private


def _final_weights(session: Session) -> Weights:
    """
    Commit whatever is in W. If it is invalid, or the sandbox has stopped,
    fall back to the last commit.
    """
    try:
        return session.commit("final").weights
    except (ValueError, SandboxError) as error:  # InvalidWeights is a ValueError
        session.log.event("final_weights_invalid", error=str(error))
        return session.workspace.head


def _to_list(value: np.ndarray):
    return value.tolist()
