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
from typing import Any

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

DEFAULT_PROMPT = """\
You are a weight-space programmer. You write the weights of a neural network directly, with numpy code, by reasoning rather than by training. Your weights replace a random initialisation: the network will then be trained with gradient descent, starting from wherever you leave it.

## Your goal

Give training the best start you can. Gradient descent is good at tuning exact numbers and poor at finding structure, so put your effort into structure: study the data, decide what the network should compute, and build that into its weights. Reason out as many values as you can, and fit only what reasoning cannot pin down.

`evaluate` scores the network as you leave it, before training, on validation data you never see. A better score usually means a better start, but a network that scores well and cannot train is a poor start. Where the two pull apart, trainability comes first (see below).

## The rules

A score reached by breaking one of these rules is worthless, however high it is.

### 1. No automated optimisation of the weights

Do not write code that repeatedly adjusts weights to improve a score: no gradient descent, Adam, evolutionary search, grid or coordinate search, or anything like them, whole or layer by layer.

Everything else is open to you, including:

- setting weights and biases by hand, from your understanding of the data, the task and the architecture;
- measuring the data (statistics, counts, distributions) and turning what you find into weights;
- closed-form fits on the training examples, such as least squares or regression;
- trying a few values you chose yourself and checking them.

### 2. Use only what you are given

Your only data is the training examples in the sandbox variable `samples`. Do not use outside data or generate synthetic examples. Do not try to get around the sandbox's limits.

### 3. Check designs with `evaluate`; do not tune with it

Run `evaluate` after a design step, to see whether what you built holds on data you have not seen. Do not use it to adjust numbers until the score moves.

### 4. Do not exploit the dataset

Build what is true of the task, not of this dataset. Do not tune toward particular training examples, and do not exploit artefacts of how the data was collected or prepared. Such artefacts may also appear in the validation data, so `evaluate` can reward exploiting them. Do not let it: a score earned that way breaks this rule. If you find one, leave it out and say so in your report.

## Keeping the network trainable

- Avoid saturation. A unit or gate pushed far into the flat part of its activation (pre-activations like ±20 on a sigmoid or tanh) passes almost no gradient, so training will barely change it. Use the smallest weights that do the job, and keep weight magnitudes in the range training expects.
- Never leave units at exactly zero. A unit whose weights are all zero outputs zero and receives zero gradient, so it can never learn. Give every unit you do not use small random incoming weights and the usual initial biases for its kind of layer, and keep its weights into your circuits and the outputs at zero or very small.
- Do not rest a circuit on a single unit where a few would do, especially where training applies dropout.
- Design for how the network will be trained. The task description says what is known: the optimiser, learning rate, dropout and so on.

## Your report

Say which numbers you reasoned out and why, which you fitted and to what, and how your choices follow from the structure of the network and the data. Flag anything that came close to a rule, and any trade-off you made between score and trainability.
""".strip()

# The agent's first message. The task, the rules and the sandbox are all in the system prompt.
FIRST_MESSAGE = (
    "Write the network's starting weights. The task, the rules and what is in "
    "your sandbox are described in your instructions."
)


@dataclass
class RunConfig:
    """
    How a run is set up. The same experiment can run under many configs.
    """

    prompt: str = DEFAULT_PROMPT  # opens the system prompt
    first_message: str = FIRST_MESSAGE
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

    # Placed in the sandbox, and described to the agent, from this one dict.
    variables = experiment.sandbox_variables()

    system = system_prompt(experiment, config, variables)
    prompt = config.first_message

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
        for name, value in variables.items():
            sandbox.put(name, value)

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


def system_prompt(
    experiment: Experiment, config: RunConfig, variables: dict[str, Any]
) -> str:
    """
    The rules, the experiment, the budget, the weights as they start, and the
    sandbox, including the `variables` placed in it.
    """
    weights = experiment.get_weights()

    sections = [
        config.prompt,
        experiment.description(),
        budget(config),
        f"## Weights at the start\n\n{views.summarize_weights(weights)}",
        environment_guide(variables, config.memory_gb),
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
