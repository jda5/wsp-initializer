"""
Ask an agent to write starting weights for a character-level LSTM that
predicts Shakespeare, starting from all zeros.

    uv run python -m examples.shakespeare
    uv run python -m examples.shakespeare --model claude-opus-5-5 --effort max

After the run, the run folder also holds:

    final.weights.h5   the final weights, to use as the starting point for training
    generated.txt      a passage generated from the final weights

To train from the agent's weights, build the network with
`wsp.experiments.shakespeare.build_model` and call
model.load_weights("final.weights.h5") before model.fit(), using the recipe in
that module (LEARNING_RATE, CLIPNORM, BATCH_SIZE, EPOCHS, PATIENCE).

Follow a run from another terminal with `uv run python -m wsp.watch --follow`.
"""

import argparse

from dotenv import load_dotenv

from wsp.agents import ClaudeAgent
from wsp.experiments.shakespeare import ShakespeareExperiment
from wsp.run import RunConfig, run
from wsp.tools.standard_tools import standard_tools

EFFORTS = ["low", "medium", "high", "xhigh", "max"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", choices=EFFORTS, default="high")
    args = parser.parse_args()

    load_dotenv()

    experiment = ShakespeareExperiment()

    config = RunConfig(
        tools=standard_tools(evaluations=args.evaluations),
        max_turns=args.max_turns,
    )

    agent = ClaudeAgent(model=args.model, effort=args.effort)
    folder = run(experiment, agent, config)

    # The experiment holds the final weights after the run's last test score.
    experiment.model.save_weights(folder / "final.weights.h5")

    passage = experiment.generate("ROMEO:\n", length=400)
    (folder / "generated.txt").write_text(passage)
    print(f"\n{passage}")


if __name__ == "__main__":
    main()
