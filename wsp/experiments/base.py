from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

Weights = dict[str, np.ndarray]
Inputs = np.ndarray | dict[str, np.ndarray]


@dataclass(frozen=True)
class Split:
    """
    Examples and their labels. `x` is one array or, for a model with several
    inputs, a dict of arrays that each have one row per example.
    """

    x: Inputs
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


class Experiment(ABC):
    name: str

    # -------------------------------------------------------------------------------- the task

    @abstractmethod
    def description(self) -> str:
        """
        The data, the model, the training process, etc.

        This becomes the body of the system prompt.
        """

    def sandbox_variables(self) -> dict[str, Any]:
        """
        Variables placed in the agent's sandbox at the start of a run. Values
        are arrays, numbers, or dicts of them.

        By default, the whole training split as `samples` (keys: x, y). Add
        more, e.g. a vocabulary, with `{**super().sandbox_variables(), ...}`.
        """
        train = self.train()
        return {"samples": {"x": train.x, "y": train.y}}

    # -------------------------------------------------------------------------------- the data

    @abstractmethod
    def train(self) -> Split:
        """
        The data a model would normally be trained on.
        """

    @abstractmethod
    def validation(self) -> Split:
        """
        Data the agent may evaluate on, within its budget.
        """

    @abstractmethod
    def test(self) -> Split:
        """
        Held-out data. Used only to score the start and end of a run.
        """

    @abstractmethod
    def metrics(self, outputs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """
        Score model outputs. Values may be numbers or numpy arrays.
        """

    # -------------------------------------------------------------------------------- the model

    @abstractmethod
    def get_weights(self) -> Weights:
        """
        Every trainable tensor, keyed by a stable name such as "dense.kernel".
        """

    @abstractmethod
    def set_weights(self, weights: Weights) -> None:
        """
        Load weights with the same names and shapes as `get_weights` returns.
        """

    @abstractmethod
    def predict(self, x: Inputs) -> np.ndarray:
        """
        The model's outputs for a batch of inputs.
        """
