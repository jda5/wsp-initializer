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

    def take(self, rows: Any) -> Split:
        """
        The examples at `rows`, as a smaller Split.
        """
        if isinstance(self.x, dict):
            x = {name: array[rows] for name, array in self.x.items()}
        else:
            x = self.x[rows]

        return Split(x, self.y[rows])


class Experiment(ABC):
    name: str

    # -------------------------------------------------------------------------------- the task

    @abstractmethod
    def description(self) -> str:
        """
        The data, the model, the training process, etc.

        This becomes the body of the system prompt.
        """

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

    @abstractmethod
    def activations(self, x: Inputs) -> dict[str, np.ndarray]:
        """
        Every layer's output for a batch of inputs, keyed by layer name.
        """
