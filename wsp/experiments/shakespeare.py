"""
Shakespeare, one character at a time: can an agent write starting weights for
an LSTM that predicts the next character of a play, so that it trains better
than from a random initialisation?

Every weight starts at zero. The agent's final weights (saved by
examples/shakespeare.py as final.weights.h5) are then trained with the recipe
below, and compared with the same training from Keras's random initialisation.

The data is "tiny Shakespeare" (about 1.1 million characters). The score is
bits per character: how surprised the model is, on average, by the true next
character. Lower is better; guessing uniformly scores log2(65) = 6.02.
"""

import keras
import numpy as np

from .base import Split
from .keras_experiment import KerasExperiment

URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
SEQUENCE_LENGTH = 100

# The network, and how it is trained after the agent finishes. The agent is
# told all of this, and every starting point in a comparison must be trained
# the same way.
EMBEDDING_DIM = 128
UNITS = 512
DROPOUT = 0.3
LEARNING_RATE = 2e-3  # Adam, with Keras's other defaults
CLIPNORM = 1.0
BATCH_SIZE = 64
EPOCHS = 50  # at most; training stops early when the held-out loss stops improving
PATIENCE = 3


class ShakespeareExperiment(KerasExperiment):
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        units: int = UNITS,
        dropout: float = DROPOUT,
    ):
        self.name = "shakespeare"
        self.dropout = dropout

        with open(keras.utils.get_file("shakespeare.txt", URL), encoding="utf-8") as f:
            text = f.read()

        self.vocab = np.array(sorted(set(text)))
        ids = np.searchsorted(self.vocab, list(text))

        # The first 90% of the text for training, as the training script does.
        # The rest is split in half: validation for `evaluate`, and test.
        a, b = int(len(ids) * 0.90), int(len(ids) * 0.95)
        self.splits = {
            "train": _passages(ids[:a]),
            "validation": _passages(ids[a:b]),
            "test": _passages(ids[b:]),
        }

        super().__init__(build_model(len(self.vocab), embedding_dim, units, dropout))
        self.zero_weights()

    # -------------------------------------------------------------------------------- what the agent sees

    def description(self) -> str:
        V = len(self.vocab)
        E = self.model.get_layer("embedding").output_dim
        H = self.model.get_layer("lstm_layer_1").units

        return f"""\
## Data

The complete works of Shakespeare, abridged ("tiny Shakespeare", about 1.1
million characters). Plays are written as a speaker's name, a colon and a new
line, then their lines:

    First Citizen:
    Before we proceed any further, hear me speak.

The text uses {V} characters. Each is given an ID: the sandbox variable `vocab`
holds them in order, so `vocab[i]` is the character with ID i:

    {"".join(self.vocab)!r}

The text is cut into chunks of {SEQUENCE_LENGTH + 1} characters, one after another. Each chunk
gives a passage: `x` is its first {SEQUENCE_LENGTH} characters, as IDs, and `y` its last {SEQUENCE_LENGTH},
so at every position the model must predict the next character: `y[t]` is the
character after `x[t]`. `samples` holds every training passage, in order, so it
covers the first 90% of the text:

```python
"".join(vocab[samples["x"][0]])         # the first passage, as text
chunks = np.concatenate([samples["x"], samples["y"][:, -1:]], axis=1)
text = "".join(vocab[chunks.ravel()])   # the whole training text
char_id = {{c: i for i, c in enumerate(vocab)}}
```

## Model

{self.describe_model()}

## How they connect

The embedding turns each character into {E} numbers: row c of
embedding.embeddings is what character c looks like to the network.

The first LSTM reads those, one character at a time; the second LSTM reads the
first one's h. Each keeps two vectors of {H} numbers, a hidden state h and a
cell state c, both starting at zero for every passage. For a layer with input
x_t at step t:

    z = x_t @ kernel + h_(t-1) @ recurrent_kernel + bias

    i = sigmoid(z[0:{H}])              input gate
    f = sigmoid(z[{H}:{2 * H}])           forget gate
    g = tanh(z[{2 * H}:{3 * H}])            candidate
    o = sigmoid(z[{3 * H}:{4 * H}])          output gate

    c_t = f * c_(t-1) + i * g
    h_t = o * tanh(c_t)

The last dimension of each layer's kernel, recurrent_kernel and bias holds the
four gates in that order, {H} columns each. For example, the input gate of unit
j is column j, and its output gate is column {3 * H} + j.

The output layer reads the second LSTM's h after every character:

    logits = h_t @ output.kernel + output.bias
    p = softmax(logits)

p[k] is the probability that the next character is vocab[k].

Each LSTM is followed by dropout of {self.dropout}. It acts only in training: each number
of h is set to zero with probability {self.dropout}, and the rest are divided by
{1 - self.dropout:g}. `evaluate` and the test score run without it.

## Training

After you finish, the network is trained from your weights:

- Adam with learning rate {LEARNING_RATE} (Keras's other defaults). Each weight's gradient
  is clipped to a norm of at most {CLIPNORM}.
- The loss is the cross-entropy of the next character, from the logits.
- Batches of {BATCH_SIZE} passages, shuffled every epoch.
- Up to {EPOCHS} epochs, stopping once the loss on held-out text has not improved
  for {PATIENCE} epochs. The best weights are kept.

Your weights replace Keras's usual random initialisation, which draws the
embeddings uniformly from ±0.05, the kernels from a Glorot uniform distribution
and the recurrent kernels as random orthogonal matrices, and sets every bias to
zero except the LSTM forget gates, which start at 1.

## Metrics

- bits_per_character: the average of -log2 p[true next character] over every
  position of every passage. Lower is better; uniform guessing scores
  log2({V}) = {np.log2(V):.2f}.
- accuracy: how often the most likely character is the true one.

## Where you start

Every weight and bias is zero, so the model gives every character the same
probability and scores {np.log2(V):.2f} bits per character."""

    def sandbox_variables(self) -> dict:
        return {**super().sandbox_variables(), "vocab": self.vocab}

    # -------------------------------------------------------------------------------- the data

    def train(self) -> Split:
        return self.splits["train"]

    def validation(self) -> Split:
        return self.splits["validation"]

    def test(self) -> Split:
        return self.splits["test"]

    def metrics(self, split: Split) -> dict:
        logits = self.predict(split.x).astype(np.float64)  # type: ignore
        top = logits.max(axis=-1, keepdims=True)
        log_p = logits - top - np.log(np.exp(logits - top).sum(axis=-1, keepdims=True))
        log_p_true = np.take_along_axis(log_p, split.y[..., None], axis=-1)

        return {
            "bits_per_character": float(-log_p_true.mean() / np.log(2)),
            "accuracy": float((logits.argmax(axis=-1) == split.y).mean()),
        }

    # -------------------------------------------------------------------------------- generating text

    def generate(
        self, start: str, length: int = 500, temperature: float = 0.8, seed: int = 0
    ) -> str:
        """
        Continue `start` with `length` characters, each drawn from the
        model's predictions given the last SEQUENCE_LENGTH characters.
        """
        rng = np.random.default_rng(seed)
        ids = list(np.searchsorted(self.vocab, list(start)))

        for _ in range(length):
            window = np.array([ids[-SEQUENCE_LENGTH:]])
            logits = np.asarray(self.model(window, training=False))[0, -1] / temperature
            p = np.exp(logits - logits.max())
            ids.append(rng.choice(len(self.vocab), p=p / p.sum()))

        return "".join(self.vocab[ids])


# -------------------------------------------------------------------------------- building blocks


def build_model(
    characters: int, embedding_dim: int, units: int, dropout: float
) -> keras.Model:
    """
    The network, with Keras's usual random initialisation. Build it again with
    this to train from random weights, or to load the agent's final.weights.h5.
    """
    return keras.Sequential(
        [
            keras.Input(shape=(None,), dtype="int32"),
            keras.layers.Embedding(characters, embedding_dim, name="embedding"),
            keras.layers.LSTM(units, return_sequences=True, name="lstm_layer_1"),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(units, return_sequences=True, name="lstm_layer_2"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(characters, name="output"),
        ]
    )


def _passages(ids: np.ndarray) -> Split:
    """
    Cut the text into chunks of SEQUENCE_LENGTH + 1 characters, one after
    another, as the training script does. Each chunk gives a passage (all but
    its last character) and its targets (all but its first).
    """
    count = len(ids) // (SEQUENCE_LENGTH + 1)
    chunks = ids[: count * (SEQUENCE_LENGTH + 1)].reshape(count, SEQUENCE_LENGTH + 1)
    return Split(chunks[:, :-1], chunks[:, 1:])
