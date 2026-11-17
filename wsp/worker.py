"""
The sandbox worker. It runs inside the Docker container, keeps the agent's
variables alive between cells, and answers one request per line of stdin:

    {"command": "run", "code": "..."}           -> {"output": "...", "changed": {name: count}}
    {"command": "put", "name": "W", "data": ...}  place a dict of arrays in the namespace
    {"command": "get_weights"}                  -> {"data": ...}

Arrays travel as base64-encoded .npz files (see `encode` and `decode`).

This file needs only numpy, so the image holds nothing else. The harness
imports `encode`, `decode`, `count_changes` and `summarize` from it too.
"""

import ast
import base64
import io
import json
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

Weights = dict[str, np.ndarray]

# Longer cell output is cut here, before it is sent; the harness cuts it further.
MAX_OUTPUT_CHARS = 1_000_000

# Short printing: large arrays show their corners with "...".
# legacy="1.25" prints scalars as 0.5 rather than np.float32(0.5).
PRINT_OPTIONS = {
    "precision": 3,
    "suppress": True,
    "threshold": 1000,
    "edgeitems": 3,
    "linewidth": 250,
    "legacy": "1.25",
}


def main() -> None:
    requests, replies = _protocol_streams()
    namespace = {"np": np, "W": {}}
    namespace["info"] = lambda: print(summarize(namespace["W"]))

    while True:
        try:
            line = requests.readline()
        except KeyboardInterrupt:
            continue  # an interrupt that arrived between cells

        if not line:
            return  # the harness has gone

        try:
            reply = {"ok": True, "result": _handle(namespace, json.loads(line))}
        except (Exception, KeyboardInterrupt) as error:
            reply = {"ok": False, "error": f"{type(error).__name__}: {error}"}

        replies.write(json.dumps(reply) + "\n")
        replies.flush()


def _handle(namespace: dict, request: dict) -> object:
    match request["command"]:
        case "run":
            before = _copy(namespace["W"])
            output = run_cell(request["code"], namespace)
            return {
                "output": output[:MAX_OUTPUT_CHARS],
                "changed": count_changes(before, namespace["W"]),
            }

        case "put":
            namespace[request["name"]] = decode(request["data"])
            return None

        case "get_weights":
            return {"data": encode(namespace["W"])}

    raise ValueError(f"Unknown command: {request['command']}")


# -------------------------------------------------------------------------------- running code


def run_cell(code: str, namespace: dict) -> str:
    """
    Run a cell, notebook style: print the value of the last expression, and
    print errors (showing only the agent's own code) instead of raising them.
    """
    buffer = io.StringIO()

    with (
        np.printoptions(**PRINT_OPTIONS),
        redirect_stdout(buffer),
        redirect_stderr(buffer),
    ):
        try:
            tree = ast.parse(code)

            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)

            exec(compile(tree, "<cell>", "exec"), namespace)

            if last is not None:
                value = eval(compile(last, "<cell>", "eval"), namespace)
                if value is not None:
                    print(repr(value))

        except (Exception, KeyboardInterrupt) as error:
            frames = [
                f
                for f in traceback.extract_tb(error.__traceback__)
                if f.filename == "<cell>"
            ]
            print(
                "".join(
                    traceback.format_list(frames)
                    + traceback.format_exception_only(error)
                )
            )

    return buffer.getvalue()


def _protocol_streams():
    """
    Keep private copies of stdin and stdout for the protocol, then point the
    originals elsewhere, so agent code that reads stdin or writes straight to
    file descriptor 1 can never mix with the messages.
    """
    requests = os.fdopen(os.dup(0), "r", encoding="utf-8")
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")

    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(2, 1)

    return requests, replies


# -------------------------------------------------------------------------------- shared with the harness


def encode(arrays: Weights) -> str:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode(data: str) -> Weights:
    # allow_pickle=False: data written by agent code can never run code on the host.
    with np.load(io.BytesIO(base64.b64decode(data)), allow_pickle=False) as file:
        return {name: file[name] for name in file.files}


def count_changes(before: Weights, after: object) -> dict[str, int]:
    """
    How many values differ in each tensor. Unchanged tensors are left out.
    """
    if not isinstance(after, dict):
        return {}

    changed = {}
    for name, array in after.items():
        old, new = before.get(name), np.asarray(array)

        if old is None or old.shape != new.shape:
            changed[name] = int(new.size)
        elif (count := int(np.sum(old != new))) > 0:
            changed[name] = count

    return changed


def summarize(weights: Weights) -> str:
    """
    One line per tensor: its name, shape and statistics.
    """
    lines = [f"{'name':30} {'shape':16} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}"]

    for name, array in weights.items():
        a = np.asarray(array, dtype=np.float64)
        lines.append(
            f"{name:30} {str(a.shape):16} "
            f"{a.mean():9.4f} {a.std():9.4f} {a.min():9.4f} {a.max():9.4f}"
        )

    return "\n".join(lines)


def _copy(weights: object) -> Weights:
    if not isinstance(weights, dict):
        return {}  # the agent replaced W with something else; checks on the host catch it
    return {name: np.array(array, copy=True) for name, array in weights.items()}


if __name__ == "__main__":
    main()
