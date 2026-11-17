"""Regression tests for context-window fuse weight generation.

The weight helpers are pure functions, but their module imports ComfyUI, which
is not available in a plain test environment. Load just the helper under test
from the source AST so the test exercises the real implementation.
"""

from __future__ import annotations

import ast
import pathlib
import unittest


_SOURCE = pathlib.Path(__file__).resolve().parents[1] / "animatediff" / "context.py"


def _load_weights_function(name: str):
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_SOURCE), "exec"), namespace)
    return namespace[name]


class DelayedReverseSawtoothWeightsTest(unittest.TestCase):
    def test_weight_count_matches_context_length(self) -> None:
        make_weights = _load_weights_function("create_weights_delayed_reverse_sawtooth")
        for length in range(1, 16):
            with self.subTest(length=length):
                self.assertEqual(len(make_weights(length)), length)

    def test_odd_length_keeps_the_peak_at_the_middle(self) -> None:
        make_weights = _load_weights_function("create_weights_delayed_reverse_sawtooth")
        weights = make_weights(5)
        self.assertEqual(weights, [0.01, 0.01, 3, 2, 1])


if __name__ == "__main__":
    unittest.main()
