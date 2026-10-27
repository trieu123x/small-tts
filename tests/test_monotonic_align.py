"""The NumPy fallback in the vendored monotonic_align must match core.pyx exactly."""

import ast
from pathlib import Path

import numpy as np
import pytest

INIT = Path(__file__).resolve().parents[1] / "third_party" / "vits_xphonebert" / "monotonic_align" / "__init__.py"


def _load_numpy_impl():
    """Exec only maximum_path_numpy so the test needs neither torch nor the compiled extension."""
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "maximum_path_numpy")
    scope = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(INIT), "exec"), scope)
    return scope["maximum_path_numpy"]


def reference_each(path, value, t_y, t_x, max_neg_val=-1e9):
    """Line-by-line transcription of maximum_path_each in core.pyx."""
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg_val if x == y else value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


@pytest.mark.parametrize("seed", range(5))
def test_numpy_fallback_matches_reference(seed):
    rng = np.random.default_rng(seed)
    b, t_y_max, t_x_max = 3, 40, 12
    values = rng.normal(size=(b, t_y_max, t_x_max)).astype(np.float32)
    t_ys = rng.integers(t_x_max, t_y_max + 1, size=b).astype(np.int32)
    t_xs = rng.integers(2, t_x_max + 1, size=b).astype(np.int32)

    got = np.zeros(values.shape, dtype=np.int32)
    _load_numpy_impl()(got, values.copy(), t_ys, t_xs)

    want = np.zeros(values.shape, dtype=np.int32)
    ref_values = values.copy()
    for i in range(b):
        reference_each(want[i], ref_values[i], int(t_ys[i]), int(t_xs[i]))

    np.testing.assert_array_equal(got, want)
    # A monotonic alignment assigns every frame to exactly one token.
    for i in range(b):
        assert (got[i, : t_ys[i]].sum(axis=1) == 1).all()
