import numpy as np
import torch

try:
    from .monotonic_align.core import maximum_path_c
except ImportError:  # tts-amis patch: no compiled extension (e.g. Windows without a C compiler)
    maximum_path_c = None


def maximum_path_numpy(paths, values, t_ys, t_xs, max_neg_val=-1e9):
    """tts-amis patch: NumPy port of core.pyx (same in-place semantics), vectorized along x. Slow; for CPU smoke tests."""
    for i in range(paths.shape[0]):
        path, value, t_y, t_x = paths[i], values[i], int(t_ys[i]), int(t_xs[i])
        for y in range(t_y):
            lo, hi = max(0, t_x + y - t_y), min(t_x, y + 1)
            if lo >= hi:
                continue
            x = np.arange(lo, hi)
            if y == 0:
                v_cur = np.full(len(x), max_neg_val, dtype=np.float32)
                v_prev = np.where(x == 0, 0.0, max_neg_val)
            else:
                v_cur = np.where(x == y, max_neg_val, value[y - 1, x])
                v_prev = np.where(x == 0, max_neg_val, value[y - 1, np.maximum(x - 1, 0)])
            value[y, lo:hi] += np.maximum(v_prev, v_cur).astype(np.float32)
        index = t_x - 1
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
                index = index - 1


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    (maximum_path_c or maximum_path_numpy)(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
