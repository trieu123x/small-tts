"""Deterministic train/holdout split; the holdout doubles as the VITS validation set."""

import random


def holdout(ids: list[str], n: int = 10, seed: int = 1234) -> tuple[list[str], list[str]]:
    """Return (train_ids, holdout_ids), both sorted. Same input and seed always give the same split."""
    unique = sorted(set(ids))
    if len(unique) != len(ids):
        raise ValueError("duplicate ids")
    if not 0 < n < len(unique):
        raise ValueError(f"holdout size {n} must be between 1 and {len(unique) - 1}")
    picked = set(random.Random(seed).sample(unique, n))
    return sorted(i for i in unique if i not in picked), sorted(picked)
