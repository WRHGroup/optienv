# src/optiverse/utils/pareto.py
from __future__ import annotations
import numpy as np

__all__ = ["nondominated_mask", "epsilon_archive"]


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of non-dominated points for *minimization*.

    Parameters
    ----------
    F : (n, m) array-like
        Objective matrix (minimized). Row = solution, Col = objective.

    Returns
    -------
    mask : (n,) bool
        True for non-dominated points (Pareto front).
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must be a 2D array")
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # A point j dominates i if it is no worse in all objectives and strictly better in one.
        dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominates_i[i] = False
        mask[dominates_i] = False
    return mask


def epsilon_archive(F: np.ndarray, eps: float | np.ndarray) -> np.ndarray:
    """
    Epsilon-box archive for *minimization* that reduces density by keeping
    at most one point per epsilon-box, preserving input order within boxes.

    Parameters
    ----------
    F : (n, m) array-like
        Objective matrix (minimized).
    eps : float or (m,) array-like
        Epsilon box widths (per objective). Must be > 0.
        If you have normalized objectives in [0,1], a scalar eps in [~0.005, 0.05]
        is typical (choose based on the granularity you want).

    Returns
    -------
    idx_keep : (k,) int
        Indices of points to keep in the *input order*.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must be 2D")
    n, m = F.shape
    eps = np.asarray(eps, dtype=float)
    if eps.ndim == 0:
        if eps <= 0:
            raise ValueError("eps must be > 0")
        eps = np.full(m, float(eps))
    if eps.shape != (m,):
        raise ValueError(f"eps must be scalar or shape ({m},); got {eps.shape}")
    if np.any(eps <= 0):
        raise ValueError("all eps values must be > 0")

    # Map each point to its epsilon-box id
    boxes = np.floor(F / eps).astype(np.int64)

    # Keep first point encountered in each box (stable; preserves order).
    seen: dict[tuple[int, ...], int] = {}
    keep_idx = []
    for i, key in enumerate(map(tuple, boxes)):
        if key not in seen:
            seen[key] = i
            keep_idx.append(i)

    # Optionally prune dominated points among the kept representatives.
    kept = F[keep_idx]
    nd = nondominated_mask(kept)
    return np.asarray(keep_idx, dtype=np.int64)[nd]