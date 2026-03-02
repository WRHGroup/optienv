
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable
import numpy as np


def batched_evaluate(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    population: np.ndarray,
    max_workers: int | None = None,
) -> np.ndarray:
    """Evaluate a population in parallel using processes.

    Parameters
    ----------
    evaluate_fn: callable
        Function mapping x (n_var,) to f (n_obj,).
    population: np.ndarray
        (n_pop, n_var) array.
    max_workers: int | None
        Number of processes; None uses ProcessPool default.
    """
    results: list[np.ndarray] = [None] * len(population)  # type: ignore
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(evaluate_fn, x.copy()): i for i, x in enumerate(population)}
        for fut in as_completed(futs):
            idx = futs[fut]
            results[idx] = np.asarray(fut.result(), dtype=float)
    return np.vstack(results)
