
from __future__ import annotations
import numpy as np

def tournament_selection(scores: np.ndarray, k: int = 2) -> int:
    idx = np.random.randint(0, scores.shape[0], size=k)
    s = scores[idx]
    return idx[np.argmin(s.sum(axis=1))]
