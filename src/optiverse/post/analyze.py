
from __future__ import annotations
import numpy as np
from pathlib import Path
from ..utils.pareto import nondominated_mask


def load_fitness_npy(path: Path) -> np.ndarray:
    return np.load(path)


def pareto_size(F: np.ndarray) -> int:
    return int(nondominated_mask(F).sum())
