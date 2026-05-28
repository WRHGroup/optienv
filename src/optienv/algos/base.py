
from __future__ import annotations
from typing import Protocol
import numpy as np

class EvolutionaryAlgorithm(Protocol):
    def initialize(self, n_pop: int, n_var: int, bounds: list[tuple[float, float]]) -> np.ndarray: ...
    def step(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray: ...
