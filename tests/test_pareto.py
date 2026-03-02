
import numpy as np
from optiverse.utils.pareto import nondominated_mask, epsilon_archive

def test_nd_mask():
    F = np.array([[0,1],[1,0],[0.5,0.5],[2,2]])
    assert nondominated_mask(F).sum() == 3

def test_eps_archive():
    F = np.array([[0.01,0.01],[0.02,0.02],[1.0,1.0]])
    idx = epsilon_archive(F, 0.05)
    assert set(idx.tolist()).issubset({0,2})
