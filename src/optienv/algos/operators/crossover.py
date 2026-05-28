
from __future__ import annotations
import numpy as np

def sbx(p1: np.ndarray, p2: np.ndarray, eta: float = 15.0, p: float = 0.9,
        lb: np.ndarray | None = None, ub: np.ndarray | None = None):
    d = p1.shape[0]
    c1, c2 = p1.copy(), p2.copy()
    if np.random.rand() > p:
        return c1, c2
    if lb is None or ub is None:
        lb = np.full(d, -np.inf); ub = np.full(d, np.inf)
    for i in range(d):
        if np.random.rand() <= 0.5 and abs(p1[i]-p2[i]) > 1e-14:
            x1, x2 = sorted([p1[i], p2[i]])
            r = np.random.rand(); a = 2 - (1 + 2*(x1-lb[i])/(x2-x1))**-(eta+1)
            if r <= 1/a: b = (r*a)**(1/(eta+1))
            else: b = (1/(2-r*a))**(1/(eta+1))
            c1[i] = 0.5*((x1+x2) - b*(x2-x1))
            a = 2 - (1 + 2*(ub[i]-x2)/(x2-x1))**-(eta+1)
            if r <= 1/a: b = (r*a)**(1/(eta+1))
            else: b = (1/(2-r*a))**(1/(eta+1))
            c2[i] = 0.5*((x1+x2) + b*(x2-x1))
            c1[i] = np.clip(c1[i], lb[i], ub[i]); c2[i] = np.clip(c2[i], lb[i], ub[i])
    return c1, c2
