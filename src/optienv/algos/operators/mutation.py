
from __future__ import annotations
import numpy as np

def polynomial_mutation(x: np.ndarray, eta: float = 20.0, p: float | None = None,
                        lb: np.ndarray | None = None, ub: np.ndarray | None = None) -> np.ndarray:
    d = x.shape[0]
    if p is None: p = 1.0/d
    if lb is None or ub is None:
        lb = np.full(d, -np.inf); ub = np.full(d, np.inf)
    y = x.copy()
    for i in range(d):
        if np.random.rand() < p:
            d1 = (y[i]-lb[i])/(ub[i]-lb[i]+1e-12); d2 = (ub[i]-y[i])/(ub[i]-lb[i]+1e-12)
            r = np.random.rand(); m = 1.0/(eta+1.0)
            if r < 0.5:
                xy = 1.0 - d1; val = 2.0*r + (1-2*r)*(xy**(eta+1)); dq = val**m - 1.0
            else:
                xy = 1.0 - d2; val = 2.0*(1-r) + 2*(r-0.5)*(xy**(eta+1)); dq = 1.0 - val**m
            y[i] = np.clip(y[i] + dq*(ub[i]-lb[i]), lb[i], ub[i])
    return y
