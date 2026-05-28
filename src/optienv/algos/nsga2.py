
from __future__ import annotations
import numpy as np
from .operators.selection import tournament_selection
from .operators.crossover import sbx
from .operators.mutation import polynomial_mutation


def _crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n == 0: return d
    for j in range(m):
        order = np.argsort(F[:, j])
        d[order[0]] = d[order[-1]] = np.inf
        span = F[order[-1], j] - F[order[0], j]
        if span == 0: continue
        for i in range(1, n-1):
            d[order[i]] += (F[order[i+1], j] - F[order[i-1], j]) / span
    return d


def _fast_nd_sort(F: np.ndarray) -> list[np.ndarray]:
    n = F.shape[0]
    S = [set() for _ in range(n)]        # S[p] : set of solutions dominated by p
    n_dom = np.zeros(n, int)             # n_dom[p] : number of solutions that dominate p
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            # p dominates q (minimization)
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].add(q)
                n_dom[q] += 1            # <-- FIX: count dominators for q
            # q dominates p (minimization)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                S[q].add(p)              # <-- mirror S update
                n_dom[p] += 1

    # first front
    fronts[0] = [p for p in range(n) if n_dom[p] == 0]

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    # return non-empty as arrays
    return [np.array(f, dtype=int) for f in fronts if f]

class NSGA2:
    def __init__(self, population_size: int = 100, pc: float = 0.9, pm: float | None = None,
                 eta_c: float = 15.0, eta_m: float = 20.0, seed: int | None = None):
        self.n = population_size; self.pc = pc; self.pm = pm; self.eta_c = eta_c; self.eta_m = eta_m
        if seed is not None: np.random.seed(seed)

    def initialize(self, n_var: int, bounds: list[tuple[float, float]]):
        lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
        return np.random.rand(self.n, n_var)*(ub-lb) + lb

    def ask(self, pop: np.ndarray, fit: np.ndarray, bounds: list[tuple[float, float]]):
        lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
        fronts = _fast_nd_sort(fit)
        rank = np.empty(len(pop), int)
        for r, f in enumerate(fronts): rank[f] = r
        crowd = np.zeros(len(pop));
        for f in fronts: crowd[f] = _crowding_distance(fit[f])
        scores = np.vstack([rank, -crowd]).T
        parents = [(tournament_selection(scores, 2), tournament_selection(scores, 2)) for _ in range(self.n)]
        off = []
        for i, j in parents:
            c1, c2 = sbx(pop[i], pop[j], eta=self.eta_c, p=self.pc, lb=lb, ub=ub)
            c1 = polynomial_mutation(c1, eta=self.eta_m, p=self.pm, lb=lb, ub=ub)
            c2 = polynomial_mutation(c2, eta=self.eta_m, p=self.pm, lb=lb, ub=ub)
            off.append(c1); off.append(c2)
        return np.asarray(off[:self.n])

    def tell(self, pop: np.ndarray, fit: np.ndarray,
            new_pop: np.ndarray, new_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Combine parents and offspring
        P = np.vstack([pop, new_pop])
        F = np.vstack([fit, new_fit])

        fronts = _fast_nd_sort(F)   # list of index arrays, e.g., [F1, F2, ...]
        nxtp, nxtf = [], []
        taken = 0                   # <-- number of survivors already selected

        for fr in fronts:
            fr_size = len(fr)
            if taken + fr_size <= self.n:
                nxtp.append(P[fr])
                nxtf.append(F[fr])
                taken += fr_size
            else:
                k = self.n - taken                  # how many more we can take
                if k > 0:
                    crowd = _crowding_distance(F[fr])        # crowding aligned to fr
                    keep_local = np.argsort(-crowd)[:k]      # top-k by crowding
                    keep_idx = fr[keep_local]                # map back to global idx
                    nxtp.append(P[keep_idx])
                    nxtf.append(F[keep_idx])
                    taken += k
                break

        # Stack and (optionally) assert the invariant
        Pn = np.vstack(nxtp)
        Fn = np.vstack(nxtf)
        # Optionally enforce:
        # assert Pn.shape[0] == self.n, f"Survivors={Pn.shape[0]} but pop_size={self.n}"
        return Pn, Fn
