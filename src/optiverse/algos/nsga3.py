# src/optiverse/algos/nsga3.py
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

# -------------------------- utility: non-dominated sorting --------------------------
def _fast_non_dominated_sort(F: np.ndarray) -> List[np.ndarray]:
    """
    Return list of fronts; each front is an array of indices.
    Minimization assumed.
    """
    n = F.shape[0]
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for i in range(n):
        Fi = F[i]
        for j in range(i + 1, n):
            Fj = F[j]
            # dominance compare
            if np.all(Fi <= Fj) and np.any(Fi < Fj):
                S[i].append(j)
                n_dom[j] += 1
            elif np.all(Fj <= Fi) and np.any(Fj < Fi):
                S[j].append(i)
                n_dom[i] += 1

    fronts[0] = [i for i in range(n) if n_dom[i] == 0]
    k = 0
    while fronts[k]:
        next_front: List[int] = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    fronts.pop()  # remove last empty
    return [np.array(fr, dtype=int) for fr in fronts]


# -------------------------- operators: SBX & polynomial mutation --------------------
def _sbx_crossover(p1: np.ndarray, p2: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                   eta_c: float = 30.0, p_c: float = 1.0, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    n = p1.size
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() <= p_c:
        u = rng.random(n)
        beta = np.empty(n, dtype=float)
        mask = u <= 0.5
        beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta_c + 1.0))
        beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta_c + 1.0))
        child1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        child2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
        c1 = np.clip(child1, lb, ub)
        c2 = np.clip(child2, lb, ub)
    return c1, c2


def _poly_mutation(x: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                   eta_m: float = 20.0, p_m: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = x.size
    if p_m is None:
        p_m = 1.0 / n
    y = x.copy()
    r = rng.random(n)
    mutate = r <= p_m
    if np.any(mutate):
        u = rng.random(np.sum(mutate))
        delta = np.empty_like(u)
        mask = u < 0.5
        delta[mask] = (2.0 * u[mask]) ** (1.0 / (1.0 + eta_m)) - 1.0
        delta[~mask] = 1.0 - (2.0 * (1.0 - u[~mask])) ** (1.0 / (1.0 + eta_m))
        span = (ub[mutate] - lb[mutate])
        y[mutate] = np.clip(y[mutate] + delta * span, lb[mutate], ub[mutate])
    return y


# -------------------------- reference directions (Das–Dennis) ----------------------
def _das_dennis(M: int, p: int) -> np.ndarray:
    """
    Simplex-lattice reference points; each row sums to 1.0 (non-negative).
    Returns array of shape (R, M).
    """
    # generate all compositions of p into M non-negative integers
    parts = []

    def rec(k, remaining, acc):
        if k == M - 1:
            parts.append(acc + [remaining])
            return
        for v in range(remaining + 1):
            rec(k + 1, remaining - v, acc + [v])

    rec(0, p, [])
    P = np.array(parts, dtype=float) / float(p)
    return P  # shape (R, M)


# -------------------------- NSGA-III core ------------------------------------------
class NSGA3:
    """
    Minimal NSGA-III with SBX + polynomial mutation.
    - Non-dominated sorting for ranking (as NSGA-II).
    - Environmental selection on splitting front via reference-direction niching:
        * normalize objectives with min-max (robust fallback),
        * associate each point to closest reference dir by perpendicular distance,
        * fill under-represented niches first (prefer min distance if niche empty).
    This is faithful to Deb & Jain (2014) while using a practical min-max normalization,
    which is a common engineering simplification when hyperplane intercepts are unstable.
    """

    def __init__(self,
                 population_size: int,
                 n_obj: int,
                 ref_dirs: Optional[np.ndarray] = None,
                 ref_parts: int = 12,
                 eta_c: float = 30.0,
                 eta_m: float = 20.0,
                 p_c: float = 1.0,
                 p_m: Optional[float] = None,
                 seed: Optional[int] = None):
        self.n = int(population_size)
        self.n_obj = int(n_obj)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.p_c = float(p_c)
        self.p_m = p_m
        self.rng = np.random.default_rng(seed)

        if ref_dirs is None:
            self.ref_dirs = _das_dennis(self.n_obj, int(ref_parts))
        else:
            ref_dirs = np.asarray(ref_dirs, dtype=float)
            if ref_dirs.ndim != 2 or ref_dirs.shape[1] != self.n_obj:
                raise ValueError(f"ref_dirs must have shape (R, {self.n_obj})")
            # normalize rows to sum to 1 and be non-zero
            row_sum = ref_dirs.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            self.ref_dirs = ref_dirs / row_sum

        # unit-length direction vectors for perpendicular distance
        norms = np.linalg.norm(self.ref_dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.ref_unit = self.ref_dirs / norms

    # ---------- GA outer API ----------
    def initialize(self, n_var: int, bounds: List[Tuple[float, float]]) -> np.ndarray:
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
        pop = self.rng.random((self.n, n_var)) * (ub - lb) + lb
        self._bounds = (lb, ub)
        return pop

    def ask(self, pop: np.ndarray, fit: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        # Variation: binary tournament by rank (then random as tie-break).
        n, d = pop.shape

        # >>> read bounds from the call, not from self._bounds <<<
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)

        if not hasattr(self, "_rank"):
            # first ask after initialize/resume: make random mating
            parents_idx = self.rng.integers(0, n, size=(n,))
        else:
            inv = 1.0 / (1.0 + self._rank[:n])   # safe if _rank is longer from R=2N
            prob = inv / inv.sum()
            parents_idx = self.rng.choice(n, size=(n,), p=prob)

        children = []
        for i in range(0, n, 2):
            i1 = parents_idx[i % n]
            i2 = parents_idx[(i + 1) % n]
            c1, c2 = _sbx_crossover(pop[i1], pop[i2], lb, ub, self.eta_c, self.p_c, self.rng)
            c1 = _poly_mutation(c1, lb, ub, self.eta_m, self.p_m, self.rng)
            c2 = _poly_mutation(c2, lb, ub, self.eta_m, self.p_m, self.rng)
            children.append(c1)
            children.append(c2)
        return np.vstack(children)[:n]

    def tell(self, pop: np.ndarray, fit: np.ndarray,
             new_pop: np.ndarray, new_fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Elitist survival from R = parents ∪ offspring (2N)
        P = np.vstack([pop, new_pop])
        F = np.vstack([fit, new_fit])

        fronts = _fast_non_dominated_sort(F)
        next_idx: List[int] = []
        for fr in fronts:
            if len(next_idx) + len(fr) <= self.n:
                next_idx.extend(fr.tolist())
            else:
                # NSGA-III niching on the splitting front
                needed = self.n - len(next_idx)
                chosen = self._niching_select(F, fronts, next_idx, fr, needed)
                next_idx.extend(chosen.tolist())
                break

        next_idx = np.array(next_idx, dtype=int)
        self._rank = self._rank_from_fronts(fronts, F.shape[0])
        return P[next_idx], F[next_idx]

    # ---------- helpers ----------
    def _rank_from_fronts(self, fronts: List[np.ndarray], n: int) -> np.ndarray:
        rank = np.full(n, fill_value=1_000_000, dtype=int)
        for r, fr in enumerate(fronts, start=0):
            rank[fr] = r
        return rank

    def _niching_select(self, F_all: np.ndarray, fronts: List[np.ndarray],
                        selected_idx: List[int], split_front: np.ndarray, k: int) -> np.ndarray:
        """
        Select k indices from split_front using reference-direction niching (NSGA-III).
        Steps:
          1) Normalize objectives (min-max over all candidates in R for robustness).
          2) Associate already-selected points and split_front points to ref dirs by ⟂ distance.
          3) Repeatedly pick the niche with minimum count and take the closest candidate.
        """
        # 1) normalization (min-max fallback)
        zmin = F_all.min(axis=0)
        zmax = F_all.max(axis=0)
        span = np.where((zmax - zmin) > 1e-12, (zmax - zmin), 1.0)
        F_norm = (F_all - zmin) / span

        # 2) association: for each point, closest ref dir by perpendicular distance
        def assoc_idx_and_dist(Fn: np.ndarray):
            # project onto each unit ref to compute perpendicular distance
            # dist = || y - (y⋅r) r ||, r has unit norm
            # expand: shape (R, M) against (K, M)
            y = Fn  # (K, M)
            dot = y @ self.ref_unit.T                      # (K, R)
            proj = dot[..., None] * self.ref_unit[None]    # (K, R, M)
            perp = np.linalg.norm(y[:, None, :] - proj, axis=2)  # (K, R)
            j = np.argmin(perp, axis=1)
            d = perp[np.arange(perp.shape[0]), j]
            return j, d

        R = F_all.shape[0]
        all_idx = np.arange(R)
        selected_idx_arr = np.array(selected_idx, dtype=int)

        # precompute association for selected and for split front
        if selected_idx_arr.size > 0:
            sel_dirs, _ = assoc_idx_and_dist(F_norm[selected_idx_arr])
        else:
            sel_dirs = np.array([], dtype=int)
        split_dirs, split_dist = assoc_idx_and_dist(F_norm[split_front])

        # niche counts from already-selected
        niche_count = np.zeros(self.ref_unit.shape[0], dtype=int)
        if sel_dirs.size > 0:
            for j in sel_dirs:
                niche_count[j] += 1

        # candidates per niche for the split front
        cand_per_niche = {j: [] for j in range(self.ref_unit.shape[0])}
        for idx_local, (idx_global, j, d) in enumerate(zip(split_front, split_dirs, split_dist)):
            cand_per_niche[j].append((idx_global, d))

        # iteratively pick k
        chosen: List[int] = []
        for _ in range(k):
            # eligible niches (have candidates)
            eligible = [j for j, lst in cand_per_niche.items() if len(lst) > 0]
            if not eligible:
                # no candidates left due to numerical ties; fall back: random from remaining split_front
                remaining = [i for i in split_front if i not in chosen]
                self.rng.shuffle(remaining)
                chosen.extend(remaining[:(k - len(chosen))])
                break

            # min niche count among eligible
            counts = np.array([niche_count[j] for j in eligible])
            minc = counts.min()
            J = [j for j in eligible if niche_count[j] == minc]
            # pick one niche at random among minima
            j_pick = self.rng.choice(J)

            # pick candidate: if niche empty (count==0), choose smallest perp distance; else random
            lst = cand_per_niche[j_pick]
            if niche_count[j_pick] == 0:
                # choose with smallest distance
                lst.sort(key=lambda t: t[1])
                idx_global, _ = lst.pop(0)
            else:
                # random selection from that niche
                r = self.rng.integers(0, len(lst))
                idx_global, _ = lst.pop(r)

            niche_count[j_pick] += 1
            chosen.append(idx_global)

        return np.array(chosen, dtype=int)