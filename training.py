"""Training problem and derivative-free optimisers (point 2).

Training problem:  max_theta psi(theta), where psi aggregates the results of the learned
heuristic over a collection of training instances:
    "raw"   : mean of the values f(x^(j))           (formulation of Lecture 2)
    "ratio" : mean of the ratios f(x^(j)) / f*_j     (every instance weighs the same)
    "min"   : smallest ratio                         (worst case)

All optimisers maximise f(theta) with exactly `budget` evaluations of f, so that they can be
compared under an identical evaluation budget.
"""
import math
import random

from knapsack_family import ukp_family, dp_opt, greedy
from learned_heuristic import learned


# ---------------------------------------------------------------- data
def make_dataset(n_instances, m_range, k_range, seed):
    """Random members F(m, k): k uniform in k_range, then m uniform in m_range (m >= k + 1)."""
    rng = random.Random(seed)
    data = []
    for _ in range(n_instances):
        k = rng.randint(*k_range)
        m = rng.randint(max(m_range[0], k + 1), m_range[1])
        data.append(ukp_family(m, k))
    return data


def ensure_optimum(instances):
    """Fill f_star with the exact DP optimum when no closed form is known."""
    for inst in instances:
        if inst.f_star is None:
            inst.f_star = dp_opt(inst)
    return instances


# ---------------------------------------------------------------- reward
def aggregate(values, optima, kind):
    if kind == "raw":
        return sum(values) / len(values)
    ratios = [v / f for v, f in zip(values, optima)]
    return min(ratios) if kind == "min" else sum(ratios) / len(ratios)


def reward(theta, instances, phi, state, kind="ratio"):
    """psi(theta) for the learned heuristic with configuration (phi, state)."""
    values = [learned(inst, theta, phi, state)[1] for inst in instances]
    return aggregate(values, [inst.f_star for inst in instances], kind)


def greedy_reward(instances, kind="ratio"):
    values = [greedy(inst)[1] for inst in instances]
    return aggregate(values, [inst.f_star for inst in instances], kind)


# ---------------------------------------------------------------- budget bookkeeping
class BudgetExhausted(Exception):
    pass


class Counter:
    """Wraps f: counts evaluations, stops at the budget, records the best-so-far value."""

    def __init__(self, f, budget):
        self.f, self.budget = f, budget
        self.n, self.best, self.best_theta, self.history = 0, -math.inf, None, []

    def __call__(self, theta):
        if self.n >= self.budget:
            raise BudgetExhausted
        v = self.f(theta)
        self.n += 1
        if v > self.best:
            self.best, self.best_theta = v, list(theta)
        self.history.append(self.best)
        return v


def _clip_box(theta, bound):
    return [min(max(t, -bound), bound) for t in theta]


def _random_point(rng, dim, bound):
    return [rng.uniform(-bound, bound) for _ in range(dim)]


# ---------------------------------------------------------------- 1. random search
def random_search(f, dim, budget, seed, bound=1000.0):
    """Baseline: sample theta uniformly in [-bound, bound]^dim and keep the best."""
    rng = random.Random(seed)
    F = Counter(f, budget)
    try:
        while True:
            F(_random_point(rng, dim, bound))
    except BudgetExhausted:
        pass
    return F.best_theta, F.best, F.history


# ---------------------------------------------------------------- 2. simulated annealing
def simulated_annealing(f, dim, budget, seed, bound=1000.0,
                        step0=0.3, step_min=0.001, T0=0.05, T_end=1e-4):
    """Gaussian moves with decreasing size; a worse candidate is accepted with probability
    exp(delta / (T |current value|)), i.e. temperatures are relative, so the same settings
    work for the "raw" and the "ratio" rewards."""
    rng = random.Random(seed)
    F = Counter(f, budget)
    alpha = (T_end / T0) ** (1.0 / max(budget - 1, 1))
    try:
        theta = _random_point(rng, dim, bound)
        val = F(theta)
        T = T0
        while True:
            frac = F.n / budget
            step = bound * (step0 * (1 - frac) + step_min * frac)
            cand = _clip_box([t + rng.gauss(0, step) for t in theta], bound)
            cval = F(cand)
            delta = cval - val
            if delta >= 0 or rng.random() < math.exp(delta / (T * max(abs(val), 1e-12))):
                theta, val = cand, cval
            T *= alpha
    except BudgetExhausted:
        pass
    return F.best_theta, F.best, F.history


# ---------------------------------------------------------------- 3. genetic algorithm
def genetic_algorithm(f, dim, budget, seed, bound=1000.0, pop_size=20,
                      elite=2, tournament=3, p_cross=0.9, mut0=0.3, mut_min=0.001):
    """Tournament selection, uniform crossover, Gaussian mutation of decreasing size, elitism."""
    rng = random.Random(seed)
    F = Counter(f, budget)
    pop, fit = [], []
    try:
        for _ in range(pop_size):
            p = _random_point(rng, dim, bound)
            pop.append(p)
            fit.append(F(p))
        while True:
            frac = F.n / budget
            mut = bound * (mut0 * (1 - frac) + mut_min * frac)
            order = sorted(range(len(pop)), key=lambda i: fit[i], reverse=True)
            new_pop = [pop[i] for i in order[:elite]]
            new_fit = [fit[i] for i in order[:elite]]

            def select():
                idx = rng.sample(range(len(pop)), tournament)
                return pop[max(idx, key=lambda i: fit[i])]

            while len(new_pop) < pop_size:
                p1, p2 = select(), select()
                if rng.random() < p_cross:
                    child = [g1 if rng.random() < 0.5 else g2 for g1, g2 in zip(p1, p2)]
                else:
                    child = list(p1)
                child = _clip_box([g + rng.gauss(0, mut) for g in child], bound)
                new_pop.append(child)
                new_fit.append(F(child))
            pop, fit = new_pop, new_fit
    except BudgetExhausted:
        pass
    return F.best_theta, F.best, F.history


# ---------------------------------------------------------------- 4. Nelder-Mead
def nelder_mead(f, dim, budget, seed, bound=1000.0, init_step=0.1, tol=1e-9,
                alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Nelder-Mead simplex (maximisation) with random restarts until the budget is used.

    A run is restarted from a new random point when the simplex has collapsed
    (all vertices have the same value, e.g. on a plateau of the staircase).
    """
    rng = random.Random(seed)
    F = Counter(f, budget)
    try:
        while True:
            x0 = _random_point(rng, dim, bound)
            simplex = [x0] + [_clip_box([x0[j] + (init_step * bound if j == i else 0.0)
                                         for j in range(dim)], bound) for i in range(dim)]
            vals = [F(x) for x in simplex]
            while max(vals) - min(vals) > tol * max(1.0, abs(max(vals))):
                order = sorted(range(dim + 1), key=lambda i: vals[i], reverse=True)
                simplex = [simplex[i] for i in order]
                vals = [vals[i] for i in order]
                centroid = [sum(x[j] for x in simplex[:-1]) / dim for j in range(dim)]
                worst = simplex[-1]

                def move(coef):  # point centroid + coef * (centroid - worst)
                    return _clip_box([c + coef * (c - w) for c, w in zip(centroid, worst)], bound)

                xr = move(alpha)
                fr = F(xr)
                if fr > vals[0]:                        # better than the best: try to expand
                    xe = move(gamma)
                    fe = F(xe)
                    simplex[-1], vals[-1] = (xe, fe) if fe > fr else (xr, fr)
                elif fr > vals[-2]:                     # better than the second worst: accept
                    simplex[-1], vals[-1] = xr, fr
                else:                                   # contract towards the centroid
                    xc = move(-rho)
                    fc = F(xc)
                    if fc > vals[-1]:
                        simplex[-1], vals[-1] = xc, fc
                    else:                               # shrink towards the best vertex
                        best = simplex[0]
                        for i in range(1, dim + 1):
                            simplex[i] = [b + sigma * (x - b) for b, x in zip(best, simplex[i])]
                            vals[i] = F(simplex[i])
    except BudgetExhausted:
        pass
    return F.best_theta, F.best, F.history


OPTIMISERS = {"RS": random_search, "SA": simulated_annealing,
              "GA": genetic_algorithm, "NM": nelder_mead}


def train(instances, phi, state, dim, method, budget, seed, kind="ratio", **kw):
    """Train theta for the configuration (phi, state) with the given optimiser and reward."""
    f = lambda theta: reward(theta, instances, phi, state, kind)
    return OPTIMISERS[method](f, dim, budget, seed, **kw)