"""Benchmark family F(m, k) for the Unbounded Knapsack Problem (point 1).

Items are sorted by non-increasing density c_i / a_i.
"""
from dataclasses import dataclass


@dataclass
class UKPInstance:
    a: list            # weights
    c: list            # values
    b: int             # capacity
    f_star: int = None # known optimal value (None if unknown)
    x_star: list = None
    name: str = ""

    @property
    def n(self):
        return len(self.a)

    def value(self, x):
        return sum(ci * xi for ci, xi in zip(self.c, x))

    def is_feasible(self, x):
        return all(xi >= 0 for xi in x) and sum(ai * xi for ai, xi in zip(self.a, x)) <= self.b


def ukp_family(m, k):
    """Instance F(m, k): n = 2k, b = 2m, a_i = m+k-i+1, c_i = 2a_i - 1."""
    assert k >= 2 and m >= k + 1
    n = 2 * k
    a = [m + k - i + 1 for i in range(1, n + 1)]
    c = [2 * ai - 1 for ai in a]
    x_star = [0] * n
    x_star[k - 1] = x_star[k + 1] = 1          # items k and k+2 (1-based)
    return UKPInstance(a, c, 2 * m, 4 * m - 2, x_star, f"F({m},{k})")


def greedy(inst):
    """Density-ordered Greedy of the lecture."""
    x, r = [], inst.b
    for ai in inst.a:
        xi = r // ai
        x.append(xi)
        r -= ai * xi
    return x, inst.value(x)


def dp_opt(inst):
    """Exact UKP optimum by dynamic programming (for validation)."""
    best = [0] * (inst.b + 1)
    for cap in range(1, inst.b + 1):
        best[cap] = max([best[cap - 1]] +
                        [best[cap - ai] + ci for ai, ci in zip(inst.a, inst.c) if ai <= cap])
    return best[inst.b]


if __name__ == "__main__":
    # Validation test: closed form == DP, x* feasible, Greedy value as in Lemma 1
    for k in range(2, 8):
        for m in range(k + 1, 60):
            inst = ukp_family(m, k)
            assert inst.is_feasible(inst.x_star)
            assert inst.value(inst.x_star) == inst.f_star == dp_opt(inst)
            assert greedy(inst)[1] == 2 * m + 2 * k - 1
    print("All tests passed.")

    # Table of the report (n = 10)
    print(f"{'m':>6} {'b':>6} {'greedy':>7} {'opt':>6} {'ratio':>7}")
    for m in [6, 10, 50, 100, 1000]:
        inst = ukp_family(m, 5)
        _, fg = greedy(inst)
        print(f"{m:>6} {inst.b:>6} {fg:>7} {inst.f_star:>6} {fg / inst.f_star:>7.4f}")