"""Point 3: Greedy and the learned heuristics with an explicit count of elementary operations.

Counting convention (RAM model, unit cost), identical for every algorithm:
    1  each arithmetic operation +, -, *, /
    1  each floor; an integer division floor(r / a) therefore costs 2
    1  each comparison
    1  each array access: reading a_i or c_i (once per iteration), writing x_i
    2  per iteration for the loop control (test i <= n and increment i + 1),
       plus 1 for the final test that exits the loop
    0  local scalar variables (a, c, r, q, z, ...) and the constants theta_j

The heuristics are written in their cheapest form (clip by two comparisons, terms with the
same denominator grouped), and return exactly the same x as learned() in learned_heuristic.py.
"""
import math


def greedy_counted(inst):
    """Greedy of the lecture. Returns (x, number of operations) = (x, 8n + 1)."""
    n, r, ops = inst.n, inst.b, 0
    x = [0] * n
    for i in range(n):
        ops += 2                      # loop control
        a = inst.a[i]; ops += 1       # read a_i
        q = r // a; ops += 2          # division + floor
        x[i] = q; ops += 1            # write x_i
        r = r - a * q; ops += 2       # multiplication + subtraction
    ops += 1                          # final loop test
    return x, ops


def lecture_counted(inst, theta):
    """clip + lecture state (1, c/a, c/r, a/r, r/b). At most 24n + 2 operations."""
    t0, t1, t2, t3, t4 = theta
    n, r, ops = inst.n, inst.b, 0
    kappa = t4 / inst.b; ops += 1     # computed once, before the loop
    x = [0] * n
    for i in range(n):
        ops += 2                      # loop control
        a = inst.a[i]; ops += 1       # read a_i
        ops += 1                      # test a > r
        if a > r:
            x[i] = 0; ops += 1        # item does not fit
            continue
        c = inst.c[i]; ops += 1       # read c_i
        # z = t0 + t1 c/a + (t2 c + t3 a)/r + kappa r
        z = t0 + t1 * (c / a) + (t2 * c + t3 * a) / r + kappa * r
        ops += 2 + 4 + 1 + 3          # t1*(c/a) | (t2 c + t3 a)/r | kappa r | 3 additions
        ops += 1                      # test z <= 0
        if z <= 0:                    # phi = 1: as Greedy
            q = r // a; ops += 2
            x[i] = q; ops += 1
            r = r - a * q; ops += 2
            continue
        ops += 1                      # test z >= 1
        if z >= 1:                    # phi = 0: skip
            x[i] = 0; ops += 1
            continue
        q = math.floor((r / a) * (1 - z)); ops += 4   # ramp: 1 - z, r/a, *, floor
        x[i] = q; ops += 1
        r = r - a * q; ops += 2
    ops += 1                          # final loop test
    return x, ops


def waste_counted(inst, theta):
    """clip + waste state (1, a/r, w), w = (r - a floor(r/a))/r. At most 22n + 1 operations."""
    t0, t1, t2 = theta
    n, r, ops = inst.n, inst.b, 0
    x = [0] * n
    for i in range(n):
        ops += 2                      # loop control
        a = inst.a[i]; ops += 1       # read a_i
        ops += 1                      # test a > r
        if a > r:
            x[i] = 0; ops += 1
            continue
        q = r // a; ops += 2          # division + floor
        rho = r - a * q; ops += 2     # remainder: the hole left by Greedy
        z = t0 + (t1 * a + t2 * rho) / r; ops += 5
        ops += 1                      # test z <= 0
        if z <= 0:                    # phi = 1: reuse q and rho
            x[i] = q; ops += 1
            r = rho
            continue
        ops += 1                      # test z >= 1
        if z >= 1:
            x[i] = 0; ops += 1
            continue
        q = math.floor((r / a) * (1 - z)); ops += 4
        x[i] = q; ops += 1
        r = r - a * q; ops += 2
    ops += 1
    return x, ops


COUNTED = {"lecture": lecture_counted, "waste": waste_counted}
BOUND = {"greedy": (8, 1), "lecture": (24, 2), "waste": (22, 1)}   # ops <= alpha n + beta