"""Point 4, Section 4.5: a failure case of the selected heuristic.

    python run_point4_failure.py

Finds, among the 1000 random instances of Section 4.3, the one where the selected heuristic
has the lowest ratio to the optimum, prints its data, the optimal and Greedy solutions, and the
decision of the heuristic item by item (score z and branch), and writes the first rows of this
trace as a LaTeX table  ->  results/point4_failure_table.tex
"""
import math
import os

from knapsack_family import greedy
from run_point4_unseen import random_dataset, lecture_plain, THETA

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
N_ROWS = 4                                # rows of the LaTeX table (the items that matter)


def optimal_solution(inst):
    """Dynamic programming with backtracking: an optimal x, not only the optimal value."""
    best, arg = [0] * (inst.b + 1), [-1] * (inst.b + 1)
    for cap in range(1, inst.b + 1):
        best[cap] = best[cap - 1]
        for i, (a, c) in enumerate(zip(inst.a, inst.c)):
            if a <= cap and best[cap - a] + c > best[cap]:
                best[cap], arg[cap] = best[cap - a] + c, i
    x, cap = [0] * inst.n, inst.b
    while cap > 0:
        if arg[cap] == -1:                # capacity cap - 1 is as good: leave one unit empty
            cap -= 1
        else:
            x[arg[cap]] += 1
            cap -= inst.a[arg[cap]]
    return x


def trace(inst, theta):
    """Decision of the heuristic for each item: (a, c, r, z, branch, x_i)."""
    t0, t1, t2, t3, t4 = theta
    r, rows = inst.b, []
    for a, c in zip(inst.a, inst.c):
        if a > r:
            rows.append((a, c, r, None, "does not fit", 0))
            continue
        z = t0 + t1 * (c / a) + (t2 * c + t3 * a) / r + t4 * r / inst.b
        if z <= 0:
            q, branch = r // a, "taken ($\\phi=1$)"
        elif z >= 1:
            q, branch = 0, "skipped ($\\phi=0$)"
        else:
            q, branch = math.floor((r / a) * (1 - z)), "ramp"
        rows.append((a, c, r, z, branch, q))
        r -= a * q
    return rows


def main():
    data = random_dataset(1000, seed=2100)
    ratios = [i.value(lecture_plain(i.a, i.c, i.b, THETA)) / i.f_star for i in data]
    j = min(range(len(data)), key=lambda i: ratios[i])
    inst = data[j]
    x_g, v_g = greedy(inst)
    x_opt = optimal_solution(inst)
    rows = trace(inst, THETA)
    x_h = [row[5] for row in rows]

    print(f"instance {j} of the random set: n = {inst.n}, b = {inst.b}")
    print(f"  a = {inst.a}\n  c = {inst.c}")
    print(f"  optimum  {inst.f_star}   x* = {x_opt}")
    print(f"  Greedy   {v_g}   x = {x_g}   ratio {v_g / inst.f_star:.3f}")
    print(f"  learned  {inst.value(x_h)}   x = {x_h}   ratio {ratios[j]:.3f}")
    print("\n   a    c    c/a    r      z   decision")
    for a, c, r, z, branch, q in rows:
        zs = f"{z:7.1f}" if z is not None else "      -"
        print(f"{a:4}{c:5}{c / a:7.2f}{r:5}{zs}   {branch}, x = {q}")

    lines = [r"\begin{tabular}{@{}rrrrrll@{}}", r"\toprule",
             r"$a_i$ & $c_i$ & $c_i/a_i$ & $r$ & $z$ & learned heuristic & Greedy \\",
             r"\midrule"]
    for (a, c, r, z, branch, q), qg in list(zip(rows, x_g))[:N_ROWS]:
        zs = f"${z:.1f}$" if z is not None else "--"
        lines.append(f"${a}$ & ${c}$ & ${c / a:.2f}$ & ${r}$ & {zs} & {branch}, $x_i={q}$ & "
                     f"$x_i={qg}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(OUT, "point4_failure_table.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nsaved", path)


if __name__ == "__main__":
    main()