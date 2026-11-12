"""Point 4, Section 4.3: the selected heuristic against Greedy on unseen instances.

    python run_point4_unseen.py

Evaluation sets (none of them was used for training or for selecting theta):
    test      200 instances F(m, k), m in [10, 200], k in {6, 7, 8}, seed 2000
    random    1000 random instances (weights and values in [10, 100]), optimum by DP
plus the family F(m, k) for m from 10 to 10 000 (generalisation in size). The results on
G(m, k, p) (items of F, capacity p m) are only printed, for the discussion of Section 4.6.

Outputs:
    results/point4_unseen.png        left: ratio vs m on F; right: Greedy vs learned on random
    results/point4_unseen_table.tex  ratios, improvement, operations and time per set
"""
import math
import os
import random
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knapsack_family import UKPInstance, ukp_family, greedy, dp_opt
from training import make_dataset
from operation_count import greedy_counted, lecture_counted

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

THETA = (-1000.0, -17.18, 438.3, -535.8, 864.2)          # selected heuristic (Section 2.4)
K_RANGE = (6, 8)                                          # k drawn in {6, 7, 8}
KS = [6, 7, 8]
C_GREEDY, C_LEARNED = "#8a8a85", "#2a78d6"
K_COLORS = {6: "#2a78d6", 7: "#eb6834", 8: "#1baf7a"}


# ---------------------------------------------------------------- instances
def g_family(m, k, p):
    """G(m, k, p): items of F(m, k), capacity p m. Optimum 2pm - p (p units filling the
    knapsack) when m > (p+1)(k-1) and m > (p-1)k."""
    base = ukp_family(m, k)
    return UKPInstance(base.a, base.c, p * m, 2 * p * m - p, None, f"G({m},{k},{p})")


def g_dataset(n_instances, p, seed):
    rng = random.Random(seed)
    data = []
    for _ in range(n_instances):
        k = rng.randint(*K_RANGE)
        m_min = max((p + 1) * (k - 1), (p - 1) * k) + 1
        data.append(g_family(rng.randint(m_min, 200), k, p))
    return data


def random_dataset(n_instances, seed):
    """Random UKP instances: n in [11, 15], weights and values uniform in [10, 100],
    capacity in [100, 400], items sorted by density, optimum by dynamic programming."""
    rng = random.Random(seed)
    data = []
    for _ in range(n_instances):
        items = [(rng.randint(10, 100), rng.randint(10, 100)) for _ in range(rng.randint(11, 15))]
        items.sort(key=lambda ac: ac[1] / ac[0], reverse=True)
        inst = UKPInstance([a for a, _ in items], [c for _, c in items], rng.randint(100, 400))
        inst.f_star = dp_opt(inst)
        data.append(inst)
    return data


# ---------------------------------------------------------------- algorithms
def lecture_plain(a, c, b, theta):
    """Selected heuristic in the form of Section 3.2, without counter (for timing)."""
    t0, t1, t2, t3, t4 = theta
    kappa = t4 / b
    r, x = b, []
    for ai, ci in zip(a, c):
        if ai > r:
            x.append(0)
            continue
        z = t0 + t1 * (ci / ai) + (t2 * ci + t3 * ai) / r + kappa * r
        if z <= 0:
            q = r // ai
        elif z >= 1:
            q = 0
        else:
            q = math.floor((r / ai) * (1 - z))
        x.append(q)
        r -= ai * q
    return x


def greedy_plain(a, c, b):
    """Greedy of the lecture, without counter (for timing)."""
    r, x = b, []
    for ai in a:
        q = r // ai
        x.append(q)
        r -= ai * q
    return x


def time_per_instance(f, instances, repeat=7, number=200):
    """Median over `repeat` runs of the time per instance (each run solves every instance
    `number` times)."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            for inst in instances:
                f(inst)
        times.append((time.perf_counter() - t0) / (number * len(instances)))
    return statistics.median(times)


def evaluate(instances):
    vg = [greedy(i)[1] for i in instances]
    vh = [i.value(lecture_plain(i.a, i.c, i.b, THETA)) for i in instances]
    rg = [v / i.f_star for v, i in zip(vg, instances)]
    rh = [v / i.f_star for v, i in zip(vh, instances)]
    return {
        "mean_g": statistics.mean(rg), "worst_g": min(rg),
        "mean_h": statistics.mean(rh), "worst_h": min(rh),
        "optimal": sum(r == 1 for r in rh),
        "better": sum(h > g for h, g in zip(vh, vg)),
        "equal": sum(h == g for h, g in zip(vh, vg)),
        "worse": sum(h < g for h, g in zip(vh, vg)),
        "improvement": statistics.mean((h - g) / g for h, g in zip(vh, vg)),
        "ops_g": sum(greedy_counted(i)[1] for i in instances),
        "ops_h": sum(lecture_counted(i, THETA)[1] for i in instances),
        "time_g": time_per_instance(lambda i: greedy_plain(i.a, i.c, i.b), instances),
        "time_h": time_per_instance(lambda i: lecture_plain(i.a, i.c, i.b, THETA), instances),
    }


def learned_is_optimal(m, k):
    inst = ukp_family(m, k)
    return inst.value(lecture_plain(inst.a, inst.c, inst.b, THETA)) == inst.f_star


def main():
    SETS = {
        "test F": make_dataset(200, (10, 200), K_RANGE, seed=2000),
        "random": random_dataset(1000, seed=2100),
        "G, p=3": g_dataset(150, 3, seed=2003),          # printed only (Section 4.6)
        "G, p=4": g_dataset(150, 4, seed=2004),
        "G, p=5": g_dataset(150, 5, seed=2005),
    }
    res = {name: evaluate(data) for name, data in SETS.items()}
    shown = ["test F", "random"]                         # sets of the figure and the table

    print(f"{'set':8}{'n':>5}{'Greedy':>15}{'learned':>15}{'opt':>5}{'better':>8}{'equal':>7}"
          f"{'worse':>7}{'improv.':>9}{'ops ratio':>10}{'time ratio':>11}")
    for name, r in res.items():
        print(f"{name:8}{len(SETS[name]):5}{r['mean_g']:9.3f}/{r['worst_g']:.3f}"
              f"{r['mean_h']:9.3f}/{r['worst_h']:.3f}{r['optimal']:5}{r['better']:8}{r['equal']:7}"
              f"{r['worse']:7}{r['improvement']:9.1%}{r['ops_h'] / r['ops_g']:10.2f}"
              f"{r['time_h'] / r['time_g']:11.2f}")

    # ---------------------------------------------------------------- generalisation in size
    M_VALUES = sorted({int(round(10 ** (1 + 3 * j / 60))) for j in range(61)})   # 10 .. 10 000
    curve = {k: [] for k in KS}
    for k in KS:
        for m in M_VALUES:
            if m >= k + 1:
                inst = ukp_family(m, k)
                curve[k].append((m, inst.value(lecture_plain(inst.a, inst.c, inst.b, THETA))
                                 / inst.f_star, greedy(inst)[1] / inst.f_star))
    for k in KS:                                  # exact limit: first m where the optimum is missed
        first_miss = next(m for m in range(k + 1, 10 ** 5) if not learned_is_optimal(m, k))
        print(f"k = {k}: optimal for every m < {first_miss} "
              f"(prediction of Section 2.1: about {657 * k - 777})")

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.3, 1]})

    axl.axvspan(10, 200, color="#e6e6e2", alpha=0.6, lw=0)
    axl.text(14, 0.43, "training\nrange", fontsize=7, color="#5a5a55")
    for k in KS:
        ms = [m for m, _, _ in curve[k]]
        axl.plot(ms, [rg for _, _, rg in curve[k]], color=C_GREEDY, lw=1.2,
                 ls=[":", "--", "-."][k - 6])
        axl.plot(ms, [rh for _, rh, _ in curve[k]], color=K_COLORS[k], lw=1.8, label=f"learned, $k={k}$")
    axl.plot([], [], color=C_GREEDY, lw=1.2, label="Greedy")
    axl.set_xscale("log")
    axl.set_ylim(0.4, 1.03)
    axl.set_xlabel("size $m$ of $\\mathcal{F}(m,k)$")
    axl.set_ylabel("ratio to the optimum")
    axl.legend(frameon=False, fontsize=7, loc="center", bbox_to_anchor=(0.55, 0.62))

    labels = {"test F": "test $\\mathcal{F}$", "random": "random"}
    rand = SETS["random"]
    rg = [greedy(i)[1] / i.f_star for i in rand]
    rh = [i.value(lecture_plain(i.a, i.c, i.b, THETA)) / i.f_star for i in rand]
    worse = [(g, h) for g, h in zip(rg, rh) if h < g]
    axr.plot([0, 1], [0, 1], color="#c3c3be", lw=1, zorder=0)
    axr.scatter(rg, rh, s=12, color=C_LEARNED, alpha=0.25, edgecolor="none",
                label=f"same as Greedy ({sum(h == g for g, h in zip(rg, rh))})")
    axr.scatter([g for g, _ in worse], [h for _, h in worse], s=18, color="#d03b3b",
                edgecolor="none", label=f"worse than Greedy ({len(worse)})")
    g_min, h_min = min(worse, key=lambda gh: gh[1])
    axr.scatter([g_min], [h_min], s=110, facecolor="none", edgecolor="#3a3a36", lw=1)
    axr.annotate("Section 4.5", (g_min, h_min), xytext=(g_min - 0.12, h_min + 0.12),
                 fontsize=7, color="#3a3a36", arrowprops=dict(arrowstyle="-", color="#3a3a36",
                                                             lw=0.8))
    axr.set_xlim(0.76, 1.01)
    axr.set_ylim(0, 1.03)
    axr.set_xlabel("ratio of Greedy")
    axr.set_ylabel("ratio of the learned heuristic")
    axr.set_title("random instances", fontsize=8.5)
    axr.legend(frameon=False, fontsize=7, loc="center left")
    for ax in (axl, axr):
        ax.grid(color="#e6e6e2", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(OUT, "point4_unseen.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)

    # ---------------------------------------------------------------- LaTeX table
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"set & improvement & better / equal / worse & operations & time \\", r"\midrule"]
    for name in shown:
        r = res[name]
        lines.append(f"{labels[name]} & ${r['improvement'] * 100:+.1f}\\%$ & "
                     f"${r['better']}$ / ${r['equal']}$ / ${r['worse']}$ & "
                     f"${r['ops_h'] / r['ops_g']:.2f}\\times$ & ${r['time_h'] / r['time_g']:.2f}\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(OUT, "point4_unseen_table.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("saved", path)


if __name__ == "__main__":
    main()