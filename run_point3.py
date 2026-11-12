"""Point 3: empirical verification of the bound "at most 3 times the operations of Greedy".

    python run_point3.py

Counts the elementary operations (convention of operation_count.py) of Greedy and of the
learned heuristics on instances of increasing size n, and draws
    results/point3_operations.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knapsack_family import UKPInstance, ukp_family
from operation_count import greedy_counted, lecture_counted, waste_counted

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

THETA_LECTURE = (-1000.0, -17.18, 438.3, -535.8, 864.2)   # selected heuristic (Section 2.4)
THETA_WASTE = (-94.96, 62.13, 969.8)                      # best clip + waste run
THETA_RAMP = (1 - 1e-6, 0.0, 0.0, 0.0, 0.0)              # z just below 1: always on the ramp
SIZES = [11, 21, 51, 101, 201, 501, 1001, 2001]           # n = 2k - 1 for the family
COLORS = {"Greedy": "#8a8a85", "lecture": "#2a78d6", "waste": "#1baf7a", "ramp": "#eb6834"}


def counts(n):
    """Operation counts of every algorithm on instances with n items."""
    k = (n + 1) // 2
    family = ukp_family(10 * k, k)                                   # n = 2k - 1, m = 10k
    large = UKPInstance(family.a, family.c, 100 * sum(family.a))     # same items, all fit
    return {
        "Greedy (family)": greedy_counted(family)[1],
        "lecture (family)": lecture_counted(family, THETA_LECTURE)[1],
        "waste (family)": waste_counted(family, THETA_WASTE)[1],
        "Greedy (worst)": greedy_counted(large)[1],
        "lecture (worst)": lecture_counted(large, THETA_RAMP)[1],
    }


rows = {n: counts(n) for n in SIZES}

print(f"{'n':>6}" + "".join(f"{k:>18}" for k in rows[SIZES[0]]))
for n, r in rows.items():
    print(f"{n:>6}" + "".join(f"{v:>18}" for v in r.values()))

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.8, 3.0))

g = [rows[n]["Greedy (family)"] for n in SIZES]
ax.plot(SIZES, g, color=COLORS["Greedy"], lw=2, label="Greedy: $8n+1$")
ax.plot(SIZES, [3 * v for v in g], color="#3a3a36", lw=1.2, ls="--", zorder=5,
        label="3 × Greedy")
ax.plot(SIZES, [rows[n]["lecture (worst)"] for n in SIZES], color=COLORS["ramp"], lw=2,
        marker="o", ms=3, label="lecture, worst case")
ax.plot(SIZES, [rows[n]["lecture (family)"] for n in SIZES], color=COLORS["lecture"], lw=2,
        marker="o", ms=3, label="lecture, family")
ax.plot(SIZES, [rows[n]["waste (family)"] for n in SIZES], color=COLORS["waste"], lw=2,
        marker="o", ms=3, label="waste, family")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of items $n$")
ax.set_ylabel("elementary operations")
ax.legend(frameon=False, fontsize=7)
ax.grid(color="#e6e6e2", lw=0.6)
ax.set_axisbelow(True)

fig.tight_layout()
path = os.path.join(OUT, "point3_operations.png")
fig.savefig(path, dpi=200)
print("saved", path)

# ratios to Greedy, quoted in the text
for n in (SIZES[0], SIZES[-1]):
    r = rows[n]
    print(f"n = {n}: lecture family {r['lecture (family)'] / r['Greedy (family)']:.2f}, "
          f"waste family {r['waste (family)'] / r['Greedy (family)']:.2f}, "
          f"lecture worst {r['lecture (worst)'] / r['Greedy (worst)']:.3f}")