"""Point 4, Section 4.2: results on the training instances.

    python run_point4_training.py

1. Trains clip + lecture with the four derivative-free methods (10 seeds, budget 2000,
   mean-ratio reward, as in Figure 1) and records the best-so-far training reward after
   each evaluation  ->  results/point4_training.png
2. Prints, for the selected heuristic on its 40 training instances: ratios, improvement
   over Greedy, exact operation counts and wall-clock time per instance.
3. Writes the table of Section 4.2  ->  results/point4_training_table.tex
"""
import math
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knapsack_family import greedy
from training import make_dataset, train, reward
from operation_count import greedy_counted, lecture_counted

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

BUDGET, N_TRAIN, SEEDS = 2000, 40, list(range(1, 11))
K_RANGE, WIDE = (6, 8), (10, 200)
METHODS = ["RS", "SA", "GA", "NM"]
COLORS = {"RS": "#2a78d6", "SA": "#eb6834", "GA": "#1baf7a", "NM": "#eda100"}
THETA = (-1000.0, -17.18, 438.3, -535.8, 864.2)          # selected heuristic (Section 2.4)
SELECTED_SEED = 1                                         # its training set


# ---------------------------------------------------------------- plain implementations
def greedy_plain(a, c, b):
    """Greedy of the lecture, without counter (for timing)."""
    r, x = b, []
    for ai in a:
        q = r // ai
        x.append(q)
        r -= ai * q
    return x


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


def time_per_instance(f, instances, repeat=7, number=200):
    """Median over `repeat` runs of the time per instance, each run solving every instance
    `number` times."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            for inst in instances:
                f(inst)
        times.append((time.perf_counter() - t0) / (number * len(instances)))
    return statistics.median(times)


# ---------------------------------------------------------------- 1. training curves
curves, final, worst_inst, train_time = {}, {}, {}, {}
for method in METHODS:
    curves[method], final[method], worst_inst[method], train_time[method] = [], [], [], []
    for seed in SEEDS:
        tr = make_dataset(N_TRAIN, WIDE, K_RANGE, seed=seed)
        t0 = time.perf_counter()
        theta, best, hist = train(tr, "clip", "lecture", 5, method, BUDGET, seed)
        train_time[method].append(time.perf_counter() - t0)
        curves[method].append(hist)
        final[method].append(best)
        worst_inst[method].append(reward(theta, tr, "clip", "lecture", "min"))
        if method == "SA" and seed == SELECTED_SEED:
            print("selected run reproduced:", [round(t, 2) for t in theta])

print("\nmethod  seeds  mean final  worst final  mean worst instance  time per run (s)")
for m in METHODS:
    print(f"{m:6}{len(SEEDS):7}{statistics.mean(final[m]):12.4f}{min(final[m]):13.4f}"
          f"{statistics.mean(worst_inst[m]):21.4f}{statistics.median(train_time[m]):18.2f}")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(3.8, 3.4), sharex=True, sharey=True)
evals = range(1, BUDGET + 1)
for ax, m in zip(axes.flat, METHODS):
    for h in curves[m]:                                   # one thin line per seed
        ax.plot(evals, h, color=COLORS[m], lw=0.8, alpha=0.45)
    mean = [statistics.mean(h[i] for h in curves[m]) for i in range(BUDGET)]
    ax.plot(evals, mean, color="#3a3a36", lw=1.6)        # mean over the seeds
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(m, fontsize=9, pad=2)
    ax.grid(color="#e6e6e2", lw=0.6)
    ax.set_axisbelow(True)
for ax in axes[1]:
    ax.set_xlabel("evaluations of $\\psi$")
for ax in axes[:, 0]:
    ax.set_ylabel("best training ratio")
fig.tight_layout()
path = os.path.join(OUT, "point4_training.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print("\nsaved", path)

# ---------------------------------------------------------------- 2. selected heuristic
tr = make_dataset(N_TRAIN, WIDE, K_RANGE, seed=SELECTED_SEED)
val_g = [greedy(i)[1] for i in tr]
val_h = [i.value(lecture_plain(i.a, i.c, i.b, THETA)) for i in tr]
ops_g = sum(greedy_counted(i)[1] for i in tr)
ops_h = sum(lecture_counted(i, THETA)[1] for i in tr)
time_g = time_per_instance(lambda i: greedy_plain(i.a, i.c, i.b), tr)
time_h = time_per_instance(lambda i: lecture_plain(i.a, i.c, i.b, THETA), tr)

print(f"\nselected heuristic on its {N_TRAIN} training instances")
print(f"  mean ratio      Greedy {statistics.mean(v / i.f_star for v, i in zip(val_g, tr)):.4f}"
      f"   learned {statistics.mean(v / i.f_star for v, i in zip(val_h, tr)):.4f}")
print(f"  worst ratio     Greedy {min(v / i.f_star for v, i in zip(val_g, tr)):.4f}"
      f"   learned {min(v / i.f_star for v, i in zip(val_h, tr)):.4f}")
print(f"  mean improvement over Greedy "
      f"{statistics.mean((h - g) / g for h, g in zip(val_h, val_g)):.1%}")
print(f"  operations      Greedy {ops_g}   learned {ops_h}   ratio {ops_h / ops_g:.2f}")
print(f"  time/instance   Greedy {time_g * 1e6:.2f} us   learned {time_h * 1e6:.2f} us"
      f"   ratio {time_h / time_g:.2f}")

# ---------------------------------------------------------------- 3. LaTeX table
ratio_g = [v / i.f_star for v, i in zip(val_g, tr)]
ratio_h = [v / i.f_star for v, i in zip(val_h, tr)]
lines = [
    r"\begin{tabular}{@{}lcc@{}}",
    r"\toprule",
    r" & Greedy & learned \\",
    r"\midrule",
    f"mean ratio & ${statistics.mean(ratio_g):.3f}$ & ${statistics.mean(ratio_h):.3f}$ \\\\",
    f"worst ratio & ${min(ratio_g):.3f}$ & ${min(ratio_h):.3f}$ \\\\",
    f"operations & ${ops_g}$ & ${ops_h}$ \\\\",
    f"time per instance & ${time_g * 1e6:.2f}$~$\\mu$s & ${time_h * 1e6:.2f}$~$\\mu$s \\\\",
    r"\bottomrule",
    r"\end{tabular}",
]
path = os.path.join(OUT, "point4_training_table.tex")
with open(path, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("saved", path)