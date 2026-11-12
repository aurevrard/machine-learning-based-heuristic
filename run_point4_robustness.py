"""Point 4, Section 4.4: robustness across seeds and derivative-free methods.

    python run_point4_robustness.py

Evaluates the 40 parameters of clip + lecture trained in Figure 1 (4 methods x 10 seeds,
read from results/point2_all_configs.json, produced by fig_point2.py) on:
    training   the training set of each run (seed = repetition)
    test F     200 instances F(m, k), m in [10, 200], seed 2000 (as in Section 4.3)
    larger m   200 instances F(m, k), m in [201, 2000], seed 2200
    random     the 1000 random instances of Section 4.3

Output: results/point4_robustness.png   one panel per set; per method, mean over the seeds
        of the mean ratio (dot) and range over the seeds (line); dashed line: Greedy; star:
        the selected parameter (SA, seed 1).
"""
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knapsack_family import greedy
from training import make_dataset, reward
from run_point4_unseen import random_dataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PATH_JSON = os.path.join(OUT, "point2_all_configs.json")
METHODS = ["RS", "SA", "GA", "NM"]
COLORS = {"RS": "#2a78d6", "SA": "#eb6834", "GA": "#1baf7a", "NM": "#eda100"}
K_RANGE = (6, 8)


def main():
    if not os.path.exists(PATH_JSON):
        sys.exit(f"{PATH_JSON} not found: run 'python fig_point2.py' first.")
    runs = json.load(open(PATH_JSON))["runs"]
    thetas = {m: [run["theta"] for run in runs[f"clip|lecture|{m}"]] for m in METHODS}

    test = make_dataset(200, (10, 200), K_RANGE, seed=2000)
    large = make_dataset(200, (201, 2000), K_RANGE, seed=2200)
    rand = random_dataset(1000, seed=2100)
    sets = {"training": None, "test $\\mathcal{F}$": test,
            "larger $m$": large, "random": rand}

    # ratio[set][method] = list over the seeds of the mean ratio
    ratio, greedy_ratio = {}, {}
    for name, data in sets.items():
        ratio[name] = {}
        for m in METHODS:
            vals = []
            for seed, th in enumerate(thetas[m], start=1):
                inst = make_dataset(40, (10, 200), K_RANGE, seed=seed) if data is None else data
                vals.append(reward(th, inst, "clip", "lecture", "ratio"))
            ratio[name][m] = vals
        if data is None:          # Greedy on the training sets, averaged over the seeds
            greedy_ratio[name] = statistics.mean(
                statistics.mean(greedy(i)[1] / i.f_star for i in
                                make_dataset(40, (10, 200), K_RANGE, seed=s)) for s in range(1, 11))
        else:
            greedy_ratio[name] = statistics.mean(greedy(i)[1] / i.f_star for i in data)

    print(f"{'set':22}{'Greedy':>8}" + "".join(f"{m:>22}" for m in METHODS))
    for name in sets:
        line = f"{name:22}{greedy_ratio[name]:8.3f}"
        for m in METHODS:
            v = ratio[name][m]
            line += f"   {statistics.mean(v):.3f} [{min(v):.3f}, {max(v):.3f}]"
        print(line)
    for m in METHODS:
        n_opt = sum(v == 1 for v in ratio["test $\\mathcal{F}$"][m])
        print(f"{m}: {n_opt} of 10 parameters optimal on every test instance")

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.6))
    xlims = {"training": (0.5, 1.01), "test $\\mathcal{F}$": (0.5, 1.01),
             "larger $m$": (0.0, 1.02), "random": (0.6, 1.01)}
    for ax, name in zip(axes.flat, sets):
        ax.axvline(greedy_ratio[name], color="#3a3a36", lw=1.2, ls="--", zorder=1)
        for j, m in enumerate(METHODS):
            y = len(METHODS) - 1 - j
            v = ratio[name][m]
            ax.plot([min(v), max(v)], [y, y], color=COLORS[m], lw=2.2, alpha=0.45,
                    solid_capstyle="round")
            ax.scatter(statistics.mean(v), y, color=COLORS[m], s=36, zorder=3)
            if m == "SA":                                  # the selected parameter (seed 1)
                ax.scatter(v[0], y + 0.3, marker="*", s=60, color="#3a3a36", zorder=5)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS[::-1])
        ax.set_ylim(-0.6, len(METHODS) - 0.2)
        ax.set_xlim(*xlims[name])
        ax.set_title(name, fontsize=8.5)
        ax.grid(axis="x", color="#e6e6e2", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("mean ratio to the optimum")
    handles = [plt.Line2D([], [], color="#3a3a36", ls="--", lw=1.2),
               plt.Line2D([], [], color="#8a8a85", marker="o", ls="-", lw=2.2, alpha=0.6),
               plt.Line2D([], [], color="#3a3a36", marker="*", ls="", markersize=8)]
    fig.legend(handles, ["Greedy", "mean over the seeds (dot), worst to best seed (line)",
                         "selected parameter"], loc="lower center", ncol=3, frameon=False,
               fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(OUT, "point4_robustness.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()