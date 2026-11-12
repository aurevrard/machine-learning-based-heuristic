"""Point 2: the two figures of Section 2.3.

Figure 1  results/point2_all_configs.png
    Every configuration (phi x state) and every method, ratio reward, seeds 1..10.
    Per method: mean validation ratio +- std, mean worst ratio (star), number of seeds
    reaching >= 0.95, and median number of evaluations needed to reach 99% of the
    final training value. The meaning of these marks is given in the LaTeX caption.

Figure 2  results/point2_variants.png
    Secondary experiments for clip + lecture state, seeds 1..5: rewards, size of the
    search box, narrow range of sizes. Reads results/point2_runs.csv (run_point2.py).

Usage (from the root of the project):
    python run_point2.py            produces results/point2_runs.csv (needed by Figure 2)
    python fig_point2.py            trains the 240 runs of Figure 1 (about 2 minutes)
    python fig_point2.py --replot   redraws both figures from the saved results
"""
import csv
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learned_heuristic import STATE_DIM
from training import make_dataset, train, reward, greedy_reward

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
PATH_JSON = os.path.join(OUT, "point2_all_configs.json")
PATH_CSV = os.path.join(OUT, "point2_runs.csv")

BUDGET, N_TRAIN, SEEDS = 2000, 40, list(range(1, 11))
K_RANGE, WIDE = (6, 8), (10, 200)
PHIS, STATES = ["clip", "sigmoid"], ["lecture", "minimal", "waste"]
METHODS = ["RS", "SA", "GA", "NM"]
COLORS = {"RS": "#2a78d6", "SA": "#eb6834", "GA": "#1baf7a", "NM": "#eda100"}
VALID = make_dataset(200, WIDE, K_RANGE, seed=1000)          # validation set (m in [10, 200])
GREEDY = greedy_reward(VALID, "ratio")

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


# =============================================================== Figure 1
def evals_to_99(hist):
    """Number of evaluations needed to reach 99% of the final best training value."""
    target = 0.99 * hist[-1]
    return next(i + 1 for i, h in enumerate(hist) if h >= target)


def run_all_configs():
    """Train every (phi, state, method) for 10 seeds and evaluate on the validation set."""
    results = {}
    for phi in PHIS:
        for state in STATES:
            for method in METHODS:
                runs = []
                for seed in SEEDS:
                    tr = make_dataset(N_TRAIN, WIDE, K_RANGE, seed=seed)
                    theta, best, hist = train(tr, phi, state, STATE_DIM[state], method,
                                              BUDGET, seed)
                    runs.append({"valid": reward(theta, VALID, phi, state, "ratio"),
                                 "worst": reward(theta, VALID, phi, state, "min"),
                                 "evals99": evals_to_99(hist), "train": best,
                                 "theta": list(theta)})
                results[f"{phi}|{state}|{method}"] = runs
            print("done", phi, state)
    with open(PATH_JSON, "w") as fh:
        json.dump({"greedy_valid": GREEDY, "runs": results}, fh, indent=1)
    return results


def load_all_configs():
    """Read the saved results of Figure 1, or explain what is missing."""
    if not os.path.exists(PATH_JSON):
        sys.exit(f"{PATH_JSON} not found: run 'python fig_point2.py' once without --replot.")
    results = json.load(open(PATH_JSON))["runs"]
    if any("worst" not in run for runs in results.values() for run in runs):
        sys.exit(f"{PATH_JSON} comes from an older version (no worst ratio): "
                 "run 'python fig_point2.py' once without --replot.")
    return results


def figure_all_configs(results):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3), sharey=True)
    for r, phi in enumerate(PHIS):
        for c, state in enumerate(STATES):
            ax = axes[r, c]
            ax.axhline(GREEDY, color="#8a8a85", lw=1, ls="--", zorder=0)
            for x, method in enumerate(METHODS):
                runs = results[f"{phi}|{state}|{method}"]
                vals = [run["valid"] for run in runs]
                mean, sd = statistics.mean(vals), statistics.stdev(vals)
                ax.bar(x, mean, width=0.62, color=COLORS[method], alpha=0.55, edgecolor="none")
                ax.errorbar(x, mean, yerr=[[sd], [min(sd, 1.0 - mean)]], color="#3a3a36",
                            lw=1, capsize=2.5)
                worst = statistics.mean(run["worst"] for run in runs)
                ax.scatter(x, worst, marker="*", s=45, color="#3a3a36", zorder=3)
                good = sum(v >= 0.95 for v in vals)
                ax.text(x, 1.04, f"{good}", ha="center", va="bottom", fontsize=7,
                        color="#3a3a36")
                med = statistics.median(run["evals99"] for run in runs)
                ax.text(x, 0.02, f"{med:.0f}", ha="center", va="bottom", fontsize=6.5,
                        color="#3a3a36")
            ax.set_xticks(range(len(METHODS)))
            ax.set_xticklabels(METHODS)
            ax.set_ylim(0, 1.12)
            ax.set_title(f"{phi} + {state}", fontsize=8.5)
            ax.grid(axis="y", color="#e6e6e2", lw=0.6)
            ax.set_axisbelow(True)
        axes[r, 0].set_ylabel("validation mean ratio")
    axes[0, 0].text(-0.45, GREEDY + 0.02, "Greedy", fontsize=6.5, color="#5a5a55")
    fig.tight_layout()
    path = os.path.join(OUT, "point2_all_configs.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


# =============================================================== Figure 2
GREEDY_BY_SET = {"test_wide": GREEDY,
                 "test_narrow": greedy_reward(make_dataset(200, (20, 39), K_RANGE, seed=1001))}

VARIANTS_LEFT = [
    ("reference", lambda r: r["experiment"] == "main" and r["reward"] == "ratio", "test_wide"),
    ("raw reward", lambda r: r["experiment"] == "main" and r["reward"] == "raw", "test_wide"),
    ("worst-case\nreward", lambda r: r["experiment"] == "worst", "test_wide"),
    ("$L=100$", lambda r: r["experiment"] == "box" and float(r["bound"]) == 100, "test_wide"),
    ("$L=10\\,000$", lambda r: r["experiment"] == "box" and float(r["bound"]) == 10000,
     "test_wide")]
VARIANTS_RIGHT = [
    ("validated on\n$m\\in[20,39]$", lambda r: r["experiment"] == "narrow", "test_narrow"),
    ("validated on\n$m\\in[10,200]$", lambda r: r["experiment"] == "narrow", "test_wide")]


def draw_variants(ax, rows, variants):
    width = 0.19
    for g, (label, keep, col) in enumerate(variants):
        ax.hlines(GREEDY_BY_SET[col], g - 0.46, g + 0.46, color="#8a8a85", lw=1, ls="--",
                  zorder=0)
        for j, method in enumerate(METHODS):
            sel = [r for r in rows if keep(r) and r["method"] == method]
            mean_vals = [float(r[col + "_ratio"]) for r in sel]
            worst_vals = [float(r[col + "_min"]) for r in sel]
            x = g + (j - 1.5) * (width + 0.02)
            mean, sd = statistics.mean(mean_vals), statistics.stdev(mean_vals)
            ax.bar(x, mean - 0.3, bottom=0.3, width=width, color=COLORS[method], alpha=0.55,
                   edgecolor="none", label=method if g == 0 else None)
            ax.errorbar(x, mean, yerr=[[sd], [min(sd, 1.0 - mean)]], color="#3a3a36",
                        lw=0.9, capsize=2)
            ax.scatter(x, statistics.mean(worst_vals), marker="*", s=40, color="#3a3a36",
                       zorder=3, label="worst ratio" if (g == 0 and j == 0) else None)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v[0] for v in variants])
    ax.set_ylim(0.3, 1.03)
    ax.grid(axis="y", color="#e6e6e2", lw=0.6)
    ax.set_axisbelow(True)


def figure_variants():
    if not os.path.exists(PATH_CSV):
        print(f"{PATH_CSV} not found: run 'python run_point2.py' first. Figure 2 skipped.")
        return
    rows = [r for r in csv.DictReader(open(PATH_CSV))
            if r["phi"] == "clip" and r["state"] == "lecture" and int(r["rep"]) <= 5]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True,
                                   gridspec_kw={"width_ratios": [5, 2.3]})
    draw_variants(axl, rows, VARIANTS_LEFT)
    draw_variants(axr, rows, VARIANTS_RIGHT)
    axl.set_title("training choices (trained and validated on $m\\in[10,200]$)", fontsize=8.5)
    axr.set_title("training on $m\\in[20,39]$", fontsize=8.5)
    axl.set_ylabel("validation ratio")
    axl.text(-0.46, GREEDY + 0.012, "Greedy", fontsize=6.5, color="#5a5a55")
    handles, labels = axl.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(OUT, "point2_variants.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


# =============================================================== main
if __name__ == "__main__":
    results = load_all_configs() if "--replot" in sys.argv else run_all_configs()
    figure_all_configs(results)
    figure_variants()