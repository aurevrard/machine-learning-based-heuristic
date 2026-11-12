"""Point 2: training experiments.

    python run_point2.py           full experiment (about 5-10 minutes)
    python run_point2.py --quick   quick check while developing (a few seconds)

Every repetition r uses the same seed r for the training set and for the optimiser
(paired design): within a repetition, all methods and configurations see the same
training instances. Test sets are fixed and shared by all runs.

Outputs (folder results/):
    point2_runs.csv         one line per training run
    point2_convergence.png  best-so-far reward vs number of evaluations
"""
import csv
import os
import statistics
import sys
import time

from learned_heuristic import STATE_DIM
from training import make_dataset, train, reward, greedy_reward

QUICK = "--quick" in sys.argv

# ---------------------------------------------------------------- settings
BUDGET = 300 if QUICK else 2000           # evaluations of psi per run (same for every method)
N_TRAIN = 40                              # training instances per repetition
REPS = [1] if QUICK else list(range(1, 11))
REPS_SECONDARY = [1] if QUICK else list(range(1, 6))
K_RANGE = (6, 8)                          # n = 2k - 1 between 11 and 15
WIDE, NARROW = (10, 200), (20, 39)        # ranges of m for training
BOUND = 1000.0                            # search box [-L, L]^d
PHIS = ["clip", "sigmoid"]
STATES = ["lecture", "minimal", "waste"]
METHODS = ["RS", "SA", "GA", "NM"]

OUT = "results"
os.makedirs(OUT, exist_ok=True)

# fixed held-out sets, shared by all runs
TEST = {
    "test_wide": make_dataset(200, WIDE, K_RANGE, seed=1000),
    "test_narrow": make_dataset(200, NARROW, K_RANGE, seed=1001),
    "test_big_m": make_dataset(200, (201, 600), K_RANGE, seed=1002),
}

rows, curves = [], {}


def run(experiment, m_range, phi, state, method, kind, rep, bound=BOUND):
    """Train one configuration and evaluate it on the training set and the test sets."""
    train_set = make_dataset(N_TRAIN, m_range, K_RANGE, seed=rep)
    t0 = time.perf_counter()
    theta, best, hist = train(train_set, phi, state, STATE_DIM[state], method,
                              BUDGET, seed=rep, kind=kind, bound=bound)
    row = {"experiment": experiment, "m_range": f"{m_range[0]}-{m_range[1]}",
           "phi": phi, "state": state, "method": method, "reward": kind, "rep": rep,
           "bound": bound, "time_s": round(time.perf_counter() - t0, 3),
           "train_ratio": reward(theta, train_set, phi, state, "ratio"),
           "train_min": reward(theta, train_set, phi, state, "min"),
           "greedy_train_ratio": greedy_reward(train_set, "ratio")}
    for name, data in TEST.items():
        row[f"{name}_ratio"] = reward(theta, data, phi, state, "ratio")
        row[f"{name}_min"] = reward(theta, data, phi, state, "min")
    row["theta"] = " ".join(f"{t:.4g}" for t in theta)
    rows.append(row)
    return hist


def summary(title, selection, columns):
    """Print mean +- std over repetitions for each group of runs."""
    print(f"\n=== {title} ===")
    groups = {}
    for r in rows:
        if selection(r):
            key = (r["phi"], r["state"], r["method"], r["reward"], r["m_range"], r["bound"])
            groups.setdefault(key, []).append(r)
    header = f"{'phi':8}{'state':9}{'meth':5}{'reward':7}{'m':8}{'L':>7}"
    print(header + "".join(f"{c:>22}" for c in columns))
    for key, rs in groups.items():
        line = f"{key[0]:8}{key[1]:9}{key[2]:5}{key[3]:7}{key[4]:8}{key[5]:>7.0f}"
        for c in columns:
            vals = [r[c] for r in rs]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            line += f"{statistics.mean(vals):>14.4f} +-{sd:6.4f}"
        print(line)


# ---------------------------------------------------------------- Greedy reference
print("Greedy on the test sets:")
for name, data in TEST.items():
    print(f"  {name:12} mean ratio {greedy_reward(data, 'ratio'):.4f}"
          f"   worst ratio {greedy_reward(data, 'min'):.4f}")

# ---------------------------------------------------------------- 1. main experiment
# 6 configurations x 4 methods x 2 rewards x 10 repetitions, wide range of m
for phi in PHIS:
    for state in STATES:
        for method in METHODS:
            for kind in ["ratio", "raw"]:
                for rep in REPS:
                    hist = run("main", WIDE, phi, state, method, kind, rep)
                    if phi == "clip" and state == "lecture" and kind == "ratio":
                        curves.setdefault(method, []).append(hist)
        print(f"done: {phi} {state}")

summary("Main experiment (wide range of m)", lambda r: r["experiment"] == "main",
        ["train_ratio", "test_wide_ratio", "test_wide_min", "test_big_m_ratio"])

# ---------------------------------------------------------------- 2. narrow range of m
for phi in PHIS:
    for state in STATES:
        for method in METHODS:
            for rep in REPS_SECONDARY:
                run("narrow", NARROW, phi, state, method, "ratio", rep)

summary("Narrow range of m (20-39)", lambda r: r["experiment"] == "narrow",
        ["train_ratio", "test_narrow_ratio", "test_narrow_min", "test_wide_ratio"])

# ---------------------------------------------------------------- 3. worst-case reward
for method in METHODS:
    for rep in REPS_SECONDARY:
        run("worst", WIDE, "clip", "lecture", method, "min", rep)

summary("Worst-case reward (clip, lecture)",
        lambda r: r["experiment"] == "worst" or (r["experiment"] == "main" and r["phi"] == "clip"
                                                 and r["state"] == "lecture" and r["rep"] in REPS_SECONDARY),
        ["test_wide_ratio", "test_wide_min"])

# ---------------------------------------------------------------- 4. size of the search box
for bound in [100.0, 10000.0]:
    for method in METHODS:
        for rep in REPS_SECONDARY:
            run("box", WIDE, "clip", "lecture", method, "ratio", rep, bound=bound)

summary("Size of the search box L (clip, lecture, ratio)",
        lambda r: r["experiment"] == "box" or (r["experiment"] == "main" and r["phi"] == "clip"
                                               and r["state"] == "lecture" and r["reward"] == "ratio"
                                               and r["rep"] in REPS_SECONDARY),
        ["train_ratio", "test_wide_ratio"])

# ---------------------------------------------------------------- outputs
with open(os.path.join(OUT, "point2_runs.csv"), "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for method, hs in curves.items():
        mean = [statistics.mean(h[i] for h in hs) for i in range(BUDGET)]
        ax.plot(range(1, BUDGET + 1), mean, label=method)
    ax.set_xlabel("evaluations of $\\psi$")
    ax.set_ylabel("best mean ratio (train)")
    ax.set_title("Convergence, clip + lecture state, ratio reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "point2_convergence.png"), dpi=200)
except ImportError:
    print("matplotlib not installed: no convergence plot")

print(f"\n{len(rows)} runs saved in {OUT}/point2_runs.csv")