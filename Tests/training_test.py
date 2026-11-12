import math

from knapsack_family import ukp_family
from training import (OPTIMISERS, make_dataset, ensure_optimum, reward, greedy_reward, train)


def sphere(theta):
    """Smooth test function with known maximum 0 at theta = (3, -2)."""
    return -((theta[0] - 3) ** 2 + (theta[1] + 2) ** 2)


def test_budget_is_respected():
    # every optimiser uses exactly `budget` evaluations (fair comparison)
    for name, opt in OPTIMISERS.items():
        calls = []
        f = lambda th: calls.append(1) or sphere(th)
        _, _, hist = opt(f, 2, 157, seed=0, bound=10)
        assert len(calls) == 157 and len(hist) == 157, name


def test_history_is_best_so_far():
    # the recorded curve never decreases and ends at the returned best value
    for name, opt in OPTIMISERS.items():
        theta, best, hist = opt(sphere, 2, 300, seed=1, bound=10)
        assert all(h2 >= h1 for h1, h2 in zip(hist, hist[1:])), name
        assert hist[-1] == best == sphere(theta), name


def test_optimisers_find_known_maximum():
    # sanity check on a smooth problem whose maximum is known
    for name in ("SA", "GA", "NM"):
        theta, best, _ = OPTIMISERS[name](sphere, 2, 3000, seed=2, bound=10)
        assert math.dist(theta, (3, -2)) < 0.3, name
    theta, best, _ = OPTIMISERS["RS"](sphere, 2, 3000, seed=2, bound=10)
    assert math.dist(theta, (3, -2)) < 1.0          # random search is coarser


def test_reproducible_with_seed():
    for name, opt in OPTIMISERS.items():
        assert opt(sphere, 2, 200, seed=5, bound=10) == opt(sphere, 2, 200, seed=5, bound=10), name


def test_reward_kinds():
    data = [ukp_family(m, 6) for m in (10, 50, 100)]
    optimal = (-600.0, 0.0, 400.0, -400.0, 400.0)       # hand-made theta, optimal for m <= 200
    assert reward(optimal, data, "clip", "lecture", "ratio") == 1.0
    assert reward(optimal, data, "clip", "lecture", "min") == 1.0
    assert reward(optimal, data, "clip", "lecture", "raw") == sum(i.f_star for i in data) / 3
    # phi = 1 everywhere reproduces Greedy for every reward
    for kind in ("raw", "ratio", "min"):
        assert reward((-1e6, 0.0), data, "clip", "minimal", kind) == greedy_reward(data, kind)


def test_raw_and_ratio_rank_differently():
    # the example of the report: raw mean prefers theta_B, ratio mean prefers theta_A
    values_A, values_B, optima = [78, 760], [51, 798], [78, 798]
    from training import aggregate
    assert aggregate(values_B, optima, "raw") > aggregate(values_A, optima, "raw")
    assert aggregate(values_A, optima, "ratio") > aggregate(values_B, optima, "ratio")


def test_ensure_optimum_uses_dp():
    inst = ukp_family(10, 6)
    inst.f_star = None
    assert ensure_optimum([inst])[0].f_star == 38


def test_training_beats_greedy():
    # after training on the family, the heuristic beats Greedy on held-out instances
    tr = make_dataset(20, (10, 100), (6, 8), seed=0)
    te = make_dataset(50, (10, 100), (6, 8), seed=1)
    for method in ("SA", "GA"):
        theta, _, _ = train(tr, "clip", "waste", 3, method, budget=1500, seed=0)
        assert reward(theta, te, "clip", "waste") > greedy_reward(te) + 0.1, method