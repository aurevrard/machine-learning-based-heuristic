import random

from knapsack_family import UKPInstance, ukp_family, greedy
from learned_heuristic import learned
from operation_count import greedy_counted, lecture_counted, waste_counted, BOUND

THETA_LECTURE = [(-600.0, 0.0, 400.0, -400.0, 400.0),          # theta* of the report
                 (-699.5, 0.0, 402.0, -404.0, 500.0),          # theta** of the report
                 (-1000.0, -17.18, 438.3, -535.8, 864.2),      # selected parameter
                 (0.5, 0.0, 0.0, 0.0, 0.0)]                    # always on the ramp
THETA_WASTE = [(-1.0, 0.0, 8.0), (-94.96, 62.13, 969.8), (0.5, 0.0, 0.0)]


def random_instance(rng, n):
    """Random UKP instance with items sorted by non-increasing density."""
    items = [(rng.randint(1, 60), rng.randint(1, 100)) for _ in range(n)]
    items.sort(key=lambda ac: ac[1] / ac[0], reverse=True)
    return UKPInstance([a for a, _ in items], [c for _, c in items], rng.randint(1, 500))


def instances():
    rng = random.Random(0)
    data = [ukp_family(m, k) for k in (2, 6, 8) for m in range(k + 1, 300, 7)]
    data += [random_instance(rng, rng.randint(1, 40)) for _ in range(400)]
    return data


def test_greedy_counted_is_greedy():
    # same solution as the Greedy of point 1, and exactly 8n + 1 operations
    for inst in instances():
        x, ops = greedy_counted(inst)
        assert x == greedy(inst)[0]
        assert ops == 8 * inst.n + 1


def test_counted_versions_give_the_trained_heuristic():
    # the cheap implementation returns exactly the x of learned(), used for training
    for inst in instances():
        for th in THETA_LECTURE:
            assert lecture_counted(inst, th)[0] == learned(inst, th, "clip", "lecture")[0]
        for th in THETA_WASTE:
            assert waste_counted(inst, th)[0] == learned(inst, th, "clip", "waste")[0]


def test_at_most_three_times_greedy():
    # bound of Section 3 on every instance, and the per-item bounds alpha n + beta
    for inst in instances():
        g = greedy_counted(inst)[1]
        for name, f, thetas in (("lecture", lecture_counted, THETA_LECTURE),
                                ("waste", waste_counted, THETA_WASTE)):
            alpha, beta = BOUND[name]
            for th in thetas:
                ops = f(inst, th)[1]
                assert ops <= alpha * inst.n + beta
                assert ops <= 3 * g


def test_worst_case_is_reached():
    # with z = 0.5 every fitting item is on the ramp: the bound 24n + 2 is attained
    inst = UKPInstance([1] * 10, [1] * 10, 10 ** 6)
    assert lecture_counted(inst, (0.5, 0, 0, 0, 0))[1] == 24 * 10 + 2
    assert waste_counted(inst, (0.5, 0, 0))[1] == 22 * 10 + 1