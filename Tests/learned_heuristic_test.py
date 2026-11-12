import random

from knapsack_family import UKPInstance, ukp_family, greedy, dp_opt
from learned_heuristic import learned, phi_clip, phi_sigmoid, PHIS, STATES, STATE_DIM


def random_instance(rng):
    """Random UKP instance sorted by density (not from the family)."""
    n = rng.randint(2, 12)
    items = [(rng.randint(1, 40), rng.randint(1, 60)) for _ in range(n)]
    items.sort(key=lambda it: it[1] / it[0], reverse=True)
    return UKPInstance([a for a, _ in items], [c for _, c in items], rng.randint(1, 150))


def test_phi_ranges():
    # clip reaches 0 and 1 exactly; sigmoid stays strictly inside (0, 1) for moderate z
    assert phi_clip((1.0,), (-5.0,)) == 1.0
    assert phi_clip((1.0,), (5.0,)) == 0.0
    assert phi_clip((1.0,), (0.25,)) == 0.75
    assert 0.0 < phi_sigmoid((1.0,), (3.0,)) < 1.0


def test_feasible_by_construction():
    # any theta, any configuration, any instance: the point is feasible and integer
    rng = random.Random(0)
    for _ in range(300):
        inst = random_instance(rng)
        for phi in PHIS:
            for state in STATES:
                theta = [rng.uniform(-50, 50) for _ in range(STATE_DIM[state])]
                x, v = learned(inst, theta, phi, state)
                assert inst.is_feasible(x)
                assert all(isinstance(xi, int) for xi in x)
                assert v <= dp_opt(inst)


def test_phi_one_gives_greedy():
    # theta^T s very negative  =>  phi = 1  =>  same point as Greedy
    rng = random.Random(1)
    for _ in range(200):
        inst = random_instance(rng)
        for state in STATES:
            theta = [-1e6] + [0.0] * (STATE_DIM[state] - 1)
            assert learned(inst, theta, "clip", state) == greedy(inst)


def test_minimal_rule_takes_one_type():
    # theta = (-m, 2m), rule "skip if a_i/b_i > 1/2": skips the heavy items, then takes
    # 2 units of weight m-1 -> 4m - 6, close to but below the optimum 4m - 2
    for k in range(2, 8):
        for m in range(k + 2, 60):
            inst = ukp_family(m, k)
            assert learned(inst, (-m, 2 * m), "clip", "minimal")[1] == 4 * m - 6


def test_waste_rule_takes_one_type():
    # theta = (-1, 0, 8), rule "skip if w_i >= 1/4": same behaviour, 4m - 6, for every m >= 2k
    for k in range(2, 8):
        for m in range(max(2 * k, 8), 400):
            inst = ukp_family(m, k)
            assert learned(inst, (-1.0, 0.0, 8.0), "clip", "waste")[1] == 4 * m - 6


def test_lecture_state_can_be_optimal():
    # with b_i/b in the state, theta = (-3m-1, 0, 0, 2m, 2m) takes item m+1 once, skips the
    # other heavy items, then takes its partner m-1: the optimal pair, value 4m - 2
    for k in range(2, 8):
        for m in range(k + 1, 60):
            inst = ukp_family(m, k)
            theta = (-3 * m - 1, 0.0, 0.0, 2 * m, 2 * m)
            assert learned(inst, theta, "clip", "lecture")[1] == 4 * m - 2


def test_sigmoid_loses_one_unit_at_threshold():
    # one item of weight 6, capacity 12 (a/b = 1/2 exactly): clip takes 2 units, sigmoid only 1
    inst = UKPInstance([6], [11], 12)
    assert learned(inst, (-6, 12), "clip", "minimal")[0] == [2]
    assert learned(inst, (6, -12), "sigmoid", "minimal")[0] == [1]