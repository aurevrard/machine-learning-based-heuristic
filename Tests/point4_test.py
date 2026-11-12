from knapsack_family import dp_opt, greedy
from learned_heuristic import learned
from run_point4_unseen import g_family, g_dataset, random_dataset, lecture_plain, THETA
from run_point4_failure import optimal_solution, trace


def test_g_family_closed_form_optimum():
    # the optimum 2pm - p of G(m, k, p) agrees with dynamic programming
    for p in (2, 3, 4):
        for k in (2, 3, 4):
            m_min = max((p + 1) * (k - 1), (p - 1) * k) + 1
            for m in range(max(m_min, k + 1), m_min + 25):
                inst = g_family(m, k, p)
                assert dp_opt(inst) == inst.f_star == 2 * p * m - p


def test_g_family_greedy_value():
    # Greedy takes p - 1 units of the heaviest item, nothing else fits
    for p in (3, 4, 5):
        for inst in g_dataset(30, p, seed=7):
            k = (inst.n + 1) // 2
            m = inst.b // p
            assert greedy(inst)[1] == (p - 1) * (2 * m + 2 * k - 1)


def test_random_instances():
    # reproducible, optimum by DP, and both heuristics are feasible and below the optimum
    data = random_dataset(50, seed=3)
    assert [i.b for i in data] == [i.b for i in random_dataset(50, seed=3)]
    for inst in data:
        assert all(d1 >= d2 for d1, d2 in zip([c / a for a, c in zip(inst.a, inst.c)][:-1],
                                              [c / a for a, c in zip(inst.a, inst.c)][1:]))
        x = lecture_plain(inst.a, inst.c, inst.b, THETA)
        assert inst.is_feasible(x) and inst.value(x) <= inst.f_star
        assert greedy(inst)[1] <= inst.f_star


def test_plain_implementation_is_the_trained_heuristic():
    for inst in random_dataset(50, seed=4) + g_dataset(30, 3, seed=5):
        assert lecture_plain(inst.a, inst.c, inst.b, THETA) == learned(inst, THETA, "clip",
                                                                       "lecture")[0]


def test_failure_case_tools():
    # the backtracked solution is feasible and optimal, and the trace reproduces the heuristic
    for inst in random_dataset(50, seed=6):
        x = optimal_solution(inst)
        assert inst.is_feasible(x) and inst.value(x) == inst.f_star
        assert [row[5] for row in trace(inst, THETA)] == lecture_plain(inst.a, inst.c, inst.b,
                                                                        THETA)