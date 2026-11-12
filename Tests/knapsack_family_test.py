from knapsack_family import UKPInstance, ukp_family, greedy, dp_opt


def test_dp_on_known_examples():
    # exact DP checked on optima known independently of the AI
    for m in range(2, 30):        # lecture two-item example: optimum 4m - 2
        assert dp_opt(UKPInstance([m + 1, m], [2*m + 1, 2*m - 1], 2*m)) == 4*m - 2
    assert dp_opt(UKPInstance([3, 2], [5, 3], 4)) == 6     # TP2, exercise 2(c)


def test_family_structure():
    # n = 2k - 1, weight m removed, items sorted by decreasing density
    for k in range(2, 8):
        for m in range(k + 1, 60):
            inst = ukp_family(m, k)
            assert inst.n == 2 * k - 1
            assert m not in inst.a
            dens = [ci / ai for ai, ci in zip(inst.a, inst.c)]
            assert all(d1 > d2 for d1, d2 in zip(dens, dens[1:]))


def test_closed_form_optimum():
    # x* feasible, two positive components, value 4m - 2 equal to the DP optimum
    for k in range(2, 8):
        for m in range(k + 1, 60):
            inst = ukp_family(m, k)
            assert inst.is_feasible(inst.x_star)
            assert sum(1 for xi in inst.x_star if xi > 0) == 2
            assert inst.value(inst.x_star) == inst.f_star == dp_opt(inst)


def test_no_single_type_optimum():
    # no optimal solution uses a single item type: t units of item i are always worse than f*
    for k in range(2, 8):
        for m in range(k + 1, 60):
            inst = ukp_family(m, k)
            for ai, ci in zip(inst.a, inst.c):
                assert (inst.b // ai) * ci < inst.f_star


def test_greedy_value():
    # Greedy takes one unit of item 1 only: value 2m + 2k - 1
    for k in range(2, 8):
        for m in range(k + 1, 60):
            x, v = greedy(ukp_family(m, k))
            assert x[0] == 1 and sum(x) == 1
            assert v == 2 * m + 2 * k - 1


def test_by_hand_F_10_6():
    # checked on paper: weights 16..11, 9..5 (no 10), b = 20, Greedy 31, optimum 38 (11 + 9)
    inst = ukp_family(10, 6)
    assert inst.a == [16, 15, 14, 13, 12, 11, 9, 8, 7, 6, 5]
    assert greedy(inst)[1] == 31
    assert inst.f_star == 38 and inst.x_star[5] == inst.x_star[6] == 1