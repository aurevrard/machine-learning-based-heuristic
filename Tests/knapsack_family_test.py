from knapsack_family import ukp_family, greedy, dp_opt

def test_closed_form_optimum():
    for k in range(2, 8):
        for m in range(k + 1, 60):
            inst = ukp_family(m, k)
            assert inst.is_feasible(inst.x_star)
            assert inst.value(inst.x_star) == inst.f_star == dp_opt(inst)

def test_greedy_value():
    for k in range(2, 8):
        for m in range(k + 1, 60):
            assert greedy(ukp_family(m, k))[1] == 2 * m + 2 * k - 1

def test_dp_on_lecture_example():
    from knapsack_family import UKPInstance
    for m in range(2, 30):
        assert dp_opt(UKPInstance([m + 1, m], [2*m + 1, 2*m - 1], 2*m)) == 4*m - 2