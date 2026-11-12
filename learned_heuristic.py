"""Learned one-pass heuristic of Lecture 2 (point 2).

    x_i = floor( (b_i / a_i) * phi_theta(s_i) ),   b_{i+1} = b_i - a_i x_i

Since 0 <= phi <= 1, every generated point is feasible by construction.
A configuration is a pair (phi, state), chosen by name.
"""
import math


#phi_theta
def phi_clip(theta, s):
    """clip(1 - ReLU(theta^T s), 0, 1): exactly 0 or 1 on whole intervals."""
    z = sum(t * si for t, si in zip(theta, s))
    return min(max(1 - max(z, 0.0), 0.0), 1.0)


def phi_sigmoid(theta, s):
    """sigma(theta^T s) = 1 / (1 + exp(-z))."""
    z = sum(t * si for t, si in zip(theta, s))
    if z < -50:                       # avoid overflow in exp
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


PHIS = {"clip": phi_clip, "sigmoid": phi_sigmoid}


#states s_i
def state_lecture(ai, ci, r, b):
    """State suggested in Lecture 2: (1, c_i/a_i, c_i/b_i, a_i/b_i, b_i/b)."""
    return (1.0, ci / ai, ci / r, ai / r, r / b)


def state_minimal(ai, ci, r, b):
    """Minimal state: (1, a_i/b_i), share of the remaining capacity used by one unit."""
    return (1.0, ai / r)


def state_waste(ai, ci, r, b):
    """(1, a_i/b_i, w_i) with w_i = share of b_i left empty if we take item i at maximum."""
    q = r // ai
    return (1.0, ai / r, (r - q * ai) / r)


STATES = {"lecture": state_lecture, "minimal": state_minimal, "waste": state_waste}
STATE_DIM = {"lecture": 5, "minimal": 2, "waste": 3}


#heuristic
def learned(inst, theta, phi="clip", state="minimal"):
    """Return (x, f(x)) for the learned heuristic with parameter theta."""
    phi_f, state_f = PHIS[phi], STATES[state]
    x, r = [], inst.b
    for ai, ci in zip(inst.a, inst.c):
        if ai > r:                    # item does not fit: x_i = 0 whatever phi (also covers r = 0)
            x.append(0)
            continue
        s = state_f(ai, ci, r, inst.b)
        xi = math.floor((r / ai) * phi_f(theta, s))
        x.append(xi)
        r -= ai * xi
    return x, inst.value(x)