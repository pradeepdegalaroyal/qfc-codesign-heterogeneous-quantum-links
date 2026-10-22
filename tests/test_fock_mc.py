import numpy as np
import pytest

from qfclink import physics as ph
from qfclink.fock import pure_mode_pair, simulate_bsm
from qfclink.montecarlo import simulate_heralding


@pytest.mark.parametrize("overlap", [0.0, 0.3, 0.7, 1.0])
def test_fock_fidelity_equals_half_one_plus_v(overlap):
    res = simulate_bsm(*pure_mode_pair(overlap))
    assert res.herald_probability == pytest.approx(0.5)
    assert res.mean_fidelity == pytest.approx((1 + overlap**2) / 2, abs=1e-12)


def test_fock_mixed_states_use_trace_overlap():
    rng = np.random.default_rng(3)
    vecs = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ea = [(0.3, vecs[0]), (0.7, vecs[1])]
    eb = [(0.6, vecs[2]), (0.4, vecs[3])]
    vis = sum(pa * pb * abs(np.vdot(a, b)) ** 2 for pa, a in ea for pb, b in eb)
    assert simulate_bsm(ea, eb).mean_fidelity == pytest.approx((1 + vis) / 2, abs=1e-12)


def test_monte_carlo_agrees_with_analytic_model():
    eta_a, eta_b, mu, coh = 0.1, 0.2, 0.01, 0.8
    pp, p1, p2 = ph.herald_probabilities(eta_a, eta_b, 1 - np.exp(-mu))
    f = float(ph.bell_diagonal(pp, p1 + p2, coh)[0])
    mc = simulate_heralding(eta_a, eta_b, mu, coh, 300_000, np.random.default_rng(0))
    assert abs(mc.p_herald - float(pp + p1 + p2)) < 5 * mc.p_herald_se
    assert abs(mc.fidelity - f) < 5 * mc.fidelity_se
