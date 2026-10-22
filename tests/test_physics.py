import numpy as np
import pytest
import torch

from qfclink import physics as ph


def test_unfiltered_overlap_matches_lorentzian_formula():
    la, lb = 0.5e9, 10e6
    vis, ta, _ = ph.filtered_overlap(la, lb, 1e15, 1e15)
    expected = 4 * la * lb / (la + lb) ** 2
    assert float(vis) == pytest.approx(expected, rel=1e-6)
    assert float(ta) == pytest.approx(1.0, abs=1e-5)


def test_matched_filter_restores_indistinguishability():
    la, lb = 1e9, 1e6  # bandwidth ratio r = 1000
    vis_raw, _, _ = ph.filtered_overlap(la, lb, 1e15, 1e15)
    vis_f, ta, _ = ph.filtered_overlap(la, lb, lb, 1e15)
    assert float(vis_raw) < 0.01
    assert float(vis_f) > 0.99
    assert float(ta) == pytest.approx(lb / (la + lb), rel=1e-9)


def test_overlap_is_bounded_and_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(50):
        la, lb, fa, fb = 10 ** rng.uniform(6, 10, 4)
        v1, _, _ = ph.filtered_overlap(la, lb, fa, fb)
        v2, _, _ = ph.filtered_overlap(lb, la, fb, fa)
        assert 0.0 <= float(v1) <= 1.0 + 1e-12
        assert float(v1) == pytest.approx(float(v2), rel=1e-9)


def test_detuning_reduces_visibility():
    v0, _, _ = ph.filtered_overlap(10e6, 10e6, 1e15, 1e15, 0.0)
    v1, _, _ = ph.filtered_overlap(10e6, 10e6, 1e15, 1e15, 10e6)
    assert float(v0) == pytest.approx(1.0)
    # Identical Lorentzians detuned by one FWHM: V = 1 / (1 + (delta/Gamma)^2) = 0.5
    assert float(v1) == pytest.approx(0.5, rel=1e-9)


def test_fibre_attenuation_typical_values():
    assert 0.25 < float(ph.fibre_attenuation_db_per_km(1310.0)) < 0.35
    assert 0.15 < float(ph.fibre_attenuation_db_per_km(1550.0)) < 0.21


def test_pump_degeneracy_and_noise():
    # 780 nm -> 1560 nm needs a 1560 nm pump: degenerate and maximally noisy
    assert float(ph.pump_wavelength_nm(780.0, 1560.0)) == pytest.approx(1560.0)
    near = float(ph.raman_noise_per_mode(780.0, 1555.0, 5e-5, 1e-7))
    far = float(ph.raman_noise_per_mode(780.0, 1310.0, 5e-5, 1e-7))
    assert near > 100 * far
    assert far == pytest.approx(1e-7, rel=1e-3)


def test_conversion_efficiency_peaks_at_full_pump():
    p = torch.linspace(0.01, 1.0, 100, dtype=torch.float64)
    eff = ph.conversion_efficiency(p, 0.5)
    assert float(eff.max()) == pytest.approx(0.5)
    assert int(torch.argmax(eff)) == 99


def test_hashing_yield_limits():
    assert float(ph.hashing_yield(torch.tensor([1.0, 0, 0, 0], dtype=torch.float64))) == pytest.approx(1.0)
    assert float(ph.hashing_yield(torch.full((4,), 0.25, dtype=torch.float64))) == pytest.approx(-1.0)
    eigs = ph.bell_diagonal(1.0, 0.0, 0.0)  # F = 0.5 rank-2 state: zero hashing yield
    assert float(ph.hashing_yield(eigs)) == pytest.approx(0.0, abs=1e-12)


def test_herald_probabilities_noise_free():
    pp, p1, p2 = ph.herald_probabilities(0.3, 0.2, 0.0)
    assert float(pp) == pytest.approx(0.03)
    assert float(p1) == 0.0 and float(p2) == 0.0
