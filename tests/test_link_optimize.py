import numpy as np
import pytest
import torch

from qfclink import DesignSpace, OperatingPoint, evaluate, evaluate_point, grid_search, optimise, random_search
from qfclink.scenarios import calibrate, heterogeneous_nominal, homogeneous_nominal, sample_hardware
from qfclink.stats import paired_comparison

STD_O = dict(target_nm=1350.0, pump_a=1.0, pump_b=1.0, filter_a_hz=20e6, filter_b_hz=100e6)
FILTERS_ONLY_O = DesignSpace(("filter_a_hz", "filter_b_hz"), {"target_nm": 1350.0, "pump_a": 1.0, "pump_b": 1.0})


def test_evaluate_outputs_are_physical():
    out = evaluate_point(heterogeneous_nominal(), 50.0, STD_O)
    for key in ("eta_a", "eta_b", "p_noise", "p_herald", "visibility"):
        assert 0.0 <= out[key] <= 1.0
    assert 0.25 <= out["fidelity"] <= 1.0
    assert out["ebit_rate"] >= 0.0


def test_rate_decreases_with_distance():
    hw = heterogeneous_nominal()
    rates = [evaluate_point(hw, d, STD_O)["rate_hz"] for d in (10, 50, 100)]
    assert rates[0] > rates[1] > rates[2]


def test_gradients_are_finite():
    params = torch.tensor([1400.0, 0.8, 0.8, 1e8, 1e8], dtype=torch.float64, requires_grad=True)
    out = evaluate(homogeneous_nominal(), OperatingPoint(*params), 50.0)
    out["objective"].backward()
    assert torch.isfinite(params.grad).all()


def test_codesign_beats_standard_on_nominal_hardware():
    hw = heterogeneous_nominal()
    std = optimise(hw, 50.0, FILTERS_ONLY_O, n_starts=4, steps=60, lr=0.15)
    full = optimise(hw, 50.0, DesignSpace(("target_nm", "pump_a", "pump_b", "filter_a_hz", "filter_b_hz")),
                    n_starts=4, steps=60, lr=0.15, warm_starts=[dict(STD_O)])
    assert float(full.ebit_rate) >= float(std.ebit_rate)


def test_batched_optimisation_over_distances():
    d = torch.tensor([[10.0], [80.0]], dtype=torch.float64)
    res = optimise(heterogeneous_nominal(), d, FILTERS_ONLY_O, n_starts=3, steps=20, n_problems=2)
    assert res.ebit_rate.shape == (2,)
    assert float(res.ebit_rate[0]) > float(res.ebit_rate[1])


def test_random_and_grid_search_run():
    hw = heterogeneous_nominal()
    space = DesignSpace(("target_nm", "pump_b"), {"pump_a": 1.0, "filter_a_hz": 20e6, "filter_b_hz": 100e6})
    rs = random_search(hw, 50.0, space, 500)
    gs = grid_search(hw, 50.0, space, [10, 5])
    assert gs.n_evals == 50 and rs.n_evals == 500
    assert 1260 <= rs.point()["target_nm"] <= 1625


def test_design_space_validation():
    with pytest.raises(ValueError):
        DesignSpace(("pump_a",), {})


def test_hardware_sampling_is_reproducible():
    a = sample_hardware(heterogeneous_nominal(), np.random.default_rng(5))
    b = sample_hardware(heterogeneous_nominal(), np.random.default_rng(5))
    assert a == b
    assert calibrate(a, np.random.default_rng(1), 0.1) != a


def test_paired_comparison():
    res = paired_comparison([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    assert res["median_rel_gain"] == pytest.approx(1.0)
    assert res["wins"] == 6 and res["wilcoxon_p"] < 0.05
