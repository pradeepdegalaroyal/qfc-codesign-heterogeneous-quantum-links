"""Representative hardware scenarios and hardware-uncertainty sampling.

The nominal values are *order-of-magnitude assumptions* chosen to be
representative of current platforms; they are not measurements of a specific
device. Every experiment that depends on them also sweeps or randomises them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .link import Compressor, Converter, Detector, Emitter, LinkHardware


def heterogeneous_nominal() -> LinkHardware:
    """Broadband solid-state emitter (~930 nm, GHz-class linewidth) linked to a
    narrowband atom-like memory (780 nm, ~10 MHz). Bandwidth ratio r = 50."""
    a = Emitter(
        name="broadband solid-state emitter (930 nm)",
        wavelength_nm=930.0,
        linewidth_hz=0.5e9,
        collection_eff=0.3,
        t2_s=10e-3,
        indistinguishability=0.90,
    )
    b = Emitter(
        name="narrowband atom-like memory (780 nm)",
        wavelength_nm=780.0,
        linewidth_hz=10e6,
        collection_eff=0.4,
        t2_s=100e-3,
        indistinguishability=0.98,
    )
    return LinkHardware(emitter_a=a, emitter_b=b)


def homogeneous_nominal() -> LinkHardware:
    """Two identical colour-centre-like nodes at 737 nm (reference case)."""
    a = Emitter(
        name="colour-centre node (737 nm)",
        wavelength_nm=737.0,
        linewidth_hz=0.1e9,
        collection_eff=0.4,
        t2_s=1.0,
        indistinguishability=0.95,
    )
    return LinkHardware(emitter_a=a, emitter_b=replace(a))


def _lognormal(rng: np.random.Generator, value: float, sigma: float) -> float:
    return float(value * np.exp(rng.normal(0.0, sigma)))


def _bounded(rng: np.random.Generator, value: float, rel: float, hi: float = 1.0) -> float:
    return float(min(hi, value * rng.uniform(1 - rel, 1 + rel)))


def sample_hardware(nominal: LinkHardware, rng: np.random.Generator) -> LinkHardware:
    """Draw one 'true' hardware instance around the nominal parameters.

    Noise processes (Raman strength, noise floor, dark counts) are log-normal
    with sigma = 0.5 (factor ~1.6); linewidths and T2 log-normal with
    sigma = 0.2-0.3; efficiencies uniform +-20 %; a residual detuning of up to
    0.3 narrow linewidths.
    """
    def emitter(e: Emitter) -> Emitter:
        return replace(
            e,
            linewidth_hz=_lognormal(rng, e.linewidth_hz, 0.2),
            collection_eff=_bounded(rng, e.collection_eff, 0.2),
            t2_s=_lognormal(rng, e.t2_s, 0.3),
            indistinguishability=_bounded(rng, e.indistinguishability, 0.03),
        )

    conv = nominal.converter
    narrow = min(nominal.emitter_a.linewidth_hz, nominal.emitter_b.linewidth_hz)
    return replace(
        nominal,
        emitter_a=emitter(nominal.emitter_a),
        emitter_b=emitter(nominal.emitter_b),
        converter=replace(
            conv,
            external_eff=_bounded(rng, conv.external_eff, 0.2),
            raman_strength=_lognormal(rng, conv.raman_strength, 0.5),
            noise_floor=_lognormal(rng, conv.noise_floor, 0.5),
            raman_cutoff_cm1=float(conv.raman_cutoff_cm1 + rng.normal(0.0, 30.0)),
        ),
        detector=replace(
            nominal.detector,
            efficiency=_bounded(rng, nominal.detector.efficiency, 0.1),
            dark_rate_hz=_lognormal(rng, nominal.detector.dark_rate_hz, 0.5),
        ),
        detuning_hz=float(abs(rng.normal(0.0, 0.15 * narrow))),
    )


def calibrate(true_hw: LinkHardware, rng: np.random.Generator, rel_error: float = 0.1) -> LinkHardware:
    """Simulate imperfect calibration: every parameter of ``true_hw`` is
    estimated with a multiplicative log-normal error of ``rel_error``."""
    def est(x: float) -> float:
        return _lognormal(rng, x, rel_error)

    def emitter(e: Emitter) -> Emitter:
        return replace(
            e,
            linewidth_hz=est(e.linewidth_hz),
            collection_eff=min(1.0, est(e.collection_eff)),
            t2_s=est(e.t2_s),
            indistinguishability=min(1.0, est(e.indistinguishability)),
        )

    c = true_hw.converter
    return replace(
        true_hw,
        emitter_a=emitter(true_hw.emitter_a),
        emitter_b=emitter(true_hw.emitter_b),
        converter=replace(
            c,
            external_eff=min(1.0, est(c.external_eff)),
            raman_strength=est(c.raman_strength),
            noise_floor=est(c.noise_floor),
            raman_cutoff_cm1=c.raman_cutoff_cm1 * float(np.exp(rng.normal(0, 0.03))),
        ),
        detector=replace(
            true_hw.detector,
            efficiency=min(1.0, est(true_hw.detector.efficiency)),
            dark_rate_hz=est(true_hw.detector.dark_rate_hz),
        ),
        detuning_hz=est(true_hw.detuning_hz + 1.0),
    )


__all__ = [
    "Compressor",
    "Converter",
    "Detector",
    "Emitter",
    "LinkHardware",
    "heterogeneous_nominal",
    "homogeneous_nominal",
    "sample_hardware",
    "calibrate",
]
