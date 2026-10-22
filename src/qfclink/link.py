"""End-to-end model of a two-node heralded-entanglement link with quantum
frequency conversion (QFC) and bandwidth matching.

Topology::

    emitter A --QFC_A--[compressor]--filter_A--fibre L/2--\
                                                          BSM station (50:50 BS + 2 SPDs)
    emitter B --QFC_B--------------- filter_B--fibre L/2--/

Both photons are converted to a common telecom target wavelength, filtered,
sent through L/2 of fibre each and interfered in a time-bin two-click Bell-state
measurement (Barrett-Kok type). The model returns the herald rate, the
heralded-state fidelity and the hashing-bound distillable-entanglement rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Union

import torch

from . import physics as ph

Number = Union[float, torch.Tensor]


@dataclass(frozen=True)
class Emitter:
    """A quantum node (emitter + spin memory) that emits a photon entangled
    with its spin."""

    name: str
    wavelength_nm: float
    linewidth_hz: float
    collection_eff: float  # probability a photon reaches the QFC input fibre
    t2_s: float  # spin-memory coherence time during the herald wait
    indistinguishability: float = 1.0  # ceiling on HOM visibility (e.g. spectral diffusion)


@dataclass(frozen=True)
class Converter:
    """PPLN-waveguide QFC stage (one per node; same technology on both)."""

    external_eff: float = 0.5  # coupling/transmission losses at unit internal efficiency
    raman_strength: float = 5.0e-5  # photons/mode at P_max inside the Raman band (n_th = 1)
    noise_floor: float = 1.0e-7  # photons/mode at P_max far from the Raman band
    base_filter_hz: float = 20.0e9  # broad grating filter after the converter
    raman_cutoff_cm1: float = 900.0


@dataclass(frozen=True)
class Detector:
    efficiency: float = 0.8
    dark_rate_hz: float = 10.0


@dataclass(frozen=True)
class Compressor:
    """Active bandwidth compressor (e.g. time lens / chirped-pump conversion)
    acting on photon A only."""

    efficiency: float = 0.4
    max_factor: float = 100.0
    mode_fidelity: float = 0.95  # residual temporal-shape mismatch (visibility ceiling)


@dataclass(frozen=True)
class LinkHardware:
    emitter_a: Emitter
    emitter_b: Emitter
    converter: Converter = field(default_factory=Converter)
    detector: Detector = field(default_factory=Detector)
    compressor: Compressor = field(default_factory=Compressor)
    detuning_hz: float = 0.0  # residual frequency offset between converted photons
    reset_time_s: float = 10.0e-6  # per-attempt node preparation time
    gate_factor: float = 3.0  # detection gate length in photon durations
    noise_scale: float = 1.0  # global multiplier on converter noise (robustness sweeps)

    def with_(self, **kwargs: object) -> "LinkHardware":
        """Return a copy with top-level fields replaced."""
        return replace(self, **kwargs)  # type: ignore[arg-type]


@dataclass
class OperatingPoint:
    """Design variables of the link (tensors allow batching / autograd)."""

    target_nm: Number
    pump_a: Number  # P/P_max of converter A
    pump_b: Number  # P/P_max of converter B
    filter_a_hz: Number
    filter_b_hz: Number
    compression: Number = 1.0  # bandwidth compression factor on photon A

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {k: ph.t(v) for k, v in self.__dict__.items()}


def evaluate(
    hw: LinkHardware,
    op: OperatingPoint,
    distance_km: Number,
    compress: bool = False,
) -> dict[str, torch.Tensor]:
    """Evaluate the link at operating point ``op`` and total length ``distance_km``.

    Returns a dict of tensors (broadcast over the batch shape of ``op``):
    ``eta_a``, ``eta_b`` (end-to-end detection probabilities), ``visibility``,
    ``p_noise`` (noise click probability per bin), ``p_herald``, ``fidelity``,
    ``yield`` (hashing bound per herald, unclamped), ``rate_hz`` (heralds/s),
    ``ebit_rate`` (clamped distillable-entanglement rate, ebit/s) and
    ``objective`` (unclamped ebit rate used for gradient ascent).
    """
    ea, eb = hw.emitter_a, hw.emitter_b
    conv, det, comp = hw.converter, hw.detector, hw.compressor
    target = ph.t(op.target_nm)
    length = ph.t(distance_km)
    kappa = ph.t(op.compression) if compress else ph.t(1.0)

    lw_a = ph.t(ea.linewidth_hz) / kappa
    lw_b = ph.t(eb.linewidth_hz)
    vis, trans_a, trans_b = ph.filtered_overlap(
        lw_a, lw_b, op.filter_a_hz, op.filter_b_hz, hw.detuning_hz
    )
    vis_ceiling = ea.indistinguishability * eb.indistinguishability
    if compress:
        vis_ceiling = vis_ceiling * comp.mode_fidelity
    vis = vis * vis_ceiling

    # Detection gate covers ``gate_factor`` photon durations of the longer photon.
    gate = hw.gate_factor * torch.maximum(
        ph.photon_duration_s(lw_a, op.filter_a_hz), ph.photon_duration_s(lw_b, op.filter_b_hz)
    )
    capture = 1.0 - math.exp(-hw.gate_factor)

    eta_fib = ph.fibre_transmission(target, length / 2.0)
    qfc_a = ph.conversion_efficiency(op.pump_a, conv.external_eff)
    qfc_b = ph.conversion_efficiency(op.pump_b, conv.external_eff)
    comp_eff = comp.efficiency if compress else 1.0
    downstream = eta_fib * det.efficiency
    eta_a = ea.collection_eff * qfc_a * comp_eff * trans_a * downstream * capture
    eta_b = eb.collection_eff * qfc_b * trans_b * downstream * capture

    # Pump-induced noise photons reaching the detectors, per time bin.
    def arm_noise(signal_nm: float, pump: Number, filt: Number) -> torch.Tensor:
        per_mode = ph.raman_noise_per_mode(
            signal_nm, target, conv.raman_strength, conv.noise_floor, conv.raman_cutoff_cm1
        )
        modes = ph.noise_bandwidth_hz(filt, conv.base_filter_hz) * gate
        return hw.noise_scale * per_mode * ph.t(pump) * modes * downstream

    mu = (
        arm_noise(ea.wavelength_nm, op.pump_a, op.filter_a_hz) * comp_eff
        + arm_noise(eb.wavelength_nm, op.pump_b, op.filter_b_hz)
        + 2.0 * det.dark_rate_hz * gate
    )
    p_noise = -torch.expm1(-mu)

    p_pair, p_one, p_two = ph.herald_probabilities(eta_a, eta_b, p_noise)
    wait = length * 1.0e3 / ph.C_FIBRE_M_PER_S
    dephasing = torch.exp(-wait * (1.0 / ea.t2_s + 1.0 / eb.t2_s))
    coherence = vis * dephasing
    eigs = ph.bell_diagonal(p_pair, p_one + p_two, coherence)
    yld = ph.hashing_yield(eigs)
    p_herald = p_pair + p_one + p_two
    attempt_rate = 1.0 / (hw.reset_time_s + wait)
    rate = attempt_rate * p_herald
    return {
        "eta_a": eta_a,
        "eta_b": eta_b,
        "visibility": vis,
        "p_noise": p_noise,
        "p_herald": p_herald,
        "fidelity": eigs[..., 0],
        "yield": yld,
        "rate_hz": rate,
        "ebit_rate": rate * torch.clamp(yld, min=0.0),
        "objective": rate * yld,
        "gate_s": gate,
        "attempt_rate_hz": attempt_rate * torch.ones_like(rate),
    }


def standard_point(hw: LinkHardware, target_nm: float) -> OperatingPoint:
    """Literature-style 'maximum-efficiency' operating point: both converters at
    full pump power and both filters matched to the narrower photon linewidth.
    (In the experiments the filters of the standard baseline are additionally
    tuned; this point is used as a warm start and sanity reference.)"""
    narrow = min(hw.emitter_a.linewidth_hz, hw.emitter_b.linewidth_hz)
    return OperatingPoint(target_nm, 1.0, 1.0, narrow, narrow)
