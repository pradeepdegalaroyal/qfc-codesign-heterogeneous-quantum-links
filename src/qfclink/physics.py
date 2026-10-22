"""Closed-form physics of a frequency-converted single-photon link.

All functions accept Python floats or ``torch`` tensors (float64) and are
differentiable, so the full link model can be optimised with autograd.

Conventions
-----------
* Linewidths and filter bandwidths are FWHM in Hz.
* A transform-limited photon from a spontaneously emitting two-level system has
  a one-sided exponential temporal amplitude ``sqrt(2a) exp(-a t)`` with
  amplitude decay rate ``a = pi * FWHM`` (rad/s), i.e. a Lorentzian spectrum.
* A Lorentzian filter (e.g. a filter cavity) has amplitude response
  ``f / (f - i w)`` with ``f = pi * FWHM_filter``.
* Wavelengths are vacuum wavelengths in nm; Raman shifts are in cm^-1.
"""

from __future__ import annotations

import math
from typing import Union

import torch

Number = Union[float, torch.Tensor]

DTYPE = torch.float64
C_FIBRE_M_PER_S = 2.04e8  # group velocity in standard silica fibre (n_g ~ 1.47)
KT_CM1_ROOM = 207.2  # k_B T / (h c) at 298 K, in cm^-1


def t(x: Number) -> torch.Tensor:
    """Convert ``x`` to a float64 tensor without breaking the autograd graph."""
    if isinstance(x, torch.Tensor):
        return x if x.dtype in (DTYPE, torch.complex128) else x.to(DTYPE)
    return torch.as_tensor(x, dtype=DTYPE)


# --------------------------------------------------------------------------
# Spectral-temporal mode overlap (bandwidth matching)
# --------------------------------------------------------------------------
def filtered_overlap(
    linewidth_a_hz: Number,
    linewidth_b_hz: Number,
    filter_a_hz: Number,
    filter_b_hz: Number,
    detuning_hz: Number = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-photon interference visibility of two Lorentzian-filtered photons.

    Photon A (linewidth ``linewidth_a_hz``) passes a Lorentzian filter of FWHM
    ``filter_a_hz``; photon B likewise with ``filter_b_hz``. Photon B and its
    filter are detuned by ``detuning_hz`` from photon A. The overlap integral
    is evaluated in closed form by contour integration::

        <B'|A'> = sqrt(4ab) f_A f_B (a + b' + f_A + f_B')
                  / ((a + b')(a + f_B')(b' + f_A)(f_A + f_B'))

    with ``b' = b - i delta`` and ``f_B' = f_B - i delta``. The expression has no
    removable singularities, which keeps autograd stable.

    Returns
    -------
    visibility, transmission_a, transmission_b
        ``visibility = |<B'|A'>|^2 / (T_A T_B)`` is the HOM visibility of the
        *normalised* filtered photons; ``T_X`` is the filter transmission.
    """
    a = math.pi * t(linewidth_a_hz)
    b = math.pi * t(linewidth_b_hz)
    fa = math.pi * t(filter_a_hz)
    fb = math.pi * t(filter_b_hz)
    d = 2.0 * math.pi * t(detuning_hz)
    bc = torch.complex(b, -d * torch.ones_like(b))
    fbc = torch.complex(fb, -d * torch.ones_like(fb))
    num = a + bc + fa + fbc
    den = (a + bc) * (a + fbc) * (bc + fa) * (fa + fbc)
    overlap = torch.sqrt(4.0 * a * b) * fa * fb * num / den
    trans_a = fa / (a + fa)
    trans_b = fb / (b + fb)
    vis = (overlap.real**2 + overlap.imag**2) / (trans_a * trans_b)
    return vis, trans_a, trans_b


def photon_duration_s(linewidth_hz: Number, filter_hz: Number) -> torch.Tensor:
    """Characteristic duration (s) of a filtered photon: sum of the intensity
    lifetimes of the emitter and of the filter response."""
    return 1.0 / (2.0 * math.pi * t(linewidth_hz)) + 1.0 / (2.0 * math.pi * t(filter_hz))


def noise_bandwidth_hz(filter_hz: Number, base_filter_hz: Number) -> torch.Tensor:
    """Equivalent noise bandwidth (Hz) of a Lorentzian filter in series with a
    broader Lorentzian 'base' filter (e.g. the converter's grating filter)."""
    f = t(filter_hz)
    g = t(base_filter_hz)
    return 0.5 * math.pi * f * g / (f + g)


# --------------------------------------------------------------------------
# Fibre transmission
# --------------------------------------------------------------------------
def fibre_attenuation_db_per_km(wavelength_nm: Number) -> torch.Tensor:
    """Empirical attenuation of low-water-peak standard single-mode fibre.

    Rayleigh scattering (0.8 dB/km um^4 / lambda^4) + 0.01 dB/km excess
    + infrared absorption tail. Gives ~0.28 dB/km at 1310 nm and ~0.17 dB/km
    at 1550 nm, typical of G.652.D-class fibre. Valid for ~1260-1650 nm.
    """
    lam_um = t(wavelength_nm) / 1000.0
    return 0.8 / lam_um**4 + 0.01 + 6.0e11 * torch.exp(-48.0 / lam_um)


def fibre_transmission(wavelength_nm: Number, length_km: Number) -> torch.Tensor:
    """Power transmission of ``length_km`` of fibre at ``wavelength_nm``."""
    return 10.0 ** (-fibre_attenuation_db_per_km(wavelength_nm) * t(length_km) / 10.0)


# --------------------------------------------------------------------------
# Quantum frequency conversion (PPLN difference-frequency generation)
# --------------------------------------------------------------------------
def pump_wavelength_nm(signal_nm: Number, target_nm: Number) -> torch.Tensor:
    """Pump wavelength for DFG ``1/lambda_t = 1/lambda_s - 1/lambda_p``."""
    return 1.0 / (1.0 / t(signal_nm) - 1.0 / t(target_nm))


def raman_shift_cm1(signal_nm: Number, target_nm: Number) -> torch.Tensor:
    """Signed shift (cm^-1) of the target band relative to the pump.

    Positive: target is blue of the pump (anti-Stokes side).
    Negative: target is red of the pump (Stokes side).
    """
    k_t = 1.0e7 / t(target_nm)
    k_p = 1.0e7 / pump_wavelength_nm(signal_nm, target_nm)
    return k_t - k_p


def conversion_efficiency(pump_fraction: Number, external_eff: Number) -> torch.Tensor:
    """Device efficiency ``eta_ext * sin^2(pi/2 * sqrt(P/P_max))``.

    ``pump_fraction = P / P_max`` where ``P_max`` gives unit internal efficiency.
    """
    p = t(pump_fraction)
    return t(external_eff) * torch.sin(0.5 * math.pi * torch.sqrt(p)) ** 2


def raman_noise_per_mode(
    signal_nm: Number,
    target_nm: Number,
    raman_strength: Number,
    noise_floor: Number,
    cutoff_cm1: float = 900.0,
    edge_cm1: float = 50.0,
    min_shift_cm1: float = 10.0,
    kt_cm1: float = KT_CM1_ROOM,
) -> torch.Tensor:
    """Pump-induced noise photons per temporal-spectral mode at full pump.

    Phenomenological model of spontaneous Raman scattering of the strong pump
    in the nonlinear crystal: a flat Raman response that rolls off above the
    highest phonon frequency (``cutoff_cm1``), weighted by the Bose-Einstein
    occupation ``n_th`` on the anti-Stokes side and ``n_th + 1`` on the Stokes
    side, plus a wavelength-independent floor (e.g. residual SPDC / leakage).
    Values are photons per mode (per unit time-bandwidth product) at the
    converter output for ``P = P_max``; noise scales linearly with pump power.
    """
    shift = raman_shift_cm1(signal_nm, target_nm)
    mag = torch.sqrt(shift**2 + min_shift_cm1**2)
    response = torch.sigmoid((cutoff_cm1 - mag) / edge_cm1)
    n_th = 1.0 / torch.expm1(mag / kt_cm1)
    occupation = torch.where(shift >= 0, n_th, n_th + 1.0)
    return t(raman_strength) * response * occupation + t(noise_floor)


# --------------------------------------------------------------------------
# Heralded two-photon (time-bin) entanglement swapping
# --------------------------------------------------------------------------
def herald_probabilities(
    eta_a: Number, eta_b: Number, p_noise: Number
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-attempt herald probabilities of the time-bin two-click protocol.

    The station heralds when both the early and the late time bin register a
    click. ``eta_X`` is the end-to-end detection probability of photon X and
    ``p_noise`` the probability of at least one noise/dark click per bin.

    Returns ``(P_pair, P_one_noise, P_two_noise)``: herald from a genuine
    photon pair in opposite bins; from one photon plus a noise click (including
    a same-bin photon pair plus noise); and from noise in both bins.
    """
    ea, eb, pn = t(eta_a), t(eta_b), t(p_noise)
    p_pair = 0.5 * ea * eb
    p_one = (ea * (1 - eb) + eb * (1 - ea) + 0.5 * ea * eb) * pn
    p_two = (1 - ea) * (1 - eb) * pn**2
    return p_pair, p_one, p_two


def bell_diagonal(
    p_pair: Number, p_noise_events: Number, coherence: Number
) -> torch.Tensor:
    """Bell-diagonal eigenvalues of the heralded spin-spin state.

    Genuine pairs give ``(1+c)/2 |Psi_target> + (1-c)/2 |Psi_other>`` where
    ``c`` is the effective two-photon coherence (HOM visibility times memory
    dephasing). Noise-involved heralds give the maximally mixed state.
    Returns a tensor with last dimension 4: (target, other, Phi+, Phi-).
    """
    pp, pn, c = t(p_pair), t(p_noise_events), t(coherence)
    total = pp + pn
    w = pp / total
    mix = (1 - w) / 4.0
    lam1 = w * (1 + c) / 2 + mix
    lam2 = w * (1 - c) / 2 + mix
    return torch.stack([lam1, lam2, mix, mix], dim=-1)


def hashing_yield(eigs: torch.Tensor) -> torch.Tensor:
    """Hashing-bound distillable-entanglement yield ``1 - H(lambda)`` (ebit per
    heralded pair) of a Bell-diagonal state. May be negative (no yield)."""
    p = torch.clamp(eigs, min=1e-300)
    return 1.0 + torch.sum(p * torch.log2(p), dim=-1)
