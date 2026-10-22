"""First-principles Fock-space simulation of the time-bin Bell-state measurement.

Used to validate the closed-form result that a genuine photon pair heralds a
spin-spin state with fidelity ``(1 + V)/2`` where ``V = Tr(rho_A rho_B)`` is the
two-photon (HOM) visibility of the photons' internal (spectral) states.

Each node emits ``(|0> a^dag_E + |1> a^dag_L)/sqrt(2)`` (spin 0 <-> early time
bin). Photons carry a K-dimensional internal (spectral) degree of freedom that
the detectors do not resolve. A 50:50 beam splitter maps input ports
(A, B) -> outputs (c, d). Two-photon amplitudes are tracked exactly, including
bosonic symmetrisation; threshold detectors project onto one photon in
(det1, early) and one in (det2, late), tracing out the internal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

PSI_PLUS = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_MINUS = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)

MixedMode = Sequence[tuple[float, np.ndarray]]  # [(probability, internal amplitude vector)]


@dataclass
class BSMResult:
    """Heralded spin states for the four click patterns (c/d at early, c/d at late)."""

    probabilities: dict[tuple[str, str], float]
    states: dict[tuple[str, str], np.ndarray]
    fidelities: dict[tuple[str, str], float]

    @property
    def herald_probability(self) -> float:
        return float(sum(self.probabilities.values()))

    @property
    def mean_fidelity(self) -> float:
        p = self.probabilities
        return float(sum(p[k] * self.fidelities[k] for k in p) / self.herald_probability)


def _mode(det: int, tbin: int, k: int, dim: int) -> int:
    return (det * 2 + tbin) * dim + k


def simulate_bsm(mode_a: MixedMode, mode_b: MixedMode) -> BSMResult:
    """Exact two-photon simulation of the heralded entanglement swap.

    ``mode_a`` / ``mode_b`` are ensembles of internal states (spectral
    decompositions of the single-photon spectral density matrices).
    """
    dim = len(mode_a[0][1])
    n_modes = 4 * dim
    patterns = [(d1, d2) for d1 in ("c", "d") for d2 in ("c", "d")]
    rho = {p: np.zeros((4, 4), dtype=complex) for p in patterns}
    for pa, phi_a in mode_a:
        phi_a = np.asarray(phi_a, dtype=complex) / np.linalg.norm(phi_a)
        for pb, phi_b in mode_b:
            phi_b = np.asarray(phi_b, dtype=complex) / np.linalg.norm(phi_b)
            # amp[spin_a, spin_b] = two-photon amplitude matrix over output modes
            amp = np.zeros((2, 2, n_modes, n_modes), dtype=complex)
            for sa in (0, 1):
                ca = np.zeros(n_modes, dtype=complex)
                for k in range(dim):  # port A -> (c + d)/sqrt2
                    ca[_mode(0, sa, k, dim)] += phi_a[k] / np.sqrt(2)
                    ca[_mode(1, sa, k, dim)] += phi_a[k] / np.sqrt(2)
                for sb in (0, 1):
                    cb = np.zeros(n_modes, dtype=complex)
                    for k in range(dim):  # port B -> (c - d)/sqrt2
                        cb[_mode(0, sb, k, dim)] += phi_b[k] / np.sqrt(2)
                        cb[_mode(1, sb, k, dim)] -= phi_b[k] / np.sqrt(2)
                    amp[sa, sb] = 0.5 * np.outer(ca, cb)  # 1/2 from the two spin superpositions
            for d1, d2 in patterns:
                i1 = 0 if d1 == "c" else 1
                i2 = 0 if d2 == "c" else 1
                for k1 in range(dim):
                    for k2 in range(dim):
                        m1, m2 = _mode(i1, 0, k1, dim), _mode(i2, 1, k2, dim)
                        # <1_m1 1_m2| a^dag_x a^dag_y |0> = delta(x,m1)delta(y,m2) + delta(x,m2)delta(y,m1)
                        v = (amp[:, :, m1, m2] + amp[:, :, m2, m1]).reshape(4)
                        rho[(d1, d2)] += pa * pb * np.outer(v, v.conj())
    probs, states, fids = {}, {}, {}
    for p in patterns:
        prob = float(np.real(np.trace(rho[p])))
        probs[p] = prob
        state = rho[p] / prob
        states[p] = state
        target = PSI_PLUS if p[0] == p[1] else PSI_MINUS
        fids[p] = float(np.real(target.conj() @ state @ target))
    return BSMResult(probs, states, fids)


def pure_mode_pair(overlap: float, dim: int = 2) -> tuple[MixedMode, MixedMode]:
    """Two pure internal states with ``|<a|b>| = overlap``."""
    a = np.zeros(dim)
    a[0] = 1.0
    b = np.zeros(dim)
    b[0], b[1] = overlap, np.sqrt(max(0.0, 1 - overlap**2))
    return [(1.0, a)], [(1.0, b)]


def lorentzian_mode_vector(linewidth: float, filter_fwhm: float, detuning: float, grid: np.ndarray) -> np.ndarray:
    """Discretised spectral amplitude of a Lorentzian photon after a Lorentzian
    filter, ``sqrt(2a)/(a - i(w-d)) * f/(f - i(w-d))`` on the angular grid."""
    a, f, d = np.pi * linewidth, np.pi * filter_fwhm, 2 * np.pi * detuning
    w = grid - d
    return np.sqrt(2 * a) / (a - 1j * w) * f / (f - 1j * w)
