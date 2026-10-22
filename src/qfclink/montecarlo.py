"""Event-by-event Monte-Carlo of the two-click time-bin heralding protocol.

Independent check of :func:`qfclink.physics.herald_probabilities` and
:func:`qfclink.physics.bell_diagonal`: each attempt samples the time bins of
both photons, their detection, and Poissonian noise clicks per bin, applies
threshold-detector logic, and assigns each herald the conditional state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    n_attempts: int
    n_heralds: int
    p_herald: float
    p_herald_se: float
    fidelity: float
    fidelity_se: float


def simulate_heralding(
    eta_a: float,
    eta_b: float,
    mu_noise: float,
    coherence: float,
    n_attempts: int,
    rng: np.random.Generator,
    chunk: int = 2_000_000,
) -> MCResult:
    """Simulate ``n_attempts`` heralding attempts.

    ``mu_noise`` is the mean number of noise (pump-induced + dark) clicks per
    time bin; ``coherence`` the effective two-photon coherence of genuine pairs.
    """
    heralds = 0
    fid_sum = 0.0
    fid_sq = 0.0
    f_pair = 0.5 * (1 + coherence)
    f_mixed = 0.25
    done = 0
    while done < n_attempts:
        n = min(chunk, n_attempts - done)
        bin_a = rng.integers(0, 2, n)
        bin_b = rng.integers(0, 2, n)
        det_a = rng.random(n) < eta_a
        det_b = rng.random(n) < eta_b
        noise_e = rng.poisson(mu_noise, n) > 0
        noise_l = rng.poisson(mu_noise, n) > 0
        click_e = noise_e | (det_a & (bin_a == 0)) | (det_b & (bin_b == 0))
        click_l = noise_l | (det_a & (bin_a == 1)) | (det_b & (bin_b == 1))
        herald = click_e & click_l
        genuine = det_a & det_b & (bin_a != bin_b)
        n_h = int(herald.sum())
        n_g = int((herald & genuine).sum())
        heralds += n_h
        fid_sum += n_g * f_pair + (n_h - n_g) * f_mixed
        fid_sq += n_g * f_pair**2 + (n_h - n_g) * f_mixed**2
        done += n
    p = heralds / n_attempts
    mean_f = fid_sum / max(heralds, 1)
    var_f = max(fid_sq / max(heralds, 1) - mean_f**2, 0.0)
    return MCResult(
        n_attempts=n_attempts,
        n_heralds=heralds,
        p_herald=p,
        p_herald_se=float(np.sqrt(p * (1 - p) / n_attempts)),
        fidelity=mean_f,
        fidelity_se=float(np.sqrt(var_f / max(heralds, 1))),
    )
