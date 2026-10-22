# Method

## 1. Link model

```
emitter A ─QFC_A(P_A)─[compressor κ]─filter Γ_fA─ fibre L/2 ─┐
                                                               ├─ 50:50 BS + 2 SPDs (time-bin two-click BSM)
emitter B ─QFC_B(P_B)──────────────── filter Γ_fB─ fibre L/2 ─┘
```

Each node emits a photon whose time bin (early/late) is entangled with its spin.
Both photons are converted to one telecom wavelength λ_t, filtered, sent L/2 each to a
middle station and interfered. A herald requires one click in the early bin and one in the
late bin, as in a Barrett–Kok-type scheme. All code lives in `src/qfclink/`.

| Component | Model (file) |
|---|---|
| Photon modes | One-sided exponential (Lorentzian) photons with FWHM Γ; Lorentzian filters. Closed-form overlap `<B'|A'> = √(4ab) f_A f_B (a+b'+f_A+f_B') / [(a+b')(a+f_B')(b'+f_A)(f_A+f_B')]` with b' = b − iδ (detuning) from contour integration. V = \|<B'|A'>\|²/(T_A T_B), T_X = f_X/(x+f_X) (`physics.filtered_overlap`). |
| QFC efficiency | η = η_ext · sin²(π/2 √(P/P_max)) (`physics.conversion_efficiency`). |
| QFC noise | DFG pump λ_p = (1/λ_s − 1/λ_t)⁻¹. Noise photons per temporal-spectral mode = ν_R · g(\|Δ\|) · n(Δ) + ν_floor. Δ is the pump-to-target Raman shift, g is a flat Raman response rolling off above a cutoff Ω_c (nominal 900 cm⁻¹), and n = n_th on the anti-Stokes side or n_th + 1 on the Stokes side at 298 K. Noise ∝ P. Accepted modes = equivalent noise bandwidth × gate (`physics.raman_noise_per_mode`). **Phenomenological; Ω_c and ν_R are randomised and swept.** |
| Fibre | Empirical G.652.D-like attenuation: Rayleigh (0.8 dB km⁻¹ µm⁴/λ⁴) + 0.01 dB/km + IR tail (≈0.28 dB/km at 1310 nm, 0.17 at 1550 nm). |
| Detection | Efficiency 0.8, dark rate 10 Hz, gate = 3 photon durations, capture 1 − e⁻³. |
| Heralding | P_pair = ½η_Aη_B; P_1noise = [η_A(1−η_B)+η_B(1−η_A)+½η_Aη_B]·p_n; P_2noise = (1−η_A)(1−η_B)p_n². Genuine pairs give Bell-diagonal (1±c)/2 with c = V · e^{−t_wait(1/T2_A+1/T2_B)}; noise heralds give 𝟙/4 (`physics.herald_probabilities`, `bell_diagonal`). |
| Figure of merit | D_H = R_att · P_herald · max(0, 1 − S(ρ)) ebit/s, R_att = 1/(10 µs + L/c_fibre) (hashing bound). |
| Active bandwidth compression | Photon A's linewidth is divided by κ ∈ [1, 100], with efficiency η_BC (nominal 0.4) and a mode-shape visibility ceiling of 0.95. Noise per mode is conserved and attenuated by η_BC. |

**Nominal heterogeneous scenario** (`scenarios.heterogeneous_nominal`):
- A: 930 nm, Γ = 0.5 GHz, collection 0.3, T2 = 10 ms, visibility ceiling 0.90.
- B: 780 nm, Γ = 10 MHz, collection 0.4, T2 = 100 ms, ceiling 0.98. The bandwidth ratio is r = 50.
- QFC: η_ext = 0.5, ν_R = 5×10⁻⁵, ν_floor = 10⁻⁷ photons/mode at P_max, 20 GHz base filter.

**Homogeneous reference:** two 737 nm nodes with Γ = 0.1 GHz.

## 2. Model validation (E1)

1. **Fock-space simulation** (`fock.py`). The two-photon state is tracked exactly in
   2 outputs × 2 time bins × K internal (spectral) modes, with bosonic symmetrisation. This
   checks that a genuine pair heralds F = (1 + Tr ρ_Aρ_B)/2 with probability ½, for both
   pure and mixed spectral states.
2. **Numerical quadrature** of the filtered spectral overlap, with detuning, for random
   parameter draws spanning 1 MHz to 20 GHz.
3. **Event Monte Carlo** (`montecarlo.py`): samples time bins, detections and Poissonian noise
   clicks with threshold detectors, giving herald probability and fidelity in 8 regimes.

## 3. Compared strategies

All strategies use the same model, bounds and optimiser.

| Name | λ_t | Pump powers | Filters | Role |
|---|---|---|---|---|
| `std_C` | 1550 nm (min-loss C-band) | P_max (max efficiency) | optimised | literature-style baseline 1 |
| `std_O` | 1350 nm (O-band, as in Knaut et al. 2024) | P_max | optimised | literature-style baseline 2 (the standard quantum baseline) |
| `abl_pump_C/O` | fixed C / O | optimised | optimised | ablation: pump only |
| `abl_wavelength` | optimised | P_max | optimised | ablation: wavelength only |
| `codesign` | optimised | optimised | optimised | **proposed** |
| `codesign_comp` | optimised | optimised | optimised + κ | proposed + active bandwidth compression |

The baselines are deliberately strong. Their filters are tuned by the same optimiser rather
than set by a rule of thumb. For a 780 nm node, the C-band baseline is close to pump
degeneracy, so the O-band baseline is the fair literature-standard reference.

**Classical baselines.** This is a quantum-network design problem, so the classical
comparators are the *design procedures* rather than a classical communication protocol.
- *Derivative-free random search* gets 3× the gradient budget in model evaluations.
- A *dense grid* uses 25 × 10 × 10 × 12 × 12 = 360,000 points.

Both are compared with gradient-based co-design (multi-start Adam, 16 starts × 300 steps,
warm-started from the two standard points) through the differentiable model (PyTorch autograd).

**Information levels.** These test realism beyond pure simulation.
- *oracle*: optimised on the true hardware instance (upper bound).
- *calibrated*: optimised on a calibrated estimate, where every parameter carries 10 %
  log-normal error. This is the realistic setting.
- *nominal*: optimised on the nominal model only, with no calibration.

Every operating point is then **evaluated on the true hardware instance**.

## 4. Hardware seeds and statistics

Each seed draws a "true" device around the nominal values (`scenarios.sample_hardware`):
- Raman strength, noise floor and dark rate: log-normal, σ = 0.5.
- Raman cutoff: ±30 cm⁻¹.
- Linewidths: σ = 0.2. T2: σ = 0.3.
- Efficiencies: ±20 %.
- Residual detuning: half-normal, 0.15 Γ_B.

The study uses 10 seeds (0–9) and distances of 10, 25, 50, 75 and 100 km.

**Test.** For each distance, a paired two-sided Wilcoxon signed-rank test on the per-seed
log-ratios of D_H (method vs baseline). We report the median relative gain with a 10,000-sample
percentile-bootstrap 95 % CI, and wins/losses. Results are given as mean ± std over seeds.

## 5. Ablation, scaling and robustness

- **E3 scaling map.** D_H over λ_t (1262–1623 nm, 122 points) × distance (5–150 km, 30 points),
  with pumps and filters optimised at every cell, for both the heterogeneous and the
  homogeneous scenario.
- **E4 ablation.** The pump-only, wavelength-only, full and full+compression strategies above.
- **E5 robustness** (5 seeds, 50 km). Sweeps of the converter noise scale (0.01–100×), dark
  rate (1–1000 Hz), residual detuning (0–2 Γ_B), memory T2 (0.01–10×) and the *true* Raman
  cutoff (600–1200 cm⁻¹). The last sweep is a model-misspecification test: the
  `codesign (nominal model)` curve is designed assuming 900 cm⁻¹.
- **E6 bandwidth conversion** (50 km). Bandwidth ratio r = Γ_A/Γ_B from 1 to 1000 (13 points)
  × compressor efficiency η_BC from 0.005 to 1 (14 log-spaced points). Filter co-design is
  compared with compression co-design, and the crossover η*(r) is extracted.

## 6. Reproducibility

`python experiments/run_all.py` regenerates every file in `results/` and `figures/`.
`--quick` runs a < 2 min smoke version that writes to `quick_outputs/`. All randomness is
seeded: NumPy generators per seed and a `torch.Generator` for optimiser initialisation.
