# Research gap: which frequency/bandwidth-conversion operating point should a heterogeneous quantum link use?

## 1. State of the art (from the provided literature)

**Quantum networks need converters.** Quantum repeaters are the central missing element of a
quantum internet. They exist to overcome loss and noise in the channels that connect memory
nodes (Azuma et al., 2023, doi:10.1103/RevModPhys.95.045006). Most good memories and emitters
work at visible or near-infrared wavelengths, while low-loss fibre works in the telecom O-to-L
bands. Knaut et al. (2024, doi:10.1038/s41586-024-07252-z) showed how this gap is closed in
practice. They entangled two silicon-vacancy (SiV) nuclear-spin memories through 40 km of
fibre spools and a 35 km deployed loop by integrating *bidirectional quantum frequency
conversion (QFC) to 1350 nm*. That demonstration used two **identical** nodes.

**Lithium niobate is the workhorse.** Periodically poled lithium niobate (PPLN) gives
χ(2) conversion from the microwave to the UV, and a mature wafer industry supports it
(Boes et al., 2023, doi:10.1126/science.abj4396). Fabrication quality limits device losses
(Ma et al., 2027, doi:10.1016/j.triboint.2026.112616). Integrated photonic chips are making
these components compact and stable (Luo et al., 2023, doi:10.1038/s41377-023-01173-8).

**Noise, not only loss, sets the distance limit.** Record QKD systems reach their distances
mainly by suppressing noise counts. Twin-field QKD over 1002 km needed system noise of
about 0.02 Hz from ultra-low-noise SNSPDs and dual-band phase estimation (Liu et al., 2023,
doi:10.1103/PhysRevLett.130.210801). High-rate QKD relies on SNSPDs with very high counting
rates (Li et al., 2023, doi:10.1038/s41566-023-01166-4). A strong pump inside a converter
generates broadband Raman and parametric noise photons, so the same principle applies to QFC.

**Nodes are heterogeneous.** Candidate nodes include semiconductor spin qubits with spin-photon
interfaces (Burkard et al., 2023, doi:10.1103/RevModPhys.95.025003), emitters in waveguide-QED
structures (Sheremet et al., 2023, doi:10.1103/RevModPhys.95.015002), trapped ions (Moses et al.,
2023, doi:10.1103/PhysRevX.13.041052) and new colour centres such as V_Si in SiC (Zhao et al.,
2027, doi:10.1063/5.0326538). These platforms emit at different wavelengths, and their
linewidths range from MHz to GHz. Network-native integration of heterogeneous resources is
listed as an open challenge for multiparty protocols (Mouaji & Al-Kuwari, 2027,
doi:10.1016/j.cosrev.2026.101047). Modular quantum computing gives a direct analogy: tools that
ignore the asymmetry between low-error local operations and high-error links lose up to 30 %
in fidelity (Du et al., 2027, doi:10.1016/j.future.2026.108830). The cost of non-local
operations dominates distributed algorithms (Cardama et al., 2027, doi:10.1016/j.future.2026.108815).

Foundational background (outside the provided Scopus set): two-photon interference
(Hong, Ou & Mandel, 1987, doi:10.1103/PhysRevLett.59.2044); quantum frequency conversion
(Kumar, 1990, doi:10.1364/OL.15.001476); two-photon heralded entanglement of matter qubits
(Barrett & Kok, 2005, doi:10.1103/PhysRevA.71.060310); hashing-bound entanglement distillation
(Bennett et al., 1996, doi:10.1103/PhysRevA.54.3824).

## 2. The gap

To link two *different* nodes by two-photon interference, two conditions must hold:

1. **Frequency matching.** Both photons are converted to one common telecom wavelength λ_t.
2. **Bandwidth matching.** The broadband photon is filtered, or actively compressed, until
   it is indistinguishable from the narrowband one.

Every choice trades **efficiency against noise against indistinguishability**:

- λ_t sets the fibre loss. Through the energy-conservation constraint it also sets the
  pump wavelength of *each* converter. The pump-to-target Raman shift then sets the noise.
  One λ_t can be quiet for one node and close to pump degeneracy for the other. For example,
  780 nm → 1560 nm needs a 1560 nm pump.
- Pump power raises conversion efficiency as sin²(π/2·√(P/P_max)), but noise grows linearly in P.
- A narrow filter raises the HOM visibility but cuts transmission. Its noise bandwidth and
  the length of the detection gate are coupled.

Experiments usually choose a sensible operating point one component at a time: maximum
conversion efficiency, a band chosen by convention (O-band as in Knaut et al. 2024, or the
C-band minimum-loss window), and a filter matched to the narrow linewidth. In the provided
literature we found **no end-to-end, noise-aware treatment that jointly chooses target
wavelength, pump powers and filter bandwidths for a heterogeneous pair**. We also found none
that scores those choices with an entanglement figure of merit that accounts for fidelity,
none that tests whether the choice survives imperfect calibration, and none that maps where
active bandwidth compression beats passive filtering. This is the kind of quantitative design
rule a technical review and roadmap of conversion technologies needs.

## 3. Research question and hypotheses

**RQ.** For a heralded two-node link between a broadband solid-state emitter (930 nm,
~0.5 GHz) and a narrowband atom-like memory (780 nm, ~10 MHz), how much distillable
entanglement does noise-aware *joint* co-design of (λ_t, P_A, P_B, Γ_fA, Γ_fB) gain over
literature-style operating points? Does that gain survive realistic hardware uncertainty?

All hypotheses and thresholds below were fixed before the full experiment run.

- **H1 (primary).** At 100 km, co-design based on calibrated parameters (10 % calibration
  error) raises the hashing-bound distillable-entanglement rate by a **median ≥ 10 %** over the
  *better* of two literature-standard baselines (full pump power, tuned filters, λ_t = 1350 nm
  or 1550 nm). The comparison is paired over 10 random hardware instances, and the two-sided
  Wilcoxon signed-rank test must give p < 0.05.
  *Falsified if* the median gain is < 10 % or p ≥ 0.05.
- **H2 (robustness).** Co-design using **only the nominal model** (no calibration of the
  actual device) still beats the best standard baseline on ≥ 8/10 hardware instances at
  50 km and at 100 km.
- **H3 (bandwidth conversion).** At 50 km, active bandwidth compression beats optimised
  passive filtering if and only if its efficiency exceeds a crossover η*(r). For r ≥ 10,
  η*(r) lies within a factor of 2 of the matched-filter transmission 2/(1+r).
  *Falsified if* the measured crossover lies outside [1/(1+r), 4/(1+r)] for any r ≥ 10 on the grid.

**Primary metric.** Distillable entanglement rate D_H = R_attempt · P_herald · max(0, 1 − S(ρ))
in ebit/s (reported as ebit/hour). Here S(ρ) is the entropy of the Bell-diagonal heralded state,
the hashing bound. Secondary metrics are heralded fidelity and raw herald rate.

## 4. Scope and feasibility (explicit)

This is a **simulation study** using a closed-form, differentiable physical model. The model
is checked against a first-principles Fock-space simulation of the Bell-state measurement, an
event-level Monte Carlo and numerical quadrature. Hardware parameters are representative,
order-of-magnitude **assumptions** rather than measurements of a particular device. For that
reason they are randomised (10 seeds) and swept, including the main phenomenological parameter
of the noise model, the Raman cutoff. Everything runs on a laptop CPU in under an hour.
Nothing here claims an experimental demonstration.
