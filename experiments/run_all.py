"""Reproduce every result and figure of the project with one command.

    python experiments/run_all.py            # full run (~10 min on a laptop CPU)
    python experiments/run_all.py --quick    # smoke run (< 2 min), writes to quick_outputs/

Experiments
-----------
E1 validation   Fock-space BSM vs closed form; closed-form spectral overlap vs
                numerical quadrature; event Monte-Carlo vs analytic heralding model.
E2 landscape    pump-induced noise and fibre loss vs telecom target wavelength.
E3 map          optimal target wavelength vs distance (heterogeneous + homogeneous).
E4 main         co-design vs literature-standard operating points over hardware
                seeds and distances, with calibration error; ablations; optimiser
                baselines (random search, dense grid).
E5 robustness   sweeps of noise strength, dark counts, detuning, memory T2 and the
                Raman-cutoff model parameter.
E6 bandwidth    passive filtering vs active bandwidth compression over
                (bandwidth ratio, compressor efficiency).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy import integrate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qfclink import physics as ph  # noqa: E402
from qfclink import plotting as pl  # noqa: E402
from qfclink.fock import pure_mode_pair, simulate_bsm  # noqa: E402
from qfclink.link import LinkHardware, OperatingPoint, evaluate  # noqa: E402
from qfclink.montecarlo import simulate_heralding  # noqa: E402
from qfclink.optimize import DesignSpace, grid_search, optimise, random_search  # noqa: E402
from qfclink.scenarios import calibrate, heterogeneous_nominal, homogeneous_nominal, sample_hardware  # noqa: E402
from qfclink.stats import paired_comparison  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

FULL = ("target_nm", "pump_a", "pump_b", "filter_a_hz", "filter_b_hz")
FILTERS = ("filter_a_hz", "filter_b_hz")
O_BAND_NM = 1350.0  # O-band target used for SiV QFC in Knaut et al. (2024)
C_BAND_NM = 1550.0  # minimum-loss C-band target
HOUR = 3600.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Design spaces of the compared strategies
# --------------------------------------------------------------------------
def space_for(method: str) -> DesignSpace:
    """Design space for each method / ablation."""
    pumps_max = {"pump_a": 1.0, "pump_b": 1.0}
    if method == "std_C":
        return DesignSpace(FILTERS, {"target_nm": C_BAND_NM, **pumps_max})
    if method == "std_O":
        return DesignSpace(FILTERS, {"target_nm": O_BAND_NM, **pumps_max})
    if method == "abl_pump_C":
        return DesignSpace(("pump_a", "pump_b") + FILTERS, {"target_nm": C_BAND_NM})
    if method == "abl_pump_O":
        return DesignSpace(("pump_a", "pump_b") + FILTERS, {"target_nm": O_BAND_NM})
    if method == "abl_wavelength":
        return DesignSpace(("target_nm",) + FILTERS, dict(pumps_max))
    if method == "codesign":
        return DesignSpace(FULL)
    if method == "codesign_comp":
        return DesignSpace(FULL + ("compression",), compress=True)
    raise ValueError(method)


def warm_starts(hw: LinkHardware) -> list[dict[str, float]]:
    """Literature-style starting points (full pump, filters at the narrow linewidth)."""
    nb = min(hw.emitter_a.linewidth_hz, hw.emitter_b.linewidth_hz)
    ratio = max(hw.emitter_a.linewidth_hz / hw.emitter_b.linewidth_hz, 1.0)
    kappa = float(min(ratio, hw.compressor.max_factor))
    base = dict(pump_a=0.999, pump_b=0.999, filter_a_hz=nb, filter_b_hz=3 * nb, compression=kappa)
    return [dict(base, target_nm=O_BAND_NM), dict(base, target_nm=C_BAND_NM)]


def design_and_evaluate(
    method: str,
    design_hw: LinkHardware,
    true_hw: LinkHardware,
    distances: list[float],
    opt_cfg: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Optimise the operating point on ``design_hw`` (batched over distances)
    and evaluate it on ``true_hw``."""
    space = space_for(method)
    n = len(distances)
    d_col = torch.tensor(distances, dtype=ph.DTYPE).reshape(n, 1)
    res = optimise(
        design_hw, d_col, space, n_starts=opt_cfg["n_starts"], steps=opt_cfg["steps"],
        lr=opt_cfg["lr"], seed=seed, warm_starts=warm_starts(design_hw), n_problems=n,
    )
    with torch.no_grad():
        out = evaluate(true_hw, OperatingPoint(**res.values), d_col.reshape(n), space.compress)
    rows = []
    for i, dist in enumerate(distances):
        rows.append(
            {
                "distance_km": dist,
                "ebit_rate": float(out["ebit_rate"][i]),
                "ebits_per_hour": float(out["ebit_rate"][i]) * HOUR,
                "fidelity": float(out["fidelity"][i]),
                "herald_rate_hz": float(out["rate_hz"][i]),
                "visibility": float(out["visibility"][i]),
                **{k: float(v[i]) for k, v in res.values.items()},
            }
        )
    return rows


# --------------------------------------------------------------------------
# E1 - validation
# --------------------------------------------------------------------------
def numeric_overlap(la: float, lb: float, fa: float, fb: float, det: float) -> tuple[float, float, float]:
    """Visibility and transmissions by adaptive quadrature over angular frequency."""
    a, b, f_a, f_b, d = math.pi * la, math.pi * lb, math.pi * fa, math.pi * fb, 2 * math.pi * det

    def psi_a(w: float) -> complex:
        return math.sqrt(2 * a) / (a - 1j * w) * f_a / (f_a - 1j * w)

    def psi_b(w: float) -> complex:
        return math.sqrt(2 * b) / (b - 1j * (w - d)) * f_b / (f_b - 1j * (w - d))

    widths = [a, b, f_a, f_b]
    pts = {0.0, d}
    for w0 in widths:
        for k in range(-2, 6):
            for c in (0.0, d):
                pts.add(c + w0 * 10.0**k)
                pts.add(c - w0 * 10.0**k)
    edges = sorted(pts)

    def integ(fun: Callable[[float], float]) -> float:
        total = integrate.quad(fun, -np.inf, edges[0], limit=200)[0]
        for lo, hi in zip(edges[:-1], edges[1:]):
            total += integrate.quad(fun, lo, hi, limit=200)[0]
        total += integrate.quad(fun, edges[-1], np.inf, limit=200)[0]
        return total / (2 * math.pi)

    o_re = integ(lambda w: (np.conj(psi_b(w)) * psi_a(w)).real)
    o_im = integ(lambda w: (np.conj(psi_b(w)) * psi_a(w)).imag)
    ta = integ(lambda w: abs(psi_a(w)) ** 2)
    tb = integ(lambda w: abs(psi_b(w)) ** 2)
    return (o_re**2 + o_im**2) / (ta * tb), ta, tb


def exp_validation(cfg: dict[str, Any], res_dir: Path, fig_dir: Path) -> dict[str, Any]:
    vcfg = cfg["validation"]
    rng = np.random.default_rng(1234)
    # (a) Fock-space BSM, pure and mixed internal states
    fock_rows = []
    for ov in np.linspace(0.0, 1.0, 21):
        r = simulate_bsm(*pure_mode_pair(float(ov)))
        fock_rows.append({"kind": "pure", "V": ov**2, "F_fock": r.mean_fidelity, "F_closed": (1 + ov**2) / 2, "p_herald": r.herald_probability})
    for _ in range(vcfg["n_mixed_fock"]):
        dim = 3

        def ensemble() -> list[tuple[float, np.ndarray]]:
            p = rng.dirichlet([1.0, 1.0])
            vecs = rng.normal(size=(2, dim)) + 1j * rng.normal(size=(2, dim))
            return [(float(p[i]), vecs[i] / np.linalg.norm(vecs[i])) for i in range(2)]

        ea, eb = ensemble(), ensemble()
        vis = sum(pa * pb * abs(np.vdot(va, vb)) ** 2 for pa, va in ea for pb, vb in eb)
        r = simulate_bsm(ea, eb)
        fock_rows.append({"kind": "mixed", "V": vis, "F_fock": r.mean_fidelity, "F_closed": (1 + vis) / 2, "p_herald": r.herald_probability})
    fock = pd.DataFrame(fock_rows)
    fock.to_csv(res_dir / "validation_fock.csv", index=False)

    # (b) closed-form filtered overlap vs quadrature
    ov_rows = []
    for _ in range(vcfg["n_overlap_checks"]):
        la, lb = 10 ** rng.uniform(6, 9.3, 2)
        fa, fb = 10 ** rng.uniform(6, 10.3, 2)
        det = rng.uniform(0, 2) * min(la, lb)
        v_num, ta_num, tb_num = numeric_overlap(la, lb, fa, fb, det)
        v_cf, ta_cf, tb_cf = ph.filtered_overlap(la, lb, fa, fb, det)
        ov_rows.append({"linewidth_a": la, "linewidth_b": lb, "filter_a": fa, "filter_b": fb, "detuning": det,
                        "V_numeric": v_num, "V_closed": float(v_cf), "TA_numeric": ta_num, "TA_closed": float(ta_cf)})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(res_dir / "validation_overlap.csv", index=False)

    # (c) event Monte-Carlo vs analytic heralding model
    regimes = [(0.05, 0.05, 1e-4, 0.9), (0.05, 0.2, 1e-3, 0.9), (0.02, 0.3, 5e-3, 0.7), (0.1, 0.1, 1e-2, 0.8),
               (0.01, 0.4, 2e-3, 0.95), (0.2, 0.2, 5e-2, 0.6), (0.03, 0.03, 3e-3, 1.0), (0.3, 0.05, 2e-2, 0.85)]
    mc_rows = []
    for k, (ea_, eb_, mu, coh) in enumerate(regimes):
        pn = 1 - math.exp(-mu)
        p_pair, p_one, p_two = ph.herald_probabilities(ea_, eb_, pn)
        eigs = ph.bell_diagonal(p_pair, p_one + p_two, coh)
        mc = simulate_heralding(ea_, eb_, mu, coh, vcfg["mc_attempts"], np.random.default_rng(100 + k))
        mc_rows.append({"eta_a": ea_, "eta_b": eb_, "mu_noise": mu, "coherence": coh,
                        "p_herald_analytic": float(p_pair + p_one + p_two), "p_herald_mc": mc.p_herald, "p_herald_mc_se": mc.p_herald_se,
                        "F_analytic": float(eigs[0]), "F_mc": mc.fidelity, "F_mc_se": mc.fidelity_se, "n_heralds": mc.n_heralds})
    mcd = pd.DataFrame(mc_rows)
    mcd.to_csv(res_dir / "validation_mc.csv", index=False)

    summary = {
        "fock_max_abs_error_F": float((fock.F_fock - fock.F_closed).abs().max()),
        "fock_herald_probability_range": [float(fock.p_herald.min()), float(fock.p_herald.max())],
        "overlap_max_abs_error_V": float((ov.V_numeric - ov.V_closed).abs().max()),
        "overlap_max_abs_error_T": float((ov.TA_numeric - ov.TA_closed).abs().max()),
        "mc_max_abs_z_p_herald": float(((mcd.p_herald_mc - mcd.p_herald_analytic) / mcd.p_herald_mc_se).abs().max()),
        "mc_max_abs_z_fidelity": float(((mcd.F_mc - mcd.F_analytic) / mcd.F_mc_se).abs().max()),
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax = axes[0]
    pure = fock[fock.kind == "pure"]
    mixed = fock[fock.kind == "mixed"]
    vv = np.linspace(0, 1, 50)
    ax.plot(vv, (1 + vv) / 2, color=pl.MUTED, lw=1.2, label="closed form $(1+V)/2$")
    ax.plot(pure.V, pure.F_fock, "o", color=pl.SERIES[0], label="Fock sim., pure modes")
    ax.plot(mixed.V, mixed.F_fock, "s", color=pl.SERIES[1], mfc="none", label="Fock sim., mixed modes")
    ax.set_xlabel("two-photon visibility $V$")
    ax.set_ylabel("heralded fidelity $F$")
    ax.legend(loc="upper left")
    pl.panel_label(ax, "a")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=pl.MUTED, lw=1.2)
    ax.plot(ov.V_numeric, ov.V_closed, "o", color=pl.SERIES[0], ms=4, alpha=0.8)
    ax.set_xlabel("$V$ numerical quadrature")
    ax.set_ylabel("$V$ closed form")
    ax.set_title(f"max |error| = {summary['overlap_max_abs_error_V']:.1e}", fontsize=8, color=pl.INK_2)
    pl.panel_label(ax, "b")
    ax = axes[2]
    x = np.arange(len(mcd))
    ax.errorbar(x - 0.12, mcd.F_mc, yerr=1.96 * mcd.F_mc_se, fmt="o", color=pl.SERIES[0], label="Monte-Carlo (95% CI)")
    ax.plot(x + 0.12, mcd.F_analytic, "D", color=pl.SERIES[1], label="analytic model")
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{i + 1}" for i in x])
    ax.set_xlabel("regime ($\\eta_A,\\eta_B,\\mu,c$ varied)")
    ax.set_ylabel("heralded fidelity")
    ax.legend(loc="lower left")
    pl.panel_label(ax, "c")
    pl.save(fig, fig_dir / "fig1_validation.png")
    return summary


# --------------------------------------------------------------------------
# E2 - noise / loss landscape
# --------------------------------------------------------------------------
def exp_landscape(res_dir: Path, fig_dir: Path) -> None:
    conv = heterogeneous_nominal().converter
    lam = torch.linspace(1260, 1625, 731, dtype=ph.DTYPE)
    rows: dict[str, Any] = {"target_nm": lam.numpy(), "attenuation_db_per_km": ph.fibre_attenuation_db_per_km(lam).numpy()}
    sources = [(737.0, "737 nm"), (780.0, "780 nm"), (930.0, "930 nm")]
    for nm, _ in sources:
        rows[f"noise_per_mode_{int(nm)}"] = ph.raman_noise_per_mode(nm, lam, conv.raman_strength, conv.noise_floor, conv.raman_cutoff_cm1).numpy()
        rows[f"pump_nm_{int(nm)}"] = ph.pump_wavelength_nm(nm, lam).numpy()
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "landscape.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    for i, (nm, label) in enumerate(sources):
        ax.semilogy(df.target_nm, df[f"noise_per_mode_{int(nm)}"], color=pl.SERIES[i], label=f"source {label}")
    for x, name in ((O_BAND_NM, "1350"), (C_BAND_NM, "1550")):
        ax.axvline(x, color=pl.MUTED, ls="--", lw=0.8)
        ax.text(x + 3, 2e-3, name, color=pl.INK_2, fontsize=7)
    ax.set_xlabel("telecom target wavelength (nm)")
    ax.set_ylabel("pump-induced noise (photons / mode at $P_{max}$)")
    ax.legend(loc="upper left")
    pl.panel_label(ax, "a")
    ax = axes[1]
    ax.plot(df.target_nm, df.attenuation_db_per_km, color=pl.SERIES[0])
    ax.set_xlabel("telecom target wavelength (nm)")
    ax.set_ylabel("fibre attenuation (dB/km)")
    pl.panel_label(ax, "b")
    pl.save(fig, fig_dir / "fig2_noise_loss_landscape.png")


# --------------------------------------------------------------------------
# E3 - wavelength x distance map
# --------------------------------------------------------------------------
def wavelength_map(hw: LinkHardware, mcfg: dict[str, Any], seed: int) -> pd.DataFrame:
    lam = np.linspace(1262, 1623, mcfg["n_wavelengths"])
    dist = np.linspace(5, mcfg["max_distance_km"], mcfg["n_distances"])
    ll, dd = np.meshgrid(lam, dist, indexing="xy")
    n = ll.size
    space = DesignSpace(("pump_a", "pump_b") + FILTERS, {"target_nm": torch.tensor(ll.ravel(), dtype=ph.DTYPE)})
    res = optimise(hw, torch.tensor(dd.ravel(), dtype=ph.DTYPE).reshape(n, 1), space, n_starts=mcfg["n_starts"],
                   steps=mcfg["steps"], lr=0.1, seed=seed, warm_starts=warm_starts(hw), n_problems=n)
    return pd.DataFrame({"target_nm": ll.ravel(), "distance_km": dd.ravel(), "ebit_rate": res.ebit_rate.numpy(),
                         "pump_a": res.values["pump_a"].numpy(), "pump_b": res.values["pump_b"].numpy()})


def exp_map(cfg: dict[str, Any], res_dir: Path, fig_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for k, (name, hw) in enumerate((("heterogeneous", heterogeneous_nominal()), ("homogeneous", homogeneous_nominal()))):
        df = wavelength_map(hw, cfg["map"], seed=k)
        df.to_csv(res_dir / f"map_{name}.csv", index=False)
        piv = df.pivot(index="distance_km", columns="target_nm", values="ebit_rate") * HOUR
        best = piv.idxmax(axis=1)
        out[name] = {
            "best_target_nm_by_distance": {f"{d:.1f}": float(b) for d, b in best.items()},
            "best_ebits_per_hour_by_distance": {f"{d:.1f}": float(piv.loc[d].max()) for d in piv.index},
        }
        ax = axes[k]
        z = np.log10(np.maximum(piv.values, 1e-3))
        mesh = ax.pcolormesh(piv.columns, piv.index, z, cmap=pl.SEQUENTIAL, shading="auto", vmin=-3, vmax=max(z.max(), 0))
        ax.plot(best.values, best.index, color=pl.SERIES[1], lw=2, label="optimal target")
        for x in (O_BAND_NM, C_BAND_NM):
            ax.axvline(x, color=pl.INK_2, ls="--", lw=0.8)
        ax.set_xlabel("telecom target wavelength (nm)")
        ax.set_title(f"{name} link ({hw.emitter_a.wavelength_nm:.0f} nm + {hw.emitter_b.wavelength_nm:.0f} nm)")
        ax.grid(False)
        ax.legend(loc="upper right", frameon=True, facecolor=pl.SURFACE)
        cb = fig.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label("log$_{10}$ ebit / hour (pump + filters optimised)")
        pl.panel_label(ax, "ab"[k])
    axes[0].set_ylabel("total link distance (km)")
    pl.save(fig, fig_dir / "fig3_wavelength_distance_map.png")
    return out


# --------------------------------------------------------------------------
# E4 - main comparison, ablation and optimiser baselines
# --------------------------------------------------------------------------
def exp_main(cfg: dict[str, Any], res_dir: Path, fig_dir: Path) -> dict[str, Any]:
    nominal = heterogeneous_nominal()
    distances = [float(d) for d in cfg["distances_km"]]
    ocfg = cfg["optimiser"]
    rows: list[dict[str, Any]] = []
    opt_rows: list[dict[str, Any]] = []
    for seed in cfg["seeds"]:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        true_hw = sample_hardware(nominal, rng)
        cal_hw = calibrate(true_hw, np.random.default_rng(10_000 + seed), cfg["calibration_error"])
        plans = [(m, "calibrated", cal_hw) for m in ("std_C", "std_O", "codesign")]
        plans += [(m, "nominal", nominal) for m in ("std_C", "std_O", "codesign")]
        plans += [(m, "oracle", true_hw) for m in ("std_C", "std_O", "abl_pump_C", "abl_pump_O", "abl_wavelength", "codesign", "codesign_comp")]
        for method, info, design_hw in plans:
            for r in design_and_evaluate(method, design_hw, true_hw, distances, ocfg, seed):
                rows.append({"seed": seed, "method": method, "information": info, **r})
        # optimiser baselines on the full space (oracle information)
        adam_budget = ocfg["n_starts"] * ocfg["steps"]
        space = space_for("codesign")
        for d in distances:
            adam = [r for r in rows if r["seed"] == seed and r["method"] == "codesign" and r["information"] == "oracle" and r["distance_km"] == d][0]
            rs = random_search(true_hw, d, space, n_samples=cfg["random_search_budget_factor"] * adam_budget, seed=seed)
            gs = grid_search(true_hw, d, space, cfg["grid_points"])
            opt_rows.append({"seed": seed, "distance_km": d, "adam_ebit_rate": adam["ebit_rate"], "adam_evals": adam_budget,
                             "random_ebit_rate": float(rs.ebit_rate[0]), "random_evals": rs.n_evals,
                             "grid_ebit_rate": float(gs.ebit_rate[0]), "grid_evals": gs.n_evals})
        log(f"  E4 seed {seed} done in {time.time() - t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "main_results.csv", index=False)
    opt = pd.DataFrame(opt_rows)
    opt.to_csv(res_dir / "optimiser_comparison.csv", index=False)
    return analyse_main(cfg, res_dir, fig_dir)


def analyse_main(cfg: dict[str, Any], res_dir: Path, fig_dir: Path) -> dict[str, Any]:
    """Statistics and figures of E4, computed from the saved CSV files."""
    df = pd.read_csv(res_dir / "main_results.csv")
    opt = pd.read_csv(res_dir / "optimiser_comparison.csv")
    distances = [float(d) for d in cfg["distances_km"]]

    def series(method: str, info: str, d: float) -> np.ndarray:
        s = df[(df.method == method) & (df.information == info) & (df.distance_km == d)].sort_values("seed")
        return s.ebit_rate.to_numpy()

    stats_out: dict[str, Any] = {}
    agg_rows = []
    for d in distances:
        best_std_cal = np.maximum(series("std_C", "calibrated", d), series("std_O", "calibrated", d))
        best_std_orc = np.maximum(series("std_C", "oracle", d), series("std_O", "oracle", d))
        stats_out[f"{d:g}km"] = {
            "H1_codesign_calibrated_vs_best_standard_calibrated": paired_comparison(series("codesign", "calibrated", d), best_std_cal),
            "codesign_calibrated_vs_std_O_calibrated": paired_comparison(series("codesign", "calibrated", d), series("std_O", "calibrated", d)),
            "codesign_nominal_vs_best_standard_calibrated": paired_comparison(series("codesign", "nominal", d), best_std_cal),
            "codesign_oracle_vs_best_standard_oracle": paired_comparison(series("codesign", "oracle", d), best_std_orc),
            "compression_vs_filtering_oracle": paired_comparison(series("codesign_comp", "oracle", d), series("codesign", "oracle", d)),
            "calibrated_vs_oracle_codesign": paired_comparison(series("codesign", "calibrated", d), series("codesign", "oracle", d)),
        }
        for (method, info), g in df[df.distance_km == d].groupby(["method", "information"]):
            agg_rows.append({"distance_km": d, "method": method, "information": info,
                             "ebits_per_hour_mean": g.ebits_per_hour.mean(), "ebits_per_hour_std": g.ebits_per_hour.std(ddof=1),
                             "fidelity_mean": g.fidelity.mean(), "fidelity_std": g.fidelity.std(ddof=1),
                             "target_nm_mean": g.target_nm.mean(), "target_nm_std": g.target_nm.std(ddof=1),
                             "pump_b_mean": g.pump_b.mean(), "n_zero_rate": int((g.ebit_rate <= 0).sum()), "n": len(g)})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(res_dir / "main_summary.csv", index=False)
    with open(res_dir / "main_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    # ---- Figure 4: main comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    curves = [("std_C", "calibrated", "standard, C-band 1550 nm"), ("std_O", "calibrated", "standard, O-band 1350 nm"),
              ("codesign", "nominal", "co-design (nominal model)"), ("codesign", "calibrated", "co-design (calibrated)")]
    for i, (m, info, label) in enumerate(curves):
        g = agg[(agg.method == m) & (agg.information == info)].sort_values("distance_km")
        mean = g.ebits_per_hour_mean.to_numpy()
        ok = mean > 0
        ax.errorbar(g.distance_km[ok], mean[ok], yerr=g.ebits_per_hour_std.to_numpy()[ok], color=pl.SERIES[i],
                    marker=pl.MARKERS[i], capsize=3, label=label)
        if not ok.all():
            ax.plot(g.distance_km[~ok], np.full((~ok).sum(), 1e-2), marker=pl.MARKERS[i], ls="none", color=pl.SERIES[i], mfc="none")
    ax.set_yscale("log")
    ax.set_ylim(bottom=5e-3)
    ax.text(0.98, 0.04, "open markers at $10^{-2}$: mean rate = 0\n(no distillable entanglement)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=pl.INK_2)
    ax.set_xlabel("total link distance (km)")
    ax.set_ylabel("distillable entanglement (ebit / hour)")
    ax.legend(loc="lower left")
    pl.panel_label(ax, "a")
    ax = axes[1]
    for j, d in enumerate(distances):
        best_std = np.maximum(series("std_C", "calibrated", d), series("std_O", "calibrated", d))
        ratio = np.maximum(series("codesign", "calibrated", d), 1e-12) / np.maximum(best_std, 1e-12)
        jitter = np.random.default_rng(j).uniform(-0.15, 0.15, len(ratio))
        ax.plot(np.full(len(ratio), j) + jitter, ratio, "o", color=pl.SERIES[0], alpha=0.7, ms=5)
        ax.plot([j - 0.3, j + 0.3], [np.median(ratio)] * 2, color=pl.INK, lw=2)
    ax.axhline(1.0, color=pl.MUTED, ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(distances)))
    ax.set_xticklabels([f"{d:g}" for d in distances])
    ax.set_xlabel("total link distance (km)")
    ax.set_ylabel("co-design / best standard (ebit rate ratio)")
    ax.set_title("per hardware seed (dots) and median (bar)", fontsize=8, color=pl.INK_2)
    pl.panel_label(ax, "b")
    pl.save(fig, fig_dir / "fig4_main_comparison.png")

    # ---- Figure 5: ablation (oracle information)
    abl = [("std_C", "standard C-band"), ("std_O", "standard O-band"), ("abl_pump_C", "+ pump power (C-band)"),
           ("abl_pump_O", "+ pump power (O-band)"), ("abl_wavelength", "+ wavelength (full pump)"),
           ("codesign", "full co-design"), ("codesign_comp", "full co-design + compression")]
    pick = [d for d in (50.0, 100.0) if d in distances] or distances[-1:]
    fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 3.4), squeeze=False)
    for k, d in enumerate(pick):
        ax = axes[0, k]
        ref = series("std_O", "oracle", d)
        for i, (m, label) in enumerate(abl):
            r = series(m, "oracle", d) / np.maximum(ref, 1e-300)
            med, q1, q3 = np.median(r), np.quantile(r, 0.25), np.quantile(r, 0.75)
            if med < 1e-2:
                n_zero = int(np.sum(series(m, "oracle", d) <= 0))
                ax.text(1.2e-2, i, f"< 0.01 ({n_zero}/{len(r)} seeds with zero yield)", va="center", fontsize=7, color=pl.INK_2)
                continue
            ax.barh(i, med - 1e-2, left=1e-2, color=pl.SERIES[0] if m.startswith("codesign") else "#9ec5f4", height=0.6)
            ax.errorbar(med, i, xerr=[[med - q1], [q3 - med]], color=pl.INK, capsize=3, lw=1)
        ax.axvline(1.0, color=pl.MUTED, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 300)
        ax.set_yticks(range(len(abl)))
        ax.set_yticklabels([a[1] for a in abl] if k == 0 else [])
        ax.invert_yaxis()
        ax.set_xlabel("ebit rate relative to standard O-band (median, IQR)")
        ax.set_title(f"{d:g} km", fontsize=10, pad=14)
    pl.save(fig, fig_dir / "fig5_ablation.png")

    # ---- Figure 8: optimiser comparison
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, (col, label) in enumerate((("random_ebit_rate", f"random search ({cfg['random_search_budget_factor']}x Adam budget)"),
                                      ("grid_ebit_rate", f"dense grid ({int(np.prod(cfg['grid_points']))} pts)"))):
        for j, d in enumerate(distances):
            g = opt[opt.distance_km == d]
            r = g[col].to_numpy() / np.maximum(g.adam_ebit_rate.to_numpy(), 1e-300)
            ax.plot(np.full(len(r), j) + (i - 0.5) * 0.3, r, pl.MARKERS[i], color=pl.SERIES[i], alpha=0.8,
                    label=label if j == 0 else None)
    ax.axhline(1.0, color=pl.MUTED, ls="--", lw=1)
    ax.set_xticks(range(len(distances)))
    ax.set_xticklabels([f"{d:g}" for d in distances])
    ax.set_xlabel("total link distance (km)")
    ax.set_ylabel("best ebit rate / gradient co-design")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    pl.save(fig, fig_dir / "fig8_optimiser_comparison.png")
    return {"stats": stats_out, "optimiser": {
        "random_over_adam_median": float(np.median(opt.random_ebit_rate / opt.adam_ebit_rate)),
        "grid_over_adam_median": float(np.median(opt.grid_ebit_rate / opt.adam_ebit_rate)),
        "grid_over_adam_max": float(np.max(opt.grid_ebit_rate / opt.adam_ebit_rate)),
        "adam_evals": int(opt.adam_evals.iloc[0]), "random_evals": int(opt.random_evals.iloc[0]), "grid_evals": int(opt.grid_evals.iloc[0])}}


# --------------------------------------------------------------------------
# E5 - robustness sweeps
# --------------------------------------------------------------------------
def _sweep_hw(hw: LinkHardware, kind: str, value: float) -> LinkHardware:
    if kind == "noise_scale":
        return hw.with_(noise_scale=value)
    if kind == "dark_rate_hz":
        return hw.with_(detector=replace(hw.detector, dark_rate_hz=value))
    if kind == "detuning_in_narrow_linewidths":
        return hw.with_(detuning_hz=value * min(hw.emitter_a.linewidth_hz, hw.emitter_b.linewidth_hz))
    if kind == "t2_scale":
        return hw.with_(emitter_a=replace(hw.emitter_a, t2_s=hw.emitter_a.t2_s * value),
                        emitter_b=replace(hw.emitter_b, t2_s=hw.emitter_b.t2_s * value))
    if kind == "raman_cutoff_cm1":
        return hw.with_(converter=replace(hw.converter, raman_cutoff_cm1=value))
    raise ValueError(kind)


def exp_robustness(cfg: dict[str, Any], res_dir: Path, fig_dir: Path) -> None:
    rc = cfg["robustness"]
    d = float(rc["distance_km"])
    ocfg = {"n_starts": rc["n_starts"], "steps": rc["steps"], "lr": cfg["optimiser"]["lr"]}
    nominal = heterogeneous_nominal()
    rows = []
    kinds = ["noise_scale", "dark_rate_hz", "detuning_in_narrow_linewidths", "t2_scale", "raman_cutoff_cm1"]
    for kind in kinds:
        for value in rc[kind]:
            for seed in rc["seeds"]:
                true_hw = _sweep_hw(sample_hardware(nominal, np.random.default_rng(seed)), kind, float(value))
                for m in ("std_C", "std_O", "codesign"):
                    r = design_and_evaluate(m, true_hw, true_hw, [d], ocfg, seed)[0]
                    rows.append({"sweep": kind, "value": value, "seed": seed, "method": m, **r})
                r = design_and_evaluate("codesign", nominal, true_hw, [d], ocfg, 0)[0]
                rows.append({"sweep": kind, "value": value, "seed": seed, "method": "codesign_nominal_design", **r})
        log(f"  E5 sweep {kind} done")
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "robustness.csv", index=False)
    labels = {"noise_scale": "converter noise scale (x nominal)", "dark_rate_hz": "detector dark-count rate (Hz)",
              "detuning_in_narrow_linewidths": "residual detuning (narrow linewidths)", "t2_scale": "memory $T_2$ scale (x nominal)",
              "raman_cutoff_cm1": "true Raman cutoff (cm$^{-1}$)"}
    curves = [("std_C", "standard C-band"), ("std_O", "standard O-band"), ("codesign", "co-design (oracle)"),
              ("codesign_nominal_design", "co-design (nominal model)")]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.3 * len(kinds), 3.2), sharey=True)
    for k, kind in enumerate(kinds):
        ax = axes[k]
        sub = df[df.sweep == kind]
        for i, (m, label) in enumerate(curves):
            g = sub[sub.method == m].groupby("value").ebits_per_hour.agg(["mean", "std"]).reset_index()
            ok = g["mean"] > 0
            ax.errorbar(g.value[ok], g["mean"][ok], yerr=g["std"].fillna(0)[ok], marker=pl.MARKERS[i], color=pl.SERIES[i],
                        capsize=2, label=label, lw=1.6)
        if kind in ("noise_scale", "dark_rate_hz", "t2_scale"):
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(labels[kind])
        pl.panel_label(ax, "abcde"[k])
    axes[0].set_ylabel(f"ebit / hour at {d:g} km")
    axes[0].legend(loc="lower left", fontsize=7)
    pl.save(fig, fig_dir / "fig6_robustness.png")


# --------------------------------------------------------------------------
# E6 - filtering vs active bandwidth compression
# --------------------------------------------------------------------------
def exp_bandwidth(cfg: dict[str, Any], res_dir: Path, fig_dir: Path) -> dict[str, Any]:
    bc = cfg["bandwidth"]
    d = float(bc["distance_km"])
    ocfg = {"n_starts": bc["n_starts"], "steps": bc["steps"], "lr": cfg["optimiser"]["lr"]}
    nominal = heterogeneous_nominal()
    ratios = np.logspace(0, math.log10(bc["max_ratio"]), bc["n_ratios"])
    effs = np.logspace(math.log10(bc["min_efficiency"]), 0.0, bc["n_efficiencies"])
    rows = []
    for r in ratios:
        hw_r = nominal.with_(emitter_a=replace(nominal.emitter_a, linewidth_hz=float(r * nominal.emitter_b.linewidth_hz)))
        filt = design_and_evaluate("codesign", hw_r, hw_r, [d], ocfg, 0)[0]
        for e in effs:
            hw_re = hw_r.with_(compressor=replace(hw_r.compressor, efficiency=float(e)))
            comp = design_and_evaluate("codesign_comp", hw_re, hw_re, [d], ocfg, 0)[0]
            rows.append({"bandwidth_ratio": r, "compressor_efficiency": e, "filter_ebit_rate": filt["ebit_rate"],
                         "compress_ebit_rate": comp["ebit_rate"], "compression_factor": comp["compression"],
                         "filter_fidelity": filt["fidelity"], "compress_fidelity": comp["fidelity"]})
    df = pd.DataFrame(rows)
    df["log10_ratio"] = np.log10(np.maximum(df.compress_ebit_rate, 1e-15) / np.maximum(df.filter_ebit_rate, 1e-15))
    df.to_csv(res_dir / "bandwidth_crossover.csv", index=False)
    crossover = {}
    for r, g in df.groupby("bandwidth_ratio"):
        g = g.sort_values("compressor_efficiency")
        y, x = g.log10_ratio.to_numpy(), g.compressor_efficiency.to_numpy()
        idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
        if y[0] >= 0:
            crossover[f"{r:.3g}"] = float(x[0])
        elif len(idx):
            i = idx[0]
            crossover[f"{r:.3g}"] = float(x[i] + (x[i + 1] - x[i]) * (-y[i]) / (y[i + 1] - y[i]))
        else:
            crossover[f"{r:.3g}"] = None
    piv = df.pivot(index="compressor_efficiency", columns="bandwidth_ratio", values="log10_ratio")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    lim = float(np.nanmax(np.abs(piv.values.clip(-3, 3))))
    mesh = ax.pcolormesh(piv.columns, piv.index, piv.values.clip(-3, 3), cmap=pl.DIVERGING.reversed(), vmin=-lim, vmax=lim, shading="auto")
    if piv.shape[0] > 1 and piv.shape[1] > 1:
        ax.contour(piv.columns, piv.index, piv.values, levels=[0.0], colors=[pl.INK], linewidths=1.5)
    xs = [float(k) for k, v in crossover.items() if v is not None]
    ys = [v for v in crossover.values() if v is not None]
    ax.plot(xs, ys, "o", color=pl.INK, ms=4, label="crossover $\\eta^*$")
    rr = np.logspace(0, math.log10(bc["max_ratio"]), 50)
    ax.plot(rr, np.minimum(1.0, 2.0 / (1 + rr)), "--", color=pl.SERIES[1], lw=1.5, label="filter transmission $2/(1+r)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(effs.min(), effs.max())
    ax.set_xlabel("bandwidth ratio $r = \\Gamma_A/\\Gamma_B$")
    ax.set_ylabel("compressor efficiency $\\eta_{BC}$")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True, facecolor=pl.SURFACE, fontsize=7)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("log$_{10}$(compression / filtering) ebit rate")
    pl.save(fig, fig_dir / "fig7_bandwidth_crossover.png")
    return {"distance_km": d, "crossover_efficiency_by_ratio": crossover}


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="fast smoke run (< 2 min)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None, help="run only these experiments (updates summary.json)")
    parser.add_argument("--out", type=Path, default=None, help="output root (default: repo root, or quick_outputs/ with --quick)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg_path = args.config or ROOT / "configs" / ("quick.json" if args.quick else "default.json")
    cfg = json.loads(cfg_path.read_text())
    out_root = args.out or (ROOT / "quick_outputs" if args.quick else ROOT)
    res_dir, fig_dir = out_root / "results", out_root / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pl.setup()
    t_start = time.time()
    summary_path = res_dir / "summary.json"
    previous = json.loads(summary_path.read_text()) if (args.only and summary_path.exists()) else {}
    summary: dict[str, Any] = {**previous, "config": str(cfg_path.relative_to(ROOT)) if cfg_path.is_relative_to(ROOT) else str(cfg_path),
                               "torch": torch.__version__, "numpy": np.__version__}
    timings = {}
    for name, fn in (("validation", lambda: exp_validation(cfg, res_dir, fig_dir)),
                     ("landscape", lambda: exp_landscape(res_dir, fig_dir)),
                     ("map", lambda: exp_map(cfg, res_dir, fig_dir)),
                     ("main", lambda: exp_main(cfg, res_dir, fig_dir)),
                     ("main_analysis", lambda: analyse_main(cfg, res_dir, fig_dir) if args.only else None),
                     ("robustness", lambda: exp_robustness(cfg, res_dir, fig_dir)),
                     ("bandwidth", lambda: exp_bandwidth(cfg, res_dir, fig_dir))):
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        log(f"running {name}")
        result = fn()
        if result is not None:
            summary["main" if name == "main_analysis" else name] = result
        timings[name] = round(time.time() - t0, 1)
        log(f"{name} finished in {timings[name]} s")
    summary["timings_s"] = {**previous.get("timings_s", {}), **timings}
    summary["total_runtime_s"] = round(sum(summary["timings_s"].values()), 1)
    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"all done in {summary['total_runtime_s']} s -> {res_dir}")


if __name__ == "__main__":
    main()
