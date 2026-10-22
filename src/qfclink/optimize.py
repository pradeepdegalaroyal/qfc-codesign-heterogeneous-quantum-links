"""Noise-aware co-design of the link operating point.

Three optimisers are provided, all acting on the same bounded design space:

* :func:`optimise` - multi-start gradient ascent (Adam) through the
  differentiable link model (the proposed co-design method);
* :func:`random_search` - derivative-free uniform random search;
* :func:`grid_search` - dense grid (used as a near-exhaustive reference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import torch

from . import physics as ph
from .link import LinkHardware, OperatingPoint, evaluate

Number = Union[float, torch.Tensor]

PARAMS = ("target_nm", "pump_a", "pump_b", "filter_a_hz", "filter_b_hz", "compression")


def default_bounds(hw: LinkHardware) -> dict[str, tuple[float, float, str]]:
    """Physical bounds of each design variable: (low, high, scale)."""
    return {
        "target_nm": (1260.0, 1625.0, "lin"),  # O- to L-band
        "pump_a": (1e-3, 1.0, "lin"),
        "pump_b": (1e-3, 1.0, "lin"),
        "filter_a_hz": (1e6, 2e10, "log"),
        "filter_b_hz": (1e6, 2e10, "log"),
        "compression": (1.0, hw.compressor.max_factor, "log"),
    }


@dataclass
class DesignSpace:
    """Which variables are optimised (``free``) and the values of the others."""

    free: tuple[str, ...]
    fixed: dict[str, Number] = field(default_factory=dict)
    compress: bool = False

    def __post_init__(self) -> None:
        needed = set(PARAMS) - set(self.free)
        if not self.compress:
            needed.discard("compression")
            self.fixed.setdefault("compression", 1.0)
        missing = needed - set(self.fixed)
        if missing:
            raise ValueError(f"fixed values missing for {sorted(missing)}")


@dataclass
class OptResult:
    """Best operating point(s) found; tensors have shape (n_problems,)."""

    values: dict[str, torch.Tensor]
    objective: torch.Tensor
    ebit_rate: torch.Tensor
    n_evals: int

    def point(self, i: int = 0) -> dict[str, float]:
        return {k: float(v.reshape(-1)[i]) for k, v in self.values.items()}


def _decode(hw: LinkHardware, space: DesignSpace, u: torch.Tensor, fixed: Mapping[str, Number]) -> OperatingPoint:
    bounds = default_bounds(hw)
    vals: dict[str, Number] = dict(fixed)
    for j, name in enumerate(space.free):
        lo, hi, scale = bounds[name]
        s = torch.sigmoid(u[..., j])
        if scale == "lin":
            vals[name] = lo + (hi - lo) * s
        else:
            vals[name] = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * s)
    return OperatingPoint(**{k: vals[k] for k in PARAMS})


def _encode(hw: LinkHardware, space: DesignSpace, values: Mapping[str, float]) -> torch.Tensor:
    bounds = default_bounds(hw)
    out = []
    for name in space.free:
        lo, hi, scale = bounds[name]
        v = float(values[name])
        q = (v - lo) / (hi - lo) if scale == "lin" else (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        q = min(max(q, 1e-4), 1 - 1e-4)
        out.append(math.log(q / (1 - q)))
    return torch.tensor(out, dtype=ph.DTYPE)


def _broadcast_fixed(space: DesignSpace, n_problems: int, n_starts: int) -> dict[str, Number]:
    fixed: dict[str, Number] = {}
    for k, v in space.fixed.items():
        if isinstance(v, torch.Tensor) and v.numel() > 1:
            fixed[k] = v.to(ph.DTYPE).reshape(n_problems, 1).expand(n_problems, n_starts)
        else:
            fixed[k] = float(v)
    return fixed


def _select_best(values: dict[str, torch.Tensor], score: torch.Tensor, ebit: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    idx = torch.argmax(score, dim=-1, keepdim=True)
    best = {k: torch.gather(v.expand_as(score), -1, idx).squeeze(-1) for k, v in values.items()}
    return best, torch.gather(score, -1, idx).squeeze(-1), torch.gather(ebit, -1, idx).squeeze(-1)


def _values(op: OperatingPoint, shape: torch.Size) -> dict[str, torch.Tensor]:
    return {k: ph.t(v).expand(shape).detach().clone() for k, v in op.as_dict().items()}


def optimise(
    hw: LinkHardware,
    distance_km: float,
    space: DesignSpace,
    n_starts: int = 16,
    steps: int = 300,
    lr: float = 0.08,
    seed: int = 0,
    warm_starts: Optional[Sequence[Mapping[str, float]]] = None,
    n_problems: int = 1,
) -> OptResult:
    """Multi-start Adam ascent of the (unclamped) distillable-entanglement rate.

    The gradient is normalised by the current objective magnitude so that the
    effective step is a *relative* improvement; this keeps optimisation well
    conditioned across many orders of magnitude of rate. ``n_problems``
    independent problems (e.g. different fixed target wavelengths) are solved
    in one batch when ``space.fixed`` contains tensors of that length.
    """
    gen = torch.Generator().manual_seed(seed)
    nf = len(space.free)
    q = torch.rand(n_problems, n_starts, nf, generator=gen, dtype=ph.DTYPE) * 0.96 + 0.02
    u0 = torch.log(q / (1 - q))
    for j, w in enumerate(warm_starts or []):
        if j < n_starts:
            u0[:, j, :] = _encode(hw, space, w)
    u = u0.clone().requires_grad_(True)
    fixed = _broadcast_fixed(space, n_problems, n_starts)
    opt = torch.optim.Adam([u], lr=lr)
    best_obj = torch.full((n_problems, n_starts), -math.inf, dtype=ph.DTYPE)
    best_u = u0.clone()
    for _ in range(steps):
        opt.zero_grad()
        out = evaluate(hw, _decode(hw, space, u, fixed), distance_km, space.compress)
        obj = out["objective"]
        with torch.no_grad():
            improved = obj > best_obj
            best_obj = torch.where(improved, obj, best_obj)
            best_u = torch.where(improved.unsqueeze(-1), u, best_u)
        loss = -(obj / (obj.detach().abs() + 1e-300)).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        op = _decode(hw, space, best_u, fixed)
        out = evaluate(hw, op, distance_km, space.compress)
        vals, score, ebit = _select_best(_values(op, out["objective"].shape), out["objective"], out["ebit_rate"])
    return OptResult(vals, score, ebit, n_evals=steps * n_starts * n_problems)


def _evaluate_uniform(hw: LinkHardware, distance_km: float, space: DesignSpace, q: torch.Tensor, chunk: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    fixed = {k: float(v) for k, v in space.fixed.items()}
    best_score = torch.tensor(-math.inf, dtype=ph.DTYPE)
    best_vals: dict[str, torch.Tensor] = {}
    best_ebit = torch.tensor(0.0, dtype=ph.DTYPE)
    q = q.clamp(1e-9, 1 - 1e-9)
    with torch.no_grad():
        for i in range(0, q.shape[0], chunk):
            u = torch.log(q[i : i + chunk] / (1 - q[i : i + chunk]))
            op = _decode(hw, space, u, fixed)
            out = evaluate(hw, op, distance_km, space.compress)
            vals, score, ebit = _select_best(_values(op, out["objective"].shape), out["objective"], out["ebit_rate"])
            if score > best_score:
                best_score, best_vals, best_ebit = score, vals, ebit
    return {k: v.reshape(1) for k, v in best_vals.items()}, best_score.reshape(1), best_ebit.reshape(1)


def random_search(hw: LinkHardware, distance_km: float, space: DesignSpace, n_samples: int, seed: int = 0, chunk: int = 200_000) -> OptResult:
    """Uniform random search (log-uniform for bandwidths) over the free variables."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.rand(n_samples, len(space.free), generator=gen, dtype=ph.DTYPE)
    vals, score, ebit = _evaluate_uniform(hw, distance_km, space, q, chunk)
    return OptResult(vals, score, ebit, n_evals=n_samples)


def grid_search(hw: LinkHardware, distance_km: float, space: DesignSpace, points: Union[int, Sequence[int]], chunk: int = 400_000) -> OptResult:
    """Exhaustive tensor-product grid over the free variables."""
    nf = len(space.free)
    pts = [points] * nf if isinstance(points, int) else list(points)
    axes = [torch.linspace(0.0005, 0.9995, n, dtype=ph.DTYPE) for n in pts]
    q = torch.cartesian_prod(*axes).reshape(-1, nf)
    vals, score, ebit = _evaluate_uniform(hw, distance_km, space, q, chunk)
    return OptResult(vals, score, ebit, n_evals=int(np.prod(pts)))


def evaluate_point(hw: LinkHardware, distance_km: float, values: Mapping[str, float], compress: bool = False) -> dict[str, float]:
    """Evaluate a fixed operating point; returns plain floats."""
    op = OperatingPoint(**{k: float(values.get(k, 1.0)) for k in PARAMS})
    with torch.no_grad():
        out = evaluate(hw, op, distance_km, compress)
    return {k: float(v) for k, v in out.items()}
