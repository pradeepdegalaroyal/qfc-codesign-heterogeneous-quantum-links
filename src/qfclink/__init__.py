"""qfclink: noise-aware co-design of quantum frequency conversion and bandwidth
matching for heterogeneous heralded-entanglement links."""

from .link import Compressor, Converter, Detector, Emitter, LinkHardware, OperatingPoint, evaluate
from .optimize import DesignSpace, OptResult, evaluate_point, grid_search, optimise, random_search
from .scenarios import calibrate, heterogeneous_nominal, homogeneous_nominal, sample_hardware

__version__ = "0.1.0"

__all__ = [
    "Compressor",
    "Converter",
    "Detector",
    "DesignSpace",
    "Emitter",
    "LinkHardware",
    "OperatingPoint",
    "OptResult",
    "calibrate",
    "evaluate",
    "evaluate_point",
    "grid_search",
    "heterogeneous_nominal",
    "homogeneous_nominal",
    "optimise",
    "random_search",
    "sample_hardware",
]
