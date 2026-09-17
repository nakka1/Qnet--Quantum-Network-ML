"""Getting the compensator off a workstation and onto a target.

Three things live here, in the order a port needs them.

:mod:`fixed_point` is the datapath in fixed point, written with the semantics
the emitted RTL has rather than with the semantics numpy has: products at full
width, one rounding and one saturation per layer, CORDIC with a fixed iteration
count. It is the golden reference the hardware is checked against.

:mod:`quantise` calibrates a fitted compensator against training trajectories
and rewrites it, keeping the ``reset``/``predict`` interface so the quantised
model can be scored by the same closed loop as the float one.

:mod:`verilog` emits the RTL and a testbench of vectors taken from the
reference.

A note on why any of this is here. Timing the datapath shows the cost is not
where an FPGA usually helps: the 46 656 multiply-accumulates take 25 us against
a 2 ms control period, while the fifteen SO(3) logarithms in the feature window
take 125 us. The case for hardware is determinism, co-location with the photon
counting, and power — not throughput. These modules exist so that case can be
argued with measurements instead of assertions.
"""

from .fixed_point import (CordicConfig, FixedLayer, FixedNetwork,  # noqa: F401
                          FixedStandardiser, QFormat, choose_format,
                          cordic_atan2, cordic_sincos, fixed_exp_map,
                          fixed_log_map)
from .quantise import (QuantisationConfig, QuantisedCompensator,  # noqa: F401
                       build_quantised, collect_calibration)
from .verilog import emit_vectors, emit_verilog  # noqa: F401

__all__ = [
    "QFormat", "choose_format", "FixedLayer", "FixedNetwork",
    "FixedStandardiser", "CordicConfig", "cordic_atan2", "cordic_sincos",
    "fixed_log_map", "fixed_exp_map",
    "QuantisationConfig", "QuantisedCompensator", "build_quantised",
    "collect_calibration", "emit_verilog", "emit_vectors",
]
