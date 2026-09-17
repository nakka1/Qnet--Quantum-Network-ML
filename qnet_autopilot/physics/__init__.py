"""Analytic physics layer: a fast, differentiable-friendly twin of SeQUeNCe's models.

Five submodules, each answering one question a controller has to answer:

:mod:`.bell_diagonal`
    What happens to a pair's *quality* under decoherence, purification and
    swapping.  Transcribed from SeQUeNCe's BDS protocols and tested against
    them.

:mod:`.fiber`
    What the fibre does — loss, latency, and the polarisation drift that a
    real-time compensator has to predict.

:mod:`.memory`
    How fast stored entanglement decays, and how long a pair stays useful.

:mod:`.link`
    How often a span produces a pair at all, and what that converts to in
    secret key.

:mod:`.coexistence`
    What the classical traffic sharing the same fibre costs the quantum
    channel, via spontaneous Raman scattering — the coupling that makes hybrid
    allocation a trade-off rather than two separate scheduling problems.

Everything is numpy, stateless and vectorised over leading axes, so a whole
batch of links or environments evaluates in one call.
"""

from __future__ import annotations

from . import bell_diagonal, coexistence, constants, fiber, link, memory
from .coexistence import (CoexistenceParameters, coexistence_qber,
                          max_classical_power, raman_noise_counts,
                          raman_noise_power)
from .bell_diagonal import bbpssw, decohere, dejmps, fidelity, swap, twirl, werner
from .fiber import (FiberParameters, PolarizationDriftModel, PolarizationTrajectory,
                    stokes_qber, transmittance)
from .link import LinkBudget, bb84_secret_key_rate, entanglement_rate_hz
from .memory import MemoryParameters, fidelity_after, time_to_fidelity

__all__ = [
    "bell_diagonal", "constants", "fiber", "link", "memory", "coexistence",
    "werner", "fidelity", "twirl", "decohere", "bbpssw", "dejmps", "swap",
    "FiberParameters", "PolarizationDriftModel", "PolarizationTrajectory",
    "transmittance", "stokes_qber",
    "MemoryParameters", "fidelity_after", "time_to_fidelity",
    "LinkBudget", "entanglement_rate_hz", "bb84_secret_key_rate",
    "CoexistenceParameters", "raman_noise_power", "raman_noise_counts",
    "coexistence_qber", "max_classical_power",
]
