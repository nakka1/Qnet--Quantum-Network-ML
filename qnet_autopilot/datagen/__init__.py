"""Data generators: polarisation drift, link-level entanglement, hybrid traffic."""

from __future__ import annotations

from . import entanglement, polarization, schema, traffic
from .entanglement import LinkExperiment, generate_link_dataset
from .polarization import (chronological_split, generate_polarization_dataset,
                           make_supervised_windows)
from .schema import Dataset, DatasetMetadata, load
from .traffic import TrafficProfile, generate_capacity_trace, generate_demand_stream

__all__ = [
    "schema", "polarization", "entanglement", "traffic",
    "Dataset", "DatasetMetadata", "load",
    "generate_polarization_dataset", "make_supervised_windows", "chronological_split",
    "LinkExperiment", "generate_link_dataset",
    "TrafficProfile", "generate_capacity_trace", "generate_demand_stream",
]
