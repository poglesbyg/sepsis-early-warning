"""The experiment layer: one module per question, explicitly registered.

A self-describing registry with auto-discovery was considered and rejected. It
would be an abstraction and a schema built for experiments that may never exist,
to avoid edits in four files. What is here instead is the part that actually
earns its keep -- a shared fit-and-score helper so the experiments do not each
reimplement it -- plus this list, which is read top to bottom and can be
understood without documentation.

Adding an experiment means writing the module and adding one line below.
"""

from __future__ import annotations

from typing import Callable

from .common import ExperimentResult, assert_partition, block_columns, fit_and_score
from . import ablation, leakage

# Explicit, ordered, greppable. Report sections appear in this order.
REGISTRY: dict[str, Callable[..., ExperimentResult]] = {
    "leakage": leakage.run,
    "ablation": ablation.run,
}

__all__ = [
    "REGISTRY",
    "ExperimentResult",
    "assert_partition",
    "block_columns",
    "fit_and_score",
]
