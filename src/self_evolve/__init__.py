"""Output-distribution repair for self-evolving agents."""

from .benchmark import ACTION_LABELS, LogicRouteTask, generate_tasks
from .controller import DistributionRepairHead
from .repair import repair_distribution

__all__ = [
    "ACTION_LABELS",
    "DistributionRepairHead",
    "LogicRouteTask",
    "generate_tasks",
    "repair_distribution",
]

