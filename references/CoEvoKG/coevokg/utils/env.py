"""Environment-variable helpers for CoEvoKG runtime settings."""

from __future__ import annotations

import os
from typing import Optional


def get_coevokg_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read COEVOKG_<name>."""
    return os.getenv(f"COEVOKG_{name}", default)


def setdefault_coevokg_env(name: str, value: object) -> str:
    """Set COEVOKG_<name> unless it is already provided."""
    key = f"COEVOKG_{name}"
    os.environ.setdefault(key, str(value))
    return os.environ[key]
