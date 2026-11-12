# Copyright 2026 Mengdan Xue
# SPDX-License-Identifier: Apache-2.0
"""Simulated defaults only on explicit opt-in.

Many realtime modules fall back to a ``Simulated*Provider`` when no provider is
passed. A missing provider is a wiring state, not a request for made-up figures,
so the fallback goes through the same rule as ``PnLStreamer``: it is built only
when synthetic data is explicitly allowed (``ALLOW_MOCK_AGENT_DATA=true``, see
``mock_guard``); otherwise construction raises ``MockDataDisabledError`` naming
what to pass instead. Passing a real provider never needs the flag.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from .mock_guard import require_real_data

T = TypeVar("T")


def simulated_default(factory: Callable[[], T], context: str) -> T:
    """Build the simulated fallback for ``context`` - only if synthetic data is allowed."""
    require_real_data(context, "pass a real provider instead of relying on the simulated default")
    return factory()
