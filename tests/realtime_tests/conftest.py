# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the realtime engine tests."""

from __future__ import annotations

import pytest

from quantrisk_realtime.core.mock_guard import MOCK_FLAG_ENV_VAR


@pytest.fixture(autouse=True)
def allow_synthetic_pnl(monkeypatch):
    """Let the P&L streamer tests run without a portfolio service.

    ``PnLStreamer`` refuses to invent P&L numbers unless synthetic data is
    explicitly allowed, because a missing ``portfolio_service`` is a wiring
    state and not a request for made-up figures.

    These suites exercise subscription bookkeeping, broadcasting and heartbeat
    plumbing - not the correctness of any P&L number - so they opt in through
    the same public flag a developer would use. The guard itself is covered by
    ``tests/agents/test_agent_mock_guard.py``.
    """
    monkeypatch.setenv(MOCK_FLAG_ENV_VAR, "true")
