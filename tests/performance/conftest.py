"""Keep the wall-clock benchmark suite out of the correctness gates.

Every test in this package measures elapsed time and asserts against a
threshold, so its result depends on the machine and on whatever else the
runner is doing. ``pytest.ini`` already states the policy - ``performance:
wall-clock performance tests that must run without coverage instrumentation``
- and both CI gates already filter with ``-m "not performance"``. What was
missing is the marker itself: these tests carry ``@pytest.mark.benchmark`` but
never ``@pytest.mark.performance``, so the filter did not match them and they
ran inside the gates anyway.

That is how ``test_p99_latency`` turned ``main`` red. It asserts P99 < 100ms,
and the coverage gate runs it under ``--cov`` instrumentation, which inflates
the very number being asserted; it measured 115.24ms. The uninstrumented gate
passed on the same commit.

Marking the package here applies the existing policy to the whole directory
instead of relying on each new benchmark remembering the decorator. Nothing is
skipped or deleted, and no threshold is touched: the tests still run, and still
assert exactly what they asserted before, whenever the suite is invoked
directly or with ``-m performance``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PERFORMANCE_SUITE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark only this package.

    ``pytest_collection_modifyitems`` is a session hook: even though this
    conftest lives in a subdirectory, it is handed every item pytest collected
    for the whole run. Marking unconditionally would put ``performance`` on the
    entire backend suite and leave both gates selecting nothing - which is a
    far worse failure than the one this file exists to fix, and a silent one.
    """
    for item in items:
        path = getattr(item, "path", None)
        if path is None:
            continue
        if Path(path).resolve().is_relative_to(PERFORMANCE_SUITE):
            item.add_marker(pytest.mark.performance)
