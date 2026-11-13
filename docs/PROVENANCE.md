# Provenance: what was copied, what was changed

quantrisk-realtime is an extraction, not a rewrite. This page records what came from where and every change made while taking it out of its host, so a reader can tell packaging from original logic.

## Source

- **Host project**: TraderX ("QuantRiskPro V2.0"), the author's private trading-management prototype (2025–2026; Java + FastAPI + React), written alone. See `NOTICE`.
- **Copied on**: 2026-11-12 from TraderX revision `156d265e` (files taken with `git show 156d265e:<path>`, so the host's uncommitted state played no part), into a fresh git history.
- **Siblings**: [quantrisk-core](https://github.com/MengdanXue/quantrisk-core) (indicators, strategies, backtesting) and [agent-runtime-kit](https://github.com/MengdanXue/agent-runtime-kit) (agent runtime) were extracted from the same host earlier.

## Copied from `backend-python/`

| Host path | Here | Size |
|---|---|---|
| `src/engine/realtime/**` | `src/quantrisk_realtime/realtime/` | 56 files, ~73.8k lines |
| `src/engine/models/**` except `signal_providers.py` | `src/quantrisk_realtime/models/` | 4 files, ~1.8k lines |
| `src/engine/online_learning/**` | `src/quantrisk_realtime/online_learning/` | 3 files, ~0.7k lines |
| `src/datasources/**` | `src/quantrisk_realtime/datasources/` | 9 files, ~4.9k lines |
| `src/exchanges/**` | `src/quantrisk_realtime/exchanges/` | 4 files, ~2.5k lines |
| `src/observability/**` | `src/quantrisk_realtime/observability/` | 4 files, ~1.3k lines |
| `src/core/mock_guard.py` | `src/quantrisk_realtime/core/mock_guard.py` | the rule for when synthetic data may be served |
| `tests/realtime_tests/**`, `tests/datasources/**`, `tests/exchange_tests/**`, `tests/performance/test_indicator_benchmarks.py` + its `conftest.py`, `tests/property/test_indicator_properties.py`, `tests/test_websocket_performance.py` | `tests/` (same layout) | 64 test files |

The selection rule: a host package came along only if it imports nothing else from the host (checked for absolute `src.*` imports and relative `..x` imports), and a test only if every host module it imports came along.

## Not copied

| Host path | Why |
|---|---|
| `src/middleware/` | imports the host's `config.system_parameters` and, at runtime, `database.session`: it is the host application's JWT auth and rate limiting, not a reusable piece |
| `src/engine/models/signal_providers.py` | adapters for the host's decision engine V2 (`engine/decision`), which depends on host services and configuration |
| `src/core/container.py` | the host's dependency-injection container; nothing copied needs it |
| `tests/engine/test_realtime_risk.py`, `tests/test_oms_ownership.py`, `tests/test_oms_persistence.py`, `tests/test_observability.py`, `tests/integration/*`, `tests/integration_tests/*` that touch these packages | they also import host modules that were not copied (`src.api`, `src.database`, `src.models`, `src.tools`, `src.backtest`, `src.strategies`, `src.main`) |
| everything else in TraderX | other packages are tied to the host's database, services and configuration, or were extracted into the sibling repositories; the host's documents, data files, uploads and reports are not part of any extraction |

## Changes made during extraction

Changes 1–4 are mechanical or boundary-related; change 5 is the one deliberate behaviour change. **No computation or threshold was altered.**

1. **Package rename**: `src.engine.realtime` → `quantrisk_realtime.realtime`; `src.engine.models` / `online_learning` → `quantrisk_realtime.models` / `online_learning`; `src.datasources`, `src.exchanges`, `src.observability`, `src.core` → `quantrisk_realtime.*`. In the tests, `engine.realtime` (imported after the host's `sys.path` insert) became `quantrisk_realtime.realtime`, including in `mock.patch` target strings. One source file (`realtime/pnl_streamer.py`, its `mock_guard` import) and 65 test files changed.
2. **Boundaries**: `models/__init__.py` no longer imports or exports the four names from `signal_providers.py`; `core/__init__.py` is a new docstring-only module (the host's version re-exported `container`, which was not copied); `quantrisk_realtime/__init__.py` is new.
3. **Test configuration** (in `pyproject.toml`, from the host's `pytest.ini`): `asyncio_mode = auto`, function-scoped asyncio fixtures, and the `performance` marker the copied `tests/performance/conftest.py` applies, so wall-clock benchmarks stay out of the correctness run exactly as in the host's CI.
4. **New test**: `tests/test_imports.py` imports every module, because `models`, `online_learning` and `observability` came without tests of their own.
5. **Simulated fallbacks need an explicit opt-in** (behaviour change, approved by the author on 2026-11-12). In the host, 21 entry points silently built a `Simulated*Provider` when none was passed (40 `x or SimulatedX()` defaults in 19 classes and `create_reconciliation_engine`, plus two in `create_order_flow_analyzer`), `RealtimeMarginMonitor` invented margin figures without a portfolio service, and `MarketDataAggregator.register_source` registered a `SimulatedDataSource` without a handler. These now go through `require_real_data` — via the new `core/simulated.py` (`simulated_default`), and in `RealtimeMarginMonitor` directly, with `MockDataDisabledError` re-raised ahead of its catch-all handler — the rule `PnLStreamer` already followed. With `ALLOW_MOCK_AGENT_DATA=true` everything behaves as in the host; the copied tests already set it. New tests: `tests/test_simulated_defaults.py`. `quick_reconcile` was left alone: it fills in-memory providers with the positions the caller passes, so nothing is invented.

## Known behaviour carried over unchanged

- Source comments and docstrings are partly in Chinese, as in the host.
