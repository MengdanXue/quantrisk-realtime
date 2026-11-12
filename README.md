# quantrisk-realtime

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![tests](https://github.com/MengdanXue/quantrisk-realtime/actions/workflows/tests.yml/badge.svg)](https://github.com/MengdanXue/quantrisk-realtime/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Real-time trading analytics — option pricing and Greeks, volatility surfaces, risk monitors, portfolio construction, execution analytics, signal generation — plus market-data and exchange adapters, extracted from the author's private trading-system prototype (TraderX / QuantRiskPro).

> **Read this first.** This is a large body of code (about 85k lines) whose **numerical correctness is largely unverified**. Its tests check that the code runs and behaves as each module's author intended, not that the numbers match an independent reference. Many modules can run on **simulated data providers**, but only when you ask for it (see below). Treat it as material to study and build on, not as a source of figures. Nothing here is investment advice, and nothing here places orders with a broker.

What came from where, and what changed on the way out, is recorded in [NOTICE](NOTICE) and [docs/PROVENANCE.md](docs/PROVENANCE.md). Sibling packages from the same host: [quantrisk-core](https://github.com/MengdanXue/quantrisk-core) (indicators, strategies, backtesting) and [agent-runtime-kit](https://github.com/MengdanXue/agent-runtime-kit) (an auditable LLM agent runtime).

## Install

```bash
pip install -e .                 # numpy, pandas, scipy, aiohttp, fastapi
pip install -e ".[dev]"          # + pytest, pytest-asyncio, pytest-benchmark, hypothesis and the stream extras the tests use
```

Extras: `stream` (websockets, redis, prometheus_client), `observability` (prometheus_client, opentelemetry), `ml` (scikit-learn, xgboost, torch, arch, statsmodels — imported lazily), `data` (akshare, ccxt), `alerts` (aiosmtplib). `fastapi` is a core dependency because importing `quantrisk_realtime.realtime` loads its P&L WebSocket router.

## What is in it

| Package | Files | What it covers |
|---|---|---|
| `quantrisk_realtime.realtime` | 56 | option pricing (Black–Scholes, binomial tree, Monte Carlo, Longstaff–Schwartz, barriers, finite differences), Greeks, volatility surfaces, derivatives indicators; risk monitors (drawdown, margin, liquidity, counterparty, portfolio, compliance, risk limits); portfolio optimisation, risk parity, rebalancing, position sizing; execution (order manager, smart order router, algo execution, slippage, market impact); signals (alpha, momentum, sentiment, order-flow, order-book imbalance, regime detection); P&L and market-data streaming |
| `quantrisk_realtime.models`, `online_learning` | 7 | ML, statistical and factor models; online learning |
| `quantrisk_realtime.datasources`, `exchanges` | 13 | market-data sources and exchange adapters over public APIs |
| `quantrisk_realtime.observability` | 4 | Prometheus metrics, OpenTelemetry tracing, structured logging |
| `quantrisk_realtime.core.mock_guard` | 1 | the single rule for when synthetic data may be served |

A spot check that runs as written:

```python
from quantrisk_realtime.realtime.option_pricing_engine import BlackScholesModel, OptionType

BlackScholesModel.price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.0, 0.2)   # 10.4506
BlackScholesModel.price(OptionType.PUT,  100, 100, 1.0, 0.05, 0.0, 0.2)   # 5.5735
```

Both values match the textbook Black–Scholes figures for these inputs, and put–call parity holds to rounding. That is one point, checked once — not a validation of the module.

## Simulated data only on request

Twenty-one `realtime` entry points (19 classes, 2 factory functions) ship a `Simulated*Provider` for each data source they read — among them `LiquidityRiskMonitor`, `SlippageTracker`, `ComplianceMonitor`, `CounterpartyRiskMonitor`, `CorrelationTracker`, `GreeksCalculator`, `VolatilitySurfaceManager`, `DerivativesIndicators`, `PortfolioRebalancer`, `RiskLimitsEngine`, `create_order_flow_analyzer` and `create_reconciliation_engine`. In the host they used it silently whenever no provider was passed. Here, as `PnLStreamer` already did, a missing provider is treated as a wiring gap rather than a request for made-up figures:

- pass your providers — nothing else is needed;
- pass no provider — construction raises `MockDataDisabledError` naming the class, the parameter and the simulated provider it would have used;
- set `ALLOW_MOCK_AGENT_DATA=true` (`core/mock_guard.py`) — the simulated providers are built as before, for local development; their numbers are random or fixed placeholders.

The same rule covers `RealtimeMarginMonitor` without a portfolio service (it used to invent margin figures) and `MarketDataAggregator.register_source` without a handler (it used to register a `SimulatedDataSource`). Passing a simulated provider explicitly is allowed without the flag: that is the caller's choice, not a silent fallback.

## Tests

```bash
pytest -q -m "not performance"     # correctness suite
pytest -q -m performance           # wall-clock benchmarks (machine-dependent)
```

On 2026-11-12, in a fresh Python 3.12 environment with `.[dev]`: **3416 passed, 3 skipped** (the three skips are timing-sensitive stop-loss tests marked as such in the host) and **13 benchmarks passed**; CI gives the same 3416 / 3 on Python 3.10, 3.11 and 3.12. The correctness suite needs no database or API key, and no network: it also passes with every outbound connection blocked (checked the same day); the realtime tests set `ALLOW_MOCK_AGENT_DATA` for themselves, because they exercise subscription, broadcasting and bookkeeping on simulated providers, not the figures; `tests/test_simulated_defaults.py` checks the opt-in rule for every guarded entry point.

## What is and is not verified

| | Verified here | Not verified |
|---|---|---|
| `realtime` (55 modules + `__init__`) | 54 have a test file (3000+ cases on module behaviour: inputs, state, thresholds, edge cases, async plumbing); Black–Scholes call/put spot-checked once against textbook values | **numerical correctness** against independent references (pricing beyond that spot check, Greeks, surfaces, VaR-type risk figures, optimisers); `dynamic_slippage_model` has no test file |
| `models`, `online_learning`, `observability` | every module imports | everything else — they came without tests |
| `datasources`, `exchanges` | adapter logic against mocked responses | any live API |
| Simulated data | every one of the 21 guarded entry points refuses without the opt-in and builds with it; an explicitly passed provider needs no flag | — |
| Packaging | every one of the 83 importable modules and packages imports with `.[dev]` | — |

No return figures appear in this repository.

## License

Apache-2.0. Some source headers carry the original project name (TraderX); see [NOTICE](NOTICE).
