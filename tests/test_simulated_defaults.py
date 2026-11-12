# Copyright 2026 Mengdan Xue
# SPDX-License-Identifier: Apache-2.0
"""Simulated fallbacks are built only on explicit opt-in (ALLOW_MOCK_AGENT_DATA).

Every entry point that used to fall back to a ``Simulated*Provider`` silently now
refuses without the flag, builds with it, and never needs it when the caller passes
the provider - including a simulated one, which is then the caller's explicit choice.
"""

import importlib

import pytest

from quantrisk_realtime.core.mock_guard import MOCK_FLAG_ENV_VAR, MockDataDisabledError

R = "quantrisk_realtime.realtime."


def _cfg(module, name):
    return lambda: (getattr(importlib.import_module(R + module), name)(),)


# (module, callable, positional args) for every guarded entry point
ENTRIES = [
    ("benchmark_tracker", "BenchmarkTracker", None),
    ("compliance_monitor", "ComplianceMonitor", None),
    ("correlation_tracker", "CorrelationTracker", None),
    ("counterparty_risk_monitor", "CounterpartyRiskMonitor", None),
    ("derivatives_indicators", "DerivativesIndicators", _cfg("derivatives_indicators", "DerivativesConfig")),
    ("execution_monitor", "ExecutionMonitor", None),
    ("funding_rate_monitor", "FundingRateMonitor", None),
    ("greeks_calculator", "GreeksCalculator", _cfg("greeks_calculator", "CalculatorConfig")),
    ("hedging_engine", "HedgingEngine", _cfg("hedging_engine", "EngineConfig")),
    ("liquidity_risk_monitor", "LiquidityRiskMonitor", None),
    ("market_regime_detector", "MarketRegimeDetector", None),
    ("ml_feature_engine", "MLFeatureEngine", _cfg("ml_feature_engine", "EngineConfig")),
    ("ml_inference_service", "MLInferenceService", None),
    ("order_flow_analyzer", "create_order_flow_analyzer", None),
    ("portfolio_rebalancer", "PortfolioRebalancer", None),
    ("position_reconciliation", "create_reconciliation_engine", None),
    ("risk_limits_engine", "RiskLimitsEngine", None),
    ("sentiment_analyzer", "SentimentAnalyzer", None),
    ("slippage_tracker", "SlippageTracker", None),
    ("trade_signal_aggregator", "TradeSignalAggregator", None),
    ("volatility_surface_manager", "VolatilitySurfaceManager", _cfg("volatility_surface_manager", "SurfaceManagerConfig")),
]


def _build(module, name, args):
    target = getattr(importlib.import_module(R + module), name)
    return target(*(args() if args else ()))


@pytest.mark.parametrize("module,name,args", ENTRIES, ids=[e[1] for e in ENTRIES])
def test_without_opt_in_a_missing_provider_refuses(monkeypatch, module, name, args):
    monkeypatch.delenv(MOCK_FLAG_ENV_VAR, raising=False)
    with pytest.raises(MockDataDisabledError, match="simulated"):
        _build(module, name, args)


@pytest.mark.parametrize("module,name,args", ENTRIES, ids=[e[1] for e in ENTRIES])
def test_with_opt_in_the_simulated_default_is_built(monkeypatch, module, name, args):
    monkeypatch.setenv(MOCK_FLAG_ENV_VAR, "true")
    assert _build(module, name, args) is not None


def test_passing_the_providers_needs_no_opt_in(monkeypatch):
    monkeypatch.delenv(MOCK_FLAG_ENV_VAR, raising=False)
    from quantrisk_realtime.realtime import liquidity_risk_monitor as L

    monitor = L.LiquidityRiskMonitor(order_book_provider=L.SimulatedOrderBookProvider(),
                                     volume_provider=L.SimulatedVolumeDataProvider(),
                                     position_provider=L.SimulatedPositionDataProvider())
    assert isinstance(monitor._volume_provider, L.SimulatedVolumeDataProvider)


@pytest.mark.asyncio
async def test_margin_monitor_without_a_portfolio_service_refuses_instead_of_inventing_margin(monkeypatch):
    from quantrisk_realtime.realtime.margin_monitor import RealtimeMarginMonitor

    monkeypatch.delenv(MOCK_FLAG_ENV_VAR, raising=False)
    monitor = RealtimeMarginMonitor(account_id="acct-1")
    with pytest.raises(MockDataDisabledError, match="margin status for account acct-1"):
        await monitor._get_margin_status()          # not swallowed by the catch-all handler
    monkeypatch.setenv(MOCK_FLAG_ENV_VAR, "true")
    status = await monitor._get_margin_status()
    assert status.total_equity == 1000000.0         # the host's synthetic figures, now opt-in only


def test_aggregator_without_a_handler_refuses_the_simulated_source(monkeypatch):
    from quantrisk_realtime.realtime.market_data_aggregator import DataSource, DataSourceConfig, MarketDataAggregator

    config = DataSourceConfig(source_id="s1", source_type=DataSource.SIMULATION, name="s1")
    monkeypatch.delenv(MOCK_FLAG_ENV_VAR, raising=False)
    with pytest.raises(MockDataDisabledError, match="market data for source s1"):
        MarketDataAggregator().register_source(config)
    monkeypatch.setenv(MOCK_FLAG_ENV_VAR, "true")
    assert MarketDataAggregator().register_source(config) is True
