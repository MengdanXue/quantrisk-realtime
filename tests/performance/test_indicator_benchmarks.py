# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Performance benchmarks for indicator calculations.

These tests ensure that critical indicator calculations meet
latency requirements for real-time trading:
- PCR calculation: < 10ms
- Volatility skew: < 20ms
- Sentiment analysis: < 30ms
- Full snapshot: < 50ms
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.derivatives_indicators import (
    OptionType,
    PCRType,
    OptionData,
    FuturesContract,
    DerivativesConfig,
    DerivativesIndicators,
    SimulatedOptionDataProvider,
    SimulatedFuturesDataProvider,
)
from quantrisk_realtime.realtime.pnl_streamer import (
    PnLStreamer,
    StreamerConfig,
    PortfolioPnL,
    PositionPnL,
)
# Note: CorrelationTracker uses TrackerConfig, not CorrelationConfig
# Correlation benchmarks use numpy directly for simplicity


# ============ Fixtures ============


@pytest.fixture
def large_option_chain() -> List[OptionData]:
    """Generate a large option chain for benchmark testing."""
    now = datetime.now()
    options = []
    spot = 100.0

    # Generate options for multiple expiries and strikes
    for days_to_expiry in [7, 14, 30, 60, 90]:
        expiry = now + timedelta(days=days_to_expiry)
        for strike in range(80, 121, 1):  # 41 strikes
            # Call option
            options.append(OptionData(
                symbol="TEST",
                option_symbol=f"TEST{expiry.strftime('%y%m%d')}C{strike}",
                option_type=OptionType.CALL,
                strike=float(strike),
                expiry=expiry,
                spot_price=spot,
                volume=np.random.randint(100, 10000),
                open_interest=np.random.randint(1000, 50000),
                bid=max(0, spot - strike) + np.random.uniform(0.5, 2.0),
                ask=max(0, spot - strike) + np.random.uniform(2.0, 3.0),
                last=max(0, spot - strike) + np.random.uniform(1.0, 2.5),
                implied_vol=0.20 + abs(strike - spot) * 0.002 + np.random.uniform(-0.02, 0.02),
                delta=0.5 - (strike - spot) * 0.02,
            ))

            # Put option
            options.append(OptionData(
                symbol="TEST",
                option_symbol=f"TEST{expiry.strftime('%y%m%d')}P{strike}",
                option_type=OptionType.PUT,
                strike=float(strike),
                expiry=expiry,
                spot_price=spot,
                volume=np.random.randint(100, 10000),
                open_interest=np.random.randint(1000, 50000),
                bid=max(0, strike - spot) + np.random.uniform(0.5, 2.0),
                ask=max(0, strike - spot) + np.random.uniform(2.0, 3.0),
                last=max(0, strike - spot) + np.random.uniform(1.0, 2.5),
                implied_vol=0.20 + abs(strike - spot) * 0.003 + np.random.uniform(-0.02, 0.02),
                delta=-0.5 + (strike - spot) * 0.02,
            ))

    return options  # 410 options (5 expiries * 41 strikes * 2 types)


@pytest.fixture
def derivatives_config():
    """Create config for derivatives indicators."""
    return DerivativesConfig(
        update_interval_ms=1000,
        history_size=100,
    )


@pytest.fixture
def derivatives_indicators(derivatives_config):
    """Create derivatives indicators instance."""
    return DerivativesIndicators(
        config=derivatives_config,
        option_provider=SimulatedOptionDataProvider(),
        futures_provider=SimulatedFuturesDataProvider(),
    )


@pytest.fixture
def pnl_streamer():
    """Create PnL streamer instance."""
    config = StreamerConfig(
        update_interval_seconds=1.0,
        heartbeat_interval_seconds=30.0,
    )
    return PnLStreamer(config=config)


@pytest.fixture
def large_portfolio():
    """Generate a large portfolio for benchmark testing."""
    positions = []
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'QCOM']

    for i in range(100):  # 100 positions
        symbol = symbols[i % len(symbols)]
        positions.append(PositionPnL(
            position_id=i + 1,
            symbol=f"{symbol}_{i}",
            current_price=np.random.uniform(50, 500),
            average_cost=np.random.uniform(45, 480),
            quantity=np.random.randint(10, 1000),
            direction=1 if np.random.random() > 0.3 else -1,
            unrealized_pnl=np.random.uniform(-5000, 10000),
            market_value=np.random.uniform(5000, 500000),
        ))

    return PortfolioPnL(
        portfolio_id=1,
        account_id="BENCH001",
        total_unrealized_pnl=sum(p.unrealized_pnl for p in positions),
        total_market_value=sum(p.market_value for p in positions),
        positions=positions,
    )


@pytest.fixture
def price_series():
    """Generate price series for correlation testing."""
    np.random.seed(42)
    n_days = 252  # 1 year of trading days
    n_assets = 20

    # Generate correlated returns
    returns = np.random.multivariate_normal(
        mean=[0.0001] * n_assets,
        cov=np.eye(n_assets) * 0.0004,
        size=n_days,
    )

    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return prices


# ============ PCR Calculation Benchmarks ============


class TestPCRBenchmarks:
    """Benchmark tests for PCR calculations."""

    @pytest.mark.benchmark(group="pcr")
    def test_pcr_calculation_speed(self, benchmark, derivatives_indicators, large_option_chain):
        """
        PCR calculation should complete within 10ms.

        Target: < 10ms for 400+ options
        """
        provider = derivatives_indicators._option_provider
        provider.set_option_chain("TEST", large_option_chain)
        derivatives_indicators.add_symbol("TEST")

        async def calculate_pcr():
            return await derivatives_indicators._calculate_pcr("TEST", large_option_chain)

        import asyncio
        result = benchmark(lambda: asyncio.run(calculate_pcr()))

        # Verify result is valid
        assert result is not None
        assert len(result) > 0

        # Check performance (10ms = 0.010 seconds)
        assert benchmark.stats["mean"] < 0.010, f"PCR calculation too slow: {benchmark.stats['mean']*1000:.2f}ms"

    @pytest.mark.benchmark(group="pcr")
    def test_pcr_interpretation_speed(self, benchmark, derivatives_indicators):
        """
        PCR interpretation should be instant (< 1ms).
        """
        pcr_values = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]

        def interpret_all():
            return [derivatives_indicators._interpret_pcr(v) for v in pcr_values]

        result = benchmark(interpret_all)

        assert len(result) == len(pcr_values)
        assert benchmark.stats["mean"] < 0.001, "PCR interpretation too slow"


# ============ Volatility Skew Benchmarks ============


class TestVolatilitySkewBenchmarks:
    """Benchmark tests for volatility skew calculations."""

    @pytest.mark.benchmark(group="skew")
    def test_volatility_skew_calculation(self, benchmark, derivatives_indicators, large_option_chain):
        """
        Volatility skew calculation should complete within 20ms.

        Target: < 20ms for 400+ options
        """
        provider = derivatives_indicators._option_provider
        provider.set_option_chain("TEST", large_option_chain)
        derivatives_indicators.add_symbol("TEST")

        async def calculate_skew():
            return await derivatives_indicators._calculate_volatility_skew("TEST", large_option_chain)

        import asyncio
        result = benchmark(lambda: asyncio.run(calculate_skew()))

        assert isinstance(result, list)
        assert benchmark.stats["mean"] < 0.020, f"Skew calculation too slow: {benchmark.stats['mean']*1000:.2f}ms"


# ============ Sentiment Analysis Benchmarks ============


class TestSentimentBenchmarks:
    """Benchmark tests for sentiment analysis."""

    @pytest.mark.benchmark(group="sentiment")
    def test_sentiment_calculation(self, benchmark, derivatives_indicators, large_option_chain):
        """
        Sentiment calculation should complete within 30ms.

        Target: < 30ms for 400+ options
        """
        provider = derivatives_indicators._option_provider
        provider.set_option_chain("TEST", large_option_chain)
        derivatives_indicators.add_symbol("TEST")

        async def calculate_sentiment():
            return await derivatives_indicators._calculate_sentiment("TEST", large_option_chain)

        import asyncio
        result = benchmark(lambda: asyncio.run(calculate_sentiment()))

        assert result is not None
        assert benchmark.stats["mean"] < 0.030, f"Sentiment calculation too slow: {benchmark.stats['mean']*1000:.2f}ms"


# ============ Full Update Cycle Benchmarks ============


class TestFullUpdateBenchmarks:
    """Benchmark tests for full indicator update cycles."""

    @pytest.mark.benchmark(group="full_update")
    def test_full_indicator_update(self, benchmark, derivatives_indicators, large_option_chain):
        """
        Full indicator update should complete within 50ms.

        This includes PCR, skew, sentiment, and term structure.
        Target: < 50ms for complete update cycle
        """
        provider = derivatives_indicators._option_provider
        provider.set_option_chain("TEST", large_option_chain)
        derivatives_indicators.add_symbol("TEST")

        async def full_update():
            await derivatives_indicators._update_indicators("TEST")

        import asyncio
        benchmark(lambda: asyncio.run(full_update()))

        # Verify data was populated
        assert derivatives_indicators.get_all_pcr("TEST") is not None

        assert benchmark.stats["mean"] < 0.050, f"Full update too slow: {benchmark.stats['mean']*1000:.2f}ms"


# ============ PnL Streaming Benchmarks ============


class TestPnLStreamingBenchmarks:
    """Benchmark tests for PnL streaming operations."""

    @pytest.mark.benchmark(group="pnl")
    def test_portfolio_pnl_serialization(self, benchmark, large_portfolio):
        """
        Portfolio P&L serialization should be fast.

        Target: < 5ms for 100 positions
        """
        result = benchmark(lambda: large_portfolio.to_dict(include_positions=True))

        assert "positions" in result
        assert len(result["positions"]) == 100
        assert benchmark.stats["mean"] < 0.005, f"Serialization too slow: {benchmark.stats['mean']*1000:.2f}ms"

    @pytest.mark.benchmark(group="pnl")
    def test_pnl_update_threshold_check(self, benchmark, pnl_streamer, large_portfolio):
        """
        P&L update threshold check should be instant.

        Target: < 1ms
        """
        # Set up cache
        pnl_streamer._portfolio_pnl_cache[1] = large_portfolio

        # Create slightly modified portfolio
        modified = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=large_portfolio.total_unrealized_pnl * 1.02,
            total_market_value=large_portfolio.total_market_value * 1.001,
        )

        result = benchmark(lambda: pnl_streamer._should_send_update(1, modified))

        assert isinstance(result, bool)
        assert benchmark.stats["mean"] < 0.001, "Threshold check too slow"


# ============ Correlation Calculation Benchmarks ============


class TestCorrelationBenchmarks:
    """Benchmark tests for correlation calculations."""

    @pytest.mark.benchmark(group="correlation")
    def test_correlation_matrix_calculation(self, benchmark, price_series):
        """
        Correlation matrix calculation for 20 assets.

        Target: < 10ms for 252 days x 20 assets
        """
        def calculate_correlation():
            return np.corrcoef(price_series.T)

        result = benchmark(calculate_correlation)

        assert result.shape == (20, 20)
        assert benchmark.stats["mean"] < 0.010, f"Correlation calculation too slow: {benchmark.stats['mean']*1000:.2f}ms"

    @pytest.mark.benchmark(group="correlation")
    def test_rolling_correlation(self, benchmark, price_series):
        """
        Rolling correlation calculation.

        Target: < 50ms for 252 days rolling window
        """
        window = 20

        def rolling_corr():
            results = []
            for i in range(window, len(price_series)):
                window_data = price_series[i-window:i]
                corr = np.corrcoef(window_data.T)
                results.append(corr[0, 1])  # First pair correlation
            return results

        result = benchmark(rolling_corr)

        assert len(result) == len(price_series) - window
        assert benchmark.stats["mean"] < 0.050, f"Rolling correlation too slow: {benchmark.stats['mean']*1000:.2f}ms"


# ============ Memory Efficiency Benchmarks ============


class TestMemoryBenchmarks:
    """Benchmark tests for memory efficiency."""

    @pytest.mark.benchmark(group="memory")
    def test_option_data_memory(self, benchmark, large_option_chain):
        """
        Option chain should have reasonable memory footprint.
        """
        import sys

        def measure_size():
            return sum(sys.getsizeof(opt) for opt in large_option_chain)

        size = benchmark(measure_size)

        # Each option should be < 1KB
        avg_size = size / len(large_option_chain)
        assert avg_size < 1024, f"Option data too large: {avg_size:.0f} bytes each"

    @pytest.mark.benchmark(group="memory")
    def test_snapshot_creation(self, benchmark, derivatives_indicators, large_option_chain):
        """
        Snapshot creation should be memory efficient.
        """
        provider = derivatives_indicators._option_provider
        provider.set_option_chain("TEST", large_option_chain)
        derivatives_indicators.add_symbol("TEST")

        import asyncio
        asyncio.run(derivatives_indicators._update_indicators("TEST"))

        def get_snapshot():
            return derivatives_indicators.get_snapshot("TEST")

        result = benchmark(get_snapshot)

        assert result is not None
        assert benchmark.stats["mean"] < 0.001, "Snapshot retrieval too slow"


# ============ Throughput Benchmarks ============


class TestThroughputBenchmarks:
    """Benchmark tests for throughput."""

    @pytest.mark.benchmark(group="throughput")
    def test_tick_processing_throughput(self, benchmark):
        """
        Measure tick processing throughput.

        Target: > 10,000 ticks/second
        """
        # Simulate tick data
        ticks = [
            {"symbol": f"SYM{i%10}", "price": 100 + np.random.randn(), "volume": 100}
            for i in range(1000)
        ]

        processed = []

        def process_ticks():
            processed.clear()
            for tick in ticks:
                # Simulate basic processing
                processed.append({
                    "symbol": tick["symbol"],
                    "price": tick["price"],
                    "vwap": tick["price"] * tick["volume"],
                })
            return len(processed)

        result = benchmark(process_ticks)

        assert result == 1000

        # Calculate throughput
        ticks_per_second = 1000 / benchmark.stats["mean"]
        assert ticks_per_second > 10000, f"Throughput too low: {ticks_per_second:.0f} ticks/sec"


# ============ Latency Percentile Benchmarks ============


class TestLatencyPercentiles:
    """Test latency percentiles for critical operations."""

    @pytest.mark.benchmark(group="latency", min_rounds=100)
    def test_p99_latency(self, benchmark, derivatives_indicators, large_option_chain):
        """
        P99 latency should be within acceptable bounds.

        Target: P99 < 100ms
        """
        provider = derivatives_indicators._option_provider
        provider.set_option_chain("TEST", large_option_chain)
        derivatives_indicators.add_symbol("TEST")

        async def update():
            await derivatives_indicators._update_indicators("TEST")

        import asyncio
        benchmark(lambda: asyncio.run(update()))

        # P99 should be under 100ms
        # Note: pytest-benchmark calculates percentiles automatically
        mean = benchmark.stats["mean"]
        stddev = benchmark.stats.get("stddev", mean * 0.1)

        # Estimate P99 (mean + 2.33 * stddev for normal distribution)
        estimated_p99 = mean + 2.33 * stddev
        assert estimated_p99 < 0.100, f"Estimated P99 too high: {estimated_p99*1000:.2f}ms"
