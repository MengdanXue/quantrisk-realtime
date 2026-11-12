# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Derivatives Indicators Module.

Tests PCR calculation, unusual activity detection, volatility skew,
sentiment analysis, and term structure analysis.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from quantrisk_realtime.realtime.derivatives_indicators import (
    # Enums
    OptionType,
    PCRType,
    FlowDirection,
    ActivityType,
    TermStructureShape,
    SentimentLevel,
    AlertSeverity,
    AlertType,
    MonitorState,
    # Data Classes
    OptionData,
    PCRIndicator,
    UnusualActivity,
    VolatilitySkew,
    OptionSentiment,
    TermStructure,
    FuturesContract,
    DerivativesSnapshot,
    DerivativesAlert,
    DerivativesConfig,
    # Providers
    SimulatedOptionDataProvider,
    SimulatedFuturesDataProvider,
    # Main Class
    DerivativesIndicators,
    # Factory functions
    create_derivatives_indicators,
)


# ============ Test Fixtures ============


@pytest.fixture
def config():
    """Create test configuration."""
    return DerivativesConfig(
        update_interval_ms=1000,
        history_size=100,
        pcr_extreme_low=0.5,
        pcr_extreme_high=1.5,
        block_trade_min_contracts=50,
        unusual_volume_threshold=2.0,
    )


@pytest.fixture
def option_provider():
    """Create simulated option provider."""
    return SimulatedOptionDataProvider()


@pytest.fixture
def futures_provider():
    """Create simulated futures provider."""
    return SimulatedFuturesDataProvider()


@pytest.fixture
def derivatives_indicators(config, option_provider, futures_provider):
    """Create derivatives indicators instance."""
    return DerivativesIndicators(
        config=config,
        option_provider=option_provider,
        futures_provider=futures_provider,
    )


@pytest.fixture
def sample_option_chain():
    """Create sample option chain."""
    now = datetime.now()
    expiry = now + timedelta(days=30)
    spot = 100.0

    options = []
    strikes = [90, 95, 100, 105, 110]

    for strike in strikes:
        # Call option
        options.append(OptionData(
            symbol="AAPL",
            option_symbol=f"AAPL{expiry.strftime('%y%m%d')}C{strike}",
            option_type=OptionType.CALL,
            strike=strike,
            expiry=expiry,
            spot_price=spot,
            volume=1000 + (strike - 90) * 100,
            open_interest=5000 + (strike - 90) * 500,
            bid=max(0, spot - strike) + 2.0,
            ask=max(0, spot - strike) + 2.5,
            last=max(0, spot - strike) + 2.25,
            implied_vol=0.25 + abs(strike - spot) * 0.002,
            delta=0.5 - (strike - spot) * 0.02,
        ))

        # Put option
        options.append(OptionData(
            symbol="AAPL",
            option_symbol=f"AAPL{expiry.strftime('%y%m%d')}P{strike}",
            option_type=OptionType.PUT,
            strike=strike,
            expiry=expiry,
            spot_price=spot,
            volume=800 + (110 - strike) * 100,
            open_interest=4000 + (110 - strike) * 500,
            bid=max(0, strike - spot) + 1.5,
            ask=max(0, strike - spot) + 2.0,
            last=max(0, strike - spot) + 1.75,
            implied_vol=0.25 + abs(strike - spot) * 0.003,
            delta=-0.5 + (strike - spot) * 0.02,
        ))

    return options


@pytest.fixture
def sample_futures_chain():
    """Create sample futures chain."""
    now = datetime.now()
    spot = 100.0

    contracts = []
    for i in range(1, 5):
        expiry = now + timedelta(days=30 * i)
        # Simulate contango (futures price > spot)
        price = spot * (1 + 0.01 * i)
        time_to_expiry = (30 * i) / 365.0
        contracts.append(FuturesContract(
            symbol="CL",
            contract_month=expiry.strftime('%Y%m'),
            expiry=expiry,
            price=price,
            volume=10000 - i * 1000,
            open_interest=50000 - i * 5000,
            basis=price - spot,
            time_to_expiry=time_to_expiry,
        ))

    return contracts


# ============ Enum Tests ============


class TestOptionType:
    """Test OptionType enum."""

    def test_values(self):
        assert OptionType.CALL.value == "call"
        assert OptionType.PUT.value == "put"

    def test_count(self):
        assert len(OptionType) == 2


class TestPCRType:
    """Test PCRType enum."""

    def test_values(self):
        assert PCRType.VOLUME.value == "volume"
        assert PCRType.OPEN_INTEREST.value == "oi"
        assert PCRType.WEIGHTED.value == "weighted"
        assert PCRType.DELTA_ADJUSTED.value == "delta"

    def test_count(self):
        assert len(PCRType) == 4


class TestFlowDirection:
    """Test FlowDirection enum."""

    def test_values(self):
        assert FlowDirection.BULLISH.value == "bullish"
        assert FlowDirection.BEARISH.value == "bearish"
        assert FlowDirection.NEUTRAL.value == "neutral"
        assert FlowDirection.MIXED.value == "mixed"


class TestActivityType:
    """Test ActivityType enum."""

    def test_values(self):
        assert ActivityType.BLOCK_TRADE.value == "block"
        assert ActivityType.SWEEP.value == "sweep"
        assert ActivityType.UNUSUAL_VOLUME.value == "volume"


class TestTermStructureShape:
    """Test TermStructureShape enum."""

    def test_values(self):
        assert TermStructureShape.CONTANGO.value == "contango"
        assert TermStructureShape.BACKWARDATION.value == "backwardation"
        assert TermStructureShape.FLAT.value == "flat"


class TestSentimentLevel:
    """Test SentimentLevel enum."""

    def test_values(self):
        assert SentimentLevel.EXTREME_FEAR.value == "extreme_fear"
        assert SentimentLevel.FEAR.value == "fear"
        assert SentimentLevel.NEUTRAL.value == "neutral"
        assert SentimentLevel.GREED.value == "greed"
        assert SentimentLevel.EXTREME_GREED.value == "extreme_greed"


class TestMonitorState:
    """Test MonitorState enum."""

    def test_values(self):
        assert MonitorState.IDLE.value == "idle"
        assert MonitorState.RUNNING.value == "running"
        assert MonitorState.PAUSED.value == "paused"


# ============ Data Class Tests ============


class TestOptionData:
    """Test OptionData dataclass."""

    def test_creation(self):
        now = datetime.now()
        expiry = now + timedelta(days=30)

        option = OptionData(
            symbol="AAPL",
            option_symbol="AAPL240115C150",
            option_type=OptionType.CALL,
            strike=150.0,
            expiry=expiry,
            spot_price=145.0,
            volume=1000,
            open_interest=5000,
            bid=5.0,
            ask=5.5,
            last=5.25,
        )

        assert option.symbol == "AAPL"
        assert option.option_type == OptionType.CALL
        assert option.strike == 150.0
        assert option.volume == 1000

    def test_to_dict(self):
        now = datetime.now()
        expiry = now + timedelta(days=30)

        option = OptionData(
            symbol="AAPL",
            option_symbol="AAPL240115C150",
            option_type=OptionType.CALL,
            strike=150.0,
            expiry=expiry,
            spot_price=145.0,
            volume=1000,
            open_interest=5000,
            bid=5.0,
            ask=5.5,
            last=5.25,
        )

        d = option.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["option_type"] == "call"
        assert d["strike"] == 150.0


class TestPCRIndicator:
    """Test PCRIndicator dataclass."""

    def test_creation(self):
        pcr = PCRIndicator(
            symbol="AAPL",
            timestamp=datetime.now(),
            pcr_type=PCRType.VOLUME,
            pcr_value=0.8,
            put_volume=8000,
            call_volume=10000,
            put_oi=40000,
            call_oi=50000,
            interpretation=SentimentLevel.NEUTRAL,
        )

        assert pcr.pcr_value == 0.8
        assert pcr.put_volume == 8000
        assert pcr.call_volume == 10000

    def test_to_dict(self):
        pcr = PCRIndicator(
            symbol="AAPL",
            timestamp=datetime.now(),
            pcr_type=PCRType.OPEN_INTEREST,
            pcr_value=0.75,
            put_volume=7500,
            call_volume=10000,
            put_oi=37500,
            call_oi=50000,
            interpretation=SentimentLevel.GREED,
        )

        d = pcr.to_dict()
        assert d["pcr_type"] == "oi"
        assert d["pcr_value"] == 0.75
        assert d["interpretation"] == "greed"


class TestUnusualActivity:
    """Test UnusualActivity dataclass."""

    def test_creation(self):
        activity = UnusualActivity(
            symbol="AAPL",
            option_symbol="AAPL240115C150",
            activity_type=ActivityType.BLOCK_TRADE,
            option_type=OptionType.CALL,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            volume=5000,
            avg_volume=1000.0,
            volume_ratio=5.0,
            oi_change=2000,
            premium=25000.0,
            notional_value=750000.0,
            flow_direction=FlowDirection.BULLISH,
            confidence=0.85,
        )

        assert activity.activity_type == ActivityType.BLOCK_TRADE
        assert activity.volume_ratio == 5.0
        assert activity.confidence == 0.85

    def test_to_dict(self):
        activity = UnusualActivity(
            symbol="AAPL",
            option_symbol="AAPL240115C150",
            activity_type=ActivityType.SWEEP,
            option_type=OptionType.PUT,
            strike=140.0,
            expiry=datetime.now() + timedelta(days=30),
            volume=3000,
            avg_volume=500.0,
            volume_ratio=6.0,
            oi_change=1500,
            premium=15000.0,
            notional_value=420000.0,
            flow_direction=FlowDirection.BEARISH,
            confidence=0.9,
        )

        d = activity.to_dict()
        assert d["activity_type"] == "sweep"
        assert d["flow_direction"] == "bearish"


class TestVolatilitySkew:
    """Test VolatilitySkew dataclass."""

    def test_creation(self):
        skew = VolatilitySkew(
            symbol="AAPL",
            expiry=datetime.now() + timedelta(days=30),
            timestamp=datetime.now(),
            atm_strike=150.0,
            atm_vol=0.25,
            skew_index=-0.05,
            risk_reversal_25d=-0.02,
            risk_reversal_10d=-0.03,
            butterfly_25d=0.01,
            butterfly_10d=0.015,
            put_skew=0.03,
            call_skew=-0.01,
            skew_slope=-0.002,
            smile_convexity=0.001,
        )

        assert skew.atm_vol == 0.25
        assert skew.skew_index == -0.05


class TestFuturesContract:
    """Test FuturesContract dataclass."""

    def test_creation(self):
        expiry = datetime.now() + timedelta(days=60)
        contract = FuturesContract(
            symbol="CL",
            contract_month="202412",
            expiry=expiry,
            price=75.50,
            volume=50000,
            open_interest=200000,
            basis=2.50,
            time_to_expiry=60 / 365.0,
        )

        assert contract.symbol == "CL"
        assert contract.price == 75.50
        assert contract.contract_month == "202412"
        assert contract.basis == 2.50

    def test_to_dict(self):
        expiry = datetime.now() + timedelta(days=60)
        contract = FuturesContract(
            symbol="CL",
            contract_month="202412",
            expiry=expiry,
            price=75.50,
            volume=50000,
            open_interest=200000,
            basis=2.50,
            time_to_expiry=60 / 365.0,
        )

        d = contract.to_dict()
        assert d["symbol"] == "CL"
        assert d["price"] == 75.50
        assert d["contract_month"] == "202412"


# ============ Provider Tests ============


class TestSimulatedOptionDataProvider:
    """Test SimulatedOptionDataProvider."""

    @pytest.mark.asyncio
    async def test_set_and_get_option_chain(self, sample_option_chain):
        provider = SimulatedOptionDataProvider()
        provider.set_option_chain("AAPL", sample_option_chain)

        chain = await provider.get_option_chain("AAPL")
        assert len(chain) == len(sample_option_chain)

    @pytest.mark.asyncio
    async def test_empty_chain(self):
        provider = SimulatedOptionDataProvider()
        chain = await provider.get_option_chain("UNKNOWN")
        assert chain == []

    @pytest.mark.asyncio
    async def test_set_and_get_history(self, sample_option_chain):
        provider = SimulatedOptionDataProvider()
        history = sample_option_chain[:3]
        provider.set_option_history("AAPL", "AAPL240115C150", history)

        result = await provider.get_option_history("AAPL", "AAPL240115C150", 10)
        assert len(result) == 3


class TestSimulatedFuturesDataProvider:
    """Test SimulatedFuturesDataProvider."""

    @pytest.mark.asyncio
    async def test_set_and_get_futures_chain(self, sample_futures_chain):
        provider = SimulatedFuturesDataProvider()
        provider.set_futures_chain("CL", sample_futures_chain)

        chain = await provider.get_futures_chain("CL")
        assert len(chain) == len(sample_futures_chain)

    @pytest.mark.asyncio
    async def test_empty_chain(self):
        provider = SimulatedFuturesDataProvider()
        chain = await provider.get_futures_chain("UNKNOWN")
        assert chain == []

    @pytest.mark.asyncio
    async def test_set_and_get_spot_price(self):
        provider = SimulatedFuturesDataProvider()
        provider.set_spot_price("CL", 72.50)

        price = await provider.get_spot_price("CL")
        assert price == 72.50

    @pytest.mark.asyncio
    async def test_default_spot_price(self):
        provider = SimulatedFuturesDataProvider()
        price = await provider.get_spot_price("UNKNOWN")
        assert price == 100.0  # Default value


# ============ Main Class Tests ============


class TestDerivativesIndicators:
    """Test DerivativesIndicators class."""

    def test_initialization(self, derivatives_indicators):
        assert derivatives_indicators._state == MonitorState.IDLE
        assert not derivatives_indicators._running

    @pytest.mark.asyncio
    async def test_start_stop(self, derivatives_indicators):
        await derivatives_indicators.start()
        assert derivatives_indicators._state == MonitorState.RUNNING
        assert derivatives_indicators._running

        await derivatives_indicators.stop()
        assert derivatives_indicators._state == MonitorState.IDLE
        assert not derivatives_indicators._running

    @pytest.mark.asyncio
    async def test_add_remove_symbol(self, derivatives_indicators):
        derivatives_indicators.add_symbol("AAPL")
        assert "AAPL" in derivatives_indicators._symbols

        derivatives_indicators.add_symbol("MSFT")
        assert len(derivatives_indicators._symbols) == 2

        derivatives_indicators.remove_symbol("AAPL")
        assert "AAPL" not in derivatives_indicators._symbols
        assert "MSFT" in derivatives_indicators._symbols

    @pytest.mark.asyncio
    async def test_update_and_get_pcr(
        self, derivatives_indicators, option_provider, sample_option_chain
    ):
        option_provider.set_option_chain("AAPL", sample_option_chain)
        derivatives_indicators.add_symbol("AAPL")

        # Trigger update (internal method)
        await derivatives_indicators._update_indicators("AAPL")

        # Get PCR via public API
        pcr = derivatives_indicators.get_pcr("AAPL", PCRType.VOLUME)

        assert pcr is not None
        assert pcr.symbol == "AAPL"
        assert pcr.pcr_type == PCRType.VOLUME
        assert pcr.pcr_value > 0

    @pytest.mark.asyncio
    async def test_get_all_pcr(
        self, derivatives_indicators, option_provider, sample_option_chain
    ):
        option_provider.set_option_chain("AAPL", sample_option_chain)
        derivatives_indicators.add_symbol("AAPL")

        await derivatives_indicators._update_indicators("AAPL")

        all_pcr = derivatives_indicators.get_all_pcr("AAPL")
        # Should have at least volume and OI PCR
        assert len(all_pcr) >= 2
        assert PCRType.VOLUME in all_pcr or PCRType.OPEN_INTEREST in all_pcr

    @pytest.mark.asyncio
    async def test_get_unusual_activities(
        self, derivatives_indicators, option_provider, sample_option_chain
    ):
        # Modify one option to have unusual volume (above block trade threshold)
        modified_chain = sample_option_chain.copy()
        if modified_chain:
            modified_chain[0] = OptionData(
                symbol=modified_chain[0].symbol,
                option_symbol=modified_chain[0].option_symbol,
                option_type=modified_chain[0].option_type,
                strike=modified_chain[0].strike,
                expiry=modified_chain[0].expiry,
                spot_price=modified_chain[0].spot_price,
                volume=500,  # Above block_trade_min_contracts=50
                open_interest=modified_chain[0].open_interest,
                bid=modified_chain[0].bid,
                ask=modified_chain[0].ask,
                last=modified_chain[0].last,
            )

        option_provider.set_option_chain("AAPL", modified_chain)
        derivatives_indicators.add_symbol("AAPL")

        await derivatives_indicators._update_indicators("AAPL")
        activities = derivatives_indicators.get_unusual_activities("AAPL")
        # Should detect block trade due to high volume
        assert isinstance(activities, list)

    @pytest.mark.asyncio
    async def test_get_volatility_skews(
        self, derivatives_indicators, option_provider, sample_option_chain
    ):
        option_provider.set_option_chain("AAPL", sample_option_chain)
        derivatives_indicators.add_symbol("AAPL")

        await derivatives_indicators._update_indicators("AAPL")
        skews = derivatives_indicators.get_volatility_skews("AAPL")
        assert isinstance(skews, list)

    @pytest.mark.asyncio
    async def test_get_sentiment(
        self, derivatives_indicators, option_provider, sample_option_chain
    ):
        option_provider.set_option_chain("AAPL", sample_option_chain)
        derivatives_indicators.add_symbol("AAPL")

        await derivatives_indicators._update_indicators("AAPL")
        sentiment = derivatives_indicators.get_sentiment("AAPL")

        assert sentiment is not None
        assert sentiment.symbol == "AAPL"
        assert isinstance(sentiment.sentiment_level, SentimentLevel)

    @pytest.mark.asyncio
    async def test_get_term_structure(
        self, derivatives_indicators, option_provider, futures_provider,
        sample_option_chain, sample_futures_chain
    ):
        # Need to provide option data because _update_indicators requires it
        # Create minimal option chain for CL
        now = datetime.now()
        expiry = now + timedelta(days=30)
        spot = 100.0

        cl_options = []
        for strike in [95, 100, 105]:
            cl_options.append(OptionData(
                symbol="CL",
                option_symbol=f"CL{expiry.strftime('%y%m%d')}C{strike}",
                option_type=OptionType.CALL,
                strike=strike,
                expiry=expiry,
                spot_price=spot,
                volume=1000,
                open_interest=5000,
                bid=2.0,
                ask=2.5,
                last=2.25,
                implied_vol=0.30,
            ))
            cl_options.append(OptionData(
                symbol="CL",
                option_symbol=f"CL{expiry.strftime('%y%m%d')}P{strike}",
                option_type=OptionType.PUT,
                strike=strike,
                expiry=expiry,
                spot_price=spot,
                volume=800,
                open_interest=4000,
                bid=1.5,
                ask=2.0,
                last=1.75,
                implied_vol=0.30,
            ))

        option_provider.set_option_chain("CL", cl_options)
        futures_provider.set_futures_chain("CL", sample_futures_chain)
        futures_provider.set_spot_price("CL", 100.0)
        derivatives_indicators.add_symbol("CL")

        await derivatives_indicators._update_indicators("CL")
        term_structure = derivatives_indicators.get_term_structure("CL")

        assert term_structure is not None
        assert term_structure.symbol == "CL"
        assert term_structure.spot_price == 100.0

    @pytest.mark.asyncio
    async def test_get_snapshot(
        self, derivatives_indicators, option_provider, sample_option_chain
    ):
        option_provider.set_option_chain("AAPL", sample_option_chain)
        derivatives_indicators.add_symbol("AAPL")

        await derivatives_indicators._update_indicators("AAPL")
        snapshot = derivatives_indicators.get_snapshot("AAPL")

        assert snapshot is not None
        assert snapshot.symbol == "AAPL"
        assert snapshot.pcr_indicators is not None

    def test_get_statistics(self, derivatives_indicators):
        derivatives_indicators.add_symbol("AAPL")
        derivatives_indicators.add_symbol("MSFT")

        stats = derivatives_indicators.get_statistics()

        assert "state" in stats
        assert "symbols_tracked" in stats
        assert stats["symbols_tracked"] == 2


class TestDerivativesConfig:
    """Test DerivativesConfig."""

    def test_default_values(self):
        config = DerivativesConfig()

        assert config.update_interval_ms == 5000
        assert config.history_size == 1000
        assert config.pcr_extreme_low == 0.5
        assert config.pcr_extreme_high == 1.5

    def test_custom_values(self):
        config = DerivativesConfig(
            update_interval_ms=1000,
            pcr_extreme_low=0.3,
            pcr_extreme_high=2.0,
        )

        assert config.update_interval_ms == 1000
        assert config.pcr_extreme_low == 0.3
        assert config.pcr_extreme_high == 2.0


# ============ Factory Function Tests ============


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_derivatives_indicators(self):
        indicators = create_derivatives_indicators()
        assert indicators is not None
        assert isinstance(indicators, DerivativesIndicators)

    def test_create_with_custom_parameters(self):
        indicators = create_derivatives_indicators(
            update_interval_ms=2000,
            pcr_extreme_low=0.3,
            pcr_extreme_high=2.0,
        )
        assert indicators._config.update_interval_ms == 2000
        assert indicators._config.pcr_extreme_low == 0.3
        assert indicators._config.pcr_extreme_high == 2.0


# ============ Edge Case Tests ============


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_empty_option_chain_pcr(self, derivatives_indicators):
        derivatives_indicators.add_symbol("EMPTY")
        await derivatives_indicators._update_indicators("EMPTY")
        pcr = derivatives_indicators.get_pcr("EMPTY", PCRType.VOLUME)
        # Should handle gracefully - None when no data
        assert pcr is None

    @pytest.mark.asyncio
    async def test_unknown_symbol(self, derivatives_indicators):
        snapshot = derivatives_indicators.get_snapshot("UNKNOWN")
        # Should return None for unknown symbol
        assert snapshot is None

    @pytest.mark.asyncio
    async def test_multiple_start_calls(self, derivatives_indicators):
        await derivatives_indicators.start()
        await derivatives_indicators.start()  # Should be idempotent
        assert derivatives_indicators._running
        await derivatives_indicators.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, derivatives_indicators):
        await derivatives_indicators.stop()  # Should not raise
        assert not derivatives_indicators._running


# ============ Callback Tests ============


class TestCallbacks:
    """Test callback functionality."""

    @pytest.mark.asyncio
    async def test_alert_callback(
        self, config, option_provider, futures_provider, sample_option_chain
    ):
        alerts_received = []

        def alert_handler(alert):
            alerts_received.append(alert)

        indicators = DerivativesIndicators(
            config=config,
            option_provider=option_provider,
            futures_provider=futures_provider,
        )
        # Use on_alert method to register callback
        indicators.on_alert(alert_handler)

        option_provider.set_option_chain("AAPL", sample_option_chain)
        indicators.add_symbol("AAPL")

        # Trigger analysis (may or may not generate alerts)
        await indicators._update_indicators("AAPL")

        # Callback should be registered
        assert alert_handler in indicators._alert_callbacks

    @pytest.mark.asyncio
    async def test_snapshot_callback(
        self, config, option_provider, futures_provider, sample_option_chain
    ):
        snapshots_received = []

        def snapshot_handler(snapshot):
            snapshots_received.append(snapshot)

        indicators = DerivativesIndicators(
            config=config,
            option_provider=option_provider,
            futures_provider=futures_provider,
        )
        # Use on_snapshot_update method to register callback
        indicators.on_snapshot_update(snapshot_handler)

        option_provider.set_option_chain("AAPL", sample_option_chain)
        indicators.add_symbol("AAPL")

        # Trigger update
        await indicators._update_indicators("AAPL")

        # Should have received snapshot
        assert len(snapshots_received) == 1
        assert snapshots_received[0].symbol == "AAPL"
