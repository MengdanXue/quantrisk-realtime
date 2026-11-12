# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for PnL WebSocket and Streamer modules.

Tests cover:
1. PnL Streamer lifecycle (start, stop, pause, resume)
2. Subscription management (subscribe, unsubscribe, disconnect)
3. P&L calculation and broadcasting
4. WebSocket message handling
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from quantrisk_realtime.realtime.pnl_streamer import (
    # Enums
    SubscriptionType,
    UpdateType,
    StreamerState,
    # Data Classes
    PositionPnL,
    PortfolioPnL,
    PnLUpdate,
    StreamerConfig,
    # Main Class
    PnLStreamer,
    # Factory
    create_pnl_streamer,
)


# ============ Test Fixtures ============


@pytest.fixture
def config():
    """Create test configuration."""
    return StreamerConfig(
        update_interval_seconds=0.1,  # Fast updates for testing
        heartbeat_interval_seconds=1.0,
        min_pnl_change_threshold=0.01,
        min_price_change_threshold=0.001,
        max_subscriptions_per_client=10,
    )


@pytest.fixture
def mock_portfolio_service():
    """Create mock portfolio service."""
    service = MagicMock()
    service.get_portfolio_summary = AsyncMock(return_value={
        'unrealized_pnl': 1500.0,
        'realized_pnl_today': 500.0,
        'total_market_value': 100000.0,
        'total_cost_basis': 98500.0,
        'equity': 120000.0,
        'cash': 20000.0,
        'position_count': 5,
        'margin_ratio': 0.2,
        'positions': [
            {
                'position_id': 101,
                'symbol': 'AAPL',
                'current_price': 150.0,
                'average_cost': 145.0,
                'quantity': 100,
                'direction': 1,
                'unrealized_pnl': 500.0,
                'market_value': 15000.0,
            }
        ]
    })
    return service


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock(return_value='{"type": "ping"}')
    return ws


@pytest.fixture
def pnl_streamer(config):
    """Create PnL streamer instance."""
    return PnLStreamer(config=config)


@pytest.fixture
def pnl_streamer_with_service(config, mock_portfolio_service):
    """Create PnL streamer with mock service."""
    return PnLStreamer(
        portfolio_service=mock_portfolio_service,
        config=config,
    )


# ============ Enum Tests ============


class TestSubscriptionType:
    """Test SubscriptionType enum."""

    def test_values(self):
        assert SubscriptionType.PORTFOLIO.value == "portfolio"
        assert SubscriptionType.POSITION.value == "position"
        assert SubscriptionType.ACCOUNT.value == "account"

    def test_count(self):
        assert len(SubscriptionType) == 3


class TestUpdateType:
    """Test UpdateType enum."""

    def test_values(self):
        assert UpdateType.FULL.value == "full"
        assert UpdateType.INCREMENTAL.value == "incremental"
        assert UpdateType.HEARTBEAT.value == "heartbeat"


class TestStreamerState:
    """Test StreamerState enum."""

    def test_values(self):
        assert StreamerState.STOPPED.value == "stopped"
        assert StreamerState.STARTING.value == "starting"
        assert StreamerState.RUNNING.value == "running"
        assert StreamerState.PAUSED.value == "paused"
        assert StreamerState.ERROR.value == "error"


# ============ Data Class Tests ============


class TestPositionPnL:
    """Test PositionPnL dataclass."""

    def test_creation(self):
        pos = PositionPnL(
            position_id=101,
            symbol="AAPL",
            current_price=150.0,
            average_cost=145.0,
            quantity=100,
            direction=1,
            unrealized_pnl=500.0,
            market_value=15000.0,
        )

        assert pos.position_id == 101
        assert pos.symbol == "AAPL"
        assert pos.unrealized_pnl == 500.0

    def test_to_dict(self):
        pos = PositionPnL(
            position_id=101,
            symbol="AAPL",
            current_price=150.0,
            average_cost=145.0,
            quantity=100,
            direction=1,
            unrealized_pnl=500.0,
        )

        d = pos.to_dict()
        assert d["position_id"] == 101
        assert d["symbol"] == "AAPL"
        assert "updated_at" in d


class TestPortfolioPnL:
    """Test PortfolioPnL dataclass."""

    def test_creation(self):
        pnl = PortfolioPnL(
            portfolio_id=1,
            account_id="ACC001",
            total_unrealized_pnl=1500.0,
            total_realized_pnl_today=500.0,
            total_market_value=100000.0,
        )

        assert pnl.portfolio_id == 1
        assert pnl.total_unrealized_pnl == 1500.0

    def test_to_dict_without_positions(self):
        pnl = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=1500.0,
        )

        d = pnl.to_dict(include_positions=False)
        assert d["portfolio_id"] == 1
        assert "positions" not in d

    def test_to_dict_with_positions(self):
        pos = PositionPnL(position_id=101, symbol="AAPL")
        pnl = PortfolioPnL(
            portfolio_id=1,
            positions=[pos],
        )

        d = pnl.to_dict(include_positions=True)
        assert "positions" in d
        assert len(d["positions"]) == 1
        assert d["positions"][0]["symbol"] == "AAPL"


class TestPnLUpdate:
    """Test PnLUpdate dataclass."""

    def test_creation(self):
        update = PnLUpdate(
            update_type=UpdateType.INCREMENTAL,
            subscription_type=SubscriptionType.PORTFOLIO,
            sequence=123,
        )

        assert update.update_type == UpdateType.INCREMENTAL
        assert update.sequence == 123

    def test_to_json_portfolio(self):
        pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1000.0)
        update = PnLUpdate(
            update_type=UpdateType.FULL,
            subscription_type=SubscriptionType.PORTFOLIO,
            portfolio_pnl=pnl,
            sequence=1,
        )

        json_str = update.to_json()
        data = json.loads(json_str)

        assert data["type"] == "pnl_update"
        assert data["update_type"] == "full"
        assert data["subscription_type"] == "portfolio"
        assert data["portfolio"]["portfolio_id"] == 1

    def test_to_json_incremental(self):
        update = PnLUpdate(
            update_type=UpdateType.INCREMENTAL,
            subscription_type=SubscriptionType.POSITION,
            pnl_change=50.25,
            price_change=0.5,
            sequence=100,
        )

        json_str = update.to_json()
        data = json.loads(json_str)

        assert data["pnl_change"] == 50.25
        assert data["price_change"] == 0.5


class TestStreamerConfig:
    """Test StreamerConfig dataclass."""

    def test_default_values(self):
        config = StreamerConfig()

        assert config.update_interval_seconds == 1.0
        assert config.heartbeat_interval_seconds == 30.0
        assert config.max_subscriptions_per_client == 50

    def test_custom_values(self):
        config = StreamerConfig(
            update_interval_seconds=0.5,
            max_subscriptions_per_client=100,
        )

        assert config.update_interval_seconds == 0.5
        assert config.max_subscriptions_per_client == 100


# ============ PnL Streamer Tests ============


class TestPnLStreamerLifecycle:
    """Test PnL streamer lifecycle management."""

    def test_initial_state(self, pnl_streamer):
        assert pnl_streamer.state == StreamerState.STOPPED
        assert not pnl_streamer.is_running

    @pytest.mark.asyncio
    async def test_start(self, pnl_streamer):
        await pnl_streamer.start()

        assert pnl_streamer.state == StreamerState.RUNNING
        assert pnl_streamer.is_running

        await pnl_streamer.stop()

    @pytest.mark.asyncio
    async def test_stop(self, pnl_streamer):
        await pnl_streamer.start()
        await pnl_streamer.stop()

        assert pnl_streamer.state == StreamerState.STOPPED
        assert not pnl_streamer.is_running

    @pytest.mark.asyncio
    async def test_double_start(self, pnl_streamer):
        await pnl_streamer.start()
        await pnl_streamer.start()  # Should be idempotent

        assert pnl_streamer.state == StreamerState.RUNNING

        await pnl_streamer.stop()

    @pytest.mark.asyncio
    async def test_double_stop(self, pnl_streamer):
        await pnl_streamer.start()
        await pnl_streamer.stop()
        await pnl_streamer.stop()  # Should be idempotent

        assert pnl_streamer.state == StreamerState.STOPPED

    @pytest.mark.asyncio
    async def test_pause_resume(self, pnl_streamer):
        await pnl_streamer.start()

        await pnl_streamer.pause()
        assert pnl_streamer.state == StreamerState.PAUSED

        await pnl_streamer.resume()
        assert pnl_streamer.state == StreamerState.RUNNING

        await pnl_streamer.stop()


class TestPnLStreamerSubscriptions:
    """Test subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_portfolio(self, pnl_streamer, mock_websocket):
        result = await pnl_streamer.subscribe_portfolio(mock_websocket, 1)

        assert result is True
        assert mock_websocket in pnl_streamer._portfolio_subscribers[1]
        assert "portfolio:1" in pnl_streamer._client_subscriptions[mock_websocket]

    @pytest.mark.asyncio
    async def test_subscribe_position(self, pnl_streamer, mock_websocket):
        result = await pnl_streamer.subscribe_position(mock_websocket, 101)

        assert result is True
        assert mock_websocket in pnl_streamer._position_subscribers[101]

    @pytest.mark.asyncio
    async def test_subscribe_account(self, pnl_streamer, mock_websocket):
        result = await pnl_streamer.subscribe_account(mock_websocket, "ACC001")

        assert result is True
        assert mock_websocket in pnl_streamer._account_subscribers["ACC001"]

    @pytest.mark.asyncio
    async def test_subscription_limit(self, config, mock_websocket):
        config.max_subscriptions_per_client = 2
        streamer = PnLStreamer(config=config)

        # First two should succeed
        result1 = await streamer.subscribe_portfolio(mock_websocket, 1)
        result2 = await streamer.subscribe_portfolio(mock_websocket, 2)

        assert result1 is True
        assert result2 is True

        # Third should fail
        result3 = await streamer.subscribe_portfolio(mock_websocket, 3)
        assert result3 is False

    @pytest.mark.asyncio
    async def test_unsubscribe_specific(self, pnl_streamer, mock_websocket):
        await pnl_streamer.subscribe_portfolio(mock_websocket, 1)
        await pnl_streamer.subscribe_portfolio(mock_websocket, 2)

        await pnl_streamer.unsubscribe(mock_websocket, "portfolio:1")

        assert mock_websocket not in pnl_streamer._portfolio_subscribers[1]
        assert mock_websocket in pnl_streamer._portfolio_subscribers[2]

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, pnl_streamer, mock_websocket):
        await pnl_streamer.subscribe_portfolio(mock_websocket, 1)
        await pnl_streamer.subscribe_position(mock_websocket, 101)

        await pnl_streamer.unsubscribe(mock_websocket)

        assert len(pnl_streamer._client_subscriptions[mock_websocket]) == 0

    @pytest.mark.asyncio
    async def test_handle_disconnect(self, pnl_streamer, mock_websocket):
        await pnl_streamer.subscribe_portfolio(mock_websocket, 1)
        await pnl_streamer.subscribe_position(mock_websocket, 101)

        await pnl_streamer.handle_disconnect(mock_websocket)

        assert mock_websocket not in pnl_streamer._portfolio_subscribers[1]
        assert mock_websocket not in pnl_streamer._client_subscriptions

    def test_subscriber_count(self, pnl_streamer):
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        pnl_streamer._portfolio_subscribers[1].add(ws1)
        pnl_streamer._portfolio_subscribers[1].add(ws2)
        pnl_streamer._position_subscribers[101].add(ws1)

        # ws1 is counted once even though it's in two sets
        assert pnl_streamer.subscriber_count == 2


class TestPnLCalculation:
    """Test P&L calculation."""

    @pytest.mark.asyncio
    async def test_calculate_with_service(self, pnl_streamer_with_service):
        streamer = pnl_streamer_with_service

        pnl = await streamer._calculate_portfolio_pnl(1)

        assert pnl is not None
        assert pnl.portfolio_id == 1
        assert pnl.total_unrealized_pnl == 1500.0
        assert pnl.total_realized_pnl_today == 500.0
        assert len(pnl.positions) == 1
        assert pnl.positions[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_calculate_mock_pnl(self, pnl_streamer):
        pnl = await pnl_streamer._generate_mock_pnl(1)

        assert pnl is not None
        assert pnl.portfolio_id == 1
        # Mock data is random but should be set
        assert pnl.updated_at is not None

    @pytest.mark.asyncio
    async def test_should_send_update_no_cache(self, pnl_streamer):
        pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1000.0)

        # No cache - should send
        result = pnl_streamer._should_send_update(1, pnl)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_send_update_significant_change(self, pnl_streamer):
        # Set cache
        cached = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=1000.0,
            total_market_value=100000.0,
        )
        pnl_streamer._portfolio_pnl_cache[1] = cached

        # Significant change (2% pnl change)
        new_pnl = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=1020.0,
            total_market_value=100000.0,
        )

        result = pnl_streamer._should_send_update(1, new_pnl)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_send_update_no_change(self, pnl_streamer):
        # Set cache
        cached = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=1000.0,
            total_market_value=100000.0,
        )
        pnl_streamer._portfolio_pnl_cache[1] = cached

        # Minimal change (less than threshold)
        new_pnl = PortfolioPnL(
            portfolio_id=1,
            total_unrealized_pnl=1000.05,  # 0.005% change
            total_market_value=100000.0,
        )

        result = pnl_streamer._should_send_update(1, new_pnl)
        assert result is False


class TestPnLBroadcasting:
    """Test P&L broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_portfolio_update(self, pnl_streamer, mock_websocket):
        await pnl_streamer.subscribe_portfolio(mock_websocket, 1)

        pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1500.0)
        await pnl_streamer._broadcast_portfolio_update(1, pnl)

        # Should have sent message
        assert mock_websocket.send_text.called

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_subscribers(self, pnl_streamer):
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await pnl_streamer.subscribe_portfolio(ws1, 1)
        await pnl_streamer.subscribe_portfolio(ws2, 1)

        pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1500.0)
        await pnl_streamer._broadcast_portfolio_update(1, pnl)

        assert ws1.send_text.called
        assert ws2.send_text.called

    @pytest.mark.asyncio
    async def test_broadcast_handles_dead_connection(self, pnl_streamer):
        ws_live = AsyncMock()
        ws_dead = AsyncMock()
        ws_dead.send_text.side_effect = Exception("Connection closed")

        await pnl_streamer.subscribe_portfolio(ws_live, 1)
        await pnl_streamer.subscribe_portfolio(ws_dead, 1)

        pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1500.0)
        await pnl_streamer._broadcast_portfolio_update(1, pnl)

        # Dead connection should be cleaned up
        assert ws_dead not in pnl_streamer._portfolio_subscribers[1]
        # Live connection should still be there
        assert ws_live in pnl_streamer._portfolio_subscribers[1]

    @pytest.mark.asyncio
    async def test_send_initial_portfolio_data(self, pnl_streamer, mock_websocket):
        await pnl_streamer._send_initial_portfolio_data(mock_websocket, 1)

        # Should have sent initial data
        assert mock_websocket.send_text.called

    @pytest.mark.asyncio
    async def test_broadcast_heartbeat(self, pnl_streamer, mock_websocket):
        await pnl_streamer.subscribe_portfolio(mock_websocket, 1)

        await pnl_streamer._broadcast_heartbeat()

        # Should have sent heartbeat
        mock_websocket.send_text.assert_called()
        call_arg = mock_websocket.send_text.call_args[0][0]
        data = json.loads(call_arg)
        assert data["type"] == "pnl_heartbeat"


class TestPnLStreamerStatistics:
    """Test statistics gathering."""

    def test_get_statistics(self, pnl_streamer):
        stats = pnl_streamer.get_statistics()

        assert "state" in stats
        assert stats["state"] == "stopped"
        assert "subscriber_count" in stats
        assert "update_count" in stats
        assert "error_count" in stats
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_statistics_after_start(self, pnl_streamer):
        await pnl_streamer.start()

        stats = pnl_streamer.get_statistics()

        assert stats["state"] == "running"
        assert stats["uptime_seconds"] is not None
        assert stats["uptime_seconds"] >= 0

        await pnl_streamer.stop()


class TestPnLStreamerCallbacks:
    """Test callback functionality."""

    @pytest.mark.asyncio
    async def test_pnl_update_callback(self, config):
        updates_received = []

        async def on_update(update):
            updates_received.append(update)

        streamer = PnLStreamer(
            config=config,
            on_pnl_update=on_update,
        )

        ws = AsyncMock()
        await streamer.subscribe_portfolio(ws, 1)

        pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1500.0)
        await streamer._broadcast_portfolio_update(1, pnl)

        assert len(updates_received) == 1
        assert updates_received[0].portfolio_pnl.portfolio_id == 1

    @pytest.mark.asyncio
    async def test_error_callback(self, config):
        errors_received = []

        async def on_error(error):
            errors_received.append(error)

        streamer = PnLStreamer(
            config=config,
            on_error=on_error,
        )

        # The on_error callback is called in the update loop when errors occur
        # We can verify it's set correctly
        assert streamer.on_error == on_error


# ============ Factory Function Tests ============


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_pnl_streamer_default(self):
        streamer = create_pnl_streamer()

        assert streamer is not None
        assert isinstance(streamer, PnLStreamer)
        assert streamer.config.update_interval_seconds == 1.0

    def test_create_pnl_streamer_custom(self):
        service = MagicMock()
        streamer = create_pnl_streamer(
            portfolio_service=service,
            update_interval=0.5,
        )

        assert streamer.portfolio_service == service
        assert streamer.config.update_interval_seconds == 0.5


# ============ Integration Tests ============


class TestPnLStreamerIntegration:
    """Integration tests for the full streaming workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, pnl_streamer_with_service):
        streamer = pnl_streamer_with_service
        ws = AsyncMock()

        # Start streamer
        await streamer.start()
        assert streamer.is_running

        # Subscribe
        result = await streamer.subscribe_portfolio(ws, 1)
        assert result is True

        # Check initial data was sent
        assert ws.send_text.called

        # Simulate update loop iteration
        pnl = await streamer._calculate_portfolio_pnl(1)
        assert pnl is not None

        # Disconnect
        await streamer.handle_disconnect(ws)
        assert ws not in streamer._portfolio_subscribers[1]

        # Stop
        await streamer.stop()
        assert not streamer.is_running

    @pytest.mark.asyncio
    async def test_multiple_portfolio_subscriptions(self, pnl_streamer):
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await pnl_streamer.subscribe_portfolio(ws1, 1)
        await pnl_streamer.subscribe_portfolio(ws1, 2)
        await pnl_streamer.subscribe_portfolio(ws2, 1)

        # ws1 subscribed to 2 portfolios, ws2 to 1
        assert len(pnl_streamer._client_subscriptions[ws1]) == 2
        assert len(pnl_streamer._client_subscriptions[ws2]) == 1

        # Portfolio 1 has 2 subscribers
        assert len(pnl_streamer._portfolio_subscribers[1]) == 2

        # Total unique subscribers
        assert pnl_streamer.subscriber_count == 2


class TestSequenceNumbers:
    """Test sequence number management."""

    @pytest.mark.asyncio
    async def test_sequence_increments(self, pnl_streamer, mock_websocket):
        await pnl_streamer.subscribe_portfolio(mock_websocket, 1)

        initial_seq = pnl_streamer._sequence

        pnl = PortfolioPnL(portfolio_id=1, total_unrealized_pnl=1000.0)
        await pnl_streamer._broadcast_portfolio_update(1, pnl)

        assert pnl_streamer._sequence == initial_seq + 1

        await pnl_streamer._broadcast_portfolio_update(1, pnl)
        assert pnl_streamer._sequence == initial_seq + 2

    @pytest.mark.asyncio
    async def test_heartbeat_increments_sequence(self, pnl_streamer, mock_websocket):
        await pnl_streamer.subscribe_portfolio(mock_websocket, 1)

        initial_seq = pnl_streamer._sequence

        await pnl_streamer._broadcast_heartbeat()

        assert pnl_streamer._sequence == initial_seq + 1
