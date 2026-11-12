"""
Tests for Resilient WebSocket Implementation

This module tests:
- Connection and disconnection
- Reconnection strategies
- Message sending and receiving
- Heartbeat mechanism
- Message queue management
- Subscription recovery
- Connection pool
- Metrics reporting
"""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Dict, Any

from quantrisk_realtime.realtime.websocket_reconnect import (
    ResilientWebSocket,
    WebSocketConnectionPool,
    ConnectionState,
    ExponentialBackoff,
    LinearBackoff,
    FixedInterval,
    JitteredBackoff,
    ConnectionMetrics,
    Message
)


# Mock WebSocket for testing
class MockWebSocket:
    """Mock WebSocket for testing"""

    def __init__(self, fail_on_connect=False, fail_on_send=False, fail_after_n_messages=None):
        self.fail_on_connect = fail_on_connect
        self.fail_on_send = fail_on_send
        self.fail_after_n_messages = fail_after_n_messages
        self.sent_messages = []
        self.received_messages = []
        self.message_count = 0
        self.closed = False
        self.ping_count = 0

    async def send(self, message):
        """Mock send"""
        if self.fail_on_send:
            raise Exception("Send failed")

        if self.fail_after_n_messages and self.message_count >= self.fail_after_n_messages:
            from websockets.exceptions import ConnectionClosed
            raise ConnectionClosed(None, None)

        self.sent_messages.append(message)
        self.message_count += 1

    async def recv(self):
        """Mock receive"""
        if self.fail_after_n_messages and self.message_count >= self.fail_after_n_messages:
            from websockets.exceptions import ConnectionClosed
            raise ConnectionClosed(None, None)

        if not self.received_messages:
            await asyncio.sleep(0.1)
            return json.dumps({"type": "heartbeat"})

        return self.received_messages.pop(0)

    async def close(self):
        """Mock close"""
        self.closed = True

    async def ping(self):
        """Mock ping"""
        self.ping_count += 1

    def add_message(self, message):
        """Add message to receive queue"""
        self.received_messages.append(json.dumps(message))


@pytest.fixture(autouse=True)
async def cleanup_websocket_clients(monkeypatch):
    """Close every test client so background work cannot leak between tests."""

    clients = []
    original_init = ResilientWebSocket.__init__

    def tracked_init(client, *args, **kwargs):
        original_init(client, *args, **kwargs)
        clients.append(client)

    monkeypatch.setattr(ResilientWebSocket, "__init__", tracked_init)
    yield

    for client in clients:
        await client.disconnect()
        client._json_executor.shutdown(wait=True, cancel_futures=True)


class TestBackoffStrategies:
    """Test backoff strategy implementations"""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation"""
        strategy = ExponentialBackoff(base_delay=1.0, max_delay=60.0, multiplier=2.0)

        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(2) == 4.0
        assert strategy.get_delay(3) == 8.0
        assert strategy.get_delay(10) == 60.0  # capped at max_delay

    def test_linear_backoff(self):
        """Test linear backoff calculation"""
        strategy = LinearBackoff(base_delay=1.0, max_delay=30.0, increment=2.0)

        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(1) == 3.0
        assert strategy.get_delay(2) == 5.0
        assert strategy.get_delay(3) == 7.0
        assert strategy.get_delay(20) == 30.0  # capped at max_delay

    def test_fixed_interval(self):
        """Test fixed interval backoff"""
        strategy = FixedInterval(delay=5.0)

        assert strategy.get_delay(0) == 5.0
        assert strategy.get_delay(1) == 5.0
        assert strategy.get_delay(10) == 5.0

    def test_jittered_backoff(self):
        """Test jittered backoff adds randomness"""
        base_strategy = FixedInterval(delay=10.0)
        strategy = JitteredBackoff(base_strategy, jitter_factor=0.3)

        delays = [strategy.get_delay(0) for _ in range(100)]

        # All delays should be between 10.0 and 13.0
        assert all(10.0 <= d <= 13.0 for d in delays)
        # Delays should vary (not all the same)
        assert len(set(delays)) > 1

    def test_backoff_reset(self):
        """Test backoff strategy reset"""
        strategy = ExponentialBackoff()

        # Get some delays
        strategy.get_delay(5)

        # Reset should allow getting delays again from start
        strategy.reset()
        assert strategy.get_delay(0) == 1.0


class TestConnectionMetrics:
    """Test connection metrics tracking"""

    def test_metrics_initialization(self):
        """Test metrics are initialized correctly"""
        metrics = ConnectionMetrics()

        assert metrics.connection_attempts == 0
        assert metrics.successful_connections == 0
        assert metrics.total_messages_sent == 0
        assert metrics.average_latency_ms == 0.0

    def test_latency_recording(self):
        """Test latency recording and averaging"""
        metrics = ConnectionMetrics()

        metrics.record_latency(10.0)
        assert metrics.average_latency_ms == 10.0

        metrics.record_latency(20.0)
        assert metrics.average_latency_ms == 15.0

        metrics.record_latency(30.0)
        assert metrics.average_latency_ms == 20.0

    def test_latency_sample_limit(self):
        """Test latency samples are limited to 100"""
        metrics = ConnectionMetrics()

        # Record 150 samples
        for i in range(150):
            metrics.record_latency(float(i))

        # Should only keep last 100
        assert len(metrics.latency_samples) == 100

    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary"""
        metrics = ConnectionMetrics()
        metrics.connection_attempts = 5
        metrics.successful_connections = 4
        metrics.total_messages_sent = 100

        result = metrics.to_dict()

        assert result["connection_attempts"] == 5
        assert result["successful_connections"] == 4
        assert result["total_messages_sent"] == 100
        assert "average_latency_ms" in result

    def test_uptime_calculation(self):
        """Test connection uptime calculation"""
        metrics = ConnectionMetrics()

        # No connection yet
        assert metrics._calculate_uptime() is None

        # Connected
        metrics.last_connect_time = time.time()
        uptime = metrics._calculate_uptime()
        assert uptime is not None
        assert uptime >= 0

        # Disconnected
        metrics.last_disconnect_time = metrics.last_connect_time + 10.0
        uptime = metrics._calculate_uptime()
        assert uptime == 10.0


class TestMessage:
    """Test message wrapper"""

    def test_message_creation(self):
        """Test message creation"""
        msg = Message(
            id="test_123",
            data={"key": "value"},
            timestamp=time.time()
        )

        assert msg.id == "test_123"
        assert msg.data == {"key": "value"}
        assert not msg.acknowledged
        assert msg.retry_count == 0
        assert not msg.compressed

    def test_message_acknowledgment(self):
        """Test message acknowledgment"""
        msg = Message(id="test", data={}, timestamp=time.time())

        assert not msg.acknowledged
        msg.acknowledged = True
        assert msg.acknowledged


@pytest.mark.asyncio
class TestResilientWebSocket:
    """Test ResilientWebSocket implementation"""

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket"""
        return MockWebSocket()

    @pytest.fixture
    async def websocket_client(self):
        """Create WebSocket client for testing"""
        client = ResilientWebSocket(
            url="ws://localhost:8000",
            backoff_strategy=FixedInterval(delay=0.1),
            heartbeat_interval=0.5,
            heartbeat_timeout=2.0
        )
        yield client
        await client.disconnect()

    async def test_initialization(self):
        """Test WebSocket initialization"""
        ws = ResilientWebSocket(url="ws://test.com")

        assert ws.url == "ws://test.com"
        assert ws.state == ConnectionState.DISCONNECTED
        assert ws.websocket is None
        assert ws.reconnect_attempts == 0

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_successful_connection(self, mock_connect, mock_websocket):
        """Test successful connection"""
        # websockets.connect() is an async function, must return awaitable
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        ws = ResilientWebSocket(
            url="ws://test.com",
            heartbeat_interval=10.0
        )

        success = await ws.connect()

        assert success
        assert ws.state == ConnectionState.CONNECTED
        assert ws.metrics.successful_connections == 1
        assert ws.is_healthy

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_failed_connection(self, mock_connect):
        """Test failed connection"""
        mock_connect.side_effect = Exception("Connection failed")

        ws = ResilientWebSocket(url="ws://test.com")

        success = await ws.connect()

        assert not success
        assert ws.state == ConnectionState.DISCONNECTED
        assert ws.metrics.connection_attempts == 1
        assert ws.metrics.successful_connections == 0

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_disconnect(self, mock_connect, mock_websocket):
        """Test disconnection"""
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        ws = ResilientWebSocket(url="ws://test.com", heartbeat_interval=10.0)
        await ws.connect()

        await ws.disconnect()

        assert ws.state == ConnectionState.CLOSED
        assert mock_websocket.closed

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_send_message(self, mock_connect, mock_websocket):
        """Test sending messages"""
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        ws = ResilientWebSocket(url="ws://test.com", heartbeat_interval=10.0)
        await ws.connect()

        # Wait for send loop to start
        await asyncio.sleep(0.1)

        success = await ws.send({"test": "data"})
        assert success

        # Wait for message to be sent
        await asyncio.sleep(0.2)

        assert len(mock_websocket.sent_messages) > 0

    async def test_send_without_connection(self):
        """Test sending message without connection"""
        ws = ResilientWebSocket(url="ws://test.com")

        success = await ws.send({"test": "data"})

        assert not success

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_message_queue_full(self, mock_connect, mock_websocket):
        """Test message queue overflow"""
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        ws = ResilientWebSocket(
            url="ws://test.com",
            message_queue_size=5,
            heartbeat_interval=10.0
        )
        await ws.connect()

        # Fill queue
        for i in range(10):
            success = await ws.send({"msg": i})
            if i < 5:
                assert success
            else:
                # Queue should be full
                await asyncio.sleep(0.1)

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_subscribe_unsubscribe(self, mock_connect, mock_websocket):
        """Test channel subscription"""
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        ws = ResilientWebSocket(url="ws://test.com", heartbeat_interval=10.0)
        await ws.connect()
        await asyncio.sleep(0.1)

        await ws.subscribe("test_channel")
        assert "test_channel" in ws.subscriptions

        await ws.unsubscribe("test_channel")
        assert "test_channel" not in ws.subscriptions

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_subscription_recovery(self, mock_connect, mock_websocket):
        """Test subscription recovery after reconnection"""
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        ws = ResilientWebSocket(url="ws://test.com", heartbeat_interval=10.0)
        await ws.connect()

        # Subscribe to channels
        await ws.subscribe("channel1")
        await ws.subscribe("channel2")

        # Simulate reconnection
        await ws.disconnect()
        mock_websocket.closed = False  # Reset for reconnection
        await ws.connect()
        await asyncio.sleep(0.2)

        # Check subscriptions were restored
        assert "channel1" in ws.subscriptions
        assert "channel2" in ws.subscriptions

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_message_acknowledgment(self, mock_connect, mock_websocket):
        """Test message acknowledgment"""
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        ws = ResilientWebSocket(url="ws://test.com", heartbeat_interval=10.0)
        await ws.connect()
        await asyncio.sleep(0.1)

        # Send message requiring ack
        await ws.send({"test": "data"}, message_id="msg_123", require_ack=True)

        assert "msg_123" in ws.pending_messages

        # Acknowledge message
        ws.acknowledge_message("msg_123")

        assert "msg_123" not in ws.pending_messages

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_message_deduplication(self, mock_connect, mock_websocket):
        """Test message deduplication"""
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        received_messages = []

        async def on_message(msg):
            received_messages.append(msg)

        ws = ResilientWebSocket(url="ws://test.com", heartbeat_interval=10.0)
        ws.set_callbacks(on_message=on_message)

        await ws.connect()
        await asyncio.sleep(0.1)

        # Send duplicate messages
        test_msg = {"id": "unique_123", "data": "test"}
        mock_websocket.add_message(test_msg)
        mock_websocket.add_message(test_msg)  # Duplicate

        await asyncio.sleep(0.3)

        # Heartbeat count depends on event-loop scheduling; the business
        # message itself must still be delivered exactly once.
        business_messages = [
            message for message in received_messages
            if message.get("id") == "unique_123"
        ]
        assert len(business_messages) == 1

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_callbacks(self, mock_connect, mock_websocket):
        """Test event callbacks"""
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        callbacks_called = {
            "connect": False,
            "disconnect": False,
            "message": False
        }

        async def on_connect():
            callbacks_called["connect"] = True

        async def on_disconnect():
            callbacks_called["disconnect"] = True

        async def on_message(msg):
            callbacks_called["message"] = True

        ws = ResilientWebSocket(url="ws://test.com", heartbeat_interval=10.0)
        ws.set_callbacks(
            on_connect=on_connect,
            on_disconnect=on_disconnect,
            on_message=on_message
        )

        await ws.connect()
        await asyncio.sleep(0.1)

        assert callbacks_called["connect"]

        # Add message to trigger callback
        mock_websocket.add_message({"test": "data"})
        await asyncio.sleep(0.2)

        await ws.disconnect()
        await asyncio.sleep(0.1)

        assert callbacks_called["disconnect"]

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_reconnection_on_connection_lost(self, mock_connect):
        """Test automatic reconnection when connection is lost"""
        # First connection succeeds
        mock_ws1 = MockWebSocket(fail_after_n_messages=2)
        # Second connection succeeds
        mock_ws2 = MockWebSocket()

        call_count = [0]
        async def mock_connect_async(*args, **kwargs):
            ws = mock_ws1 if call_count[0] == 0 else mock_ws2
            call_count[0] += 1
            return ws
        mock_connect.side_effect = mock_connect_async

        reconnect_called = False

        async def on_reconnect():
            nonlocal reconnect_called
            reconnect_called = True

        ws = ResilientWebSocket(
            url="ws://test.com",
            backoff_strategy=FixedInterval(delay=0.1),
            heartbeat_interval=10.0
        )
        ws.set_callbacks(on_reconnect=on_reconnect)

        await ws.connect()
        await asyncio.sleep(0.1)

        # Send messages to trigger failure
        await ws.send({"msg": 1})
        await ws.send({"msg": 2})
        await ws.send({"msg": 3})

        # Wait for reconnection
        await asyncio.sleep(0.5)

        assert ws.state == ConnectionState.CONNECTED
        assert ws.metrics.successful_reconnections > 0

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_max_reconnect_attempts(self, mock_connect):
        """Test max reconnection attempts limit"""
        mock_connect.side_effect = Exception("Connection failed")

        ws = ResilientWebSocket(
            url="ws://test.com",
            backoff_strategy=FixedInterval(delay=0.1),
            max_reconnect_attempts=3
        )

        await ws.connect()  # Initial connection fails

        # Manually trigger reconnection
        await ws._handle_connection_lost()

        # Wait for reconnection attempts
        await asyncio.sleep(0.5)

        # Should have attempted reconnection
        assert ws.reconnect_attempts > 0

    async def test_get_metrics(self):
        """Test getting connection metrics"""
        ws = ResilientWebSocket(url="ws://test.com")

        metrics = ws.get_metrics()

        assert "connection_attempts" in metrics
        assert "successful_connections" in metrics
        assert "total_messages_sent" in metrics
        assert "average_latency_ms" in metrics

    async def test_health_status(self):
        """Test getting health status"""
        ws = ResilientWebSocket(url="ws://test.com")

        health = ws.get_health_status()

        assert "is_healthy" in health
        assert "state" in health
        assert "reconnect_attempts" in health
        assert "pending_messages" in health

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_message_compression(self, mock_connect, mock_websocket):
        """Test message compression"""
        async def mock_connect_async(*args, **kwargs):
            return mock_websocket
        mock_connect.side_effect = mock_connect_async

        ws = ResilientWebSocket(
            url="ws://test.com",
            enable_compression=True,
            compression_threshold=100,
            heartbeat_interval=10.0
        )
        await ws.connect()
        await asyncio.sleep(0.1)

        # Send large message
        large_data = {"data": "x" * 200}
        await ws.send(large_data)

        await asyncio.sleep(0.2)

        # Message should be sent (possibly compressed)
        assert ws.metrics.total_messages_sent > 0

    async def test_message_id_generation(self):
        """Test unique message ID generation"""
        ws = ResilientWebSocket(url="ws://test.com")

        id1 = ws._generate_message_id()
        id2 = ws._generate_message_id()

        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0


@pytest.mark.asyncio
class TestWebSocketConnectionPool:
    """Test WebSocket connection pool"""

    async def test_pool_initialization(self):
        """Test pool initialization"""
        pool = WebSocketConnectionPool(max_connections=5)

        assert pool.max_connections == 5
        assert len(pool.connections) == 0

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_get_connection(self, mock_connect):
        """Test getting connection from pool"""
        async def mock_connect_async(*args, **kwargs):
            return MockWebSocket()
        mock_connect.side_effect = mock_connect_async

        pool = WebSocketConnectionPool(max_connections=5)

        conn1 = await pool.get_connection("ws://test1.com", heartbeat_interval=10.0)

        assert conn1 is not None
        assert len(pool.connections) == 1

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_reuse_existing_connection(self, mock_connect):
        """Test reusing existing connection"""
        async def mock_connect_async(*args, **kwargs):
            return MockWebSocket()
        mock_connect.side_effect = mock_connect_async

        pool = WebSocketConnectionPool(max_connections=5)

        conn1 = await pool.get_connection("ws://test.com", heartbeat_interval=10.0)
        conn2 = await pool.get_connection("ws://test.com", heartbeat_interval=10.0)

        assert conn1 is conn2
        assert len(pool.connections) == 1

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_pool_max_connections(self, mock_connect):
        """Test pool max connections limit"""
        async def mock_connect_async(*args, **kwargs):
            return MockWebSocket()
        mock_connect.side_effect = mock_connect_async

        pool = WebSocketConnectionPool(max_connections=2)

        conn1 = await pool.get_connection("ws://test1.com", heartbeat_interval=10.0)
        conn2 = await pool.get_connection("ws://test2.com", heartbeat_interval=10.0)

        # Should raise error when exceeding max
        with pytest.raises(RuntimeError):
            await pool.get_connection("ws://test3.com", heartbeat_interval=10.0)

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_close_all_connections(self, mock_connect):
        """Test closing all connections"""
        async def mock_connect_async(*args, **kwargs):
            return MockWebSocket()
        mock_connect.side_effect = mock_connect_async

        pool = WebSocketConnectionPool(max_connections=5)

        await pool.get_connection("ws://test1.com", heartbeat_interval=10.0)
        await pool.get_connection("ws://test2.com", heartbeat_interval=10.0)

        assert len(pool.connections) == 2

        await pool.close_all()

        assert len(pool.connections) == 0

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_pool_metrics(self, mock_connect):
        """Test getting pool metrics"""
        async def mock_connect_async(*args, **kwargs):
            return MockWebSocket()
        mock_connect.side_effect = mock_connect_async

        pool = WebSocketConnectionPool(max_connections=5)

        await pool.get_connection("ws://test1.com", heartbeat_interval=10.0)
        await pool.get_connection("ws://test2.com", heartbeat_interval=10.0)

        metrics = pool.get_pool_metrics()

        assert len(metrics) == 2
        assert "ws://test1.com" in metrics
        assert "ws://test2.com" in metrics

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_pool_health(self, mock_connect):
        """Test getting pool health status"""
        async def mock_connect_async(*args, **kwargs):
            return MockWebSocket()
        mock_connect.side_effect = mock_connect_async

        pool = WebSocketConnectionPool(max_connections=5)

        await pool.get_connection("ws://test1.com", heartbeat_interval=10.0)

        health = pool.get_pool_health()

        assert health["total_connections"] == 1
        assert health["max_connections"] == 5
        assert "connections" in health


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for complete workflows"""

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_full_lifecycle(self, mock_connect):
        """Test complete connection lifecycle"""
        mock_ws = MockWebSocket()
        async def mock_connect_async(*args, **kwargs):
            return mock_ws
        mock_connect.side_effect = mock_connect_async

        # Track events
        events = []

        async def on_connect():
            events.append("connect")

        async def on_disconnect():
            events.append("disconnect")

        async def on_message(msg):
            events.append(f"message:{msg.get('type')}")

        # Create and configure client
        ws = ResilientWebSocket(
            url="ws://test.com",
            heartbeat_interval=10.0
        )
        ws.set_callbacks(
            on_connect=on_connect,
            on_disconnect=on_disconnect,
            on_message=on_message
        )

        # Connect
        await ws.connect()
        await asyncio.sleep(0.1)
        assert "connect" in events

        # Subscribe
        await ws.subscribe("test_channel")
        assert "test_channel" in ws.subscriptions

        # Send message
        await ws.send({"action": "test"})
        await asyncio.sleep(0.1)
        assert ws.metrics.total_messages_sent > 0

        # Receive message
        mock_ws.add_message({"type": "data", "value": 123})
        await asyncio.sleep(0.2)

        # Disconnect
        await ws.disconnect()
        await asyncio.sleep(0.1)
        assert "disconnect" in events

        # Verify metrics
        metrics = ws.get_metrics()
        assert metrics["successful_connections"] > 0

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_reconnection_with_subscriptions(self, mock_connect):
        """Test reconnection preserves subscriptions"""
        mock_ws1 = MockWebSocket(fail_after_n_messages=2)
        mock_ws2 = MockWebSocket()
        call_count = [0]
        async def mock_connect_async(*args, **kwargs):
            ws = mock_ws1 if call_count[0] == 0 else mock_ws2
            call_count[0] += 1
            return ws
        mock_connect.side_effect = mock_connect_async

        ws = ResilientWebSocket(
            url="ws://test.com",
            backoff_strategy=FixedInterval(delay=0.1),
            heartbeat_interval=10.0
        )

        # Connect and subscribe
        await ws.connect()
        await ws.subscribe("channel1")
        await ws.subscribe("channel2")

        # Trigger disconnection
        await ws.send({"msg": 1})
        await ws.send({"msg": 2})
        await ws.send({"msg": 3})

        # Wait for reconnection
        await asyncio.sleep(0.5)

        # Verify subscriptions are preserved
        assert "channel1" in ws.subscriptions
        assert "channel2" in ws.subscriptions

    @patch('quantrisk_realtime.realtime.websocket_reconnect.websockets.connect')
    async def test_multiple_connections_in_pool(self, mock_connect):
        """Test managing multiple connections in pool"""
        async def mock_connect_async(*args, **kwargs):
            return MockWebSocket()
        mock_connect.side_effect = mock_connect_async

        pool = WebSocketConnectionPool(max_connections=5)

        # Create multiple connections
        conn1 = await pool.get_connection("ws://test1.com", heartbeat_interval=10.0)
        conn2 = await pool.get_connection("ws://test2.com", heartbeat_interval=10.0)
        conn3 = await pool.get_connection("ws://test3.com", heartbeat_interval=10.0)

        # All should be connected
        assert conn1.is_connected()
        assert conn2.is_connected()
        assert conn3.is_connected()

        # Get health status
        health = pool.get_pool_health()
        assert health["total_connections"] == 3

        # Close all
        await pool.close_all()
        assert len(pool.connections) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
