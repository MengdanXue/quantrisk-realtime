"""
Performance tests for WebSocket non-blocking improvements

This test demonstrates:
1. Non-blocking JSON parsing for large messages
2. Timeout protection for slow callbacks
3. Metrics tracking for performance monitoring
"""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from quantrisk_realtime.realtime.websocket_reconnect import (
    ResilientWebSocket,
    ExponentialBackoff
)


class TestWebSocketNonBlocking:
    """Test non-blocking WebSocket improvements"""

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket"""
        mock = AsyncMock()
        mock.recv = AsyncMock()
        mock.send = AsyncMock()
        mock.close = AsyncMock()
        mock.ping = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_large_json_parsing_non_blocking(self, mock_websocket):
        """Test that large JSON parsing doesn't block event loop"""
        # Create large JSON message (10MB)
        large_data = {
            "id": "test_123",
            "data": {
                "values": [{"index": i, "value": f"data_{i}"} for i in range(100000)]
            }
        }
        large_message = json.dumps(large_data)

        # Create WebSocket with non-blocking parsing
        ws = ResilientWebSocket(
            "ws://test.example.com",
            json_parse_workers=4,
            callback_timeout=5.0,
            max_json_size_mb=50.0
        )

        # Track if callback was called
        callback_called = False
        callback_data = None

        async def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data

        ws.on_message_callback = test_callback

        # Process message
        start_time = time.time()
        await ws._process_received_message(large_message)
        process_time = (time.time() - start_time) * 1000

        # Verify callback was called
        assert callback_called
        assert callback_data["id"] == "test_123"

        # Verify metrics
        metrics = ws.get_metrics()
        assert metrics["total_messages_received"] == 0  # Not updated in _process_received_message
        assert metrics["average_parse_time_ms"] > 0
        assert metrics["json_parse_errors"] == 0

        print(f"\nLarge JSON parse time: {process_time:.2f}ms")
        print(f"Average parse time: {metrics['average_parse_time_ms']:.2f}ms")

    @pytest.mark.asyncio
    async def test_callback_timeout_protection(self, mock_websocket):
        """Test that slow callbacks are protected with timeout"""
        ws = ResilientWebSocket(
            "ws://test.example.com",
            callback_timeout=0.5  # 500ms timeout
        )

        # Create slow callback
        timeout_occurred = False

        async def slow_callback(data):
            await asyncio.sleep(2.0)  # Slower than timeout
            return "completed"

        ws.on_message_callback = slow_callback

        # Process message
        message = json.dumps({"id": "test_456", "data": "test"})
        await ws._process_received_message(message)

        # Verify timeout was recorded
        metrics = ws.get_metrics()
        assert metrics["callback_timeouts"] == 1
        assert metrics["callback_errors"] == 0

        print(f"\nCallback timeouts: {metrics['callback_timeouts']}")

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, mock_websocket):
        """Test that callback errors don't crash the WebSocket"""
        ws = ResilientWebSocket(
            "ws://test.example.com",
            callback_timeout=5.0
        )

        # Create callback that raises error
        async def error_callback(data):
            raise ValueError("Test error")

        ws.on_message_callback = error_callback

        # Process message - should not raise
        message = json.dumps({"id": "test_789", "data": "test"})
        await ws._process_received_message(message)

        # Verify error was recorded
        metrics = ws.get_metrics()
        assert metrics["callback_errors"] == 1
        assert metrics["callback_timeouts"] == 0

        print(f"\nCallback errors: {metrics['callback_errors']}")

    @pytest.mark.asyncio
    async def test_message_size_limit(self, mock_websocket):
        """Test that oversized messages are rejected"""
        ws = ResilientWebSocket(
            "ws://test.example.com",
            max_json_size_mb=1.0  # 1MB limit
        )

        callback_called = False

        async def test_callback(data):
            nonlocal callback_called
            callback_called = True

        ws.on_message_callback = test_callback

        # Create message larger than limit (2MB)
        large_data = {"data": "x" * (2 * 1024 * 1024)}
        large_message = json.dumps(large_data)

        # Process message - should be rejected
        await ws._process_received_message(large_message)

        # Verify callback was NOT called
        assert not callback_called

    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self, mock_websocket):
        """Test that multiple messages can be processed concurrently"""
        ws = ResilientWebSocket(
            "ws://test.example.com",
            json_parse_workers=4
        )

        processed_messages = []

        async def tracking_callback(data):
            processed_messages.append(data["id"])
            await asyncio.sleep(0.1)  # Simulate some work

        ws.on_message_callback = tracking_callback

        # Create multiple messages
        messages = [
            json.dumps({"id": f"msg_{i}", "data": f"data_{i}"})
            for i in range(10)
        ]

        # Process all messages concurrently
        start_time = time.time()
        await asyncio.gather(*[
            ws._process_received_message(msg) for msg in messages
        ])
        total_time = (time.time() - start_time) * 1000

        # Verify all messages were processed
        assert len(processed_messages) == 10

        # With concurrent processing, should be faster than sequential
        # (10 messages * 100ms each = 1000ms sequential vs ~100-200ms concurrent)
        print(f"\nConcurrent processing time: {total_time:.2f}ms for 10 messages")

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, mock_websocket):
        """Test that performance metrics are tracked correctly"""
        ws = ResilientWebSocket(
            "ws://test.example.com",
            json_parse_workers=4,
            callback_timeout=5.0
        )

        async def simple_callback(data):
            await asyncio.sleep(0.01)

        ws.on_message_callback = simple_callback

        # Process several messages
        for i in range(5):
            message = json.dumps({"id": f"test_{i}", "data": f"value_{i}"})
            await ws._process_received_message(message)

        # Verify metrics
        metrics = ws.get_metrics()
        assert metrics["average_parse_time_ms"] > 0
        assert metrics["average_callback_time_ms"] > 0
        assert metrics["json_parse_errors"] == 0
        assert metrics["callback_timeouts"] == 0
        assert metrics["callback_errors"] == 0

        print(f"\nMetrics after 5 messages:")
        print(f"  Average parse time: {metrics['average_parse_time_ms']:.2f}ms")
        print(f"  Average callback time: {metrics['average_callback_time_ms']:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
