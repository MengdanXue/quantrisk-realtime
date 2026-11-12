"""
Resilient WebSocket Implementation with Auto-Reconnection

This module provides a robust WebSocket client with:
- Automatic reconnection with configurable backoff strategies
- Connection state management
- Heartbeat mechanism
- Message queue buffering
- Subscription state recovery
- Connection pooling
- Health checks
- Metrics reporting
"""

import asyncio
import json
import logging
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict
import random
from concurrent.futures import ThreadPoolExecutor

try:
    import websockets
    from websockets.asyncio.client import ClientConnection as WebSocketClientProtocol
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    try:
        # Fallback for older websockets versions
        import websockets
        from websockets.client import WebSocketClientProtocol
        from websockets.exceptions import ConnectionClosed, WebSocketException
    except ImportError:
        websockets = None
        WebSocketClientProtocol = None
        ConnectionClosed = Exception
        WebSocketException = Exception


logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies"""

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        pass

    @abstractmethod
    def reset(self):
        """Reset strategy state"""
        pass


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff strategy"""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier

    def get_delay(self, attempt: int) -> float:
        """Calculate exponential delay"""
        delay = self.base_delay * (self.multiplier ** attempt)
        return min(delay, self.max_delay)

    def reset(self):
        """Reset strategy (no state to reset)"""
        pass


class LinearBackoff(BackoffStrategy):
    """Linear backoff strategy"""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        increment: float = 1.0
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.increment = increment

    def get_delay(self, attempt: int) -> float:
        """Calculate linear delay"""
        delay = self.base_delay + (attempt * self.increment)
        return min(delay, self.max_delay)

    def reset(self):
        """Reset strategy (no state to reset)"""
        pass


class FixedInterval(BackoffStrategy):
    """Fixed interval backoff strategy"""

    def __init__(self, delay: float = 5.0):
        self.delay = delay

    def get_delay(self, attempt: int) -> float:
        """Return fixed delay"""
        return self.delay

    def reset(self):
        """Reset strategy (no state to reset)"""
        pass


class JitteredBackoff(BackoffStrategy):
    """Backoff with jitter to avoid thundering herd"""

    def __init__(
        self,
        base_strategy: BackoffStrategy,
        jitter_factor: float = 0.3
    ):
        self.base_strategy = base_strategy
        self.jitter_factor = jitter_factor

    def get_delay(self, attempt: int) -> float:
        """Add jitter to base strategy delay"""
        base_delay = self.base_strategy.get_delay(attempt)
        jitter = base_delay * self.jitter_factor * random.random()
        return base_delay + jitter

    def reset(self):
        """Reset base strategy"""
        self.base_strategy.reset()


@dataclass
class ConnectionMetrics:
    """Connection metrics for monitoring"""
    connection_attempts: int = 0
    successful_connections: int = 0
    reconnection_attempts: int = 0
    successful_reconnections: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    last_connect_time: Optional[float] = None
    last_disconnect_time: Optional[float] = None
    average_latency_ms: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    json_parse_errors: int = 0
    callback_timeouts: int = 0
    callback_errors: int = 0
    average_parse_time_ms: float = 0.0
    parse_time_samples: List[float] = field(default_factory=list)
    average_callback_time_ms: float = 0.0
    callback_time_samples: List[float] = field(default_factory=list)

    def record_latency(self, latency_ms: float):
        """Record latency sample"""
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 100:
            self.latency_samples.pop(0)
        self.average_latency_ms = sum(self.latency_samples) / len(self.latency_samples)

    def record_parse_time(self, parse_time_ms: float):
        """Record JSON parse time sample"""
        self.parse_time_samples.append(parse_time_ms)
        if len(self.parse_time_samples) > 100:
            self.parse_time_samples.pop(0)
        self.average_parse_time_ms = sum(self.parse_time_samples) / len(self.parse_time_samples)

    def record_callback_time(self, callback_time_ms: float):
        """Record callback execution time sample"""
        self.callback_time_samples.append(callback_time_ms)
        if len(self.callback_time_samples) > 100:
            self.callback_time_samples.pop(0)
        self.average_callback_time_ms = sum(self.callback_time_samples) / len(self.callback_time_samples)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "connection_attempts": self.connection_attempts,
            "successful_connections": self.successful_connections,
            "reconnection_attempts": self.reconnection_attempts,
            "successful_reconnections": self.successful_reconnections,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "last_connect_time": self.last_connect_time,
            "last_disconnect_time": self.last_disconnect_time,
            "average_latency_ms": self.average_latency_ms,
            "connection_uptime": self._calculate_uptime(),
            "json_parse_errors": self.json_parse_errors,
            "callback_timeouts": self.callback_timeouts,
            "callback_errors": self.callback_errors,
            "average_parse_time_ms": self.average_parse_time_ms,
            "average_callback_time_ms": self.average_callback_time_ms
        }

    def _calculate_uptime(self) -> Optional[float]:
        """Calculate connection uptime in seconds"""
        if self.last_connect_time and not self.last_disconnect_time:
            return time.time() - self.last_connect_time
        elif self.last_connect_time and self.last_disconnect_time:
            return self.last_disconnect_time - self.last_connect_time
        return None


@dataclass
class Message:
    """Message wrapper for queue management"""
    id: str
    data: Any
    timestamp: float
    acknowledged: bool = False
    retry_count: int = 0
    compressed: bool = False


class ResilientWebSocket:
    """
    Resilient WebSocket client with auto-reconnection

    Features:
    - Automatic reconnection with configurable backoff
    - Connection state management
    - Heartbeat mechanism
    - Message queue buffering
    - Subscription state recovery
    - Message deduplication
    - Compression support
    """

    def __init__(
        self,
        url: str,
        backoff_strategy: Optional[BackoffStrategy] = None,
        max_reconnect_attempts: int = -1,
        heartbeat_interval: float = 30.0,
        heartbeat_timeout: float = 10.0,
        message_queue_size: int = 1000,
        enable_compression: bool = False,
        compression_threshold: int = 1024,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
        extra_headers: Optional[Dict[str, str]] = None,
        json_parse_workers: int = 4,
        callback_timeout: float = 5.0,
        max_json_size_mb: float = 50.0
    ):
        """
        Initialize ResilientWebSocket

        Args:
            url: WebSocket URL
            backoff_strategy: Reconnection backoff strategy
            max_reconnect_attempts: Max reconnection attempts (-1 for unlimited)
            heartbeat_interval: Heartbeat interval in seconds
            heartbeat_timeout: Heartbeat timeout in seconds
            message_queue_size: Max message queue size
            enable_compression: Enable message compression
            compression_threshold: Compress messages larger than this (bytes)
            ping_interval: WebSocket ping interval
            ping_timeout: WebSocket ping timeout
            extra_headers: Additional headers for connection
            json_parse_workers: Number of thread pool workers for JSON parsing
            callback_timeout: Timeout for message callbacks in seconds
            max_json_size_mb: Maximum JSON message size in MB (for validation)
        """
        if websockets is None:
            raise ImportError("websockets library is required. Install with: pip install websockets")

        self.url = url
        self.backoff_strategy = backoff_strategy or ExponentialBackoff()
        self.max_reconnect_attempts = max_reconnect_attempts
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.message_queue_size = message_queue_size
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.extra_headers = extra_headers or {}
        self.callback_timeout = callback_timeout
        self.max_json_size_mb = max_json_size_mb
        self.max_json_size_bytes = int(max_json_size_mb * 1024 * 1024)

        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.reconnect_attempts = 0

        # Message management
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=message_queue_size)
        self.pending_messages: Dict[str, Message] = {}
        self.received_message_ids: Set[str] = set()
        self.max_dedup_cache_size = 10000

        # Subscriptions
        self.subscriptions: Set[str] = set()

        # Tasks
        self.receive_task: Optional[asyncio.Task] = None
        self.send_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.reconnect_task: Optional[asyncio.Task] = None

        # Callbacks
        self.on_connect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        self.on_reconnect_callback: Optional[Callable] = None
        self.on_message_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None

        # Metrics
        self.metrics = ConnectionMetrics()

        # Health check
        self.last_heartbeat_time = time.time()
        self.is_healthy = False

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Thread pool for non-blocking JSON parsing
        self._json_executor = ThreadPoolExecutor(
            max_workers=json_parse_workers,
            thread_name_prefix="ws_json_"
        )

    async def connect(self) -> bool:
        """
        Connect to WebSocket server

        Returns:
            True if connected successfully, False otherwise
        """
        if self.state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            logger.warning("Already connected or connecting")
            return False

        self.state = ConnectionState.CONNECTING
        self.metrics.connection_attempts += 1

        try:
            logger.info(f"Connecting to WebSocket: {self.url}")
            self.websocket = await websockets.connect(
                self.url,
                extra_headers=self.extra_headers,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout
            )

            self.state = ConnectionState.CONNECTED
            self.metrics.successful_connections += 1
            self.metrics.last_connect_time = time.time()
            self.reconnect_attempts = 0
            self.backoff_strategy.reset()
            self.is_healthy = True

            # Start background tasks
            self.receive_task = asyncio.create_task(self._receive_loop())
            self.send_task = asyncio.create_task(self._send_loop())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Restore subscriptions
            await self._restore_subscriptions()

            # Trigger callback
            if self.on_connect_callback:
                try:
                    await self.on_connect_callback()
                except Exception as e:
                    logger.error(f"Error in on_connect callback: {e}")

            logger.info("WebSocket connected successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.state = ConnectionState.DISCONNECTED
            if self.on_error_callback:
                try:
                    await self.on_error_callback(e)
                except Exception as callback_error:
                    logger.error(f"Error in on_error callback: {callback_error}")
            return False

    async def disconnect(self):
        """Gracefully disconnect from WebSocket"""
        if self.state == ConnectionState.CLOSED:
            return

        logger.info("Disconnecting WebSocket...")
        self.state = ConnectionState.CLOSED

        # Cancel all tasks
        for task in [self.receive_task, self.send_task, self.heartbeat_task, self.reconnect_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        # Shutdown thread pool
        if hasattr(self, '_json_executor'):
            self._json_executor.shutdown(wait=True, cancel_futures=True)

        self.metrics.last_disconnect_time = time.time()
        self.is_healthy = False

        # Trigger callback
        if self.on_disconnect_callback:
            try:
                await self.on_disconnect_callback()
            except Exception as e:
                logger.error(f"Error in on_disconnect callback: {e}")

        logger.info("WebSocket disconnected")

    async def send(self, data: Any, message_id: Optional[str] = None, require_ack: bool = False) -> bool:
        """
        Send message through WebSocket

        Args:
            data: Data to send (will be JSON serialized)
            message_id: Optional message ID for deduplication
            require_ack: Whether to require acknowledgment

        Returns:
            True if message queued successfully
        """
        if self.state != ConnectionState.CONNECTED:
            logger.warning("Cannot send message: not connected")
            return False

        try:
            message_id = message_id or self._generate_message_id()
            message = Message(
                id=message_id,
                data=data,
                timestamp=time.time()
            )

            if require_ack:
                self.pending_messages[message_id] = message

            await self.message_queue.put(message)
            return True

        except asyncio.QueueFull:
            logger.error("Message queue is full")
            return False
        except Exception as e:
            logger.error(f"Failed to queue message: {e}")
            return False

    async def subscribe(self, channel: str):
        """Subscribe to a channel"""
        self.subscriptions.add(channel)
        if self.state == ConnectionState.CONNECTED:
            await self.send({"action": "subscribe", "channel": channel})

    async def unsubscribe(self, channel: str):
        """Unsubscribe from a channel"""
        self.subscriptions.discard(channel)
        if self.state == ConnectionState.CONNECTED:
            await self.send({"action": "unsubscribe", "channel": channel})

    def acknowledge_message(self, message_id: str):
        """Acknowledge a message"""
        if message_id in self.pending_messages:
            self.pending_messages[message_id].acknowledged = True
            del self.pending_messages[message_id]

    def get_metrics(self) -> Dict[str, Any]:
        """Get connection metrics"""
        return self.metrics.to_dict()

    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.state == ConnectionState.CONNECTED

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            "is_healthy": self.is_healthy,
            "state": self.state.value,
            "reconnect_attempts": self.reconnect_attempts,
            "last_heartbeat": self.last_heartbeat_time,
            "time_since_heartbeat": time.time() - self.last_heartbeat_time,
            "pending_messages": len(self.pending_messages),
            "queue_size": self.message_queue.qsize()
        }

    def set_callbacks(
        self,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
        on_reconnect: Optional[Callable] = None,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """Set event callbacks"""
        if on_connect:
            self.on_connect_callback = on_connect
        if on_disconnect:
            self.on_disconnect_callback = on_disconnect
        if on_reconnect:
            self.on_reconnect_callback = on_reconnect
        if on_message:
            self.on_message_callback = on_message
        if on_error:
            self.on_error_callback = on_error

    async def _receive_loop(self):
        """Background task to receive messages"""
        try:
            while self.state == ConnectionState.CONNECTED:
                if not self.websocket:
                    break

                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.heartbeat_timeout
                    )

                    # Update metrics
                    self.metrics.total_messages_received += 1
                    self.metrics.total_bytes_received += len(message)
                    self.last_heartbeat_time = time.time()

                    # Process message
                    await self._process_received_message(message)

                except asyncio.TimeoutError:
                    logger.warning("Receive timeout, connection may be dead")
                    await self._handle_connection_lost()
                    break
                except ConnectionClosed:
                    logger.warning("Connection closed by server")
                    await self._handle_connection_lost()
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    await self._handle_connection_lost()
                    break

        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Receive loop error: {e}")

    async def _send_loop(self):
        """Background task to send queued messages"""
        try:
            while self.state == ConnectionState.CONNECTED:
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )

                    if not self.websocket:
                        # Put message back in queue
                        await self.message_queue.put(message)
                        break

                    # Serialize message
                    data = self._serialize_message(message)

                    # Compress if enabled
                    if self.enable_compression and len(data) > self.compression_threshold:
                        data = self._compress_message(data)
                        message.compressed = True

                    # Send message
                    send_start = time.time()
                    await self.websocket.send(data)
                    latency_ms = (time.time() - send_start) * 1000

                    # Update metrics
                    self.metrics.total_messages_sent += 1
                    self.metrics.total_bytes_sent += len(data)
                    self.metrics.record_latency(latency_ms)

                except asyncio.TimeoutError:
                    continue
                except ConnectionClosed:
                    logger.warning("Connection closed during send")
                    await self._handle_connection_lost()
                    break
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
                    message.retry_count += 1
                    if message.retry_count < 3:
                        await self.message_queue.put(message)

        except asyncio.CancelledError:
            logger.debug("Send loop cancelled")
        except Exception as e:
            logger.error(f"Send loop error: {e}")

    async def _heartbeat_loop(self):
        """Background task to send heartbeats"""
        try:
            while self.state == ConnectionState.CONNECTED:
                await asyncio.sleep(self.heartbeat_interval)

                # Check if we've received any messages recently
                time_since_last = time.time() - self.last_heartbeat_time
                if time_since_last > self.heartbeat_timeout:
                    logger.warning("Heartbeat timeout, connection may be dead")
                    self.is_healthy = False
                    await self._handle_connection_lost()
                    break

                # Send heartbeat
                try:
                    if self.websocket:
                        await self.websocket.ping()
                except Exception as e:
                    logger.error(f"Heartbeat failed: {e}")
                    await self._handle_connection_lost()
                    break

        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")

    async def _handle_connection_lost(self):
        """Handle lost connection and initiate reconnection"""
        if self.state == ConnectionState.CLOSED:
            return

        logger.warning("Connection lost, initiating reconnection...")
        self.state = ConnectionState.DISCONNECTED
        self.is_healthy = False

        # Cancel tasks
        for task in [self.receive_task, self.send_task, self.heartbeat_task]:
            if task and not task.done():
                task.cancel()

        # Close websocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket during cleanup: {e}")
            self.websocket = None

        # Start reconnection
        if self.reconnect_task is None or self.reconnect_task.done():
            self.reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self):
        """Background task to handle reconnection"""
        self.state = ConnectionState.RECONNECTING

        try:
            while True:
                # Check max attempts
                if self.max_reconnect_attempts > 0 and self.reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached")
                    self.state = ConnectionState.CLOSED
                    break

                # Calculate delay
                delay = self.backoff_strategy.get_delay(self.reconnect_attempts)
                logger.info(f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_attempts + 1})")
                await asyncio.sleep(delay)

                # Attempt reconnection
                self.metrics.reconnection_attempts += 1
                self.reconnect_attempts += 1

                success = await self.connect()
                if success:
                    self.metrics.successful_reconnections += 1
                    logger.info("Reconnected successfully")

                    # Trigger callback
                    if self.on_reconnect_callback:
                        try:
                            await self.on_reconnect_callback()
                        except Exception as e:
                            logger.error(f"Error in on_reconnect callback: {e}")

                    break

        except asyncio.CancelledError:
            logger.debug("Reconnect loop cancelled")
        except Exception as e:
            logger.error(f"Reconnect loop error: {e}")

    async def _restore_subscriptions(self):
        """Restore subscriptions after reconnection"""
        for channel in self.subscriptions:
            try:
                await self.send({"action": "subscribe", "channel": channel})
                logger.debug(f"Restored subscription: {channel}")
            except Exception as e:
                logger.error(f"Failed to restore subscription {channel}: {e}")

    async def _process_received_message(self, raw_message: str):
        """Process received message with non-blocking JSON parsing and timeout protection"""
        parse_start = time.time()

        try:
            # Check message size
            message_size = len(raw_message)
            if message_size > self.max_json_size_bytes:
                logger.error(
                    f"Message size ({message_size} bytes) exceeds maximum "
                    f"({self.max_json_size_bytes} bytes). Dropping message."
                )
                return

            # Decompress if needed
            if self.enable_compression:
                try:
                    raw_message = self._decompress_message(raw_message)
                except Exception as exc:
                    # Expected for uncompressed frames; logged at debug so a genuine
                    # decompression bug is still visible when tracing.
                    logger.debug("Message not decompressed, using raw payload: %s", exc)

            # Parse JSON in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            try:
                message_data = await loop.run_in_executor(
                    self._json_executor,
                    json.loads,
                    raw_message
                )

                # Record parse time
                parse_time_ms = (time.time() - parse_start) * 1000
                self.metrics.record_parse_time(parse_time_ms)

                if parse_time_ms > 100:
                    logger.warning(
                        f"Slow JSON parse detected: {parse_time_ms:.2f}ms "
                        f"for {message_size} byte message"
                    )

            except json.JSONDecodeError as e:
                self.metrics.json_parse_errors += 1
                logger.error(f"Failed to parse message JSON: {e}")
                return
            except Exception as e:
                self.metrics.json_parse_errors += 1
                logger.error(f"Error parsing JSON in thread pool: {e}")
                return

            # Check for duplicate
            message_id = message_data.get("id")
            if message_id and message_id in self.received_message_ids:
                logger.debug(f"Duplicate message received: {message_id}")
                return

            # Add to dedup cache
            if message_id:
                self.received_message_ids.add(message_id)
                if len(self.received_message_ids) > self.max_dedup_cache_size:
                    # Remove oldest (approximate with set pop)
                    self.received_message_ids.pop()

            # Trigger callback with timeout protection
            if self.on_message_callback:
                callback_start = time.time()
                try:
                    # Ensure callback is awaitable
                    callback_result = self.on_message_callback(message_data)

                    # If it's a coroutine, await it with timeout
                    if asyncio.iscoroutine(callback_result):
                        await asyncio.wait_for(
                            callback_result,
                            timeout=self.callback_timeout
                        )
                    # If it's synchronous, run it in executor to avoid blocking
                    elif callable(callback_result):
                        await loop.run_in_executor(
                            None,  # Use default executor for callbacks
                            callback_result
                        )

                    # Record callback time
                    callback_time_ms = (time.time() - callback_start) * 1000
                    self.metrics.record_callback_time(callback_time_ms)

                    if callback_time_ms > 1000:
                        logger.warning(
                            f"Slow callback detected: {callback_time_ms:.2f}ms"
                        )

                except asyncio.TimeoutError:
                    self.metrics.callback_timeouts += 1
                    logger.warning(
                        f"Message callback timed out after {self.callback_timeout}s"
                    )
                except Exception as e:
                    self.metrics.callback_errors += 1
                    logger.error(f"Error in on_message callback: {e}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _serialize_message(self, message: Message) -> str:
        """Serialize message to JSON"""
        payload = {
            "id": message.id,
            "data": message.data,
            "timestamp": message.timestamp
        }
        return json.dumps(payload)

    def _compress_message(self, data: str) -> bytes:
        """Compress message using zlib"""
        return zlib.compress(data.encode('utf-8'))

    def _decompress_message(self, data: bytes) -> str:
        """Decompress message using zlib"""
        return zlib.decompress(data).decode('utf-8')

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"{int(time.time() * 1000000)}_{random.randint(0, 999999)}"


class WebSocketConnectionPool:
    """
    Connection pool for managing multiple WebSocket connections
    """

    def __init__(self, max_connections: int = 10):
        """
        Initialize connection pool

        Args:
            max_connections: Maximum number of connections
        """
        self.max_connections = max_connections
        self.connections: Dict[str, ResilientWebSocket] = {}
        self._lock = asyncio.Lock()

    async def get_connection(
        self,
        url: str,
        **kwargs
    ) -> ResilientWebSocket:
        """
        Get or create connection for URL

        Args:
            url: WebSocket URL
            **kwargs: Additional arguments for ResilientWebSocket

        Returns:
            ResilientWebSocket instance
        """
        async with self._lock:
            if url in self.connections:
                conn = self.connections[url]
                if conn.is_connected():
                    return conn
                else:
                    # Try to reconnect
                    await conn.connect()
                    return conn

            if len(self.connections) >= self.max_connections:
                raise RuntimeError("Connection pool is full")

            # Create new connection
            conn = ResilientWebSocket(url, **kwargs)
            await conn.connect()
            self.connections[url] = conn
            return conn

    async def close_all(self):
        """Close all connections in pool"""
        async with self._lock:
            for conn in self.connections.values():
                await conn.disconnect()
            self.connections.clear()

    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get metrics for all connections"""
        return {
            url: conn.get_metrics()
            for url, conn in self.connections.items()
        }

    def get_pool_health(self) -> Dict[str, Any]:
        """Get health status for all connections"""
        return {
            "total_connections": len(self.connections),
            "max_connections": self.max_connections,
            "connections": {
                url: conn.get_health_status()
                for url, conn in self.connections.items()
            }
        }
