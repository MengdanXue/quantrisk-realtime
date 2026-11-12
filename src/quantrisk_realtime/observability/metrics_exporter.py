"""
Metrics Exporter for TraderX

Phase 3.2: Prometheus metrics collection and export.

Features:
1. Predefined metrics for trading system monitoring
2. FastAPI middleware for automatic API metrics collection
3. Decorators for function execution time measurement
4. Singleton registry for centralized metrics management
5. Support for Histogram, Counter, and Gauge metrics
"""

import time
import functools
from typing import Callable, Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass, field

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

try:
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    Request = Any
    Response = Any
    BaseHTTPMiddleware = object


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    namespace: str = "traderx"
    subsystem: str = ""
    collect_api_metrics: bool = True
    collect_function_metrics: bool = True
    latency_buckets: List[float] = field(
        default_factory=lambda: [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        ]
    )


class MetricsRegistry:
    """
    Singleton registry for all Prometheus metrics.

    Provides centralized access to metrics and ensures
    only one instance of each metric exists.
    """

    _instance: Optional['MetricsRegistry'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.config = MetricsConfig()
        self.registry = CollectorRegistry()
        self._metrics: Dict[str, Any] = {}

        # Initialize all metrics
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize all predefined metrics."""

        # Trading Latency (Histogram)
        self.trading_latency = Histogram(
            'trading_latency_seconds',
            'Latency of trading operations in seconds',
            ['operation'],
            buckets=self.config.latency_buckets,
            registry=self.registry,
        )
        self._metrics['trading_latency'] = self.trading_latency

        # Order Counter
        self.orders_total = Counter(
            'orders_total',
            'Total number of orders',
            ['symbol', 'side', 'status'],
            registry=self.registry,
        )
        self._metrics['orders_total'] = self.orders_total

        # Position Value (Gauge)
        self.position_value = Gauge(
            'position_value_usd',
            'Current position value in USD',
            ['symbol'],
            registry=self.registry,
        )
        self._metrics['position_value'] = self.position_value

        # WebSocket Connections (Gauge)
        self.websocket_connections = Gauge(
            'websocket_connections',
            'Number of active WebSocket connections',
            registry=self.registry,
        )
        self._metrics['websocket_connections'] = self.websocket_connections

        # Anomaly Detection Counter
        self.anomalies_detected = Counter(
            'anomalies_detected_total',
            'Total number of anomalies detected',
            ['type', 'severity'],
            registry=self.registry,
        )
        self._metrics['anomalies_detected'] = self.anomalies_detected

        # Circuit Breaker State (Gauge)
        # 0 = closed, 1 = open, 2 = half-open
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['service'],
            registry=self.registry,
        )
        self._metrics['circuit_breaker_state'] = self.circuit_breaker_state

        # API Request Counter
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry,
        )
        self._metrics['api_requests_total'] = self.api_requests_total

        # API Request Duration (Histogram)
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['endpoint', 'method'],
            buckets=self.config.latency_buckets,
            registry=self.registry,
        )
        self._metrics['api_request_duration'] = self.api_request_duration

    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get the content type for metrics export."""
        return CONTENT_TYPE_LATEST

    def reset_metrics(self) -> None:
        """Reset all metrics (primarily for testing)."""
        # Note: Prometheus metrics cannot be truly reset,
        # so we need to recreate the registry
        self.registry = CollectorRegistry()
        self._metrics.clear()
        self._init_metrics()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry instance."""
    return MetricsRegistry()


# Decorator for measuring function execution time
def measure_time(
    metric_name: str = 'trading_latency_seconds',
    operation_label: Optional[str] = None,
) -> Callable:
    """
    Decorator to measure function execution time.

    Usage:
        @measure_time(operation_label="order_submission")
        def submit_order(order):
            # Order submission logic
            pass

    Args:
        metric_name: Name of the histogram metric to use
        operation_label: Label for the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            registry = get_metrics_registry()
            metric = registry.get_metric(metric_name.replace('_seconds', ''))

            if metric is None:
                # If metric doesn't exist, just call the function
                return func(*args, **kwargs)

            # Use operation_label if provided, otherwise use function name
            operation = operation_label or func.__name__

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                # Record the observation with the operation label
                if hasattr(metric, 'labels'):
                    metric.labels(operation=operation).observe(duration)
                else:
                    metric.observe(duration)

        return wrapper
    return decorator


@contextmanager
def measure_operation(operation: str):
    """
    Context manager to measure operation duration.

    Usage:
        with measure_operation("order_processing"):
            # Process order
            pass
    """
    registry = get_metrics_registry()
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        registry.trading_latency.labels(operation=operation).observe(duration)


# FastAPI Middleware
if FASTAPI_AVAILABLE:
    class PrometheusMiddleware(BaseHTTPMiddleware):
        """
        FastAPI middleware for automatic API metrics collection.

        Collects:
        - Request count by endpoint, method, and status
        - Request duration by endpoint and method

        Usage:
            from fastapi import FastAPI
            from observability.metrics_exporter import PrometheusMiddleware

            app = FastAPI()
            app.add_middleware(PrometheusMiddleware)
        """

        def __init__(self, app, skip_paths: Optional[List[str]] = None):
            super().__init__(app)
            self.skip_paths = skip_paths or ['/metrics', '/health', '/docs', '/openapi.json']
            self.registry = get_metrics_registry()

        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            # Skip metrics collection for certain paths
            if request.url.path in self.skip_paths:
                return await call_next(request)

            # Record start time
            start_time = time.perf_counter()

            # Process request
            response = await call_next(request)

            # Record metrics
            duration = time.perf_counter() - start_time

            # Sanitize endpoint path (remove path parameters)
            endpoint = self._sanitize_endpoint(request.url.path)
            method = request.method
            status = response.status_code

            # Record request count
            self.registry.api_requests_total.labels(
                endpoint=endpoint,
                method=method,
                status=str(status),
            ).inc()

            # Record request duration
            self.registry.api_request_duration.labels(
                endpoint=endpoint,
                method=method,
            ).observe(duration)

            return response

        def _sanitize_endpoint(self, path: str) -> str:
            """
            Sanitize endpoint path to avoid high cardinality.

            Converts /api/orders/123 to /api/orders/{id}
            """
            # Simple implementation - can be enhanced with route matching
            parts = path.split('/')
            sanitized_parts = []

            for part in parts:
                # Replace numeric IDs with placeholder
                if part.isdigit():
                    sanitized_parts.append('{id}')
                # Replace UUIDs with placeholder
                elif len(part) == 36 and part.count('-') == 4:
                    sanitized_parts.append('{uuid}')
                else:
                    sanitized_parts.append(part)

            return '/'.join(sanitized_parts)


# Helper functions for common metric operations

def record_order(symbol: str, side: str, status: str) -> None:
    """Record an order event."""
    registry = get_metrics_registry()
    registry.orders_total.labels(
        symbol=symbol,
        side=side,
        status=status,
    ).inc()


def update_position_value(symbol: str, value_usd: float) -> None:
    """Update position value for a symbol."""
    registry = get_metrics_registry()
    registry.position_value.labels(symbol=symbol).set(value_usd)


def increment_websocket_connections(delta: int = 1) -> None:
    """Increment or decrement WebSocket connection count."""
    registry = get_metrics_registry()
    if delta > 0:
        registry.websocket_connections.inc(delta)
    else:
        registry.websocket_connections.dec(abs(delta))


def record_anomaly(anomaly_type: str, severity: str) -> None:
    """Record an anomaly detection event."""
    registry = get_metrics_registry()
    registry.anomalies_detected.labels(
        type=anomaly_type,
        severity=severity,
    ).inc()


def set_circuit_breaker_state(service: str, state: str) -> None:
    """
    Set circuit breaker state.

    Args:
        service: Name of the service
        state: State (closed, open, half_open)
    """
    registry = get_metrics_registry()
    state_map = {
        'closed': 0,
        'open': 1,
        'half_open': 2,
    }
    registry.circuit_breaker_state.labels(service=service).set(
        state_map.get(state, 0)
    )


def export_metrics() -> tuple[bytes, str]:
    """
    Export metrics for HTTP endpoint.

    Returns:
        Tuple of (metrics_data, content_type)
    """
    registry = get_metrics_registry()
    return registry.export_metrics(), registry.get_content_type()


# FastAPI endpoint helper
if FASTAPI_AVAILABLE:
    def create_metrics_endpoint():
        """
        Create a FastAPI endpoint for metrics export.

        Usage:
            from fastapi import FastAPI
            from observability.metrics_exporter import create_metrics_endpoint

            app = FastAPI()

            @app.get("/metrics")
            async def metrics():
                return create_metrics_endpoint()
        """
        from fastapi.responses import Response as FastAPIResponse

        def metrics_handler():
            data, content_type = export_metrics()
            return FastAPIResponse(
                content=data,
                media_type=content_type,
            )

        return metrics_handler()
