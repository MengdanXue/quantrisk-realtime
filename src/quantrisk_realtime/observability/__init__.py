"""
Observability module for TraderX

Provides:
- Distributed tracing (OpenTelemetry)
- Structured logging
- Metrics collection (Prometheus)
"""

from .tracing import (
    TracingManager,
    TracingConfig,
    get_tracing_manager,
    trace,
    span,
)
from .structured_logging import (
    StructuredLogger,
    LogConfig,
    get_structured_logger,
    log_context,
)
from .metrics_exporter import (
    MetricsRegistry,
    MetricsConfig,
    get_metrics_registry,
    measure_time,
    measure_operation,
    record_order,
    update_position_value,
    increment_websocket_connections,
    record_anomaly,
    set_circuit_breaker_state,
    export_metrics,
)

__all__ = [
    # Tracing
    "TracingManager",
    "TracingConfig",
    "get_tracing_manager",
    "trace",
    "span",
    # Logging
    "StructuredLogger",
    "LogConfig",
    "get_structured_logger",
    "log_context",
    # Metrics
    "MetricsRegistry",
    "MetricsConfig",
    "get_metrics_registry",
    "measure_time",
    "measure_operation",
    "record_order",
    "update_position_value",
    "increment_websocket_connections",
    "record_anomaly",
    "set_circuit_breaker_state",
    "export_metrics",
]
