"""
OpenTelemetry Tracing Integration for TraderX

Phase 3.1: Distributed tracing for request tracking across services.

Features:
1. Automatic trace context propagation
2. Span creation for key operations
3. Trace correlation with logs
4. Custom span attributes for trading context
5. Integration with Jaeger/Zipkin exporters
"""

import logging
import functools
import asyncio
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from datetime import datetime

# Try to import OpenTelemetry - graceful fallback if not installed
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.propagate import set_global_textmap, get_global_textmap
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    otel_trace = None

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class TracingConfig:
    """Configuration for tracing."""
    def __init__(
        self,
        service_name: str = "traderx-python",
        enabled: bool = True,
        sample_rate: float = 1.0,
        exporter_type: str = "console",  # console, jaeger, zipkin, otlp
        exporter_endpoint: Optional[str] = None,
        log_correlation: bool = True,
    ):
        self.service_name = service_name
        self.enabled = enabled and OTEL_AVAILABLE
        self.sample_rate = sample_rate
        self.exporter_type = exporter_type
        self.exporter_endpoint = exporter_endpoint
        self.log_correlation = log_correlation


class NoOpSpan:
    """No-op span for when tracing is disabled."""
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class NoOpTracer:
    """No-op tracer for when tracing is disabled."""
    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs):
        return NoOpSpan()


class TracingManager:
    """
    Manages distributed tracing for the application.

    Usage:
        manager = TracingManager(TracingConfig(service_name="traderx"))
        manager.initialize()

        # Use decorator
        @manager.trace("my_operation")
        def my_function():
            pass

        # Use context manager
        with manager.span("my_span") as span:
            span.set_attribute("key", "value")
    """

    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        self._tracer: Any = None
        self._initialized = False
        self._propagator = None

    def initialize(self) -> bool:
        """Initialize the tracing system."""
        if not self.config.enabled:
            logger.info("Tracing disabled")
            self._tracer = NoOpTracer()
            return False

        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not installed, tracing disabled")
            self._tracer = NoOpTracer()
            return False

        try:
            # Create resource
            resource = Resource.create({
                SERVICE_NAME: self.config.service_name,
                "service.version": "1.0.0",
                "deployment.environment": "development",
            })

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Setup exporter
            exporter = self._create_exporter()
            if exporter:
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)

            # Set as global provider
            otel_trace.set_tracer_provider(provider)

            # Setup propagator
            self._propagator = TraceContextTextMapPropagator()
            set_global_textmap(self._propagator)

            # Get tracer
            self._tracer = otel_trace.get_tracer(
                self.config.service_name,
                "1.0.0"
            )

            self._initialized = True
            logger.info(
                "Tracing initialized: service=%s, exporter=%s",
                self.config.service_name,
                self.config.exporter_type
            )
            return True

        except Exception as e:
            logger.error("Failed to initialize tracing: %s", e)
            self._tracer = NoOpTracer()
            return False

    def _create_exporter(self):
        """Create the appropriate span exporter."""
        if self.config.exporter_type == "console":
            return ConsoleSpanExporter()

        elif self.config.exporter_type == "jaeger":
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                return JaegerExporter(
                    agent_host_name=self.config.exporter_endpoint or "localhost",
                    agent_port=6831,
                )
            except ImportError:
                logger.warning("Jaeger exporter not installed")
                return ConsoleSpanExporter()

        elif self.config.exporter_type == "zipkin":
            try:
                from opentelemetry.exporter.zipkin.json import ZipkinExporter
                return ZipkinExporter(
                    endpoint=self.config.exporter_endpoint or "http://localhost:9411/api/v2/spans"
                )
            except ImportError:
                logger.warning("Zipkin exporter not installed")
                return ConsoleSpanExporter()

        elif self.config.exporter_type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                return OTLPSpanExporter(
                    endpoint=self.config.exporter_endpoint or "localhost:4317"
                )
            except ImportError:
                logger.warning("OTLP exporter not installed")
                return ConsoleSpanExporter()

        return ConsoleSpanExporter()

    @property
    def tracer(self):
        """Get the tracer instance."""
        if not self._tracer:
            self._tracer = NoOpTracer()
        return self._tracer

    @contextmanager
    def span(
        self,
        name: str,
        kind: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a span as a context manager.

        Usage:
            with tracing.span("my_operation", attributes={"key": "value"}) as span:
                # do work
                span.set_attribute("result", "success")
        """
        if not self.config.enabled:
            yield NoOpSpan()
            return

        span_kind = kind if kind else (SpanKind.INTERNAL if OTEL_AVAILABLE else None)

        with self.tracer.start_as_current_span(
            name,
            kind=span_kind,
            attributes=attributes or {},
        ) as span:
            try:
                yield span
            except Exception as e:
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                raise

    def trace(
        self,
        name: Optional[str] = None,
        kind: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable[[F], F]:
        """
        Decorator to trace a function.

        Usage:
            @tracing.trace("my_operation")
            def my_function():
                pass

            @tracing.trace()  # Uses function name
            async def my_async_function():
                pass
        """
        def decorator(func: F) -> F:
            span_name = name or func.__name__

            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with self.span(span_name, kind=kind, attributes=attributes) as span:
                        # Add function context
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        try:
                            result = await func(*args, **kwargs)
                            return result
                        except Exception as e:
                            span.set_attribute("error.type", type(e).__name__)
                            span.set_attribute("error.message", str(e))
                            raise
                return async_wrapper  # type: ignore
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.span(span_name, kind=kind, attributes=attributes) as span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        try:
                            result = func(*args, **kwargs)
                            return result
                        except Exception as e:
                            span.set_attribute("error.type", type(e).__name__)
                            span.set_attribute("error.message", str(e))
                            raise
                return sync_wrapper  # type: ignore

        return decorator

    def trace_trading_operation(
        self,
        operation_type: str,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> Callable[[F], F]:
        """
        Specialized decorator for trading operations.

        Usage:
            @tracing.trace_trading_operation("order_submit", symbol="BTCUSDT")
            async def submit_order(order):
                pass
        """
        attributes = {
            "trading.operation_type": operation_type,
        }
        if symbol:
            attributes["trading.symbol"] = symbol
        if order_id:
            attributes["trading.order_id"] = order_id

        return self.trace(
            name=f"trading.{operation_type}",
            attributes=attributes
        )

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID for log correlation."""
        if not self.config.enabled or not OTEL_AVAILABLE:
            return None

        current_span = otel_trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            return format(current_span.get_span_context().trace_id, '032x')
        return None

    def get_current_span_id(self) -> Optional[str]:
        """Get the current span ID for log correlation."""
        if not self.config.enabled or not OTEL_AVAILABLE:
            return None

        current_span = otel_trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            return format(current_span.get_span_context().span_id, '016x')
        return None

    def inject_context(self, carrier: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into a carrier (e.g., HTTP headers)."""
        if not self.config.enabled or not self._propagator:
            return carrier

        get_global_textmap().inject(carrier)
        return carrier

    def extract_context(self, carrier: Dict[str, str]):
        """Extract trace context from a carrier."""
        if not self.config.enabled or not self._propagator:
            return None

        return get_global_textmap().extract(carrier)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the current span."""
        if not self.config.enabled or not OTEL_AVAILABLE:
            return

        current_span = otel_trace.get_current_span()
        if current_span:
            current_span.add_event(name, attributes=attributes or {})

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current span."""
        if not self.config.enabled or not OTEL_AVAILABLE:
            return

        current_span = otel_trace.get_current_span()
        if current_span:
            current_span.set_attribute(key, value)


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager(config: Optional[TracingConfig] = None) -> TracingManager:
    """Get or create the global tracing manager."""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager(config)
        _tracing_manager.initialize()
    return _tracing_manager


def trace(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Convenience decorator using the global tracing manager."""
    return get_tracing_manager().trace(name, attributes=attributes)


def span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Convenience context manager using the global tracing manager."""
    return get_tracing_manager().span(name, attributes=attributes)
