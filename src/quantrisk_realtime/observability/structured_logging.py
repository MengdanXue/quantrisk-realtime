"""
Structured Logging for TraderX

Phase 3.2: Unified structured logging with JSON format,
trace correlation, and contextual fields.

Features:
1. JSON-formatted log output
2. Automatic trace ID correlation
3. Contextual logging with thread-local context
4. Log level filtering
5. Multiple output handlers
"""

import logging
import json
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import traceback


class LogConfig:
    """Configuration for structured logging."""
    def __init__(
        self,
        service_name: str = "traderx-python",
        log_level: str = "INFO",
        json_format: bool = True,
        include_trace_id: bool = True,
        include_timestamp: bool = True,
        include_caller: bool = True,
        output_file: Optional[str] = None,
        max_message_length: int = 10000,
    ):
        self.service_name = service_name
        self.log_level = log_level
        self.json_format = json_format
        self.include_trace_id = include_trace_id
        self.include_timestamp = include_timestamp
        self.include_caller = include_caller
        self.output_file = output_file
        self.max_message_length = max_message_length


class LogContext:
    """Thread-local log context storage."""
    _local = threading.local()

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get the current context dictionary."""
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        return cls._local.context

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a context value."""
        cls.get_context()[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return cls.get_context().get(key, default)

    @classmethod
    def remove(cls, key: str) -> None:
        """Remove a context value."""
        cls.get_context().pop(key, None)

    @classmethod
    def clear(cls) -> None:
        """Clear all context."""
        cls._local.context = {}

    @classmethod
    def update(cls, **kwargs) -> None:
        """Update multiple context values."""
        cls.get_context().update(kwargs)


@contextmanager
def log_context(**kwargs):
    """
    Context manager for temporary log context.

    Usage:
        with log_context(order_id="123", user_id="456"):
            logger.info("Processing order")  # Will include order_id and user_id
    """
    old_values = {}
    for key, value in kwargs.items():
        old_values[key] = LogContext.get(key)
        LogContext.set(key, value)

    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                LogContext.remove(key)
            else:
                LogContext.set(key, old_value)


class JsonFormatter(logging.Formatter):
    """JSON log formatter with structured fields."""

    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "level": record.levelname,
            "logger": record.name,
            "message": self._truncate_message(record.getMessage()),
        }

        # Add timestamp
        if self.config.include_timestamp:
            log_entry["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add service name
        log_entry["service"] = self.config.service_name

        # Add caller info
        if self.config.include_caller:
            log_entry["caller"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add trace correlation
        if self.config.include_trace_id:
            trace_id = LogContext.get("trace_id")
            span_id = LogContext.get("span_id")
            if trace_id:
                log_entry["trace_id"] = trace_id
            if span_id:
                log_entry["span_id"] = span_id

        # Add thread-local context
        context = LogContext.get_context()
        if context:
            # Filter out trace_id and span_id as they're handled separately
            extra_context = {
                k: v for k, v in context.items()
                if k not in ("trace_id", "span_id")
            }
            if extra_context:
                log_entry["context"] = extra_context

        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_entry["extra"] = record.extra_fields

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None,
            }

        return json.dumps(log_entry, default=str, ensure_ascii=False)

    def _truncate_message(self, message: str) -> str:
        """Truncate long messages."""
        if len(message) > self.config.max_message_length:
            return message[:self.config.max_message_length] + "... (truncated)"
        return message


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with context."""

    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config

    def format(self, record: logging.LogRecord) -> str:
        parts = []

        # Timestamp
        if self.config.include_timestamp:
            parts.append(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

        # Level
        parts.append(f"[{record.levelname:8}]")

        # Logger name
        parts.append(f"[{record.name}]")

        # Trace ID
        if self.config.include_trace_id:
            trace_id = LogContext.get("trace_id")
            if trace_id:
                parts.append(f"[trace:{trace_id[:8]}]")

        # Context
        context = LogContext.get_context()
        context_items = {
            k: v for k, v in context.items()
            if k not in ("trace_id", "span_id")
        }
        if context_items:
            context_str = " ".join(f"{k}={v}" for k, v in context_items.items())
            parts.append(f"[{context_str}]")

        # Message
        parts.append(record.getMessage())

        # Caller
        if self.config.include_caller:
            parts.append(f"({record.filename}:{record.lineno})")

        result = " ".join(parts)

        # Exception
        if record.exc_info:
            result += "\n" + "".join(traceback.format_exception(*record.exc_info))

        return result


class StructuredLogger:
    """
    Structured logger with context support.

    Usage:
        logger = StructuredLogger("myapp")

        # Simple logging
        logger.info("User logged in", user_id="123")

        # With context
        with log_context(request_id="abc"):
            logger.info("Processing request")

        # Trading-specific
        logger.log_trade("order_submitted", order_id="123", symbol="BTCUSDT")
    """

    def __init__(self, name: str, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self._logger = logging.getLogger(name)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup log handlers."""
        self._logger.setLevel(getattr(logging, self.config.log_level.upper()))
        self._logger.handlers.clear()

        # Create formatter
        if self.config.json_format:
            formatter = JsonFormatter(self.config)
        else:
            formatter = TextFormatter(self.config)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # File handler (optional)
        if self.config.output_file:
            file_handler = logging.FileHandler(
                self.config.output_file,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def _log(
        self,
        level: int,
        message: str,
        exc_info: bool = False,
        **kwargs
    ) -> None:
        """Internal log method."""
        extra = {'extra_fields': kwargs} if kwargs else {}
        self._logger.log(level, message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, message, exc_info=True, **kwargs)

    # Trading-specific log methods

    def log_trade(
        self,
        event: str,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log a trading event."""
        trade_context = {
            "event": event,
            "domain": "trading",
        }
        if order_id:
            trade_context["order_id"] = order_id
        if symbol:
            trade_context["symbol"] = symbol
        if side:
            trade_context["side"] = side
        if quantity:
            trade_context["quantity"] = quantity
        if price:
            trade_context["price"] = price

        trade_context.update(kwargs)
        self.info(f"Trade event: {event}", **trade_context)

    def log_market_data(
        self,
        event: str,
        symbol: str,
        **kwargs
    ) -> None:
        """Log a market data event."""
        self.info(
            f"Market data: {event}",
            domain="market_data",
            event=event,
            symbol=symbol,
            **kwargs
        )

    def log_risk(
        self,
        event: str,
        level: str = "info",
        **kwargs
    ) -> None:
        """Log a risk management event."""
        log_method = getattr(self, level, self.info)
        log_method(
            f"Risk event: {event}",
            domain="risk",
            event=event,
            **kwargs
        )

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs
    ) -> None:
        """Log a performance metric."""
        self.info(
            f"Performance: {operation}",
            domain="performance",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **kwargs
        )


# Global logger instances
_loggers: Dict[str, StructuredLogger] = {}
_global_config: Optional[LogConfig] = None


def configure_logging(config: LogConfig) -> None:
    """Configure global logging settings."""
    global _global_config
    _global_config = config


def get_structured_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, _global_config)
    return _loggers[name]


# Convenience functions for quick logging
def get_logger(name: str) -> StructuredLogger:
    """Alias for get_structured_logger."""
    return get_structured_logger(name)
