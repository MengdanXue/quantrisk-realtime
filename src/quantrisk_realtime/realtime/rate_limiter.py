"""
Trading Rate Limiter Module

Provides multiple rate limiting algorithms for trading operations:
- Token Bucket: Smooth rate limiting with burst capacity
- Sliding Window: Precise time-based rate limiting
- Leaky Bucket: Constant output rate
- Fixed Window: Simple time-window based limiting

Supports multiple dimensions:
- Account-level limiting
- Symbol-level limiting
- Operation type limiting
- Global limiting

Features:
- Decorator support
- FastAPI middleware
- Redis distributed limiting
- Rate limit status queries
- Configurable callbacks
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithm types"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"


class RateLimitDimension(str, Enum):
    """Rate limiting dimensions"""
    ACCOUNT = "account"
    SYMBOL = "symbol"
    ORDER_TYPE = "order_type"
    GLOBAL = "global"
    CUSTOM = "custom"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    # Basic limits
    max_requests_per_second: Optional[int] = None
    max_requests_per_minute: Optional[int] = None
    max_requests_per_hour: Optional[int] = None
    max_requests_per_day: Optional[int] = None

    # Trading specific limits
    max_position_value: Optional[float] = None
    max_order_value: Optional[float] = None

    # Algorithm configuration
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    burst_capacity: Optional[int] = None  # For token bucket

    # Callback configuration
    on_limit_exceeded: Optional[Callable] = None
    on_limit_warning: Optional[Callable] = None
    warning_threshold: float = 0.8  # Trigger warning at 80% capacity


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: float  # Unix timestamp
    retry_after: Optional[float] = None  # Seconds until retry
    dimension: Optional[str] = None
    key: Optional[str] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers"""
        headers = {
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_time)),
        }
        if self.retry_after:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    def __init__(self, result: RateLimitResult):
        self.result = result
        super().__init__(
            f"Rate limit exceeded. Retry after {result.retry_after:.2f}s"
        )


class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters"""

    def __init__(self, config: RateLimitConfig):
        self.config = config

    @abstractmethod
    async def check_limit(self, key: str, weight: int = 1) -> RateLimitResult:
        """Check if request is within rate limit"""
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key"""
        pass

    @abstractmethod
    async def get_status(self, key: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        pass


class TokenBucketLimiter(BaseRateLimiter):
    """Token bucket rate limiter

    Allows burst traffic while maintaining average rate.
    Tokens are added at a constant rate and consumed by requests.
    """

    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

        # Calculate token refill rate
        if config.max_requests_per_second:
            self.refill_rate = config.max_requests_per_second
            self.refill_period = 1.0
        elif config.max_requests_per_minute:
            self.refill_rate = config.max_requests_per_minute / 60.0
            self.refill_period = 1.0
        else:
            self.refill_rate = 10.0  # Default
            self.refill_period = 1.0

        self.capacity = config.burst_capacity or int(self.refill_rate * 2)

    async def check_limit(self, key: str, weight: int = 1) -> RateLimitResult:
        async with self._lock:
            now = time.time()

            if key not in self.buckets:
                self.buckets[key] = {
                    "tokens": self.capacity,
                    "last_refill": now
                }

            bucket = self.buckets[key]

            # Refill tokens
            time_passed = now - bucket["last_refill"]
            tokens_to_add = time_passed * self.refill_rate
            bucket["tokens"] = min(
                self.capacity,
                bucket["tokens"] + tokens_to_add
            )
            bucket["last_refill"] = now

            # Check if enough tokens
            if bucket["tokens"] >= weight:
                bucket["tokens"] -= weight
                remaining = int(bucket["tokens"])

                # Trigger warning callback
                if (self.config.on_limit_warning and
                    remaining < self.capacity * (1 - self.config.warning_threshold)):
                    await self._trigger_callback(self.config.on_limit_warning, key)

                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_time=now + (self.capacity - bucket["tokens"]) / self.refill_rate,
                    key=key
                )
            else:
                # Calculate retry time
                tokens_needed = weight - bucket["tokens"]
                retry_after = tokens_needed / self.refill_rate

                # Trigger limit exceeded callback
                if self.config.on_limit_exceeded:
                    await self._trigger_callback(self.config.on_limit_exceeded, key)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=now + retry_after,
                    retry_after=retry_after,
                    key=key
                )

    async def reset(self, key: str) -> None:
        async with self._lock:
            if key in self.buckets:
                self.buckets[key] = {
                    "tokens": self.capacity,
                    "last_refill": time.time()
                }

    async def get_status(self, key: str) -> Dict[str, Any]:
        async with self._lock:
            if key not in self.buckets:
                return {
                    "tokens": self.capacity,
                    "capacity": self.capacity,
                    "refill_rate": self.refill_rate
                }

            bucket = self.buckets[key]
            now = time.time()
            time_passed = now - bucket["last_refill"]
            current_tokens = min(
                self.capacity,
                bucket["tokens"] + time_passed * self.refill_rate
            )

            return {
                "tokens": current_tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "last_refill": bucket["last_refill"]
            }

    async def _trigger_callback(self, callback: Callable, key: str) -> None:
        """Trigger callback function"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(key)
            else:
                callback(key)
        except Exception as e:
            # Log but don't fail on callback errors
            print(f"Rate limiter callback error: {e}")


class SlidingWindowLimiter(BaseRateLimiter):
    """Sliding window rate limiter

    Maintains precise time-based windows of requests.
    More accurate than fixed window but more memory intensive.
    """

    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.windows: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = asyncio.Lock()

        # Determine window size and limit
        if config.max_requests_per_second:
            self.window_size = 1.0
            self.max_requests = config.max_requests_per_second
        elif config.max_requests_per_minute:
            self.window_size = 60.0
            self.max_requests = config.max_requests_per_minute
        elif config.max_requests_per_hour:
            self.window_size = 3600.0
            self.max_requests = config.max_requests_per_hour
        else:
            self.window_size = 60.0
            self.max_requests = 60

    async def check_limit(self, key: str, weight: int = 1) -> RateLimitResult:
        async with self._lock:
            now = time.time()
            window = self.windows[key]

            # Remove old requests outside the window
            cutoff = now - self.window_size
            while window and window[0] < cutoff:
                window.popleft()

            current_count = len(window)

            if current_count + weight <= self.max_requests:
                # Add request timestamps (weight times)
                for _ in range(weight):
                    window.append(now)

                remaining = self.max_requests - len(window)

                # Calculate reset time (when oldest request expires)
                reset_time = window[0] + self.window_size if window else now

                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_time=reset_time,
                    key=key
                )
            else:
                # Calculate retry time (when enough requests will expire)
                if window:
                    oldest_time = window[0]
                    retry_after = oldest_time + self.window_size - now
                else:
                    retry_after = self.window_size

                if self.config.on_limit_exceeded:
                    await self._trigger_callback(self.config.on_limit_exceeded, key)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=now + retry_after,
                    retry_after=retry_after,
                    key=key
                )

    async def reset(self, key: str) -> None:
        async with self._lock:
            if key in self.windows:
                self.windows[key].clear()

    async def get_status(self, key: str) -> Dict[str, Any]:
        async with self._lock:
            now = time.time()
            window = self.windows[key]

            # Remove old requests
            cutoff = now - self.window_size
            while window and window[0] < cutoff:
                window.popleft()

            return {
                "current_count": len(window),
                "max_requests": self.max_requests,
                "window_size": self.window_size,
                "oldest_request": window[0] if window else None
            }

    async def _trigger_callback(self, callback: Callable, key: str) -> None:
        """Trigger callback function"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(key)
            else:
                callback(key)
        except Exception as e:
            logger.error(f"Rate limit callback failed for key '{key}': {e}", exc_info=True)


class LeakyBucketLimiter(BaseRateLimiter):
    """Leaky bucket rate limiter

    Processes requests at a constant rate.
    Excess requests are queued or rejected.
    """

    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

        # Calculate leak rate
        if config.max_requests_per_second:
            self.leak_rate = config.max_requests_per_second
        elif config.max_requests_per_minute:
            self.leak_rate = config.max_requests_per_minute / 60.0
        else:
            self.leak_rate = 10.0

        self.capacity = config.burst_capacity or int(self.leak_rate * 10)

    async def check_limit(self, key: str, weight: int = 1) -> RateLimitResult:
        async with self._lock:
            now = time.time()

            if key not in self.buckets:
                self.buckets[key] = {
                    "level": 0.0,
                    "last_leak": now
                }

            bucket = self.buckets[key]

            # Leak water from bucket
            time_passed = now - bucket["last_leak"]
            leaked = time_passed * self.leak_rate
            bucket["level"] = max(0, bucket["level"] - leaked)
            bucket["last_leak"] = now

            # Check if bucket has capacity
            if bucket["level"] + weight <= self.capacity:
                bucket["level"] += weight
                remaining = int(self.capacity - bucket["level"])

                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_time=now + bucket["level"] / self.leak_rate,
                    key=key
                )
            else:
                # Calculate when bucket will have capacity
                overflow = bucket["level"] + weight - self.capacity
                retry_after = overflow / self.leak_rate

                if self.config.on_limit_exceeded:
                    await self._trigger_callback(self.config.on_limit_exceeded, key)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=now + retry_after,
                    retry_after=retry_after,
                    key=key
                )

    async def reset(self, key: str) -> None:
        async with self._lock:
            if key in self.buckets:
                self.buckets[key] = {
                    "level": 0.0,
                    "last_leak": time.time()
                }

    async def get_status(self, key: str) -> Dict[str, Any]:
        async with self._lock:
            if key not in self.buckets:
                return {
                    "level": 0.0,
                    "capacity": self.capacity,
                    "leak_rate": self.leak_rate
                }

            bucket = self.buckets[key]
            now = time.time()
            time_passed = now - bucket["last_leak"]
            current_level = max(0, bucket["level"] - time_passed * self.leak_rate)

            return {
                "level": current_level,
                "capacity": self.capacity,
                "leak_rate": self.leak_rate,
                "last_leak": bucket["last_leak"]
            }

    async def _trigger_callback(self, callback: Callable, key: str) -> None:
        """Trigger callback function"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(key)
            else:
                callback(key)
        except Exception as e:
            logger.error(f"Leaky bucket callback failed for key '{key}': {e}", exc_info=True)


class FixedWindowLimiter(BaseRateLimiter):
    """Fixed window rate limiter

    Simple time-window based limiting.
    Fast but can allow bursts at window boundaries.
    """

    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.windows: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

        # Determine window size and limit
        if config.max_requests_per_second:
            self.window_size = 1.0
            self.max_requests = config.max_requests_per_second
        elif config.max_requests_per_minute:
            self.window_size = 60.0
            self.max_requests = config.max_requests_per_minute
        elif config.max_requests_per_hour:
            self.window_size = 3600.0
            self.max_requests = config.max_requests_per_hour
        else:
            self.window_size = 60.0
            self.max_requests = 60

    async def check_limit(self, key: str, weight: int = 1) -> RateLimitResult:
        async with self._lock:
            now = time.time()
            window_id = int(now / self.window_size)

            if key not in self.windows or self.windows[key]["window_id"] != window_id:
                self.windows[key] = {
                    "window_id": window_id,
                    "count": 0,
                    "start_time": window_id * self.window_size
                }

            window = self.windows[key]

            if window["count"] + weight <= self.max_requests:
                window["count"] += weight
                remaining = self.max_requests - window["count"]
                reset_time = (window_id + 1) * self.window_size

                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_time=reset_time,
                    key=key
                )
            else:
                reset_time = (window_id + 1) * self.window_size
                retry_after = reset_time - now

                if self.config.on_limit_exceeded:
                    await self._trigger_callback(self.config.on_limit_exceeded, key)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=retry_after,
                    key=key
                )

    async def reset(self, key: str) -> None:
        async with self._lock:
            if key in self.windows:
                del self.windows[key]

    async def get_status(self, key: str) -> Dict[str, Any]:
        async with self._lock:
            now = time.time()
            window_id = int(now / self.window_size)

            if key not in self.windows or self.windows[key]["window_id"] != window_id:
                return {
                    "count": 0,
                    "max_requests": self.max_requests,
                    "window_size": self.window_size,
                    "window_start": window_id * self.window_size
                }

            window = self.windows[key]
            return {
                "count": window["count"],
                "max_requests": self.max_requests,
                "window_size": self.window_size,
                "window_start": window["start_time"]
            }

    async def _trigger_callback(self, callback: Callable, key: str) -> None:
        """Trigger callback function"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(key)
            else:
                callback(key)
        except Exception as e:
            logger.error(f"Fixed window callback failed for key '{key}': {e}", exc_info=True)


class RedisRateLimiter(BaseRateLimiter):
    """Redis-based distributed rate limiter

    Supports distributed rate limiting across multiple instances.
    Uses Redis for shared state.
    """

    def __init__(self, config: RateLimitConfig, redis_url: str = "redis://localhost"):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisRateLimiter")

        super().__init__(config)
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self._connected = False

        # Determine window size and limit
        if config.max_requests_per_second:
            self.window_size = 1
            self.max_requests = config.max_requests_per_second
        elif config.max_requests_per_minute:
            self.window_size = 60
            self.max_requests = config.max_requests_per_minute
        else:
            self.window_size = 60
            self.max_requests = 60

    async def connect(self) -> None:
        """Connect to Redis"""
        if not self._connected:
            self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)
            self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self._connected and self.redis:
            await self.redis.close()
            self._connected = False

    async def check_limit(self, key: str, weight: int = 1) -> RateLimitResult:
        if not self._connected:
            await self.connect()

        now = int(time.time())
        window_id = now // self.window_size
        redis_key = f"ratelimit:{key}:{window_id}"

        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        pipe.get(redis_key)
        pipe.incr(redis_key)
        pipe.expire(redis_key, self.window_size * 2)
        results = await pipe.execute()

        current_count = int(results[0]) if results[0] else 0
        new_count = int(results[1])

        if current_count + weight <= self.max_requests:
            remaining = self.max_requests - new_count
            reset_time = (window_id + 1) * self.window_size

            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=reset_time,
                key=key
            )
        else:
            # Rollback the increment
            await self.redis.decr(redis_key)

            reset_time = (window_id + 1) * self.window_size
            retry_after = reset_time - now

            if self.config.on_limit_exceeded:
                await self._trigger_callback(self.config.on_limit_exceeded, key)

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after,
                key=key
            )

    async def reset(self, key: str) -> None:
        if not self._connected:
            await self.connect()

        # Delete all keys matching the pattern
        pattern = f"ratelimit:{key}:*"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)

    async def get_status(self, key: str) -> Dict[str, Any]:
        if not self._connected:
            await self.connect()

        now = int(time.time())
        window_id = now // self.window_size
        redis_key = f"ratelimit:{key}:{window_id}"

        count = await self.redis.get(redis_key)
        current_count = int(count) if count else 0

        return {
            "count": current_count,
            "max_requests": self.max_requests,
            "window_size": self.window_size,
            "window_start": window_id * self.window_size
        }

    async def _trigger_callback(self, callback: Callable, key: str) -> None:
        """Trigger callback function"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(key)
            else:
                callback(key)
        except Exception as e:
            logger.error(f"Redis rate limiter callback failed for key '{key}': {e}", exc_info=True)


class TradingRateLimiter:
    """Main trading rate limiter with multi-dimensional limiting

    Supports multiple dimensions and algorithms for comprehensive
    rate limiting of trading operations.
    """

    def __init__(
        self,
        global_config: Optional[RateLimitConfig] = None,
        use_redis: bool = False,
        redis_url: str = "redis://localhost"
    ):
        self.global_config = global_config or RateLimitConfig()
        self.use_redis = use_redis
        self.redis_url = redis_url

        # Dimension-specific configurations
        self.dimension_configs: Dict[RateLimitDimension, RateLimitConfig] = {}

        # Limiters by dimension
        self.limiters: Dict[RateLimitDimension, BaseRateLimiter] = {}

        # Initialize global limiter
        self._initialize_limiter(RateLimitDimension.GLOBAL, self.global_config)

    def _initialize_limiter(
        self,
        dimension: RateLimitDimension,
        config: RateLimitConfig
    ) -> None:
        """Initialize limiter for a dimension"""
        if self.use_redis and dimension == RateLimitDimension.GLOBAL:
            self.limiters[dimension] = RedisRateLimiter(config, self.redis_url)
        else:
            # Choose algorithm based on config
            if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                self.limiters[dimension] = TokenBucketLimiter(config)
            elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                self.limiters[dimension] = SlidingWindowLimiter(config)
            elif config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                self.limiters[dimension] = LeakyBucketLimiter(config)
            elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                self.limiters[dimension] = FixedWindowLimiter(config)
            else:
                self.limiters[dimension] = TokenBucketLimiter(config)

    def configure_dimension(
        self,
        dimension: RateLimitDimension,
        config: RateLimitConfig
    ) -> None:
        """Configure rate limiting for a specific dimension"""
        self.dimension_configs[dimension] = config
        self._initialize_limiter(dimension, config)

    async def check_limit(
        self,
        dimension: RateLimitDimension,
        key: str,
        weight: int = 1,
        **kwargs
    ) -> RateLimitResult:
        """Check rate limit for a specific dimension and key

        Args:
            dimension: Rate limit dimension
            key: Identifier for the dimension (account_id, symbol, etc.)
            weight: Weight of the request (default 1)
            **kwargs: Additional context (order_value, etc.)

        Returns:
            RateLimitResult with allow/deny decision
        """
        limiter = self.limiters.get(dimension)
        if not limiter:
            # No limiter configured for this dimension, allow
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                reset_time=time.time() + 3600,
                dimension=dimension.value,
                key=key
            )

        # Check position value limits
        if (self.dimension_configs.get(dimension) and
            self.dimension_configs[dimension].max_position_value):
            position_value = kwargs.get('position_value', 0)
            if position_value > self.dimension_configs[dimension].max_position_value:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=time.time(),
                    retry_after=None,
                    dimension=dimension.value,
                    key=key
                )

        # Check order value limits
        if (self.dimension_configs.get(dimension) and
            self.dimension_configs[dimension].max_order_value):
            order_value = kwargs.get('order_value', 0)
            if order_value > self.dimension_configs[dimension].max_order_value:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=time.time(),
                    retry_after=None,
                    dimension=dimension.value,
                    key=key
                )

        result = await limiter.check_limit(key, weight)
        result.dimension = dimension.value
        return result

    async def check_all_limits(
        self,
        account_id: Optional[str] = None,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None,
        weight: int = 1,
        **kwargs
    ) -> List[RateLimitResult]:
        """Check all applicable rate limits

        Returns list of results for all dimensions that apply.
        """
        results = []

        # Check global limit
        global_result = await self.check_limit(
            RateLimitDimension.GLOBAL,
            "global",
            weight,
            **kwargs
        )
        results.append(global_result)

        # Check account limit
        if account_id and RateLimitDimension.ACCOUNT in self.limiters:
            account_result = await self.check_limit(
                RateLimitDimension.ACCOUNT,
                account_id,
                weight,
                **kwargs
            )
            results.append(account_result)

        # Check symbol limit
        if symbol and RateLimitDimension.SYMBOL in self.limiters:
            symbol_result = await self.check_limit(
                RateLimitDimension.SYMBOL,
                symbol,
                weight,
                **kwargs
            )
            results.append(symbol_result)

        # Check order type limit
        if order_type and RateLimitDimension.ORDER_TYPE in self.limiters:
            order_type_result = await self.check_limit(
                RateLimitDimension.ORDER_TYPE,
                order_type,
                weight,
                **kwargs
            )
            results.append(order_type_result)

        return results

    async def is_allowed(
        self,
        account_id: Optional[str] = None,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None,
        weight: int = 1,
        raise_on_limit: bool = False,
        **kwargs
    ) -> Tuple[bool, List[RateLimitResult]]:
        """Check if request is allowed under all applicable limits

        Args:
            account_id: Account identifier
            symbol: Trading symbol
            order_type: Order type
            weight: Request weight
            raise_on_limit: Whether to raise exception on limit exceeded
            **kwargs: Additional context

        Returns:
            Tuple of (allowed, results)

        Raises:
            RateLimitExceeded: If raise_on_limit=True and limit exceeded
        """
        results = await self.check_all_limits(
            account_id, symbol, order_type, weight, **kwargs
        )

        # Check if all limits allow the request
        allowed = all(result.allowed for result in results)

        if not allowed and raise_on_limit:
            # Find first failed result
            failed_result = next(r for r in results if not r.allowed)
            raise RateLimitExceeded(failed_result)

        return allowed, results

    async def reset(
        self,
        dimension: RateLimitDimension,
        key: str
    ) -> None:
        """Reset rate limit for a specific dimension and key"""
        limiter = self.limiters.get(dimension)
        if limiter:
            await limiter.reset(key)

    async def reset_all(
        self,
        account_id: Optional[str] = None,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None
    ) -> None:
        """Reset all applicable rate limits"""
        if account_id and RateLimitDimension.ACCOUNT in self.limiters:
            await self.reset(RateLimitDimension.ACCOUNT, account_id)

        if symbol and RateLimitDimension.SYMBOL in self.limiters:
            await self.reset(RateLimitDimension.SYMBOL, symbol)

        if order_type and RateLimitDimension.ORDER_TYPE in self.limiters:
            await self.reset(RateLimitDimension.ORDER_TYPE, order_type)

    async def get_status(
        self,
        dimension: RateLimitDimension,
        key: str
    ) -> Dict[str, Any]:
        """Get rate limit status for a specific dimension and key"""
        limiter = self.limiters.get(dimension)
        if not limiter:
            return {"error": "No limiter configured for dimension"}

        return await limiter.get_status(key)

    async def get_all_status(
        self,
        account_id: Optional[str] = None,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get status for all applicable rate limits"""
        status = {}

        # Global status
        status["global"] = await self.get_status(RateLimitDimension.GLOBAL, "global")

        # Account status
        if account_id and RateLimitDimension.ACCOUNT in self.limiters:
            status["account"] = await self.get_status(
                RateLimitDimension.ACCOUNT, account_id
            )

        # Symbol status
        if symbol and RateLimitDimension.SYMBOL in self.limiters:
            status["symbol"] = await self.get_status(
                RateLimitDimension.SYMBOL, symbol
            )

        # Order type status
        if order_type and RateLimitDimension.ORDER_TYPE in self.limiters:
            status["order_type"] = await self.get_status(
                RateLimitDimension.ORDER_TYPE, order_type
            )

        return status

    def rate_limit(
        self,
        dimension: RateLimitDimension = RateLimitDimension.GLOBAL,
        key_extractor: Optional[Callable] = None,
        weight_extractor: Optional[Callable] = None,
        raise_on_limit: bool = True
    ):
        """Decorator for rate limiting functions

        Args:
            dimension: Rate limit dimension
            key_extractor: Function to extract key from function args
            weight_extractor: Function to extract weight from function args
            raise_on_limit: Whether to raise exception on limit exceeded

        Example:
            @rate_limiter.rate_limit(
                dimension=RateLimitDimension.ACCOUNT,
                key_extractor=lambda *args, **kwargs: kwargs.get('account_id'),
                raise_on_limit=True
            )
            async def place_order(account_id: str, symbol: str, ...):
                ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract key
                if key_extractor:
                    key = key_extractor(*args, **kwargs)
                else:
                    key = kwargs.get('account_id') or kwargs.get('symbol') or 'default'

                # Extract weight
                weight = 1
                if weight_extractor:
                    weight = weight_extractor(*args, **kwargs)

                # Check rate limit
                result = await self.check_limit(dimension, key, weight)

                if not result.allowed:
                    if raise_on_limit:
                        raise RateLimitExceeded(result)
                    return None

                # Call original function
                return await func(*args, **kwargs)

            return wrapper
        return decorator


# FastAPI middleware
try:
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.status import HTTP_429_TOO_MANY_REQUESTS

    class RateLimitMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for rate limiting"""

        def __init__(
            self,
            app,
            rate_limiter: TradingRateLimiter,
            key_extractor: Optional[Callable[[Request], str]] = None
        ):
            super().__init__(app)
            self.rate_limiter = rate_limiter
            self.key_extractor = key_extractor or self._default_key_extractor

        def _default_key_extractor(self, request: Request) -> str:
            """Extract key from request (default: client IP)"""
            return request.client.host if request.client else "unknown"

        async def dispatch(self, request: Request, call_next):
            # Extract key from request
            key = self.key_extractor(request)

            # Check rate limit
            result = await self.rate_limiter.check_limit(
                RateLimitDimension.GLOBAL,
                key
            )

            # Add rate limit headers
            response_headers = result.to_headers()

            if not result.allowed:
                return Response(
                    content="Rate limit exceeded",
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    headers=response_headers
                )

            # Process request
            response = await call_next(request)

            # Add rate limit headers to response
            for header_key, header_value in response_headers.items():
                response.headers[header_key] = header_value

            return response

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# Example usage
if __name__ == "__main__":
    async def example_usage():
        # Create rate limiter
        global_config = RateLimitConfig(
            max_requests_per_second=10,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_capacity=20
        )

        rate_limiter = TradingRateLimiter(global_config=global_config)

        # Configure account-level limiting
        account_config = RateLimitConfig(
            max_requests_per_minute=100,
            max_position_value=100000.0,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW
        )
        rate_limiter.configure_dimension(RateLimitDimension.ACCOUNT, account_config)

        # Configure symbol-level limiting
        symbol_config = RateLimitConfig(
            max_requests_per_second=5,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET
        )
        rate_limiter.configure_dimension(RateLimitDimension.SYMBOL, symbol_config)

        # Test rate limiting
        print("Testing rate limiter...")

        for i in range(15):
            allowed, results = await rate_limiter.is_allowed(
                account_id="ACC001",
                symbol="BTCUSDT",
                order_type="LIMIT"
            )

            print(f"\nRequest {i+1}:")
            print(f"  Allowed: {allowed}")

            for result in results:
                if not result.allowed:
                    print(f"  Dimension: {result.dimension}")
                    print(f"  Remaining: {result.remaining}")
                    print(f"  Retry after: {result.retry_after:.2f}s")

            if allowed:
                print("  Request processed successfully")
            else:
                print("  Request rejected due to rate limit")

            await asyncio.sleep(0.1)

        # Get status
        print("\n" + "="*50)
        print("Rate Limit Status:")
        status = await rate_limiter.get_all_status(
            account_id="ACC001",
            symbol="BTCUSDT"
        )

        for dimension, dim_status in status.items():
            print(f"\n{dimension.upper()}:")
            for key, value in dim_status.items():
                print(f"  {key}: {value}")

    asyncio.run(example_usage())
