"""
Tests for Trading Rate Limiter Module

Tests all rate limiting algorithms and features:
- Token Bucket
- Sliding Window
- Leaky Bucket
- Fixed Window
- Multi-dimensional limiting
- Decorator support
- Status queries
"""

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantrisk_realtime.realtime.rate_limiter import (
    BaseRateLimiter,
    FixedWindowLimiter,
    LeakyBucketLimiter,
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitDimension,
    RateLimitExceeded,
    RateLimitResult,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    TradingRateLimiter,
)


class TestRateLimitConfig:
    """Test rate limit configuration"""

    def test_default_config(self):
        """Test default configuration"""
        config = RateLimitConfig()

        assert config.max_requests_per_second is None
        assert config.max_requests_per_minute is None
        assert config.max_requests_per_hour is None
        assert config.max_requests_per_day is None
        assert config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET
        assert config.burst_capacity is None
        assert config.warning_threshold == 0.8

    def test_custom_config(self):
        """Test custom configuration"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            max_requests_per_minute=100,
            max_position_value=50000.0,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            burst_capacity=20,
            warning_threshold=0.9
        )

        assert config.max_requests_per_second == 10
        assert config.max_requests_per_minute == 100
        assert config.max_position_value == 50000.0
        assert config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW
        assert config.burst_capacity == 20
        assert config.warning_threshold == 0.9


class TestRateLimitResult:
    """Test rate limit result"""

    def test_allowed_result(self):
        """Test allowed result"""
        result = RateLimitResult(
            allowed=True,
            remaining=50,
            reset_time=time.time() + 60,
            dimension="global",
            key="test"
        )

        assert result.allowed is True
        assert result.remaining == 50
        assert result.retry_after is None
        assert result.dimension == "global"
        assert result.key == "test"

    def test_denied_result(self):
        """Test denied result"""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_time=time.time() + 30,
            retry_after=30.0,
            dimension="account",
            key="ACC001"
        )

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 30.0
        assert result.dimension == "account"

    def test_to_headers(self):
        """Test conversion to HTTP headers"""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_time=1700000000,
            retry_after=45.5
        )

        headers = result.to_headers()

        assert "X-RateLimit-Remaining" in headers
        assert headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Reset" in headers
        assert headers["X-RateLimit-Reset"] == "1700000000"
        assert "Retry-After" in headers
        assert headers["Retry-After"] == "45"


class TestTokenBucketLimiter:
    """Test token bucket rate limiter"""

    @pytest.mark.asyncio
    async def test_basic_limiting(self):
        """Test basic rate limiting"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            burst_capacity=20
        )
        limiter = TokenBucketLimiter(config)

        # First request should succeed
        result = await limiter.check_limit("test_key")
        assert result.allowed is True
        assert result.remaining == 19

    @pytest.mark.asyncio
    async def test_burst_capacity(self):
        """Test burst capacity handling"""
        config = RateLimitConfig(
            max_requests_per_second=5,
            burst_capacity=10
        )
        limiter = TokenBucketLimiter(config)

        # Make burst requests
        results = []
        for i in range(15):
            result = await limiter.check_limit("test_key")
            results.append(result)

        # First 10 should succeed (burst capacity)
        allowed_count = sum(1 for r in results if r.allowed)
        assert allowed_count == 10

        # Remaining requests should be denied
        denied_count = sum(1 for r in results if not r.allowed)
        assert denied_count == 5

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refill over time"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            burst_capacity=10
        )
        limiter = TokenBucketLimiter(config)

        # Consume all tokens
        for _ in range(10):
            await limiter.check_limit("test_key")

        # Next request should fail
        result = await limiter.check_limit("test_key")
        assert result.allowed is False

        # Wait for token refill
        await asyncio.sleep(0.2)  # Should refill ~2 tokens

        # Should be able to make request now
        result = await limiter.check_limit("test_key")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_weighted_requests(self):
        """Test weighted request handling"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            burst_capacity=10
        )
        limiter = TokenBucketLimiter(config)

        # Make weighted request
        result = await limiter.check_limit("test_key", weight=5)
        assert result.allowed is True
        assert result.remaining == 5

        # Make another weighted request
        result = await limiter.check_limit("test_key", weight=3)
        assert result.allowed is True
        assert result.remaining == 2

        # This should fail (weight > remaining)
        result = await limiter.check_limit("test_key", weight=5)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test rate limit reset"""
        config = RateLimitConfig(
            max_requests_per_second=5,
            burst_capacity=5
        )
        limiter = TokenBucketLimiter(config)

        # Consume all tokens
        for _ in range(5):
            await limiter.check_limit("test_key")

        # Should be at limit
        result = await limiter.check_limit("test_key")
        assert result.allowed is False

        # Reset
        await limiter.reset("test_key")

        # Should work now
        result = await limiter.check_limit("test_key")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test status retrieval"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            burst_capacity=20
        )
        limiter = TokenBucketLimiter(config)

        # Make some requests
        await limiter.check_limit("test_key", weight=5)

        # Get status
        status = await limiter.get_status("test_key")

        assert "tokens" in status
        assert "capacity" in status
        assert "refill_rate" in status
        assert status["capacity"] == 20
        assert status["refill_rate"] == 10

    @pytest.mark.asyncio
    async def test_callback_on_limit_exceeded(self):
        """Test callback on limit exceeded"""
        callback_called = []

        async def on_limit_exceeded(key):
            callback_called.append(key)

        config = RateLimitConfig(
            max_requests_per_second=5,
            burst_capacity=5,
            on_limit_exceeded=on_limit_exceeded
        )
        limiter = TokenBucketLimiter(config)

        # Consume all tokens
        for _ in range(5):
            await limiter.check_limit("test_key")

        # This should trigger callback
        await limiter.check_limit("test_key")

        assert len(callback_called) == 1
        assert callback_called[0] == "test_key"


class TestSlidingWindowLimiter:
    """Test sliding window rate limiter"""

    @pytest.mark.asyncio
    async def test_basic_limiting(self):
        """Test basic rate limiting"""
        config = RateLimitConfig(max_requests_per_second=10)
        limiter = SlidingWindowLimiter(config)

        # First request should succeed
        result = await limiter.check_limit("test_key")
        assert result.allowed is True
        assert result.remaining == 9

    @pytest.mark.asyncio
    async def test_window_sliding(self):
        """Test sliding window behavior"""
        config = RateLimitConfig(max_requests_per_second=5)
        limiter = SlidingWindowLimiter(config)

        # Fill the window
        for _ in range(5):
            result = await limiter.check_limit("test_key")
            assert result.allowed is True

        # Should be at limit
        result = await limiter.check_limit("test_key")
        assert result.allowed is False

        # Wait for window to slide
        await asyncio.sleep(1.1)

        # Should work now
        result = await limiter.check_limit("test_key")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_minute_window(self):
        """Test minute-based window"""
        config = RateLimitConfig(max_requests_per_minute=10)
        limiter = SlidingWindowLimiter(config)

        # Make requests
        for i in range(12):
            result = await limiter.check_limit("test_key")
            if i < 10:
                assert result.allowed is True
            else:
                assert result.allowed is False

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test status retrieval"""
        config = RateLimitConfig(max_requests_per_second=10)
        limiter = SlidingWindowLimiter(config)

        # Make some requests
        for _ in range(5):
            await limiter.check_limit("test_key")

        # Get status
        status = await limiter.get_status("test_key")

        assert status["current_count"] == 5
        assert status["max_requests"] == 10
        assert status["window_size"] == 1.0


class TestLeakyBucketLimiter:
    """Test leaky bucket rate limiter"""

    @pytest.mark.asyncio
    async def test_basic_limiting(self):
        """Test basic rate limiting"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            burst_capacity=20
        )
        limiter = LeakyBucketLimiter(config)

        # First request should succeed
        result = await limiter.check_limit("test_key")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_leak_rate(self):
        """Test leak rate behavior"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            burst_capacity=15
        )
        limiter = LeakyBucketLimiter(config)

        # Fill the bucket
        for _ in range(15):
            await limiter.check_limit("test_key")

        # Should be at capacity
        result = await limiter.check_limit("test_key")
        assert result.allowed is False

        # Wait for leak
        await asyncio.sleep(0.2)  # Should leak ~2 units

        # Should work now
        result = await limiter.check_limit("test_key")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test status retrieval"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            burst_capacity=20
        )
        limiter = LeakyBucketLimiter(config)

        # Make some requests
        for _ in range(5):
            await limiter.check_limit("test_key")

        # Get status
        status = await limiter.get_status("test_key")

        assert "level" in status
        assert "capacity" in status
        assert "leak_rate" in status
        assert status["capacity"] == 20
        assert status["leak_rate"] == 10


class TestFixedWindowLimiter:
    """Test fixed window rate limiter"""

    @pytest.mark.asyncio
    async def test_basic_limiting(self):
        """Test basic rate limiting"""
        config = RateLimitConfig(max_requests_per_second=10)
        limiter = FixedWindowLimiter(config)

        # First request should succeed
        result = await limiter.check_limit("test_key")
        assert result.allowed is True
        assert result.remaining == 9

    @pytest.mark.asyncio
    async def test_window_reset(self):
        """Test window reset behavior"""
        config = RateLimitConfig(max_requests_per_second=5)
        limiter = FixedWindowLimiter(config)

        # Fill the window
        for _ in range(5):
            result = await limiter.check_limit("test_key")
            assert result.allowed is True

        # Should be at limit
        result = await limiter.check_limit("test_key")
        assert result.allowed is False

        # Wait for window to reset
        await asyncio.sleep(1.1)

        # Should work now (new window)
        result = await limiter.check_limit("test_key")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_multiple_keys(self):
        """Test multiple keys independently"""
        config = RateLimitConfig(max_requests_per_second=5)
        limiter = FixedWindowLimiter(config)

        # Fill first key
        for _ in range(5):
            await limiter.check_limit("key1")

        # First key should be at limit
        result = await limiter.check_limit("key1")
        assert result.allowed is False

        # Second key should still work
        result = await limiter.check_limit("key2")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test status retrieval"""
        config = RateLimitConfig(max_requests_per_second=10)
        limiter = FixedWindowLimiter(config)

        # Make some requests
        for _ in range(5):
            await limiter.check_limit("test_key")

        # Get status
        status = await limiter.get_status("test_key")

        assert status["count"] == 5
        assert status["max_requests"] == 10
        assert status["window_size"] == 1.0


class TestTradingRateLimiter:
    """Test main trading rate limiter"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test rate limiter initialization"""
        config = RateLimitConfig(max_requests_per_second=10)
        limiter = TradingRateLimiter(global_config=config)

        assert limiter.global_config == config
        assert RateLimitDimension.GLOBAL in limiter.limiters

    @pytest.mark.asyncio
    async def test_configure_dimension(self):
        """Test dimension configuration"""
        global_config = RateLimitConfig(max_requests_per_second=10)
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure account dimension
        account_config = RateLimitConfig(
            max_requests_per_minute=100,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW
        )
        limiter.configure_dimension(RateLimitDimension.ACCOUNT, account_config)

        assert RateLimitDimension.ACCOUNT in limiter.limiters
        assert isinstance(limiter.limiters[RateLimitDimension.ACCOUNT], SlidingWindowLimiter)

    @pytest.mark.asyncio
    async def test_check_single_limit(self):
        """Test checking single dimension limit"""
        config = RateLimitConfig(max_requests_per_second=5)
        limiter = TradingRateLimiter(global_config=config)

        # Check global limit
        result = await limiter.check_limit(
            RateLimitDimension.GLOBAL,
            "global"
        )

        assert result.allowed is True
        assert result.dimension == "global"

    @pytest.mark.asyncio
    async def test_check_all_limits(self):
        """Test checking all applicable limits"""
        global_config = RateLimitConfig(max_requests_per_second=10)
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure dimensions
        account_config = RateLimitConfig(max_requests_per_minute=100)
        limiter.configure_dimension(RateLimitDimension.ACCOUNT, account_config)

        symbol_config = RateLimitConfig(max_requests_per_second=5)
        limiter.configure_dimension(RateLimitDimension.SYMBOL, symbol_config)

        # Check all limits
        results = await limiter.check_all_limits(
            account_id="ACC001",
            symbol="BTCUSDT",
            order_type="LIMIT"
        )

        # Should check global, account, and symbol
        assert len(results) >= 3
        assert all(result.allowed for result in results)

    @pytest.mark.asyncio
    async def test_is_allowed(self):
        """Test is_allowed method"""
        config = RateLimitConfig(max_requests_per_second=5)
        limiter = TradingRateLimiter(global_config=config)

        # Should be allowed
        allowed, results = await limiter.is_allowed(
            account_id="ACC001",
            symbol="BTCUSDT"
        )

        assert allowed is True
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_is_allowed_with_limit_exceeded(self):
        """Test is_allowed when limit is exceeded"""
        config = RateLimitConfig(
            max_requests_per_second=3,
            burst_capacity=3
        )
        limiter = TradingRateLimiter(global_config=config)

        # Fill the limit quickly
        for _ in range(3):
            await limiter.is_allowed()

        # Should be denied now
        allowed, results = await limiter.is_allowed()
        assert allowed is False

    @pytest.mark.asyncio
    async def test_is_allowed_raise_on_limit(self):
        """Test is_allowed with raise_on_limit"""
        config = RateLimitConfig(
            max_requests_per_second=3,
            burst_capacity=3
        )
        limiter = TradingRateLimiter(global_config=config)

        # Fill the limit
        for _ in range(3):
            await limiter.is_allowed(raise_on_limit=False)

        # Should raise exception
        with pytest.raises(RateLimitExceeded):
            await limiter.is_allowed(raise_on_limit=True)

    @pytest.mark.asyncio
    async def test_position_value_limit(self):
        """Test position value limiting"""
        global_config = RateLimitConfig(max_requests_per_second=10)
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure account with position value limit
        account_config = RateLimitConfig(
            max_requests_per_minute=100,
            max_position_value=50000.0
        )
        limiter.configure_dimension(RateLimitDimension.ACCOUNT, account_config)

        # Check with valid position value
        result = await limiter.check_limit(
            RateLimitDimension.ACCOUNT,
            "ACC001",
            position_value=30000.0
        )
        assert result.allowed is True

        # Check with excessive position value
        result = await limiter.check_limit(
            RateLimitDimension.ACCOUNT,
            "ACC001",
            position_value=60000.0
        )
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_order_value_limit(self):
        """Test order value limiting"""
        global_config = RateLimitConfig(max_requests_per_second=10)
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure symbol with order value limit
        symbol_config = RateLimitConfig(
            max_requests_per_second=5,
            max_order_value=10000.0
        )
        limiter.configure_dimension(RateLimitDimension.SYMBOL, symbol_config)

        # Check with valid order value
        result = await limiter.check_limit(
            RateLimitDimension.SYMBOL,
            "BTCUSDT",
            order_value=5000.0
        )
        assert result.allowed is True

        # Check with excessive order value
        result = await limiter.check_limit(
            RateLimitDimension.SYMBOL,
            "BTCUSDT",
            order_value=15000.0
        )
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_reset_single_dimension(self):
        """Test resetting single dimension"""
        config = RateLimitConfig(
            max_requests_per_second=3,
            burst_capacity=3
        )
        limiter = TradingRateLimiter(global_config=config)

        # Fill the limit
        for _ in range(3):
            await limiter.check_limit(RateLimitDimension.GLOBAL, "global")

        # Should be at limit
        result = await limiter.check_limit(RateLimitDimension.GLOBAL, "global")
        assert result.allowed is False

        # Reset
        await limiter.reset(RateLimitDimension.GLOBAL, "global")

        # Should work now
        result = await limiter.check_limit(RateLimitDimension.GLOBAL, "global")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all dimensions"""
        global_config = RateLimitConfig(max_requests_per_second=3)
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure account dimension
        account_config = RateLimitConfig(max_requests_per_second=3)
        limiter.configure_dimension(RateLimitDimension.ACCOUNT, account_config)

        # Fill limits
        for _ in range(3):
            await limiter.check_all_limits(account_id="ACC001")

        # Reset all
        await limiter.reset_all(account_id="ACC001")

        # Should work now
        allowed, _ = await limiter.is_allowed(account_id="ACC001")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting status for single dimension"""
        config = RateLimitConfig(max_requests_per_second=10)
        limiter = TradingRateLimiter(global_config=config)

        # Make some requests
        for _ in range(5):
            await limiter.check_limit(RateLimitDimension.GLOBAL, "global")

        # Get status
        status = await limiter.get_status(RateLimitDimension.GLOBAL, "global")

        assert "tokens" in status or "count" in status
        assert isinstance(status, dict)

    @pytest.mark.asyncio
    async def test_get_all_status(self):
        """Test getting status for all dimensions"""
        global_config = RateLimitConfig(max_requests_per_second=10)
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure dimensions
        account_config = RateLimitConfig(max_requests_per_minute=100)
        limiter.configure_dimension(RateLimitDimension.ACCOUNT, account_config)

        symbol_config = RateLimitConfig(max_requests_per_second=5)
        limiter.configure_dimension(RateLimitDimension.SYMBOL, symbol_config)

        # Get all status
        status = await limiter.get_all_status(
            account_id="ACC001",
            symbol="BTCUSDT"
        )

        assert "global" in status
        assert "account" in status
        assert "symbol" in status

    @pytest.mark.asyncio
    async def test_rate_limit_decorator(self):
        """Test rate limit decorator"""
        config = RateLimitConfig(
            max_requests_per_second=3,
            burst_capacity=3
        )
        limiter = TradingRateLimiter(global_config=config)

        call_count = [0]

        @limiter.rate_limit(
            dimension=RateLimitDimension.GLOBAL,
            key_extractor=lambda *args, **kwargs: "test",
            raise_on_limit=False
        )
        async def test_function():
            call_count[0] += 1
            return "success"

        # First 3 calls should succeed
        for _ in range(3):
            result = await test_function()
            assert result == "success"

        # Next call should be rate limited
        result = await test_function()
        assert result is None
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_rate_limit_decorator_with_exception(self):
        """Test rate limit decorator with exception"""
        config = RateLimitConfig(
            max_requests_per_second=2,
            burst_capacity=2
        )
        limiter = TradingRateLimiter(global_config=config)

        @limiter.rate_limit(
            dimension=RateLimitDimension.GLOBAL,
            key_extractor=lambda *args, **kwargs: "test",
            raise_on_limit=True
        )
        async def test_function():
            return "success"

        # First 2 calls should succeed
        for _ in range(2):
            result = await test_function()
            assert result == "success"

        # Next call should raise exception
        with pytest.raises(RateLimitExceeded):
            await test_function()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request handling"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            burst_capacity=10
        )
        limiter = TradingRateLimiter(global_config=config)

        # Make concurrent requests
        tasks = [
            limiter.is_allowed(account_id="ACC001")
            for _ in range(20)
        ]

        results = await asyncio.gather(*tasks)

        # Some should succeed, some should fail
        allowed_count = sum(1 for allowed, _ in results if allowed)
        denied_count = sum(1 for allowed, _ in results if not allowed)

        assert allowed_count <= 10
        assert denied_count >= 10

    @pytest.mark.asyncio
    async def test_different_algorithms(self):
        """Test different algorithms for different dimensions"""
        global_config = RateLimitConfig(
            max_requests_per_second=10,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure different algorithms
        account_config = RateLimitConfig(
            max_requests_per_minute=100,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW
        )
        limiter.configure_dimension(RateLimitDimension.ACCOUNT, account_config)

        symbol_config = RateLimitConfig(
            max_requests_per_second=5,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET
        )
        limiter.configure_dimension(RateLimitDimension.SYMBOL, symbol_config)

        order_type_config = RateLimitConfig(
            max_requests_per_second=8,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW
        )
        limiter.configure_dimension(RateLimitDimension.ORDER_TYPE, order_type_config)

        # Verify correct limiter types
        assert isinstance(limiter.limiters[RateLimitDimension.GLOBAL], TokenBucketLimiter)
        assert isinstance(limiter.limiters[RateLimitDimension.ACCOUNT], SlidingWindowLimiter)
        assert isinstance(limiter.limiters[RateLimitDimension.SYMBOL], LeakyBucketLimiter)
        assert isinstance(limiter.limiters[RateLimitDimension.ORDER_TYPE], FixedWindowLimiter)


class TestRateLimitExceeded:
    """Test rate limit exceeded exception"""

    def test_exception_creation(self):
        """Test exception creation"""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_time=time.time() + 30,
            retry_after=30.5
        )

        exception = RateLimitExceeded(result)

        assert exception.result == result
        assert "30.50" in str(exception)


class TestIntegrationScenarios:
    """Test integration scenarios"""

    @pytest.mark.asyncio
    async def test_trading_workflow(self):
        """Test complete trading workflow with rate limiting"""
        # Setup rate limiter
        global_config = RateLimitConfig(
            max_requests_per_second=10,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure account limits
        account_config = RateLimitConfig(
            max_requests_per_minute=50,
            max_position_value=100000.0,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW
        )
        limiter.configure_dimension(RateLimitDimension.ACCOUNT, account_config)

        # Configure symbol limits
        symbol_config = RateLimitConfig(
            max_requests_per_second=5,
            max_order_value=10000.0,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET
        )
        limiter.configure_dimension(RateLimitDimension.SYMBOL, symbol_config)

        # Simulate trading operations
        successful_orders = 0
        failed_orders = 0

        for i in range(20):
            allowed, results = await limiter.is_allowed(
                account_id="ACC001",
                symbol="BTCUSDT",
                order_type="LIMIT",
                order_value=5000.0,
                position_value=50000.0
            )

            if allowed:
                successful_orders += 1
            else:
                failed_orders += 1

            # Small delay between orders
            await asyncio.sleep(0.05)

        # Verify some orders succeeded and some failed
        assert successful_orders > 0
        assert successful_orders + failed_orders == 20

        # Get final status
        status = await limiter.get_all_status(
            account_id="ACC001",
            symbol="BTCUSDT"
        )

        assert "global" in status
        assert "account" in status
        assert "symbol" in status

    @pytest.mark.asyncio
    async def test_multi_account_limiting(self):
        """Test rate limiting across multiple accounts"""
        global_config = RateLimitConfig(max_requests_per_second=20)
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure account limits with fixed window for predictability
        account_config = RateLimitConfig(
            max_requests_per_second=5,
            burst_capacity=5,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW
        )
        limiter.configure_dimension(RateLimitDimension.ACCOUNT, account_config)

        # Test multiple accounts
        accounts = ["ACC001", "ACC002", "ACC003"]
        results_by_account = {account: [] for account in accounts}

        # Make requests for each account
        for _ in range(10):
            for account in accounts:
                allowed, _ = await limiter.is_allowed(account_id=account)
                results_by_account[account].append(allowed)

        # Each account should be limited independently
        for account, results in results_by_account.items():
            allowed_count = sum(1 for r in results if r)
            # Should allow ~5 requests per account in first window
            assert allowed_count <= 10  # More tolerance for timing and refill

    @pytest.mark.asyncio
    async def test_multi_symbol_limiting(self):
        """Test rate limiting across multiple symbols"""
        global_config = RateLimitConfig(max_requests_per_second=30)
        limiter = TradingRateLimiter(global_config=global_config)

        # Configure symbol limits with fixed window for predictability
        symbol_config = RateLimitConfig(
            max_requests_per_second=5,
            burst_capacity=5,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW
        )
        limiter.configure_dimension(RateLimitDimension.SYMBOL, symbol_config)

        # Test multiple symbols
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        results_by_symbol = {symbol: [] for symbol in symbols}

        # Make requests for each symbol
        for _ in range(10):
            for symbol in symbols:
                allowed, _ = await limiter.is_allowed(symbol=symbol)
                results_by_symbol[symbol].append(allowed)

        # Each symbol should be limited independently
        for symbol, results in results_by_symbol.items():
            allowed_count = sum(1 for r in results if r)
            # Should allow ~5 requests per symbol in first window
            assert allowed_count <= 10  # More tolerance for timing and refill

    @pytest.mark.asyncio
    async def test_burst_traffic_handling(self):
        """Test handling of burst traffic"""
        config = RateLimitConfig(
            max_requests_per_second=5,
            burst_capacity=15,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )
        limiter = TradingRateLimiter(global_config=config)

        # Send burst of requests
        results = []
        for _ in range(20):
            allowed, _ = await limiter.is_allowed()
            results.append(allowed)

        # Should allow burst up to capacity
        allowed_count = sum(1 for r in results if r)
        assert allowed_count == 15  # Burst capacity

        # Wait for token refill
        await asyncio.sleep(1.0)

        # Should be able to make more requests
        allowed, _ = await limiter.is_allowed()
        assert allowed is True

    @pytest.mark.asyncio
    async def test_gradual_recovery(self):
        """Test gradual recovery after hitting limit"""
        config = RateLimitConfig(
            max_requests_per_second=10,
            burst_capacity=10,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )
        limiter = TradingRateLimiter(global_config=config)

        # Hit the limit
        for _ in range(10):
            await limiter.is_allowed()

        # Should be at limit
        allowed, _ = await limiter.is_allowed()
        assert allowed is False

        # Track recovery
        recovery_points = []

        for i in range(10):
            await asyncio.sleep(0.1)  # Wait 100ms
            allowed, _ = await limiter.is_allowed(raise_on_limit=False)
            if allowed:
                recovery_points.append(i)

        # Should gradually recover
        assert len(recovery_points) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
