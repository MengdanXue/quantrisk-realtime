"""
数据源基类测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from quantrisk_realtime.datasources.base import (
    DataSource,
    DataSourceConfig,
    DataSourceState,
    DataSourceError,
    RateLimitError,
    DataNotFoundError,
    MarketType,
    DataFrequency,
    OHLCV,
    Quote,
    DataSourceStats,
    RateLimiter,
    DataCache,
)


class TestOHLCV:
    """OHLCV数据结构测试"""

    def test_ohlcv_creation(self):
        """测试OHLCV创建"""
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000,
            symbol="BTC/USDT",
            source="test",
        )

        assert ohlcv.open == 100.0
        assert ohlcv.high == 105.0
        assert ohlcv.low == 95.0
        assert ohlcv.close == 102.0
        assert ohlcv.volume == 1000000
        assert ohlcv.symbol == "BTC/USDT"

    def test_ohlcv_to_dict(self):
        """测试OHLCV转字典"""
        ohlcv = OHLCV(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000,
            symbol="BTC/USDT",
            source="test",
        )

        d = ohlcv.to_dict()
        assert d['open'] == 100.0
        assert d['close'] == 102.0
        assert d['symbol'] == "BTC/USDT"
        assert '2024-01-01' in d['timestamp']


class TestQuote:
    """Quote数据结构测试"""

    def test_quote_creation(self):
        """测试Quote创建"""
        quote = Quote(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.5,
            last_price=100.2,
            volume=1000000,
            source="test",
        )

        assert quote.bid == 100.0
        assert quote.ask == 100.5
        assert quote.last_price == 100.2

    def test_quote_mid_price(self):
        """测试中间价"""
        quote = Quote(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            bid=100.0,
            ask=101.0,
        )

        assert quote.mid_price == 100.5

    def test_quote_spread(self):
        """测试价差"""
        quote = Quote(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            bid=100.0,
            ask=101.0,
        )

        assert quote.spread == 1.0
        assert quote.spread_pct == pytest.approx(0.995, rel=0.01)

    def test_quote_to_dict(self):
        """测试Quote转字典"""
        quote = Quote(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            bid=100.0,
            ask=101.0,
            last_price=100.5,
        )

        d = quote.to_dict()
        assert d['bid'] == 100.0
        assert d['ask'] == 101.0
        assert d['mid_price'] == 100.5
        assert d['spread'] == 1.0


class TestDataSourceStats:
    """数据源统计测试"""

    def test_stats_creation(self):
        """测试统计创建"""
        stats = DataSourceStats()

        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0

    def test_stats_to_dict(self):
        """测试统计转字典"""
        stats = DataSourceStats(
            total_requests=100,
            successful_requests=90,
            failed_requests=10,
            cache_hits=50,
            cache_misses=40,
        )

        d = stats.to_dict()
        assert d['total_requests'] == 100
        assert d['success_rate'] == 0.9
        assert d['cache_hit_rate'] == pytest.approx(0.556, rel=0.01)


class TestRateLimiter:
    """限流器测试"""

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """测试获取令牌"""
        limiter = RateLimiter(rate_per_second=10.0, burst=5)

        # 第一次应该立即获取
        wait_time = await limiter.acquire()
        assert wait_time == 0

    @pytest.mark.asyncio
    async def test_rate_limiter_burst(self):
        """测试突发限制"""
        limiter = RateLimiter(rate_per_second=10.0, burst=3)

        # 连续获取应该消耗突发配额
        for _ in range(3):
            wait_time = await limiter.acquire()
            assert wait_time == 0

        # 第4次应该需要等待
        wait_time = await limiter.acquire()
        assert wait_time > 0


class TestDataCache:
    """缓存测试"""

    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """测试缓存设置和获取"""
        cache = DataCache(default_ttl=60.0)

        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        """测试缓存过期"""
        cache = DataCache(default_ttl=0.1)

        await cache.set("key1", "value1")
        await asyncio.sleep(0.2)

        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """测试缓存删除"""
        cache = DataCache()

        await cache.set("key1", "value1")
        deleted = await cache.delete("key1")

        assert deleted is True
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """测试缓存清空"""
        cache = DataCache()

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


class TestDataSourceConfig:
    """配置测试"""

    def test_config_defaults(self):
        """测试默认配置"""
        config = DataSourceConfig()

        assert config.enabled is True
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_config_custom(self):
        """测试自定义配置"""
        config = DataSourceConfig(
            name="test",
            timeout=60.0,
            max_retries=5,
        )

        assert config.name == "test"
        assert config.timeout == 60.0
        assert config.max_retries == 5

    def test_config_to_dict(self):
        """测试配置转字典"""
        config = DataSourceConfig(name="test")
        d = config.to_dict()

        assert d['name'] == "test"
        assert 'timeout' in d
        assert 'max_retries' in d


class TestExceptions:
    """异常测试"""

    def test_data_source_error(self):
        """测试数据源错误"""
        error = DataSourceError("Test error")
        assert str(error) == "Test error"

    def test_rate_limit_error(self):
        """测试限流错误"""
        error = RateLimitError("Rate limited", retry_after=60.0)
        assert error.retry_after == 60.0

    def test_data_not_found_error(self):
        """测试数据未找到错误"""
        error = DataNotFoundError("Data not found")
        assert str(error) == "Data not found"


class TestEnums:
    """枚举测试"""

    def test_market_type(self):
        """测试市场类型枚举"""
        assert MarketType.STOCK_US.value == "stock_us"
        assert MarketType.CRYPTO.value == "crypto"

    def test_data_frequency(self):
        """测试数据频率枚举"""
        assert DataFrequency.DAILY.value == "1d"
        assert DataFrequency.MINUTE_1.value == "1m"

    def test_data_source_state(self):
        """测试数据源状态枚举"""
        assert DataSourceState.IDLE.value == "idle"
        assert DataSourceState.RUNNING.value == "running"
