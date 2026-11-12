"""
数据源基类 (Data Source Base Classes)

提供所有数据源的通用接口和基础设施。

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


# ==================== 异常类 ====================

class DataSourceError(Exception):
    """数据源基础异常"""
    pass


class RateLimitError(DataSourceError):
    """请求限流异常"""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class DataNotFoundError(DataSourceError):
    """数据未找到异常"""
    pass


class ConnectionError(DataSourceError):
    """连接异常"""
    pass


class AuthenticationError(DataSourceError):
    """认证异常"""
    pass


# ==================== 枚举类型 ====================

class DataSourceState(str, Enum):
    """数据源状态"""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    DISCONNECTED = "disconnected"


class MarketType(str, Enum):
    """市场类型"""
    STOCK_US = "stock_us"           # 美股
    STOCK_CN = "stock_cn"           # A股
    STOCK_HK = "stock_hk"           # 港股
    CRYPTO = "crypto"               # 加密货币
    FOREX = "forex"                 # 外汇
    FUTURES = "futures"             # 期货
    OPTIONS = "options"             # 期权
    INDEX = "index"                 # 指数
    MACRO = "macro"                 # 宏观经济


class DataFrequency(str, Enum):
    """数据频率"""
    TICK = "tick"
    SECOND = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


# ==================== 数据类 ====================

@dataclass
class DataSourceConfig:
    """数据源配置基类"""
    name: str = "base"
    enabled: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_second: float = 10.0
    rate_limit_per_minute: float = 60.0
    cache_ttl: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'enabled': self.enabled,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'rate_limit_per_second': self.rate_limit_per_second,
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'cache_ttl': self.cache_ttl,
        }


@dataclass
class OHLCV:
    """OHLCV数据结构"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol,
            'source': self.source,
        }


@dataclass
class Quote:
    """实时报价数据结构"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_price: float = 0.0
    last_size: float = 0.0
    volume: float = 0.0
    source: str = ""

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        if self.mid_price > 0:
            return self.spread / self.mid_price * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'last_price': self.last_price,
            'last_size': self.last_size,
            'volume': self.volume,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'source': self.source,
        }


@dataclass
class DataSourceStats:
    """数据源统计信息"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'rate_limited_requests': self.rate_limited_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'avg_latency_ms': self.avg_latency_ms,
            'success_rate': self.successful_requests / max(1, self.total_requests),
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'last_error': self.last_error,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
        }


# ==================== 限流器 ====================

class RateLimiter:
    """令牌桶限流器"""

    def __init__(
        self,
        rate_per_second: float = 10.0,
        rate_per_minute: float = 60.0,
        burst: int = 10
    ):
        self.rate_per_second = rate_per_second
        self.rate_per_minute = rate_per_minute
        self.burst = burst

        self._tokens = float(burst)
        self._last_update = time.monotonic()
        self._minute_requests: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        获取令牌

        Returns:
            等待的秒数
        """
        async with self._lock:
            now = time.monotonic()

            # 更新令牌
            elapsed = now - self._last_update
            self._tokens = min(
                self.burst,
                self._tokens + elapsed * self.rate_per_second
            )
            self._last_update = now

            # 检查分钟限制
            current_time = datetime.now()
            minute_ago = current_time - timedelta(minutes=1)
            while self._minute_requests and self._minute_requests[0] < minute_ago:
                self._minute_requests.popleft()

            if len(self._minute_requests) >= self.rate_per_minute:
                wait_time = (self._minute_requests[0] + timedelta(minutes=1) - current_time).total_seconds()
                return max(0, wait_time)

            # 检查令牌
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._minute_requests.append(current_time)
                return 0

            # 需要等待
            wait_time = (tokens - self._tokens) / self.rate_per_second
            return wait_time

    async def wait_and_acquire(self, tokens: int = 1) -> None:
        """等待并获取令牌"""
        wait_time = await self.acquire(tokens)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            await self.acquire(tokens)


# ==================== 缓存 ====================

class DataCache:
    """简单的内存缓存"""

    def __init__(self, default_ttl: float = 60.0, max_size: int = 10000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, tuple] = {}  # key -> (value, expire_time)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self._lock:
            if key in self._cache:
                value, expire_time = self._cache[key]
                if datetime.now() < expire_time:
                    return value
                else:
                    del self._cache[key]
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """设置缓存值"""
        async with self._lock:
            # 清理过期条目
            if len(self._cache) >= self.max_size:
                await self._cleanup()

            expire_time = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
            self._cache[key] = (value, expire_time)

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """清空缓存"""
        async with self._lock:
            self._cache.clear()

    async def _cleanup(self) -> None:
        """清理过期条目"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, expire_time) in self._cache.items()
            if expire_time < now
        ]
        for key in expired_keys:
            del self._cache[key]

        # 如果还是太大，删除最旧的
        if len(self._cache) >= self.max_size:
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1][1]
            )
            for key, _ in sorted_items[:len(self._cache) // 2]:
                del self._cache[key]


# ==================== 数据源基类 ====================

class DataSource(ABC):
    """数据源抽象基类"""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.state = DataSourceState.IDLE
        self.stats = DataSourceStats()
        self._rate_limiter = RateLimiter(
            rate_per_second=config.rate_limit_per_second,
            rate_per_minute=config.rate_limit_per_minute,
        )
        self._cache = DataCache(default_ttl=config.cache_ttl)
        self._callbacks: List[Callable] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        pass

    @property
    @abstractmethod
    def supported_markets(self) -> List[MarketType]:
        """支持的市场类型"""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """连接数据源"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        frequency: DataFrequency = DataFrequency.DAILY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """获取OHLCV数据"""
        pass

    @abstractmethod
    async def fetch_quote(self, symbol: str) -> Optional[Quote]:
        """获取实时报价"""
        pass

    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """批量获取报价"""
        results = {}
        for symbol in symbols:
            quote = await self.fetch_quote(symbol)
            if quote:
                results[symbol] = quote
        return results

    # ==================== 辅助方法 ====================

    async def _request_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """带重试的请求"""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # 等待限流
                await self._rate_limiter.wait_and_acquire()

                # 记录请求
                start_time = time.monotonic()
                self.stats.total_requests += 1
                self.stats.last_request_time = datetime.now()

                # 执行请求
                result = await func(*args, **kwargs)

                # 更新统计
                latency = (time.monotonic() - start_time) * 1000
                self._update_latency(latency)
                self.stats.successful_requests += 1

                return result

            except RateLimitError as e:
                self.stats.rate_limited_requests += 1
                self.state = DataSourceState.RATE_LIMITED
                if e.retry_after:
                    await asyncio.sleep(e.retry_after)
                else:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                last_error = e

            except Exception as e:
                self.stats.failed_requests += 1
                self.stats.last_error = str(e)
                self.stats.last_error_time = datetime.now()
                last_error = e

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise last_error or DataSourceError("Unknown error")

    def _update_latency(self, latency_ms: float) -> None:
        """更新平均延迟"""
        total = self.stats.successful_requests
        if total <= 1:
            self.stats.avg_latency_ms = latency_ms
        else:
            # 移动平均
            self.stats.avg_latency_ms = (
                self.stats.avg_latency_ms * (total - 1) + latency_ms
            ) / total

    async def _get_cached(self, key: str) -> Optional[Any]:
        """获取缓存"""
        result = await self._cache.get(key)
        if result is not None:
            self.stats.cache_hits += 1
        else:
            self.stats.cache_misses += 1
        return result

    async def _set_cached(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """设置缓存"""
        await self._cache.set(key, value, ttl)

    def get_stats(self) -> DataSourceStats:
        """获取统计信息"""
        return self.stats

    def add_callback(self, callback: Callable) -> None:
        """添加回调"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """移除回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _notify_callbacks(self, event: str, data: Any) -> None:
        """通知回调"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
