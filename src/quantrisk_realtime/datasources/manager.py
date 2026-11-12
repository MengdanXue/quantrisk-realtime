"""
数据源管理器 (Data Source Manager)

统一管理和协调多个数据源:
- 自动选择最佳数据源
- 故障转移
- 数据聚合
- 健康检查

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

from .base import (
    DataSource,
    DataSourceConfig,
    DataSourceState,
    DataSourceError,
    MarketType,
    DataFrequency,
    OHLCV,
    Quote,
    DataSourceStats,
)

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class SelectionStrategy(str, Enum):
    """数据源选择策略"""
    PRIORITY = "priority"           # 按优先级
    ROUND_ROBIN = "round_robin"     # 轮询
    FASTEST = "fastest"             # 最快响应
    RANDOM = "random"               # 随机


class AggregationMethod(str, Enum):
    """数据聚合方法"""
    FIRST = "first"                 # 第一个成功的
    AVERAGE = "average"             # 平均值
    MEDIAN = "median"               # 中位数
    WEIGHTED = "weighted"           # 加权平均


# ==================== 配置类 ====================

@dataclass
class DataSourceEntry:
    """数据源条目"""
    source: DataSource
    priority: int = 0
    weight: float = 1.0
    enabled: bool = True
    markets: List[MarketType] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.source.name,
            'priority': self.priority,
            'weight': self.weight,
            'enabled': self.enabled,
            'state': self.source.state.value,
            'markets': [m.value for m in self.markets],
        }


@dataclass
class ManagerConfig:
    """管理器配置"""
    selection_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
    aggregation_method: AggregationMethod = AggregationMethod.FIRST
    enable_fallback: bool = True
    fallback_timeout: float = 5.0
    health_check_interval: float = 60.0
    max_concurrent_requests: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            'selection_strategy': self.selection_strategy.value,
            'aggregation_method': self.aggregation_method.value,
            'enable_fallback': self.enable_fallback,
            'fallback_timeout': self.fallback_timeout,
            'health_check_interval': self.health_check_interval,
            'max_concurrent_requests': self.max_concurrent_requests,
        }


# ==================== 数据类 ====================

@dataclass
class HealthStatus:
    """健康状态"""
    source_name: str
    is_healthy: bool
    state: DataSourceState
    latency_ms: float
    error_rate: float
    last_check: datetime
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_name': self.source_name,
            'is_healthy': self.is_healthy,
            'state': self.state.value,
            'latency_ms': self.latency_ms,
            'error_rate': self.error_rate,
            'last_check': self.last_check.isoformat(),
            'last_error': self.last_error,
        }


@dataclass
class AggregatedQuote:
    """聚合报价"""
    symbol: str
    timestamp: datetime
    quotes: Dict[str, Quote]
    best_bid: float
    best_ask: float
    avg_price: float
    spread: float
    sources_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'avg_price': self.avg_price,
            'spread': self.spread,
            'sources_count': self.sources_count,
            'quotes': {k: v.to_dict() for k, v in self.quotes.items()},
        }


# ==================== 主类 ====================

class DataSourceManager:
    """数据源管理器"""

    def __init__(self, config: Optional[ManagerConfig] = None):
        self.config = config or ManagerConfig()
        self._sources: Dict[str, DataSourceEntry] = {}
        self._health_status: Dict[str, HealthStatus] = {}
        self._round_robin_index: int = 0
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

    # ==================== 数据源管理 ====================

    def register(
        self,
        source: DataSource,
        priority: int = 0,
        weight: float = 1.0,
        markets: Optional[List[MarketType]] = None,
    ) -> None:
        """注册数据源"""
        entry = DataSourceEntry(
            source=source,
            priority=priority,
            weight=weight,
            markets=markets or source.supported_markets,
        )
        self._sources[source.name] = entry
        logger.info(f"注册数据源: {source.name}, 优先级: {priority}")

    def unregister(self, name: str) -> bool:
        """注销数据源"""
        if name in self._sources:
            del self._sources[name]
            if name in self._health_status:
                del self._health_status[name]
            logger.info(f"注销数据源: {name}")
            return True
        return False

    def get_source(self, name: str) -> Optional[DataSource]:
        """获取数据源"""
        entry = self._sources.get(name)
        return entry.source if entry else None

    def get_sources_for_market(self, market: MarketType) -> List[DataSource]:
        """获取支持特定市场的数据源"""
        sources = []
        for entry in self._sources.values():
            if entry.enabled and market in entry.markets:
                sources.append(entry.source)
        return sources

    def enable_source(self, name: str) -> bool:
        """启用数据源"""
        if name in self._sources:
            self._sources[name].enabled = True
            return True
        return False

    def disable_source(self, name: str) -> bool:
        """禁用数据源"""
        if name in self._sources:
            self._sources[name].enabled = False
            return True
        return False

    # ==================== 生命周期 ====================

    async def start(self) -> None:
        """启动管理器"""
        self._running = True
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # 连接所有数据源
        connect_tasks = {}
        for name, entry in self._sources.items():
            if entry.enabled:
                connect_tasks[name] = entry.source.connect()

        if connect_tasks:
            results = await asyncio.gather(*connect_tasks.values(), return_exceptions=True)
            for name, result in zip(connect_tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"连接{name}失败: {result}")

        # 启动健康检查
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"数据源管理器已启动，管理 {len(self._sources)} 个数据源")

    async def stop(self) -> None:
        """停止管理器"""
        self._running = False

        # 停止健康检查
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # 断开所有数据源
        disconnect_tasks = []
        for entry in self._sources.values():
            disconnect_tasks.append(entry.source.disconnect())

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        logger.info("数据源管理器已停止")

    # ==================== 数据获取 ====================

    async def fetch_ohlcv(
        self,
        symbol: str,
        market: Optional[MarketType] = None,
        frequency: DataFrequency = DataFrequency.DAILY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        source_name: Optional[str] = None,
    ) -> List[OHLCV]:
        """获取OHLCV数据"""
        # 选择数据源
        sources = self._select_sources(market, source_name)
        if not sources:
            raise DataSourceError("No available data sources")

        # 尝试获取数据
        last_error = None
        for source in sources:
            try:
                async with self._semaphore:
                    return await asyncio.wait_for(
                        source.fetch_ohlcv(symbol, frequency, start_date, end_date, limit),
                        timeout=self.config.fallback_timeout
                    )
            except asyncio.TimeoutError:
                logger.warning(f"{source.name}超时，尝试下一个数据源")
                last_error = DataSourceError(f"Timeout from {source.name}")
            except Exception as e:
                logger.warning(f"{source.name}错误: {e}，尝试下一个数据源")
                last_error = e

            if not self.config.enable_fallback:
                break

        raise last_error or DataSourceError("All sources failed")

    async def fetch_quote(
        self,
        symbol: str,
        market: Optional[MarketType] = None,
        source_name: Optional[str] = None,
    ) -> Optional[Quote]:
        """获取实时报价"""
        sources = self._select_sources(market, source_name)
        if not sources:
            return None

        last_error = None
        for source in sources:
            try:
                async with self._semaphore:
                    quote = await asyncio.wait_for(
                        source.fetch_quote(symbol),
                        timeout=self.config.fallback_timeout
                    )
                    if quote:
                        return quote
            except asyncio.TimeoutError:
                logger.warning(f"{source.name}超时")
                last_error = DataSourceError(f"Timeout from {source.name}")
            except Exception as e:
                logger.warning(f"{source.name}错误: {e}")
                last_error = e

            if not self.config.enable_fallback:
                break

        return None

    async def fetch_aggregated_quote(
        self,
        symbol: str,
        market: Optional[MarketType] = None,
    ) -> Optional[AggregatedQuote]:
        """获取聚合报价（从多个数据源）"""
        sources = self._select_sources(market)
        if not sources:
            return None

        # 并行获取所有报价
        tasks = {}
        for source in sources:
            tasks[source.name] = source.fetch_quote(symbol)

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        quotes = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Quote) and result:
                quotes[name] = result

        if not quotes:
            return None

        # 聚合
        bids = [q.bid for q in quotes.values() if q.bid > 0]
        asks = [q.ask for q in quotes.values() if q.ask > 0]
        prices = [q.last_price for q in quotes.values() if q.last_price > 0]

        best_bid = max(bids) if bids else 0
        best_ask = min(asks) if asks else 0
        avg_price = sum(prices) / len(prices) if prices else 0

        return AggregatedQuote(
            symbol=symbol,
            timestamp=datetime.now(),
            quotes=quotes,
            best_bid=best_bid,
            best_ask=best_ask,
            avg_price=avg_price,
            spread=best_ask - best_bid if best_bid and best_ask else 0,
            sources_count=len(quotes),
        )

    async def fetch_quotes(
        self,
        symbols: List[str],
        market: Optional[MarketType] = None,
    ) -> Dict[str, Quote]:
        """批量获取报价"""
        tasks = {symbol: self.fetch_quote(symbol, market) for symbol in symbols}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        quotes = {}
        for symbol, result in zip(tasks.keys(), results):
            if isinstance(result, Quote) and result:
                quotes[symbol] = result

        return quotes

    # ==================== 数据源选择 ====================

    def _select_sources(
        self,
        market: Optional[MarketType] = None,
        source_name: Optional[str] = None,
    ) -> List[DataSource]:
        """选择数据源"""
        # 指定数据源
        if source_name:
            entry = self._sources.get(source_name)
            if entry and entry.enabled:
                return [entry.source]
            return []

        # 获取可用数据源
        available = []
        for entry in self._sources.values():
            if not entry.enabled:
                continue
            if entry.source.state != DataSourceState.RUNNING:
                continue
            if market and market not in entry.markets:
                continue
            available.append(entry)

        if not available:
            # 尝试未连接但启用的数据源
            for entry in self._sources.values():
                if entry.enabled:
                    if market is None or market in entry.markets:
                        available.append(entry)

        if not available:
            return []

        # 根据策略排序
        if self.config.selection_strategy == SelectionStrategy.PRIORITY:
            available.sort(key=lambda e: e.priority, reverse=True)
        elif self.config.selection_strategy == SelectionStrategy.FASTEST:
            available.sort(
                key=lambda e: self._health_status.get(
                    e.source.name, HealthStatus(
                        source_name=e.source.name,
                        is_healthy=True,
                        state=DataSourceState.IDLE,
                        latency_ms=float('inf'),
                        error_rate=0,
                        last_check=datetime.now(),
                    )
                ).latency_ms
            )
        elif self.config.selection_strategy == SelectionStrategy.ROUND_ROBIN:
            self._round_robin_index = (self._round_robin_index + 1) % len(available)
            available = available[self._round_robin_index:] + available[:self._round_robin_index]

        return [e.source for e in available]

    # ==================== 健康检查 ====================

    async def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            try:
                await self._check_all_health()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查错误: {e}")
                await asyncio.sleep(10)

    async def _check_all_health(self) -> None:
        """检查所有数据源健康状态"""
        for name, entry in self._sources.items():
            if not entry.enabled:
                continue

            try:
                start = datetime.now()
                # 简单的健康检查：尝试获取一个报价
                # 可以根据数据源类型定制
                stats = entry.source.get_stats()
                latency = (datetime.now() - start).total_seconds() * 1000

                error_rate = 0
                if stats.total_requests > 0:
                    error_rate = stats.failed_requests / stats.total_requests

                self._health_status[name] = HealthStatus(
                    source_name=name,
                    is_healthy=entry.source.state == DataSourceState.RUNNING,
                    state=entry.source.state,
                    latency_ms=stats.avg_latency_ms or latency,
                    error_rate=error_rate,
                    last_check=datetime.now(),
                    last_error=stats.last_error,
                )
            except Exception as e:
                self._health_status[name] = HealthStatus(
                    source_name=name,
                    is_healthy=False,
                    state=DataSourceState.ERROR,
                    latency_ms=float('inf'),
                    error_rate=1.0,
                    last_check=datetime.now(),
                    last_error=str(e),
                )

    async def check_health(self, name: str) -> Optional[HealthStatus]:
        """检查单个数据源健康状态"""
        entry = self._sources.get(name)
        if not entry:
            return None

        await self._check_all_health()
        return self._health_status.get(name)

    def get_health_status(self) -> Dict[str, HealthStatus]:
        """获取所有健康状态"""
        return self._health_status.copy()

    # ==================== 统计信息 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计"""
        source_stats = {}
        for name, entry in self._sources.items():
            source_stats[name] = {
                'entry': entry.to_dict(),
                'stats': entry.source.get_stats().to_dict(),
                'health': self._health_status.get(name, {}).to_dict() if name in self._health_status else None,
            }

        return {
            'config': self.config.to_dict(),
            'sources_count': len(self._sources),
            'enabled_count': sum(1 for e in self._sources.values() if e.enabled),
            'healthy_count': sum(1 for h in self._health_status.values() if h.is_healthy),
            'sources': source_stats,
        }

    def get_source_names(self) -> List[str]:
        """获取所有数据源名称"""
        return list(self._sources.keys())


# ==================== 工厂函数 ====================

def create_manager(
    selection_strategy: SelectionStrategy = SelectionStrategy.PRIORITY,
    enable_fallback: bool = True,
) -> DataSourceManager:
    """创建数据源管理器"""
    config = ManagerConfig(
        selection_strategy=selection_strategy,
        enable_fallback=enable_fallback,
    )
    return DataSourceManager(config)


async def create_default_manager() -> DataSourceManager:
    """创建默认配置的管理器（包含所有免费数据源）"""
    from .akshare_source import create_akshare_source
    from .ccxt_source import create_ccxt_source, ExchangeType
    from .coingecko_source import create_coingecko_source
    from .fear_greed_source import create_fear_greed_source

    manager = create_manager()

    # 注册A股数据源
    manager.register(
        create_akshare_source(),
        priority=10,
        markets=[MarketType.STOCK_CN],
    )

    # 注册加密货币数据源
    manager.register(
        create_ccxt_source(ExchangeType.BINANCE),
        priority=10,
        markets=[MarketType.CRYPTO],
    )

    manager.register(
        create_coingecko_source(),
        priority=5,
        markets=[MarketType.CRYPTO],
    )

    # 注册情绪数据源
    manager.register(
        create_fear_greed_source(),
        priority=10,
        markets=[MarketType.CRYPTO, MarketType.STOCK_US],
    )

    return manager
