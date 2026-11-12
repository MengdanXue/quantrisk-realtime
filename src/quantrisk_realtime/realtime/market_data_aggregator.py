"""
实时市场数据聚合器 (Market Data Aggregator)

提供统一的市场数据聚合和分发功能:
- 多数据源接入 (交易所、券商、第三方)
- 行情数据标准化
- 订阅管理和分发
- 数据质量监控
- Bar聚合 (Tick -> Bar)
- 市场状态监控

Copyright (c) 2025 TraderX. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import logging
import statistics
from abc import ABC, abstractmethod
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============== Enums ==============

class DataSource(str, Enum):
    """数据源类型"""
    EXCHANGE = "exchange"           # 交易所直连
    BROKER = "broker"               # 券商数据
    VENDOR = "vendor"               # 第三方数据供应商
    AGGREGATOR = "aggregator"       # 聚合数据源
    SIMULATION = "simulation"       # 模拟数据


class MarketState(str, Enum):
    """市场状态"""
    PRE_MARKET = "pre_market"       # 盘前
    OPEN = "open"                   # 开市
    LUNCH_BREAK = "lunch_break"     # 午休
    POST_MARKET = "post_market"     # 盘后
    CLOSED = "closed"               # 休市
    HALTED = "halted"               # 停牌
    AUCTION = "auction"             # 集合竞价
    UNKNOWN = "unknown"             # 未知


class BarInterval(str, Enum):
    """K线周期"""
    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_10 = "10s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


class SubscriptionType(str, Enum):
    """订阅类型"""
    TICK = "tick"                   # 逐笔成交
    QUOTE = "quote"                 # 报价
    BAR = "bar"                     # K线
    DEPTH = "depth"                 # 深度
    TRADE = "trade"                 # 成交
    ORDER_BOOK = "order_book"       # 订单簿
    ALL = "all"                     # 全部


class DataQuality(str, Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"         # 优秀
    GOOD = "good"                   # 良好
    ACCEPTABLE = "acceptable"       # 可接受
    POOR = "poor"                   # 差
    CRITICAL = "critical"           # 严重问题


# ============== Data Classes ==============

@dataclass
class Tick:
    """逐笔成交数据"""
    symbol: str
    price: float
    size: float
    timestamp: datetime
    source: DataSource
    trade_id: str = ""
    side: str = ""                  # 'buy', 'sell', 'unknown'
    exchange: str = ""

    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = f"{self.symbol}_{self.timestamp.timestamp()}"


@dataclass
class Quote:
    """报价数据"""
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp: datetime
    source: DataSource
    exchange: str = ""
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0

    def __post_init__(self):
        if self.mid_price == 0.0 and self.bid_price > 0 and self.ask_price > 0:
            self.mid_price = (self.bid_price + self.ask_price) / 2
        if self.spread == 0.0:
            self.spread = self.ask_price - self.bid_price
        if self.spread_bps == 0.0 and self.mid_price > 0:
            self.spread_bps = (self.spread / self.mid_price) * 10000


@dataclass
class Bar:
    """K线数据"""
    symbol: str
    interval: BarInterval
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    timestamp: datetime                 # Bar开始时间
    close_timestamp: datetime           # Bar结束时间
    source: DataSource
    trade_count: int = 0
    exchange: str = ""

    @property
    def typical_price(self) -> float:
        """典型价格"""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        """价格区间"""
        return self.high - self.low

    @property
    def body(self) -> float:
        """实体大小"""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """是否阳线"""
        return self.close > self.open


@dataclass
class MarketDepth:
    """市场深度数据"""
    symbol: str
    bids: List[Tuple[float, float]]     # [(price, size), ...]
    asks: List[Tuple[float, float]]     # [(price, size), ...]
    timestamp: datetime
    source: DataSource
    exchange: str = ""

    @property
    def bid_depth(self) -> float:
        """买盘深度"""
        return sum(size for _, size in self.bids)

    @property
    def ask_depth(self) -> float:
        """卖盘深度"""
        return sum(size for _, size in self.asks)

    @property
    def imbalance(self) -> float:
        """买卖盘不平衡度"""
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / total


@dataclass
class DataSourceConfig:
    """数据源配置"""
    source_id: str
    source_type: DataSource
    name: str
    priority: int = 1                   # 优先级(1最高)
    enabled: bool = True
    latency_threshold_ms: float = 100   # 延迟阈值
    symbols: Set[str] = field(default_factory=set)
    reconnect_interval_sec: int = 5
    max_reconnect_attempts: int = 10
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subscription:
    """订阅信息"""
    subscription_id: str
    symbol: str
    subscription_type: SubscriptionType
    interval: Optional[BarInterval] = None
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_update: Optional[datetime] = None
    update_count: int = 0


@dataclass
class QualityMetrics:
    """数据质量指标"""
    symbol: str
    source: DataSource
    timestamp: datetime
    latency_ms: float = 0.0
    gap_count: int = 0                  # 数据缺口数量
    stale_count: int = 0                # 过期数据数量
    quality: DataQuality = DataQuality.GOOD
    error_rate: float = 0.0
    uptime: float = 1.0
    message_rate: float = 0.0           # 消息/秒

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'source': self.source.value,
            'timestamp': self.timestamp.isoformat(),
            'latency_ms': self.latency_ms,
            'gap_count': self.gap_count,
            'stale_count': self.stale_count,
            'quality': self.quality.value,
            'error_rate': self.error_rate,
            'uptime': self.uptime,
            'message_rate': self.message_rate,
        }


@dataclass
class AggregatorConfig:
    """聚合器配置"""
    enable_bar_aggregation: bool = True
    bar_intervals: List[BarInterval] = field(default_factory=lambda: [
        BarInterval.MINUTE_1, BarInterval.MINUTE_5
    ])
    enable_quality_monitoring: bool = True
    quality_check_interval_sec: int = 30
    stale_threshold_sec: float = 5.0
    latency_alert_threshold_ms: float = 500
    max_symbols: int = 1000
    buffer_size: int = 10000


# ============== Bar Builder ==============

class BarBuilder:
    """K线构建器"""

    def __init__(self, symbol: str, interval: BarInterval, source: DataSource):
        self.symbol = symbol
        self.interval = interval
        self.source = source
        self._reset()

    def _reset(self):
        """重置状态"""
        self._open: Optional[float] = None
        self._high: float = float('-inf')
        self._low: float = float('inf')
        self._close: float = 0.0
        self._volume: float = 0.0
        self._value: float = 0.0        # 成交额
        self._trade_count: int = 0
        self._start_time: Optional[datetime] = None

    def add_tick(self, tick: Tick) -> Optional[Bar]:
        """
        添加tick数据

        Returns:
            如果完成一根Bar则返回Bar，否则返回None
        """
        # 计算Bar时间边界
        bar_start = self._get_bar_start(tick.timestamp)

        # 检查是否需要完成当前Bar
        completed_bar = None
        if self._start_time is not None and bar_start > self._start_time:
            completed_bar = self._build_bar()
            self._reset()

        # 更新当前Bar
        if self._open is None:
            self._open = tick.price
            self._start_time = bar_start

        self._high = max(self._high, tick.price)
        self._low = min(self._low, tick.price)
        self._close = tick.price
        self._volume += tick.size
        self._value += tick.price * tick.size
        self._trade_count += 1

        return completed_bar

    def _get_bar_start(self, timestamp: datetime) -> datetime:
        """获取Bar开始时间"""
        interval_seconds = self._get_interval_seconds()

        # 对齐到周期边界
        epoch = datetime(1970, 1, 1)
        total_seconds = (timestamp - epoch).total_seconds()
        bar_start_seconds = (total_seconds // interval_seconds) * interval_seconds

        return epoch + timedelta(seconds=bar_start_seconds)

    def _get_interval_seconds(self) -> int:
        """获取周期秒数"""
        mapping = {
            BarInterval.TICK: 0,
            BarInterval.SECOND_1: 1,
            BarInterval.SECOND_5: 5,
            BarInterval.SECOND_10: 10,
            BarInterval.SECOND_30: 30,
            BarInterval.MINUTE_1: 60,
            BarInterval.MINUTE_5: 300,
            BarInterval.MINUTE_15: 900,
            BarInterval.MINUTE_30: 1800,
            BarInterval.HOUR_1: 3600,
            BarInterval.HOUR_4: 14400,
            BarInterval.DAY_1: 86400,
        }
        return mapping.get(self.interval, 60)

    def _build_bar(self) -> Optional[Bar]:
        """构建Bar"""
        if self._open is None or self._start_time is None:
            return None

        interval_seconds = self._get_interval_seconds()
        close_time = self._start_time + timedelta(seconds=interval_seconds)

        vwap = self._value / self._volume if self._volume > 0 else self._close

        return Bar(
            symbol=self.symbol,
            interval=self.interval,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            vwap=vwap,
            timestamp=self._start_time,
            close_timestamp=close_time,
            source=self.source,
            trade_count=self._trade_count,
        )

    def force_complete(self) -> Optional[Bar]:
        """强制完成当前Bar"""
        if self._open is None:
            return None
        bar = self._build_bar()
        self._reset()
        return bar


# ============== Data Source Base ==============

class DataSourceHandler(ABC):
    """数据源处理器基类"""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    @abstractmethod
    async def connect(self) -> bool:
        """连接数据源"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str], sub_type: SubscriptionType) -> bool:
        """订阅标的"""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """取消订阅"""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    def add_callback(self, event_type: str, callback: Callable):
        """添加回调"""
        self._callbacks[event_type].append(callback)

    def remove_callback(self, event_type: str, callback: Callable):
        """移除回调"""
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)

    async def _emit(self, event_type: str, data: Any):
        """触发事件"""
        for callback in self._callbacks[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class SimulatedDataSource(DataSourceHandler):
    """模拟数据源"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self._symbols: Set[str] = set()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        self._connected = True
        logger.info(f"Simulated data source {self.config.source_id} connected")
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._connected = False
        logger.info(f"Simulated data source {self.config.source_id} disconnected")

    async def subscribe(self, symbols: List[str], sub_type: SubscriptionType) -> bool:
        self._symbols.update(symbols)
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._generate_data())
        return True

    async def unsubscribe(self, symbols: List[str]) -> bool:
        self._symbols.difference_update(symbols)
        if not self._symbols and self._running:
            self._running = False
        return True

    async def _generate_data(self):
        """生成模拟数据"""
        import random

        # 初始价格
        prices = {symbol: 100.0 + random.random() * 100 for symbol in self._symbols}

        while self._running:
            try:
                for symbol in list(self._symbols):
                    if symbol not in prices:
                        prices[symbol] = 100.0 + random.random() * 100

                    # 随机价格变动
                    change = (random.random() - 0.5) * 0.02 * prices[symbol]
                    prices[symbol] += change

                    tick = Tick(
                        symbol=symbol,
                        price=round(prices[symbol], 4),
                        size=random.randint(1, 100) * 10,
                        timestamp=datetime.now(),
                        source=DataSource.SIMULATION,
                        side='buy' if random.random() > 0.5 else 'sell',
                    )

                    await self._emit('tick', tick)

                    # 生成报价
                    spread = prices[symbol] * 0.001  # 0.1% spread
                    quote = Quote(
                        symbol=symbol,
                        bid_price=round(prices[symbol] - spread/2, 4),
                        bid_size=random.randint(100, 1000),
                        ask_price=round(prices[symbol] + spread/2, 4),
                        ask_size=random.randint(100, 1000),
                        timestamp=datetime.now(),
                        source=DataSource.SIMULATION,
                    )

                    await self._emit('quote', quote)

                await asyncio.sleep(0.1)  # 100ms interval

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating simulated data: {e}")
                await asyncio.sleep(1)


# ============== Market Data Aggregator ==============

class MarketDataAggregator:
    """
    市场数据聚合器

    功能:
    - 多数据源管理
    - 数据标准化和聚合
    - 订阅管理
    - Bar聚合
    - 数据质量监控
    """

    def __init__(self, config: Optional[AggregatorConfig] = None):
        self.config = config or AggregatorConfig()

        # 数据源管理
        self._sources: Dict[str, DataSourceHandler] = {}
        self._source_configs: Dict[str, DataSourceConfig] = {}

        # 订阅管理
        self._subscriptions: Dict[str, Subscription] = {}
        self._symbol_subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # 数据缓存
        self._latest_ticks: Dict[str, Tick] = {}
        self._latest_quotes: Dict[str, Quote] = {}
        self._latest_bars: Dict[str, Dict[BarInterval, Bar]] = defaultdict(dict)
        self._market_depth: Dict[str, MarketDepth] = {}

        # Bar构建器
        self._bar_builders: Dict[str, Dict[BarInterval, BarBuilder]] = defaultdict(dict)

        # 回调
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # 数据质量
        self._quality_metrics: Dict[str, Dict[DataSource, QualityMetrics]] = defaultdict(dict)
        self._latency_history: Dict[str, List[float]] = defaultdict(list)

        # 状态
        self._running = False
        self._quality_task: Optional[asyncio.Task] = None
        self._subscription_counter = 0

        logger.info("MarketDataAggregator initialized")

    # ============== Source Management ==============

    def register_source(self, config: DataSourceConfig, handler: Optional[DataSourceHandler] = None) -> bool:
        """
        注册数据源

        Args:
            config: 数据源配置
            handler: 数据源处理器(可选,默认使用模拟数据源)
        """
        if config.source_id in self._sources:
            logger.warning(f"Source {config.source_id} already registered")
            return False

        if handler is None:
            handler = simulated_default(
                lambda: SimulatedDataSource(config),
                f"market data for source {config.source_id} (no handler passed, SimulatedDataSource)",
            )

        self._sources[config.source_id] = handler
        self._source_configs[config.source_id] = config

        # 设置回调
        handler.add_callback('tick', lambda t: asyncio.create_task(self._on_tick(t)))
        handler.add_callback('quote', lambda q: asyncio.create_task(self._on_quote(q)))
        handler.add_callback('depth', lambda d: asyncio.create_task(self._on_depth(d)))

        logger.info(f"Registered data source: {config.source_id} ({config.source_type.value})")
        return True

    def unregister_source(self, source_id: str) -> bool:
        """注销数据源"""
        if source_id not in self._sources:
            return False

        del self._sources[source_id]
        del self._source_configs[source_id]
        logger.info(f"Unregistered data source: {source_id}")
        return True

    def get_source(self, source_id: str) -> Optional[DataSourceHandler]:
        """获取数据源处理器"""
        return self._sources.get(source_id)

    def get_sources(self) -> Dict[str, DataSourceConfig]:
        """获取所有数据源配置"""
        return dict(self._source_configs)

    async def connect_all(self) -> Dict[str, bool]:
        """连接所有数据源"""
        results = {}
        for source_id, handler in self._sources.items():
            try:
                results[source_id] = await handler.connect()
            except Exception as e:
                logger.error(f"Failed to connect {source_id}: {e}")
                results[source_id] = False
        return results

    async def disconnect_all(self) -> None:
        """断开所有数据源"""
        for handler in self._sources.values():
            try:
                await handler.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")

    # ============== Subscription Management ==============

    def subscribe(
        self,
        symbol: str,
        sub_type: SubscriptionType = SubscriptionType.QUOTE,
        interval: Optional[BarInterval] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        订阅市场数据

        Args:
            symbol: 标的代码
            sub_type: 订阅类型
            interval: Bar周期(仅Bar订阅需要)
            callback: 数据回调

        Returns:
            订阅ID
        """
        self._subscription_counter += 1
        sub_id = f"sub_{self._subscription_counter}"

        subscription = Subscription(
            subscription_id=sub_id,
            symbol=symbol,
            subscription_type=sub_type,
            interval=interval,
            callback=callback,
        )

        self._subscriptions[sub_id] = subscription
        self._symbol_subscriptions[symbol].add(sub_id)

        # 初始化Bar构建器
        if sub_type in (SubscriptionType.BAR, SubscriptionType.ALL):
            intervals = [interval] if interval else self.config.bar_intervals
            for itv in intervals:
                if itv not in self._bar_builders[symbol]:
                    # 使用优先级最高的数据源
                    source = self._get_primary_source()
                    self._bar_builders[symbol][itv] = BarBuilder(
                        symbol, itv, source or DataSource.SIMULATION
                    )

        # 向数据源订阅 (仅在事件循环运行时)
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._subscribe_to_sources(symbol, sub_type))
        except RuntimeError:
            # 没有运行中的事件循环,跳过异步订阅
            pass

        logger.info(f"Subscribed {symbol} ({sub_type.value}), sub_id={sub_id}")
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions[subscription_id]
        symbol = subscription.symbol

        # 移除订阅
        del self._subscriptions[subscription_id]
        self._symbol_subscriptions[symbol].discard(subscription_id)

        # 如果没有其他订阅,清理资源
        if not self._symbol_subscriptions[symbol]:
            del self._symbol_subscriptions[symbol]
            if symbol in self._bar_builders:
                del self._bar_builders[symbol]
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._unsubscribe_from_sources(symbol))
            except RuntimeError:
                pass

        logger.info(f"Unsubscribed {subscription_id}")
        return True

    def get_subscriptions(self) -> Dict[str, Subscription]:
        """获取所有订阅"""
        return dict(self._subscriptions)

    def get_symbol_subscriptions(self, symbol: str) -> List[Subscription]:
        """获取标的的所有订阅"""
        return [
            self._subscriptions[sub_id]
            for sub_id in self._symbol_subscriptions.get(symbol, set())
        ]

    async def _subscribe_to_sources(self, symbol: str, sub_type: SubscriptionType):
        """向数据源订阅"""
        for handler in self._sources.values():
            if handler.is_connected:
                try:
                    await handler.subscribe([symbol], sub_type)
                except Exception as e:
                    logger.error(f"Error subscribing to source: {e}")

    async def _unsubscribe_from_sources(self, symbol: str):
        """从数据源取消订阅"""
        for handler in self._sources.values():
            if handler.is_connected:
                try:
                    await handler.unsubscribe([symbol])
                except Exception as e:
                    logger.error(f"Error unsubscribing from source: {e}")

    # ============== Data Callbacks ==============

    def add_callback(self, event_type: str, callback: Callable):
        """添加全局回调"""
        self._callbacks[event_type].append(callback)

    def remove_callback(self, event_type: str, callback: Callable):
        """移除全局回调"""
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)

    async def _emit(self, event_type: str, data: Any):
        """触发事件"""
        for callback in self._callbacks[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def _on_tick(self, tick: Tick):
        """处理Tick数据"""
        symbol = tick.symbol

        # 更新最新tick
        self._latest_ticks[symbol] = tick

        # 更新延迟监控
        latency = (datetime.now() - tick.timestamp).total_seconds() * 1000
        self._latency_history[symbol].append(latency)
        if len(self._latency_history[symbol]) > 100:
            self._latency_history[symbol] = self._latency_history[symbol][-100:]

        # Bar聚合
        if self.config.enable_bar_aggregation and symbol in self._bar_builders:
            for interval, builder in self._bar_builders[symbol].items():
                bar = builder.add_tick(tick)
                if bar:
                    self._latest_bars[symbol][interval] = bar
                    await self._emit('bar', bar)
                    await self._notify_bar_subscribers(bar)

        # 触发回调
        await self._emit('tick', tick)
        await self._notify_tick_subscribers(tick)

    async def _on_quote(self, quote: Quote):
        """处理Quote数据"""
        symbol = quote.symbol
        self._latest_quotes[symbol] = quote

        await self._emit('quote', quote)
        await self._notify_quote_subscribers(quote)

    async def _on_depth(self, depth: MarketDepth):
        """处理深度数据"""
        symbol = depth.symbol
        self._market_depth[symbol] = depth

        await self._emit('depth', depth)
        await self._notify_depth_subscribers(depth)

    async def _notify_tick_subscribers(self, tick: Tick):
        """通知Tick订阅者"""
        for sub_id in self._symbol_subscriptions.get(tick.symbol, set()):
            sub = self._subscriptions.get(sub_id)
            if sub and sub.callback and sub.subscription_type in (SubscriptionType.TICK, SubscriptionType.ALL):
                sub.update_count += 1
                sub.last_update = datetime.now()
                try:
                    if asyncio.iscoroutinefunction(sub.callback):
                        await sub.callback(tick)
                    else:
                        sub.callback(tick)
                except Exception as e:
                    logger.error(f"Subscription callback error: {e}")

    async def _notify_quote_subscribers(self, quote: Quote):
        """通知Quote订阅者"""
        for sub_id in self._symbol_subscriptions.get(quote.symbol, set()):
            sub = self._subscriptions.get(sub_id)
            if sub and sub.callback and sub.subscription_type in (SubscriptionType.QUOTE, SubscriptionType.ALL):
                sub.update_count += 1
                sub.last_update = datetime.now()
                try:
                    if asyncio.iscoroutinefunction(sub.callback):
                        await sub.callback(quote)
                    else:
                        sub.callback(quote)
                except Exception as e:
                    logger.error(f"Subscription callback error: {e}")

    async def _notify_bar_subscribers(self, bar: Bar):
        """通知Bar订阅者"""
        for sub_id in self._symbol_subscriptions.get(bar.symbol, set()):
            sub = self._subscriptions.get(sub_id)
            if sub and sub.callback and sub.subscription_type in (SubscriptionType.BAR, SubscriptionType.ALL):
                if sub.interval is None or sub.interval == bar.interval:
                    sub.update_count += 1
                    sub.last_update = datetime.now()
                    try:
                        if asyncio.iscoroutinefunction(sub.callback):
                            await sub.callback(bar)
                        else:
                            sub.callback(bar)
                    except Exception as e:
                        logger.error(f"Subscription callback error: {e}")

    async def _notify_depth_subscribers(self, depth: MarketDepth):
        """通知深度订阅者"""
        for sub_id in self._symbol_subscriptions.get(depth.symbol, set()):
            sub = self._subscriptions.get(sub_id)
            if sub and sub.callback and sub.subscription_type in (SubscriptionType.DEPTH, SubscriptionType.ORDER_BOOK, SubscriptionType.ALL):
                sub.update_count += 1
                sub.last_update = datetime.now()
                try:
                    if asyncio.iscoroutinefunction(sub.callback):
                        await sub.callback(depth)
                    else:
                        sub.callback(depth)
                except Exception as e:
                    logger.error(f"Subscription callback error: {e}")

    # ============== Data Access ==============

    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """获取最新Tick"""
        return self._latest_ticks.get(symbol)

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """获取最新Quote"""
        return self._latest_quotes.get(symbol)

    def get_latest_bar(self, symbol: str, interval: BarInterval) -> Optional[Bar]:
        """获取最新Bar"""
        return self._latest_bars.get(symbol, {}).get(interval)

    def get_market_depth(self, symbol: str) -> Optional[MarketDepth]:
        """获取市场深度"""
        return self._market_depth.get(symbol)

    def get_all_quotes(self) -> Dict[str, Quote]:
        """获取所有最新报价"""
        return dict(self._latest_quotes)

    def get_all_ticks(self) -> Dict[str, Tick]:
        """获取所有最新tick"""
        return dict(self._latest_ticks)

    def get_mid_price(self, symbol: str) -> Optional[float]:
        """获取中间价"""
        quote = self._latest_quotes.get(symbol)
        if quote:
            return quote.mid_price
        tick = self._latest_ticks.get(symbol)
        if tick:
            return tick.price
        return None

    def get_spread(self, symbol: str) -> Optional[float]:
        """获取买卖价差"""
        quote = self._latest_quotes.get(symbol)
        return quote.spread if quote else None

    # ============== Quality Monitoring ==============

    def _get_primary_source(self) -> Optional[DataSource]:
        """获取主数据源"""
        if not self._source_configs:
            return None

        # 按优先级排序
        sorted_configs = sorted(
            self._source_configs.values(),
            key=lambda c: c.priority
        )
        return sorted_configs[0].source_type if sorted_configs else None

    async def start_quality_monitoring(self):
        """启动数据质量监控"""
        if self._quality_task:
            return

        self._running = True
        self._quality_task = asyncio.create_task(self._quality_monitor_loop())
        logger.info("Quality monitoring started")

    async def stop_quality_monitoring(self):
        """停止数据质量监控"""
        self._running = False
        if self._quality_task:
            self._quality_task.cancel()
            try:
                await self._quality_task
            except asyncio.CancelledError:
                pass
            self._quality_task = None
        logger.info("Quality monitoring stopped")

    async def _quality_monitor_loop(self):
        """质量监控循环"""
        while self._running:
            try:
                await self._check_data_quality()
                await asyncio.sleep(self.config.quality_check_interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Quality monitoring error: {e}")
                await asyncio.sleep(5)

    async def _check_data_quality(self):
        """检查数据质量"""
        now = datetime.now()

        for symbol in list(self._symbol_subscriptions.keys()):
            # 检查Tick数据
            tick = self._latest_ticks.get(symbol)
            if tick:
                age = (now - tick.timestamp).total_seconds()
                latency_list = self._latency_history.get(symbol, [])
                avg_latency = statistics.mean(latency_list) if latency_list else 0

                # 确定质量等级
                quality = DataQuality.EXCELLENT
                if age > self.config.stale_threshold_sec:
                    quality = DataQuality.POOR
                elif avg_latency > self.config.latency_alert_threshold_ms:
                    quality = DataQuality.ACCEPTABLE
                elif avg_latency > self.config.latency_alert_threshold_ms / 2:
                    quality = DataQuality.GOOD

                metrics = QualityMetrics(
                    symbol=symbol,
                    source=tick.source,
                    timestamp=now,
                    latency_ms=avg_latency,
                    stale_count=1 if age > self.config.stale_threshold_sec else 0,
                    quality=quality,
                    message_rate=len(latency_list) / self.config.quality_check_interval_sec if latency_list else 0,
                )

                self._quality_metrics[symbol][tick.source] = metrics

                # 质量告警
                if quality in (DataQuality.POOR, DataQuality.CRITICAL):
                    await self._emit('quality_alert', {
                        'symbol': symbol,
                        'quality': quality.value,
                        'latency_ms': avg_latency,
                        'age_sec': age,
                    })

    def get_quality_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取数据质量指标"""
        if symbol:
            metrics = self._quality_metrics.get(symbol, {})
            return {
                source.value: m.to_dict()
                for source, m in metrics.items()
            }

        return {
            sym: {
                source.value: m.to_dict()
                for source, m in metrics.items()
            }
            for sym, metrics in self._quality_metrics.items()
        }

    # ============== Statistics ==============

    def get_statistics(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        return {
            'sources': {
                source_id: {
                    'type': config.source_type.value,
                    'connected': self._sources[source_id].is_connected if source_id in self._sources else False,
                    'priority': config.priority,
                }
                for source_id, config in self._source_configs.items()
            },
            'subscriptions': {
                'total': len(self._subscriptions),
                'by_type': self._count_subscriptions_by_type(),
                'symbols': len(self._symbol_subscriptions),
            },
            'data': {
                'ticks': len(self._latest_ticks),
                'quotes': len(self._latest_quotes),
                'bars': sum(len(bars) for bars in self._latest_bars.values()),
                'depth': len(self._market_depth),
            },
            'quality_monitoring': {
                'enabled': self.config.enable_quality_monitoring,
                'running': self._running,
                'symbols_monitored': len(self._quality_metrics),
            },
            'config': {
                'bar_aggregation': self.config.enable_bar_aggregation,
                'bar_intervals': [i.value for i in self.config.bar_intervals],
                'max_symbols': self.config.max_symbols,
            },
        }

    def _count_subscriptions_by_type(self) -> Dict[str, int]:
        """按类型统计订阅数量"""
        counts: Dict[str, int] = defaultdict(int)
        for sub in self._subscriptions.values():
            counts[sub.subscription_type.value] += 1
        return dict(counts)

    # ============== Lifecycle ==============

    async def start(self) -> bool:
        """启动聚合器"""
        # 连接所有数据源
        results = await self.connect_all()

        # 启动质量监控
        if self.config.enable_quality_monitoring:
            await self.start_quality_monitoring()

        connected_count = sum(1 for v in results.values() if v)
        logger.info(f"MarketDataAggregator started, {connected_count}/{len(results)} sources connected")
        return connected_count > 0

    async def stop(self) -> None:
        """停止聚合器"""
        # 停止质量监控
        await self.stop_quality_monitoring()

        # 断开所有数据源
        await self.disconnect_all()

        logger.info("MarketDataAggregator stopped")


# ============== Factory Functions ==============

def create_aggregator(
    enable_bar_aggregation: bool = True,
    bar_intervals: Optional[List[str]] = None,
    enable_quality_monitoring: bool = True,
    **kwargs
) -> MarketDataAggregator:
    """
    创建市场数据聚合器

    Args:
        enable_bar_aggregation: 是否启用Bar聚合
        bar_intervals: Bar周期列表
        enable_quality_monitoring: 是否启用质量监控
        **kwargs: 其他配置参数
    """
    intervals = []
    if bar_intervals:
        for itv in bar_intervals:
            try:
                intervals.append(BarInterval(itv))
            except ValueError:
                logger.warning(f"Invalid bar interval: {itv}")

    config = AggregatorConfig(
        enable_bar_aggregation=enable_bar_aggregation,
        bar_intervals=intervals or [BarInterval.MINUTE_1, BarInterval.MINUTE_5],
        enable_quality_monitoring=enable_quality_monitoring,
        **{k: v for k, v in kwargs.items() if hasattr(AggregatorConfig, k)}
    )

    return MarketDataAggregator(config)


def create_source_config(
    source_id: str,
    name: str,
    source_type: str = 'simulation',
    priority: int = 1,
    **kwargs
) -> DataSourceConfig:
    """
    创建数据源配置

    Args:
        source_id: 数据源ID
        name: 数据源名称
        source_type: 数据源类型
        priority: 优先级
        **kwargs: 其他配置参数
    """
    try:
        src_type = DataSource(source_type)
    except ValueError:
        src_type = DataSource.SIMULATION

    return DataSourceConfig(
        source_id=source_id,
        source_type=src_type,
        name=name,
        priority=priority,
        **{k: v for k, v in kwargs.items() if hasattr(DataSourceConfig, k)}
    )


def create_tick(
    symbol: str,
    price: float,
    size: float,
    timestamp: Optional[datetime] = None,
    source: str = 'simulation',
    **kwargs
) -> Tick:
    """创建Tick数据"""
    try:
        src = DataSource(source)
    except ValueError:
        src = DataSource.SIMULATION

    return Tick(
        symbol=symbol,
        price=price,
        size=size,
        timestamp=timestamp or datetime.now(),
        source=src,
        **{k: v for k, v in kwargs.items() if k in ['trade_id', 'side', 'exchange']}
    )


def create_quote(
    symbol: str,
    bid_price: float,
    bid_size: float,
    ask_price: float,
    ask_size: float,
    timestamp: Optional[datetime] = None,
    source: str = 'simulation',
    **kwargs
) -> Quote:
    """创建Quote数据"""
    try:
        src = DataSource(source)
    except ValueError:
        src = DataSource.SIMULATION

    return Quote(
        symbol=symbol,
        bid_price=bid_price,
        bid_size=bid_size,
        ask_price=ask_price,
        ask_size=ask_size,
        timestamp=timestamp or datetime.now(),
        source=src,
        **{k: v for k, v in kwargs.items() if k in ['exchange']}
    )


def create_market_depth(
    symbol: str,
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    timestamp: Optional[datetime] = None,
    source: str = 'simulation',
    **kwargs
) -> MarketDepth:
    """创建市场深度数据"""
    try:
        src = DataSource(source)
    except ValueError:
        src = DataSource.SIMULATION

    return MarketDepth(
        symbol=symbol,
        bids=bids,
        asks=asks,
        timestamp=timestamp or datetime.now(),
        source=src,
        **{k: v for k, v in kwargs.items() if k in ['exchange']}
    )


async def quick_aggregate(
    symbols: List[str],
    duration_sec: float = 5.0,
    source_type: str = 'simulation',
) -> Dict[str, Any]:
    """
    快速聚合测试

    Args:
        symbols: 标的列表
        duration_sec: 运行时长(秒)
        source_type: 数据源类型
    """
    aggregator = create_aggregator()

    # 注册模拟数据源
    config = create_source_config(
        source_id='sim_1',
        name='Simulator',
        source_type=source_type,
    )
    aggregator.register_source(config)

    # 收集数据
    collected_ticks: List[Tick] = []
    collected_quotes: List[Quote] = []
    collected_bars: List[Bar] = []

    def on_tick(tick: Tick):
        collected_ticks.append(tick)

    def on_quote(quote: Quote):
        collected_quotes.append(quote)

    def on_bar(bar: Bar):
        collected_bars.append(bar)

    aggregator.add_callback('tick', on_tick)
    aggregator.add_callback('quote', on_quote)
    aggregator.add_callback('bar', on_bar)

    # 启动
    await aggregator.start()

    # 订阅
    for symbol in symbols:
        aggregator.subscribe(symbol, SubscriptionType.ALL)

    # 运行
    await asyncio.sleep(duration_sec)

    # 停止
    await aggregator.stop()

    return {
        'duration_sec': duration_sec,
        'symbols': symbols,
        'ticks_collected': len(collected_ticks),
        'quotes_collected': len(collected_quotes),
        'bars_collected': len(collected_bars),
        'statistics': aggregator.get_statistics(),
    }


__all__ = [
    # Enums
    'DataSource',
    'MarketState',
    'BarInterval',
    'SubscriptionType',
    'DataQuality',
    # Data Classes
    'Tick',
    'Quote',
    'Bar',
    'MarketDepth',
    'DataSourceConfig',
    'Subscription',
    'QualityMetrics',
    'AggregatorConfig',
    # Bar Builder
    'BarBuilder',
    # Data Source
    'DataSourceHandler',
    'SimulatedDataSource',
    # Aggregator
    'MarketDataAggregator',
    # Factory Functions
    'create_aggregator',
    'create_source_config',
    'create_tick',
    'create_quote',
    'create_market_depth',
    'quick_aggregate',
]
