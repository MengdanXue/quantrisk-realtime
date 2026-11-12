"""
订单流分析器 (Order Flow Analyzer)

提供实时订单流分析功能:
- 订单簿分析 (Order Book Analysis)
- 交易流分析 (Trade Flow Analysis)
- 微观结构指标 (Microstructure Indicators)
- 订单流信号 (Order Flow Signals)

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Deque
from collections import defaultdict, deque
import numpy as np
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class OrderSide(str, Enum):
    """订单方向"""
    BID = "bid"
    ASK = "ask"


class TradeType(str, Enum):
    """交易类型"""
    AGGRESSIVE = "aggressive"  # 主动成交
    PASSIVE = "passive"        # 被动成交
    UNKNOWN = "unknown"


class OrderFlowSignal(str, Enum):
    """订单流信号"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class ToxicityLevel(str, Enum):
    """订单流毒性水平"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class InstitutionalActivity(str, Enum):
    """机构活动类型"""
    BUYING = "buying"
    SELLING = "selling"
    NEUTRAL = "neutral"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class AnalyzerState(str, Enum):
    """分析器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ==================== 数据类 ====================

@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    quantity: float
    num_orders: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'price': self.price,
            'quantity': self.quantity,
            'num_orders': self.num_orders,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class OrderBook:
    """订单簿快照"""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def bid_price(self) -> float:
        """最优买价"""
        return self.bids[0].price if self.bids else 0.0

    @property
    def ask_price(self) -> float:
        """最优卖价"""
        return self.asks[0].price if self.asks else 0.0

    @property
    def spread(self) -> float:
        """买卖价差"""
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @property
    def mid_price(self) -> float:
        """中间价"""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'bids': [b.to_dict() for b in self.bids[:5]],  # Top 5
            'asks': [a.to_dict() for a in self.asks[:5]],  # Top 5
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'spread': self.spread,
            'mid_price': self.mid_price,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    price: float
    quantity: float
    side: OrderSide           # 主动方向
    trade_type: TradeType
    timestamp: datetime = field(default_factory=datetime.now)
    trade_id: Optional[str] = None

    @property
    def notional(self) -> float:
        """名义价值"""
        return self.price * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'quantity': self.quantity,
            'side': self.side.value,
            'trade_type': self.trade_type.value,
            'notional': self.notional,
            'timestamp': self.timestamp.isoformat(),
            'trade_id': self.trade_id,
        }


@dataclass
class OrderBookMetrics:
    """订单簿指标"""
    symbol: str

    # 基础指标
    bid_ask_spread: float
    spread_bps: float           # 价差基点
    mid_price: float

    # 订单簿失衡
    order_book_imbalance: float  # OBI: (bid_vol - ask_vol) / (bid_vol + ask_vol)

    # 深度指标
    bid_depth: float            # 买单总量
    ask_depth: float            # 卖单总量
    total_depth: float
    depth_ratio: float          # bid_depth / ask_depth

    # 加权价格
    volume_weighted_mid: float  # VWMP

    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'bid_ask_spread': round(self.bid_ask_spread, 6),
            'spread_bps': round(self.spread_bps, 2),
            'mid_price': round(self.mid_price, 4),
            'order_book_imbalance': round(self.order_book_imbalance, 4),
            'bid_depth': round(self.bid_depth, 2),
            'ask_depth': round(self.ask_depth, 2),
            'total_depth': round(self.total_depth, 2),
            'depth_ratio': round(self.depth_ratio, 4),
            'volume_weighted_mid': round(self.volume_weighted_mid, 4),
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class TradeFlowMetrics:
    """交易流指标"""
    symbol: str

    # 交易流失衡
    trade_flow_imbalance: float  # (buy_vol - sell_vol) / total_vol

    # 成交量时钟
    volume_clock: float          # 成交量累计
    volume_rate: float           # 成交量速率 (per minute)

    # 交易到达率
    trade_arrival_rate: float    # 交易频率 (trades per second)

    # 主动/被动分类
    aggressive_buy_ratio: float
    aggressive_sell_ratio: float
    passive_ratio: float

    # 统计
    total_trades: int
    buy_volume: float
    sell_volume: float
    total_volume: float

    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'trade_flow_imbalance': round(self.trade_flow_imbalance, 4),
            'volume_clock': round(self.volume_clock, 2),
            'volume_rate': round(self.volume_rate, 2),
            'trade_arrival_rate': round(self.trade_arrival_rate, 4),
            'aggressive_buy_ratio': round(self.aggressive_buy_ratio, 4),
            'aggressive_sell_ratio': round(self.aggressive_sell_ratio, 4),
            'passive_ratio': round(self.passive_ratio, 4),
            'total_trades': self.total_trades,
            'buy_volume': round(self.buy_volume, 2),
            'sell_volume': round(self.sell_volume, 2),
            'total_volume': round(self.total_volume, 2),
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class MicrostructureIndicators:
    """微观结构指标"""
    symbol: str

    # Kyle's Lambda (价格影响)
    kyles_lambda: float          # 每单位成交量的价格影响

    # Amihud Illiquidity
    amihud_illiquidity: float    # |return| / dollar_volume

    # Roll's Spread Estimator
    roll_spread: float           # 基于价格协方差的价差估计

    # VPIN (Volume-Synchronized PIN)
    vpin: float                  # 订单流毒性指标

    # 额外指标
    price_impact: float          # 价格影响系数
    effective_spread: float      # 有效价差
    realized_spread: float       # 已实现价差

    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'kyles_lambda': round(self.kyles_lambda, 8),
            'amihud_illiquidity': round(self.amihud_illiquidity, 8),
            'roll_spread': round(self.roll_spread, 6),
            'vpin': round(self.vpin, 4),
            'price_impact': round(self.price_impact, 8),
            'effective_spread': round(self.effective_spread, 6),
            'realized_spread': round(self.realized_spread, 6),
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class LargeOrderDetection:
    """大单检测"""
    symbol: str
    trade: Trade
    size_percentile: float       # 成交量分位数
    is_block_trade: bool         # 是否大宗交易
    estimated_impact: float      # 预估市场影响 (bps)

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'trade': self.trade.to_dict(),
            'size_percentile': round(self.size_percentile, 2),
            'is_block_trade': self.is_block_trade,
            'estimated_impact': round(self.estimated_impact, 2),
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class IcebergDetection:
    """冰山订单检测"""
    symbol: str
    price: float
    side: OrderSide
    detected_volume: float       # 已检测到的隐藏量
    refresh_count: int           # 订单刷新次数
    confidence: float            # 检测置信度

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'side': self.side.value,
            'detected_volume': round(self.detected_volume, 2),
            'refresh_count': self.refresh_count,
            'confidence': round(self.confidence, 2),
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class ToxicityAnalysis:
    """订单流毒性分析"""
    symbol: str
    toxicity_level: ToxicityLevel
    vpin_score: float
    imbalance_score: float
    adverse_selection_score: float
    overall_score: float         # 综合毒性得分 0-1

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'toxicity_level': self.toxicity_level.value,
            'vpin_score': round(self.vpin_score, 4),
            'imbalance_score': round(self.imbalance_score, 4),
            'adverse_selection_score': round(self.adverse_selection_score, 4),
            'overall_score': round(self.overall_score, 4),
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class InstitutionalSignal:
    """机构活动信号"""
    symbol: str
    activity_type: InstitutionalActivity
    detected_volume: float
    average_price: float
    time_span_minutes: float
    conviction_score: float      # 确信度评分 0-1

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'activity_type': self.activity_type.value,
            'detected_volume': round(self.detected_volume, 2),
            'average_price': round(self.average_price, 4),
            'time_span_minutes': round(self.time_span_minutes, 2),
            'conviction_score': round(self.conviction_score, 4),
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class OrderFlowSignalResult:
    """订单流信号结果"""
    symbol: str
    signal: OrderFlowSignal
    strength: float              # 信号强度 0-1
    confidence: float            # 置信度 0-1

    # 贡献因素
    order_book_contribution: float
    trade_flow_contribution: float
    microstructure_contribution: float

    # 附加信息
    reasoning: str

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'signal': self.signal.value,
            'strength': round(self.strength, 4),
            'confidence': round(self.confidence, 4),
            'order_book_contribution': round(self.order_book_contribution, 4),
            'trade_flow_contribution': round(self.trade_flow_contribution, 4),
            'microstructure_contribution': round(self.microstructure_contribution, 4),
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class AnalyzerConfig:
    """分析器配置"""
    # 时间窗口
    orderbook_window_seconds: float = 60.0
    trade_window_seconds: float = 300.0
    microstructure_window_seconds: float = 600.0

    # 大单检测
    block_trade_percentile: float = 95.0  # 95分位数
    large_trade_threshold: float = 10000.0  # 最小大单金额

    # 冰山检测
    iceberg_refresh_threshold: int = 3
    iceberg_confidence_threshold: float = 0.7

    # VPIN参数
    vpin_buckets: int = 50
    vpin_window: int = 20

    # 信号阈值
    signal_strength_threshold: float = 0.5
    signal_confidence_threshold: float = 0.6

    # 毒性阈值
    toxicity_high_threshold: float = 0.7
    toxicity_medium_threshold: float = 0.4

    # 更新频率
    update_interval_seconds: float = 1.0


# ==================== 抽象基类 ====================

class OrderBookProvider(ABC):
    """订单簿数据提供者"""

    @abstractmethod
    async def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """获取订单簿"""
        pass

    @abstractmethod
    async def subscribe_order_book(
        self,
        symbol: str,
        callback: Callable[[OrderBook], None]
    ) -> None:
        """订阅订单簿更新"""
        pass


class TradeDataProvider(ABC):
    """交易数据提供者"""

    @abstractmethod
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Trade]:
        """获取最近交易"""
        pass

    @abstractmethod
    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Trade], None]
    ) -> None:
        """订阅交易流"""
        pass


# ==================== 模拟实现 ====================

class SimulatedOrderBookProvider(OrderBookProvider):
    """模拟订单簿提供者"""

    def __init__(self):
        self._order_books: Dict[str, OrderBook] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    async def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """获取订单簿"""
        if symbol not in self._order_books:
            # 生成模拟订单簿
            mid_price = 100.0
            self._order_books[symbol] = self._generate_order_book(symbol, mid_price)

        return self._order_books[symbol]

    async def subscribe_order_book(
        self,
        symbol: str,
        callback: Callable[[OrderBook], None]
    ) -> None:
        """订阅订单簿更新"""
        self._callbacks[symbol].append(callback)

    def _generate_order_book(self, symbol: str, mid_price: float) -> OrderBook:
        """生成模拟订单簿"""
        spread = mid_price * 0.0001  # 1 bps spread

        bids = [
            OrderBookLevel(mid_price - spread/2 - i*spread, 100 + i*10, 1 + i)
            for i in range(10)
        ]

        asks = [
            OrderBookLevel(mid_price + spread/2 + i*spread, 100 + i*10, 1 + i)
            for i in range(10)
        ]

        return OrderBook(symbol=symbol, bids=bids, asks=asks)

    def update_order_book(self, order_book: OrderBook) -> None:
        """更新订单簿并触发回调"""
        self._order_books[order_book.symbol] = order_book

        for callback in self._callbacks.get(order_book.symbol, []):
            try:
                callback(order_book)
            except Exception as e:
                logger.error(f"Error in order book callback: {e}")


class SimulatedTradeDataProvider(TradeDataProvider):
    """模拟交易数据提供者"""

    def __init__(self):
        self._trades: Dict[str, Deque[Trade]] = defaultdict(lambda: deque(maxlen=1000))
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Trade]:
        """获取最近交易"""
        trades = list(self._trades[symbol])
        return trades[-limit:] if len(trades) > limit else trades

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Trade], None]
    ) -> None:
        """订阅交易流"""
        self._callbacks[symbol].append(callback)

    def add_trade(self, trade: Trade) -> None:
        """添加交易并触发回调"""
        self._trades[trade.symbol].append(trade)

        for callback in self._callbacks.get(trade.symbol, []):
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")


# ==================== 核心分析器 ====================

class OrderFlowAnalyzer:
    """订单流分析器"""

    def __init__(
        self,
        order_book_provider: OrderBookProvider,
        trade_provider: TradeDataProvider,
        config: Optional[AnalyzerConfig] = None
    ):
        self.order_book_provider = order_book_provider
        self.trade_provider = trade_provider
        self.config = config or AnalyzerConfig()

        # 状态
        self._state = AnalyzerState.IDLE

        # 数据缓存
        self._order_books: Dict[str, OrderBook] = {}
        self._trades: Dict[str, Deque[Trade]] = defaultdict(lambda: deque(maxlen=1000))
        self._price_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))

        # 指标缓存
        self._orderbook_metrics: Dict[str, OrderBookMetrics] = {}
        self._tradeflow_metrics: Dict[str, TradeFlowMetrics] = {}
        self._microstructure: Dict[str, MicrostructureIndicators] = {}

        # 检测结果
        self._large_orders: Dict[str, List[LargeOrderDetection]] = defaultdict(list)
        self._iceberg_orders: Dict[str, List[IcebergDetection]] = defaultdict(list)

        # 回调
        self._signal_callbacks: List[Callable[[OrderFlowSignalResult], None]] = []
        self._large_order_callbacks: List[Callable[[LargeOrderDetection], None]] = []
        self._toxicity_callbacks: List[Callable[[ToxicityAnalysis], None]] = []

        # 任务
        self._tasks: List[asyncio.Task] = []

    async def start(self, symbols: List[str]) -> None:
        """启动分析器"""
        if self._state == AnalyzerState.RUNNING:
            logger.warning("Analyzer already running")
            return

        self._state = AnalyzerState.RUNNING

        # 订阅数据
        for symbol in symbols:
            await self.order_book_provider.subscribe_order_book(
                symbol, self._on_order_book_update
            )
            await self.trade_provider.subscribe_trades(
                symbol, self._on_trade_update
            )

        # 启动分析任务
        for symbol in symbols:
            task = asyncio.create_task(self._analyze_loop(symbol))
            self._tasks.append(task)

        logger.info(f"Order Flow Analyzer started for {len(symbols)} symbols")

    async def stop(self) -> None:
        """停止分析器"""
        self._state = AnalyzerState.IDLE

        # 取消所有任务
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("Order Flow Analyzer stopped")

    def register_signal_callback(
        self,
        callback: Callable[[OrderFlowSignalResult], None]
    ) -> None:
        """注册信号回调"""
        self._signal_callbacks.append(callback)

    def register_large_order_callback(
        self,
        callback: Callable[[LargeOrderDetection], None]
    ) -> None:
        """注册大单检测回调"""
        self._large_order_callbacks.append(callback)

    def register_toxicity_callback(
        self,
        callback: Callable[[ToxicityAnalysis], None]
    ) -> None:
        """注册毒性检测回调"""
        self._toxicity_callbacks.append(callback)

    # ==================== 数据更新处理 ====================

    def _on_order_book_update(self, order_book: OrderBook) -> None:
        """订单簿更新处理"""
        self._order_books[order_book.symbol] = order_book

    def _on_trade_update(self, trade: Trade) -> None:
        """交易更新处理"""
        self._trades[trade.symbol].append(trade)
        self._price_history[trade.symbol].append(trade.price)

        # 检测大单
        self._detect_large_order(trade)

    # ==================== 分析循环 ====================

    async def _analyze_loop(self, symbol: str) -> None:
        """分析循环"""
        while self._state == AnalyzerState.RUNNING:
            try:
                # 计算订单簿指标
                orderbook_metrics = await self.calculate_orderbook_metrics(symbol)
                if orderbook_metrics:
                    self._orderbook_metrics[symbol] = orderbook_metrics

                # 计算交易流指标
                tradeflow_metrics = await self.calculate_tradeflow_metrics(symbol)
                if tradeflow_metrics:
                    self._tradeflow_metrics[symbol] = tradeflow_metrics

                # 计算微观结构指标
                microstructure = await self.calculate_microstructure_indicators(symbol)
                if microstructure:
                    self._microstructure[symbol] = microstructure

                # 检测冰山订单
                await self._detect_iceberg_orders(symbol)

                # 分析毒性
                toxicity = await self.analyze_toxicity(symbol)
                if toxicity:
                    self._emit_toxicity_signal(toxicity)

                # 生成订单流信号
                signal = await self.generate_signal(symbol)
                if signal:
                    self._emit_signal(signal)

                # 等待下一个周期
                await asyncio.sleep(self.config.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analyze loop for {symbol}: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    # ==================== 订单簿分析 ====================

    async def calculate_orderbook_metrics(
        self,
        symbol: str
    ) -> Optional[OrderBookMetrics]:
        """计算订单簿指标"""
        order_book = self._order_books.get(symbol)
        if not order_book or not order_book.bids or not order_book.asks:
            return None

        # 基础指标
        spread = order_book.spread
        mid_price = order_book.mid_price
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0

        # 订单簿深度
        bid_depth = sum(level.quantity for level in order_book.bids[:10])
        ask_depth = sum(level.quantity for level in order_book.asks[:10])
        total_depth = bid_depth + ask_depth
        depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 1.0

        # 订单簿失衡 (OBI)
        obi = (bid_depth - ask_depth) / (bid_depth + ask_depth) if total_depth > 0 else 0

        # 成交量加权中间价 (VWMP)
        vwmp = self._calculate_vwmp(order_book)

        return OrderBookMetrics(
            symbol=symbol,
            bid_ask_spread=spread,
            spread_bps=spread_bps,
            mid_price=mid_price,
            order_book_imbalance=obi,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_depth=total_depth,
            depth_ratio=depth_ratio,
            volume_weighted_mid=vwmp,
        )

    def _calculate_vwmp(self, order_book: OrderBook) -> float:
        """计算成交量加权中间价"""
        bid_value = sum(l.price * l.quantity for l in order_book.bids[:5])
        ask_value = sum(l.price * l.quantity for l in order_book.asks[:5])

        bid_volume = sum(l.quantity for l in order_book.bids[:5])
        ask_volume = sum(l.quantity for l in order_book.asks[:5])

        total_volume = bid_volume + ask_volume

        if total_volume > 0:
            return (bid_value + ask_value) / total_volume

        return order_book.mid_price

    # ==================== 交易流分析 ====================

    async def calculate_tradeflow_metrics(
        self,
        symbol: str
    ) -> Optional[TradeFlowMetrics]:
        """计算交易流指标"""
        trades = list(self._trades.get(symbol, []))
        if not trades:
            return None

        # 过滤时间窗口内的交易
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.trade_window_seconds)
        recent_trades = [t for t in trades if t.timestamp >= window_start]

        if not recent_trades:
            return None

        # 统计
        total_trades = len(recent_trades)
        buy_volume = sum(t.quantity for t in recent_trades if t.side == OrderSide.BID)
        sell_volume = sum(t.quantity for t in recent_trades if t.side == OrderSide.ASK)
        total_volume = buy_volume + sell_volume

        # 交易流失衡
        tfi = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0

        # 成交量时钟和速率
        time_span = (now - recent_trades[0].timestamp).total_seconds()
        volume_rate = (total_volume / time_span * 60) if time_span > 0 else 0

        # 交易到达率
        trade_arrival_rate = total_trades / time_span if time_span > 0 else 0

        # 主动/被动分类
        aggressive_buy = sum(
            1 for t in recent_trades
            if t.trade_type == TradeType.AGGRESSIVE and t.side == OrderSide.BID
        )
        aggressive_sell = sum(
            1 for t in recent_trades
            if t.trade_type == TradeType.AGGRESSIVE and t.side == OrderSide.ASK
        )
        passive = total_trades - aggressive_buy - aggressive_sell

        aggressive_buy_ratio = aggressive_buy / total_trades if total_trades > 0 else 0
        aggressive_sell_ratio = aggressive_sell / total_trades if total_trades > 0 else 0
        passive_ratio = passive / total_trades if total_trades > 0 else 0

        return TradeFlowMetrics(
            symbol=symbol,
            trade_flow_imbalance=tfi,
            volume_clock=total_volume,
            volume_rate=volume_rate,
            trade_arrival_rate=trade_arrival_rate,
            aggressive_buy_ratio=aggressive_buy_ratio,
            aggressive_sell_ratio=aggressive_sell_ratio,
            passive_ratio=passive_ratio,
            total_trades=total_trades,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            total_volume=total_volume,
        )

    # ==================== 微观结构指标 ====================

    async def calculate_microstructure_indicators(
        self,
        symbol: str
    ) -> Optional[MicrostructureIndicators]:
        """计算微观结构指标"""
        trades = list(self._trades.get(symbol, []))
        if len(trades) < 10:
            return None

        # Kyle's Lambda
        kyles_lambda = self._calculate_kyles_lambda(trades)

        # Amihud Illiquidity
        amihud = self._calculate_amihud_illiquidity(trades)

        # Roll's Spread
        roll_spread = self._calculate_roll_spread(trades)

        # VPIN
        vpin = self._calculate_vpin(trades)

        # 价格影响
        price_impact = self._calculate_price_impact(trades)

        # 有效价差和已实现价差
        effective_spread, realized_spread = self._calculate_spreads(symbol, trades)

        return MicrostructureIndicators(
            symbol=symbol,
            kyles_lambda=kyles_lambda,
            amihud_illiquidity=amihud,
            roll_spread=roll_spread,
            vpin=vpin,
            price_impact=price_impact,
            effective_spread=effective_spread,
            realized_spread=realized_spread,
        )

    def _calculate_kyles_lambda(self, trades: List[Trade]) -> float:
        """
        计算Kyle's Lambda (价格影响系数)
        Lambda = Cov(ΔP, Q) / Var(Q)
        """
        if len(trades) < 20:
            return 0.0

        recent_trades = trades[-50:]

        # 价格变化
        price_changes = [
            recent_trades[i].price - recent_trades[i-1].price
            for i in range(1, len(recent_trades))
        ]

        # 成交量 (带方向)
        volumes = [
            t.quantity if t.side == OrderSide.BID else -t.quantity
            for t in recent_trades[1:]
        ]

        if len(price_changes) != len(volumes):
            return 0.0

        # 计算协方差和方差
        cov = np.cov(price_changes, volumes)[0, 1] if len(price_changes) > 1 else 0
        var = np.var(volumes) if len(volumes) > 1 else 1

        return cov / var if var > 0 else 0.0

    def _calculate_amihud_illiquidity(self, trades: List[Trade]) -> float:
        """
        计算Amihud非流动性指标
        Amihud = Average(|return| / dollar_volume)
        """
        if len(trades) < 10:
            return 0.0

        recent_trades = trades[-50:]

        illiquidity_measures = []
        for i in range(1, len(recent_trades)):
            price_return = abs(
                (recent_trades[i].price - recent_trades[i-1].price) / recent_trades[i-1].price
            )
            dollar_volume = recent_trades[i].notional

            if dollar_volume > 0:
                illiquidity_measures.append(price_return / dollar_volume)

        return np.mean(illiquidity_measures) if illiquidity_measures else 0.0

    def _calculate_roll_spread(self, trades: List[Trade]) -> float:
        """
        计算Roll's Spread Estimator
        Roll = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
        """
        if len(trades) < 20:
            return 0.0

        prices = [t.price for t in trades[-100:]]

        # 计算价格变化
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

        if len(price_changes) < 2:
            return 0.0

        # 计算自协方差
        auto_cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]

        # Roll's spread
        if auto_cov < 0:
            return 2 * math.sqrt(-auto_cov)

        return 0.0

    def _calculate_vpin(self, trades: List[Trade]) -> float:
        """
        计算VPIN (Volume-Synchronized Probability of Informed Trading)
        """
        if len(trades) < self.config.vpin_buckets:
            return 0.0

        recent_trades = trades[-self.config.vpin_buckets * 2:]

        # 计算总成交量
        total_volume = sum(t.quantity for t in recent_trades)
        bucket_size = total_volume / self.config.vpin_buckets

        # 分桶
        buckets = []
        current_bucket_buy = 0
        current_bucket_sell = 0
        current_volume = 0

        for trade in recent_trades:
            if trade.side == OrderSide.BID:
                current_bucket_buy += trade.quantity
            else:
                current_bucket_sell += trade.quantity

            current_volume += trade.quantity

            if current_volume >= bucket_size:
                buckets.append(abs(current_bucket_buy - current_bucket_sell))
                current_bucket_buy = 0
                current_bucket_sell = 0
                current_volume = 0

        # 计算VPIN
        if len(buckets) >= self.config.vpin_window:
            recent_buckets = buckets[-self.config.vpin_window:]
            vpin = sum(recent_buckets) / (len(recent_buckets) * bucket_size)
            return min(vpin, 1.0)

        return 0.0

    def _calculate_price_impact(self, trades: List[Trade]) -> float:
        """计算价格影响系数"""
        if len(trades) < 10:
            return 0.0

        recent_trades = trades[-30:]

        impacts = []
        for i in range(1, len(recent_trades)):
            price_change = recent_trades[i].price - recent_trades[i-1].price
            volume = recent_trades[i].quantity

            if volume > 0 and recent_trades[i-1].price > 0:
                impact = (price_change / recent_trades[i-1].price) / volume
                impacts.append(abs(impact))

        return np.mean(impacts) if impacts else 0.0

    def _calculate_spreads(
        self,
        symbol: str,
        trades: List[Trade]
    ) -> tuple[float, float]:
        """计算有效价差和已实现价差"""
        order_book = self._order_books.get(symbol)
        if not order_book or not trades:
            return 0.0, 0.0

        mid_price = order_book.mid_price
        if mid_price <= 0:
            return 0.0, 0.0

        # 有效价差 = 2 * |Trade Price - Mid Price|
        recent_trades = trades[-10:]
        effective_spreads = [
            2 * abs(t.price - mid_price) / mid_price
            for t in recent_trades
        ]
        effective_spread = np.mean(effective_spreads) if effective_spreads else 0.0

        # 已实现价差 (简化版)
        realized_spread = effective_spread * 0.7  # 通常约为有效价差的70%

        return effective_spread, realized_spread

    # ==================== 订单流信号检测 ====================

    def _detect_large_order(self, trade: Trade) -> None:
        """检测大单"""
        symbol = trade.symbol

        # 计算成交量分位数
        all_trades = list(self._trades[symbol])
        if len(all_trades) < 20:
            return

        volumes = [t.quantity for t in all_trades[-100:]]
        percentile = np.percentile(volumes, self.config.block_trade_percentile)

        is_large = (
            trade.quantity > percentile and
            trade.notional > self.config.large_trade_threshold
        )

        if is_large:
            # 估算市场影响
            order_book = self._order_books.get(symbol)
            if order_book:
                depth = sum(l.quantity for l in order_book.bids[:5])
                impact_bps = (trade.quantity / depth * 10) if depth > 0 else 5
            else:
                impact_bps = 5

            detection = LargeOrderDetection(
                symbol=symbol,
                trade=trade,
                size_percentile=self.config.block_trade_percentile,
                is_block_trade=True,
                estimated_impact=impact_bps,
            )

            self._large_orders[symbol].append(detection)
            self._emit_large_order_detection(detection)

    async def _detect_iceberg_orders(self, symbol: str) -> None:
        """检测冰山订单"""
        order_book = self._order_books.get(symbol)
        if not order_book:
            return

        # 检测买单侧
        self._detect_iceberg_side(order_book, OrderSide.BID)

        # 检测卖单侧
        self._detect_iceberg_side(order_book, OrderSide.ASK)

    def _detect_iceberg_side(self, order_book: OrderBook, side: OrderSide) -> None:
        """检测单侧冰山订单"""
        # 简化实现: 检测订单量异常稳定的价格水平
        levels = order_book.bids if side == OrderSide.BID else order_book.asks

        for level in levels[:3]:  # 只检查前3档
            # 检测逻辑: 如果某价格水平订单量始终维持在相近水平，可能是冰山单
            # 这里使用简化版本，实际需要维护历史订单簿快照

            if level.quantity > 0:
                # 模拟检测结果
                confidence = 0.6  # 简化的置信度

                if confidence >= self.config.iceberg_confidence_threshold:
                    detection = IcebergDetection(
                        symbol=order_book.symbol,
                        price=level.price,
                        side=side,
                        detected_volume=level.quantity * 3,  # 估算隐藏量
                        refresh_count=3,
                        confidence=confidence,
                    )

                    # 避免重复
                    existing = [
                        d for d in self._iceberg_orders[order_book.symbol]
                        if d.price == level.price and d.side == side
                    ]

                    if not existing:
                        self._iceberg_orders[order_book.symbol].append(detection)

    async def analyze_toxicity(
        self,
        symbol: str
    ) -> Optional[ToxicityAnalysis]:
        """分析订单流毒性"""
        microstructure = self._microstructure.get(symbol)
        orderbook_metrics = self._orderbook_metrics.get(symbol)
        tradeflow_metrics = self._tradeflow_metrics.get(symbol)

        if not all([microstructure, orderbook_metrics, tradeflow_metrics]):
            return None

        # VPIN分数
        vpin_score = microstructure.vpin

        # 失衡分数
        imbalance_score = abs(orderbook_metrics.order_book_imbalance)

        # 逆向选择分数 (基于价格影响)
        adverse_selection_score = min(microstructure.price_impact * 10000, 1.0)

        # 综合毒性得分
        overall_score = (
            vpin_score * 0.4 +
            imbalance_score * 0.3 +
            adverse_selection_score * 0.3
        )

        # 确定毒性水平
        if overall_score >= self.config.toxicity_high_threshold:
            level = ToxicityLevel.HIGH
        elif overall_score >= self.config.toxicity_medium_threshold:
            level = ToxicityLevel.MEDIUM
        else:
            level = ToxicityLevel.LOW

        return ToxicityAnalysis(
            symbol=symbol,
            toxicity_level=level,
            vpin_score=vpin_score,
            imbalance_score=imbalance_score,
            adverse_selection_score=adverse_selection_score,
            overall_score=overall_score,
        )

    async def detect_institutional_activity(
        self,
        symbol: str
    ) -> Optional[InstitutionalSignal]:
        """检测机构活动"""
        # 分析大单模式
        large_orders = self._large_orders.get(symbol, [])
        if len(large_orders) < 3:
            return None

        # 最近的大单
        recent_large_orders = large_orders[-10:]

        # 统计
        buy_volume = sum(
            lo.trade.quantity for lo in recent_large_orders
            if lo.trade.side == OrderSide.BID
        )
        sell_volume = sum(
            lo.trade.quantity for lo in recent_large_orders
            if lo.trade.side == OrderSide.ASK
        )

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return None

        # 平均价格
        avg_price = sum(
            lo.trade.price * lo.trade.quantity for lo in recent_large_orders
        ) / total_volume

        # 时间跨度
        time_span = (
            recent_large_orders[-1].timestamp - recent_large_orders[0].timestamp
        ).total_seconds() / 60

        # 确定活动类型
        imbalance = (buy_volume - sell_volume) / total_volume

        if imbalance > 0.6:
            activity = InstitutionalActivity.BUYING
        elif imbalance < -0.6:
            activity = InstitutionalActivity.SELLING
        elif imbalance > 0.3:
            activity = InstitutionalActivity.ACCUMULATION
        elif imbalance < -0.3:
            activity = InstitutionalActivity.DISTRIBUTION
        else:
            activity = InstitutionalActivity.NEUTRAL

        # 确信度
        conviction = abs(imbalance)

        return InstitutionalSignal(
            symbol=symbol,
            activity_type=activity,
            detected_volume=total_volume,
            average_price=avg_price,
            time_span_minutes=time_span,
            conviction_score=conviction,
        )

    async def generate_signal(
        self,
        symbol: str
    ) -> Optional[OrderFlowSignalResult]:
        """生成订单流信号"""
        orderbook_metrics = self._orderbook_metrics.get(symbol)
        tradeflow_metrics = self._tradeflow_metrics.get(symbol)
        microstructure = self._microstructure.get(symbol)

        if not all([orderbook_metrics, tradeflow_metrics, microstructure]):
            return None

        # 订单簿贡献
        ob_contribution = orderbook_metrics.order_book_imbalance

        # 交易流贡献
        tf_contribution = tradeflow_metrics.trade_flow_imbalance

        # 微观结构贡献 (基于VPIN的反向)
        ms_contribution = -microstructure.vpin if microstructure.vpin > 0.5 else 0

        # 综合信号强度
        raw_signal = (
            ob_contribution * 0.4 +
            tf_contribution * 0.4 +
            ms_contribution * 0.2
        )

        # 确定信号类型和强度
        if raw_signal > 0.5:
            signal = OrderFlowSignal.STRONG_BUY
            strength = min(raw_signal, 1.0)
        elif raw_signal > 0.2:
            signal = OrderFlowSignal.BUY
            strength = raw_signal
        elif raw_signal < -0.5:
            signal = OrderFlowSignal.STRONG_SELL
            strength = min(abs(raw_signal), 1.0)
        elif raw_signal < -0.2:
            signal = OrderFlowSignal.SELL
            strength = abs(raw_signal)
        else:
            signal = OrderFlowSignal.NEUTRAL
            strength = 0.5

        # 置信度 (基于数据质量)
        confidence = min(
            1.0,
            (tradeflow_metrics.total_trades / 100) * 0.5 +
            (1 - microstructure.vpin) * 0.5
        )

        # 推理说明
        reasoning = self._generate_reasoning(
            signal, orderbook_metrics, tradeflow_metrics, microstructure
        )

        result = OrderFlowSignalResult(
            symbol=symbol,
            signal=signal,
            strength=strength,
            confidence=confidence,
            order_book_contribution=ob_contribution,
            trade_flow_contribution=tf_contribution,
            microstructure_contribution=ms_contribution,
            reasoning=reasoning,
        )

        # 只发送满足阈值的信号
        if (strength >= self.config.signal_strength_threshold and
            confidence >= self.config.signal_confidence_threshold):
            return result

        return None

    def _generate_reasoning(
        self,
        signal: OrderFlowSignal,
        ob_metrics: OrderBookMetrics,
        tf_metrics: TradeFlowMetrics,
        ms_indicators: MicrostructureIndicators
    ) -> str:
        """生成信号推理说明"""
        reasons = []

        # 订单簿因素
        if abs(ob_metrics.order_book_imbalance) > 0.3:
            direction = "买" if ob_metrics.order_book_imbalance > 0 else "卖"
            reasons.append(f"订单簿{direction}单占优 ({ob_metrics.order_book_imbalance:.2%})")

        # 交易流因素
        if abs(tf_metrics.trade_flow_imbalance) > 0.3:
            direction = "买" if tf_metrics.trade_flow_imbalance > 0 else "卖"
            reasons.append(f"交易流{direction}单主导 ({tf_metrics.trade_flow_imbalance:.2%})")

        # 毒性因素
        if ms_indicators.vpin > 0.6:
            reasons.append(f"高订单流毒性 (VPIN={ms_indicators.vpin:.2f})")

        # 深度因素
        if ob_metrics.depth_ratio > 1.5:
            reasons.append(f"买单深度优势 (比率={ob_metrics.depth_ratio:.2f})")
        elif ob_metrics.depth_ratio < 0.67:
            reasons.append(f"卖单深度优势 (比率={ob_metrics.depth_ratio:.2f})")

        return "; ".join(reasons) if reasons else "综合订单流分析"

    # ==================== 数据获取接口 ====================

    def get_orderbook_metrics(self, symbol: str) -> Optional[OrderBookMetrics]:
        """获取订单簿指标"""
        return self._orderbook_metrics.get(symbol)

    def get_tradeflow_metrics(self, symbol: str) -> Optional[TradeFlowMetrics]:
        """获取交易流指标"""
        return self._tradeflow_metrics.get(symbol)

    def get_microstructure_indicators(
        self,
        symbol: str
    ) -> Optional[MicrostructureIndicators]:
        """获取微观结构指标"""
        return self._microstructure.get(symbol)

    def get_large_orders(self, symbol: str) -> List[LargeOrderDetection]:
        """获取大单检测结果"""
        return self._large_orders.get(symbol, [])

    def get_iceberg_orders(self, symbol: str) -> List[IcebergDetection]:
        """获取冰山订单检测结果"""
        return self._iceberg_orders.get(symbol, [])

    def get_state(self) -> AnalyzerState:
        """获取分析器状态"""
        return self._state

    # ==================== 回调触发 ====================

    def _emit_signal(self, signal: OrderFlowSignalResult) -> None:
        """触发信号回调"""
        for callback in self._signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

    def _emit_large_order_detection(self, detection: LargeOrderDetection) -> None:
        """触发大单检测回调"""
        for callback in self._large_order_callbacks:
            try:
                callback(detection)
            except Exception as e:
                logger.error(f"Error in large order callback: {e}")

    def _emit_toxicity_signal(self, toxicity: ToxicityAnalysis) -> None:
        """触发毒性检测回调"""
        for callback in self._toxicity_callbacks:
            try:
                callback(toxicity)
            except Exception as e:
                logger.error(f"Error in toxicity callback: {e}")


# ==================== 工厂函数 ====================

def create_order_flow_analyzer(
    order_book_provider: Optional[OrderBookProvider] = None,
    trade_provider: Optional[TradeDataProvider] = None,
    **config_kwargs
) -> OrderFlowAnalyzer:
    """
    创建订单流分析器

    Args:
        order_book_provider: 订单簿数据提供者
        trade_provider: 交易数据提供者
        **config_kwargs: 配置参数

    Returns:
        OrderFlowAnalyzer实例
    """
    # 使用模拟提供者（如果未提供）
    if order_book_provider is None:
        order_book_provider = simulated_default(SimulatedOrderBookProvider, "data from create_order_flow_analyzer's simulated order_book_provider (SimulatedOrderBookProvider)")

    if trade_provider is None:
        trade_provider = simulated_default(SimulatedTradeDataProvider, "data from create_order_flow_analyzer's simulated trade_provider (SimulatedTradeDataProvider)")

    # 创建配置
    config = AnalyzerConfig(**config_kwargs) if config_kwargs else AnalyzerConfig()

    return OrderFlowAnalyzer(
        order_book_provider=order_book_provider,
        trade_provider=trade_provider,
        config=config,
    )


def create_order_book(
    symbol: str,
    bids: List[tuple[float, float]],
    asks: List[tuple[float, float]]
) -> OrderBook:
    """
    创建订单簿

    Args:
        symbol: 交易品种
        bids: 买单列表 [(price, quantity), ...]
        asks: 卖单列表 [(price, quantity), ...]

    Returns:
        OrderBook实例
    """
    bid_levels = [
        OrderBookLevel(price=price, quantity=qty)
        for price, qty in bids
    ]

    ask_levels = [
        OrderBookLevel(price=price, quantity=qty)
        for price, qty in asks
    ]

    return OrderBook(symbol=symbol, bids=bid_levels, asks=ask_levels)


def create_trade(
    symbol: str,
    price: float,
    quantity: float,
    side: str,
    trade_type: str = "unknown"
) -> Trade:
    """
    创建交易记录

    Args:
        symbol: 交易品种
        price: 价格
        quantity: 数量
        side: 方向 ('bid' 或 'ask')
        trade_type: 交易类型

    Returns:
        Trade实例
    """
    return Trade(
        symbol=symbol,
        price=price,
        quantity=quantity,
        side=OrderSide(side),
        trade_type=TradeType(trade_type),
    )


async def quick_analyze(
    symbol: str,
    order_book: OrderBook,
    trades: List[Trade]
) -> Dict[str, Any]:
    """
    快速订单流分析

    Args:
        symbol: 交易品种
        order_book: 订单簿
        trades: 交易列表

    Returns:
        分析结果字典
    """
    # 创建分析器
    analyzer = create_order_flow_analyzer()

    # 注入数据
    analyzer._order_books[symbol] = order_book
    analyzer._trades[symbol].extend(trades)

    # 计算指标
    ob_metrics = await analyzer.calculate_orderbook_metrics(symbol)
    tf_metrics = await analyzer.calculate_tradeflow_metrics(symbol)
    ms_indicators = await analyzer.calculate_microstructure_indicators(symbol)
    signal = await analyzer.generate_signal(symbol)

    return {
        'orderbook_metrics': ob_metrics.to_dict() if ob_metrics else None,
        'tradeflow_metrics': tf_metrics.to_dict() if tf_metrics else None,
        'microstructure_indicators': ms_indicators.to_dict() if ms_indicators else None,
        'signal': signal.to_dict() if signal else None,
    }
