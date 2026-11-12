"""
订单簿失衡分析器 (Order Book Imbalance Analyzer)

提供实时订单簿失衡分析功能:
- 订单簿快照分析 (Order Book Snapshot Analysis)
- 买卖失衡度计算 (Volume Imbalance)
- 加权失衡指标 (Weighted Imbalance by Price Levels)
- 订单流失衡 (Order Flow Imbalance)
- 深度失衡分析 (Depth Imbalance Analysis)
- 失衡动量指标 (Imbalance Momentum)
- 价格冲击预测 (Price Impact Prediction)
- 微观结构信号生成 (Microstructure Signal Generation)
- 多档位失衡热力图 (Multi-Level Imbalance Heatmap)
- 失衡历史统计 (Imbalance Historical Statistics)

Copyright (c) 2025 TraderX. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Deque
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import logging
import math
import statistics
import uuid

logger = logging.getLogger(__name__)


# ============== Enums ==============

class ImbalanceDirection(str, Enum):
    """失衡方向"""
    BUY_HEAVY = "buy_heavy"           # 买盘占优
    SELL_HEAVY = "sell_heavy"         # 卖盘占优
    BALANCED = "balanced"             # 均衡
    NEUTRAL = "neutral"               # 中性


class ImbalanceSeverity(str, Enum):
    """失衡严重程度"""
    EXTREME = "extreme"               # 极端 (>80%)
    HIGH = "high"                     # 高 (60-80%)
    MODERATE = "moderate"             # 中等 (40-60%)
    LOW = "low"                       # 低 (20-40%)
    MINIMAL = "minimal"               # 轻微 (<20%)


class SignalType(str, Enum):
    """信号类型"""
    STRONG_BUY = "strong_buy"         # 强买入
    BUY = "buy"                       # 买入
    WEAK_BUY = "weak_buy"             # 弱买入
    NEUTRAL = "neutral"               # 中性
    WEAK_SELL = "weak_sell"           # 弱卖出
    SELL = "sell"                     # 卖出
    STRONG_SELL = "strong_sell"       # 强卖出


class SignalStrength(str, Enum):
    """信号强度"""
    VERY_STRONG = "very_strong"       # 非常强
    STRONG = "strong"                 # 强
    MODERATE = "moderate"             # 中等
    WEAK = "weak"                     # 弱
    VERY_WEAK = "very_weak"           # 非常弱


class AnalyzerState(str, Enum):
    """分析器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class ImbalanceTrend(str, Enum):
    """失衡趋势"""
    INCREASING = "increasing"         # 失衡加剧
    DECREASING = "decreasing"         # 失衡减弱
    STABLE = "stable"                 # 稳定
    REVERSING = "reversing"           # 反转


# ============== Data Classes ==============

@dataclass
class PriceLevel:
    """价格档位"""
    price: float
    quantity: float
    order_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'price': self.price,
            'quantity': self.quantity,
            'order_count': self.order_count,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class OrderBookSnapshot:
    """
    订单簿快照

    记录特定时刻的完整订单簿状态
    """
    symbol: str
    timestamp: datetime
    bids: List[PriceLevel]              # 买单档位 (从高到低排序)
    asks: List[PriceLevel]              # 卖单档位 (从低到高排序)

    # 汇总信息
    total_bid_volume: float = 0.0       # 总买单量
    total_ask_volume: float = 0.0       # 总卖单量
    total_bid_orders: int = 0           # 总买单数
    total_ask_orders: int = 0           # 总卖单数

    # 最优价格
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    mid_price: Optional[float] = None
    spread: Optional[float] = None
    spread_pct: Optional[float] = None

    def __post_init__(self):
        """初始化后计算汇总信息"""
        if self.bids:
            self.total_bid_volume = sum(level.quantity for level in self.bids)
            self.total_bid_orders = sum(level.order_count for level in self.bids)
            self.best_bid = self.bids[0].price

        if self.asks:
            self.total_ask_volume = sum(level.quantity for level in self.asks)
            self.total_ask_orders = sum(level.order_count for level in self.asks)
            self.best_ask = self.asks[0].price

        if self.best_bid and self.best_ask:
            self.mid_price = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid
            self.spread_pct = (self.spread / self.mid_price * 100) if self.mid_price > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bids': [level.to_dict() for level in self.bids[:10]],  # 只返回前10档
            'asks': [level.to_dict() for level in self.asks[:10]],
            'total_bid_volume': self.total_bid_volume,
            'total_ask_volume': self.total_ask_volume,
            'total_bid_orders': self.total_bid_orders,
            'total_ask_orders': self.total_ask_orders,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
        }


@dataclass
class ImbalanceMetrics:
    """
    失衡指标

    各种失衡度量指标的集合
    """
    timestamp: datetime
    symbol: str

    # 基础失衡指标
    volume_imbalance: float             # 成交量失衡 (-1 to 1)
    order_imbalance: float              # 订单数失衡 (-1 to 1)

    # 加权失衡指标
    weighted_imbalance: float           # 价格加权失衡
    depth_weighted_imbalance: float     # 深度加权失衡

    # 订单流失衡
    order_flow_imbalance: float         # 订单流失衡
    aggressive_flow_imbalance: float    # 激进订单流失衡

    # 失衡方向和程度
    direction: ImbalanceDirection
    severity: ImbalanceSeverity

    # 动量指标
    imbalance_momentum: float = 0.0     # 失衡动量
    imbalance_acceleration: float = 0.0 # 失衡加速度

    # 趋势
    trend: ImbalanceTrend = ImbalanceTrend.STABLE

    # 统计信息
    sample_levels: int = 0              # 采样档位数
    confidence: float = 0.0             # 置信度

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'volume_imbalance': self.volume_imbalance,
            'order_imbalance': self.order_imbalance,
            'weighted_imbalance': self.weighted_imbalance,
            'depth_weighted_imbalance': self.depth_weighted_imbalance,
            'order_flow_imbalance': self.order_flow_imbalance,
            'aggressive_flow_imbalance': self.aggressive_flow_imbalance,
            'direction': self.direction.value,
            'severity': self.severity.value,
            'imbalance_momentum': self.imbalance_momentum,
            'imbalance_acceleration': self.imbalance_acceleration,
            'trend': self.trend.value,
            'sample_levels': self.sample_levels,
            'confidence': self.confidence,
        }


@dataclass
class DepthAnalysis:
    """
    深度失衡分析

    订单簿深度的详细分析
    """
    timestamp: datetime
    symbol: str

    # 各档位深度
    bid_depth_by_level: Dict[int, float] = field(default_factory=dict)  # 档位 -> 买单量
    ask_depth_by_level: Dict[int, float] = field(default_factory=dict)  # 档位 -> 卖单量

    # 累计深度
    cumulative_bid_depth: List[float] = field(default_factory=list)
    cumulative_ask_depth: List[float] = field(default_factory=list)

    # 深度比率
    depth_ratio_by_level: Dict[int, float] = field(default_factory=dict)  # 各档位买卖比

    # 深度不对称性
    depth_asymmetry: float = 0.0        # 深度不对称系数

    # 深度集中度
    bid_concentration: float = 0.0      # 买单集中度 (前N档占比)
    ask_concentration: float = 0.0      # 卖单集中度

    # 深度强度
    total_depth: float = 0.0            # 总深度
    effective_depth: float = 0.0        # 有效深度 (考虑价格距离)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'bid_depth_by_level': {str(k): v for k, v in self.bid_depth_by_level.items()},
            'ask_depth_by_level': {str(k): v for k, v in self.ask_depth_by_level.items()},
            'cumulative_bid_depth': self.cumulative_bid_depth,
            'cumulative_ask_depth': self.cumulative_ask_depth,
            'depth_ratio_by_level': {str(k): v for k, v in self.depth_ratio_by_level.items()},
            'depth_asymmetry': self.depth_asymmetry,
            'bid_concentration': self.bid_concentration,
            'ask_concentration': self.ask_concentration,
            'total_depth': self.total_depth,
            'effective_depth': self.effective_depth,
        }


@dataclass
class ImbalanceSignal:
    """
    失衡信号

    基于订单簿失衡生成的交易信号
    """
    signal_id: str
    timestamp: datetime
    symbol: str

    # 信号类型和强度
    signal_type: SignalType
    signal_strength: SignalStrength

    # 信号置信度
    confidence: float                   # 0-1

    # 支撑信息
    volume_imbalance: float
    weighted_imbalance: float
    order_flow_imbalance: float

    # 价格预测
    predicted_direction: str            # "up" / "down" / "neutral"
    predicted_price_impact: float       # 预期价格冲击 (%)

    # 时间有效性
    valid_until: datetime
    ttl_seconds: float = 60.0           # 信号有效期 (秒)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not hasattr(self, 'valid_until') or self.valid_until is None:
            self.valid_until = self.timestamp + timedelta(seconds=self.ttl_seconds)

    def is_valid(self, current_time: datetime) -> bool:
        """检查信号是否仍然有效"""
        return current_time <= self.valid_until

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'signal_strength': self.signal_strength.value,
            'confidence': self.confidence,
            'volume_imbalance': self.volume_imbalance,
            'weighted_imbalance': self.weighted_imbalance,
            'order_flow_imbalance': self.order_flow_imbalance,
            'predicted_direction': self.predicted_direction,
            'predicted_price_impact': self.predicted_price_impact,
            'valid_until': self.valid_until.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'metadata': self.metadata,
        }


@dataclass
class ImbalanceConfig:
    """失衡分析配置"""
    # 订单簿采样配置
    max_depth_levels: int = 20          # 最大深度档位
    sampling_levels: int = 10           # 采样档位数

    # 失衡计算配置
    imbalance_threshold: float = 0.1    # 失衡阈值 (10%)
    extreme_imbalance_threshold: float = 0.5  # 极端失衡阈值 (50%)

    # 加权配置
    distance_weight_decay: float = 0.5  # 距离加权衰减系数
    volume_weight_power: float = 1.0    # 成交量加权指数

    # 订单流配置
    order_flow_window: int = 100        # 订单流窗口大小
    aggressive_order_threshold: float = 0.001  # 激进订单阈值 (价格偏离%)

    # 信号生成配置
    signal_threshold: float = 0.3       # 信号生成阈值
    signal_ttl_seconds: float = 60.0    # 信号有效期 (秒)
    min_confidence: float = 0.5         # 最小置信度

    # 价格冲击预测配置
    price_impact_alpha: float = 0.5     # 价格冲击系数

    # 动量计算配置
    momentum_window: int = 20           # 动量窗口

    # 历史数据配置
    history_size: int = 1000            # 历史记录大小

    # 更新配置
    update_interval: float = 0.1        # 更新间隔 (秒)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_depth_levels': self.max_depth_levels,
            'sampling_levels': self.sampling_levels,
            'imbalance_threshold': self.imbalance_threshold,
            'extreme_imbalance_threshold': self.extreme_imbalance_threshold,
            'distance_weight_decay': self.distance_weight_decay,
            'volume_weight_power': self.volume_weight_power,
            'order_flow_window': self.order_flow_window,
            'aggressive_order_threshold': self.aggressive_order_threshold,
            'signal_threshold': self.signal_threshold,
            'signal_ttl_seconds': self.signal_ttl_seconds,
            'min_confidence': self.min_confidence,
            'price_impact_alpha': self.price_impact_alpha,
            'momentum_window': self.momentum_window,
            'history_size': self.history_size,
            'update_interval': self.update_interval,
        }


@dataclass
class ImbalanceStatistics:
    """
    失衡历史统计

    历史失衡数据的统计分析
    """
    symbol: str
    period_start: datetime
    period_end: datetime

    # 失衡统计
    total_samples: int                  # 总样本数
    avg_volume_imbalance: float         # 平均成交量失衡
    std_volume_imbalance: float         # 成交量失衡标准差
    max_volume_imbalance: float         # 最大成交量失衡
    min_volume_imbalance: float         # 最小成交量失衡

    # 方向分布
    buy_heavy_count: int = 0            # 买盘占优次数
    sell_heavy_count: int = 0           # 卖盘占优次数
    balanced_count: int = 0             # 均衡次数

    # 信号统计
    total_signals: int = 0              # 总信号数
    buy_signals: int = 0                # 买入信号数
    sell_signals: int = 0               # 卖出信号数
    avg_signal_confidence: float = 0.0  # 平均信号置信度

    # 准确性统计 (如果有实际价格数据)
    signal_accuracy: Optional[float] = None  # 信号准确率
    avg_prediction_error: Optional[float] = None  # 平均预测误差

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_samples': self.total_samples,
            'avg_volume_imbalance': self.avg_volume_imbalance,
            'std_volume_imbalance': self.std_volume_imbalance,
            'max_volume_imbalance': self.max_volume_imbalance,
            'min_volume_imbalance': self.min_volume_imbalance,
            'buy_heavy_count': self.buy_heavy_count,
            'sell_heavy_count': self.sell_heavy_count,
            'balanced_count': self.balanced_count,
            'total_signals': self.total_signals,
            'buy_signals': self.buy_signals,
            'sell_signals': self.sell_signals,
            'avg_signal_confidence': self.avg_signal_confidence,
            'signal_accuracy': self.signal_accuracy,
            'avg_prediction_error': self.avg_prediction_error,
        }


# ============== Main Analyzer ==============

class OrderBookImbalanceAnalyzer:
    """
    订单簿失衡分析器

    功能:
    - 实时订单簿快照分析
    - 多维度失衡指标计算
    - 深度失衡分析
    - 订单流失衡追踪
    - 失衡动量和趋势
    - 价格冲击预测
    - 交易信号生成
    - 历史统计分析
    """

    def __init__(
        self,
        config: Optional[ImbalanceConfig] = None,
        signal_callback: Optional[Callable[[ImbalanceSignal], None]] = None,
    ):
        self.config = config or ImbalanceConfig()
        self.signal_callback = signal_callback

        # 状态
        self._state = AnalyzerState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        # 当前数据
        self._current_snapshot: Optional[OrderBookSnapshot] = None
        self._current_metrics: Optional[ImbalanceMetrics] = None
        self._current_depth_analysis: Optional[DepthAnalysis] = None

        # 历史数据
        self._snapshot_history: Deque[OrderBookSnapshot] = deque(maxlen=self.config.history_size)
        self._metrics_history: Deque[ImbalanceMetrics] = deque(maxlen=self.config.history_size)
        self._depth_history: Deque[DepthAnalysis] = deque(maxlen=self.config.history_size)

        # 订单流数据
        self._order_flow: Deque[Tuple[datetime, float, str]] = deque(maxlen=self.config.order_flow_window)
        # (时间戳, 成交量, 方向: "buy"/"sell")

        # 信号历史
        self._signals: Deque[ImbalanceSignal] = deque(maxlen=self.config.history_size)

        # 失衡热力图数据 (档位 -> 历史失衡值)
        self._imbalance_heatmap: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        logger.info("OrderBookImbalanceAnalyzer initialized")

    @property
    def state(self) -> AnalyzerState:
        """获取分析器状态"""
        return self._state

    # ============== Lifecycle ==============

    async def start(self):
        """启动分析器"""
        if self._state == AnalyzerState.RUNNING:
            return

        self._state = AnalyzerState.RUNNING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("OrderBookImbalanceAnalyzer started")

    async def stop(self):
        """停止分析器"""
        self._state = AnalyzerState.IDLE
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("OrderBookImbalanceAnalyzer stopped")

    async def pause(self):
        """暂停分析器"""
        if self._state == AnalyzerState.RUNNING:
            self._state = AnalyzerState.PAUSED
            logger.info("OrderBookImbalanceAnalyzer paused")

    async def resume(self):
        """恢复分析器"""
        if self._state == AnalyzerState.PAUSED:
            self._state = AnalyzerState.RUNNING
            logger.info("OrderBookImbalanceAnalyzer resumed")

    async def _monitor_loop(self):
        """监控循环"""
        while self._state in (AnalyzerState.RUNNING, AnalyzerState.PAUSED):
            try:
                if self._state == AnalyzerState.RUNNING:
                    # 定期更新分析
                    await self._periodic_update()

                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self._state = AnalyzerState.ERROR

    async def _periodic_update(self):
        """定期更新"""
        # 清理过期信号
        self._cleanup_expired_signals()

    # ============== Order Book Update ==============

    def update_order_book(
        self,
        symbol: str,
        bids: List[Tuple[float, float, int]],  # (价格, 数量, 订单数)
        asks: List[Tuple[float, float, int]],
        timestamp: Optional[datetime] = None,
    ) -> OrderBookSnapshot:
        """
        更新订单簿

        Args:
            symbol: 交易对
            bids: 买单列表 [(价格, 数量, 订单数)]
            asks: 卖单列表 [(价格, 数量, 订单数)]
            timestamp: 时间戳

        Returns:
            订单簿快照
        """
        timestamp = timestamp or datetime.now()

        # 创建价格档位
        bid_levels = [
            PriceLevel(price=price, quantity=qty, order_count=count, timestamp=timestamp)
            for price, qty, count in sorted(bids, key=lambda x: x[0], reverse=True)
        ]

        ask_levels = [
            PriceLevel(price=price, quantity=qty, order_count=count, timestamp=timestamp)
            for price, qty, count in sorted(asks, key=lambda x: x[0])
        ]

        # 限制档位数
        bid_levels = bid_levels[:self.config.max_depth_levels]
        ask_levels = ask_levels[:self.config.max_depth_levels]

        # 创建快照
        snapshot = OrderBookSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            bids=bid_levels,
            asks=ask_levels,
        )

        # 存储快照
        self._current_snapshot = snapshot
        self._snapshot_history.append(snapshot)

        # 自动计算失衡指标
        self._calculate_all_metrics(snapshot)

        return snapshot

    # ============== Imbalance Calculation ==============

    def calculate_volume_imbalance(self, snapshot: Optional[OrderBookSnapshot] = None) -> float:
        """
        计算成交量失衡

        Returns:
            失衡值 (-1 to 1): 正值表示买盘占优，负值表示卖盘占优
        """
        snapshot = snapshot or self._current_snapshot
        if not snapshot:
            return 0.0

        bid_volume = snapshot.total_bid_volume
        ask_volume = snapshot.total_ask_volume

        if bid_volume + ask_volume == 0:
            return 0.0

        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        return max(-1.0, min(1.0, imbalance))

    def calculate_weighted_imbalance(
        self,
        snapshot: Optional[OrderBookSnapshot] = None,
        levels: Optional[int] = None,
    ) -> float:
        """
        计算加权失衡指标

        根据价格距离对各档位进行加权，近端档位权重更高

        Args:
            snapshot: 订单簿快照
            levels: 采样档位数

        Returns:
            加权失衡值 (-1 to 1)
        """
        snapshot = snapshot or self._current_snapshot
        if not snapshot or not snapshot.mid_price:
            return 0.0

        levels = levels or self.config.sampling_levels
        mid_price = snapshot.mid_price

        weighted_bid = 0.0
        weighted_ask = 0.0

        # 买单加权
        for i, level in enumerate(snapshot.bids[:levels]):
            distance = abs(level.price - mid_price) / mid_price
            weight = math.exp(-distance / self.config.distance_weight_decay)
            weighted_bid += level.quantity * weight

        # 卖单加权
        for i, level in enumerate(snapshot.asks[:levels]):
            distance = abs(level.price - mid_price) / mid_price
            weight = math.exp(-distance / self.config.distance_weight_decay)
            weighted_ask += level.quantity * weight

        if weighted_bid + weighted_ask == 0:
            return 0.0

        imbalance = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)
        return max(-1.0, min(1.0, imbalance))

    def calculate_order_flow_imbalance(self, window: Optional[int] = None) -> float:
        """
        计算订单流失衡

        基于最近的订单流数据计算失衡

        Args:
            window: 窗口大小

        Returns:
            订单流失衡 (-1 to 1)
        """
        window = window or self.config.order_flow_window

        if len(self._order_flow) < 2:
            return 0.0

        recent_flow = list(self._order_flow)[-window:]

        buy_volume = sum(vol for _, vol, direction in recent_flow if direction == "buy")
        sell_volume = sum(vol for _, vol, direction in recent_flow if direction == "sell")

        if buy_volume + sell_volume == 0:
            return 0.0

        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        return max(-1.0, min(1.0, imbalance))

    def analyze_depth_imbalance(
        self,
        snapshot: Optional[OrderBookSnapshot] = None,
    ) -> DepthAnalysis:
        """
        深度失衡分析

        分析订单簿各档位的深度分布和失衡情况

        Args:
            snapshot: 订单簿快照

        Returns:
            深度分析结果
        """
        snapshot = snapshot or self._current_snapshot
        if not snapshot:
            return DepthAnalysis(
                timestamp=datetime.now(),
                symbol="",
            )

        analysis = DepthAnalysis(
            timestamp=snapshot.timestamp,
            symbol=snapshot.symbol,
        )

        # 各档位深度
        for i, level in enumerate(snapshot.bids[:self.config.sampling_levels]):
            analysis.bid_depth_by_level[i] = level.quantity

        for i, level in enumerate(snapshot.asks[:self.config.sampling_levels]):
            analysis.ask_depth_by_level[i] = level.quantity

        # 累计深度
        cumulative_bid = 0.0
        cumulative_ask = 0.0

        for i in range(self.config.sampling_levels):
            cumulative_bid += analysis.bid_depth_by_level.get(i, 0.0)
            cumulative_ask += analysis.ask_depth_by_level.get(i, 0.0)
            analysis.cumulative_bid_depth.append(cumulative_bid)
            analysis.cumulative_ask_depth.append(cumulative_ask)

        # 深度比率
        for i in range(self.config.sampling_levels):
            bid_depth = analysis.bid_depth_by_level.get(i, 0.0)
            ask_depth = analysis.ask_depth_by_level.get(i, 0.0)

            if ask_depth > 0:
                analysis.depth_ratio_by_level[i] = bid_depth / ask_depth
            else:
                analysis.depth_ratio_by_level[i] = float('inf') if bid_depth > 0 else 1.0

        # 深度不对称性
        if cumulative_ask > 0:
            analysis.depth_asymmetry = (cumulative_bid - cumulative_ask) / cumulative_ask
        else:
            analysis.depth_asymmetry = 0.0

        # 深度集中度 (前3档占比)
        if cumulative_bid > 0:
            top3_bid = sum(analysis.cumulative_bid_depth[:3])
            analysis.bid_concentration = (top3_bid / cumulative_bid) if len(analysis.cumulative_bid_depth) >= 3 else 1.0

        if cumulative_ask > 0:
            top3_ask = sum(analysis.cumulative_ask_depth[:3])
            analysis.ask_concentration = (top3_ask / cumulative_ask) if len(analysis.cumulative_ask_depth) >= 3 else 1.0

        # 总深度和有效深度
        analysis.total_depth = cumulative_bid + cumulative_ask

        # 有效深度: 考虑价格距离的加权深度
        if snapshot.mid_price:
            effective_bid = 0.0
            effective_ask = 0.0

            for level in snapshot.bids[:self.config.sampling_levels]:
                distance = abs(level.price - snapshot.mid_price) / snapshot.mid_price
                weight = math.exp(-distance / self.config.distance_weight_decay)
                effective_bid += level.quantity * weight

            for level in snapshot.asks[:self.config.sampling_levels]:
                distance = abs(level.price - snapshot.mid_price) / snapshot.mid_price
                weight = math.exp(-distance / self.config.distance_weight_decay)
                effective_ask += level.quantity * weight

            analysis.effective_depth = effective_bid + effective_ask

        # 存储分析结果
        self._current_depth_analysis = analysis
        self._depth_history.append(analysis)

        return analysis

    def _calculate_all_metrics(self, snapshot: OrderBookSnapshot):
        """计算所有失衡指标"""
        # 基础失衡
        volume_imb = self.calculate_volume_imbalance(snapshot)

        # 订单数失衡
        total_bid_orders = snapshot.total_bid_orders
        total_ask_orders = snapshot.total_ask_orders

        if total_bid_orders + total_ask_orders > 0:
            order_imb = (total_bid_orders - total_ask_orders) / (total_bid_orders + total_ask_orders)
        else:
            order_imb = 0.0

        # 加权失衡
        weighted_imb = self.calculate_weighted_imbalance(snapshot)

        # 深度分析
        depth_analysis = self.analyze_depth_imbalance(snapshot)
        depth_weighted_imb = depth_analysis.depth_asymmetry

        # 订单流失衡
        order_flow_imb = self.calculate_order_flow_imbalance()

        # 激进订单流失衡 (简化计算)
        aggressive_flow_imb = order_flow_imb * 1.2 if abs(order_flow_imb) > 0.5 else order_flow_imb

        # 确定方向
        direction = self._determine_direction(volume_imb)

        # 确定严重程度
        severity = self._determine_severity(abs(volume_imb))

        # 计算动量和加速度
        momentum = self._calculate_momentum()
        acceleration = self._calculate_acceleration()

        # 确定趋势
        trend = self._determine_trend()

        # 置信度
        confidence = self._calculate_confidence(snapshot)

        # 创建指标对象
        metrics = ImbalanceMetrics(
            timestamp=snapshot.timestamp,
            symbol=snapshot.symbol,
            volume_imbalance=volume_imb,
            order_imbalance=order_imb,
            weighted_imbalance=weighted_imb,
            depth_weighted_imbalance=depth_weighted_imb,
            order_flow_imbalance=order_flow_imb,
            aggressive_flow_imbalance=aggressive_flow_imb,
            direction=direction,
            severity=severity,
            imbalance_momentum=momentum,
            imbalance_acceleration=acceleration,
            trend=trend,
            sample_levels=min(len(snapshot.bids), len(snapshot.asks), self.config.sampling_levels),
            confidence=confidence,
        )

        # 存储指标
        self._current_metrics = metrics
        self._metrics_history.append(metrics)

        # 更新热力图
        self._update_heatmap(snapshot)

        # 生成信号
        self._try_generate_signal(metrics)

    def _determine_direction(self, imbalance: float) -> ImbalanceDirection:
        """确定失衡方向"""
        threshold = self.config.imbalance_threshold

        if imbalance > threshold:
            return ImbalanceDirection.BUY_HEAVY
        elif imbalance < -threshold:
            return ImbalanceDirection.SELL_HEAVY
        elif abs(imbalance) < threshold / 2:
            return ImbalanceDirection.BALANCED
        else:
            return ImbalanceDirection.NEUTRAL

    def _determine_severity(self, abs_imbalance: float) -> ImbalanceSeverity:
        """确定失衡严重程度"""
        if abs_imbalance >= 0.8:
            return ImbalanceSeverity.EXTREME
        elif abs_imbalance >= 0.6:
            return ImbalanceSeverity.HIGH
        elif abs_imbalance >= 0.4:
            return ImbalanceSeverity.MODERATE
        elif abs_imbalance >= 0.2:
            return ImbalanceSeverity.LOW
        else:
            return ImbalanceSeverity.MINIMAL

    def _calculate_momentum(self) -> float:
        """计算失衡动量"""
        if len(self._metrics_history) < 2:
            return 0.0

        window = min(self.config.momentum_window, len(self._metrics_history))
        recent_metrics = list(self._metrics_history)[-window:]

        if len(recent_metrics) < 2:
            return 0.0

        # 简单动量: 最近N期的平均变化率
        imbalances = [m.volume_imbalance for m in recent_metrics]

        changes = [imbalances[i] - imbalances[i-1] for i in range(1, len(imbalances))]

        if not changes:
            return 0.0

        momentum = statistics.mean(changes)
        return momentum

    def _calculate_acceleration(self) -> float:
        """计算失衡加速度"""
        if len(self._metrics_history) < 3:
            return 0.0

        window = min(self.config.momentum_window, len(self._metrics_history))
        recent_metrics = list(self._metrics_history)[-window:]

        if len(recent_metrics) < 3:
            return 0.0

        imbalances = [m.volume_imbalance for m in recent_metrics]

        # 一阶差分 (动量)
        first_diff = [imbalances[i] - imbalances[i-1] for i in range(1, len(imbalances))]

        # 二阶差分 (加速度)
        second_diff = [first_diff[i] - first_diff[i-1] for i in range(1, len(first_diff))]

        if not second_diff:
            return 0.0

        acceleration = statistics.mean(second_diff)
        return acceleration

    def _determine_trend(self) -> ImbalanceTrend:
        """确定失衡趋势"""
        if len(self._metrics_history) < 5:
            return ImbalanceTrend.STABLE

        recent_metrics = list(self._metrics_history)[-10:]
        imbalances = [m.volume_imbalance for m in recent_metrics]

        # 计算趋势斜率
        n = len(imbalances)
        x = list(range(n))

        # 简单线性回归
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(imbalances)

        numerator = sum((x[i] - x_mean) * (imbalances[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return ImbalanceTrend.STABLE

        slope = numerator / denominator

        # 根据斜率判断趋势
        if abs(slope) > 0.05:
            if slope > 0:
                # 失衡增加 (向买盘倾斜)
                return ImbalanceTrend.INCREASING
            else:
                # 失衡减少 (向卖盘倾斜)
                return ImbalanceTrend.DECREASING

        # 检查是否反转
        if len(imbalances) >= 4:
            recent_change = imbalances[-1] - imbalances[-2]
            previous_change = imbalances[-2] - imbalances[-3]

            if recent_change * previous_change < 0 and abs(recent_change) > 0.1:
                return ImbalanceTrend.REVERSING

        return ImbalanceTrend.STABLE

    def _calculate_confidence(self, snapshot: OrderBookSnapshot) -> float:
        """
        计算置信度

        基于多个因素:
        - 订单簿深度
        - 档位数量
        - 价差大小
        - 历史稳定性
        """
        confidence = 0.0

        # 因素1: 深度 (0-0.3)
        total_volume = snapshot.total_bid_volume + snapshot.total_ask_volume
        if total_volume > 1000:
            confidence += 0.3
        elif total_volume > 100:
            confidence += 0.2
        elif total_volume > 10:
            confidence += 0.1

        # 因素2: 档位数 (0-0.2)
        levels = min(len(snapshot.bids), len(snapshot.asks))
        if levels >= self.config.sampling_levels:
            confidence += 0.2
        elif levels >= self.config.sampling_levels / 2:
            confidence += 0.1

        # 因素3: 价差 (0-0.2)
        if snapshot.spread_pct:
            if snapshot.spread_pct < 0.1:
                confidence += 0.2
            elif snapshot.spread_pct < 0.5:
                confidence += 0.1

        # 因素4: 历史稳定性 (0-0.3)
        if len(self._metrics_history) >= 10:
            recent = list(self._metrics_history)[-10:]
            imbalances = [m.volume_imbalance for m in recent]
            std = statistics.stdev(imbalances)

            if std < 0.1:
                confidence += 0.3
            elif std < 0.3:
                confidence += 0.2
            elif std < 0.5:
                confidence += 0.1

        return min(1.0, confidence)

    # ============== Price Impact Prediction ==============

    def predict_price_impact(
        self,
        volume: float,
        side: str,  # "buy" or "sell"
        snapshot: Optional[OrderBookSnapshot] = None,
    ) -> float:
        """
        预测价格冲击

        估算给定成交量对价格的冲击

        Args:
            volume: 成交量
            side: 方向 ("buy" 或 "sell")
            snapshot: 订单簿快照

        Returns:
            预期价格冲击 (%)
        """
        snapshot = snapshot or self._current_snapshot
        if not snapshot or not snapshot.mid_price:
            return 0.0

        mid_price = snapshot.mid_price

        # 选择对应方向的订单簿
        levels = snapshot.asks if side == "buy" else snapshot.bids

        # 计算需要消耗的档位
        remaining_volume = volume
        total_cost = 0.0

        for level in levels:
            if remaining_volume <= 0:
                break

            consumed = min(remaining_volume, level.quantity)
            total_cost += consumed * level.price
            remaining_volume -= consumed

        if volume <= 0:
            return 0.0

        # 平均成交价格
        avg_price = total_cost / (volume - remaining_volume) if (volume - remaining_volume) > 0 else mid_price

        # 价格冲击百分比
        impact_pct = abs((avg_price - mid_price) / mid_price * 100)

        # 应用冲击系数
        impact_pct *= self.config.price_impact_alpha

        return impact_pct if side == "buy" else -impact_pct

    # ============== Signal Generation ==============

    def generate_signals(
        self,
        metrics: Optional[ImbalanceMetrics] = None,
    ) -> Optional[ImbalanceSignal]:
        """
        生成交易信号

        基于失衡指标生成交易信号

        Args:
            metrics: 失衡指标

        Returns:
            交易信号 (如果满足条件)
        """
        metrics = metrics or self._current_metrics
        if not metrics:
            return None

        # 检查是否满足信号生成条件
        if metrics.confidence < self.config.min_confidence:
            return None

        # 综合失衡分数
        composite_score = (
            metrics.volume_imbalance * 0.3 +
            metrics.weighted_imbalance * 0.3 +
            metrics.order_flow_imbalance * 0.2 +
            metrics.depth_weighted_imbalance * 0.2
        )

        # 信号阈值
        threshold = self.config.signal_threshold

        if abs(composite_score) < threshold:
            return None

        # 确定信号类型和强度
        signal_type, signal_strength = self._determine_signal_type_and_strength(composite_score)

        # 预测方向
        predicted_direction = "up" if composite_score > 0 else "down"

        # 预测价格冲击
        predicted_impact = self._predict_signal_price_impact(metrics, composite_score)

        # 创建信号
        signal = ImbalanceSignal(
            signal_id=str(uuid.uuid4())[:8],
            timestamp=metrics.timestamp,
            symbol=metrics.symbol,
            signal_type=signal_type,
            signal_strength=signal_strength,
            confidence=metrics.confidence,
            volume_imbalance=metrics.volume_imbalance,
            weighted_imbalance=metrics.weighted_imbalance,
            order_flow_imbalance=metrics.order_flow_imbalance,
            predicted_direction=predicted_direction,
            predicted_price_impact=predicted_impact,
            ttl_seconds=self.config.signal_ttl_seconds,
            metadata={
                'composite_score': composite_score,
                'imbalance_momentum': metrics.imbalance_momentum,
                'trend': metrics.trend.value,
            },
        )

        # 存储信号
        self._signals.append(signal)

        # 回调
        if self.signal_callback:
            self.signal_callback(signal)

        return signal

    def _try_generate_signal(self, metrics: ImbalanceMetrics):
        """尝试生成信号"""
        try:
            self.generate_signals(metrics)
        except Exception as e:
            logger.error(f"Signal generation error: {e}")

    def _determine_signal_type_and_strength(
        self,
        composite_score: float,
    ) -> Tuple[SignalType, SignalStrength]:
        """确定信号类型和强度"""
        abs_score = abs(composite_score)

        # 确定强度
        if abs_score >= 0.7:
            strength = SignalStrength.VERY_STRONG
        elif abs_score >= 0.6:
            strength = SignalStrength.STRONG
        elif abs_score >= 0.4:
            strength = SignalStrength.MODERATE
        elif abs_score >= 0.3:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.VERY_WEAK

        # 确定类型
        if composite_score >= 0.6:
            signal_type = SignalType.STRONG_BUY
        elif composite_score >= 0.4:
            signal_type = SignalType.BUY
        elif composite_score >= 0.2:
            signal_type = SignalType.WEAK_BUY
        elif composite_score <= -0.6:
            signal_type = SignalType.STRONG_SELL
        elif composite_score <= -0.4:
            signal_type = SignalType.SELL
        elif composite_score <= -0.2:
            signal_type = SignalType.WEAK_SELL
        else:
            signal_type = SignalType.NEUTRAL

        return signal_type, strength

    def _predict_signal_price_impact(
        self,
        metrics: ImbalanceMetrics,
        composite_score: float,
    ) -> float:
        """预测信号的价格冲击"""
        # 简化预测: 基于综合分数和动量
        base_impact = abs(composite_score) * 1.0  # 基础冲击

        # 动量影响
        momentum_impact = metrics.imbalance_momentum * 0.5

        # 总预测冲击
        total_impact = base_impact + abs(momentum_impact)

        # 方向
        if composite_score > 0:
            return total_impact
        else:
            return -total_impact

    def _cleanup_expired_signals(self):
        """清理过期信号"""
        now = datetime.now()
        # deque不支持直接过滤，转换为列表再重建
        valid_signals = [s for s in self._signals if s.is_valid(now)]
        self._signals.clear()
        self._signals.extend(valid_signals)

    # ============== Heatmap ==============

    def _update_heatmap(self, snapshot: OrderBookSnapshot):
        """更新失衡热力图"""
        for i in range(min(len(snapshot.bids), len(snapshot.asks), self.config.sampling_levels)):
            if i < len(snapshot.bids) and i < len(snapshot.asks):
                bid_qty = snapshot.bids[i].quantity
                ask_qty = snapshot.asks[i].quantity

                if bid_qty + ask_qty > 0:
                    level_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
                    self._imbalance_heatmap[i].append(level_imbalance)

    def get_imbalance_heatmap(self, levels: Optional[int] = None) -> Dict[int, List[float]]:
        """
        获取失衡热力图

        返回各档位的历史失衡数据

        Args:
            levels: 档位数

        Returns:
            档位 -> 失衡历史列表
        """
        levels = levels or self.config.sampling_levels

        heatmap = {}
        for level in range(levels):
            if level in self._imbalance_heatmap:
                heatmap[level] = list(self._imbalance_heatmap[level])

        return heatmap

    # ============== Order Flow ==============

    def add_order_flow(
        self,
        volume: float,
        direction: str,  # "buy" or "sell"
        timestamp: Optional[datetime] = None,
    ):
        """
        添加订单流数据

        Args:
            volume: 成交量
            direction: 方向 ("buy" 或 "sell")
            timestamp: 时间戳
        """
        timestamp = timestamp or datetime.now()
        self._order_flow.append((timestamp, volume, direction))

    # ============== Query Methods ==============

    def get_current_snapshot(self) -> Optional[OrderBookSnapshot]:
        """获取当前订单簿快照"""
        return self._current_snapshot

    def get_current_metrics(self) -> Optional[ImbalanceMetrics]:
        """获取当前失衡指标"""
        return self._current_metrics

    def get_current_depth_analysis(self) -> Optional[DepthAnalysis]:
        """获取当前深度分析"""
        return self._current_depth_analysis

    def get_latest_signal(self) -> Optional[ImbalanceSignal]:
        """获取最新信号"""
        if self._signals:
            return self._signals[-1]
        return None

    def get_signals(
        self,
        limit: int = 100,
        valid_only: bool = False,
    ) -> List[ImbalanceSignal]:
        """
        获取信号列表

        Args:
            limit: 数量限制
            valid_only: 仅返回有效信号

        Returns:
            信号列表
        """
        signals = list(self._signals)[-limit:]

        if valid_only:
            now = datetime.now()
            signals = [s for s in signals if s.is_valid(now)]

        return signals

    def get_statistics(self) -> ImbalanceStatistics:
        """
        获取失衡统计信息

        Returns:
            统计结果
        """
        if not self._metrics_history:
            return ImbalanceStatistics(
                symbol="",
                period_start=datetime.now(),
                period_end=datetime.now(),
                total_samples=0,
                avg_volume_imbalance=0.0,
                std_volume_imbalance=0.0,
                max_volume_imbalance=0.0,
                min_volume_imbalance=0.0,
            )

        metrics_list = list(self._metrics_history)

        # 时间范围
        period_start = metrics_list[0].timestamp
        period_end = metrics_list[-1].timestamp

        # 失衡统计
        imbalances = [m.volume_imbalance for m in metrics_list]

        avg_imb = statistics.mean(imbalances)
        std_imb = statistics.stdev(imbalances) if len(imbalances) > 1 else 0.0
        max_imb = max(imbalances)
        min_imb = min(imbalances)

        # 方向分布
        buy_heavy = sum(1 for m in metrics_list if m.direction == ImbalanceDirection.BUY_HEAVY)
        sell_heavy = sum(1 for m in metrics_list if m.direction == ImbalanceDirection.SELL_HEAVY)
        balanced = sum(1 for m in metrics_list if m.direction == ImbalanceDirection.BALANCED)

        # 信号统计
        signals_list = list(self._signals)
        total_signals = len(signals_list)
        buy_signals = sum(
            1 for s in signals_list
            if s.signal_type in (SignalType.STRONG_BUY, SignalType.BUY, SignalType.WEAK_BUY)
        )
        sell_signals = sum(
            1 for s in signals_list
            if s.signal_type in (SignalType.STRONG_SELL, SignalType.SELL, SignalType.WEAK_SELL)
        )

        avg_confidence = statistics.mean([s.confidence for s in signals_list]) if signals_list else 0.0

        return ImbalanceStatistics(
            symbol=metrics_list[0].symbol if metrics_list else "",
            period_start=period_start,
            period_end=period_end,
            total_samples=len(metrics_list),
            avg_volume_imbalance=avg_imb,
            std_volume_imbalance=std_imb,
            max_volume_imbalance=max_imb,
            min_volume_imbalance=min_imb,
            buy_heavy_count=buy_heavy,
            sell_heavy_count=sell_heavy,
            balanced_count=balanced,
            total_signals=total_signals,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            avg_signal_confidence=avg_confidence,
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        获取摘要信息

        Returns:
            摘要字典
        """
        summary = {
            'state': self._state.value,
            'current_snapshot': self._current_snapshot.to_dict() if self._current_snapshot else None,
            'current_metrics': self._current_metrics.to_dict() if self._current_metrics else None,
            'current_depth_analysis': self._current_depth_analysis.to_dict() if self._current_depth_analysis else None,
            'latest_signal': self.get_latest_signal().to_dict() if self.get_latest_signal() else None,
            'statistics': self.get_statistics().to_dict(),
            'snapshot_history_size': len(self._snapshot_history),
            'metrics_history_size': len(self._metrics_history),
            'signals_count': len(self._signals),
            'config': self.config.to_dict(),
        }

        return summary


# ============== Factory Functions ==============

def create_order_book_analyzer(
    max_depth_levels: int = 20,
    sampling_levels: int = 10,
    signal_threshold: float = 0.3,
    **kwargs
) -> OrderBookImbalanceAnalyzer:
    """
    创建订单簿失衡分析器

    Args:
        max_depth_levels: 最大深度档位
        sampling_levels: 采样档位数
        signal_threshold: 信号生成阈值
        **kwargs: 其他配置参数

    Returns:
        分析器实例
    """
    config = ImbalanceConfig(
        max_depth_levels=max_depth_levels,
        sampling_levels=sampling_levels,
        signal_threshold=signal_threshold,
        **{k: v for k, v in kwargs.items() if hasattr(ImbalanceConfig, k)}
    )

    return OrderBookImbalanceAnalyzer(config=config)


# ============== Exports ==============

__all__ = [
    # Enums
    'ImbalanceDirection',
    'ImbalanceSeverity',
    'SignalType',
    'SignalStrength',
    'AnalyzerState',
    'ImbalanceTrend',
    # Data Classes
    'PriceLevel',
    'OrderBookSnapshot',
    'ImbalanceMetrics',
    'DepthAnalysis',
    'ImbalanceSignal',
    'ImbalanceConfig',
    'ImbalanceStatistics',
    # Main Class
    'OrderBookImbalanceAnalyzer',
    # Factory Functions
    'create_order_book_analyzer',
]
