"""
滑点追踪器模块 (Slippage Tracker)

提供全面的滑点分析功能:
- 滑点计算 (价格滑点、时间滑点、市场影响滑点)
- 滑点统计 (按品种、时间、订单规模、市场状态)
- 滑点预测 (基于历史、订单簿、波动率)
- 滑点归因 (执行延迟、市场条件、订单类型、流动性)
- 滑点优化建议 (最佳执行时间、订单拆分、限价vs市价)
"""

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict, deque
import numpy as np
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class SlippageType(str, Enum):
    """滑点类型"""
    PRICE = "price"                      # 价格滑点
    TIMING = "timing"                    # 时间滑点
    MARKET_IMPACT = "market_impact"      # 市场影响滑点
    TOTAL = "total"                      # 总滑点


class AttributionCategory(str, Enum):
    """归因类别"""
    EXECUTION_DELAY = "execution_delay"   # 执行延迟
    MARKET_CONDITION = "market_condition"  # 市场条件
    ORDER_TYPE = "order_type"            # 订单类型
    LIQUIDITY = "liquidity"              # 流动性
    ORDER_SIZE = "order_size"            # 订单大小
    TIME_OF_DAY = "time_of_day"          # 时间段
    VOLATILITY = "volatility"            # 波动率


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class MarketCondition(str, Enum):
    """市场状态"""
    NORMAL = "normal"          # 正常
    VOLATILE = "volatile"      # 波动
    ILLIQUID = "illiquid"      # 流动性差
    TRENDING = "trending"      # 趋势
    RANGING = "ranging"        # 震荡


class AlertLevel(str, Enum):
    """警报级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TrackerState(str, Enum):
    """追踪器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


# ==================== 数据类 ====================

@dataclass
class OrderRecord:
    """订单记录"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float

    # 价格信息
    decision_price: float = 0.0          # 决策价格
    arrival_price: float = 0.0           # 到达价格
    submission_price: float = 0.0        # 提交价格
    execution_price: float = 0.0         # 成交价格

    # 时间信息
    decision_time: Optional[datetime] = None    # 决策时间
    arrival_time: Optional[datetime] = None     # 到达时间
    submission_time: Optional[datetime] = None  # 提交时间
    execution_time: Optional[datetime] = None   # 成交时间

    # 市场数据
    bid_price: float = 0.0               # 买价
    ask_price: float = 0.0               # 卖价
    mid_price: float = 0.0               # 中间价
    spread: float = 0.0                  # 价差

    # 订单簿数据
    bid_depth: float = 0.0               # 买盘深度
    ask_depth: float = 0.0               # 卖盘深度

    # 市场状况
    volatility: float = 0.0              # 波动率
    volume: float = 0.0                  # 成交量
    market_condition: MarketCondition = MarketCondition.NORMAL

    # 元数据
    portfolio_id: str = ""
    strategy: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def spread_bps(self) -> float:
        """价差(bps)"""
        if self.mid_price > 0 and self.spread > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0

    @property
    def order_size_pct(self) -> float:
        """订单大小占比(相对日成交量)"""
        if self.volume > 0:
            return self.quantity / self.volume
        return 0.0

    @property
    def execution_delay_seconds(self) -> float:
        """执行延迟(秒)"""
        if self.decision_time and self.execution_time:
            return (self.execution_time - self.decision_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'decision_price': self.decision_price,
            'arrival_price': self.arrival_price,
            'execution_price': self.execution_price,
            'decision_time': self.decision_time.isoformat() if self.decision_time else None,
            'execution_time': self.execution_time.isoformat() if self.execution_time else None,
            'spread_bps': round(self.spread_bps, 2),
            'execution_delay_seconds': round(self.execution_delay_seconds, 2),
            'market_condition': self.market_condition.value,
            'portfolio_id': self.portfolio_id,
            'strategy': self.strategy,
        }


@dataclass
class SlippageResult:
    """滑点分析结果"""
    order_id: str
    symbol: str
    side: OrderSide

    # 滑点分解 (bps)
    price_slippage: float           # 价格滑点
    timing_slippage: float          # 时间滑点
    market_impact_slippage: float   # 市场影响滑点
    total_slippage: float           # 总滑点

    # 货币金额
    slippage_cost: float            # 滑点成本(货币)

    # 执行质量指标
    execution_quality_score: float  # 执行质量得分 (0-100)

    # 分析时间
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'price_slippage': round(self.price_slippage, 2),
            'timing_slippage': round(self.timing_slippage, 2),
            'market_impact_slippage': round(self.market_impact_slippage, 2),
            'total_slippage': round(self.total_slippage, 2),
            'slippage_cost': round(self.slippage_cost, 2),
            'execution_quality_score': round(self.execution_quality_score, 1),
            'analyzed_at': self.analyzed_at.isoformat(),
        }


@dataclass
class SlippageStatistics:
    """滑点统计"""
    dimension: str                      # 统计维度 (symbol/time/size/condition)
    dimension_value: str                # 维度值

    # 基础统计
    total_orders: int
    total_notional: float

    # 滑点统计 (bps)
    avg_price_slippage: float
    avg_timing_slippage: float
    avg_market_impact_slippage: float
    avg_total_slippage: float

    # 分布统计
    median_slippage: float
    std_slippage: float
    max_slippage: float
    min_slippage: float

    # 方向分解
    buy_avg_slippage: float
    sell_avg_slippage: float

    # 成本统计
    total_slippage_cost: float          # 总滑点成本

    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'dimension': self.dimension,
            'dimension_value': self.dimension_value,
            'total_orders': self.total_orders,
            'total_notional': round(self.total_notional, 2),
            'avg_price_slippage': round(self.avg_price_slippage, 2),
            'avg_timing_slippage': round(self.avg_timing_slippage, 2),
            'avg_market_impact_slippage': round(self.avg_market_impact_slippage, 2),
            'avg_total_slippage': round(self.avg_total_slippage, 2),
            'median_slippage': round(self.median_slippage, 2),
            'std_slippage': round(self.std_slippage, 2),
            'buy_avg_slippage': round(self.buy_avg_slippage, 2),
            'sell_avg_slippage': round(self.sell_avg_slippage, 2),
            'total_slippage_cost': round(self.total_slippage_cost, 2),
            'analyzed_at': self.analyzed_at.isoformat(),
        }


@dataclass
class SlippagePredict:
    """滑点预测"""
    symbol: str
    side: OrderSide
    quantity: float

    # 预测值 (bps)
    predicted_slippage: float
    confidence: float                   # 置信度 (0-1)

    # 预测区间
    slippage_low: float                 # 最低估计
    slippage_high: float                # 最高估计

    # 预测依据
    prediction_method: str              # 预测方法
    historical_avg: float               # 历史平均
    orderbook_estimate: float           # 订单簿估计
    volatility_estimate: float          # 波动率估计

    # 风险评估
    risk_level: str                     # 风险级别: low/medium/high

    predicted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'predicted_slippage': round(self.predicted_slippage, 2),
            'confidence': round(self.confidence, 2),
            'slippage_low': round(self.slippage_low, 2),
            'slippage_high': round(self.slippage_high, 2),
            'prediction_method': self.prediction_method,
            'risk_level': self.risk_level,
            'predicted_at': self.predicted_at.isoformat(),
        }


@dataclass
class SlippageAttribution:
    """滑点归因"""
    order_id: str
    symbol: str
    total_slippage: float               # 总滑点 (bps)

    # 归因分解 (贡献占比, 0-1)
    execution_delay_contribution: float
    market_condition_contribution: float
    order_type_contribution: float
    liquidity_contribution: float
    order_size_contribution: float
    time_of_day_contribution: float
    volatility_contribution: float

    # 详细说明
    attribution_details: Dict[str, Any] = field(default_factory=dict)

    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'total_slippage': round(self.total_slippage, 2),
            'execution_delay_contribution': round(self.execution_delay_contribution, 3),
            'market_condition_contribution': round(self.market_condition_contribution, 3),
            'order_type_contribution': round(self.order_type_contribution, 3),
            'liquidity_contribution': round(self.liquidity_contribution, 3),
            'order_size_contribution': round(self.order_size_contribution, 3),
            'time_of_day_contribution': round(self.time_of_day_contribution, 3),
            'volatility_contribution': round(self.volatility_contribution, 3),
            'attribution_details': self.attribution_details,
            'analyzed_at': self.analyzed_at.isoformat(),
        }


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    symbol: str
    side: OrderSide
    quantity: float

    # 建议类型
    recommendation_type: str            # 建议类型

    # 时间建议
    best_execution_time: Optional[str]  # 最佳执行时间
    time_reasoning: str                 # 时间选择理由

    # 订单拆分建议
    suggested_split: int                # 建议拆分数量
    split_interval_seconds: float       # 拆分间隔(秒)
    split_reasoning: str                # 拆分理由

    # 订单类型建议
    suggested_order_type: OrderType     # 建议订单类型
    limit_price: Optional[float]        # 建议限价
    order_type_reasoning: str           # 订单类型理由

    # 预期改善
    expected_slippage_reduction: float  # 预期滑点减少 (bps)
    expected_cost_saving: float         # 预期成本节省 (货币)

    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'recommendation_type': self.recommendation_type,
            'best_execution_time': self.best_execution_time,
            'time_reasoning': self.time_reasoning,
            'suggested_split': self.suggested_split,
            'split_interval_seconds': self.split_interval_seconds,
            'split_reasoning': self.split_reasoning,
            'suggested_order_type': self.suggested_order_type.value,
            'limit_price': self.limit_price,
            'order_type_reasoning': self.order_type_reasoning,
            'expected_slippage_reduction': round(self.expected_slippage_reduction, 2),
            'expected_cost_saving': round(self.expected_cost_saving, 2),
            'generated_at': self.generated_at.isoformat(),
        }


@dataclass
class SlippageAlert:
    """滑点警报"""
    alert_id: str
    order_id: str
    symbol: str
    level: AlertLevel

    # 警报内容
    message: str
    actual_slippage: float              # 实际滑点 (bps)
    expected_slippage: float            # 预期滑点 (bps)
    threshold: float                    # 阈值

    # 建议
    recommendation: str

    triggered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'level': self.level.value,
            'message': self.message,
            'actual_slippage': round(self.actual_slippage, 2),
            'expected_slippage': round(self.expected_slippage, 2),
            'threshold': round(self.threshold, 2),
            'recommendation': self.recommendation,
            'triggered_at': self.triggered_at.isoformat(),
        }


@dataclass
class TrackerConfig:
    """追踪器配置"""
    # 滑点阈值
    warning_slippage_bps: float = 10.0      # 警告阈值 (bps)
    critical_slippage_bps: float = 30.0     # 严重阈值 (bps)

    # 统计窗口
    history_window_days: int = 30           # 历史窗口 (天)
    max_history_orders: int = 10000         # 最大历史订单数

    # 预测参数
    prediction_confidence_threshold: float = 0.6  # 预测置信度阈值
    orderbook_depth_levels: int = 5         # 订单簿深度级数

    # 归因权重
    attribution_weights: Dict[str, float] = field(default_factory=lambda: {
        'execution_delay': 0.25,
        'market_condition': 0.20,
        'order_type': 0.15,
        'liquidity': 0.15,
        'order_size': 0.10,
        'time_of_day': 0.10,
        'volatility': 0.05,
    })

    # 优化建议参数
    large_order_threshold: float = 0.05      # 大订单阈值 (占日成交量比例)
    split_threshold_bps: float = 20.0        # 拆分建议阈值 (bps)

    # 回调函数
    enable_alerts: bool = True


# ==================== 数据提供者接口 ====================

class OrderBookProvider(ABC):
    """订单簿数据提供者"""

    @abstractmethod
    async def get_orderbook(
        self,
        symbol: str,
        depth: int = 5
    ) -> Dict[str, Any]:
        """
        获取订单簿

        Returns:
            {
                'bids': [(price, size), ...],
                'asks': [(price, size), ...]
            }
        """
        pass


class MarketDataProvider(ABC):
    """市场数据提供者"""

    @abstractmethod
    async def get_market_data(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        获取市场数据

        Returns:
            {
                'bid': float,
                'ask': float,
                'last': float,
                'volume': float,
                'volatility': float,
            }
        """
        pass


class HistoricalDataProvider(ABC):
    """历史数据提供者"""

    @abstractmethod
    async def get_historical_slippage(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """获取历史滑点数据 (bps)"""
        pass


# ==================== 模拟数据提供者 ====================

class SimulatedOrderBookProvider(OrderBookProvider):
    """模拟订单簿提供者"""

    def __init__(self):
        self._orderbooks: Dict[str, Dict[str, Any]] = {}

    def set_orderbook(self, symbol: str, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """设置订单簿"""
        self._orderbooks[symbol] = {
            'bids': bids,
            'asks': asks,
        }

    async def get_orderbook(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        """获取订单簿"""
        if symbol in self._orderbooks:
            ob = self._orderbooks[symbol]
            return {
                'bids': ob['bids'][:depth],
                'asks': ob['asks'][:depth],
            }

        # 默认订单簿
        base_price = 100.0
        return {
            'bids': [(base_price - i * 0.01, 100 - i * 10) for i in range(depth)],
            'asks': [(base_price + i * 0.01, 100 - i * 10) for i in range(depth)],
        }


class SimulatedMarketDataProvider(MarketDataProvider):
    """模拟市场数据提供者"""

    def __init__(self):
        self._market_data: Dict[str, Dict[str, Any]] = {}

    def set_market_data(self, symbol: str, data: Dict[str, Any]):
        """设置市场数据"""
        self._market_data[symbol] = data

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """获取市场数据"""
        if symbol in self._market_data:
            return self._market_data[symbol]

        # 默认市场数据
        return {
            'bid': 99.99,
            'ask': 100.01,
            'last': 100.00,
            'volume': 1000000,
            'volatility': 0.20,
        }


class SimulatedHistoricalDataProvider(HistoricalDataProvider):
    """模拟历史数据提供者"""

    def __init__(self):
        self._historical_slippage: Dict[str, List[float]] = defaultdict(list)

    def add_historical_slippage(self, symbol: str, slippage_bps: float):
        """添加历史滑点数据"""
        self._historical_slippage[symbol].append(slippage_bps)

    async def get_historical_slippage(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """获取历史滑点数据"""
        if symbol in self._historical_slippage:
            return self._historical_slippage[symbol]

        # 默认: 随机生成一些历史数据
        np.random.seed(42)
        n_days = (end_date - start_date).days
        return list(np.random.normal(5.0, 3.0, n_days))


# ==================== 滑点追踪器 ====================

class SlippageTracker:
    """滑点追踪器"""

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        orderbook_provider: Optional[OrderBookProvider] = None,
        market_data_provider: Optional[MarketDataProvider] = None,
        historical_data_provider: Optional[HistoricalDataProvider] = None,
    ):
        self.config = config or TrackerConfig()
        self.orderbook_provider = orderbook_provider or simulated_default(SimulatedOrderBookProvider, "data from SlippageTracker's simulated orderbook_provider (SimulatedOrderBookProvider)")
        self.market_data_provider = market_data_provider or simulated_default(SimulatedMarketDataProvider, "data from SlippageTracker's simulated market_data_provider (SimulatedMarketDataProvider)")
        self.historical_data_provider = historical_data_provider or simulated_default(SimulatedHistoricalDataProvider, "data from SlippageTracker's simulated historical_data_provider (SimulatedHistoricalDataProvider)")

        # 状态
        self._state = TrackerState.IDLE

        # 存储
        self._order_history: List[OrderRecord] = []
        self._slippage_results: Dict[str, SlippageResult] = {}
        self._statistics_cache: Dict[str, Dict[str, SlippageStatistics]] = {}

        # 警报回调
        self._alert_callbacks: List[Callable] = []

        logger.info("SlippageTracker initialized")

    @property
    def state(self) -> TrackerState:
        """获取状态"""
        return self._state

    def add_alert_callback(self, callback: Callable):
        """添加警报回调"""
        self._alert_callbacks.append(callback)

    async def start(self):
        """启动追踪器"""
        if self._state == TrackerState.RUNNING:
            logger.warning("Tracker already running")
            return

        self._state = TrackerState.RUNNING
        logger.info("SlippageTracker started")

    async def stop(self):
        """停止追踪器"""
        if self._state == TrackerState.STOPPED:
            logger.warning("Tracker already stopped")
            return

        self._state = TrackerState.STOPPED
        logger.info("SlippageTracker stopped")

    async def add_order(self, order: OrderRecord):
        """添加订单记录"""
        self._order_history.append(order)

        # 维护历史大小
        if len(self._order_history) > self.config.max_history_orders:
            self._order_history = self._order_history[-self.config.max_history_orders:]

        logger.debug(f"Added order {order.order_id} to history")

    async def calculate_slippage(self, order: OrderRecord) -> SlippageResult:
        """
        计算滑点

        Args:
            order: 订单记录

        Returns:
            滑点分析结果
        """
        # 1. 价格滑点 (Price Slippage)
        price_slippage = self._calculate_price_slippage(order)

        # 2. 时间滑点 (Timing Slippage)
        timing_slippage = self._calculate_timing_slippage(order)

        # 3. 市场影响滑点 (Market Impact Slippage)
        market_impact_slippage = self._calculate_market_impact_slippage(order)

        # 4. 总滑点
        total_slippage = price_slippage + timing_slippage + market_impact_slippage

        # 5. 货币成本
        notional = order.quantity * order.execution_price
        slippage_cost = notional * total_slippage / 10000

        # 6. 执行质量得分
        execution_quality_score = self._calculate_execution_quality(
            total_slippage, order
        )

        result = SlippageResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price_slippage=price_slippage,
            timing_slippage=timing_slippage,
            market_impact_slippage=market_impact_slippage,
            total_slippage=total_slippage,
            slippage_cost=slippage_cost,
            execution_quality_score=execution_quality_score,
        )

        # 缓存结果
        self._slippage_results[order.order_id] = result

        # 检查警报
        if self.config.enable_alerts:
            await self._check_alerts(order, result)

        return result

    async def get_statistics(
        self,
        dimension: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, SlippageStatistics]:
        """
        获取滑点统计

        Args:
            dimension: 统计维度 ('symbol'/'time'/'size'/'condition')
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            统计结果字典
        """
        # 筛选订单
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.config.history_window_days)
        if end_date is None:
            end_date = datetime.now()

        filtered_orders = [
            o for o in self._order_history
            if o.execution_time and start_date <= o.execution_time <= end_date
        ]

        if not filtered_orders:
            return {}

        # 按维度分组
        if dimension == 'symbol':
            groups = self._group_by_symbol(filtered_orders)
        elif dimension == 'time':
            groups = self._group_by_time(filtered_orders)
        elif dimension == 'size':
            groups = self._group_by_size(filtered_orders)
        elif dimension == 'condition':
            groups = self._group_by_condition(filtered_orders)
        else:
            raise ValueError(f"Unknown dimension: {dimension}")

        # 计算每组统计
        statistics = {}
        for dimension_value, orders in groups.items():
            stat = await self._calculate_group_statistics(
                dimension, dimension_value, orders
            )
            statistics[dimension_value] = stat

        # 缓存结果
        cache_key = f"{dimension}_{start_date.date()}_{end_date.date()}"
        self._statistics_cache[cache_key] = statistics

        return statistics

    async def predict_slippage(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> SlippagePredict:
        """
        预测滑点

        Args:
            symbol: 交易品种
            side: 交易方向
            quantity: 订单数量

        Returns:
            滑点预测
        """
        # 1. 基于历史的预测
        historical_avg = await self._predict_from_history(symbol, side)

        # 2. 基于订单簿的预测
        orderbook_estimate = await self._predict_from_orderbook(symbol, side, quantity)

        # 3. 基于波动率的预测
        volatility_estimate = await self._predict_from_volatility(symbol, quantity)

        # 4. 综合预测
        weights = [0.4, 0.4, 0.2]  # 历史、订单簿、波动率权重
        predicted_slippage = (
            weights[0] * historical_avg +
            weights[1] * orderbook_estimate +
            weights[2] * volatility_estimate
        )

        # 5. 计算置信度
        confidence = self._calculate_prediction_confidence(
            historical_avg, orderbook_estimate, volatility_estimate
        )

        # 6. 区间估计
        std = abs(predicted_slippage) * 0.3
        slippage_low = predicted_slippage - std
        slippage_high = predicted_slippage + std

        # 7. 风险评估
        risk_level = self._assess_risk_level(predicted_slippage)

        return SlippagePredict(
            symbol=symbol,
            side=side,
            quantity=quantity,
            predicted_slippage=predicted_slippage,
            confidence=confidence,
            slippage_low=slippage_low,
            slippage_high=slippage_high,
            prediction_method='weighted_ensemble',
            historical_avg=historical_avg,
            orderbook_estimate=orderbook_estimate,
            volatility_estimate=volatility_estimate,
            risk_level=risk_level,
        )

    async def attribute_slippage(self, order_id: str) -> SlippageAttribution:
        """
        滑点归因分析

        Args:
            order_id: 订单ID

        Returns:
            滑点归因结果
        """
        # 获取订单和滑点结果
        order = self._find_order(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        result = self._slippage_results.get(order_id)
        if not result:
            raise ValueError(f"Slippage result for {order_id} not found")

        total_slippage = result.total_slippage

        # 计算各因素贡献
        execution_delay_contrib = self._attribute_execution_delay(order, total_slippage)
        market_condition_contrib = self._attribute_market_condition(order, total_slippage)
        order_type_contrib = self._attribute_order_type(order, total_slippage)
        liquidity_contrib = self._attribute_liquidity(order, total_slippage)
        order_size_contrib = self._attribute_order_size(order, total_slippage)
        time_of_day_contrib = self._attribute_time_of_day(order, total_slippage)
        volatility_contrib = self._attribute_volatility(order, total_slippage)

        # 归一化贡献 (确保总和为1)
        total_contrib = (
            execution_delay_contrib +
            market_condition_contrib +
            order_type_contrib +
            liquidity_contrib +
            order_size_contrib +
            time_of_day_contrib +
            volatility_contrib
        )

        if total_contrib > 0:
            norm_factor = 1.0 / total_contrib
        else:
            norm_factor = 0

        attribution = SlippageAttribution(
            order_id=order_id,
            symbol=order.symbol,
            total_slippage=total_slippage,
            execution_delay_contribution=execution_delay_contrib * norm_factor,
            market_condition_contribution=market_condition_contrib * norm_factor,
            order_type_contribution=order_type_contrib * norm_factor,
            liquidity_contribution=liquidity_contrib * norm_factor,
            order_size_contribution=order_size_contrib * norm_factor,
            time_of_day_contribution=time_of_day_contrib * norm_factor,
            volatility_contribution=volatility_contrib * norm_factor,
            attribution_details={
                'execution_delay_seconds': order.execution_delay_seconds,
                'market_condition': order.market_condition.value,
                'order_type': order.order_type.value,
                'spread_bps': order.spread_bps,
                'order_size_pct': order.order_size_pct,
                'volatility': order.volatility,
            }
        )

        return attribution

    async def generate_optimization_recommendation(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float
    ) -> OptimizationRecommendation:
        """
        生成优化建议

        Args:
            symbol: 交易品种
            side: 交易方向
            quantity: 订单数量
            current_price: 当前价格

        Returns:
            优化建议
        """
        # 预测滑点
        prediction = await self.predict_slippage(symbol, side, quantity)

        # 获取市场数据
        market_data = await self.market_data_provider.get_market_data(symbol)

        # 计算订单大小占比
        order_size_pct = quantity / market_data['volume']

        # 1. 时间建议
        best_time, time_reasoning = self._recommend_execution_time(
            symbol, market_data
        )

        # 2. 订单拆分建议
        suggested_split, split_interval, split_reasoning = self._recommend_order_split(
            order_size_pct, prediction.predicted_slippage
        )

        # 3. 订单类型建议
        suggested_type, limit_price, type_reasoning = self._recommend_order_type(
            side, current_price, market_data, prediction.predicted_slippage
        )

        # 4. 预期改善
        expected_reduction = self._estimate_slippage_reduction(
            prediction.predicted_slippage,
            suggested_split,
            suggested_type
        )

        notional = quantity * current_price
        expected_saving = notional * expected_reduction / 10000

        recommendation = OptimizationRecommendation(
            symbol=symbol,
            side=side,
            quantity=quantity,
            recommendation_type='comprehensive',
            best_execution_time=best_time,
            time_reasoning=time_reasoning,
            suggested_split=suggested_split,
            split_interval_seconds=split_interval,
            split_reasoning=split_reasoning,
            suggested_order_type=suggested_type,
            limit_price=limit_price,
            order_type_reasoning=type_reasoning,
            expected_slippage_reduction=expected_reduction,
            expected_cost_saving=expected_saving,
        )

        return recommendation

    def get_recent_orders(self, n: int = 100) -> List[OrderRecord]:
        """获取最近的订单"""
        return self._order_history[-n:]

    def get_slippage_result(self, order_id: str) -> Optional[SlippageResult]:
        """获取滑点结果"""
        return self._slippage_results.get(order_id)

    def clear_history(self):
        """清除历史数据"""
        self._order_history.clear()
        self._slippage_results.clear()
        self._statistics_cache.clear()
        logger.info("History cleared")

    # ==================== 私有方法 ====================

    def _calculate_price_slippage(self, order: OrderRecord) -> float:
        """
        计算价格滑点

        价格滑点 = 执行价格与到达价格的差异
        """
        if order.arrival_price <= 0 or order.execution_price <= 0:
            return 0.0

        price_diff = order.execution_price - order.arrival_price

        # 对于买入，价格上涨是负滑点
        # 对于卖出，价格下跌是负滑点
        if order.side == OrderSide.BUY:
            slippage_bps = (price_diff / order.arrival_price) * 10000
        else:
            slippage_bps = (-price_diff / order.arrival_price) * 10000

        return slippage_bps

    def _calculate_timing_slippage(self, order: OrderRecord) -> float:
        """
        计算时间滑点

        时间滑点 = 决策价格与到达价格的差异
        """
        if order.decision_price <= 0 or order.arrival_price <= 0:
            return 0.0

        price_diff = order.arrival_price - order.decision_price

        if order.side == OrderSide.BUY:
            slippage_bps = (price_diff / order.decision_price) * 10000
        else:
            slippage_bps = (-price_diff / order.decision_price) * 10000

        return slippage_bps

    def _calculate_market_impact_slippage(self, order: OrderRecord) -> float:
        """
        计算市场影响滑点

        使用平方根模型估计市场影响
        """
        if order.volume <= 0 or order.volatility <= 0:
            return 0.0

        # 参与率
        participation_rate = order.quantity / order.volume

        # 日波动率
        daily_vol = order.volatility / math.sqrt(252)

        # 平方根模型: Impact = coef * sqrt(participation_rate) * volatility
        coefficient = 0.1
        impact_bps = coefficient * math.sqrt(participation_rate) * daily_vol * 10000

        return impact_bps

    def _calculate_execution_quality(
        self,
        total_slippage: float,
        order: OrderRecord
    ) -> float:
        """
        计算执行质量得分 (0-100)

        考虑:
        - 滑点大小
        - 执行延迟
        - 市场条件
        """
        score = 100.0

        # 1. 滑点惩罚 (每1bps扣0.5分)
        slippage_penalty = abs(total_slippage) * 0.5
        score -= slippage_penalty

        # 2. 延迟惩罚 (每秒扣0.1分)
        delay_penalty = order.execution_delay_seconds * 0.1
        score -= delay_penalty

        # 3. 市场条件调整
        if order.market_condition == MarketCondition.VOLATILE:
            score += 5  # 波动市场容忍度
        elif order.market_condition == MarketCondition.ILLIQUID:
            score += 10  # 流动性差容忍度更高

        return max(0, min(100, score))

    async def _check_alerts(self, order: OrderRecord, result: SlippageResult):
        """检查并触发警报"""
        total_slippage = result.total_slippage

        # 确定警报级别
        level = None
        if abs(total_slippage) > self.config.critical_slippage_bps:
            level = AlertLevel.CRITICAL
        elif abs(total_slippage) > self.config.warning_slippage_bps:
            level = AlertLevel.WARNING

        if level:
            alert = SlippageAlert(
                alert_id=f"SLIPPAGE_ALERT_{order.order_id}_{datetime.now().timestamp()}",
                order_id=order.order_id,
                symbol=order.symbol,
                level=level,
                message=f"High slippage detected: {total_slippage:.2f} bps",
                actual_slippage=total_slippage,
                expected_slippage=5.0,  # 简化: 使用固定预期
                threshold=self.config.warning_slippage_bps if level == AlertLevel.WARNING else self.config.critical_slippage_bps,
                recommendation="Consider splitting the order or using limit orders",
            )

            # 调用回调
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    def _group_by_symbol(self, orders: List[OrderRecord]) -> Dict[str, List[OrderRecord]]:
        """按品种分组"""
        groups = defaultdict(list)
        for order in orders:
            groups[order.symbol].append(order)
        return dict(groups)

    def _group_by_time(self, orders: List[OrderRecord]) -> Dict[str, List[OrderRecord]]:
        """按时间段分组"""
        groups = defaultdict(list)
        for order in orders:
            if order.execution_time:
                hour = order.execution_time.hour
                if 9 <= hour < 11:
                    bucket = '09:00-11:00'
                elif 11 <= hour < 13:
                    bucket = '11:00-13:00'
                elif 13 <= hour < 15:
                    bucket = '13:00-15:00'
                else:
                    bucket = '15:00-16:00'
                groups[bucket].append(order)
        return dict(groups)

    def _group_by_size(self, orders: List[OrderRecord]) -> Dict[str, List[OrderRecord]]:
        """按订单大小分组"""
        groups = defaultdict(list)
        for order in orders:
            size_pct = order.order_size_pct
            if size_pct < 0.01:
                bucket = 'small'
            elif size_pct < 0.05:
                bucket = 'medium'
            else:
                bucket = 'large'
            groups[bucket].append(order)
        return dict(groups)

    def _group_by_condition(self, orders: List[OrderRecord]) -> Dict[str, List[OrderRecord]]:
        """按市场状态分组"""
        groups = defaultdict(list)
        for order in orders:
            groups[order.market_condition.value].append(order)
        return dict(groups)

    async def _calculate_group_statistics(
        self,
        dimension: str,
        dimension_value: str,
        orders: List[OrderRecord]
    ) -> SlippageStatistics:
        """计算分组统计"""
        # 计算滑点
        slippages = []
        buy_slippages = []
        sell_slippages = []
        total_cost = 0.0
        total_notional = 0.0

        price_slippages = []
        timing_slippages = []
        impact_slippages = []

        for order in orders:
            result = self._slippage_results.get(order.order_id)
            if not result:
                # 如果没有缓存，计算滑点
                result = await self.calculate_slippage(order)

            slippages.append(result.total_slippage)
            price_slippages.append(result.price_slippage)
            timing_slippages.append(result.timing_slippage)
            impact_slippages.append(result.market_impact_slippage)

            if order.side == OrderSide.BUY:
                buy_slippages.append(result.total_slippage)
            else:
                sell_slippages.append(result.total_slippage)

            total_cost += result.slippage_cost
            notional = order.quantity * order.execution_price
            total_notional += notional

        # 统计值
        avg_total = np.mean(slippages) if slippages else 0.0
        median = np.median(slippages) if slippages else 0.0
        std = np.std(slippages) if slippages else 0.0
        max_slip = max(slippages) if slippages else 0.0
        min_slip = min(slippages) if slippages else 0.0

        avg_price = np.mean(price_slippages) if price_slippages else 0.0
        avg_timing = np.mean(timing_slippages) if timing_slippages else 0.0
        avg_impact = np.mean(impact_slippages) if impact_slippages else 0.0

        buy_avg = np.mean(buy_slippages) if buy_slippages else 0.0
        sell_avg = np.mean(sell_slippages) if sell_slippages else 0.0

        return SlippageStatistics(
            dimension=dimension,
            dimension_value=dimension_value,
            total_orders=len(orders),
            total_notional=total_notional,
            avg_price_slippage=avg_price,
            avg_timing_slippage=avg_timing,
            avg_market_impact_slippage=avg_impact,
            avg_total_slippage=avg_total,
            median_slippage=median,
            std_slippage=std,
            max_slippage=max_slip,
            min_slippage=min_slip,
            buy_avg_slippage=buy_avg,
            sell_avg_slippage=sell_avg,
            total_slippage_cost=total_cost,
        )

    async def _predict_from_history(self, symbol: str, side: OrderSide) -> float:
        """基于历史预测滑点"""
        # 获取历史订单
        historical_orders = [
            o for o in self._order_history
            if o.symbol == symbol and o.side == side
        ]

        if len(historical_orders) < 5:
            # 使用历史数据提供者
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            historical_slippage = await self.historical_data_provider.get_historical_slippage(
                symbol, start_date, end_date
            )
            return np.mean(historical_slippage) if historical_slippage else 5.0

        # 计算历史平均滑点
        recent_orders = historical_orders[-20:]  # 最近20笔
        slippages = []

        for order in recent_orders:
            result = self._slippage_results.get(order.order_id)
            if result:
                slippages.append(result.total_slippage)

        if slippages:
            return np.mean(slippages)

        return 5.0  # 默认值

    async def _predict_from_orderbook(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> float:
        """基于订单簿预测滑点"""
        orderbook = await self.orderbook_provider.get_orderbook(
            symbol, self.config.orderbook_depth_levels
        )

        if side == OrderSide.BUY:
            levels = orderbook['asks']
        else:
            levels = orderbook['bids']

        if not levels:
            return 5.0

        # 模拟市场订单执行
        remaining_qty = quantity
        total_cost = 0.0

        for price, size in levels:
            fill_qty = min(remaining_qty, size)
            total_cost += fill_qty * price
            remaining_qty -= fill_qty

            if remaining_qty <= 0:
                break

        if remaining_qty > 0:
            # 订单簿深度不足
            return 20.0  # 高滑点

        # 计算加权平均价格
        avg_price = total_cost / quantity

        # 与最优价格比较
        best_price = levels[0][0]

        # Ensure we return non-zero slippage for orderbook-based prediction
        slippage_bps = abs((avg_price - best_price) / best_price) * 10000

        # If slippage is very small, use a minimum baseline
        if slippage_bps < 0.1:
            slippage_bps = 0.5  # Minimum 0.5 bps for orderbook impact

        return slippage_bps

    async def _predict_from_volatility(self, symbol: str, quantity: float) -> float:
        """基于波动率预测滑点"""
        market_data = await self.market_data_provider.get_market_data(symbol)

        volatility = market_data.get('volatility', 0.20)
        volume = market_data.get('volume', 1000000)

        # 参与率
        participation_rate = quantity / volume if volume > 0 else 0.01

        # 日波动率
        daily_vol = volatility / math.sqrt(252)

        # 简化模型
        slippage_bps = 0.1 * math.sqrt(participation_rate) * daily_vol * 10000

        return slippage_bps

    def _calculate_prediction_confidence(
        self,
        historical: float,
        orderbook: float,
        volatility: float
    ) -> float:
        """计算预测置信度"""
        # 计算三个预测的标准差
        values = [historical, orderbook, volatility]
        std = np.std(values)
        mean = np.mean(values)

        # 相对标准差越小，置信度越高
        if mean > 0:
            cv = std / mean  # 变异系数
            confidence = 1.0 / (1.0 + cv)
        else:
            confidence = 0.5

        return min(1.0, max(0.0, confidence))

    def _assess_risk_level(self, predicted_slippage: float) -> str:
        """评估风险级别"""
        abs_slippage = abs(predicted_slippage)

        if abs_slippage < 10:
            return 'low'
        elif abs_slippage < 20:
            return 'medium'
        else:
            return 'high'

    def _find_order(self, order_id: str) -> Optional[OrderRecord]:
        """查找订单"""
        for order in self._order_history:
            if order.order_id == order_id:
                return order
        return None

    def _attribute_execution_delay(self, order: OrderRecord, total_slippage: float) -> float:
        """归因: 执行延迟"""
        delay_seconds = order.execution_delay_seconds

        # 延迟越长，贡献越大
        if delay_seconds < 1:
            contrib = 0.1
        elif delay_seconds < 5:
            contrib = 0.3
        elif delay_seconds < 10:
            contrib = 0.5
        else:
            contrib = 0.8

        return contrib * self.config.attribution_weights.get('execution_delay', 0.25)

    def _attribute_market_condition(self, order: OrderRecord, total_slippage: float) -> float:
        """归因: 市场条件"""
        condition = order.market_condition

        if condition == MarketCondition.VOLATILE:
            contrib = 0.8
        elif condition == MarketCondition.ILLIQUID:
            contrib = 0.7
        elif condition == MarketCondition.TRENDING:
            contrib = 0.4
        else:
            contrib = 0.2

        return contrib * self.config.attribution_weights.get('market_condition', 0.20)

    def _attribute_order_type(self, order: OrderRecord, total_slippage: float) -> float:
        """归因: 订单类型"""
        if order.order_type == OrderType.MARKET:
            contrib = 0.7
        elif order.order_type == OrderType.LIMIT:
            contrib = 0.3
        else:
            contrib = 0.5

        return contrib * self.config.attribution_weights.get('order_type', 0.15)

    def _attribute_liquidity(self, order: OrderRecord, total_slippage: float) -> float:
        """归因: 流动性"""
        spread_bps = order.spread_bps

        # 价差越大，流动性越差，贡献越大
        if spread_bps < 5:
            contrib = 0.2
        elif spread_bps < 10:
            contrib = 0.4
        elif spread_bps < 20:
            contrib = 0.6
        else:
            contrib = 0.9

        return contrib * self.config.attribution_weights.get('liquidity', 0.15)

    def _attribute_order_size(self, order: OrderRecord, total_slippage: float) -> float:
        """归因: 订单大小"""
        size_pct = order.order_size_pct

        # 订单越大，贡献越大
        if size_pct < 0.01:
            contrib = 0.1
        elif size_pct < 0.05:
            contrib = 0.4
        elif size_pct < 0.10:
            contrib = 0.7
        else:
            contrib = 1.0

        return contrib * self.config.attribution_weights.get('order_size', 0.10)

    def _attribute_time_of_day(self, order: OrderRecord, total_slippage: float) -> float:
        """归因: 时间段"""
        if not order.execution_time:
            return 0.0

        hour = order.execution_time.hour

        # 开盘和收盘时段滑点更大
        if hour in [9, 10, 15]:
            contrib = 0.6
        elif hour in [11, 12, 13, 14]:
            contrib = 0.3
        else:
            contrib = 0.2

        return contrib * self.config.attribution_weights.get('time_of_day', 0.10)

    def _attribute_volatility(self, order: OrderRecord, total_slippage: float) -> float:
        """归因: 波动率"""
        volatility = order.volatility

        # 波动率越高，贡献越大
        if volatility < 0.15:
            contrib = 0.2
        elif volatility < 0.30:
            contrib = 0.5
        else:
            contrib = 0.8

        return contrib * self.config.attribution_weights.get('volatility', 0.05)

    def _recommend_execution_time(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Tuple[str, str]:
        """推荐最佳执行时间"""
        current_hour = datetime.now().hour

        # 避开开盘和收盘
        if current_hour < 10:
            return "10:30-11:30", "Avoid opening volatility"
        elif current_hour >= 14:
            return "Next day 10:30-11:30", "Avoid closing volatility"
        else:
            return "Within next 30 minutes", "Normal trading hours"

    def _recommend_order_split(
        self,
        order_size_pct: float,
        predicted_slippage: float
    ) -> Tuple[int, float, str]:
        """推荐订单拆分"""
        if order_size_pct > self.config.large_order_threshold:
            # 大订单，建议拆分
            n_splits = math.ceil(order_size_pct / 0.02)  # 每片不超过2%
            n_splits = min(n_splits, 10)  # 最多拆10片
            interval = 300  # 5分钟间隔
            reasoning = f"Large order ({order_size_pct*100:.1f}% of daily volume), split to reduce market impact"
            return n_splits, interval, reasoning
        elif predicted_slippage > self.config.split_threshold_bps:
            # 预期滑点高，建议拆分
            n_splits = 3
            interval = 180  # 3分钟间隔
            reasoning = f"High predicted slippage ({predicted_slippage:.1f} bps), split to reduce impact"
            return n_splits, interval, reasoning
        else:
            # 不需要拆分
            return 1, 0, "Order size acceptable for single execution"

    def _recommend_order_type(
        self,
        side: OrderSide,
        current_price: float,
        market_data: Dict[str, Any],
        predicted_slippage: float
    ) -> Tuple[OrderType, Optional[float], str]:
        """推荐订单类型"""
        spread = market_data.get('ask', current_price) - market_data.get('bid', current_price)
        spread_bps = (spread / current_price) * 10000 if current_price > 0 else 0

        if predicted_slippage > 15:
            # 高滑点，使用限价单
            if side == OrderSide.BUY:
                limit_price = current_price * 0.999  # 略低于市价
                reasoning = f"High slippage expected ({predicted_slippage:.1f} bps), use limit order"
            else:
                limit_price = current_price * 1.001  # 略高于市价
                reasoning = f"High slippage expected ({predicted_slippage:.1f} bps), use limit order"
            return OrderType.LIMIT, limit_price, reasoning
        elif spread_bps > 10:
            # 价差大，使用限价单
            if side == OrderSide.BUY:
                limit_price = market_data.get('bid', current_price) + spread * 0.3
            else:
                limit_price = market_data.get('ask', current_price) - spread * 0.3
            reasoning = f"Wide spread ({spread_bps:.1f} bps), use limit order"
            return OrderType.LIMIT, limit_price, reasoning
        else:
            # 正常情况，使用市价单
            reasoning = f"Normal market conditions, market order acceptable"
            return OrderType.MARKET, None, reasoning

    def _estimate_slippage_reduction(
        self,
        baseline_slippage: float,
        n_splits: int,
        order_type: OrderType
    ) -> float:
        """估计滑点减少"""
        reduction = 0.0

        # 拆单减少
        if n_splits > 1:
            reduction += baseline_slippage * 0.3  # 拆单可减少30%

        # 限价单减少
        if order_type == OrderType.LIMIT:
            reduction += baseline_slippage * 0.4  # 限价单可减少40%

        return min(reduction, baseline_slippage * 0.6)  # 最多减少60%


# ==================== 工厂函数 ====================

def create_slippage_tracker(
    warning_slippage_bps: float = 10.0,
    critical_slippage_bps: float = 30.0,
    history_window_days: int = 30,
    enable_alerts: bool = True,
    orderbook_provider: Optional[OrderBookProvider] = None,
    market_data_provider: Optional[MarketDataProvider] = None,
    historical_data_provider: Optional[HistoricalDataProvider] = None,
    **kwargs
) -> SlippageTracker:
    """
    创建滑点追踪器

    Args:
        warning_slippage_bps: 警告阈值 (bps)
        critical_slippage_bps: 严重阈值 (bps)
        history_window_days: 历史窗口 (天)
        enable_alerts: 启用警报
        orderbook_provider: 订单簿提供者
        market_data_provider: 市场数据提供者
        historical_data_provider: 历史数据提供者
        **kwargs: 其他配置参数

    Returns:
        SlippageTracker实例
    """
    config = TrackerConfig(
        warning_slippage_bps=warning_slippage_bps,
        critical_slippage_bps=critical_slippage_bps,
        history_window_days=history_window_days,
        enable_alerts=enable_alerts,
        **{k: v for k, v in kwargs.items() if hasattr(TrackerConfig, k)}
    )

    return SlippageTracker(
        config=config,
        orderbook_provider=orderbook_provider,
        market_data_provider=market_data_provider,
        historical_data_provider=historical_data_provider,
    )


def create_order_record(
    order_id: str,
    symbol: str,
    side: str,
    order_type: str,
    quantity: float,
    **kwargs
) -> OrderRecord:
    """
    创建订单记录

    Args:
        order_id: 订单ID
        symbol: 交易品种
        side: 方向 ('buy' 或 'sell')
        order_type: 订单类型
        quantity: 数量
        **kwargs: 其他参数

    Returns:
        OrderRecord实例
    """
    return OrderRecord(
        order_id=order_id,
        symbol=symbol,
        side=OrderSide(side),
        order_type=OrderType(order_type),
        quantity=quantity,
        **{k: v for k, v in kwargs.items() if hasattr(OrderRecord, k)}
    )


async def quick_slippage_analysis(
    order: OrderRecord
) -> SlippageResult:
    """
    快速滑点分析

    Args:
        order: 订单记录

    Returns:
        滑点分析结果
    """
    tracker = create_slippage_tracker()
    return await tracker.calculate_slippage(order)
