"""
流动性聚合器模块 (Liquidity Aggregator)

提供多交易所流动性聚合功能:
- 多交易所流动性聚合
- 订单簿深度合并
- 最优价格发现
- 流动性评分系统
- 流动性成本估算
- 执行策略推荐
- 流动性预测模型
- 暗池流动性整合
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class LiquidityTier(str, Enum):
    """流动性层级"""
    TIER_1 = "tier_1"           # 顶级流动性
    TIER_2 = "tier_2"           # 二级流动性
    TIER_3 = "tier_3"           # 三级流动性
    DARK = "dark"               # 暗池流动性
    RETAIL = "retail"           # 零售流动性


class LiquidityType(str, Enum):
    """流动性类型"""
    DISPLAYED = "displayed"     # 显示流动性
    HIDDEN = "hidden"           # 隐藏流动性
    IMPLIED = "implied"         # 隐含流动性
    DARK_POOL = "dark_pool"     # 暗池流动性
    INDICATIVE = "indicative"   # 指示性流动性


class AggregationMethod(str, Enum):
    """聚合方法"""
    VOLUME_WEIGHTED = "volume_weighted"     # 成交量加权
    PRICE_PRIORITY = "price_priority"       # 价格优先
    TIME_PRIORITY = "time_priority"         # 时间优先
    PRO_RATA = "pro_rata"                   # 按比例分配
    SMART_BLEND = "smart_blend"             # 智能混合


class LiquidityScore(str, Enum):
    """流动性评分等级"""
    EXCELLENT = "excellent"     # 优秀 (90-100)
    GOOD = "good"              # 良好 (70-89)
    FAIR = "fair"              # 一般 (50-69)
    POOR = "poor"              # 较差 (30-49)
    VERY_POOR = "very_poor"    # 很差 (0-29)


class PriceLevel(str, Enum):
    """价格水平"""
    BEST = "best"              # 最优价格
    NBBO = "nbbo"              # 国家最优买卖价
    NEAR_TOUCH = "near_touch"  # 接近最优价
    INSIDE_SPREAD = "inside"   # 价差内
    MID_POINT = "mid_point"    # 中间价


# ==================== 数据类 ====================

@dataclass
class ExchangeLiquidity:
    """交易所流动性数据"""
    exchange_id: str
    symbol: str
    timestamp: datetime

    # 买卖盘深度 (价格 -> 数量)
    bid_levels: Dict[float, float] = field(default_factory=dict)
    ask_levels: Dict[float, float] = field(default_factory=dict)

    # 流动性类型分布
    displayed_liquidity: float = 0
    hidden_liquidity: float = 0
    implied_liquidity: float = 0

    # 流动性质量指标
    spread_bps: float = 0
    depth_score: float = 0           # 深度评分 0-100
    resilience_score: float = 0      # 恢复力评分 0-100

    # 交易活动
    trade_velocity: float = 0        # 每秒交易次数
    volume_velocity: float = 0       # 每秒成交量

    # 历史统计
    avg_spread_bps: float = 0
    avg_depth: float = 0

    # 状态
    is_active: bool = True
    last_update_ms: float = 0

    @property
    def best_bid(self) -> Optional[float]:
        """最优买价"""
        return max(self.bid_levels.keys()) if self.bid_levels else None

    @property
    def best_ask(self) -> Optional[float]:
        """最优卖价"""
        return min(self.ask_levels.keys()) if self.ask_levels else None

    @property
    def mid_price(self) -> Optional[float]:
        """中间价"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def total_bid_volume(self) -> float:
        """总买盘量"""
        return sum(self.bid_levels.values())

    @property
    def total_ask_volume(self) -> float:
        """总卖盘量"""
        return sum(self.ask_levels.values())

    @property
    def total_liquidity(self) -> float:
        """总流动性"""
        return self.displayed_liquidity + self.hidden_liquidity + self.implied_liquidity

    def to_dict(self) -> Dict[str, Any]:
        return {
            'exchange_id': self.exchange_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread_bps': round(self.spread_bps, 2),
            'total_bid_volume': round(self.total_bid_volume, 2),
            'total_ask_volume': round(self.total_ask_volume, 2),
            'displayed_liquidity': round(self.displayed_liquidity, 2),
            'hidden_liquidity': round(self.hidden_liquidity, 2),
            'depth_score': round(self.depth_score, 1),
            'resilience_score': round(self.resilience_score, 1),
        }


@dataclass
class AggregatedBook:
    """聚合订单簿"""
    symbol: str
    timestamp: datetime

    # 聚合后的价格层级
    bid_levels: List[Dict[str, Any]] = field(default_factory=list)  # [{'price', 'size', 'exchanges', 'count'}]
    ask_levels: List[Dict[str, Any]] = field(default_factory=list)

    # 最优报价
    nbbo_bid: Optional[float] = None
    nbbo_ask: Optional[float] = None
    nbbo_bid_size: float = 0
    nbbo_ask_size: float = 0
    nbbo_bid_exchanges: List[str] = field(default_factory=list)
    nbbo_ask_exchanges: List[str] = field(default_factory=list)

    # 聚合统计
    total_exchanges: int = 0
    total_bid_liquidity: float = 0
    total_ask_liquidity: float = 0
    weighted_spread_bps: float = 0

    # 深度指标
    depth_1_percent: float = 0       # 1%价格范围内的流动性
    depth_5_percent: float = 0       # 5%价格范围内的流动性

    # 流动性评分
    liquidity_score: float = 0       # 综合流动性评分 0-100
    score_grade: LiquidityScore = LiquidityScore.FAIR

    @property
    def mid_price(self) -> Optional[float]:
        """中间价"""
        if self.nbbo_bid and self.nbbo_ask:
            return (self.nbbo_bid + self.nbbo_ask) / 2
        return None

    @property
    def spread_bps(self) -> float:
        """价差(基点)"""
        if self.mid_price and self.mid_price > 0 and self.nbbo_bid and self.nbbo_ask:
            return (self.nbbo_ask - self.nbbo_bid) / self.mid_price * 10000
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'nbbo_bid': self.nbbo_bid,
            'nbbo_ask': self.nbbo_ask,
            'nbbo_bid_size': round(self.nbbo_bid_size, 2),
            'nbbo_ask_size': round(self.nbbo_ask_size, 2),
            'mid_price': self.mid_price,
            'spread_bps': round(self.spread_bps, 2),
            'total_exchanges': self.total_exchanges,
            'total_bid_liquidity': round(self.total_bid_liquidity, 2),
            'total_ask_liquidity': round(self.total_ask_liquidity, 2),
            'liquidity_score': round(self.liquidity_score, 1),
            'score_grade': self.score_grade.value,
            'bid_levels': self.bid_levels[:10],  # 前10档
            'ask_levels': self.ask_levels[:10],
        }


@dataclass
class LiquidityCost:
    """流动性成本估算"""
    symbol: str
    side: str                        # buy/sell
    quantity: float

    # 价格估算
    estimated_avg_price: float
    best_price: float
    worst_price: float

    # 成本分解
    spread_cost_bps: float           # 价差成本
    impact_cost_bps: float           # 冲击成本
    opportunity_cost_bps: float      # 机会成本
    slippage_bps: float              # 滑点
    total_cost_bps: float            # 总成本

    # 成本金额
    estimated_cost_amount: float

    # 执行建议
    suggested_method: str            # 建议执行方法
    suggested_split: int             # 建议拆分数
    suggested_exchanges: List[str] = field(default_factory=list)

    # 置信度
    confidence: float = 0            # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'estimated_avg_price': round(self.estimated_avg_price, 4),
            'best_price': round(self.best_price, 4),
            'worst_price': round(self.worst_price, 4),
            'spread_cost_bps': round(self.spread_cost_bps, 2),
            'impact_cost_bps': round(self.impact_cost_bps, 2),
            'slippage_bps': round(self.slippage_bps, 2),
            'total_cost_bps': round(self.total_cost_bps, 2),
            'estimated_cost_amount': round(self.estimated_cost_amount, 2),
            'suggested_method': self.suggested_method,
            'suggested_split': self.suggested_split,
            'suggested_exchanges': self.suggested_exchanges,
            'confidence': round(self.confidence, 2),
        }


@dataclass
class ExecutionStrategy:
    """执行策略建议"""
    strategy_name: str
    description: str

    # 执行参数
    recommended_venues: List[Dict[str, Any]]  # [{'exchange', 'allocation', 'reason'}]
    total_allocation: float

    # 执行方式
    execution_method: str            # market/limit/iceberg/twap/vwap
    urgency_level: str              # low/medium/high

    # 预期结果
    expected_fill_rate: float        # 预期成交率
    expected_cost_bps: float         # 预期成本
    expected_duration_seconds: float # 预期耗时

    # 风险评估
    price_risk: float                # 价格风险 0-1
    execution_risk: float            # 执行风险 0-1

    # 评分
    strategy_score: float = 0        # 策略得分 0-100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'description': self.description,
            'recommended_venues': self.recommended_venues,
            'total_allocation': round(self.total_allocation, 2),
            'execution_method': self.execution_method,
            'urgency_level': self.urgency_level,
            'expected_fill_rate': round(self.expected_fill_rate * 100, 2),
            'expected_cost_bps': round(self.expected_cost_bps, 2),
            'expected_duration_seconds': round(self.expected_duration_seconds, 1),
            'price_risk': round(self.price_risk, 2),
            'execution_risk': round(self.execution_risk, 2),
            'strategy_score': round(self.strategy_score, 1),
        }


@dataclass
class LiquidityPrediction:
    """流动性预测"""
    symbol: str
    prediction_horizon_seconds: int
    timestamp: datetime

    # 预测的流动性指标
    predicted_spread_bps: float
    predicted_depth: float
    predicted_volatility: float

    # 预测的价格范围
    predicted_price_range: Tuple[float, float]  # (low, high)

    # 流动性可用性
    liquidity_availability: float    # 0-1, 1表示充足

    # 最佳执行时间窗口
    optimal_execution_window: Optional[Tuple[datetime, datetime]] = None

    # 置信区间
    confidence_lower: float = 0
    confidence_upper: float = 0
    model_confidence: float = 0      # 模型置信度 0-1

    # 特殊事件
    events: List[str] = field(default_factory=list)  # 影响流动性的事件

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'prediction_horizon_seconds': self.prediction_horizon_seconds,
            'timestamp': self.timestamp.isoformat(),
            'predicted_spread_bps': round(self.predicted_spread_bps, 2),
            'predicted_depth': round(self.predicted_depth, 2),
            'predicted_volatility': round(self.predicted_volatility, 4),
            'predicted_price_range': [round(p, 4) for p in self.predicted_price_range],
            'liquidity_availability': round(self.liquidity_availability, 2),
            'model_confidence': round(self.model_confidence, 2),
            'events': self.events,
        }


@dataclass
class DarkPoolLiquidity:
    """暗池流动性数据"""
    pool_id: str
    symbol: str
    timestamp: datetime

    # 流动性指示
    indicative_size: float           # 指示性规模
    min_block_size: float            # 最小块订单

    # 匹配概率
    match_probability: float = 0     # 匹配概率 0-1

    # 历史数据
    avg_execution_size: float = 0
    avg_price_improvement_bps: float = 0
    fill_rate_history: float = 0

    # 特征
    accepts_retail: bool = False
    minimum_value: float = 0

    # 成本
    fee_bps: float = 0

    # 可用性
    is_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pool_id': self.pool_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'indicative_size': round(self.indicative_size, 2),
            'min_block_size': round(self.min_block_size, 2),
            'match_probability': round(self.match_probability, 2),
            'avg_price_improvement_bps': round(self.avg_price_improvement_bps, 2),
            'fill_rate_history': round(self.fill_rate_history * 100, 2),
            'fee_bps': round(self.fee_bps, 2),
            'is_available': self.is_available,
        }


@dataclass
class LiquidityMetrics:
    """流动性综合指标"""
    symbol: str
    timestamp: datetime

    # 基础指标
    total_exchanges: int
    total_liquidity: float
    avg_spread_bps: float

    # 深度指标
    top_of_book_liquidity: float     # 最优价流动性
    near_touch_liquidity: float      # 近价流动性
    total_depth_liquidity: float     # 总深度流动性

    # 质量指标
    liquidity_score: float           # 综合评分 0-100
    depth_quality: float             # 深度质量 0-1
    spread_quality: float            # 价差质量 0-1
    resilience_quality: float        # 恢复力质量 0-1

    # 市场微观结构
    order_book_imbalance: float      # 订单簿失衡度 -1到1
    pressure_indicator: float        # 买卖压力指标 -1到1

    # 交易活动
    trade_intensity: float           # 交易强度
    turnover_rate: float            # 换手率

    # 成本估计
    small_order_cost_bps: float     # 小订单成本(<1%)
    medium_order_cost_bps: float    # 中订单成本(1-5%)
    large_order_cost_bps: float     # 大订单成本(>5%)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'total_exchanges': self.total_exchanges,
            'total_liquidity': round(self.total_liquidity, 2),
            'avg_spread_bps': round(self.avg_spread_bps, 2),
            'liquidity_score': round(self.liquidity_score, 1),
            'depth_quality': round(self.depth_quality, 2),
            'spread_quality': round(self.spread_quality, 2),
            'resilience_quality': round(self.resilience_quality, 2),
            'order_book_imbalance': round(self.order_book_imbalance, 2),
            'trade_intensity': round(self.trade_intensity, 2),
            'small_order_cost_bps': round(self.small_order_cost_bps, 2),
            'medium_order_cost_bps': round(self.medium_order_cost_bps, 2),
            'large_order_cost_bps': round(self.large_order_cost_bps, 2),
        }


@dataclass
class AggregatorConfig:
    """聚合器配置"""
    # 数据源配置
    min_exchanges: int = 2                      # 最小交易所数量
    max_exchanges: int = 20                     # 最大交易所数量
    include_dark_pools: bool = True             # 包含暗池

    # 数据质量
    max_quote_age_ms: float = 1000              # 最大报价年龄(毫秒)
    min_depth_levels: int = 5                   # 最小深度档位

    # 聚合配置
    default_aggregation_method: AggregationMethod = AggregationMethod.SMART_BLEND
    depth_levels_to_aggregate: int = 20         # 聚合深度档位数

    # 流动性评分权重
    spread_weight: float = 0.3                  # 价差权重
    depth_weight: float = 0.4                   # 深度权重
    resilience_weight: float = 0.3              # 恢复力权重

    # 成本估算
    enable_impact_model: bool = True            # 启用冲击模型
    impact_model_type: str = "square_root"      # 冲击模型类型

    # 预测模型
    enable_prediction: bool = True              # 启用预测
    prediction_window_seconds: int = 300        # 预测窗口(秒)
    history_window_seconds: int = 3600          # 历史窗口(秒)

    # 缓存配置
    cache_enabled: bool = True
    cache_ttl_seconds: int = 10


# ==================== 流动性聚合器 ====================

class LiquidityAggregator:
    """流动性聚合器引擎"""

    def __init__(self, config: Optional[AggregatorConfig] = None):
        self.config = config or AggregatorConfig()

        # 数据存储
        self._exchange_liquidity: Dict[str, Dict[str, ExchangeLiquidity]] = defaultdict(dict)  # symbol -> exchange -> liquidity
        self._dark_pools: Dict[str, Dict[str, DarkPoolLiquidity]] = defaultdict(dict)  # symbol -> pool -> liquidity

        # 聚合缓存
        self._aggregated_books: Dict[str, AggregatedBook] = {}  # symbol -> book
        self._cache_timestamps: Dict[str, datetime] = {}

        # 历史数据(用于预测)
        self._liquidity_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # 统计
        self._stats = {
            'aggregations_performed': 0,
            'total_exchanges_tracked': 0,
            'total_symbols_tracked': 0,
            'predictions_made': 0,
            'cost_estimates_made': 0,
        }

        # 回调
        self._on_book_update: Optional[Callable[[AggregatedBook], None]] = None
        self._on_liquidity_alert: Optional[Callable[[str, str], None]] = None

    def update_exchange_liquidity(self, liquidity: ExchangeLiquidity) -> None:
        """
        更新交易所流动性数据

        Args:
            liquidity: 交易所流动性数据
        """
        self._exchange_liquidity[liquidity.symbol][liquidity.exchange_id] = liquidity
        liquidity.last_update_ms = (datetime.now() - liquidity.timestamp).total_seconds() * 1000

        # 记录历史
        if self.config.enable_prediction:
            self._liquidity_history[liquidity.symbol].append({
                'timestamp': liquidity.timestamp,
                'spread_bps': liquidity.spread_bps,
                'depth': liquidity.total_bid_volume + liquidity.total_ask_volume,
                'resilience': liquidity.resilience_score,
            })

        # 使缓存失效
        if liquidity.symbol in self._cache_timestamps:
            del self._cache_timestamps[liquidity.symbol]

        logger.debug(f"Updated liquidity for {liquidity.exchange_id}:{liquidity.symbol}")

    def update_dark_pool_liquidity(self, dark_liquidity: DarkPoolLiquidity) -> None:
        """
        更新暗池流动性数据

        Args:
            dark_liquidity: 暗池流动性数据
        """
        self._dark_pools[dark_liquidity.symbol][dark_liquidity.pool_id] = dark_liquidity
        logger.debug(f"Updated dark pool liquidity for {dark_liquidity.pool_id}:{dark_liquidity.symbol}")

    def aggregate_order_book(
        self,
        symbol: str,
        method: Optional[AggregationMethod] = None,
        max_levels: int = 20
    ) -> AggregatedBook:
        """
        聚合订单簿

        Args:
            symbol: 标的代码
            method: 聚合方法
            max_levels: 最大层级数

        Returns:
            聚合后的订单簿
        """
        # 检查缓存
        if self.config.cache_enabled and symbol in self._cache_timestamps:
            age = (datetime.now() - self._cache_timestamps[symbol]).total_seconds()
            if age < self.config.cache_ttl_seconds and symbol in self._aggregated_books:
                return self._aggregated_books[symbol]

        method = method or self.config.default_aggregation_method

        # 获取有效的交易所流动性
        exchange_data = self._get_valid_exchanges(symbol)

        if not exchange_data:
            raise ValueError(f"No valid liquidity data for {symbol}")

        # 聚合买卖盘
        aggregated_bids = self._aggregate_side(exchange_data, 'bid', max_levels)
        aggregated_asks = self._aggregate_side(exchange_data, 'ask', max_levels)

        # 计算NBBO
        nbbo = self._calculate_nbbo(exchange_data)

        # 计算深度指标
        mid_price = (nbbo['best_bid'] + nbbo['best_ask']) / 2 if nbbo.get('best_bid') and nbbo.get('best_ask') else 0
        depth_1pct = self._calculate_depth_in_range(aggregated_bids, aggregated_asks, mid_price, 0.01)
        depth_5pct = self._calculate_depth_in_range(aggregated_bids, aggregated_asks, mid_price, 0.05)

        # 计算流动性评分
        liquidity_score, score_grade = self._calculate_liquidity_score(
            exchange_data, aggregated_bids, aggregated_asks, nbbo
        )

        # 计算加权价差
        weighted_spread = self._calculate_weighted_spread(exchange_data)

        # 创建聚合订单簿
        book = AggregatedBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_levels=aggregated_bids,
            ask_levels=aggregated_asks,
            nbbo_bid=nbbo.get('best_bid'),
            nbbo_ask=nbbo.get('best_ask'),
            nbbo_bid_size=nbbo.get('best_bid_size', 0),
            nbbo_ask_size=nbbo.get('best_ask_size', 0),
            nbbo_bid_exchanges=nbbo.get('best_bid_exchanges', []),
            nbbo_ask_exchanges=nbbo.get('best_ask_exchanges', []),
            total_exchanges=len(exchange_data),
            total_bid_liquidity=sum(level['size'] for level in aggregated_bids),
            total_ask_liquidity=sum(level['size'] for level in aggregated_asks),
            weighted_spread_bps=weighted_spread,
            depth_1_percent=depth_1pct,
            depth_5_percent=depth_5pct,
            liquidity_score=liquidity_score,
            score_grade=score_grade,
        )

        # 缓存
        self._aggregated_books[symbol] = book
        self._cache_timestamps[symbol] = datetime.now()

        # 更新统计
        self._stats['aggregations_performed'] += 1

        # 触发回调
        if self._on_book_update:
            self._on_book_update(book)

        return book

    def estimate_liquidity_cost(
        self,
        symbol: str,
        side: str,
        quantity: float,
        urgency: float = 0.5
    ) -> LiquidityCost:
        """
        估算流动性成本

        Args:
            symbol: 标的代码
            side: 方向 (buy/sell)
            quantity: 数量
            urgency: 紧急度 0-1

        Returns:
            流动性成本估算
        """
        # 获取聚合订单簿
        book = self.aggregate_order_book(symbol)

        # 选择对应的盘口
        levels = book.ask_levels if side.lower() == 'buy' else book.bid_levels
        best_price = book.nbbo_ask if side.lower() == 'buy' else book.nbbo_bid

        if not levels or not best_price:
            raise ValueError(f"Insufficient liquidity data for {symbol}")

        # 计算价格冲击
        avg_price, worst_price, consumed_levels = self._calculate_price_impact(levels, quantity)

        # 价差成本
        mid_price = book.mid_price or best_price
        spread_cost_bps = abs(best_price - mid_price) / mid_price * 10000 if mid_price > 0 else 0

        # 冲击成本
        impact_cost_bps = abs(avg_price - best_price) / best_price * 10000 if best_price > 0 else 0

        # 滑点估算 (基于市场深度)
        market_depth_ratio = quantity / book.total_bid_liquidity if side.lower() == 'sell' else quantity / book.total_ask_liquidity
        slippage_bps = self._estimate_slippage(market_depth_ratio, urgency, book.liquidity_score)

        # 机会成本
        opportunity_cost_bps = slippage_bps * (1 - urgency) * 0.5

        # 总成本
        total_cost_bps = spread_cost_bps + impact_cost_bps + slippage_bps + opportunity_cost_bps

        # 成本金额
        cost_amount = avg_price * quantity * (total_cost_bps / 10000)

        # 执行建议
        suggested_method, suggested_split = self._suggest_execution_method(
            quantity, market_depth_ratio, urgency, book.liquidity_score
        )

        # 推荐交易所
        suggested_exchanges = self._suggest_exchanges(symbol, side, quantity)

        # 置信度
        confidence = self._calculate_cost_confidence(book, consumed_levels, len(book.bid_levels) + len(book.ask_levels))

        result = LiquidityCost(
            symbol=symbol,
            side=side,
            quantity=quantity,
            estimated_avg_price=avg_price,
            best_price=best_price,
            worst_price=worst_price,
            spread_cost_bps=spread_cost_bps,
            impact_cost_bps=impact_cost_bps,
            opportunity_cost_bps=opportunity_cost_bps,
            slippage_bps=slippage_bps,
            total_cost_bps=total_cost_bps,
            estimated_cost_amount=cost_amount,
            suggested_method=suggested_method,
            suggested_split=suggested_split,
            suggested_exchanges=suggested_exchanges,
            confidence=confidence,
        )

        self._stats['cost_estimates_made'] += 1

        return result

    def recommend_execution_strategy(
        self,
        symbol: str,
        side: str,
        quantity: float,
        urgency: float = 0.5,
        max_cost_bps: Optional[float] = None
    ) -> List[ExecutionStrategy]:
        """
        推荐执行策略

        Args:
            symbol: 标的代码
            side: 方向
            quantity: 数量
            urgency: 紧急度 0-1
            max_cost_bps: 最大成本限制(基点)

        Returns:
            执行策略列表(按评分排序)
        """
        book = self.aggregate_order_book(symbol)
        cost_estimate = self.estimate_liquidity_cost(symbol, side, quantity, urgency)

        strategies = []

        # 策略1: 市价单快速执行
        if urgency >= 0.7:
            market_strategy = self._create_market_strategy(
                symbol, side, quantity, book, cost_estimate, urgency
            )
            strategies.append(market_strategy)

        # 策略2: 限价单被动执行
        if urgency <= 0.5:
            limit_strategy = self._create_limit_strategy(
                symbol, side, quantity, book, cost_estimate, urgency
            )
            strategies.append(limit_strategy)

        # 策略3: 冰山单隐藏规模
        if quantity > book.depth_1_percent:
            iceberg_strategy = self._create_iceberg_strategy(
                symbol, side, quantity, book, cost_estimate, urgency
            )
            strategies.append(iceberg_strategy)

        # 策略4: TWAP时间加权
        twap_strategy = self._create_twap_strategy(
            symbol, side, quantity, book, cost_estimate, urgency
        )
        strategies.append(twap_strategy)

        # 策略5: VWAP成交量加权
        vwap_strategy = self._create_vwap_strategy(
            symbol, side, quantity, book, cost_estimate, urgency
        )
        strategies.append(vwap_strategy)

        # 策略6: 暗池优先
        if self.config.include_dark_pools and self._has_dark_liquidity(symbol):
            dark_strategy = self._create_dark_pool_strategy(
                symbol, side, quantity, book, cost_estimate, urgency
            )
            strategies.append(dark_strategy)

        # 过滤成本超限的策略
        if max_cost_bps:
            strategies = [s for s in strategies if s.expected_cost_bps <= max_cost_bps]

        # 按评分排序
        strategies.sort(key=lambda x: x.strategy_score, reverse=True)

        return strategies

    def predict_liquidity(
        self,
        symbol: str,
        horizon_seconds: int = 300
    ) -> LiquidityPrediction:
        """
        预测未来流动性

        Args:
            symbol: 标的代码
            horizon_seconds: 预测时间范围(秒)

        Returns:
            流动性预测
        """
        if not self.config.enable_prediction:
            raise ValueError("Liquidity prediction is disabled")

        history = list(self._liquidity_history[symbol])

        if len(history) < 10:
            raise ValueError(f"Insufficient history for prediction: {len(history)} points")

        # 提取时间序列
        spreads = [h['spread_bps'] for h in history]
        depths = [h['depth'] for h in history]

        # 简单预测模型(移动平均 + 趋势)
        window = min(20, len(spreads))
        predicted_spread = np.mean(spreads[-window:])
        spread_trend = (spreads[-1] - spreads[-window]) / window if window > 1 else 0
        predicted_spread += spread_trend * (horizon_seconds / 60)

        predicted_depth = np.mean(depths[-window:])
        depth_trend = (depths[-1] - depths[-window]) / window if window > 1 else 0
        predicted_depth += depth_trend * (horizon_seconds / 60)

        # 预测波动率
        if len(spreads) > 1:
            spread_volatility = np.std(spreads[-window:]) / np.mean(spreads[-window:]) if np.mean(spreads[-window:]) > 0 else 0
        else:
            spread_volatility = 0

        # 价格范围预测
        current_book = self.aggregate_order_book(symbol)
        mid_price = current_book.mid_price or 100
        price_range = (
            mid_price * (1 - spread_volatility * 2),
            mid_price * (1 + spread_volatility * 2)
        )

        # 流动性可用性评分
        liquidity_availability = min(1.0, predicted_depth / (np.mean(depths) + 1e-6)) if depths else 0.5

        # 最佳执行窗口(流动性最好的时段)
        optimal_window = self._find_optimal_execution_window(history, horizon_seconds)

        # 模型置信度
        model_confidence = min(len(history) / 100, 1.0) * (1 - spread_volatility)

        # 检测特殊事件
        events = self._detect_liquidity_events(history)

        prediction = LiquidityPrediction(
            symbol=symbol,
            prediction_horizon_seconds=horizon_seconds,
            timestamp=datetime.now(),
            predicted_spread_bps=predicted_spread,
            predicted_depth=predicted_depth,
            predicted_volatility=spread_volatility,
            predicted_price_range=price_range,
            liquidity_availability=liquidity_availability,
            optimal_execution_window=optimal_window,
            model_confidence=model_confidence,
            events=events,
        )

        self._stats['predictions_made'] += 1

        return prediction

    def get_liquidity_metrics(self, symbol: str) -> LiquidityMetrics:
        """
        获取综合流动性指标

        Args:
            symbol: 标的代码

        Returns:
            流动性综合指标
        """
        book = self.aggregate_order_book(symbol)
        exchange_data = self._get_valid_exchanges(symbol)

        # 基础指标
        total_exchanges = len(exchange_data)
        total_liquidity = book.total_bid_liquidity + book.total_ask_liquidity
        avg_spread = book.weighted_spread_bps

        # 深度分层
        mid_price = book.mid_price or 0
        top_of_book = book.nbbo_bid_size + book.nbbo_ask_size
        near_touch = book.depth_1_percent
        total_depth = total_liquidity

        # 质量指标
        depth_quality = min(1.0, total_depth / 10000)  # 归一化
        spread_quality = max(0, 1 - avg_spread / 100)  # 价差越小越好

        # 恢复力质量(从历史计算)
        resilience_scores = [ex.resilience_score for ex in exchange_data.values() if ex.resilience_score > 0]
        resilience_quality = np.mean(resilience_scores) / 100 if resilience_scores else 0.5

        # 订单簿失衡
        if book.total_bid_liquidity + book.total_ask_liquidity > 0:
            imbalance = (book.total_bid_liquidity - book.total_ask_liquidity) / (book.total_bid_liquidity + book.total_ask_liquidity)
        else:
            imbalance = 0

        # 压力指标(基于失衡和价差)
        pressure = imbalance * (1 - spread_quality)

        # 交易活动
        trade_intensities = [ex.trade_velocity for ex in exchange_data.values()]
        trade_intensity = sum(trade_intensities)

        # 估算换手率(简化)
        turnover_rate = trade_intensity * 3600 / total_liquidity if total_liquidity > 0 else 0

        # 不同订单规模的成本估算
        small_cost = self.estimate_liquidity_cost(symbol, 'buy', total_liquidity * 0.005, 0.5).total_cost_bps if total_liquidity > 0 else 0
        medium_cost = self.estimate_liquidity_cost(symbol, 'buy', total_liquidity * 0.03, 0.5).total_cost_bps if total_liquidity > 0 else 0
        large_cost = self.estimate_liquidity_cost(symbol, 'buy', total_liquidity * 0.1, 0.5).total_cost_bps if total_liquidity > 0 else 0

        return LiquidityMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            total_exchanges=total_exchanges,
            total_liquidity=total_liquidity,
            avg_spread_bps=avg_spread,
            top_of_book_liquidity=top_of_book,
            near_touch_liquidity=near_touch,
            total_depth_liquidity=total_depth,
            liquidity_score=book.liquidity_score,
            depth_quality=depth_quality,
            spread_quality=spread_quality,
            resilience_quality=resilience_quality,
            order_book_imbalance=imbalance,
            pressure_indicator=pressure,
            trade_intensity=trade_intensity,
            turnover_rate=turnover_rate,
            small_order_cost_bps=small_cost,
            medium_order_cost_bps=medium_cost,
            large_order_cost_bps=large_cost,
        )

    def get_dark_pool_opportunities(self, symbol: str, min_size: float = 0) -> List[DarkPoolLiquidity]:
        """
        获取暗池流动性机会

        Args:
            symbol: 标的代码
            min_size: 最小规模要求

        Returns:
            暗池流动性列表
        """
        dark_pools = self._dark_pools.get(symbol, {})

        opportunities = [
            pool for pool in dark_pools.values()
            if pool.is_available and pool.indicative_size >= min_size
        ]

        # 按匹配概率排序
        opportunities.sort(key=lambda x: x.match_probability, reverse=True)

        return opportunities

    def get_statistics(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        return {
            'aggregations_performed': self._stats['aggregations_performed'],
            'total_symbols_tracked': len(self._exchange_liquidity),
            'total_exchanges_tracked': sum(len(exchanges) for exchanges in self._exchange_liquidity.values()),
            'predictions_made': self._stats['predictions_made'],
            'cost_estimates_made': self._stats['cost_estimates_made'],
            'cache_hit_rate': self._calculate_cache_hit_rate(),
        }

    def set_book_update_callback(self, callback: Callable[[AggregatedBook], None]) -> None:
        """设置订单簿更新回调"""
        self._on_book_update = callback

    def set_liquidity_alert_callback(self, callback: Callable[[str, str], None]) -> None:
        """设置流动性警报回调"""
        self._on_liquidity_alert = callback

    # ==================== 私有方法 ====================

    def _get_valid_exchanges(self, symbol: str) -> Dict[str, ExchangeLiquidity]:
        """获取有效的交易所数据"""
        exchanges = self._exchange_liquidity.get(symbol, {})
        now = datetime.now()

        valid = {}
        for exchange_id, liquidity in exchanges.items():
            # 检查数据新鲜度
            age_ms = (now - liquidity.timestamp).total_seconds() * 1000
            if age_ms > self.config.max_quote_age_ms:
                continue

            # 检查是否激活
            if not liquidity.is_active:
                continue

            # 检查深度
            if len(liquidity.bid_levels) < self.config.min_depth_levels or len(liquidity.ask_levels) < self.config.min_depth_levels:
                continue

            valid[exchange_id] = liquidity

        return valid

    def _aggregate_side(
        self,
        exchange_data: Dict[str, ExchangeLiquidity],
        side: str,
        max_levels: int
    ) -> List[Dict[str, Any]]:
        """聚合单边盘口"""
        price_levels = defaultdict(lambda: {'size': 0, 'exchanges': [], 'count': 0})

        for exchange_id, liquidity in exchange_data.items():
            levels = liquidity.bid_levels if side == 'bid' else liquidity.ask_levels

            for price, size in levels.items():
                price_levels[price]['size'] += size
                price_levels[price]['exchanges'].append(exchange_id)
                price_levels[price]['count'] += 1

        # 排序并转换为列表
        sorted_prices = sorted(price_levels.keys(), reverse=(side == 'bid'))

        result = []
        for price in sorted_prices[:max_levels]:
            level_data = price_levels[price]
            result.append({
                'price': price,
                'size': level_data['size'],
                'exchanges': level_data['exchanges'],
                'exchange_count': level_data['count'],
            })

        return result

    def _calculate_nbbo(self, exchange_data: Dict[str, ExchangeLiquidity]) -> Dict[str, Any]:
        """计算全国最优买卖价"""
        best_bid = None
        best_ask = None
        best_bid_size = 0
        best_ask_size = 0
        best_bid_exchanges = []
        best_ask_exchanges = []

        for exchange_id, liquidity in exchange_data.items():
            # 最优买价
            if liquidity.best_bid:
                if best_bid is None or liquidity.best_bid > best_bid:
                    best_bid = liquidity.best_bid
                    best_bid_size = liquidity.bid_levels.get(liquidity.best_bid, 0)
                    best_bid_exchanges = [exchange_id]
                elif liquidity.best_bid == best_bid:
                    best_bid_size += liquidity.bid_levels.get(liquidity.best_bid, 0)
                    best_bid_exchanges.append(exchange_id)

            # 最优卖价
            if liquidity.best_ask:
                if best_ask is None or liquidity.best_ask < best_ask:
                    best_ask = liquidity.best_ask
                    best_ask_size = liquidity.ask_levels.get(liquidity.best_ask, 0)
                    best_ask_exchanges = [exchange_id]
                elif liquidity.best_ask == best_ask:
                    best_ask_size += liquidity.ask_levels.get(liquidity.best_ask, 0)
                    best_ask_exchanges.append(exchange_id)

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'best_bid_size': best_bid_size,
            'best_ask_size': best_ask_size,
            'best_bid_exchanges': best_bid_exchanges,
            'best_ask_exchanges': best_ask_exchanges,
        }

    def _calculate_depth_in_range(
        self,
        bid_levels: List[Dict[str, Any]],
        ask_levels: List[Dict[str, Any]],
        mid_price: float,
        range_pct: float
    ) -> float:
        """计算价格范围内的深度"""
        if mid_price <= 0:
            return 0

        lower_bound = mid_price * (1 - range_pct)
        upper_bound = mid_price * (1 + range_pct)

        depth = 0

        # 买盘深度
        for level in bid_levels:
            if level['price'] >= lower_bound:
                depth += level['size']

        # 卖盘深度
        for level in ask_levels:
            if level['price'] <= upper_bound:
                depth += level['size']

        return depth

    def _calculate_liquidity_score(
        self,
        exchange_data: Dict[str, ExchangeLiquidity],
        bid_levels: List[Dict[str, Any]],
        ask_levels: List[Dict[str, Any]],
        nbbo: Dict[str, Any]
    ) -> Tuple[float, LiquidityScore]:
        """计算流动性综合评分"""
        # 价差评分
        spread_score = 0
        if nbbo.get('best_bid') and nbbo.get('best_ask'):
            mid = (nbbo['best_bid'] + nbbo['best_ask']) / 2
            spread_bps = (nbbo['best_ask'] - nbbo['best_bid']) / mid * 10000 if mid > 0 else 100
            spread_score = max(0, 100 - spread_bps) * self.config.spread_weight

        # 深度评分
        total_depth = sum(l['size'] for l in bid_levels) + sum(l['size'] for l in ask_levels)
        depth_score = min(100, total_depth / 100) * self.config.depth_weight

        # 恢复力评分
        resilience_scores = [ex.resilience_score for ex in exchange_data.values()]
        avg_resilience = np.mean(resilience_scores) if resilience_scores else 50
        resilience_score = avg_resilience * self.config.resilience_weight

        # 综合评分
        total_score = spread_score + depth_score + resilience_score

        # 确定等级
        if total_score >= 90:
            grade = LiquidityScore.EXCELLENT
        elif total_score >= 70:
            grade = LiquidityScore.GOOD
        elif total_score >= 50:
            grade = LiquidityScore.FAIR
        elif total_score >= 30:
            grade = LiquidityScore.POOR
        else:
            grade = LiquidityScore.VERY_POOR

        return total_score, grade

    def _calculate_weighted_spread(self, exchange_data: Dict[str, ExchangeLiquidity]) -> float:
        """计算加权平均价差"""
        total_weight = 0
        weighted_sum = 0

        for liquidity in exchange_data.values():
            weight = liquidity.total_bid_volume + liquidity.total_ask_volume
            if weight > 0:
                weighted_sum += liquidity.spread_bps * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def _calculate_price_impact(
        self,
        levels: List[Dict[str, Any]],
        quantity: float
    ) -> Tuple[float, float, int]:
        """计算价格冲击"""
        remaining = quantity
        total_cost = 0
        worst_price = levels[0]['price'] if levels else 0
        consumed_levels = 0

        for level in levels:
            if remaining <= 0:
                break

            consume_qty = min(remaining, level['size'])
            total_cost += consume_qty * level['price']
            worst_price = level['price']
            remaining -= consume_qty
            consumed_levels += 1

        # 如果流动性不足，估算剩余部分
        if remaining > 0 and levels:
            # 假设价格继续恶化
            extrapolated_price = worst_price * 1.01
            total_cost += remaining * extrapolated_price
            worst_price = extrapolated_price

        avg_price = total_cost / quantity if quantity > 0 else 0

        return avg_price, worst_price, consumed_levels

    def _estimate_slippage(
        self,
        market_depth_ratio: float,
        urgency: float,
        liquidity_score: float
    ) -> float:
        """估算滑点"""
        # 基础滑点(基于市场深度比例)
        base_slippage = market_depth_ratio * 50  # 1%的市场规模 = 0.5基点

        # 紧急度调整
        urgency_factor = 1 + urgency * 0.5

        # 流动性质量调整
        quality_factor = 2 - (liquidity_score / 100)

        return base_slippage * urgency_factor * quality_factor

    def _suggest_execution_method(
        self,
        quantity: float,
        market_depth_ratio: float,
        urgency: float,
        liquidity_score: float
    ) -> Tuple[str, int]:
        """推荐执行方法"""
        # 小订单 + 高紧急度 = 市价单
        if market_depth_ratio < 0.01 and urgency > 0.7:
            return "market", 1

        # 大订单 = 需要拆分
        if market_depth_ratio > 0.05:
            suggested_split = max(3, min(10, int(market_depth_ratio * 100)))
            if urgency < 0.3:
                return "limit", suggested_split
            elif urgency < 0.7:
                return "twap", suggested_split
            else:
                return "vwap", suggested_split

        # 中等订单
        if urgency > 0.7:
            return "ioc", max(2, int(market_depth_ratio * 50))
        elif urgency < 0.3:
            return "limit", 1
        else:
            return "limit", max(2, int(market_depth_ratio * 30))

    def _suggest_exchanges(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> List[str]:
        """推荐交易所"""
        exchange_data = self._get_valid_exchanges(symbol)

        # 按流动性评分排序
        scored_exchanges = []
        for exchange_id, liquidity in exchange_data.items():
            liquidity_size = liquidity.total_ask_volume if side.lower() == 'buy' else liquidity.total_bid_volume
            score = liquidity.depth_score * 0.6 + (liquidity_size / quantity if quantity > 0 else 0) * 40
            scored_exchanges.append((exchange_id, score))

        scored_exchanges.sort(key=lambda x: x[1], reverse=True)

        # 返回前5个
        return [ex_id for ex_id, _ in scored_exchanges[:5]]

    def _calculate_cost_confidence(
        self,
        book: AggregatedBook,
        consumed_levels: int,
        total_levels: int
    ) -> float:
        """计算成本估算置信度"""
        # 数据质量因素
        data_quality = min(1.0, book.total_exchanges / 5)

        # 深度覆盖因素
        depth_coverage = 1.0 if consumed_levels < total_levels * 0.5 else 0.7

        # 流动性评分因素
        liquidity_factor = book.liquidity_score / 100

        return data_quality * depth_coverage * liquidity_factor

    def _create_market_strategy(
        self,
        symbol: str,
        side: str,
        quantity: float,
        book: AggregatedBook,
        cost: LiquidityCost,
        urgency: float
    ) -> ExecutionStrategy:
        """创建市价单策略"""
        # 推荐场所(按流动性排序)
        venues = []
        exchanges = self._suggest_exchanges(symbol, side, quantity)
        for ex in exchanges[:3]:
            venues.append({
                'exchange': ex,
                'allocation': quantity / min(3, len(exchanges)),
                'reason': 'High liquidity'
            })

        # 评分
        score = 70 + urgency * 20 - (cost.total_cost_bps / 10)

        return ExecutionStrategy(
            strategy_name="Market Order - Immediate Execution",
            description="Execute immediately at market price across multiple venues",
            recommended_venues=venues,
            total_allocation=quantity,
            execution_method="market",
            urgency_level="high",
            expected_fill_rate=0.95,
            expected_cost_bps=cost.total_cost_bps,
            expected_duration_seconds=5,
            price_risk=0.7,
            execution_risk=0.2,
            strategy_score=max(0, min(100, score)),
        )

    def _create_limit_strategy(
        self,
        symbol: str,
        side: str,
        quantity: float,
        book: AggregatedBook,
        cost: LiquidityCost,
        urgency: float
    ) -> ExecutionStrategy:
        """创建限价单策略"""
        venues = []
        exchanges = self._suggest_exchanges(symbol, side, quantity)
        for ex in exchanges[:2]:
            venues.append({
                'exchange': ex,
                'allocation': quantity / min(2, len(exchanges)),
                'reason': 'Best price available'
            })

        # 限价单成本更低但时间更长
        score = 80 - urgency * 30 - (cost.spread_cost_bps / 5)

        return ExecutionStrategy(
            strategy_name="Limit Order - Passive Execution",
            description="Post limit orders at best price, wait for fill",
            recommended_venues=venues,
            total_allocation=quantity,
            execution_method="limit",
            urgency_level="low",
            expected_fill_rate=0.7,
            expected_cost_bps=cost.spread_cost_bps * 0.5,
            expected_duration_seconds=300,
            price_risk=0.4,
            execution_risk=0.6,
            strategy_score=max(0, min(100, score)),
        )

    def _create_iceberg_strategy(
        self,
        symbol: str,
        side: str,
        quantity: float,
        book: AggregatedBook,
        cost: LiquidityCost,
        urgency: float
    ) -> ExecutionStrategy:
        """创建冰山单策略"""
        venues = []
        exchanges = self._suggest_exchanges(symbol, side, quantity)
        for ex in exchanges[:2]:
            venues.append({
                'exchange': ex,
                'allocation': quantity / min(2, len(exchanges)),
                'reason': 'Hidden liquidity available'
            })

        score = 75 - (cost.impact_cost_bps / 3)

        return ExecutionStrategy(
            strategy_name="Iceberg Order - Hidden Size",
            description="Hide order size to minimize market impact",
            recommended_venues=venues,
            total_allocation=quantity,
            execution_method="iceberg",
            urgency_level="medium",
            expected_fill_rate=0.8,
            expected_cost_bps=cost.total_cost_bps * 0.8,
            expected_duration_seconds=120,
            price_risk=0.5,
            execution_risk=0.4,
            strategy_score=max(0, min(100, score)),
        )

    def _create_twap_strategy(
        self,
        symbol: str,
        side: str,
        quantity: float,
        book: AggregatedBook,
        cost: LiquidityCost,
        urgency: float
    ) -> ExecutionStrategy:
        """创建TWAP策略"""
        venues = []
        exchanges = self._suggest_exchanges(symbol, side, quantity)
        slices = max(5, int(quantity / book.depth_1_percent * 10))

        for ex in exchanges[:3]:
            venues.append({
                'exchange': ex,
                'allocation': quantity / min(3, len(exchanges)),
                'reason': 'Time-distributed execution'
            })

        score = 70 + (1 - urgency) * 20 - (cost.impact_cost_bps / 4)

        return ExecutionStrategy(
            strategy_name="TWAP - Time Weighted Average Price",
            description=f"Split order into {slices} time-distributed slices",
            recommended_venues=venues,
            total_allocation=quantity,
            execution_method="twap",
            urgency_level="low" if urgency < 0.3 else "medium",
            expected_fill_rate=0.85,
            expected_cost_bps=cost.total_cost_bps * 0.7,
            expected_duration_seconds=slices * 30,
            price_risk=0.5,
            execution_risk=0.3,
            strategy_score=max(0, min(100, score)),
        )

    def _create_vwap_strategy(
        self,
        symbol: str,
        side: str,
        quantity: float,
        book: AggregatedBook,
        cost: LiquidityCost,
        urgency: float
    ) -> ExecutionStrategy:
        """创建VWAP策略"""
        venues = []
        exchanges = self._suggest_exchanges(symbol, side, quantity)

        for ex in exchanges[:3]:
            venues.append({
                'exchange': ex,
                'allocation': quantity / min(3, len(exchanges)),
                'reason': 'Volume-distributed execution'
            })

        score = 75 + urgency * 10 - (cost.impact_cost_bps / 3)

        return ExecutionStrategy(
            strategy_name="VWAP - Volume Weighted Average Price",
            description="Execute in proportion to market volume patterns",
            recommended_venues=venues,
            total_allocation=quantity,
            execution_method="vwap",
            urgency_level="medium",
            expected_fill_rate=0.9,
            expected_cost_bps=cost.total_cost_bps * 0.75,
            expected_duration_seconds=600,
            price_risk=0.45,
            execution_risk=0.25,
            strategy_score=max(0, min(100, score)),
        )

    def _create_dark_pool_strategy(
        self,
        symbol: str,
        side: str,
        quantity: float,
        book: AggregatedBook,
        cost: LiquidityCost,
        urgency: float
    ) -> ExecutionStrategy:
        """创建暗池策略"""
        dark_pools = self.get_dark_pool_opportunities(symbol)

        venues = []
        for pool in dark_pools[:3]:
            venues.append({
                'exchange': pool.pool_id,
                'allocation': min(quantity * 0.3, pool.indicative_size),
                'reason': f'Dark pool, {pool.match_probability:.0%} match probability'
            })

        # 剩余部分在公开市场
        allocated_dark = sum(v['allocation'] for v in venues)
        if allocated_dark < quantity:
            exchanges = self._suggest_exchanges(symbol, side, quantity - allocated_dark)
            for ex in exchanges[:2]:
                venues.append({
                    'exchange': ex,
                    'allocation': (quantity - allocated_dark) / min(2, len(exchanges)),
                    'reason': 'Lit market for remaining'
                })

        # 暗池可以获得价格改善
        avg_improvement = np.mean([p.avg_price_improvement_bps for p in dark_pools]) if dark_pools else 0

        score = 85 - urgency * 15 + avg_improvement / 2

        return ExecutionStrategy(
            strategy_name="Dark Pool Priority",
            description="Execute in dark pools first for price improvement",
            recommended_venues=venues,
            total_allocation=quantity,
            execution_method="dark_first",
            urgency_level="low",
            expected_fill_rate=0.75,
            expected_cost_bps=max(0, cost.total_cost_bps - avg_improvement),
            expected_duration_seconds=180,
            price_risk=0.3,
            execution_risk=0.5,
            strategy_score=max(0, min(100, score)),
        )

    def _has_dark_liquidity(self, symbol: str) -> bool:
        """检查是否有暗池流动性"""
        dark_pools = self._dark_pools.get(symbol, {})
        return any(pool.is_available for pool in dark_pools.values())

    def _find_optimal_execution_window(
        self,
        history: List[Dict[str, Any]],
        horizon_seconds: int
    ) -> Optional[Tuple[datetime, datetime]]:
        """查找最佳执行窗口"""
        if len(history) < 20:
            return None

        # 找到流动性最好的时段(价差最小、深度最大)
        best_idx = 0
        best_score = -float('inf')

        for i in range(len(history) - 5):
            window = history[i:i+5]
            avg_spread = np.mean([h['spread_bps'] for h in window])
            avg_depth = np.mean([h['depth'] for h in window])

            # 评分: 低价差 + 高深度
            score = avg_depth / (avg_spread + 1)

            if score > best_score:
                best_score = score
                best_idx = i

        # 投影到未来
        now = datetime.now()
        start_time = now + timedelta(seconds=best_idx * 60)
        end_time = start_time + timedelta(seconds=min(horizon_seconds, 300))

        return (start_time, end_time)

    def _detect_liquidity_events(self, history: List[Dict[str, Any]]) -> List[str]:
        """检测影响流动性的事件"""
        events = []

        if len(history) < 10:
            return events

        recent = history[-10:]

        # 检测价差突然扩大
        recent_spreads = [h['spread_bps'] for h in recent]
        if recent_spreads[-1] > np.mean(recent_spreads[:-1]) * 1.5:
            events.append("Spread widening detected")

        # 检测深度突然下降
        recent_depths = [h['depth'] for h in recent]
        if recent_depths[-1] < np.mean(recent_depths[:-1]) * 0.7:
            events.append("Liquidity drain detected")

        # 检测波动率上升
        if len(recent_spreads) > 2:
            volatility = np.std(recent_spreads) / np.mean(recent_spreads) if np.mean(recent_spreads) > 0 else 0
            if volatility > 0.3:
                events.append("High volatility period")

        return events

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        if self._stats['aggregations_performed'] == 0:
            return 0

        # 简化计算: 假设缓存有效期内的请求都命中
        cached_symbols = len(self._cache_timestamps)
        total_requests = self._stats['aggregations_performed']

        # 粗略估计
        hit_rate = min(0.9, cached_symbols / max(1, total_requests * 0.1))

        return hit_rate


# ==================== 工厂函数 ====================

def create_liquidity_aggregator(
    min_exchanges: int = 2,
    include_dark_pools: bool = True,
    enable_prediction: bool = True,
    **kwargs
) -> LiquidityAggregator:
    """
    创建流动性聚合器

    Args:
        min_exchanges: 最小交易所数量
        include_dark_pools: 是否包含暗池
        enable_prediction: 是否启用预测
        **kwargs: 其他配置参数

    Returns:
        LiquidityAggregator实例
    """
    config = AggregatorConfig(
        min_exchanges=min_exchanges,
        include_dark_pools=include_dark_pools,
        enable_prediction=enable_prediction,
        **{k: v for k, v in kwargs.items() if hasattr(AggregatorConfig, k)}
    )
    return LiquidityAggregator(config)


def create_exchange_liquidity(
    exchange_id: str,
    symbol: str,
    bid_levels: Dict[float, float],
    ask_levels: Dict[float, float],
    **kwargs
) -> ExchangeLiquidity:
    """创建交易所流动性数据"""
    return ExchangeLiquidity(
        exchange_id=exchange_id,
        symbol=symbol,
        timestamp=kwargs.get('timestamp', datetime.now()),
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        displayed_liquidity=kwargs.get('displayed_liquidity', sum(bid_levels.values()) + sum(ask_levels.values())),
        spread_bps=kwargs.get('spread_bps', 0),
        depth_score=kwargs.get('depth_score', 50),
        resilience_score=kwargs.get('resilience_score', 50),
    )


def create_dark_pool_liquidity(
    pool_id: str,
    symbol: str,
    indicative_size: float,
    **kwargs
) -> DarkPoolLiquidity:
    """创建暗池流动性数据"""
    return DarkPoolLiquidity(
        pool_id=pool_id,
        symbol=symbol,
        timestamp=kwargs.get('timestamp', datetime.now()),
        indicative_size=indicative_size,
        min_block_size=kwargs.get('min_block_size', indicative_size * 0.5),
        match_probability=kwargs.get('match_probability', 0.6),
        avg_price_improvement_bps=kwargs.get('avg_price_improvement_bps', 1.0),
        fee_bps=kwargs.get('fee_bps', 0.5),
    )
