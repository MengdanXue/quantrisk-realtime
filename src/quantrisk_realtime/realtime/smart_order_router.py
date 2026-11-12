"""
智能订单路由模块 (Smart Order Router - SOR)

提供智能订单路由功能:
- 多交易所/执行场所路由
- 最佳执行选择算法
- 流动性聚合
- 暗池接入
- 订单拆分优化
- 实时市场监控
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class VenueType(str, Enum):
    """执行场所类型"""
    EXCHANGE = "exchange"           # 交易所
    DARK_POOL = "dark_pool"         # 暗池
    ECN = "ecn"                     # 电子通讯网络
    ATS = "ats"                     # 替代交易系统
    BROKER = "broker"              # 经纪商内部化
    MARKET_MAKER = "market_maker"  # 做市商


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"                    # 立即成交或取消
    FOK = "fok"                    # 全部成交或取消
    GTC = "gtc"                    # 撤销前有效
    GTD = "gtd"                    # 指定日期前有效


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class RouteStrategy(str, Enum):
    """路由策略"""
    BEST_PRICE = "best_price"           # 最优价格
    BEST_EXECUTION = "best_execution"   # 最佳执行
    LIQUIDITY_SEEKING = "liquidity"     # 寻找流动性
    MINIMIZE_IMPACT = "min_impact"      # 最小化市场影响
    DARK_FIRST = "dark_first"           # 暗池优先
    SPRAY = "spray"                     # 喷射到多个场所
    SERIAL = "serial"                   # 串行路由


class RouteStatus(str, Enum):
    """路由状态"""
    PENDING = "pending"
    ROUTED = "routed"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


# ==================== 数据类 ====================

@dataclass
class Venue:
    """执行场所"""
    venue_id: str
    name: str
    venue_type: VenueType

    # 费用结构
    maker_fee_bps: float = 0            # 做市商费率(基点)
    taker_fee_bps: float = 0            # 吃单费率(基点)
    rebate_bps: float = 0               # 返佣(基点)

    # 能力
    min_order_size: float = 1           # 最小订单大小
    max_order_size: float = float('inf') # 最大订单大小
    supports_dark: bool = False         # 支持暗池
    supports_midpoint: bool = False     # 支持中间价

    # 性能指标
    avg_latency_ms: float = 1           # 平均延迟(毫秒)
    fill_rate: float = 0.8              # 成交率
    rejection_rate: float = 0.01        # 拒单率

    # 状态
    is_active: bool = True
    priority: int = 0                   # 优先级(越高越优先)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'venue_id': self.venue_id,
            'name': self.name,
            'venue_type': self.venue_type.value,
            'maker_fee_bps': self.maker_fee_bps,
            'taker_fee_bps': self.taker_fee_bps,
            'rebate_bps': self.rebate_bps,
            'fill_rate': self.fill_rate,
            'avg_latency_ms': self.avg_latency_ms,
            'is_active': self.is_active,
            'priority': self.priority,
        }


@dataclass
class VenueQuote:
    """场所报价"""
    venue_id: str
    symbol: str
    timestamp: datetime

    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    # 隐含流动性估计
    hidden_liquidity: float = 0

    # 报价质量
    is_stale: bool = False
    age_ms: float = 0

    @property
    def mid_price(self) -> float:
        """中间价"""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread_bps(self) -> float:
        """价差(基点)"""
        if self.mid_price <= 0:
            return 0
        return (self.ask_price - self.bid_price) / self.mid_price * 10000

    def to_dict(self) -> Dict[str, Any]:
        return {
            'venue_id': self.venue_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bid_price': self.bid_price,
            'bid_size': self.bid_size,
            'ask_price': self.ask_price,
            'ask_size': self.ask_size,
            'mid_price': self.mid_price,
            'spread_bps': round(self.spread_bps, 2),
        }


@dataclass
class RouteOrder:
    """路由订单"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None

    # 路由配置
    strategy: RouteStrategy = RouteStrategy.BEST_EXECUTION
    max_venues: int = 5                  # 最大场所数
    min_fill_rate: float = 0.5           # 最小期望成交率
    urgency: float = 0.5                 # 紧迫度 0-1

    # 约束
    allowed_venues: List[str] = field(default_factory=list)
    excluded_venues: List[str] = field(default_factory=list)
    max_dark_pct: float = 0.3            # 最大暗池比例

    # 状态
    status: RouteStatus = RouteStatus.PENDING
    filled_quantity: float = 0
    avg_fill_price: float = 0
    total_fees: float = 0

    # 路由分配
    venue_allocations: Dict[str, float] = field(default_factory=dict)
    child_orders: List[str] = field(default_factory=list)

    # 时间
    created_at: datetime = field(default_factory=datetime.now)
    routed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def remaining_quantity(self) -> float:
        """剩余数量"""
        return self.quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        """成交率"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0

    @property
    def is_complete(self) -> bool:
        """是否完成"""
        return self.status in [RouteStatus.FILLED, RouteStatus.REJECTED, RouteStatus.CANCELLED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'strategy': self.strategy.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'fill_rate': round(self.fill_rate * 100, 2),
            'avg_fill_price': round(self.avg_fill_price, 4),
            'total_fees': round(self.total_fees, 2),
            'venue_allocations': self.venue_allocations,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class ChildOrder:
    """子订单(发送到单个场所)"""
    child_id: str
    parent_id: str
    venue_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None

    # 状态
    status: RouteStatus = RouteStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    fees: float = 0

    # 时间
    sent_at: Optional[datetime] = None
    acked_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # 执行信息
    venue_order_id: Optional[str] = None
    reject_reason: Optional[str] = None

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            'child_id': self.child_id,
            'parent_id': self.parent_id,
            'venue_id': self.venue_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'status': self.status.value,
            'filled_price': self.filled_price,
            'fees': self.fees,
        }


@dataclass
class RouteDecision:
    """路由决策"""
    venue_id: str
    allocation: float                # 分配数量
    expected_fill_rate: float        # 期望成交率
    expected_cost_bps: float         # 期望成本(基点)
    reasoning: str                   # 决策原因

    def to_dict(self) -> Dict[str, Any]:
        return {
            'venue_id': self.venue_id,
            'allocation': self.allocation,
            'expected_fill_rate': round(self.expected_fill_rate * 100, 2),
            'expected_cost_bps': round(self.expected_cost_bps, 2),
            'reasoning': self.reasoning,
        }


@dataclass
class ExecutionQuality:
    """执行质量分析"""
    order_id: str
    symbol: str
    side: OrderSide

    # 基准价格
    arrival_price: float
    decision_price: float

    # 实际执行
    avg_fill_price: float
    total_quantity: float
    filled_quantity: float

    # 成本分析
    implementation_shortfall_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    venue_fees_bps: float

    # 场所分析
    venue_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 评分
    execution_score: float = 0       # 0-100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'arrival_price': self.arrival_price,
            'avg_fill_price': self.avg_fill_price,
            'fill_rate': round(self.filled_quantity / self.total_quantity * 100, 2) if self.total_quantity > 0 else 0,
            'implementation_shortfall_bps': round(self.implementation_shortfall_bps, 2),
            'market_impact_bps': round(self.market_impact_bps, 2),
            'timing_cost_bps': round(self.timing_cost_bps, 2),
            'venue_fees_bps': round(self.venue_fees_bps, 2),
            'total_cost_bps': round(
                self.implementation_shortfall_bps + self.venue_fees_bps, 2
            ),
            'execution_score': round(self.execution_score, 1),
            'venue_breakdown': self.venue_breakdown,
        }


@dataclass
class SORConfig:
    """SOR配置"""
    # 路由配置
    default_strategy: RouteStrategy = RouteStrategy.BEST_EXECUTION
    max_venues_per_order: int = 5
    min_venue_allocation: float = 100    # 最小分配到单个场所

    # 暗池配置
    dark_pool_enabled: bool = True
    max_dark_pool_pct: float = 0.3       # 最大暗池比例
    dark_pool_min_size: float = 1000     # 暗池最小订单

    # 超时配置
    quote_timeout_ms: float = 100        # 报价超时
    order_timeout_seconds: float = 60    # 订单超时

    # 费用配置
    include_fees_in_routing: bool = True

    # 风控
    max_order_value: float = 1_000_000   # 最大订单金额


# ==================== 路由策略 ====================

class RoutingStrategy(ABC):
    """路由策略基类"""

    @property
    @abstractmethod
    def strategy_type(self) -> RouteStrategy:
        """策略类型"""
        pass

    @abstractmethod
    def calculate_route(
        self,
        order: RouteOrder,
        venues: List[Venue],
        quotes: Dict[str, VenueQuote]
    ) -> List[RouteDecision]:
        """
        计算路由决策

        Args:
            order: 路由订单
            venues: 可用场所列表
            quotes: 场所报价字典

        Returns:
            路由决策列表
        """
        pass


class BestPriceStrategy(RoutingStrategy):
    """最优价格策略"""

    @property
    def strategy_type(self) -> RouteStrategy:
        return RouteStrategy.BEST_PRICE

    def calculate_route(
        self,
        order: RouteOrder,
        venues: List[Venue],
        quotes: Dict[str, VenueQuote]
    ) -> List[RouteDecision]:
        """按价格排序，优先分配到最优价格场所"""
        decisions = []
        remaining = order.remaining_quantity

        # 按价格排序报价
        sorted_quotes = sorted(
            [(v, q) for v, q in quotes.items() if v in [ven.venue_id for ven in venues]],
            key=lambda x: x[1].ask_price if order.side == OrderSide.BUY else -x[1].bid_price
        )

        for venue_id, quote in sorted_quotes:
            if remaining <= 0:
                break

            venue = next((v for v in venues if v.venue_id == venue_id), None)
            if not venue or not venue.is_active:
                continue

            # 可用数量
            available = quote.ask_size if order.side == OrderSide.BUY else quote.bid_size
            allocation = min(remaining, available, venue.max_order_size)

            if allocation < venue.min_order_size:
                continue

            price = quote.ask_price if order.side == OrderSide.BUY else quote.bid_price
            cost_bps = venue.taker_fee_bps

            decisions.append(RouteDecision(
                venue_id=venue_id,
                allocation=allocation,
                expected_fill_rate=venue.fill_rate,
                expected_cost_bps=cost_bps,
                reasoning=f"Best price at {price:.4f}",
            ))

            remaining -= allocation

        return decisions


class BestExecutionStrategy(RoutingStrategy):
    """最佳执行策略(综合考虑价格、成本、成交率)"""

    @property
    def strategy_type(self) -> RouteStrategy:
        return RouteStrategy.BEST_EXECUTION

    def calculate_route(
        self,
        order: RouteOrder,
        venues: List[Venue],
        quotes: Dict[str, VenueQuote]
    ) -> List[RouteDecision]:
        """综合评分后分配"""
        decisions = []
        remaining = order.remaining_quantity

        # 计算每个场所的综合得分
        venue_scores = []
        for venue in venues:
            if venue.venue_id not in quotes or not venue.is_active:
                continue

            quote = quotes[venue.venue_id]
            if quote.is_stale:
                continue

            # 价格得分 (0-40分)
            price = quote.ask_price if order.side == OrderSide.BUY else quote.bid_price
            mid_price = quote.mid_price
            price_score = 40 * (1 - abs(price - mid_price) / mid_price * 100) if mid_price > 0 else 0
            price_score = max(0, min(40, price_score))

            # 流动性得分 (0-30分)
            available = quote.ask_size if order.side == OrderSide.BUY else quote.bid_size
            liquidity_score = 30 * min(available / remaining, 1) if remaining > 0 else 0

            # 成本得分 (0-20分)
            cost_bps = venue.taker_fee_bps - venue.rebate_bps
            cost_score = 20 * max(0, 1 - cost_bps / 10)

            # 可靠性得分 (0-10分)
            reliability_score = 10 * venue.fill_rate * (1 - venue.rejection_rate)

            total_score = price_score + liquidity_score + cost_score + reliability_score

            venue_scores.append((venue, quote, total_score))

        # 按得分排序
        venue_scores.sort(key=lambda x: x[2], reverse=True)

        for venue, quote, score in venue_scores:
            if remaining <= 0:
                break

            available = quote.ask_size if order.side == OrderSide.BUY else quote.bid_size
            allocation = min(remaining, available, venue.max_order_size)

            if allocation < venue.min_order_size:
                continue

            decisions.append(RouteDecision(
                venue_id=venue.venue_id,
                allocation=allocation,
                expected_fill_rate=venue.fill_rate,
                expected_cost_bps=venue.taker_fee_bps,
                reasoning=f"Score: {score:.1f}/100",
            ))

            remaining -= allocation

        return decisions


class LiquiditySeekingStrategy(RoutingStrategy):
    """流动性寻找策略"""

    @property
    def strategy_type(self) -> RouteStrategy:
        return RouteStrategy.LIQUIDITY_SEEKING

    def calculate_route(
        self,
        order: RouteOrder,
        venues: List[Venue],
        quotes: Dict[str, VenueQuote]
    ) -> List[RouteDecision]:
        """优先分配到流动性最好的场所"""
        decisions = []
        remaining = order.remaining_quantity

        # 按流动性排序
        liquidity_ranking = []
        for venue in venues:
            if venue.venue_id not in quotes or not venue.is_active:
                continue

            quote = quotes[venue.venue_id]
            available = quote.ask_size if order.side == OrderSide.BUY else quote.bid_size
            # 考虑隐含流动性
            total_liquidity = available + quote.hidden_liquidity

            liquidity_ranking.append((venue, quote, total_liquidity))

        liquidity_ranking.sort(key=lambda x: x[2], reverse=True)

        for venue, quote, liquidity in liquidity_ranking:
            if remaining <= 0:
                break

            available = quote.ask_size if order.side == OrderSide.BUY else quote.bid_size
            allocation = min(remaining, available * 0.8, venue.max_order_size)

            if allocation < venue.min_order_size:
                continue

            decisions.append(RouteDecision(
                venue_id=venue.venue_id,
                allocation=allocation,
                expected_fill_rate=venue.fill_rate,
                expected_cost_bps=venue.taker_fee_bps,
                reasoning=f"Liquidity: {liquidity:.0f}",
            ))

            remaining -= allocation

        return decisions


class DarkFirstStrategy(RoutingStrategy):
    """暗池优先策略"""

    @property
    def strategy_type(self) -> RouteStrategy:
        return RouteStrategy.DARK_FIRST

    def calculate_route(
        self,
        order: RouteOrder,
        venues: List[Venue],
        quotes: Dict[str, VenueQuote]
    ) -> List[RouteDecision]:
        """优先路由到暗池，然后是公开市场"""
        decisions = []
        remaining = order.remaining_quantity
        max_dark = order.quantity * order.max_dark_pct
        dark_allocated = 0

        # 分离暗池和公开场所
        dark_venues = [v for v in venues if v.venue_type == VenueType.DARK_POOL and v.is_active]
        lit_venues = [v for v in venues if v.venue_type != VenueType.DARK_POOL and v.is_active]

        # 首先分配到暗池
        for venue in dark_venues:
            if remaining <= 0 or dark_allocated >= max_dark:
                break

            if venue.venue_id not in quotes:
                continue

            quote = quotes[venue.venue_id]
            allocation = min(
                remaining,
                max_dark - dark_allocated,
                quote.ask_size if order.side == OrderSide.BUY else quote.bid_size
            )

            if allocation < venue.min_order_size:
                continue

            decisions.append(RouteDecision(
                venue_id=venue.venue_id,
                allocation=allocation,
                expected_fill_rate=venue.fill_rate * 0.6,  # 暗池成交率通常较低
                expected_cost_bps=venue.taker_fee_bps,
                reasoning="Dark pool priority",
            ))

            remaining -= allocation
            dark_allocated += allocation

        # 然后分配到公开市场
        for venue in lit_venues:
            if remaining <= 0:
                break

            if venue.venue_id not in quotes:
                continue

            quote = quotes[venue.venue_id]
            available = quote.ask_size if order.side == OrderSide.BUY else quote.bid_size
            allocation = min(remaining, available)

            if allocation < venue.min_order_size:
                continue

            decisions.append(RouteDecision(
                venue_id=venue.venue_id,
                allocation=allocation,
                expected_fill_rate=venue.fill_rate,
                expected_cost_bps=venue.taker_fee_bps,
                reasoning="Lit market fallback",
            ))

            remaining -= allocation

        return decisions


class SprayStrategy(RoutingStrategy):
    """喷射策略(同时发送到多个场所)"""

    @property
    def strategy_type(self) -> RouteStrategy:
        return RouteStrategy.SPRAY

    def calculate_route(
        self,
        order: RouteOrder,
        venues: List[Venue],
        quotes: Dict[str, VenueQuote]
    ) -> List[RouteDecision]:
        """将订单分散到多个场所以提高成交概率"""
        decisions = []

        active_venues = [v for v in venues if v.is_active and v.venue_id in quotes]
        if not active_venues:
            return decisions

        # 限制场所数量
        max_venues = min(order.max_venues, len(active_venues))
        selected_venues = active_venues[:max_venues]

        # 均匀分配
        base_allocation = order.remaining_quantity / len(selected_venues)

        for venue in selected_venues:
            quote = quotes[venue.venue_id]
            available = quote.ask_size if order.side == OrderSide.BUY else quote.bid_size

            # 根据可用流动性调整分配
            allocation = min(base_allocation, available)

            if allocation < venue.min_order_size:
                continue

            decisions.append(RouteDecision(
                venue_id=venue.venue_id,
                allocation=allocation,
                expected_fill_rate=venue.fill_rate,
                expected_cost_bps=venue.taker_fee_bps,
                reasoning=f"Spray to {len(selected_venues)} venues",
            ))

        return decisions


# ==================== 智能订单路由引擎 ====================

class SmartOrderRouter:
    """智能订单路由引擎"""

    def __init__(self, config: Optional[SORConfig] = None):
        self.config = config or SORConfig()

        # 场所管理
        self._venues: Dict[str, Venue] = {}
        self._quotes: Dict[str, Dict[str, VenueQuote]] = defaultdict(dict)  # symbol -> venue -> quote

        # 路由策略
        self._strategies: Dict[RouteStrategy, RoutingStrategy] = {
            RouteStrategy.BEST_PRICE: BestPriceStrategy(),
            RouteStrategy.BEST_EXECUTION: BestExecutionStrategy(),
            RouteStrategy.LIQUIDITY_SEEKING: LiquiditySeekingStrategy(),
            RouteStrategy.DARK_FIRST: DarkFirstStrategy(),
            RouteStrategy.SPRAY: SprayStrategy(),
        }

        # 订单管理
        self._orders: Dict[str, RouteOrder] = {}
        self._child_orders: Dict[str, ChildOrder] = {}

        # 回调
        self._on_child_created: Optional[Callable[[ChildOrder], None]] = None
        self._on_order_completed: Optional[Callable[[RouteOrder], None]] = None

        # 统计
        self._stats = {
            'orders_routed': 0,
            'total_volume': 0,
            'total_filled': 0,
            'by_venue': defaultdict(lambda: {'volume': 0, 'filled': 0, 'fees': 0}),
            'by_strategy': defaultdict(lambda: {'count': 0, 'fill_rate': 0}),
        }

    def register_venue(self, venue: Venue) -> None:
        """注册执行场所"""
        self._venues[venue.venue_id] = venue
        logger.info(f"Venue registered: {venue.venue_id} ({venue.venue_type.value})")

    def unregister_venue(self, venue_id: str) -> None:
        """注销执行场所"""
        if venue_id in self._venues:
            del self._venues[venue_id]
            logger.info(f"Venue unregistered: {venue_id}")

    def update_quote(self, quote: VenueQuote) -> None:
        """更新场所报价"""
        self._quotes[quote.symbol][quote.venue_id] = quote

    def get_consolidated_book(self, symbol: str) -> Dict[str, Any]:
        """
        获取合并订单簿

        Args:
            symbol: 标的代码

        Returns:
            合并后的订单簿
        """
        quotes = self._quotes.get(symbol, {})

        if not quotes:
            return {'symbol': symbol, 'bids': [], 'asks': [], 'venues': 0}

        # 收集所有报价
        all_bids = []
        all_asks = []

        for venue_id, quote in quotes.items():
            if quote.is_stale:
                continue

            venue = self._venues.get(venue_id)
            if not venue or not venue.is_active:
                continue

            all_bids.append({
                'price': quote.bid_price,
                'size': quote.bid_size,
                'venue': venue_id,
            })
            all_asks.append({
                'price': quote.ask_price,
                'size': quote.ask_size,
                'venue': venue_id,
            })

        # 排序
        all_bids.sort(key=lambda x: x['price'], reverse=True)
        all_asks.sort(key=lambda x: x['price'])

        # 计算NBBO
        nbbo = {}
        if all_bids:
            nbbo['best_bid'] = all_bids[0]['price']
            nbbo['best_bid_venue'] = all_bids[0]['venue']
        if all_asks:
            nbbo['best_ask'] = all_asks[0]['price']
            nbbo['best_ask_venue'] = all_asks[0]['venue']

        return {
            'symbol': symbol,
            'bids': all_bids[:10],
            'asks': all_asks[:10],
            'venues': len(quotes),
            'nbbo': nbbo,
            'timestamp': datetime.now().isoformat(),
        }

    async def route_order(self, order: RouteOrder) -> RouteOrder:
        """
        路由订单

        Args:
            order: 路由订单

        Returns:
            更新后的订单
        """
        # 验证订单
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        if order.limit_price and order.limit_price <= 0:
            raise ValueError("Limit price must be positive")

        # 获取可用场所
        available_venues = self._get_available_venues(order)
        if not available_venues:
            order.status = RouteStatus.REJECTED
            raise ValueError("No available venues for routing")

        # 获取报价
        quotes = self._quotes.get(order.symbol, {})
        if not quotes:
            order.status = RouteStatus.REJECTED
            raise ValueError(f"No quotes available for {order.symbol}")

        # 获取路由策略
        strategy = self._strategies.get(order.strategy)
        if not strategy:
            strategy = self._strategies[self.config.default_strategy]

        # 计算路由决策
        decisions = strategy.calculate_route(order, available_venues, quotes)

        if not decisions:
            order.status = RouteStatus.REJECTED
            raise ValueError("No routing decisions generated")

        # 创建子订单
        for decision in decisions:
            child = await self._create_child_order(order, decision)
            order.child_orders.append(child.child_id)
            order.venue_allocations[decision.venue_id] = decision.allocation

        order.status = RouteStatus.ROUTED
        order.routed_at = datetime.now()

        # 保存订单
        self._orders[order.order_id] = order

        # 更新统计
        self._stats['orders_routed'] += 1
        self._stats['total_volume'] += order.quantity
        self._stats['by_strategy'][order.strategy.value]['count'] += 1

        logger.info(f"Order routed: {order.order_id} to {len(decisions)} venues")

        return order

    async def process_fill(
        self,
        child_id: str,
        filled_quantity: float,
        filled_price: float,
        fees: float = 0
    ) -> Tuple[ChildOrder, RouteOrder]:
        """
        处理子订单成交

        Args:
            child_id: 子订单ID
            filled_quantity: 成交数量
            filled_price: 成交价格
            fees: 费用

        Returns:
            (子订单, 父订单)
        """
        if child_id not in self._child_orders:
            raise ValueError(f"Child order {child_id} not found")

        child = self._child_orders[child_id]
        parent = self._orders.get(child.parent_id)

        if not parent:
            raise ValueError(f"Parent order {child.parent_id} not found")

        # 更新子订单
        child.filled_quantity += filled_quantity
        child.filled_price = (
            (child.filled_price * (child.filled_quantity - filled_quantity) +
             filled_price * filled_quantity) / child.filled_quantity
            if child.filled_quantity > 0 else filled_price
        )
        child.fees += fees
        child.filled_at = datetime.now()

        if child.filled_quantity >= child.quantity:
            child.status = RouteStatus.FILLED
        else:
            child.status = RouteStatus.PARTIAL

        # 更新父订单
        parent.filled_quantity += filled_quantity
        parent.avg_fill_price = (
            (parent.avg_fill_price * (parent.filled_quantity - filled_quantity) +
             filled_price * filled_quantity) / parent.filled_quantity
            if parent.filled_quantity > 0 else filled_price
        )
        parent.total_fees += fees

        # 更新统计
        self._stats['total_filled'] += filled_quantity
        self._stats['by_venue'][child.venue_id]['filled'] += filled_quantity
        self._stats['by_venue'][child.venue_id]['fees'] += fees

        # 检查是否完成
        if parent.filled_quantity >= parent.quantity:
            await self._complete_order(parent)
        elif parent.status == RouteStatus.ROUTED:
            parent.status = RouteStatus.PARTIAL

        return child, parent

    async def cancel_order(self, order_id: str, reason: str = "User cancelled") -> RouteOrder:
        """取消订单"""
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")

        order = self._orders[order_id]

        if order.is_complete:
            raise ValueError(f"Order {order_id} is already complete")

        # 取消所有子订单
        for child_id in order.child_orders:
            child = self._child_orders.get(child_id)
            if child and not child.status in [RouteStatus.FILLED, RouteStatus.CANCELLED]:
                child.status = RouteStatus.CANCELLED

        order.status = RouteStatus.CANCELLED
        order.completed_at = datetime.now()

        logger.info(f"Order cancelled: {order_id} - {reason}")

        return order

    def get_order(self, order_id: str) -> Optional[RouteOrder]:
        """获取订单"""
        return self._orders.get(order_id)

    def get_child_order(self, child_id: str) -> Optional[ChildOrder]:
        """获取子订单"""
        return self._child_orders.get(child_id)

    def analyze_execution(self, order_id: str) -> ExecutionQuality:
        """
        分析执行质量

        Args:
            order_id: 订单ID

        Returns:
            执行质量分析
        """
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        # 获取到达价格(首次报价的中间价)
        quotes = self._quotes.get(order.symbol, {})
        arrival_price = order.limit_price or 0
        if quotes and not arrival_price:
            first_quote = next(iter(quotes.values()), None)
            if first_quote:
                arrival_price = first_quote.mid_price

        # 计算实现差异
        if arrival_price > 0 and order.avg_fill_price > 0:
            if order.side == OrderSide.BUY:
                is_bps = (order.avg_fill_price - arrival_price) / arrival_price * 10000
            else:
                is_bps = (arrival_price - order.avg_fill_price) / arrival_price * 10000
        else:
            is_bps = 0

        # 估计市场影响
        impact_bps = abs(is_bps) * 0.6  # 简化估计
        timing_bps = abs(is_bps) * 0.3

        # 费用
        fee_bps = (order.total_fees / (order.filled_quantity * order.avg_fill_price) * 10000
                   if order.filled_quantity > 0 and order.avg_fill_price > 0 else 0)

        # 场所分析
        venue_breakdown = {}
        for child_id in order.child_orders:
            child = self._child_orders.get(child_id)
            if child:
                venue_breakdown[child.venue_id] = {
                    'quantity': child.quantity,
                    'filled': child.filled_quantity,
                    'fill_rate': round(child.filled_quantity / child.quantity * 100, 2) if child.quantity > 0 else 0,
                    'avg_price': round(child.filled_price, 4),
                    'fees': round(child.fees, 2),
                }

        # 执行评分
        fill_score = order.fill_rate * 40
        cost_score = max(0, 30 - abs(is_bps))
        speed_score = 20 if order.completed_at and order.routed_at else 10
        venue_score = 10 if len(venue_breakdown) > 1 else 5

        execution_score = fill_score + cost_score + speed_score + venue_score

        return ExecutionQuality(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            arrival_price=arrival_price,
            decision_price=order.limit_price or arrival_price,
            avg_fill_price=order.avg_fill_price,
            total_quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            implementation_shortfall_bps=is_bps,
            market_impact_bps=impact_bps,
            timing_cost_bps=timing_bps,
            venue_fees_bps=fee_bps,
            venue_breakdown=venue_breakdown,
            execution_score=execution_score,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取路由统计"""
        fill_rate = (self._stats['total_filled'] / self._stats['total_volume'] * 100
                     if self._stats['total_volume'] > 0 else 0)

        return {
            'orders_routed': self._stats['orders_routed'],
            'total_volume': self._stats['total_volume'],
            'total_filled': self._stats['total_filled'],
            'fill_rate': round(fill_rate, 2),
            'active_venues': len([v for v in self._venues.values() if v.is_active]),
            'by_venue': dict(self._stats['by_venue']),
            'by_strategy': dict(self._stats['by_strategy']),
        }

    def get_venues(self) -> List[Venue]:
        """获取所有场所"""
        return list(self._venues.values())

    def set_child_callback(self, callback: Callable[[ChildOrder], None]) -> None:
        """设置子订单创建回调"""
        self._on_child_created = callback

    def set_completion_callback(self, callback: Callable[[RouteOrder], None]) -> None:
        """设置订单完成回调"""
        self._on_order_completed = callback

    # ==================== 私有方法 ====================

    def _get_available_venues(self, order: RouteOrder) -> List[Venue]:
        """获取可用场所"""
        venues = []

        for venue in self._venues.values():
            # 检查是否激活
            if not venue.is_active:
                continue

            # 检查白名单
            if order.allowed_venues and venue.venue_id not in order.allowed_venues:
                continue

            # 检查黑名单
            if venue.venue_id in order.excluded_venues:
                continue

            # 检查订单大小约束
            if order.quantity < venue.min_order_size:
                continue

            venues.append(venue)

        # 按优先级排序
        venues.sort(key=lambda v: v.priority, reverse=True)

        return venues

    async def _create_child_order(self, parent: RouteOrder, decision: RouteDecision) -> ChildOrder:
        """创建子订单"""
        child = ChildOrder(
            child_id=f"CH-{uuid.uuid4().hex[:8]}",
            parent_id=parent.order_id,
            venue_id=decision.venue_id,
            symbol=parent.symbol,
            side=parent.side,
            quantity=decision.allocation,
            order_type=parent.order_type,
            limit_price=parent.limit_price,
        )

        child.status = RouteStatus.ROUTED
        child.sent_at = datetime.now()

        self._child_orders[child.child_id] = child

        # 更新场所统计
        self._stats['by_venue'][decision.venue_id]['volume'] += decision.allocation

        if self._on_child_created:
            self._on_child_created(child)

        return child

    async def _complete_order(self, order: RouteOrder) -> None:
        """完成订单"""
        order.status = RouteStatus.FILLED
        order.completed_at = datetime.now()

        # 更新策略统计
        strategy_stats = self._stats['by_strategy'][order.strategy.value]
        n = strategy_stats['count']
        strategy_stats['fill_rate'] = (
            (strategy_stats['fill_rate'] * (n - 1) + order.fill_rate) / n
            if n > 0 else order.fill_rate
        )

        if self._on_order_completed:
            self._on_order_completed(order)

        logger.info(f"Order completed: {order.order_id} - fill rate: {order.fill_rate:.1%}")


# ==================== 工厂函数 ====================

def create_smart_router(
    default_strategy: str = "best_execution",
    dark_pool_enabled: bool = True,
    max_dark_pool_pct: float = 0.3,
    **kwargs
) -> SmartOrderRouter:
    """
    创建智能订单路由器

    Args:
        default_strategy: 默认路由策略
        dark_pool_enabled: 是否启用暗池
        max_dark_pool_pct: 最大暗池比例
        **kwargs: 其他配置

    Returns:
        SmartOrderRouter实例
    """
    config = SORConfig(
        default_strategy=RouteStrategy(default_strategy),
        dark_pool_enabled=dark_pool_enabled,
        max_dark_pool_pct=max_dark_pool_pct,
        **{k: v for k, v in kwargs.items() if hasattr(SORConfig, k)}
    )
    return SmartOrderRouter(config)


def create_venue(
    venue_id: str,
    name: str,
    venue_type: str,
    **kwargs
) -> Venue:
    """创建执行场所"""
    return Venue(
        venue_id=venue_id,
        name=name,
        venue_type=VenueType(venue_type),
        maker_fee_bps=kwargs.get('maker_fee_bps', 0),
        taker_fee_bps=kwargs.get('taker_fee_bps', 0),
        rebate_bps=kwargs.get('rebate_bps', 0),
        fill_rate=kwargs.get('fill_rate', 0.8),
        avg_latency_ms=kwargs.get('avg_latency_ms', 1),
        is_active=kwargs.get('is_active', True),
        priority=kwargs.get('priority', 0),
    )


def create_route_order(
    symbol: str,
    side: str,
    quantity: float,
    **kwargs
) -> RouteOrder:
    """创建路由订单"""
    return RouteOrder(
        order_id=kwargs.get('order_id', f"RO-{uuid.uuid4().hex[:8]}"),
        symbol=symbol,
        side=OrderSide(side),
        quantity=quantity,
        order_type=OrderType(kwargs.get('order_type', 'limit')),
        limit_price=kwargs.get('limit_price'),
        strategy=RouteStrategy(kwargs.get('strategy', 'best_execution')),
        max_venues=kwargs.get('max_venues', 5),
        urgency=kwargs.get('urgency', 0.5),
    )


def create_venue_quote(
    venue_id: str,
    symbol: str,
    bid_price: float,
    bid_size: float,
    ask_price: float,
    ask_size: float,
    **kwargs
) -> VenueQuote:
    """创建场所报价"""
    return VenueQuote(
        venue_id=venue_id,
        symbol=symbol,
        timestamp=kwargs.get('timestamp', datetime.now()),
        bid_price=bid_price,
        bid_size=bid_size,
        ask_price=ask_price,
        ask_size=ask_size,
        hidden_liquidity=kwargs.get('hidden_liquidity', 0),
    )


async def quick_route(
    symbol: str,
    side: str,
    quantity: float,
    venues: List[Dict],
    quotes: List[Dict],
    strategy: str = "best_execution"
) -> RouteOrder:
    """
    快速路由订单

    Args:
        symbol: 标的代码
        side: 方向
        quantity: 数量
        venues: 场所配置列表
        quotes: 报价列表
        strategy: 路由策略

    Returns:
        路由后的订单
    """
    router = create_smart_router(default_strategy=strategy)

    # 注册场所
    for v in venues:
        # 排除已经明确传递的参数
        extra_params = {k: val for k, val in v.items()
                        if k not in ['venue_id', 'name', 'venue_type']}
        venue = create_venue(
            venue_id=v['venue_id'],
            name=v.get('name', v['venue_id']),
            venue_type=v.get('venue_type', 'exchange'),
            **extra_params
        )
        router.register_venue(venue)

    # 更新报价
    for q in quotes:
        quote = create_venue_quote(
            venue_id=q['venue_id'],
            symbol=symbol,
            bid_price=q['bid_price'],
            bid_size=q['bid_size'],
            ask_price=q['ask_price'],
            ask_size=q['ask_size'],
        )
        router.update_quote(quote)

    # 创建并路由订单
    order = create_route_order(symbol, side, quantity, strategy=strategy)
    return await router.route_order(order)
