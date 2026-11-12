"""
算法执行引擎模块 (Algorithmic Execution Engine)

提供智能订单执行算法:
- TWAP (时间加权平均价格)
- VWAP (成交量加权平均价格)
- POV (成交量参与率)
- IS (Implementation Shortfall最小化)
- Iceberg (冰山订单)
- Sniper (狙击算法)
- Adaptive (自适应算法)
"""

import asyncio
import logging
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class AlgoType(str, Enum):
    """算法类型"""
    TWAP = "twap"                    # 时间加权平均价格
    VWAP = "vwap"                    # 成交量加权平均价格
    POV = "pov"                      # 成交量参与率
    IS = "is"                        # Implementation Shortfall
    ICEBERG = "iceberg"             # 冰山订单
    SNIPER = "sniper"               # 狙击算法
    ADAPTIVE = "adaptive"           # 自适应算法


class AlgoStatus(str, Enum):
    """算法状态"""
    PENDING = "pending"             # 等待启动
    RUNNING = "running"             # 运行中
    PAUSED = "paused"               # 已暂停
    COMPLETED = "completed"         # 已完成
    CANCELLED = "cancelled"         # 已取消
    FAILED = "failed"               # 失败


class SliceStatus(str, Enum):
    """子订单状态"""
    PENDING = "pending"
    SENT = "sent"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class PriceType(str, Enum):
    """价格类型"""
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_IOC = "limit_ioc"         # 立即成交或取消
    LIMIT_FOK = "limit_fok"         # 全部成交或取消


class Urgency(str, Enum):
    """紧迫度"""
    LOW = "low"                     # 低 - 优先最小化市场影响
    MEDIUM = "medium"               # 中 - 平衡
    HIGH = "high"                   # 高 - 优先完成率


# ==================== 数据类 ====================

@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    last_price: float
    bid_size: float = 0
    ask_size: float = 0
    volume: float = 0               # 累计成交量
    vwap: float = 0                 # 当日VWAP
    volatility: float = 0           # 波动率
    spread: float = 0               # 买卖价差

    @property
    def mid_price(self) -> float:
        """中间价"""
        return (self.bid_price + self.ask_price) / 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'last_price': self.last_price,
            'mid_price': self.mid_price,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'volume': self.volume,
            'vwap': self.vwap,
        }


@dataclass
class OrderSlice:
    """子订单(切片)"""
    slice_id: str
    parent_algo_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price_type: PriceType = PriceType.LIMIT
    limit_price: Optional[float] = None

    # 状态
    status: SliceStatus = SliceStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    commission: float = 0

    # 时间
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # 执行信息
    broker_order_id: Optional[str] = None
    reject_reason: Optional[str] = None

    @property
    def remaining_quantity(self) -> float:
        """剩余数量"""
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        """是否完成"""
        return self.status in [SliceStatus.FILLED, SliceStatus.CANCELLED, SliceStatus.REJECTED]

    @property
    def fill_rate(self) -> float:
        """成交率"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'slice_id': self.slice_id,
            'parent_algo_id': self.parent_algo_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price_type': self.price_type.value,
            'limit_price': self.limit_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'fill_rate': self.fill_rate,
            'created_at': self.created_at.isoformat(),
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
        }


@dataclass
class AlgoOrder:
    """算法订单"""
    algo_id: str
    symbol: str
    side: OrderSide
    total_quantity: float
    algo_type: AlgoType

    # 算法参数
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    urgency: Urgency = Urgency.MEDIUM
    limit_price: Optional[float] = None         # 价格限制
    participation_rate: float = 0.1             # 参与率 (POV)
    display_quantity: Optional[float] = None    # 显示数量 (Iceberg)

    # 状态
    status: AlgoStatus = AlgoStatus.PENDING
    filled_quantity: float = 0
    filled_value: float = 0
    slices_sent: int = 0
    slices_filled: int = 0

    # 执行统计
    arrival_price: float = 0                    # 到达价格
    avg_fill_price: float = 0                   # 平均成交价
    market_vwap: float = 0                      # 市场VWAP

    # 子订单
    slices: List[OrderSlice] = field(default_factory=list)

    # 元数据
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def remaining_quantity(self) -> float:
        """剩余数量"""
        return self.total_quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        """完成率"""
        return self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0

    @property
    def avg_price(self) -> float:
        """平均成交价"""
        return self.filled_value / self.filled_quantity if self.filled_quantity > 0 else 0

    @property
    def slippage_bps(self) -> float:
        """滑点(基点)"""
        if self.arrival_price <= 0 or self.avg_fill_price <= 0:
            return 0
        diff = self.avg_fill_price - self.arrival_price
        if self.side == OrderSide.SELL:
            diff = -diff
        return (diff / self.arrival_price) * 10000

    @property
    def vs_vwap_bps(self) -> float:
        """vs VWAP (基点)"""
        if self.market_vwap <= 0 or self.avg_fill_price <= 0:
            return 0
        diff = self.avg_fill_price - self.market_vwap
        if self.side == OrderSide.SELL:
            diff = -diff
        return (diff / self.market_vwap) * 10000

    @property
    def duration(self) -> Optional[timedelta]:
        """执行时长"""
        if self.completed_at:
            return self.completed_at - self.start_time
        return datetime.now() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algo_id': self.algo_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'total_quantity': self.total_quantity,
            'algo_type': self.algo_type.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'urgency': self.urgency.value,
            'limit_price': self.limit_price,
            'participation_rate': self.participation_rate,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'fill_rate': round(self.fill_rate * 100, 2),
            'avg_fill_price': round(self.avg_price, 4),
            'arrival_price': self.arrival_price,
            'slippage_bps': round(self.slippage_bps, 2),
            'vs_vwap_bps': round(self.vs_vwap_bps, 2),
            'slices_sent': self.slices_sent,
            'slices_filled': self.slices_filled,
            'duration_seconds': self.duration.total_seconds() if self.duration else 0,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class AlgoConfig:
    """算法配置"""
    # 切片配置
    min_slice_size: float = 100              # 最小切片大小
    max_slice_size: float = 10000            # 最大切片大小
    slice_interval_seconds: float = 30        # 切片间隔(秒)

    # 价格配置
    price_tolerance_bps: float = 10          # 价格容忍度(基点)
    aggressive_offset_bps: float = 5         # 激进下单偏移(基点)

    # 风险配置
    max_participation_rate: float = 0.20     # 最大参与率
    max_spread_bps: float = 50               # 最大价差限制

    # 时间配置
    default_duration_minutes: int = 60       # 默认执行时长


@dataclass
class ExecutionReport:
    """执行报告"""
    algo_id: str
    symbol: str
    algo_type: AlgoType
    side: OrderSide

    # 订单信息
    total_quantity: float
    filled_quantity: float
    fill_rate: float

    # 价格分析
    arrival_price: float
    avg_fill_price: float
    market_vwap: float
    slippage_bps: float
    vs_vwap_bps: float

    # 执行统计
    total_slices: int
    filled_slices: int
    avg_slice_size: float
    execution_duration_seconds: float

    # 成本
    total_commission: float
    estimated_market_impact_bps: float

    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algo_id': self.algo_id,
            'symbol': self.symbol,
            'algo_type': self.algo_type.value,
            'side': self.side.value,
            'total_quantity': self.total_quantity,
            'filled_quantity': self.filled_quantity,
            'fill_rate': round(self.fill_rate * 100, 2),
            'arrival_price': self.arrival_price,
            'avg_fill_price': self.avg_fill_price,
            'market_vwap': self.market_vwap,
            'slippage_bps': round(self.slippage_bps, 2),
            'vs_vwap_bps': round(self.vs_vwap_bps, 2),
            'total_slices': self.total_slices,
            'filled_slices': self.filled_slices,
            'avg_slice_size': round(self.avg_slice_size, 2),
            'execution_duration_seconds': round(self.execution_duration_seconds, 2),
            'total_commission': round(self.total_commission, 2),
            'estimated_market_impact_bps': round(self.estimated_market_impact_bps, 2),
            'generated_at': self.generated_at.isoformat(),
        }


# ==================== 算法基类 ====================

class ExecutionAlgorithm(ABC):
    """执行算法基类"""

    def __init__(self, config: Optional[AlgoConfig] = None):
        self.config = config or AlgoConfig()
        self._market_data: Dict[str, MarketData] = {}
        self._volume_profile: Dict[str, List[float]] = {}  # 日内成交量分布

    @property
    @abstractmethod
    def algo_type(self) -> AlgoType:
        """算法类型"""
        pass

    @abstractmethod
    async def generate_slices(
        self,
        order: AlgoOrder,
        market_data: MarketData
    ) -> List[OrderSlice]:
        """
        生成子订单

        Args:
            order: 算法订单
            market_data: 市场数据

        Returns:
            子订单列表
        """
        pass

    def update_market_data(self, data: MarketData) -> None:
        """更新市场数据"""
        self._market_data[data.symbol] = data

    def set_volume_profile(self, symbol: str, profile: List[float]) -> None:
        """设置成交量分布"""
        # 归一化
        total = sum(profile)
        if total > 0:
            self._volume_profile[symbol] = [v / total for v in profile]
        else:
            self._volume_profile[symbol] = profile

    def _create_slice(
        self,
        order: AlgoOrder,
        quantity: float,
        price_type: PriceType = PriceType.LIMIT,
        limit_price: Optional[float] = None
    ) -> OrderSlice:
        """创建子订单"""
        return OrderSlice(
            slice_id=f"SL-{uuid.uuid4().hex[:8]}",
            parent_algo_id=order.algo_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price_type=price_type,
            limit_price=limit_price,
        )

    def _calculate_limit_price(
        self,
        order: AlgoOrder,
        market_data: MarketData,
        aggressive: bool = False
    ) -> float:
        """计算限价"""
        if order.side == OrderSide.BUY:
            base_price = market_data.ask_price
            if aggressive:
                # 激进买入: 高于卖价
                offset = base_price * (self.config.aggressive_offset_bps / 10000)
                return base_price + offset
            else:
                # 被动买入: 中间价附近
                return market_data.mid_price
        else:
            base_price = market_data.bid_price
            if aggressive:
                # 激进卖出: 低于买价
                offset = base_price * (self.config.aggressive_offset_bps / 10000)
                return base_price - offset
            else:
                # 被动卖出: 中间价附近
                return market_data.mid_price


# ==================== TWAP算法 ====================

class TWAPAlgorithm(ExecutionAlgorithm):
    """
    TWAP算法 (Time-Weighted Average Price)

    将订单均匀分布在时间段内执行
    """

    @property
    def algo_type(self) -> AlgoType:
        return AlgoType.TWAP

    async def generate_slices(
        self,
        order: AlgoOrder,
        market_data: MarketData
    ) -> List[OrderSlice]:
        """生成TWAP子订单"""
        slices = []

        remaining = order.remaining_quantity
        if remaining <= 0:
            return slices

        # 计算剩余时间和切片数量
        if order.end_time:
            remaining_seconds = (order.end_time - datetime.now()).total_seconds()
        else:
            remaining_seconds = self.config.default_duration_minutes * 60

        if remaining_seconds <= 0:
            # 时间已过，一次性完成
            slice_qty = min(remaining, self.config.max_slice_size)
            slices.append(self._create_slice(
                order, slice_qty, PriceType.MARKET
            ))
            return slices

        # 计算切片数量
        num_slices = max(1, int(remaining_seconds / self.config.slice_interval_seconds))
        slice_qty = remaining / num_slices

        # 约束切片大小
        slice_qty = max(self.config.min_slice_size, min(slice_qty, self.config.max_slice_size))

        # 根据紧迫度决定激进程度
        aggressive = order.urgency == Urgency.HIGH

        limit_price = self._calculate_limit_price(order, market_data, aggressive)

        # 只生成当前切片
        slices.append(self._create_slice(
            order, slice_qty, PriceType.LIMIT, limit_price
        ))

        return slices


# ==================== VWAP算法 ====================

class VWAPAlgorithm(ExecutionAlgorithm):
    """
    VWAP算法 (Volume-Weighted Average Price)

    根据历史成交量分布执行订单
    """

    @property
    def algo_type(self) -> AlgoType:
        return AlgoType.VWAP

    async def generate_slices(
        self,
        order: AlgoOrder,
        market_data: MarketData
    ) -> List[OrderSlice]:
        """生成VWAP子订单"""
        slices = []

        remaining = order.remaining_quantity
        if remaining <= 0:
            return slices

        # 获取成交量分布
        volume_profile = self._volume_profile.get(order.symbol)
        if not volume_profile:
            # 如果没有分布数据，使用默认的U型分布
            volume_profile = self._default_volume_profile()

        # 计算当前时间在交易日中的位置
        now = datetime.now()
        trading_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        trading_end = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now < trading_start:
            time_pct = 0
        elif now > trading_end:
            time_pct = 1
        else:
            total_minutes = (trading_end - trading_start).total_seconds() / 60
            elapsed_minutes = (now - trading_start).total_seconds() / 60
            time_pct = elapsed_minutes / total_minutes

        # 找到对应的成交量桶
        bucket_idx = int(time_pct * len(volume_profile))
        bucket_idx = min(bucket_idx, len(volume_profile) - 1)

        # 根据成交量分布计算切片大小
        volume_weight = volume_profile[bucket_idx]

        # 基础切片大小
        base_slice = order.total_quantity / len(volume_profile)

        # 根据成交量权重调整
        slice_qty = base_slice * volume_weight * len(volume_profile)

        # 约束
        slice_qty = max(self.config.min_slice_size, min(slice_qty, remaining))
        slice_qty = min(slice_qty, self.config.max_slice_size)

        # 根据紧迫度决定激进程度
        aggressive = order.urgency == Urgency.HIGH
        limit_price = self._calculate_limit_price(order, market_data, aggressive)

        slices.append(self._create_slice(
            order, slice_qty, PriceType.LIMIT, limit_price
        ))

        return slices

    def _default_volume_profile(self) -> List[float]:
        """默认U型成交量分布"""
        # 模拟典型的U型日内成交量分布
        # 开盘和收盘成交量高，午间成交量低
        return [
            0.12, 0.10, 0.08, 0.06, 0.05,  # 上午
            0.04, 0.04, 0.05, 0.06,         # 午间
            0.08, 0.10, 0.12, 0.10          # 下午
        ]


# ==================== POV算法 ====================

class POVAlgorithm(ExecutionAlgorithm):
    """
    POV算法 (Percentage of Volume)

    按照市场成交量的固定比例执行
    """

    @property
    def algo_type(self) -> AlgoType:
        return AlgoType.POV

    async def generate_slices(
        self,
        order: AlgoOrder,
        market_data: MarketData
    ) -> List[OrderSlice]:
        """生成POV子订单"""
        slices = []

        remaining = order.remaining_quantity
        if remaining <= 0:
            return slices

        # 获取参与率
        participation_rate = min(
            order.participation_rate,
            self.config.max_participation_rate
        )

        # 估算市场成交量
        # 假设每个切片间隔内的市场成交量
        estimated_market_volume = market_data.volume * 0.01  # 简化估计

        # 计算切片大小 = 市场成交量 * 参与率
        slice_qty = estimated_market_volume * participation_rate

        # 约束
        slice_qty = max(self.config.min_slice_size, min(slice_qty, remaining))
        slice_qty = min(slice_qty, self.config.max_slice_size)

        # POV通常使用被动价格
        limit_price = self._calculate_limit_price(order, market_data, aggressive=False)

        slices.append(self._create_slice(
            order, slice_qty, PriceType.LIMIT, limit_price
        ))

        return slices


# ==================== Implementation Shortfall算法 ====================

class ISAlgorithm(ExecutionAlgorithm):
    """
    IS算法 (Implementation Shortfall)

    最小化执行成本（市场影响 + 时机风险）
    """

    @property
    def algo_type(self) -> AlgoType:
        return AlgoType.IS

    async def generate_slices(
        self,
        order: AlgoOrder,
        market_data: MarketData
    ) -> List[OrderSlice]:
        """生成IS子订单"""
        slices = []

        remaining = order.remaining_quantity
        if remaining <= 0:
            return slices

        # IS算法核心: 平衡市场影响和时机风险
        # 高紧迫度 -> 快速执行（接受更多市场影响）
        # 低紧迫度 -> 慢速执行（减少市场影响）

        volatility = market_data.volatility if market_data.volatility > 0 else 0.25

        # 计算最优交易速度 (Almgren-Chriss模型简化版)
        if order.urgency == Urgency.HIGH:
            # 高紧迫度: 快速完成
            urgency_factor = 2.0
        elif order.urgency == Urgency.LOW:
            # 低紧迫度: 慢速执行
            urgency_factor = 0.5
        else:
            urgency_factor = 1.0

        # 基于波动率调整
        # 高波动 -> 更快执行（减少时机风险）
        volatility_factor = 1 + volatility

        # 计算切片大小
        base_slice = order.total_quantity * 0.05  # 基础5%
        slice_qty = base_slice * urgency_factor * volatility_factor

        # 约束
        slice_qty = max(self.config.min_slice_size, min(slice_qty, remaining))
        slice_qty = min(slice_qty, self.config.max_slice_size)

        # IS使用更激进的定价策略
        aggressive = order.urgency in [Urgency.HIGH, Urgency.MEDIUM]
        limit_price = self._calculate_limit_price(order, market_data, aggressive)

        slices.append(self._create_slice(
            order, slice_qty, PriceType.LIMIT, limit_price
        ))

        return slices


# ==================== Iceberg算法 ====================

class IcebergAlgorithm(ExecutionAlgorithm):
    """
    冰山订单算法

    只显示部分订单量，隐藏真实意图
    """

    @property
    def algo_type(self) -> AlgoType:
        return AlgoType.ICEBERG

    async def generate_slices(
        self,
        order: AlgoOrder,
        market_data: MarketData
    ) -> List[OrderSlice]:
        """生成冰山子订单"""
        slices = []

        remaining = order.remaining_quantity
        if remaining <= 0:
            return slices

        # 显示数量
        display_qty = order.display_quantity
        if not display_qty:
            # 默认显示总量的10%
            display_qty = order.total_quantity * 0.1

        # 约束
        slice_qty = min(display_qty, remaining)
        slice_qty = max(self.config.min_slice_size, slice_qty)
        slice_qty = min(slice_qty, self.config.max_slice_size)

        # 冰山订单使用限价
        limit_price = self._calculate_limit_price(order, market_data, aggressive=False)

        # 如果有价格限制，使用更严格的限价
        if order.limit_price:
            if order.side == OrderSide.BUY:
                limit_price = min(limit_price, order.limit_price)
            else:
                limit_price = max(limit_price, order.limit_price)

        slices.append(self._create_slice(
            order, slice_qty, PriceType.LIMIT, limit_price
        ))

        return slices


# ==================== Sniper算法 ====================

class SniperAlgorithm(ExecutionAlgorithm):
    """
    狙击算法

    等待有利价格出现时快速执行
    """

    @property
    def algo_type(self) -> AlgoType:
        return AlgoType.SNIPER

    async def generate_slices(
        self,
        order: AlgoOrder,
        market_data: MarketData
    ) -> List[OrderSlice]:
        """生成狙击子订单"""
        slices = []

        remaining = order.remaining_quantity
        if remaining <= 0:
            return slices

        # 检查是否有有利价格
        is_favorable = self._is_price_favorable(order, market_data)

        if not is_favorable:
            # 价格不利，等待
            return slices

        # 价格有利，快速执行较大的量
        slice_qty = min(remaining, order.total_quantity * 0.2)  # 一次最多20%
        slice_qty = max(self.config.min_slice_size, slice_qty)

        # 狙击时使用IOC订单
        limit_price = self._calculate_limit_price(order, market_data, aggressive=True)

        slices.append(self._create_slice(
            order, slice_qty, PriceType.LIMIT_IOC, limit_price
        ))

        return slices

    def _is_price_favorable(self, order: AlgoOrder, market_data: MarketData) -> bool:
        """检查价格是否有利"""
        # 基于到达价格判断
        if order.arrival_price <= 0:
            return True  # 没有参考价格，始终认为有利

        tolerance = self.config.price_tolerance_bps / 10000

        if order.side == OrderSide.BUY:
            # 买入时，当前价格低于到达价格为有利
            return market_data.ask_price <= order.arrival_price * (1 + tolerance)
        else:
            # 卖出时，当前价格高于到达价格为有利
            return market_data.bid_price >= order.arrival_price * (1 - tolerance)


# ==================== Adaptive算法 ====================

class AdaptiveAlgorithm(ExecutionAlgorithm):
    """
    自适应算法

    根据市场条件动态调整执行策略
    """

    def __init__(self, config: Optional[AlgoConfig] = None):
        super().__init__(config)
        self._price_history: deque = deque(maxlen=100)
        self._volume_history: deque = deque(maxlen=100)

    @property
    def algo_type(self) -> AlgoType:
        return AlgoType.ADAPTIVE

    async def generate_slices(
        self,
        order: AlgoOrder,
        market_data: MarketData
    ) -> List[OrderSlice]:
        """生成自适应子订单"""
        slices = []

        remaining = order.remaining_quantity
        if remaining <= 0:
            return slices

        # 更新历史数据
        self._price_history.append(market_data.mid_price)
        self._volume_history.append(market_data.volume)

        # 分析市场状态
        market_state = self._analyze_market_state(market_data)

        # 根据市场状态选择策略
        if market_state == 'trending_favorable':
            # 趋势有利，积极执行
            slice_qty = remaining * 0.15
            aggressive = True
            price_type = PriceType.LIMIT
        elif market_state == 'trending_unfavorable':
            # 趋势不利，谨慎执行
            slice_qty = remaining * 0.05
            aggressive = False
            price_type = PriceType.LIMIT
        elif market_state == 'high_volatility':
            # 高波动，快速执行小量
            slice_qty = remaining * 0.08
            aggressive = True
            price_type = PriceType.LIMIT_IOC
        elif market_state == 'low_liquidity':
            # 低流动性，使用冰山策略
            slice_qty = remaining * 0.05
            aggressive = False
            price_type = PriceType.LIMIT
        else:
            # 正常市场，标准执行
            slice_qty = remaining * 0.10
            aggressive = False
            price_type = PriceType.LIMIT

        # 约束
        slice_qty = max(self.config.min_slice_size, min(slice_qty, remaining))
        slice_qty = min(slice_qty, self.config.max_slice_size)

        limit_price = self._calculate_limit_price(order, market_data, aggressive)

        slices.append(self._create_slice(
            order, slice_qty, price_type, limit_price
        ))

        return slices

    def _analyze_market_state(self, market_data: MarketData) -> str:
        """分析市场状态"""
        if len(self._price_history) < 10:
            return 'normal'

        prices = list(self._price_history)

        # 计算价格趋势
        recent_avg = np.mean(prices[-5:])
        older_avg = np.mean(prices[-10:-5])
        trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

        # 计算波动率
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0

        # 判断状态
        if volatility > 0.02:  # 高波动
            return 'high_volatility'
        elif abs(trend) > 0.005:  # 明显趋势
            if trend > 0:
                return 'trending_favorable' if market_data.symbol else 'trending_unfavorable'
            else:
                return 'trending_unfavorable'
        elif market_data.bid_size + market_data.ask_size < 1000:
            return 'low_liquidity'
        else:
            return 'normal'


# ==================== 算法执行引擎 ====================

class AlgoExecutionEngine:
    """算法执行引擎"""

    def __init__(self, config: Optional[AlgoConfig] = None):
        self.config = config or AlgoConfig()

        # 算法实例
        self._algorithms: Dict[AlgoType, ExecutionAlgorithm] = {
            AlgoType.TWAP: TWAPAlgorithm(self.config),
            AlgoType.VWAP: VWAPAlgorithm(self.config),
            AlgoType.POV: POVAlgorithm(self.config),
            AlgoType.IS: ISAlgorithm(self.config),
            AlgoType.ICEBERG: IcebergAlgorithm(self.config),
            AlgoType.SNIPER: SniperAlgorithm(self.config),
            AlgoType.ADAPTIVE: AdaptiveAlgorithm(self.config),
        }

        # 订单管理
        self._orders: Dict[str, AlgoOrder] = {}
        self._active_orders: Dict[str, AlgoOrder] = {}
        self._completed_orders: List[AlgoOrder] = []

        # 回调
        self._on_slice_created: Optional[Callable[[OrderSlice], None]] = None
        self._on_order_completed: Optional[Callable[[AlgoOrder], None]] = None

        # 状态
        self._running = False
        self._execution_task: Optional[asyncio.Task] = None

    def set_slice_callback(self, callback: Callable[[OrderSlice], None]) -> None:
        """设置切片创建回调"""
        self._on_slice_created = callback

    def set_completion_callback(self, callback: Callable[[AlgoOrder], None]) -> None:
        """设置订单完成回调"""
        self._on_order_completed = callback

    async def start(self) -> None:
        """启动引擎"""
        if self._running:
            return

        self._running = True
        self._execution_task = asyncio.create_task(self._execution_loop())
        logger.info("Algo execution engine started")

    async def stop(self) -> None:
        """停止引擎"""
        self._running = False

        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass

        logger.info("Algo execution engine stopped")

    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        algo_type: str,
        **kwargs
    ) -> AlgoOrder:
        """
        提交算法订单

        Args:
            symbol: 标的代码
            side: 方向 (buy/sell)
            quantity: 总数量
            algo_type: 算法类型
            **kwargs: 其他参数

        Returns:
            算法订单
        """
        order = AlgoOrder(
            algo_id=f"ALGO-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=OrderSide(side),
            total_quantity=quantity,
            algo_type=AlgoType(algo_type),
            urgency=Urgency(kwargs.get('urgency', 'medium')),
            limit_price=kwargs.get('limit_price'),
            participation_rate=kwargs.get('participation_rate', 0.1),
            display_quantity=kwargs.get('display_quantity'),
            portfolio_id=kwargs.get('portfolio_id'),
            strategy_id=kwargs.get('strategy_id'),
        )

        # 设置时间
        if kwargs.get('duration_minutes'):
            order.end_time = order.start_time + timedelta(minutes=kwargs['duration_minutes'])
        elif kwargs.get('end_time'):
            order.end_time = kwargs['end_time']
        else:
            order.end_time = order.start_time + timedelta(minutes=self.config.default_duration_minutes)

        self._orders[order.algo_id] = order
        self._active_orders[order.algo_id] = order
        order.status = AlgoStatus.RUNNING

        logger.info(f"Algo order submitted: {order.algo_id} - {algo_type} {side} {quantity} {symbol}")

        return order

    async def cancel_order(self, algo_id: str, reason: str = "User cancelled") -> AlgoOrder:
        """取消算法订单"""
        if algo_id not in self._orders:
            raise ValueError(f"Order {algo_id} not found")

        order = self._orders[algo_id]

        if order.status in [AlgoStatus.COMPLETED, AlgoStatus.CANCELLED, AlgoStatus.FAILED]:
            raise ValueError(f"Order {algo_id} is already {order.status.value}")

        order.status = AlgoStatus.CANCELLED
        order.completed_at = datetime.now()

        # 取消所有未完成的切片
        for slice_order in order.slices:
            if slice_order.status in [SliceStatus.PENDING, SliceStatus.SENT]:
                slice_order.status = SliceStatus.CANCELLED

        if algo_id in self._active_orders:
            del self._active_orders[algo_id]

        self._completed_orders.append(order)

        logger.info(f"Algo order cancelled: {algo_id} - {reason}")

        return order

    async def pause_order(self, algo_id: str) -> AlgoOrder:
        """暂停算法订单"""
        if algo_id not in self._active_orders:
            raise ValueError(f"Active order {algo_id} not found")

        order = self._active_orders[algo_id]
        order.status = AlgoStatus.PAUSED

        logger.info(f"Algo order paused: {algo_id}")

        return order

    async def resume_order(self, algo_id: str) -> AlgoOrder:
        """恢复算法订单"""
        if algo_id not in self._orders:
            raise ValueError(f"Order {algo_id} not found")

        order = self._orders[algo_id]

        if order.status != AlgoStatus.PAUSED:
            raise ValueError(f"Order {algo_id} is not paused")

        order.status = AlgoStatus.RUNNING

        if algo_id not in self._active_orders:
            self._active_orders[algo_id] = order

        logger.info(f"Algo order resumed: {algo_id}")

        return order

    def update_market_data(self, data: MarketData) -> None:
        """更新市场数据"""
        for algo in self._algorithms.values():
            algo.update_market_data(data)

    def set_volume_profile(self, symbol: str, profile: List[float]) -> None:
        """设置成交量分布"""
        for algo in self._algorithms.values():
            algo.set_volume_profile(symbol, profile)

    async def process_fill(
        self,
        slice_id: str,
        filled_quantity: float,
        filled_price: float,
        commission: float = 0
    ) -> Tuple[OrderSlice, AlgoOrder]:
        """
        处理切片成交

        Args:
            slice_id: 切片ID
            filled_quantity: 成交数量
            filled_price: 成交价格
            commission: 佣金

        Returns:
            (切片, 算法订单)
        """
        # 查找切片
        slice_order = None
        parent_order = None

        for order in self._orders.values():
            for s in order.slices:
                if s.slice_id == slice_id:
                    slice_order = s
                    parent_order = order
                    break
            if slice_order:
                break

        if not slice_order or not parent_order:
            raise ValueError(f"Slice {slice_id} not found")

        # 更新切片
        slice_order.filled_quantity += filled_quantity
        slice_order.filled_price = (
            (slice_order.filled_price * (slice_order.filled_quantity - filled_quantity) +
             filled_price * filled_quantity) / slice_order.filled_quantity
            if slice_order.filled_quantity > 0 else filled_price
        )
        slice_order.commission += commission
        slice_order.filled_at = datetime.now()

        if slice_order.filled_quantity >= slice_order.quantity:
            slice_order.status = SliceStatus.FILLED
            parent_order.slices_filled += 1
        else:
            slice_order.status = SliceStatus.PARTIAL

        # 更新父订单
        parent_order.filled_quantity += filled_quantity
        parent_order.filled_value += filled_quantity * filled_price

        # 更新平均价格
        if parent_order.filled_quantity > 0:
            parent_order.avg_fill_price = parent_order.filled_value / parent_order.filled_quantity

        # 检查是否完成
        if parent_order.filled_quantity >= parent_order.total_quantity:
            await self._complete_order(parent_order)

        return slice_order, parent_order

    def get_order(self, algo_id: str) -> Optional[AlgoOrder]:
        """获取订单"""
        return self._orders.get(algo_id)

    def get_active_orders(self) -> List[AlgoOrder]:
        """获取活跃订单"""
        return list(self._active_orders.values())

    def get_order_history(self, limit: int = 100) -> List[AlgoOrder]:
        """获取历史订单"""
        return self._completed_orders[-limit:]

    def generate_report(self, algo_id: str) -> ExecutionReport:
        """生成执行报告"""
        order = self._orders.get(algo_id)
        if not order:
            raise ValueError(f"Order {algo_id} not found")

        total_commission = sum(s.commission for s in order.slices)
        avg_slice_size = (
            order.filled_quantity / order.slices_filled
            if order.slices_filled > 0 else 0
        )

        # 估计市场影响
        participation_rate = 0.1  # 简化估计
        volatility = 0.25  # 简化估计
        impact_bps = 10 * math.sqrt(participation_rate) * volatility * 100

        return ExecutionReport(
            algo_id=order.algo_id,
            symbol=order.symbol,
            algo_type=order.algo_type,
            side=order.side,
            total_quantity=order.total_quantity,
            filled_quantity=order.filled_quantity,
            fill_rate=order.fill_rate,
            arrival_price=order.arrival_price,
            avg_fill_price=order.avg_price,
            market_vwap=order.market_vwap,
            slippage_bps=order.slippage_bps,
            vs_vwap_bps=order.vs_vwap_bps,
            total_slices=len(order.slices),
            filled_slices=order.slices_filled,
            avg_slice_size=avg_slice_size,
            execution_duration_seconds=order.duration.total_seconds() if order.duration else 0,
            total_commission=total_commission,
            estimated_market_impact_bps=impact_bps,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取引擎统计"""
        active = len(self._active_orders)
        completed = len(self._completed_orders)

        total_filled_qty = sum(o.filled_quantity for o in self._completed_orders)
        avg_fill_rate = np.mean([o.fill_rate for o in self._completed_orders]) if self._completed_orders else 0

        by_algo_type = {}
        for order in self._completed_orders:
            algo_type = order.algo_type.value
            if algo_type not in by_algo_type:
                by_algo_type[algo_type] = {'count': 0, 'filled_qty': 0}
            by_algo_type[algo_type]['count'] += 1
            by_algo_type[algo_type]['filled_qty'] += order.filled_quantity

        return {
            'active_orders': active,
            'completed_orders': completed,
            'total_filled_quantity': total_filled_qty,
            'average_fill_rate': round(avg_fill_rate * 100, 2),
            'by_algo_type': by_algo_type,
            'running': self._running,
        }

    # ==================== 私有方法 ====================

    async def _execution_loop(self) -> None:
        """执行循环"""
        while self._running:
            try:
                await self._process_active_orders()
                await asyncio.sleep(self.config.slice_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")

    async def _process_active_orders(self) -> None:
        """处理活跃订单"""
        for algo_id, order in list(self._active_orders.items()):
            if order.status != AlgoStatus.RUNNING:
                continue

            # 检查是否超时
            if order.end_time and datetime.now() > order.end_time:
                await self._handle_timeout(order)
                continue

            # 获取市场数据
            algo = self._algorithms.get(order.algo_type)
            if not algo:
                continue

            market_data = algo._market_data.get(order.symbol)
            if not market_data:
                # 创建模拟市场数据用于测试
                market_data = MarketData(
                    symbol=order.symbol,
                    timestamp=datetime.now(),
                    bid_price=100.0,
                    ask_price=100.1,
                    last_price=100.05,
                )

            # 设置到达价格
            if order.arrival_price == 0:
                order.arrival_price = market_data.mid_price

            # 生成切片
            slices = await algo.generate_slices(order, market_data)

            for slice_order in slices:
                order.slices.append(slice_order)
                order.slices_sent += 1
                slice_order.status = SliceStatus.SENT
                slice_order.sent_at = datetime.now()

                if self._on_slice_created:
                    self._on_slice_created(slice_order)

                logger.debug(f"Slice created: {slice_order.slice_id} for {order.algo_id}")

    async def _complete_order(self, order: AlgoOrder) -> None:
        """完成订单"""
        order.status = AlgoStatus.COMPLETED
        order.completed_at = datetime.now()

        if order.algo_id in self._active_orders:
            del self._active_orders[order.algo_id]

        self._completed_orders.append(order)

        if self._on_order_completed:
            self._on_order_completed(order)

        logger.info(f"Algo order completed: {order.algo_id} - fill rate: {order.fill_rate:.1%}")

    async def _handle_timeout(self, order: AlgoOrder) -> None:
        """处理超时"""
        if order.fill_rate >= 0.95:
            # 基本完成
            await self._complete_order(order)
        else:
            # 未完成，标记失败
            order.status = AlgoStatus.FAILED
            order.completed_at = datetime.now()

            if order.algo_id in self._active_orders:
                del self._active_orders[order.algo_id]

            self._completed_orders.append(order)

            logger.warning(f"Algo order timeout: {order.algo_id} - fill rate: {order.fill_rate:.1%}")


# ==================== 工厂函数 ====================

def create_algo_engine(
    min_slice_size: float = 100,
    max_slice_size: float = 10000,
    slice_interval_seconds: float = 30,
    **kwargs
) -> AlgoExecutionEngine:
    """
    创建算法执行引擎

    Args:
        min_slice_size: 最小切片大小
        max_slice_size: 最大切片大小
        slice_interval_seconds: 切片间隔
        **kwargs: 其他配置参数

    Returns:
        AlgoExecutionEngine实例
    """
    config = AlgoConfig(
        min_slice_size=min_slice_size,
        max_slice_size=max_slice_size,
        slice_interval_seconds=slice_interval_seconds,
        **{k: v for k, v in kwargs.items() if hasattr(AlgoConfig, k)}
    )
    return AlgoExecutionEngine(config)


def create_market_data(
    symbol: str,
    bid_price: float,
    ask_price: float,
    **kwargs
) -> MarketData:
    """创建市场数据"""
    return MarketData(
        symbol=symbol,
        timestamp=kwargs.get('timestamp', datetime.now()),
        bid_price=bid_price,
        ask_price=ask_price,
        last_price=kwargs.get('last_price', (bid_price + ask_price) / 2),
        bid_size=kwargs.get('bid_size', 0),
        ask_size=kwargs.get('ask_size', 0),
        volume=kwargs.get('volume', 0),
        vwap=kwargs.get('vwap', 0),
        volatility=kwargs.get('volatility', 0.25),
    )


async def quick_algo_order(
    symbol: str,
    side: str,
    quantity: float,
    algo_type: str = 'twap',
    duration_minutes: int = 60
) -> AlgoOrder:
    """
    快速创建算法订单

    Args:
        symbol: 标的
        side: 方向
        quantity: 数量
        algo_type: 算法类型
        duration_minutes: 执行时长

    Returns:
        算法订单
    """
    engine = create_algo_engine()
    return await engine.submit_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        algo_type=algo_type,
        duration_minutes=duration_minutes,
    )
