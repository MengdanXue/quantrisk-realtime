"""
止损引擎 (Stop Loss Engine)

机构级止损管理系统，提供:
- Fixed Stop Loss 固定价格止损
- Trailing Stop Loss 追踪止损
- Time Stop Loss 时间止损
- Volatility Stop Loss 波动率止损(ATR)
- Percentage Stop Loss 百分比止损
- Partial Stop Loss 部分止损
- Stop Loss Order Management 止损订单管理
- Real-time Price Monitoring 实时价格监控
- Prometheus Metrics Integration Prometheus指标集成

作者: TraderX 量化团队
Copyright (c) 2025 TraderX. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import logging
import uuid

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class StopLossType(str, Enum):
    """止损类型"""
    FIXED = "fixed"                     # 固定价格止损
    TRAILING = "trailing"               # 追踪止损
    TIME = "time"                       # 时间止损
    VOLATILITY = "volatility"           # 波动率止损(ATR)
    PERCENTAGE = "percentage"           # 百分比止损


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"                 # 待触发
    ACTIVE = "active"                   # 激活
    TRIGGERED = "triggered"             # 已触发
    EXECUTED = "executed"               # 已执行
    CANCELLED = "cancelled"             # 已取消
    EXPIRED = "expired"                 # 已过期
    FAILED = "failed"                   # 失败


class PositionSide(str, Enum):
    """持仓方向"""
    LONG = "long"                       # 多头
    SHORT = "short"                     # 空头


class TriggerCondition(str, Enum):
    """触发条件"""
    PRICE_BELOW = "price_below"         # 价格低于
    PRICE_ABOVE = "price_above"         # 价格高于
    TIME_EXPIRED = "time_expired"       # 时间到期
    ATR_DISTANCE = "atr_distance"       # ATR距离


class EngineState(str, Enum):
    """引擎状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# ==================== 数据类 ====================

@dataclass
class StopLossConfig:
    """止损配置"""
    stop_type: StopLossType

    # 固定止损
    fixed_price: Optional[float] = None

    # 百分比止损
    percentage: Optional[float] = None          # 止损百分比 (如 0.05 = 5%)

    # 追踪止损
    trailing_distance: Optional[float] = None   # 追踪距离
    trailing_percent: Optional[float] = None    # 追踪百分比

    # 时间止损
    expire_at: Optional[datetime] = None        # 到期时间
    duration: Optional[timedelta] = None        # 持续时间

    # 波动率止损(ATR)
    atr_multiplier: Optional[float] = None      # ATR倍数
    atr_period: int = 14                        # ATR周期

    # 部分止损
    partial_close: bool = False                 # 是否部分平仓
    partial_ratio: float = 1.0                  # 平仓比例 (0-1)

    # 其他设置
    enabled: bool = True
    notes: str = ""

    def validate(self) -> bool:
        """验证配置有效性"""
        if self.stop_type == StopLossType.FIXED:
            return self.fixed_price is not None and self.fixed_price > 0
        elif self.stop_type == StopLossType.PERCENTAGE:
            return self.percentage is not None and 0 < self.percentage < 1
        elif self.stop_type == StopLossType.TRAILING:
            return (self.trailing_distance is not None or
                    self.trailing_percent is not None)
        elif self.stop_type == StopLossType.TIME:
            return (self.expire_at is not None or
                    self.duration is not None)
        elif self.stop_type == StopLossType.VOLATILITY:
            return (self.atr_multiplier is not None and
                    self.atr_multiplier > 0)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stop_type': self.stop_type.value,
            'fixed_price': self.fixed_price,
            'percentage': self.percentage,
            'trailing_distance': self.trailing_distance,
            'trailing_percent': self.trailing_percent,
            'expire_at': self.expire_at.isoformat() if self.expire_at else None,
            'duration': str(self.duration) if self.duration else None,
            'atr_multiplier': self.atr_multiplier,
            'atr_period': self.atr_period,
            'partial_close': self.partial_close,
            'partial_ratio': self.partial_ratio,
            'enabled': self.enabled,
            'notes': self.notes,
        }


@dataclass
class StopLossOrder:
    """止损订单"""
    order_id: str
    symbol: str
    side: PositionSide
    entry_price: float
    current_price: float
    position_size: float
    config: StopLossConfig

    # 订单状态
    status: OrderStatus = OrderStatus.PENDING

    # 止损价格
    stop_price: Optional[float] = None
    initial_stop_price: Optional[float] = None

    # 追踪数据
    highest_price: Optional[float] = None       # 最高价(多头)
    lowest_price: Optional[float] = None        # 最低价(空头)

    # ATR数据
    atr_value: Optional[float] = None
    price_history: List[float] = field(default_factory=list)

    # 时间数据
    created_at: datetime = field(default_factory=datetime.now)
    activated_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None
    expire_at: Optional[datetime] = None

    # 执行数据
    trigger_price: Optional[float] = None
    execution_price: Optional[float] = None
    executed_size: float = 0.0
    slippage: float = 0.0

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_stop_price(self) -> float:
        """计算止损价格"""
        if self.config.stop_type == StopLossType.FIXED:
            return self.config.fixed_price

        elif self.config.stop_type == StopLossType.PERCENTAGE:
            if self.side == PositionSide.LONG:
                return self.entry_price * (1 - self.config.percentage)
            else:
                return self.entry_price * (1 + self.config.percentage)

        elif self.config.stop_type == StopLossType.TRAILING:
            if self.side == PositionSide.LONG:
                reference_price = self.highest_price or self.entry_price
                if self.config.trailing_percent:
                    return reference_price * (1 - self.config.trailing_percent)
                else:
                    return reference_price - self.config.trailing_distance
            else:
                reference_price = self.lowest_price or self.entry_price
                if self.config.trailing_percent:
                    return reference_price * (1 + self.config.trailing_percent)
                else:
                    return reference_price + self.config.trailing_distance

        elif self.config.stop_type == StopLossType.VOLATILITY:
            # 使用current_price而不是entry_price作为参考
            reference_price = self.current_price if self.current_price else self.entry_price

            if self.atr_value and self.atr_value > 0:
                atr_distance = self.atr_value * self.config.atr_multiplier
                if self.side == PositionSide.LONG:
                    return reference_price - atr_distance
                else:
                    return reference_price + atr_distance
            else:
                # 回退到百分比止损
                fallback_percent = 0.05
                if self.side == PositionSide.LONG:
                    return self.entry_price * (1 - fallback_percent)
                else:
                    return self.entry_price * (1 + fallback_percent)

        # 默认返回入场价
        return self.entry_price

    def check_trigger(self) -> bool:
        """检查是否触发止损"""
        if not self.config.enabled or self.status != OrderStatus.ACTIVE:
            return False

        # 检查时间止损
        if self.config.stop_type == StopLossType.TIME:
            if self.expire_at and datetime.now() >= self.expire_at:
                return True

        # 检查价格止损
        if self.stop_price:
            if self.side == PositionSide.LONG:
                return self.current_price <= self.stop_price
            else:
                return self.current_price >= self.stop_price

        return False

    def update_price(self, new_price: float):
        """更新当前价格"""
        self.current_price = new_price
        self.price_history.append(new_price)

        # 更新追踪止损的最高/最低价
        if self.config.stop_type == StopLossType.TRAILING:
            if self.side == PositionSide.LONG:
                if self.highest_price is None or new_price > self.highest_price:
                    self.highest_price = new_price
            else:
                if self.lowest_price is None or new_price < self.lowest_price:
                    self.lowest_price = new_price

        # 保留最近的价格历史用于ATR计算
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]

        # 重新计算止损价格
        self.stop_price = self.calculate_stop_price()

    def calculate_atr(self, high_low_close_data: Optional[List[tuple]] = None) -> float:
        """
        计算ATR值

        Args:
            high_low_close_data: [(high, low, close), ...] 或 None (使用价格历史)
        """
        if high_low_close_data and len(high_low_close_data) >= self.config.atr_period:
            # 使用完整的HLC数据
            true_ranges = []
            for i in range(1, len(high_low_close_data)):
                high = high_low_close_data[i][0]
                low = high_low_close_data[i][1]
                prev_close = high_low_close_data[i-1][2]

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)

            if true_ranges:
                atr = sum(true_ranges[-self.config.atr_period:]) / min(len(true_ranges), self.config.atr_period)
                return atr

        elif len(self.price_history) >= self.config.atr_period + 1:
            # 使用简化的价格历史计算
            ranges = []
            for i in range(1, len(self.price_history)):
                price_range = abs(self.price_history[i] - self.price_history[i-1])
                ranges.append(price_range)

            if ranges:
                atr = sum(ranges[-self.config.atr_period:]) / min(len(ranges), self.config.atr_period)
                return atr

        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'position_size': self.position_size,
            'status': self.status.value,
            'stop_price': self.stop_price,
            'initial_stop_price': self.initial_stop_price,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'atr_value': self.atr_value,
            'created_at': self.created_at.isoformat(),
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'expire_at': self.expire_at.isoformat() if self.expire_at else None,
            'trigger_price': self.trigger_price,
            'execution_price': self.execution_price,
            'executed_size': self.executed_size,
            'slippage': self.slippage,
            'config': self.config.to_dict(),
            'metadata': self.metadata,
        }


@dataclass
class StopLossEvent:
    """止损事件"""
    event_id: str
    order_id: str
    symbol: str
    event_type: str                     # created, activated, triggered, executed, cancelled

    # 价格数据
    current_price: float
    stop_price: Optional[float] = None
    execution_price: Optional[float] = None

    # P&L数据
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None

    # 时间
    timestamp: datetime = field(default_factory=datetime.now)

    # 详情
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'event_type': self.event_type,
            'current_price': self.current_price,
            'stop_price': self.stop_price,
            'execution_price': self.execution_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
        }


@dataclass
class EngineConfig:
    """引擎配置"""
    # 价格更新频率
    update_interval: float = 1.0                # 秒

    # 执行设置
    auto_execute: bool = True                   # 自动执行止损
    simulate_execution: bool = True             # 模拟执行(测试用)

    # 滑点设置
    slippage_bps: float = 10.0                  # 滑点(基点)

    # 持久化
    enable_persistence: bool = True
    persistence_interval: float = 60.0          # 持久化间隔(秒)

    # Prometheus
    enable_prometheus: bool = PROMETHEUS_AVAILABLE
    metrics_prefix: str = "stop_loss_engine"

    # 日志
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'update_interval': self.update_interval,
            'auto_execute': self.auto_execute,
            'simulate_execution': self.simulate_execution,
            'slippage_bps': self.slippage_bps,
            'enable_persistence': self.enable_persistence,
            'persistence_interval': self.persistence_interval,
            'enable_prometheus': self.enable_prometheus,
            'metrics_prefix': self.metrics_prefix,
            'log_level': self.log_level,
        }


# ==================== 主引擎类 ====================

class StopLossEngine:
    """
    止损引擎

    支持多种止损类型:
    - 固定价格止损
    - 追踪止损
    - 时间止损
    - 波动率止损(ATR)
    - 百分比止损
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        on_trigger: Optional[Callable[[StopLossOrder], Any]] = None,
        on_execute: Optional[Callable[[StopLossOrder], Any]] = None,
    ):
        """
        初始化止损引擎

        Args:
            config: 引擎配置
            on_trigger: 触发回调函数
            on_execute: 执行回调函数
        """
        self.config = config or EngineConfig()
        self.on_trigger = on_trigger
        self.on_execute = on_execute

        # 状态
        self.state = EngineState.IDLE

        # 订单管理
        self.orders: Dict[str, StopLossOrder] = {}              # order_id -> order
        self.symbol_orders: Dict[str, Set[str]] = defaultdict(set)  # symbol -> order_ids

        # 事件历史
        self.events: List[StopLossEvent] = []

        # 价格数据
        self.latest_prices: Dict[str, float] = {}               # symbol -> price

        # 异步任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._persistence_task: Optional[asyncio.Task] = None

        # 统计
        self.stats = {
            'orders_created': 0,
            'orders_triggered': 0,
            'orders_executed': 0,
            'orders_cancelled': 0,
            'total_pnl': 0.0,
        }

        # Prometheus指标
        self._setup_prometheus()

        # 日志
        self._setup_logging()

        logger.info(f"StopLossEngine initialized with config: {self.config.to_dict()}")

    def _setup_prometheus(self):
        """设置Prometheus指标"""
        if not self.config.enable_prometheus:
            self.metrics = None
            return

        prefix = self.config.metrics_prefix

        self.metrics = {
            'orders_total': Counter(
                f'{prefix}_orders_total',
                'Total number of stop loss orders',
                ['status', 'stop_type']
            ),
            'triggers_total': Counter(
                f'{prefix}_triggers_total',
                'Total number of stop loss triggers',
                ['symbol', 'side']
            ),
            'executions_total': Counter(
                f'{prefix}_executions_total',
                'Total number of stop loss executions',
                ['symbol', 'side']
            ),
            'active_orders': Gauge(
                f'{prefix}_active_orders',
                'Number of active stop loss orders',
                ['symbol']
            ),
            'pnl_total': Gauge(
                f'{prefix}_pnl_total',
                'Total P&L from stop losses',
                ['symbol']
            ),
            'slippage_bps': Histogram(
                f'{prefix}_slippage_bps',
                'Slippage in basis points',
                ['symbol'],
                buckets=[1, 5, 10, 20, 50, 100, 200]
            ),
            'execution_latency': Histogram(
                f'{prefix}_execution_latency_seconds',
                'Execution latency in seconds',
                ['symbol'],
                buckets=[0.1, 0.5, 1, 2, 5, 10]
            ),
        }

    def _setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

    async def start(self):
        """启动引擎"""
        if self.state == EngineState.RUNNING:
            logger.warning("Engine is already running")
            return

        self.state = EngineState.RUNNING

        # 启动监控任务
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        # 启动持久化任务
        if self.config.enable_persistence:
            self._persistence_task = asyncio.create_task(self._persistence_loop())

        logger.info("StopLossEngine started")

    async def stop(self):
        """停止引擎"""
        if self.state != EngineState.RUNNING:
            logger.warning("Engine is not running")
            return

        self.state = EngineState.STOPPED

        # 取消异步任务
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self._persistence_task:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass

        logger.info("StopLossEngine stopped")

    async def pause(self):
        """暂停引擎"""
        if self.state == EngineState.RUNNING:
            self.state = EngineState.PAUSED
            logger.info("StopLossEngine paused")

    async def resume(self):
        """恢复引擎"""
        if self.state == EngineState.PAUSED:
            self.state = EngineState.RUNNING
            logger.info("StopLossEngine resumed")

    def create_order(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        position_size: float,
        config: StopLossConfig,
        order_id: Optional[str] = None,
    ) -> StopLossOrder:
        """
        创建止损订单

        Args:
            symbol: 交易对符号
            side: 持仓方向
            entry_price: 入场价格
            position_size: 持仓大小
            config: 止损配置
            order_id: 订单ID (可选)

        Returns:
            StopLossOrder: 创建的止损订单
        """
        if not config.validate():
            raise ValueError(f"Invalid stop loss config: {config.to_dict()}")

        order_id = order_id or f"sl_{uuid.uuid4().hex[:12]}"

        # 获取当前价格
        current_price = self.latest_prices.get(symbol, entry_price)

        # 创建订单
        order = StopLossOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
            config=config,
        )

        # 初始化追踪价格
        if config.stop_type == StopLossType.TRAILING:
            if side == PositionSide.LONG:
                order.highest_price = current_price
            else:
                order.lowest_price = current_price

        # 设置过期时间
        if config.stop_type == StopLossType.TIME:
            if config.expire_at:
                order.expire_at = config.expire_at
            elif config.duration:
                order.expire_at = datetime.now() + config.duration

        # 计算初始止损价格
        order.stop_price = order.calculate_stop_price()
        order.initial_stop_price = order.stop_price

        # 激活订单
        order.status = OrderStatus.ACTIVE
        order.activated_at = datetime.now()

        # 保存订单
        self.orders[order_id] = order
        self.symbol_orders[symbol].add(order_id)

        # 更新统计
        self.stats['orders_created'] += 1

        # Prometheus指标
        if self.metrics:
            self.metrics['orders_total'].labels(
                status='created',
                stop_type=config.stop_type.value
            ).inc()
            self.metrics['active_orders'].labels(symbol=symbol).inc()

        # 创建事件
        event = StopLossEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            order_id=order_id,
            symbol=symbol,
            event_type='created',
            current_price=current_price,
            stop_price=order.stop_price,
            details={
                'side': side.value,
                'entry_price': entry_price,
                'position_size': position_size,
                'config': config.to_dict(),
            }
        )
        self.events.append(event)

        logger.info(
            f"Created stop loss order: {order_id} | "
            f"{symbol} {side.value} | "
            f"Entry: {entry_price:.4f} | "
            f"Stop: {order.stop_price:.4f} | "
            f"Type: {config.stop_type.value}"
        )

        return order

    def update_price(self, symbol: str, price: float):
        """
        更新价格

        Args:
            symbol: 交易对符号
            price: 最新价格
        """
        self.latest_prices[symbol] = price

        # 更新相关订单
        if symbol in self.symbol_orders:
            for order_id in list(self.symbol_orders[symbol]):
                if order_id in self.orders:
                    order = self.orders[order_id]
                    if order.status == OrderStatus.ACTIVE:
                        order.update_price(price)

    def update_atr(self, symbol: str, atr_value: float):
        """
        更新ATR值并重新计算止损价

        Args:
            symbol: 交易对符号
            atr_value: ATR值
        """
        if symbol in self.symbol_orders:
            for order_id in list(self.symbol_orders[symbol]):
                if order_id in self.orders:
                    order = self.orders[order_id]
                    if (order.status == OrderStatus.ACTIVE and
                        order.config.stop_type == StopLossType.VOLATILITY):
                        order.atr_value = atr_value
                        # 重新计算止损价
                        new_stop = order.calculate_stop_price()
                        if new_stop != order.stop_price:
                            order.stop_price = new_stop
                            logger.debug(
                                f"Updated ATR stop price for {order_id}: "
                                f"{order.stop_price:.4f} (ATR: {atr_value:.4f})"
                            )

    def cancel_order(self, order_id: str) -> bool:
        """
        取消止损订单

        Args:
            order_id: 订单ID

        Returns:
            bool: 是否成功取消
        """
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False

        order = self.orders[order_id]

        if order.status not in [OrderStatus.ACTIVE, OrderStatus.PENDING]:
            logger.warning(f"Cannot cancel order in status: {order.status.value}")
            return False

        order.status = OrderStatus.CANCELLED

        # 从符号订单中移除
        self.symbol_orders[order.symbol].discard(order_id)

        # 更新统计
        self.stats['orders_cancelled'] += 1

        # Prometheus指标
        if self.metrics:
            self.metrics['orders_total'].labels(
                status='cancelled',
                stop_type=order.config.stop_type.value
            ).inc()
            self.metrics['active_orders'].labels(symbol=order.symbol).dec()

        # 创建事件
        event = StopLossEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            order_id=order_id,
            symbol=order.symbol,
            event_type='cancelled',
            current_price=order.current_price,
            stop_price=order.stop_price,
        )
        self.events.append(event)

        logger.info(f"Cancelled stop loss order: {order_id}")

        return True

    def get_order(self, order_id: str) -> Optional[StopLossOrder]:
        """获取订单"""
        return self.orders.get(order_id)

    def get_orders_by_symbol(self, symbol: str) -> List[StopLossOrder]:
        """获取指定符号的所有订单"""
        order_ids = self.symbol_orders.get(symbol, set())
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_active_orders(self) -> List[StopLossOrder]:
        """获取所有活跃订单"""
        return [
            order for order in self.orders.values()
            if order.status == OrderStatus.ACTIVE
        ]

    async def _monitor_loop(self):
        """监控循环"""
        logger.info("Monitor loop started")

        while self.state in [EngineState.RUNNING, EngineState.PAUSED]:
            try:
                if self.state == EngineState.RUNNING:
                    await self._check_orders()

                await asyncio.sleep(self.config.update_interval)

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
                self.state = EngineState.ERROR
                await asyncio.sleep(5)

        logger.info("Monitor loop stopped")

    async def _check_orders(self):
        """检查所有订单"""
        current_time = datetime.now()

        for order in list(self.orders.values()):
            if order.status != OrderStatus.ACTIVE:
                continue

            try:
                # 检查过期
                if order.expire_at and current_time >= order.expire_at:
                    if order.config.stop_type == StopLossType.TIME:
                        # 时间止损触发
                        await self._trigger_order(order, "time_expired")
                    else:
                        # 其他类型过期
                        order.status = OrderStatus.EXPIRED
                        self.symbol_orders[order.symbol].discard(order.order_id)
                        continue

                # 检查价格触发
                if order.check_trigger():
                    await self._trigger_order(order, "price_triggered")

            except Exception as e:
                logger.error(
                    f"Error checking order {order.order_id}: {e}",
                    exc_info=True
                )

    async def _trigger_order(self, order: StopLossOrder, trigger_reason: str):
        """
        触发止损订单

        Args:
            order: 止损订单
            trigger_reason: 触发原因
        """
        order.status = OrderStatus.TRIGGERED
        order.triggered_at = datetime.now()
        order.trigger_price = order.current_price

        # 更新统计
        self.stats['orders_triggered'] += 1

        # Prometheus指标
        if self.metrics:
            self.metrics['triggers_total'].labels(
                symbol=order.symbol,
                side=order.side.value
            ).inc()

        # 创建事件
        event = StopLossEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            order_id=order.order_id,
            symbol=order.symbol,
            event_type='triggered',
            current_price=order.current_price,
            stop_price=order.stop_price,
            details={'trigger_reason': trigger_reason}
        )
        self.events.append(event)

        logger.warning(
            f"Stop loss TRIGGERED: {order.order_id} | "
            f"{order.symbol} {order.side.value} | "
            f"Entry: {order.entry_price:.4f} | "
            f"Stop: {order.stop_price:.4f} | "
            f"Trigger: {order.trigger_price:.4f} | "
            f"Reason: {trigger_reason}"
        )

        # 执行回调
        if self.on_trigger:
            try:
                result = self.on_trigger(order)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in trigger callback: {e}", exc_info=True)

        # 自动执行
        if self.config.auto_execute:
            await self._execute_order(order)

    async def _execute_order(self, order: StopLossOrder):
        """
        执行止损订单

        Args:
            order: 止损订单
        """
        # 计算执行价格(考虑滑点)
        slippage_factor = self.config.slippage_bps / 10000.0

        if order.side == PositionSide.LONG:
            # 多头止损卖出,价格更低
            execution_price = order.trigger_price * (1 - slippage_factor)
        else:
            # 空头止损买入,价格更高
            execution_price = order.trigger_price * (1 + slippage_factor)

        # 计算执行数量
        if order.config.partial_close:
            # 确保不超过总仓位
            executed_size = min(
                order.position_size * order.config.partial_ratio,
                order.position_size
            )
        else:
            executed_size = order.position_size

        # 计算P&L
        if order.side == PositionSide.LONG:
            pnl = (execution_price - order.entry_price) * executed_size
        else:
            pnl = (order.entry_price - execution_price) * executed_size

        pnl_percent = (pnl / (order.entry_price * executed_size)) * 100

        # 计算实际滑点
        if order.stop_price:
            slippage = abs(execution_price - order.stop_price)
            slippage_bps = (slippage / order.stop_price) * 10000
        else:
            slippage = 0
            slippage_bps = 0

        # 更新订单
        order.status = OrderStatus.EXECUTED
        order.execution_price = execution_price
        order.executed_size = executed_size
        order.slippage = slippage

        # 从符号订单中移除
        self.symbol_orders[order.symbol].discard(order.order_id)

        # 更新统计
        self.stats['orders_executed'] += 1
        self.stats['total_pnl'] += pnl

        # Prometheus指标
        if self.metrics:
            self.metrics['executions_total'].labels(
                symbol=order.symbol,
                side=order.side.value
            ).inc()
            self.metrics['active_orders'].labels(symbol=order.symbol).dec()
            self.metrics['pnl_total'].labels(symbol=order.symbol).inc(pnl)
            self.metrics['slippage_bps'].labels(symbol=order.symbol).observe(slippage_bps)

            if order.triggered_at:
                latency = (datetime.now() - order.triggered_at).total_seconds()
                self.metrics['execution_latency'].labels(symbol=order.symbol).observe(latency)

        # 创建事件
        event = StopLossEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            order_id=order.order_id,
            symbol=order.symbol,
            event_type='executed',
            current_price=order.current_price,
            stop_price=order.stop_price,
            execution_price=execution_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            details={
                'executed_size': executed_size,
                'slippage': slippage,
                'slippage_bps': slippage_bps,
            }
        )
        self.events.append(event)

        logger.info(
            f"Stop loss EXECUTED: {order.order_id} | "
            f"{order.symbol} {order.side.value} | "
            f"Entry: {order.entry_price:.4f} | "
            f"Execution: {execution_price:.4f} | "
            f"Size: {executed_size:.4f} | "
            f"P&L: {pnl:.2f} ({pnl_percent:+.2f}%) | "
            f"Slippage: {slippage_bps:.1f}bps"
        )

        # 执行回调
        if self.on_execute:
            try:
                result = self.on_execute(order)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in execute callback: {e}", exc_info=True)

    async def _persistence_loop(self):
        """持久化循环"""
        logger.info("Persistence loop started")

        while self.state in [EngineState.RUNNING, EngineState.PAUSED]:
            try:
                await asyncio.sleep(self.config.persistence_interval)

                if self.state == EngineState.RUNNING:
                    await self._persist_state()

            except Exception as e:
                logger.error(f"Error in persistence loop: {e}", exc_info=True)

        logger.info("Persistence loop stopped")

    async def _persist_state(self):
        """持久化状态"""
        # 这里可以实现实际的持久化逻辑(数据库、文件等)
        logger.debug(f"Persisting state: {len(self.orders)} orders, {len(self.events)} events")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        active_orders = sum(
            1 for order in self.orders.values()
            if order.status == OrderStatus.ACTIVE
        )

        return {
            'state': self.state.value,
            'total_orders': len(self.orders),
            'active_orders': active_orders,
            'orders_created': self.stats['orders_created'],
            'orders_triggered': self.stats['orders_triggered'],
            'orders_executed': self.stats['orders_executed'],
            'orders_cancelled': self.stats['orders_cancelled'],
            'total_pnl': self.stats['total_pnl'],
            'total_events': len(self.events),
        }

    def get_events(
        self,
        symbol: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[StopLossEvent]:
        """
        获取事件历史

        Args:
            symbol: 过滤符号
            event_type: 过滤事件类型
            limit: 最大返回数量

        Returns:
            List[StopLossEvent]: 事件列表
        """
        filtered_events = self.events

        if symbol:
            filtered_events = [e for e in filtered_events if e.symbol == symbol]

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        return filtered_events[-limit:]


# ==================== 工厂函数 ====================

def create_stop_loss_engine(
    config: Optional[EngineConfig] = None,
    on_trigger: Optional[Callable] = None,
    on_execute: Optional[Callable] = None,
) -> StopLossEngine:
    """
    创建止损引擎

    Args:
        config: 引擎配置
        on_trigger: 触发回调函数
        on_execute: 执行回调函数

    Returns:
        StopLossEngine: 止损引擎实例
    """
    return StopLossEngine(
        config=config,
        on_trigger=on_trigger,
        on_execute=on_execute,
    )


def create_fixed_stop_config(
    fixed_price: float,
    partial_close: bool = False,
    partial_ratio: float = 1.0,
) -> StopLossConfig:
    """创建固定价格止损配置"""
    return StopLossConfig(
        stop_type=StopLossType.FIXED,
        fixed_price=fixed_price,
        partial_close=partial_close,
        partial_ratio=partial_ratio,
    )


def create_percentage_stop_config(
    percentage: float,
    partial_close: bool = False,
    partial_ratio: float = 1.0,
) -> StopLossConfig:
    """创建百分比止损配置"""
    return StopLossConfig(
        stop_type=StopLossType.PERCENTAGE,
        percentage=percentage,
        partial_close=partial_close,
        partial_ratio=partial_ratio,
    )


def create_trailing_stop_config(
    trailing_percent: Optional[float] = None,
    trailing_distance: Optional[float] = None,
    partial_close: bool = False,
    partial_ratio: float = 1.0,
) -> StopLossConfig:
    """创建追踪止损配置"""
    return StopLossConfig(
        stop_type=StopLossType.TRAILING,
        trailing_percent=trailing_percent,
        trailing_distance=trailing_distance,
        partial_close=partial_close,
        partial_ratio=partial_ratio,
    )


def create_time_stop_config(
    expire_at: Optional[datetime] = None,
    duration: Optional[timedelta] = None,
    partial_close: bool = False,
    partial_ratio: float = 1.0,
) -> StopLossConfig:
    """创建时间止损配置"""
    return StopLossConfig(
        stop_type=StopLossType.TIME,
        expire_at=expire_at,
        duration=duration,
        partial_close=partial_close,
        partial_ratio=partial_ratio,
    )


def create_volatility_stop_config(
    atr_multiplier: float,
    atr_period: int = 14,
    partial_close: bool = False,
    partial_ratio: float = 1.0,
) -> StopLossConfig:
    """创建波动率止损配置(ATR)"""
    return StopLossConfig(
        stop_type=StopLossType.VOLATILITY,
        atr_multiplier=atr_multiplier,
        atr_period=atr_period,
        partial_close=partial_close,
        partial_ratio=partial_ratio,
    )


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'StopLossType',
    'OrderStatus',
    'PositionSide',
    'TriggerCondition',
    'EngineState',

    # 数据类
    'StopLossConfig',
    'StopLossOrder',
    'StopLossEvent',
    'EngineConfig',

    # 主类
    'StopLossEngine',

    # 工厂函数
    'create_stop_loss_engine',
    'create_fixed_stop_config',
    'create_percentage_stop_config',
    'create_trailing_stop_config',
    'create_time_stop_config',
    'create_volatility_stop_config',
]
