"""
执行监控引擎模块
Execution Monitor Engine

提供订单执行质量监控、滑点分析、延迟追踪和执行报告功能。

功能特性:
- 订单执行追踪与状态监控
- 滑点分析与统计
- 执行延迟测量
- Fill Rate与执行效率分析
- 执行质量评分
- 实时执行告警
- 执行报告生成

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid
import statistics
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class OrderState(Enum):
    """订单状态"""
    PENDING = "pending"  # 待发送
    SUBMITTED = "submitted"  # 已提交
    ACKNOWLEDGED = "acknowledged"  # 已确认
    PARTIAL_FILLED = "partial_filled"  # 部分成交
    FILLED = "filled"  # 完全成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"  # 已拒绝
    EXPIRED = "expired"  # 已过期
    ERROR = "error"  # 错误


class ExecutionVenue(Enum):
    """执行场所"""
    EXCHANGE = "exchange"  # 交易所
    DARK_POOL = "dark_pool"  # 暗池
    INTERNALIZATION = "internalization"  # 内部化
    OTC = "otc"  # 场外
    MARKET_MAKER = "market_maker"  # 做市商


class SlippageType(Enum):
    """滑点类型"""
    POSITIVE = "positive"  # 正滑点(有利)
    NEGATIVE = "negative"  # 负滑点(不利)
    ZERO = "zero"  # 无滑点


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MonitorState(Enum):
    """监控状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ==================== 数据结构 ====================

@dataclass
class OrderExecution:
    """订单执行记录"""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop"
    requested_quantity: Decimal
    requested_price: Optional[Decimal]  # 限价单价格

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    first_fill_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 执行结果
    state: OrderState = OrderState.PENDING
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal = Decimal("0")
    fills: List[Dict[str, Any]] = field(default_factory=list)

    # 基准价格
    arrival_price: Optional[Decimal] = None  # 到达价格
    decision_price: Optional[Decimal] = None  # 决策价格
    benchmark_price: Optional[Decimal] = None  # 基准价格

    # 执行场所
    venue: Optional[ExecutionVenue] = None
    broker_id: Optional[str] = None

    # 元数据
    strategy_id: Optional[str] = None
    trader_id: Optional[str] = None
    account_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fill_rate(self) -> Decimal:
        """成交率"""
        if self.requested_quantity == 0:
            return Decimal("0")
        return self.filled_quantity / self.requested_quantity

    @property
    def is_complete(self) -> bool:
        """是否完成"""
        return self.state in [OrderState.FILLED, OrderState.CANCELLED,
                              OrderState.REJECTED, OrderState.EXPIRED, OrderState.ERROR]

    @property
    def slippage(self) -> Optional[Decimal]:
        """滑点(基于到达价格)"""
        if self.arrival_price is None or self.avg_fill_price == 0:
            return None

        if self.side == "buy":
            # 买入: 成交价高于到达价为负滑点
            return self.arrival_price - self.avg_fill_price
        else:
            # 卖出: 成交价低于到达价为负滑点
            return self.avg_fill_price - self.arrival_price

    @property
    def slippage_bps(self) -> Optional[Decimal]:
        """滑点(基点)"""
        if self.slippage is None or self.arrival_price is None or self.arrival_price == 0:
            return None
        return (self.slippage / self.arrival_price) * Decimal("10000")

    @property
    def slippage_type(self) -> SlippageType:
        """滑点类型"""
        if self.slippage is None:
            return SlippageType.ZERO
        if self.slippage > 0:
            return SlippageType.POSITIVE
        elif self.slippage < 0:
            return SlippageType.NEGATIVE
        return SlippageType.ZERO

    @property
    def submission_latency_ms(self) -> Optional[float]:
        """提交延迟(毫秒)"""
        if self.created_at and self.submitted_at:
            return (self.submitted_at - self.created_at).total_seconds() * 1000
        return None

    @property
    def ack_latency_ms(self) -> Optional[float]:
        """确认延迟(毫秒)"""
        if self.submitted_at and self.acknowledged_at:
            return (self.acknowledged_at - self.submitted_at).total_seconds() * 1000
        return None

    @property
    def first_fill_latency_ms(self) -> Optional[float]:
        """首次成交延迟(毫秒)"""
        if self.submitted_at and self.first_fill_at:
            return (self.first_fill_at - self.submitted_at).total_seconds() * 1000
        return None

    @property
    def total_execution_time_ms(self) -> Optional[float]:
        """总执行时间(毫秒)"""
        if self.submitted_at and self.completed_at:
            return (self.completed_at - self.submitted_at).total_seconds() * 1000
        return None

    @property
    def notional(self) -> Decimal:
        """成交名义金额"""
        return self.filled_quantity * self.avg_fill_price

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "requested_quantity": float(self.requested_quantity),
            "requested_price": float(self.requested_price) if self.requested_price else None,
            "state": self.state.value,
            "filled_quantity": float(self.filled_quantity),
            "avg_fill_price": float(self.avg_fill_price),
            "fill_rate": float(self.fill_rate),
            "arrival_price": float(self.arrival_price) if self.arrival_price else None,
            "slippage": float(self.slippage) if self.slippage is not None else None,
            "slippage_bps": float(self.slippage_bps) if self.slippage_bps is not None else None,
            "slippage_type": self.slippage_type.value,
            "submission_latency_ms": self.submission_latency_ms,
            "ack_latency_ms": self.ack_latency_ms,
            "first_fill_latency_ms": self.first_fill_latency_ms,
            "total_execution_time_ms": self.total_execution_time_ms,
            "notional": float(self.notional),
            "venue": self.venue.value if self.venue else None,
            "broker_id": self.broker_id,
            "strategy_id": self.strategy_id,
            "trader_id": self.trader_id,
            "account_id": self.account_id,
            "fills_count": len(self.fills),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ExecutionFill:
    """成交记录"""
    fill_id: str
    order_id: str
    quantity: Decimal
    price: Decimal
    venue: ExecutionVenue
    timestamp: datetime = field(default_factory=datetime.now)
    fees: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "quantity": float(self.quantity),
            "price": float(self.price),
            "venue": self.venue.value,
            "timestamp": self.timestamp.isoformat(),
            "fees": float(self.fees),
        }


@dataclass
class ExecutionAlert:
    """执行告警"""
    alert_id: str
    order_id: str
    severity: AlertSeverity
    alert_type: str  # "high_slippage", "slow_execution", "partial_fill", etc.
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "alert_id": self.alert_id,
            "order_id": self.order_id,
            "severity": self.severity.value,
            "alert_type": self.alert_type,
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ExecutionMetrics:
    """执行指标"""
    period_start: datetime
    period_end: datetime
    total_orders: int = 0
    completed_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0

    total_quantity: Decimal = Decimal("0")
    filled_quantity: Decimal = Decimal("0")
    total_notional: Decimal = Decimal("0")

    avg_fill_rate: Decimal = Decimal("0")
    avg_slippage_bps: Decimal = Decimal("0")
    avg_submission_latency_ms: float = 0
    avg_ack_latency_ms: float = 0
    avg_first_fill_latency_ms: float = 0
    avg_execution_time_ms: float = 0

    positive_slippage_count: int = 0
    negative_slippage_count: int = 0
    zero_slippage_count: int = 0

    venue_breakdown: Dict[str, int] = field(default_factory=dict)
    symbol_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_orders": self.total_orders,
            "completed_orders": self.completed_orders,
            "filled_orders": self.filled_orders,
            "cancelled_orders": self.cancelled_orders,
            "rejected_orders": self.rejected_orders,
            "total_quantity": float(self.total_quantity),
            "filled_quantity": float(self.filled_quantity),
            "total_notional": float(self.total_notional),
            "avg_fill_rate": float(self.avg_fill_rate),
            "avg_slippage_bps": float(self.avg_slippage_bps),
            "avg_submission_latency_ms": self.avg_submission_latency_ms,
            "avg_ack_latency_ms": self.avg_ack_latency_ms,
            "avg_first_fill_latency_ms": self.avg_first_fill_latency_ms,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "positive_slippage_count": self.positive_slippage_count,
            "negative_slippage_count": self.negative_slippage_count,
            "zero_slippage_count": self.zero_slippage_count,
            "venue_breakdown": self.venue_breakdown,
            "symbol_breakdown": self.symbol_breakdown,
        }


@dataclass
class MonitorConfig:
    """监控配置"""
    # 滑点阈值
    slippage_warning_bps: Decimal = Decimal("10")  # 10bps
    slippage_critical_bps: Decimal = Decimal("25")  # 25bps

    # 延迟阈值
    latency_warning_ms: float = 100  # 100ms
    latency_critical_ms: float = 500  # 500ms

    # 成交率阈值
    fill_rate_warning: Decimal = Decimal("0.90")  # 90%
    fill_rate_critical: Decimal = Decimal("0.50")  # 50%

    # 监控间隔
    check_interval: float = 5.0  # 5秒
    stale_order_timeout: float = 300.0  # 5分钟

    # 告警设置
    enable_alerts: bool = True
    alert_cooldown: float = 60.0  # 1分钟


@dataclass
class ExecutionReport:
    """执行报告"""
    report_id: str
    report_type: str  # "daily", "weekly", "monthly", "custom"
    generated_at: datetime = field(default_factory=datetime.now)
    metrics: Optional[ExecutionMetrics] = None
    top_slippage_orders: List[Dict[str, Any]] = field(default_factory=list)
    top_latency_orders: List[Dict[str, Any]] = field(default_factory=list)
    venue_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    broker_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "generated_at": self.generated_at.isoformat(),
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "top_slippage_orders": self.top_slippage_orders,
            "top_latency_orders": self.top_latency_orders,
            "venue_performance": self.venue_performance,
            "broker_performance": self.broker_performance,
            "recommendations": self.recommendations,
        }


# ==================== 数据提供者接口 ====================

class MarketDataProvider(ABC):
    """市场数据提供者接口"""

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """获取当前价格"""
        pass

    @abstractmethod
    async def get_bid_ask(self, symbol: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """获取买卖价"""
        pass


class SimulatedMarketDataProvider(MarketDataProvider):
    """模拟市场数据提供者"""

    def __init__(self):
        self.prices: Dict[str, Decimal] = {}
        self.spreads: Dict[str, Decimal] = {}

    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        return self.prices.get(symbol)

    async def get_bid_ask(self, symbol: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        price = self.prices.get(symbol)
        if price is None:
            return None, None
        spread = self.spreads.get(symbol, Decimal("0.01"))
        half_spread = spread / 2
        return price - half_spread, price + half_spread

    def set_price(self, symbol: str, price: Decimal, spread: Decimal = Decimal("0.01")):
        """设置价格(测试用)"""
        self.prices[symbol] = price
        self.spreads[symbol] = spread


# ==================== 核心引擎 ====================

class ExecutionMonitor:
    """执行监控引擎"""

    def __init__(
        self,
        market_data_provider: Optional[MarketDataProvider] = None,
        config: Optional[MonitorConfig] = None,
    ):
        self.market_data = market_data_provider or simulated_default(SimulatedMarketDataProvider, "data from ExecutionMonitor's simulated market_data_provider (SimulatedMarketDataProvider)")
        self.config = config or MonitorConfig()

        # 订单存储
        self.orders: Dict[str, OrderExecution] = {}
        self.active_orders: Set[str] = set()
        self.completed_orders: List[str] = []

        # 告警
        self.alerts: List[ExecutionAlert] = []
        self._alert_cooldown: Dict[str, datetime] = {}

        # 状态
        self.state = MonitorState.IDLE
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # 回调
        self._on_alert: Optional[Callable[[ExecutionAlert], None]] = None
        self._on_order_complete: Optional[Callable[[OrderExecution], None]] = None
        self._on_fill: Optional[Callable[[OrderExecution, ExecutionFill], None]] = None

        # 统计
        self._stats = {
            "total_orders_tracked": 0,
            "total_fills": 0,
            "total_alerts": 0,
            "current_active_orders": 0,
        }

    # ==================== 回调注册 ====================

    def on_alert(self, callback: Callable[[ExecutionAlert], None]):
        """注册告警回调"""
        self._on_alert = callback

    def on_order_complete(self, callback: Callable[[OrderExecution], None]):
        """注册订单完成回调"""
        self._on_order_complete = callback

    def on_fill(self, callback: Callable[[OrderExecution, ExecutionFill], None]):
        """注册成交回调"""
        self._on_fill = callback

    # ==================== 订单管理 ====================

    def track_order(self, order: OrderExecution) -> str:
        """开始追踪订单"""
        self.orders[order.order_id] = order
        self.active_orders.add(order.order_id)
        self._stats["total_orders_tracked"] += 1
        self._stats["current_active_orders"] = len(self.active_orders)

        logger.info(f"Tracking order: {order.order_id} - {order.symbol} {order.side} {order.requested_quantity}")
        return order.order_id

    def get_order(self, order_id: str) -> Optional[OrderExecution]:
        """获取订单"""
        return self.orders.get(order_id)

    def update_order_state(
        self,
        order_id: str,
        state: OrderState,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """更新订单状态"""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        old_state = order.state
        order.state = state
        ts = timestamp or datetime.now()

        if state == OrderState.SUBMITTED and order.submitted_at is None:
            order.submitted_at = ts
        elif state == OrderState.ACKNOWLEDGED and order.acknowledged_at is None:
            order.acknowledged_at = ts
        elif state in [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED, OrderState.ERROR]:
            order.completed_at = ts
            self._complete_order(order)

        logger.debug(f"Order {order_id} state: {old_state.value} -> {state.value}")
        return True

    def record_fill(
        self,
        order_id: str,
        quantity: Decimal,
        price: Decimal,
        venue: ExecutionVenue = ExecutionVenue.EXCHANGE,
        timestamp: Optional[datetime] = None,
        fees: Decimal = Decimal("0"),
    ) -> Optional[str]:
        """记录成交"""
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]
        ts = timestamp or datetime.now()

        fill_id = str(uuid.uuid4())[:8]
        fill = ExecutionFill(
            fill_id=fill_id,
            order_id=order_id,
            quantity=quantity,
            price=price,
            venue=venue,
            timestamp=ts,
            fees=fees,
        )

        # 更新订单
        old_filled = order.filled_quantity
        order.fills.append(fill.to_dict())
        order.filled_quantity += quantity

        # 计算加权平均价
        if order.filled_quantity > 0:
            old_notional = old_filled * order.avg_fill_price
            new_notional = quantity * price
            order.avg_fill_price = (old_notional + new_notional) / order.filled_quantity

        # 首次成交时间
        if order.first_fill_at is None:
            order.first_fill_at = ts

        # 执行场所
        if order.venue is None:
            order.venue = venue

        # 更新状态
        if order.filled_quantity >= order.requested_quantity:
            order.state = OrderState.FILLED
            order.completed_at = ts
            self._complete_order(order)
        elif order.filled_quantity > 0:
            order.state = OrderState.PARTIAL_FILLED

        self._stats["total_fills"] += 1

        # 回调
        if self._on_fill:
            self._on_fill(order, fill)

        # 检查滑点
        self._check_fill_slippage(order, fill)

        logger.debug(f"Fill recorded: {order_id} - {quantity}@{price}")
        return fill_id

    def set_arrival_price(self, order_id: str, price: Decimal) -> bool:
        """设置到达价格"""
        if order_id not in self.orders:
            return False
        self.orders[order_id].arrival_price = price
        return True

    def _complete_order(self, order: OrderExecution):
        """完成订单处理"""
        if order.order_id in self.active_orders:
            self.active_orders.remove(order.order_id)
            self.completed_orders.append(order.order_id)
            self._stats["current_active_orders"] = len(self.active_orders)

        # 检查执行质量
        self._check_execution_quality(order)

        # 回调
        if self._on_order_complete:
            self._on_order_complete(order)

    # ==================== 执行质量检查 ====================

    def _check_fill_slippage(self, order: OrderExecution, fill: ExecutionFill):
        """检查成交滑点"""
        if not self.config.enable_alerts:
            return

        if order.arrival_price is None:
            return

        # 计算单次成交滑点
        if order.side == "buy":
            slippage = order.arrival_price - fill.price
        else:
            slippage = fill.price - order.arrival_price

        slippage_bps = abs(slippage / order.arrival_price * Decimal("10000"))

        if slippage_bps >= self.config.slippage_critical_bps:
            self._generate_alert(
                order.order_id,
                AlertSeverity.CRITICAL,
                "high_slippage",
                f"Critical slippage: {slippage_bps:.1f}bps on fill {fill.fill_id}",
                {"slippage_bps": float(slippage_bps), "fill_price": float(fill.price)}
            )
        elif slippage_bps >= self.config.slippage_warning_bps:
            self._generate_alert(
                order.order_id,
                AlertSeverity.WARNING,
                "high_slippage",
                f"High slippage: {slippage_bps:.1f}bps on fill {fill.fill_id}",
                {"slippage_bps": float(slippage_bps), "fill_price": float(fill.price)}
            )

    def _check_execution_quality(self, order: OrderExecution):
        """检查执行质量"""
        if not self.config.enable_alerts:
            return

        # 检查成交率
        if order.state == OrderState.FILLED:
            pass  # 完全成交，无需警告
        elif order.fill_rate < self.config.fill_rate_critical:
            self._generate_alert(
                order.order_id,
                AlertSeverity.CRITICAL,
                "low_fill_rate",
                f"Critical low fill rate: {order.fill_rate:.1%}",
                {"fill_rate": float(order.fill_rate)}
            )
        elif order.fill_rate < self.config.fill_rate_warning:
            self._generate_alert(
                order.order_id,
                AlertSeverity.WARNING,
                "low_fill_rate",
                f"Low fill rate: {order.fill_rate:.1%}",
                {"fill_rate": float(order.fill_rate)}
            )

        # 检查执行时间
        if order.total_execution_time_ms:
            if order.total_execution_time_ms >= self.config.latency_critical_ms:
                self._generate_alert(
                    order.order_id,
                    AlertSeverity.CRITICAL,
                    "slow_execution",
                    f"Critical slow execution: {order.total_execution_time_ms:.0f}ms",
                    {"execution_time_ms": order.total_execution_time_ms}
                )
            elif order.total_execution_time_ms >= self.config.latency_warning_ms:
                self._generate_alert(
                    order.order_id,
                    AlertSeverity.WARNING,
                    "slow_execution",
                    f"Slow execution: {order.total_execution_time_ms:.0f}ms",
                    {"execution_time_ms": order.total_execution_time_ms}
                )

    # ==================== 告警管理 ====================

    def _generate_alert(
        self,
        order_id: str,
        severity: AlertSeverity,
        alert_type: str,
        message: str,
        details: Dict[str, Any],
    ):
        """生成告警"""
        # 检查冷却
        cooldown_key = f"{order_id}_{alert_type}"
        if cooldown_key in self._alert_cooldown:
            if datetime.now() < self._alert_cooldown[cooldown_key]:
                return

        alert_id = str(uuid.uuid4())[:8]
        alert = ExecutionAlert(
            alert_id=alert_id,
            order_id=order_id,
            severity=severity,
            alert_type=alert_type,
            message=message,
            details=details,
        )

        self.alerts.append(alert)
        self._stats["total_alerts"] += 1

        # 设置冷却
        self._alert_cooldown[cooldown_key] = (
            datetime.now() + timedelta(seconds=self.config.alert_cooldown)
        )

        if self._on_alert:
            self._on_alert(alert)

        logger.warning(f"Execution alert: {message}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def get_unacknowledged_alerts(self) -> List[ExecutionAlert]:
        """获取未确认告警"""
        return [a for a in self.alerts if not a.acknowledged]

    def get_alerts(
        self,
        order_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[ExecutionAlert]:
        """获取告警"""
        alerts = self.alerts
        if order_id:
            alerts = [a for a in alerts if a.order_id == order_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

    # ==================== 指标计算 ====================

    def calculate_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ExecutionMetrics:
        """计算执行指标"""
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - timedelta(days=1))

        metrics = ExecutionMetrics(
            period_start=start_time,
            period_end=end_time,
        )

        # 过滤订单
        orders = [
            o for o in self.orders.values()
            if o.created_at >= start_time and o.created_at <= end_time
        ]

        if not orders:
            return metrics

        metrics.total_orders = len(orders)

        # 统计各类订单
        slippages = []
        submission_latencies = []
        ack_latencies = []
        first_fill_latencies = []
        execution_times = []

        for order in orders:
            metrics.total_quantity += order.requested_quantity
            metrics.filled_quantity += order.filled_quantity
            metrics.total_notional += order.notional

            if order.is_complete:
                metrics.completed_orders += 1

            if order.state == OrderState.FILLED:
                metrics.filled_orders += 1
            elif order.state == OrderState.CANCELLED:
                metrics.cancelled_orders += 1
            elif order.state == OrderState.REJECTED:
                metrics.rejected_orders += 1

            # 滑点统计
            if order.slippage_bps is not None:
                slippages.append(float(order.slippage_bps))
                if order.slippage_type == SlippageType.POSITIVE:
                    metrics.positive_slippage_count += 1
                elif order.slippage_type == SlippageType.NEGATIVE:
                    metrics.negative_slippage_count += 1
                else:
                    metrics.zero_slippage_count += 1

            # 延迟统计
            if order.submission_latency_ms is not None:
                submission_latencies.append(order.submission_latency_ms)
            if order.ack_latency_ms is not None:
                ack_latencies.append(order.ack_latency_ms)
            if order.first_fill_latency_ms is not None:
                first_fill_latencies.append(order.first_fill_latency_ms)
            if order.total_execution_time_ms is not None:
                execution_times.append(order.total_execution_time_ms)

            # 场所分布
            if order.venue:
                venue_name = order.venue.value
                metrics.venue_breakdown[venue_name] = metrics.venue_breakdown.get(venue_name, 0) + 1

            # 标的分布
            metrics.symbol_breakdown[order.symbol] = metrics.symbol_breakdown.get(order.symbol, 0) + 1

        # 计算平均值
        if metrics.total_quantity > 0:
            metrics.avg_fill_rate = metrics.filled_quantity / metrics.total_quantity

        if slippages:
            metrics.avg_slippage_bps = Decimal(str(statistics.mean(slippages)))

        if submission_latencies:
            metrics.avg_submission_latency_ms = statistics.mean(submission_latencies)
        if ack_latencies:
            metrics.avg_ack_latency_ms = statistics.mean(ack_latencies)
        if first_fill_latencies:
            metrics.avg_first_fill_latency_ms = statistics.mean(first_fill_latencies)
        if execution_times:
            metrics.avg_execution_time_ms = statistics.mean(execution_times)

        return metrics

    # ==================== 报告生成 ====================

    def generate_report(
        self,
        report_type: str = "daily",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ExecutionReport:
        """生成执行报告"""
        end_time = end_time or datetime.now()

        if report_type == "daily":
            start_time = start_time or (end_time - timedelta(days=1))
        elif report_type == "weekly":
            start_time = start_time or (end_time - timedelta(weeks=1))
        elif report_type == "monthly":
            start_time = start_time or (end_time - timedelta(days=30))
        else:
            start_time = start_time or (end_time - timedelta(days=1))

        report_id = str(uuid.uuid4())[:8]
        report = ExecutionReport(
            report_id=report_id,
            report_type=report_type,
        )

        # 计算指标
        report.metrics = self.calculate_metrics(start_time, end_time)

        # 过滤订单
        orders = [
            o for o in self.orders.values()
            if o.created_at >= start_time and o.created_at <= end_time and o.is_complete
        ]

        # Top滑点订单
        slippage_orders = sorted(
            [o for o in orders if o.slippage_bps is not None],
            key=lambda x: abs(x.slippage_bps or 0),
            reverse=True
        )[:10]
        report.top_slippage_orders = [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "slippage_bps": float(o.slippage_bps) if o.slippage_bps else 0,
                "notional": float(o.notional),
            }
            for o in slippage_orders
        ]

        # Top延迟订单
        latency_orders = sorted(
            [o for o in orders if o.total_execution_time_ms is not None],
            key=lambda x: x.total_execution_time_ms or 0,
            reverse=True
        )[:10]
        report.top_latency_orders = [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "execution_time_ms": o.total_execution_time_ms,
            }
            for o in latency_orders
        ]

        # 场所性能
        venue_orders: Dict[str, List[OrderExecution]] = {}
        for order in orders:
            if order.venue:
                venue_name = order.venue.value
                if venue_name not in venue_orders:
                    venue_orders[venue_name] = []
                venue_orders[venue_name].append(order)

        for venue_name, v_orders in venue_orders.items():
            slippages = [float(o.slippage_bps) for o in v_orders if o.slippage_bps is not None]
            latencies = [o.total_execution_time_ms for o in v_orders if o.total_execution_time_ms is not None]

            report.venue_performance[venue_name] = {
                "order_count": len(v_orders),
                "avg_slippage_bps": statistics.mean(slippages) if slippages else 0,
                "avg_execution_time_ms": statistics.mean(latencies) if latencies else 0,
            }

        # 生成建议
        report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: ExecutionReport) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if report.metrics:
            if report.metrics.avg_slippage_bps > self.config.slippage_warning_bps:
                recommendations.append(
                    f"Average slippage ({report.metrics.avg_slippage_bps:.1f}bps) exceeds warning threshold. "
                    "Consider using limit orders or reducing order sizes."
                )

            if report.metrics.avg_execution_time_ms > self.config.latency_warning_ms:
                recommendations.append(
                    f"Average execution time ({report.metrics.avg_execution_time_ms:.0f}ms) is slow. "
                    "Review connectivity and consider co-location."
                )

            if report.metrics.avg_fill_rate < self.config.fill_rate_warning:
                recommendations.append(
                    f"Average fill rate ({report.metrics.avg_fill_rate:.1%}) is below target. "
                    "Consider adjusting order pricing or using more aggressive order types."
                )

            if report.metrics.negative_slippage_count > report.metrics.positive_slippage_count * 2:
                recommendations.append(
                    "Negative slippage significantly outweighs positive slippage. "
                    "Review execution strategy and market timing."
                )

        # 场所建议
        if report.venue_performance:
            best_venue = min(
                report.venue_performance.items(),
                key=lambda x: x[1].get("avg_slippage_bps", float("inf"))
            )
            if best_venue[1].get("order_count", 0) > 10:
                recommendations.append(
                    f"Consider routing more orders to {best_venue[0]} for better execution."
                )

        if not recommendations:
            recommendations.append("Execution quality is within acceptable parameters.")

        return recommendations

    # ==================== 实时监控 ====================

    async def start(self):
        """启动监控"""
        if self._running:
            return

        self._running = True
        self.state = MonitorState.RUNNING
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Execution monitor started")

    async def stop(self):
        """停止监控"""
        self._running = False
        self.state = MonitorState.IDLE
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Execution monitor stopped")

    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                await self._check_stale_orders()
                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self.state = MonitorState.ERROR
                await asyncio.sleep(60)

    async def _check_stale_orders(self):
        """检查过期订单"""
        now = datetime.now()
        stale_threshold = timedelta(seconds=self.config.stale_order_timeout)

        for order_id in list(self.active_orders):
            order = self.orders.get(order_id)
            if order is None:
                continue

            if order.submitted_at and (now - order.submitted_at) > stale_threshold:
                if order.state not in [OrderState.FILLED, OrderState.PARTIAL_FILLED]:
                    self._generate_alert(
                        order_id,
                        AlertSeverity.WARNING,
                        "stale_order",
                        f"Order {order_id} has been active for over {self.config.stale_order_timeout/60:.0f} minutes",
                        {"time_active_seconds": (now - order.submitted_at).total_seconds()}
                    )

    # ==================== 查询接口 ====================

    def get_active_orders(self) -> List[OrderExecution]:
        """获取活跃订单"""
        return [self.orders[oid] for oid in self.active_orders if oid in self.orders]

    def get_completed_orders(self, limit: int = 100) -> List[OrderExecution]:
        """获取已完成订单"""
        order_ids = self.completed_orders[-limit:]
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_orders_by_symbol(self, symbol: str) -> List[OrderExecution]:
        """获取标的订单"""
        return [o for o in self.orders.values() if o.symbol == symbol]

    def get_orders_by_strategy(self, strategy_id: str) -> List[OrderExecution]:
        """获取策略订单"""
        return [o for o in self.orders.values() if o.strategy_id == strategy_id]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "state": self.state.value,
            "total_orders": len(self.orders),
            "active_orders": len(self.active_orders),
            "completed_orders": len(self.completed_orders),
            "unacknowledged_alerts": len(self.get_unacknowledged_alerts()),
        }


# ==================== 工厂函数 ====================

def create_execution_monitor(
    slippage_warning_bps: float = 10,
    slippage_critical_bps: float = 25,
    latency_warning_ms: float = 100,
    latency_critical_ms: float = 500,
) -> ExecutionMonitor:
    """创建执行监控引擎"""
    config = MonitorConfig(
        slippage_warning_bps=Decimal(str(slippage_warning_bps)),
        slippage_critical_bps=Decimal(str(slippage_critical_bps)),
        latency_warning_ms=latency_warning_ms,
        latency_critical_ms=latency_critical_ms,
    )

    return ExecutionMonitor(config=config)


def create_order_execution(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    price: Optional[float] = None,
    arrival_price: Optional[float] = None,
    strategy_id: Optional[str] = None,
    trader_id: Optional[str] = None,
    account_id: Optional[str] = None,
) -> OrderExecution:
    """创建订单执行记录"""
    return OrderExecution(
        order_id=str(uuid.uuid4())[:8],
        symbol=symbol,
        side=side,
        order_type=order_type,
        requested_quantity=Decimal(str(quantity)),
        requested_price=Decimal(str(price)) if price else None,
        arrival_price=Decimal(str(arrival_price)) if arrival_price else None,
        strategy_id=strategy_id,
        trader_id=trader_id,
        account_id=account_id,
    )


async def quick_execution_report(
    monitor: ExecutionMonitor,
    days: int = 1,
) -> Dict[str, Any]:
    """快速生成执行报告"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    report = monitor.generate_report(
        report_type="custom",
        start_time=start_time,
        end_time=end_time,
    )

    return report.to_dict()


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "OrderState",
    "ExecutionVenue",
    "SlippageType",
    "AlertSeverity",
    "MonitorState",
    # 数据类
    "OrderExecution",
    "ExecutionFill",
    "ExecutionAlert",
    "ExecutionMetrics",
    "MonitorConfig",
    "ExecutionReport",
    # 接口
    "MarketDataProvider",
    # 模拟实现
    "SimulatedMarketDataProvider",
    # 核心引擎
    "ExecutionMonitor",
    # 工厂函数
    "create_execution_monitor",
    "create_order_execution",
    "quick_execution_report",
]
