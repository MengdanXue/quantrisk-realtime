"""
订单管理系统 (OMS)
Order Management System

提供完整的订单生命周期管理:
- 订单创建、验证、提交
- 订单状态追踪
- 订单修改和取消
- 订单历史和审计
- 仓位管理
- 风险检查

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ============ Enums ============

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"           # 市价单
    LIMIT = "limit"             # 限价单
    STOP = "stop"               # 止损单
    STOP_LIMIT = "stop_limit"   # 止损限价单
    TRAILING_STOP = "trailing"  # 追踪止损单
    IOC = "ioc"                 # 立即成交否则取消
    FOK = "fok"                 # 全部成交否则取消
    GTC = "gtc"                 # 撤销前有效
    GTD = "gtd"                 # 指定日期前有效


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"             # 待提交
    SUBMITTED = "submitted"         # 已提交
    ACCEPTED = "accepted"           # 已接受
    PARTIAL_FILLED = "partial"      # 部分成交
    FILLED = "filled"               # 完全成交
    CANCELLED = "cancelled"         # 已取消
    REJECTED = "rejected"           # 已拒绝
    EXPIRED = "expired"             # 已过期
    PENDING_CANCEL = "pending_cancel"  # 待取消


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class RiskCheckResult(Enum):
    """风险检查结果"""
    PASS = "pass"
    REJECT_EXPOSURE = "reject_exposure"
    REJECT_CONCENTRATION = "reject_concentration"
    REJECT_MARGIN = "reject_margin"
    REJECT_DAILY_LIMIT = "reject_daily_limit"
    REJECT_POSITION_LIMIT = "reject_position_limit"
    REJECT_ORDER_SIZE = "reject_order_size"


class TimeInForce(Enum):
    """订单有效期"""
    DAY = "day"             # 当日有效
    GTC = "gtc"             # 撤销前有效
    IOC = "ioc"             # 立即成交否则取消
    FOK = "fok"             # 全部成交否则取消
    GTD = "gtd"             # 指定日期前有效


# ============ Data Classes ============

@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_amount: Optional[float] = None     # 追踪止损金额
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_time: Optional[datetime] = None
    portfolio_id: Optional[int] = None
    # Internal authorization state. Deliberately omitted from to_dict().
    owner_user_id: Optional[int] = None
    strategy_id: Optional[str] = None
    client_order_id: Optional[str] = None

    # 状态信息
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # 元数据
    tags: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: Optional[str] = None

    @property
    def remaining_quantity(self) -> float:
        """剩余数量"""
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """是否活跃"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL_FILLED,
        ]

    @property
    def is_complete(self) -> bool:
        """是否完成"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    @property
    def notional_value(self) -> float:
        """名义价值"""
        price = self.limit_price or self.filled_price or self.stop_price or 0
        return self.quantity * price

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'average_price': self.average_price,
            'commission': self.commission,
            'remaining_quantity': self.remaining_quantity,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'portfolio_id': self.portfolio_id,
            'strategy_id': self.strategy_id,
        }


@dataclass
class Fill:
    """成交记录"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: Optional[str] = None
    execution_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class Position:
    """持仓"""
    symbol: str
    side: PositionSide
    quantity: float
    average_price: float
    market_price: float = 0.0
    portfolio_id: Optional[int] = None

    # 盈亏
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # 风控
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # 时间戳
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        """市值"""
        return self.quantity * self.market_price

    @property
    def cost_basis(self) -> float:
        """成本基础"""
        return self.quantity * self.average_price

    @property
    def pnl_percent(self) -> float:
        """盈亏百分比"""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    def update_market_price(self, price: float):
        """更新市价"""
        self.market_price = price
        self.updated_at = datetime.now()

        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.average_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.average_price - price) * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'market_price': self.market_price,
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'pnl_percent': self.pnl_percent,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'opened_at': self.opened_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


@dataclass
class RiskLimits:
    """风险限制"""
    max_position_value: float = 1_000_000       # 单个持仓最大价值
    max_position_pct: float = 0.20              # 单个持仓最大占比
    max_total_exposure: float = 5_000_000       # 总敞口上限
    max_daily_loss: float = 50_000              # 日最大亏损
    max_daily_trades: int = 100                 # 日最大交易次数
    max_order_size: float = 100_000             # 单笔订单最大金额
    max_concentration: float = 0.30             # 最大集中度
    margin_requirement: float = 0.25            # 保证金要求


@dataclass
class OMSConfig:
    """OMS配置"""
    enable_risk_checks: bool = True
    enable_position_limits: bool = True
    enable_order_validation: bool = True
    default_commission_rate: float = 0.001
    default_slippage_rate: float = 0.0005
    order_timeout_seconds: int = 86400          # 订单超时时间
    max_open_orders: int = 1000
    allow_short_selling: bool = True
    auto_cancel_on_disconnect: bool = True


# ============ Risk Checker ============

class RiskChecker:
    """风险检查器"""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._daily_trades: int = 0
        self._daily_pnl: float = 0.0
        self._last_reset: datetime = datetime.now()

    def check_order(
        self,
        order: Order,
        positions: Dict[str, Position],
        portfolio_value: float,
    ) -> Tuple[RiskCheckResult, str]:
        """
        检查订单风险

        Args:
            order: 待检查订单
            positions: 当前持仓
            portfolio_value: 组合总价值

        Returns:
            (检查结果, 原因描述)
        """
        # 重置日计数
        self._reset_daily_counters()

        # 检查订单大小
        if order.notional_value > self.limits.max_order_size:
            return RiskCheckResult.REJECT_ORDER_SIZE, \
                f"Order size {order.notional_value} exceeds limit {self.limits.max_order_size}"

        # 检查日交易次数
        if self._daily_trades >= self.limits.max_daily_trades:
            return RiskCheckResult.REJECT_DAILY_LIMIT, \
                f"Daily trade limit {self.limits.max_daily_trades} reached"

        # 检查持仓限制
        symbol = order.symbol
        current_position = positions.get(symbol)
        new_position_value = order.notional_value

        if current_position:
            if order.side == OrderSide.BUY:
                new_position_value = current_position.market_value + order.notional_value
            else:
                new_position_value = abs(current_position.market_value - order.notional_value)

        if new_position_value > self.limits.max_position_value:
            return RiskCheckResult.REJECT_POSITION_LIMIT, \
                f"Position value {new_position_value} exceeds limit {self.limits.max_position_value}"

        # 检查持仓占比
        if portfolio_value > 0:
            position_pct = new_position_value / portfolio_value
            if position_pct > self.limits.max_position_pct:
                return RiskCheckResult.REJECT_CONCENTRATION, \
                    f"Position concentration {position_pct:.2%} exceeds limit {self.limits.max_position_pct:.2%}"

        # 检查总敞口
        total_exposure = sum(p.market_value for p in positions.values())
        total_exposure += order.notional_value
        if total_exposure > self.limits.max_total_exposure:
            return RiskCheckResult.REJECT_EXPOSURE, \
                f"Total exposure {total_exposure} exceeds limit {self.limits.max_total_exposure}"

        # 检查日亏损
        if self._daily_pnl < -self.limits.max_daily_loss:
            return RiskCheckResult.REJECT_DAILY_LIMIT, \
                f"Daily loss {-self._daily_pnl} exceeds limit {self.limits.max_daily_loss}"

        return RiskCheckResult.PASS, "Risk check passed"

    def record_trade(self, pnl: float = 0.0):
        """记录交易"""
        self._daily_trades += 1
        self._daily_pnl += pnl

    def _reset_daily_counters(self):
        """重置日计数器"""
        now = datetime.now()
        if now.date() > self._last_reset.date():
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._last_reset = now


# ============ Position Manager ============

class PositionManager:
    """仓位管理器"""

    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._position_history: List[Dict[str, Any]] = []

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""
        return self._positions.copy()

    def get_active_positions(self) -> List[Position]:
        """获取活跃持仓"""
        return [p for p in self._positions.values() if p.quantity > 0]

    def update_position(self, fill: Fill) -> Position:
        """
        更新持仓

        Args:
            fill: 成交记录

        Returns:
            更新后的持仓
        """
        symbol = fill.symbol
        position = self._positions.get(symbol)

        if position is None:
            # 新建持仓
            side = PositionSide.LONG if fill.side == OrderSide.BUY else PositionSide.SHORT
            position = Position(
                symbol=symbol,
                side=side,
                quantity=fill.quantity,
                average_price=fill.price,
                market_price=fill.price,
            )
            self._positions[symbol] = position
        else:
            # 更新现有持仓
            if fill.side == OrderSide.BUY:
                if position.side == PositionSide.LONG or position.quantity == 0:
                    # 加仓或新开多仓
                    total_cost = position.quantity * position.average_price + fill.quantity * fill.price
                    position.quantity += fill.quantity
                    position.average_price = total_cost / position.quantity if position.quantity > 0 else 0
                    position.side = PositionSide.LONG
                else:
                    # 平空仓
                    realized = (position.average_price - fill.price) * min(fill.quantity, position.quantity)
                    position.realized_pnl += realized
                    position.quantity -= fill.quantity
                    if position.quantity < 0:
                        # 反向开多仓
                        position.side = PositionSide.LONG
                        position.quantity = abs(position.quantity)
                        position.average_price = fill.price
                    elif position.quantity == 0:
                        position.side = PositionSide.FLAT
            else:  # SELL
                if position.side == PositionSide.SHORT or position.quantity == 0:
                    # 加仓或新开空仓
                    total_cost = position.quantity * position.average_price + fill.quantity * fill.price
                    position.quantity += fill.quantity
                    position.average_price = total_cost / position.quantity if position.quantity > 0 else 0
                    position.side = PositionSide.SHORT
                else:
                    # 平多仓
                    realized = (fill.price - position.average_price) * min(fill.quantity, position.quantity)
                    position.realized_pnl += realized
                    position.quantity -= fill.quantity
                    if position.quantity < 0:
                        # 反向开空仓
                        position.side = PositionSide.SHORT
                        position.quantity = abs(position.quantity)
                        position.average_price = fill.price
                    elif position.quantity == 0:
                        position.side = PositionSide.FLAT

            position.updated_at = datetime.now()

        # 记录历史
        self._position_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'fill': fill.to_dict(),
            'position': position.to_dict(),
        })

        return position

    def update_market_prices(self, prices: Dict[str, float]):
        """批量更新市价"""
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].update_market_price(price)

    def get_total_exposure(self) -> float:
        """获取总敞口"""
        return sum(abs(p.market_value) for p in self._positions.values())

    def get_total_pnl(self) -> Tuple[float, float]:
        """获取总盈亏 (未实现, 已实现)"""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        realized = sum(p.realized_pnl for p in self._positions.values())
        return unrealized, realized

    def close_position(self, symbol: str, price: float) -> Optional[float]:
        """
        平仓

        Args:
            symbol: 标的代码
            price: 平仓价格

        Returns:
            实现盈亏
        """
        position = self._positions.get(symbol)
        if not position or position.quantity == 0:
            return None

        # 计算实现盈亏
        if position.side == PositionSide.LONG:
            realized = (price - position.average_price) * position.quantity
        else:
            realized = (position.average_price - price) * position.quantity

        position.realized_pnl += realized
        position.quantity = 0
        position.side = PositionSide.FLAT
        position.updated_at = datetime.now()

        return realized

    def get_position_history(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓历史"""
        if symbol:
            return [h for h in self._position_history if h['symbol'] == symbol]
        return self._position_history.copy()


# ============ Order Manager ============

class OrderManager:
    """订单管理器"""

    def __init__(self, config: Optional[OMSConfig] = None):
        self.config = config or OMSConfig()
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._fills: List[Fill] = []

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        portfolio_id: Optional[int] = None,
        owner_user_id: Optional[int] = None,
        strategy_id: Optional[str] = None,
        **kwargs
    ) -> Order:
        """
        创建订单

        Args:
            symbol: 标的代码
            side: 买卖方向
            quantity: 数量
            order_type: 订单类型
            limit_price: 限价
            stop_price: 止损价
            time_in_force: 有效期
            portfolio_id: 组合ID
            strategy_id: 策略ID

        Returns:
            创建的订单
        """
        order_id = (
            f"ORD-{datetime.now().strftime('%Y%m%d')}-"
            f"{uuid.uuid4().hex.upper()}"
        )

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            portfolio_id=portfolio_id,
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
            client_order_id=kwargs.get('client_order_id'),
            tags=kwargs.get('tags', {}),
        )

        # 验证订单
        if self.config.enable_order_validation:
            self._validate_order(order)

        self._orders[order_id] = order
        return order

    def _validate_order(self, order: Order):
        """验证订单"""
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            raise ValueError("Limit order requires limit_price")

        if order.order_type == OrderType.STOP and order.stop_price is None:
            raise ValueError("Stop order requires stop_price")

        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                raise ValueError("Stop limit order requires both stop_price and limit_price")

    def submit_order(self, order_id: str) -> Order:
        """
        提交订单

        Args:
            order_id: 订单ID

        Returns:
            更新后的订单
        """
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Order {order_id} is not in PENDING status")

        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()

        logger.info(f"Order submitted: {order_id}")
        return order

    def accept_order(self, order_id: str) -> Order:
        """接受订单"""
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        order.status = OrderStatus.ACCEPTED
        return order

    def fill_order(
        self,
        order_id: str,
        fill_quantity: float,
        fill_price: float,
        commission: float = 0.0,
    ) -> Tuple[Order, Fill]:
        """
        成交订单（改进版：增强验证和并发控制）

        改进点:
        1. 严格验证fill数量不能超过剩余数量（抛出异常而非静默截断）
        2. 验证fill数量和价格必须为正数
        3. 防止浮点精度问题导致的状态错误
        4. 改进平均价格计算的数值稳定性

        Args:
            order_id: 订单ID
            fill_quantity: 成交数量
            fill_price: 成交价格
            commission: 手续费

        Returns:
            (更新后的订单, 成交记录)

        Raises:
            ValueError: 订单不存在、不活跃、成交数量非法等
        """
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        if not order.is_active:
            raise ValueError(f"Order {order_id} is not active (status: {order.status.value})")

        # 验证成交数量（严格检查，不能超过剩余数量）
        if fill_quantity <= 0:
            raise ValueError(f"Fill quantity must be positive, got {fill_quantity}")

        # 允许微小的浮点误差（1e-6）
        if fill_quantity > order.remaining_quantity + 1e-6:
            raise ValueError(
                f"Fill quantity {fill_quantity} exceeds remaining quantity "
                f"{order.remaining_quantity} for order {order_id}"
            )

        # 验证成交价格
        if fill_price <= 0:
            raise ValueError(f"Fill price must be positive, got {fill_price}")

        # 处理浮点精度问题：如果剩余数量很小，直接完全成交
        if abs(fill_quantity - order.remaining_quantity) < 1e-6:
            fill_quantity = order.remaining_quantity

        # 创建成交记录
        fill = Fill(
            fill_id=f"FILL-{uuid.uuid4().hex[:8].upper()}",
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
        )
        self._fills.append(fill)

        # 更新订单（使用局部变量避免中间状态）
        old_filled = order.filled_quantity
        old_avg_price = order.average_price
        new_filled = old_filled + fill_quantity
        new_commission = order.commission + commission

        # 计算加权平均价格（避免除零错误）
        if new_filled > 0:
            new_avg_price = (old_filled * old_avg_price + fill_quantity * fill_price) / new_filled
        else:
            new_avg_price = 0.0

        # 原子更新订单状态
        order.filled_quantity = new_filled
        order.average_price = new_avg_price
        order.filled_price = fill_price
        order.commission = new_commission

        # 更新订单状态（处理浮点精度）
        if order.filled_quantity >= order.quantity - 1e-6:
            order.filled_quantity = order.quantity  # 确保精确相等
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            self._order_history.append(order)
            logger.info(f"Order {order_id} fully filled: {new_filled}/{order.quantity} @ avg {new_avg_price:.2f}")
        else:
            order.status = OrderStatus.PARTIAL_FILLED
            logger.info(
                f"Order {order_id} partially filled: {new_filled}/{order.quantity} "
                f"(remaining: {order.remaining_quantity:.6f}) @ avg {new_avg_price:.2f}"
            )

        return order, fill

    def cancel_order(self, order_id: str, reason: str = "User cancelled") -> Order:
        """
        取消订单

        Args:
            order_id: 订单ID
            reason: 取消原因

        Returns:
            更新后的订单
        """
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        if order.is_complete:
            raise ValueError(f"Order {order_id} is already complete")

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()
        order.rejection_reason = reason
        self._order_history.append(order)

        logger.info(f"Order cancelled: {order_id}, reason={reason}")
        return order

    def reject_order(self, order_id: str, reason: str) -> Order:
        """
        拒绝订单

        Args:
            order_id: 订单ID
            reason: 拒绝原因

        Returns:
            更新后的订单
        """
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        order.status = OrderStatus.REJECTED
        order.rejection_reason = reason
        self._order_history.append(order)

        logger.warning(f"Order rejected: {order_id}, reason={reason}")
        return order

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """
        修改订单

        Args:
            order_id: 订单ID
            quantity: 新数量
            limit_price: 新限价
            stop_price: 新止损价

        Returns:
            更新后的订单
        """
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        if order.status not in [OrderStatus.PENDING, OrderStatus.ACCEPTED]:
            raise ValueError(f"Order {order_id} cannot be modified in {order.status} status")

        if quantity is not None:
            if quantity < order.filled_quantity:
                raise ValueError("New quantity cannot be less than filled quantity")
            order.quantity = quantity

        if limit_price is not None:
            order.limit_price = limit_price

        if stop_price is not None:
            order.stop_price = stop_price

        logger.info(f"Order modified: {order_id}")
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self._orders.get(order_id)

    def get_order_for_owner(self, order_id: str, owner_user_id: int) -> Optional[Order]:
        """Return an order only when its immutable in-memory owner matches."""
        order = self._orders.get(order_id)
        if not order or order.owner_user_id != owner_user_id:
            return None
        return order

    def get_orders_for_owner(
        self,
        owner_user_id: int,
        *,
        active_only: bool = False,
    ) -> List[Order]:
        """List only orders explicitly owned by a user; legacy unowned rows fail closed."""
        orders = [
            order
            for order in self._orders.values()
            if order.owner_user_id == owner_user_id
        ]
        if active_only:
            orders = [order for order in orders if order.is_active]
        return orders

    def cancel_order_for_owner(
        self,
        order_id: str,
        owner_user_id: int,
        reason: str = "User cancelled",
    ) -> Optional[Order]:
        """Cancel an owned order, returning None for absent or foreign orders."""
        order = self.get_order_for_owner(order_id, owner_user_id)
        if order is None:
            return None
        if order.status is OrderStatus.CANCELLED:
            return order
        return self.cancel_order(order_id, reason)

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """按标的获取订单"""
        return [o for o in self._orders.values() if o.symbol == symbol]

    def get_active_orders(self) -> List[Order]:
        """获取活跃订单"""
        return [o for o in self._orders.values() if o.is_active]

    def get_pending_orders(self) -> List[Order]:
        """获取待处理订单"""
        return [o for o in self._orders.values() if o.status == OrderStatus.PENDING]

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Order]:
        """获取订单历史"""
        orders = self._order_history.copy()

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        if start_time:
            orders = [o for o in orders if o.created_at >= start_time]

        if end_time:
            orders = [o for o in orders if o.created_at <= end_time]

        return orders[-limit:]

    def get_fills(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Fill]:
        """获取成交记录"""
        fills = self._fills.copy()

        if order_id:
            fills = [f for f in fills if f.order_id == order_id]

        if symbol:
            fills = [f for f in fills if f.symbol == symbol]

        return fills[-limit:]

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        取消所有订单

        Args:
            symbol: 可选，只取消指定标的的订单

        Returns:
            取消的订单数量
        """
        count = 0
        for order in list(self._orders.values()):
            if order.is_active:
                if symbol is None or order.symbol == symbol:
                    self.cancel_order(order.order_id, "Bulk cancel")
                    count += 1
        return count

    def expire_orders(self) -> int:
        """
        过期超时订单

        Returns:
            过期的订单数量
        """
        count = 0
        now = datetime.now()
        timeout = timedelta(seconds=self.config.order_timeout_seconds)

        for order in list(self._orders.values()):
            if order.is_active:
                if order.time_in_force == TimeInForce.DAY:
                    # 日内订单：收盘后过期
                    if order.submitted_at and (now - order.submitted_at) > timeout:
                        order.status = OrderStatus.EXPIRED
                        self._order_history.append(order)
                        count += 1
                elif order.time_in_force == TimeInForce.GTD:
                    # GTD订单：到期过期
                    if order.expire_time and now > order.expire_time:
                        order.status = OrderStatus.EXPIRED
                        self._order_history.append(order)
                        count += 1

        return count

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        active_orders = self.get_active_orders()
        filled_orders = [o for o in self._order_history if o.status == OrderStatus.FILLED]
        cancelled_orders = [o for o in self._order_history if o.status == OrderStatus.CANCELLED]
        rejected_orders = [o for o in self._order_history if o.status == OrderStatus.REJECTED]

        total_commission = sum(o.commission for o in filled_orders)
        total_filled_value = sum(o.filled_quantity * o.average_price for o in filled_orders)

        return {
            'total_orders': len(self._orders) + len(self._order_history),
            'active_orders': len(active_orders),
            'filled_orders': len(filled_orders),
            'cancelled_orders': len(cancelled_orders),
            'rejected_orders': len(rejected_orders),
            'total_fills': len(self._fills),
            'total_commission': total_commission,
            'total_filled_value': total_filled_value,
        }


# ============ OMS (Main Class) ============

class OMS:
    """
    订单管理系统主类

    整合订单管理、仓位管理和风险控制
    """

    def __init__(
        self,
        config: Optional[OMSConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.config = config or OMSConfig()
        self.order_manager = OrderManager(self.config)
        self.position_manager = PositionManager()
        self.risk_checker = RiskChecker(risk_limits)

        self._portfolio_value: float = 1_000_000
        self._callbacks: Dict[str, List[Callable]] = {
            'on_order_created': [],
            'on_order_filled': [],
            'on_order_cancelled': [],
            'on_position_changed': [],
        }

        self._running: bool = False
        self._lock = asyncio.Lock()

    async def start(self):
        """启动OMS"""
        self._running = True
        logger.info("OMS started")

    async def stop(self):
        """停止OMS"""
        self._running = False
        # 取消所有活跃订单
        if self.config.auto_cancel_on_disconnect:
            self.order_manager.cancel_all_orders()
        logger.info("OMS stopped")

    def set_portfolio_value(self, value: float):
        """设置组合价值"""
        self._portfolio_value = value

    def register_callback(self, event: str, callback: Callable):
        """注册回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        提交订单

        Args:
            symbol: 标的代码
            side: 方向 (buy/sell)
            quantity: 数量
            order_type: 类型 (market/limit/stop/stop_limit)
            limit_price: 限价
            stop_price: 止损价

        Returns:
            创建的订单
        """
        async with self._lock:
            # 转换枚举
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            otype = OrderType(order_type.lower())

            # 创建订单
            order = self.order_manager.create_order(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                order_type=otype,
                limit_price=limit_price,
                stop_price=stop_price,
                **kwargs
            )

            # 风险检查
            if self.config.enable_risk_checks:
                positions = self.position_manager.get_all_positions()
                result, reason = self.risk_checker.check_order(
                    order, positions, self._portfolio_value
                )
                if result != RiskCheckResult.PASS:
                    self.order_manager.reject_order(order.order_id, reason)
                    return order

            # 提交订单
            self.order_manager.submit_order(order.order_id)

            # 触发回调
            for cb in self._callbacks['on_order_created']:
                try:
                    await cb(order) if asyncio.iscoroutinefunction(cb) else cb(order)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            return order

    async def execute_fill(
        self,
        order_id: str,
        fill_quantity: float,
        fill_price: float,
        commission: Optional[float] = None,
    ) -> Tuple[Order, Fill, Position]:
        """
        执行成交

        Args:
            order_id: 订单ID
            fill_quantity: 成交数量
            fill_price: 成交价格
            commission: 手续费

        Returns:
            (订单, 成交记录, 持仓)
        """
        async with self._lock:
            if commission is None:
                commission = fill_quantity * fill_price * self.config.default_commission_rate

            # 成交订单
            order, fill = self.order_manager.fill_order(
                order_id, fill_quantity, fill_price, commission
            )

            # 更新持仓
            position = self.position_manager.update_position(fill)

            # 记录交易
            self.risk_checker.record_trade(position.realized_pnl)

            # 触发回调
            for cb in self._callbacks['on_order_filled']:
                try:
                    await cb(order, fill) if asyncio.iscoroutinefunction(cb) else cb(order, fill)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            for cb in self._callbacks['on_position_changed']:
                try:
                    await cb(position) if asyncio.iscoroutinefunction(cb) else cb(position)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            return order, fill, position

    async def cancel_order(self, order_id: str, reason: str = "User cancelled") -> Order:
        """取消订单"""
        async with self._lock:
            order = self.order_manager.cancel_order(order_id, reason)

            for cb in self._callbacks['on_order_cancelled']:
                try:
                    await cb(order) if asyncio.iscoroutinefunction(cb) else cb(order)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            return order

    async def cancel_order_for_owner(
        self,
        order_id: str,
        owner_user_id: int,
        reason: str = "User cancelled",
    ) -> Optional[Order]:
        """Cancel within an owner scope and keep repeat cancellation idempotent."""
        async with self._lock:
            existing = self.order_manager.get_order_for_owner(order_id, owner_user_id)
            if existing is None:
                return None

            was_cancelled = existing.status is OrderStatus.CANCELLED
            order = self.order_manager.cancel_order_for_owner(
                order_id,
                owner_user_id,
                reason,
            )
            if order is not None and not was_cancelled:
                for cb in self._callbacks['on_order_cancelled']:
                    try:
                        await cb(order) if asyncio.iscoroutinefunction(cb) else cb(order)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.order_manager.get_order(order_id)

    def get_order_for_owner(self, order_id: str, owner_user_id: int) -> Optional[Order]:
        """获取归属于指定用户的订单"""
        return self.order_manager.get_order_for_owner(order_id, owner_user_id)

    def get_orders_for_owner(
        self,
        owner_user_id: int,
        *,
        active_only: bool = False,
    ) -> List[Order]:
        """获取归属于指定用户的订单列表"""
        return self.order_manager.get_orders_for_owner(
            owner_user_id,
            active_only=active_only,
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.position_manager.get_position(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""
        return self.position_manager.get_all_positions()

    def get_active_orders(self) -> List[Order]:
        """获取活跃订单"""
        return self.order_manager.get_active_orders()

    def update_market_prices(self, prices: Dict[str, float]):
        """更新市价"""
        self.position_manager.update_market_prices(prices)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取组合摘要"""
        positions = self.position_manager.get_active_positions()
        unrealized, realized = self.position_manager.get_total_pnl()
        total_exposure = self.position_manager.get_total_exposure()
        order_stats = self.order_manager.get_statistics()

        return {
            'portfolio_value': self._portfolio_value,
            'total_exposure': total_exposure,
            'exposure_ratio': total_exposure / self._portfolio_value if self._portfolio_value > 0 else 0,
            'position_count': len(positions),
            'unrealized_pnl': unrealized,
            'realized_pnl': realized,
            'total_pnl': unrealized + realized,
            'orders': order_stats,
            'positions': [p.to_dict() for p in positions],
        }


# ============ Factory Functions ============

def create_oms(
    enable_risk_checks: bool = True,
    max_position_pct: float = 0.20,
    max_daily_trades: int = 100,
    commission_rate: float = 0.001,
    **kwargs
) -> OMS:
    """
    创建OMS实例

    Args:
        enable_risk_checks: 启用风险检查
        max_position_pct: 最大仓位占比
        max_daily_trades: 日最大交易次数
        commission_rate: 手续费率

    Returns:
        OMS实例
    """
    config = OMSConfig(
        enable_risk_checks=enable_risk_checks,
        default_commission_rate=commission_rate,
        **{k: v for k, v in kwargs.items() if k in OMSConfig.__dataclass_fields__}
    )

    risk_limits = RiskLimits(
        max_position_pct=max_position_pct,
        max_daily_trades=max_daily_trades,
        **{k: v for k, v in kwargs.items() if k in RiskLimits.__dataclass_fields__}
    )

    return OMS(config, risk_limits)


def create_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    **kwargs
) -> Order:
    """创建订单对象（不提交）"""
    order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
    otype = OrderType(order_type.lower())

    return Order(
        order_id=(
            f"ORD-{datetime.now().strftime('%Y%m%d')}-"
            f"{uuid.uuid4().hex.upper()}"
        ),
        symbol=symbol,
        side=order_side,
        quantity=quantity,
        order_type=otype,
        limit_price=kwargs.get('limit_price'),
        stop_price=kwargs.get('stop_price'),
    )


__all__ = [
    # Enums
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'PositionSide',
    'RiskCheckResult',
    'TimeInForce',
    # Data Classes
    'Order',
    'Fill',
    'Position',
    'RiskLimits',
    'OMSConfig',
    # Classes
    'RiskChecker',
    'PositionManager',
    'OrderManager',
    'OMS',
    # Factory Functions
    'create_oms',
    'create_order',
]
