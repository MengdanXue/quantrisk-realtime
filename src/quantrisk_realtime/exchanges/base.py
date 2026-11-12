"""
交易所适配器基类 (Exchange Adapter Base)

定义交易所适配器的通用接口和数据结构

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"           # 待提交
    NEW = "new"                   # 新订单
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"             # 完全成交
    CANCELED = "canceled"         # 已取消
    REJECTED = "rejected"         # 已拒绝
    EXPIRED = "expired"           # 已过期


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"             # 市价单
    LIMIT = "limit"               # 限价单
    STOP_MARKET = "stop_market"   # 止损市价单
    STOP_LIMIT = "stop_limit"     # 止损限价单
    TAKE_PROFIT_MARKET = "take_profit_market"  # 止盈市价单
    TAKE_PROFIT_LIMIT = "take_profit_limit"    # 止盈限价单
    TRAILING_STOP = "trailing_stop"  # 追踪止损


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """订单有效期"""
    GTC = "gtc"     # Good Till Cancel
    IOC = "ioc"     # Immediate Or Cancel
    FOK = "fok"     # Fill Or Kill
    GTX = "gtx"     # Good Till Crossing (Post Only)
    GTD = "gtd"     # Good Till Date


class PositionSide(str, Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"   # 双向持仓


# ==================== 数据类 ====================

@dataclass
class Order:
    """订单"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    reduce_only: bool = False
    position_side: PositionSide = PositionSide.BOTH
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        """是否为未完成订单"""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING]

    @property
    def is_filled(self) -> bool:
        """是否已成交"""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> float:
        """剩余数量"""
        return self.quantity - self.filled_quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'time_in_force': self.time_in_force.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_open': self.is_open,
            'remaining_quantity': self.remaining_quantity,
        }


@dataclass
class Position:
    """持仓"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    mark_price: float = 0.0
    liquidation_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin: float = 0.0
    leverage: int = 1
    isolated: bool = False
    updated_at: Optional[datetime] = None

    @property
    def notional_value(self) -> float:
        """名义价值"""
        return abs(self.quantity) * self.mark_price

    @property
    def pnl_pct(self) -> float:
        """盈亏百分比"""
        if self.entry_price == 0:
            return 0
        if self.side == PositionSide.LONG:
            return (self.mark_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.mark_price) / self.entry_price * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'mark_price': self.mark_price,
            'liquidation_price': self.liquidation_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'margin': self.margin,
            'leverage': self.leverage,
            'notional_value': self.notional_value,
            'pnl_pct': self.pnl_pct,
        }


@dataclass
class Balance:
    """余额"""
    asset: str
    free: float
    locked: float
    total: float = 0.0

    def __post_init__(self):
        if self.total == 0:
            self.total = self.free + self.locked

    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset': self.asset,
            'free': self.free,
            'locked': self.locked,
            'total': self.total,
        }


@dataclass
class AccountInfo:
    """账户信息"""
    account_id: str
    balances: List[Balance]
    positions: List[Position] = field(default_factory=list)
    total_wallet_balance: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_margin_balance: float = 0.0
    available_balance: float = 0.0
    max_withdraw_amount: float = 0.0
    can_trade: bool = True
    can_withdraw: bool = True
    can_deposit: bool = True
    updated_at: Optional[datetime] = None

    def get_balance(self, asset: str) -> Optional[Balance]:
        """获取指定资产余额"""
        for b in self.balances:
            if b.asset == asset:
                return b
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'account_id': self.account_id,
            'balances': [b.to_dict() for b in self.balances],
            'positions': [p.to_dict() for p in self.positions],
            'total_wallet_balance': self.total_wallet_balance,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'available_balance': self.available_balance,
            'can_trade': self.can_trade,
        }


@dataclass
class Trade:
    """成交记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    commission: float
    commission_asset: str
    timestamp: datetime
    is_maker: bool = False
    realized_pnl: float = 0.0

    @property
    def value(self) -> float:
        """成交金额"""
        return self.price * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'price': self.price,
            'quantity': self.quantity,
            'value': self.value,
            'commission': self.commission,
            'commission_asset': self.commission_asset,
            'timestamp': self.timestamp.isoformat(),
            'is_maker': self.is_maker,
        }


@dataclass
class Ticker:
    """行情快照"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    change_24h: float
    change_pct_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime

    @property
    def mid_price(self) -> float:
        """中间价"""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """价差"""
        return self.ask - self.bid

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'volume_24h': self.volume_24h,
            'change_pct_24h': self.change_pct_24h,
            'timestamp': self.timestamp.isoformat(),
        }


# ==================== 配置类 ====================

@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: str = ""
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""         # 部分交易所需要
    testnet: bool = False        # 是否使用测试网
    timeout: float = 30.0
    recv_window: int = 5000
    rate_limit_per_second: float = 10.0
    rate_limit_per_minute: float = 1200.0


# ==================== 异常类 ====================

class ExchangeError(Exception):
    """交易所错误"""
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


class OrderError(ExchangeError):
    """订单错误"""
    pass


class InsufficientFundsError(OrderError):
    """资金不足"""
    pass


class RateLimitError(ExchangeError):
    """速率限制错误"""
    def __init__(self, message: str, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


# ==================== 适配器基类 ====================

class ExchangeAdapter(ABC):
    """交易所适配器基类"""

    def __init__(self, config: ExchangeConfig):
        self.config = config
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """交易所名称"""
        pass

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    # ==================== 连接管理 ====================

    @abstractmethod
    async def connect(self) -> bool:
        """连接交易所"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    # ==================== 账户相关 ====================

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        pass

    @abstractmethod
    async def get_balance(self, asset: str) -> Optional[Balance]:
        """获取资产余额"""
        pass

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取持仓"""
        pass

    # ==================== 订单管理 ====================

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        **kwargs,
    ) -> Order:
        """创建订单"""
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """取消订单"""
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """取消所有订单"""
        pass

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """获取订单"""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未完成订单"""
        pass

    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """获取历史订单"""
        pass

    # ==================== 市场数据 ====================

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """获取行情"""
        pass

    @abstractmethod
    async def get_tickers(self, symbols: Optional[List[str]] = None) -> List[Ticker]:
        """批量获取行情"""
        pass

    # ==================== 交易记录 ====================

    @abstractmethod
    async def get_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Trade]:
        """获取成交记录"""
        pass

    # ==================== 便捷方法 ====================

    async def market_buy(
        self,
        symbol: str,
        quantity: float,
        **kwargs,
    ) -> Order:
        """市价买入"""
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            **kwargs,
        )

    async def market_sell(
        self,
        symbol: str,
        quantity: float,
        **kwargs,
    ) -> Order:
        """市价卖出"""
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            **kwargs,
        )

    async def limit_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        **kwargs,
    ) -> Order:
        """限价买入"""
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            **kwargs,
        )

    async def limit_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        **kwargs,
    ) -> Order:
        """限价卖出"""
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            **kwargs,
        )

    async def close_position(
        self,
        symbol: str,
        position: Position,
    ) -> Order:
        """平仓"""
        side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        return await self.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
            reduce_only=True,
        )

    # ==================== 回调管理 ====================

    def on(self, event: str, callback: Callable) -> None:
        """注册事件回调"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """移除事件回调"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    async def _emit(self, event: str, data: Any) -> None:
        """触发事件"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"回调错误 {event}: {e}")


import asyncio
