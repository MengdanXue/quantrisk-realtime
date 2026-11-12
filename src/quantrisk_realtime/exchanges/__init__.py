"""
交易所适配器模块 (Exchange Adapter Module)

提供与交易所的实盘交易接口:
- 订单管理
- 持仓查询
- 账户信息
- 实时数据流

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from .base import (
    ExchangeAdapter,
    ExchangeConfig,
    ExchangeError,
    OrderError,
    InsufficientFundsError,
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    Position,
    Balance,
    AccountInfo,
    Trade,
    Ticker,
)

from .binance_adapter import (
    BinanceAdapter,
    BinanceConfig,
    BinanceFuturesAdapter,
    create_binance_spot,
    create_binance_futures,
)

from .binance_websocket import (
    BinanceWebSocket,
    WebSocketConfig,
    StreamType,
    KlineInterval,
    ConnectionState,
    TickerData,
    TradeData,
    KlineData,
    DepthData,
    DepthLevel,
    BookTickerData,
    MarkPriceData,
    UserBalanceUpdate,
    UserPositionUpdate,
    UserOrderUpdate,
    create_spot_websocket,
    create_futures_websocket,
)

__all__ = [
    # Base
    'ExchangeAdapter',
    'ExchangeConfig',
    'ExchangeError',
    'OrderError',
    'InsufficientFundsError',
    'Order',
    'OrderStatus',
    'OrderType',
    'OrderSide',
    'TimeInForce',
    'Position',
    'Balance',
    'AccountInfo',
    'Trade',
    'Ticker',
    # Binance REST
    'BinanceAdapter',
    'BinanceConfig',
    'BinanceFuturesAdapter',
    'create_binance_spot',
    'create_binance_futures',
    # Binance WebSocket
    'BinanceWebSocket',
    'WebSocketConfig',
    'StreamType',
    'KlineInterval',
    'ConnectionState',
    'TickerData',
    'TradeData',
    'KlineData',
    'DepthData',
    'DepthLevel',
    'BookTickerData',
    'MarkPriceData',
    'UserBalanceUpdate',
    'UserPositionUpdate',
    'UserOrderUpdate',
    'create_spot_websocket',
    'create_futures_websocket',
]
