"""
Binance WebSocket 适配器 (Binance WebSocket Adapter)

提供实时市场数据和账户更新流:
- 行情数据流 (Ticker, Depth, Trades)
- K线数据流 (Kline/Candlestick)
- 账户更新流 (Balance, Position, Order)
- 用户数据流 (User Data Stream)

WebSocket文档: https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Coroutine
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class StreamType(str, Enum):
    """数据流类型"""
    TICKER = "ticker"           # 24小时行情
    MINI_TICKER = "miniTicker"  # 精简行情
    TRADE = "trade"             # 成交
    AGG_TRADE = "aggTrade"      # 聚合成交
    KLINE = "kline"             # K线
    DEPTH = "depth"             # 深度
    DEPTH_UPDATE = "depthUpdate"  # 增量深度
    BOOK_TICKER = "bookTicker"  # 最优挂单
    MARK_PRICE = "markPrice"    # 标记价格
    FUNDING_RATE = "fundingRate"  # 资金费率
    USER_DATA = "userData"      # 用户数据


class KlineInterval(str, Enum):
    """K线间隔"""
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MO1 = "1M"


class ConnectionState(str, Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


# ==================== 数据类 ====================

@dataclass
class TickerData:
    """行情数据"""
    symbol: str
    price: float
    price_change: float
    price_change_pct: float
    high_24h: float
    low_24h: float
    volume_24h: float
    quote_volume_24h: float
    open_price: float
    close_price: float
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_binance(cls, data: Dict[str, Any]) -> 'TickerData':
        """从Binance数据创建"""
        return cls(
            symbol=data.get('s', ''),
            price=float(data.get('c', 0)),
            price_change=float(data.get('p', 0)),
            price_change_pct=float(data.get('P', 0)),
            high_24h=float(data.get('h', 0)),
            low_24h=float(data.get('l', 0)),
            volume_24h=float(data.get('v', 0)),
            quote_volume_24h=float(data.get('q', 0)),
            open_price=float(data.get('o', 0)),
            close_price=float(data.get('c', 0)),
            timestamp=datetime.fromtimestamp(data.get('E', 0) / 1000) if data.get('E') else datetime.now(),
        )


@dataclass
class TradeData:
    """成交数据"""
    symbol: str
    trade_id: int
    price: float
    quantity: float
    buyer_maker: bool
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_binance(cls, data: Dict[str, Any]) -> 'TradeData':
        """从Binance数据创建"""
        return cls(
            symbol=data.get('s', ''),
            trade_id=data.get('t', 0),
            price=float(data.get('p', 0)),
            quantity=float(data.get('q', 0)),
            buyer_maker=data.get('m', False),
            timestamp=datetime.fromtimestamp(data.get('T', 0) / 1000) if data.get('T') else datetime.now(),
        )


@dataclass
class KlineData:
    """K线数据"""
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    is_closed: bool

    @classmethod
    def from_binance(cls, data: Dict[str, Any]) -> 'KlineData':
        """从Binance数据创建"""
        k = data.get('k', {})
        return cls(
            symbol=data.get('s', ''),
            interval=k.get('i', ''),
            open_time=datetime.fromtimestamp(k.get('t', 0) / 1000),
            close_time=datetime.fromtimestamp(k.get('T', 0) / 1000),
            open=float(k.get('o', 0)),
            high=float(k.get('h', 0)),
            low=float(k.get('l', 0)),
            close=float(k.get('c', 0)),
            volume=float(k.get('v', 0)),
            quote_volume=float(k.get('q', 0)),
            trades=k.get('n', 0),
            is_closed=k.get('x', False),
        )


@dataclass
class DepthLevel:
    """深度层级"""
    price: float
    quantity: float

    @classmethod
    def from_list(cls, data: List) -> 'DepthLevel':
        """从列表创建"""
        return cls(
            price=float(data[0]),
            quantity=float(data[1]),
        )


@dataclass
class DepthData:
    """深度数据"""
    symbol: str
    bids: List[DepthLevel]
    asks: List[DepthLevel]
    last_update_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_binance(cls, data: Dict[str, Any], symbol: str = '') -> 'DepthData':
        """从Binance数据创建"""
        return cls(
            symbol=symbol or data.get('s', ''),
            bids=[DepthLevel.from_list(b) for b in data.get('b', data.get('bids', []))],
            asks=[DepthLevel.from_list(a) for a in data.get('a', data.get('asks', []))],
            last_update_id=data.get('u', data.get('lastUpdateId', 0)),
            timestamp=datetime.fromtimestamp(data.get('E', 0) / 1000) if data.get('E') else datetime.now(),
        )


@dataclass
class BookTickerData:
    """最优挂单数据"""
    symbol: str
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_binance(cls, data: Dict[str, Any]) -> 'BookTickerData':
        """从Binance数据创建"""
        return cls(
            symbol=data.get('s', ''),
            bid_price=float(data.get('b', 0)),
            bid_qty=float(data.get('B', 0)),
            ask_price=float(data.get('a', 0)),
            ask_qty=float(data.get('A', 0)),
            timestamp=datetime.now(),
        )


@dataclass
class MarkPriceData:
    """标记价格数据（合约）"""
    symbol: str
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_time: datetime
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_binance(cls, data: Dict[str, Any]) -> 'MarkPriceData':
        """从Binance数据创建"""
        return cls(
            symbol=data.get('s', ''),
            mark_price=float(data.get('p', 0)),
            index_price=float(data.get('i', 0)),
            funding_rate=float(data.get('r', 0)),
            next_funding_time=datetime.fromtimestamp(data.get('T', 0) / 1000) if data.get('T') else datetime.now(),
            timestamp=datetime.fromtimestamp(data.get('E', 0) / 1000) if data.get('E') else datetime.now(),
        )


@dataclass
class UserBalanceUpdate:
    """用户余额更新"""
    asset: str
    wallet_balance: float
    cross_wallet_balance: float
    balance_change: float
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_binance(cls, data: Dict[str, Any]) -> 'UserBalanceUpdate':
        """从Binance数据创建"""
        return cls(
            asset=data.get('a', ''),
            wallet_balance=float(data.get('wb', 0)),
            cross_wallet_balance=float(data.get('cw', 0)),
            balance_change=float(data.get('bc', 0)),
        )


@dataclass
class UserPositionUpdate:
    """用户持仓更新"""
    symbol: str
    position_amount: float
    entry_price: float
    unrealized_pnl: float
    margin_type: str
    isolated_margin: float
    position_side: str
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_binance(cls, data: Dict[str, Any]) -> 'UserPositionUpdate':
        """从Binance数据创建"""
        return cls(
            symbol=data.get('s', ''),
            position_amount=float(data.get('pa', 0)),
            entry_price=float(data.get('ep', 0)),
            unrealized_pnl=float(data.get('up', 0)),
            margin_type=data.get('mt', ''),
            isolated_margin=float(data.get('iw', 0)),
            position_side=data.get('ps', ''),
        )


@dataclass
class UserOrderUpdate:
    """用户订单更新"""
    symbol: str
    order_id: int
    client_order_id: str
    side: str
    order_type: str
    time_in_force: str
    quantity: float
    price: float
    stop_price: float
    status: str
    filled_qty: float
    avg_price: float
    commission: float
    commission_asset: str
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_binance(cls, data: Dict[str, Any]) -> 'UserOrderUpdate':
        """从Binance数据创建"""
        return cls(
            symbol=data.get('s', ''),
            order_id=data.get('i', 0),
            client_order_id=data.get('c', ''),
            side=data.get('S', ''),
            order_type=data.get('o', ''),
            time_in_force=data.get('f', ''),
            quantity=float(data.get('q', 0)),
            price=float(data.get('p', 0)),
            stop_price=float(data.get('sp', 0)),
            status=data.get('X', ''),
            filled_qty=float(data.get('z', 0)),
            avg_price=float(data.get('ap', 0)),
            commission=float(data.get('n', 0)),
            commission_asset=data.get('N', ''),
            timestamp=datetime.fromtimestamp(data.get('T', 0) / 1000) if data.get('T') else datetime.now(),
        )


# ==================== WebSocket配置 ====================

@dataclass
class WebSocketConfig:
    """WebSocket配置"""
    # 现货WebSocket
    spot_stream_url: str = "wss://stream.binance.com:9443/ws"
    spot_combined_url: str = "wss://stream.binance.com:9443/stream"
    spot_testnet_url: str = "wss://testnet.binance.vision/ws"

    # 合约WebSocket
    futures_stream_url: str = "wss://fstream.binance.com/ws"
    futures_combined_url: str = "wss://fstream.binance.com/stream"
    futures_testnet_url: str = "wss://stream.binancefuture.com/ws"

    # 连接设置
    ping_interval: float = 30.0       # Ping间隔（秒）
    ping_timeout: float = 10.0        # Ping超时（秒）
    reconnect_delay: float = 5.0      # 重连延迟（秒）
    max_reconnect_attempts: int = 10  # 最大重连次数
    message_timeout: float = 60.0     # 消息超时（秒）

    # API设置（用于获取listenKey）
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    is_futures: bool = False


# ==================== 回调类型 ====================

# 市场数据回调
TickerCallback = Callable[[TickerData], Coroutine[Any, Any, None]]
TradeCallback = Callable[[TradeData], Coroutine[Any, Any, None]]
KlineCallback = Callable[[KlineData], Coroutine[Any, Any, None]]
DepthCallback = Callable[[DepthData], Coroutine[Any, Any, None]]
BookTickerCallback = Callable[[BookTickerData], Coroutine[Any, Any, None]]
MarkPriceCallback = Callable[[MarkPriceData], Coroutine[Any, Any, None]]

# 用户数据回调
BalanceCallback = Callable[[UserBalanceUpdate], Coroutine[Any, Any, None]]
PositionCallback = Callable[[UserPositionUpdate], Coroutine[Any, Any, None]]
OrderCallback = Callable[[UserOrderUpdate], Coroutine[Any, Any, None]]

# 错误回调
ErrorCallback = Callable[[Exception], Coroutine[Any, Any, None]]


# ==================== WebSocket客户端 ====================

class BinanceWebSocket:
    """Binance WebSocket客户端"""

    def __init__(self, config: Optional[WebSocketConfig] = None):
        self.config = config or WebSocketConfig()
        self._state = ConnectionState.DISCONNECTED
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._subscriptions: Set[str] = set()
        self._reconnect_count = 0
        self._listen_key: Optional[str] = None
        self._listen_key_task: Optional[asyncio.Task] = None

        # 回调注册
        self._ticker_callbacks: List[TickerCallback] = []
        self._trade_callbacks: List[TradeCallback] = []
        self._kline_callbacks: List[KlineCallback] = []
        self._depth_callbacks: List[DepthCallback] = []
        self._book_ticker_callbacks: List[BookTickerCallback] = []
        self._mark_price_callbacks: List[MarkPriceCallback] = []
        self._balance_callbacks: List[BalanceCallback] = []
        self._position_callbacks: List[PositionCallback] = []
        self._order_callbacks: List[OrderCallback] = []
        self._error_callbacks: List[ErrorCallback] = []

        # 任务
        self._message_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> ConnectionState:
        """获取连接状态"""
        return self._state

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._state == ConnectionState.CONNECTED

    @property
    def subscriptions(self) -> Set[str]:
        """获取订阅列表"""
        return self._subscriptions.copy()

    def _get_base_url(self) -> str:
        """获取基础URL"""
        if self.config.is_futures:
            return self.config.futures_testnet_url if self.config.testnet else self.config.futures_stream_url
        return self.config.spot_testnet_url if self.config.testnet else self.config.spot_stream_url

    def _get_combined_url(self) -> str:
        """获取组合流URL"""
        if self.config.is_futures:
            return self.config.futures_testnet_url if self.config.testnet else self.config.futures_combined_url
        return self.config.spot_testnet_url if self.config.testnet else self.config.spot_combined_url

    async def connect(self) -> None:
        """建立WebSocket连接"""
        if self._state in (ConnectionState.CONNECTING, ConnectionState.CONNECTED):
            return

        self._state = ConnectionState.CONNECTING
        logger.info("正在连接 Binance WebSocket...")

        try:
            if not self._session:
                self._session = aiohttp.ClientSession()

            # 构建URL
            if self._subscriptions:
                streams = "/".join(self._subscriptions)
                url = f"{self._get_combined_url()}?streams={streams}"
            else:
                url = self._get_base_url()

            # 建立连接
            self._ws = await self._session.ws_connect(
                url,
                heartbeat=self.config.ping_interval,
                receive_timeout=self.config.message_timeout,
            )

            self._state = ConnectionState.CONNECTED
            self._reconnect_count = 0
            logger.info("Binance WebSocket 连接成功")

            # 启动消息处理
            self._message_task = asyncio.create_task(self._message_loop())
            self._ping_task = asyncio.create_task(self._ping_loop())

        except Exception as e:
            self._state = ConnectionState.DISCONNECTED
            logger.error(f"WebSocket 连接失败: {e}")
            await self._handle_error(e)
            raise

    async def disconnect(self) -> None:
        """断开WebSocket连接"""
        if self._state in (ConnectionState.DISCONNECTED, ConnectionState.CLOSED):
            return

        self._state = ConnectionState.CLOSING
        logger.info("正在断开 Binance WebSocket...")

        # 取消任务
        if self._message_task:
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._listen_key_task:
            self._listen_key_task.cancel()
            try:
                await self._listen_key_task
            except asyncio.CancelledError:
                pass

        # 关闭WebSocket
        if self._ws:
            await self._ws.close()
            self._ws = None

        # 关闭Session
        if self._session:
            await self._session.close()
            self._session = None

        self._state = ConnectionState.CLOSED
        logger.info("Binance WebSocket 已断开")

    async def reconnect(self) -> None:
        """重新连接"""
        if self._reconnect_count >= self.config.max_reconnect_attempts:
            logger.error(f"已达到最大重连次数 ({self.config.max_reconnect_attempts})")
            return

        self._state = ConnectionState.RECONNECTING
        self._reconnect_count += 1
        logger.info(f"正在重连 ({self._reconnect_count}/{self.config.max_reconnect_attempts})...")

        await asyncio.sleep(self.config.reconnect_delay)

        try:
            # 重新连接
            await self.connect()

            # 重新订阅
            if self._subscriptions:
                await self._send_subscribe(list(self._subscriptions))

        except Exception as e:
            logger.error(f"重连失败: {e}")
            await self.reconnect()

    async def _message_loop(self) -> None:
        """消息处理循环"""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket错误: {msg.data}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket连接已关闭")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"消息处理异常: {e}")
            await self._handle_error(e)

        # 尝试重连
        if self._state == ConnectionState.CONNECTED:
            self._state = ConnectionState.DISCONNECTED
            await self.reconnect()

    async def _ping_loop(self) -> None:
        """Ping循环"""
        try:
            while self._state == ConnectionState.CONNECTED:
                await asyncio.sleep(self.config.ping_interval)
                if self._ws:
                    await self._ws.ping()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Ping异常: {e}")

    async def _handle_message(self, data: str) -> None:
        """处理收到的消息"""
        try:
            msg = json.loads(data)

            # 组合流格式
            if 'stream' in msg:
                stream = msg['stream']
                data = msg['data']
            else:
                stream = msg.get('e', '')
                data = msg

            # 根据流类型分发
            event_type = data.get('e', '')

            if event_type == '24hrTicker' or '@ticker' in str(stream):
                await self._dispatch_ticker(data)
            elif event_type == 'trade' or '@trade' in str(stream):
                await self._dispatch_trade(data)
            elif event_type == 'aggTrade' or '@aggTrade' in str(stream):
                await self._dispatch_trade(data)
            elif event_type == 'kline' or '@kline' in str(stream):
                await self._dispatch_kline(data)
            elif event_type == 'depthUpdate' or '@depth' in str(stream):
                await self._dispatch_depth(data)
            elif event_type == 'bookTicker' or '@bookTicker' in str(stream):
                await self._dispatch_book_ticker(data)
            elif event_type == 'markPriceUpdate' or '@markPrice' in str(stream):
                await self._dispatch_mark_price(data)
            elif event_type == 'ACCOUNT_UPDATE':
                await self._dispatch_account_update(data)
            elif event_type == 'ORDER_TRADE_UPDATE':
                await self._dispatch_order_update(data)
            elif event_type == 'outboundAccountPosition':
                await self._dispatch_spot_account_update(data)
            elif event_type == 'executionReport':
                await self._dispatch_spot_order_update(data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
        except Exception as e:
            logger.error(f"消息处理错误: {e}")
            await self._handle_error(e)

    async def _dispatch_ticker(self, data: Dict[str, Any]) -> None:
        """分发Ticker数据"""
        ticker = TickerData.from_binance(data)
        for callback in self._ticker_callbacks:
            try:
                await callback(ticker)
            except Exception as e:
                logger.error(f"Ticker回调错误: {e}")

    async def _dispatch_trade(self, data: Dict[str, Any]) -> None:
        """分发成交数据"""
        trade = TradeData.from_binance(data)
        for callback in self._trade_callbacks:
            try:
                await callback(trade)
            except Exception as e:
                logger.error(f"Trade回调错误: {e}")

    async def _dispatch_kline(self, data: Dict[str, Any]) -> None:
        """分发K线数据"""
        kline = KlineData.from_binance(data)
        for callback in self._kline_callbacks:
            try:
                await callback(kline)
            except Exception as e:
                logger.error(f"Kline回调错误: {e}")

    async def _dispatch_depth(self, data: Dict[str, Any]) -> None:
        """分发深度数据"""
        depth = DepthData.from_binance(data)
        for callback in self._depth_callbacks:
            try:
                await callback(depth)
            except Exception as e:
                logger.error(f"Depth回调错误: {e}")

    async def _dispatch_book_ticker(self, data: Dict[str, Any]) -> None:
        """分发最优挂单数据"""
        book = BookTickerData.from_binance(data)
        for callback in self._book_ticker_callbacks:
            try:
                await callback(book)
            except Exception as e:
                logger.error(f"BookTicker回调错误: {e}")

    async def _dispatch_mark_price(self, data: Dict[str, Any]) -> None:
        """分发标记价格数据"""
        mark = MarkPriceData.from_binance(data)
        for callback in self._mark_price_callbacks:
            try:
                await callback(mark)
            except Exception as e:
                logger.error(f"MarkPrice回调错误: {e}")

    async def _dispatch_account_update(self, data: Dict[str, Any]) -> None:
        """分发账户更新（合约）"""
        update_data = data.get('a', {})

        # 余额更新
        for b in update_data.get('B', []):
            balance = UserBalanceUpdate.from_binance(b)
            for callback in self._balance_callbacks:
                try:
                    await callback(balance)
                except Exception as e:
                    logger.error(f"Balance回调错误: {e}")

        # 持仓更新
        for p in update_data.get('P', []):
            position = UserPositionUpdate.from_binance(p)
            for callback in self._position_callbacks:
                try:
                    await callback(position)
                except Exception as e:
                    logger.error(f"Position回调错误: {e}")

    async def _dispatch_order_update(self, data: Dict[str, Any]) -> None:
        """分发订单更新（合约）"""
        order_data = data.get('o', {})
        order = UserOrderUpdate.from_binance(order_data)
        for callback in self._order_callbacks:
            try:
                await callback(order)
            except Exception as e:
                logger.error(f"Order回调错误: {e}")

    async def _dispatch_spot_account_update(self, data: Dict[str, Any]) -> None:
        """分发账户更新（现货）"""
        for b in data.get('B', []):
            balance = UserBalanceUpdate(
                asset=b.get('a', ''),
                wallet_balance=float(b.get('f', 0)),
                cross_wallet_balance=float(b.get('l', 0)),
                balance_change=0,
            )
            for callback in self._balance_callbacks:
                try:
                    await callback(balance)
                except Exception as e:
                    logger.error(f"Balance回调错误: {e}")

    async def _dispatch_spot_order_update(self, data: Dict[str, Any]) -> None:
        """分发订单更新（现货）"""
        order = UserOrderUpdate(
            symbol=data.get('s', ''),
            order_id=data.get('i', 0),
            client_order_id=data.get('c', ''),
            side=data.get('S', ''),
            order_type=data.get('o', ''),
            time_in_force=data.get('f', ''),
            quantity=float(data.get('q', 0)),
            price=float(data.get('p', 0)),
            stop_price=float(data.get('P', 0)),
            status=data.get('X', ''),
            filled_qty=float(data.get('z', 0)),
            avg_price=0,
            commission=float(data.get('n', 0)),
            commission_asset=data.get('N', ''),
            timestamp=datetime.fromtimestamp(data.get('T', 0) / 1000) if data.get('T') else datetime.now(),
        )
        for callback in self._order_callbacks:
            try:
                await callback(order)
            except Exception as e:
                logger.error(f"Order回调错误: {e}")

    async def _handle_error(self, error: Exception) -> None:
        """处理错误"""
        for callback in self._error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"错误回调异常: {e}")

    async def _send_subscribe(self, streams: List[str]) -> None:
        """发送订阅请求"""
        if not self._ws:
            return

        msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000),
        }
        await self._ws.send_str(json.dumps(msg))

    async def _send_unsubscribe(self, streams: List[str]) -> None:
        """发送取消订阅请求"""
        if not self._ws:
            return

        msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000),
        }
        await self._ws.send_str(json.dumps(msg))

    # ==================== 订阅方法 ====================

    async def subscribe_ticker(self, symbol: str) -> None:
        """订阅24小时行情"""
        stream = f"{symbol.lower()}@ticker"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def subscribe_trade(self, symbol: str) -> None:
        """订阅成交数据"""
        stream = f"{symbol.lower()}@trade"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def subscribe_agg_trade(self, symbol: str) -> None:
        """订阅聚合成交"""
        stream = f"{symbol.lower()}@aggTrade"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def subscribe_kline(self, symbol: str, interval: KlineInterval) -> None:
        """订阅K线"""
        stream = f"{symbol.lower()}@kline_{interval.value}"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def subscribe_depth(self, symbol: str, levels: int = 20) -> None:
        """订阅深度数据"""
        stream = f"{symbol.lower()}@depth{levels}"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def subscribe_book_ticker(self, symbol: str) -> None:
        """订阅最优挂单"""
        stream = f"{symbol.lower()}@bookTicker"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def subscribe_mark_price(self, symbol: str) -> None:
        """订阅标记价格（合约）"""
        stream = f"{symbol.lower()}@markPrice"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def subscribe_all_tickers(self) -> None:
        """订阅所有行情"""
        stream = "!ticker@arr"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def subscribe_all_mini_tickers(self) -> None:
        """订阅所有精简行情"""
        stream = "!miniTicker@arr"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def subscribe_all_mark_prices(self) -> None:
        """订阅所有标记价格（合约）"""
        stream = "!markPrice@arr"
        self._subscriptions.add(stream)
        if self.is_connected:
            await self._send_subscribe([stream])

    async def unsubscribe(self, stream: str) -> None:
        """取消订阅"""
        if stream in self._subscriptions:
            self._subscriptions.remove(stream)
            if self.is_connected:
                await self._send_unsubscribe([stream])

    async def unsubscribe_all(self) -> None:
        """取消所有订阅"""
        if self._subscriptions and self.is_connected:
            await self._send_unsubscribe(list(self._subscriptions))
        self._subscriptions.clear()

    # ==================== 用户数据流 ====================

    async def subscribe_user_data(self) -> None:
        """订阅用户数据流"""
        if not self.config.api_key:
            raise ValueError("需要API密钥才能订阅用户数据流")

        # 获取listenKey
        self._listen_key = await self._create_listen_key()

        # 订阅
        stream = self._listen_key
        self._subscriptions.add(stream)

        if self.is_connected:
            await self._send_subscribe([stream])

        # 启动listenKey保活任务
        self._listen_key_task = asyncio.create_task(self._keepalive_listen_key())

    async def _create_listen_key(self) -> str:
        """创建listenKey"""
        if self.config.is_futures:
            url = "https://fapi.binance.com/fapi/v1/listenKey"
        else:
            url = "https://api.binance.com/api/v3/userDataStream"

        if self.config.testnet:
            if self.config.is_futures:
                url = "https://testnet.binancefuture.com/fapi/v1/listenKey"
            else:
                url = "https://testnet.binance.vision/api/v3/userDataStream"

        headers = {"X-MBX-APIKEY": self.config.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers) as resp:
                data = await resp.json()
                return data.get('listenKey', '')

    async def _keepalive_listen_key(self) -> None:
        """保活listenKey"""
        try:
            while self._state == ConnectionState.CONNECTED:
                await asyncio.sleep(30 * 60)  # 30分钟

                if self.config.is_futures:
                    url = "https://fapi.binance.com/fapi/v1/listenKey"
                else:
                    url = "https://api.binance.com/api/v3/userDataStream"

                if self.config.testnet:
                    if self.config.is_futures:
                        url = "https://testnet.binancefuture.com/fapi/v1/listenKey"
                    else:
                        url = "https://testnet.binance.vision/api/v3/userDataStream"

                headers = {"X-MBX-APIKEY": self.config.api_key}

                async with aiohttp.ClientSession() as session:
                    async with session.put(url, headers=headers) as resp:
                        if resp.status == 200:
                            logger.debug("ListenKey保活成功")
                        else:
                            logger.warning(f"ListenKey保活失败: {resp.status}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"ListenKey保活异常: {e}")

    # ==================== 回调注册 ====================

    def on_ticker(self, callback: TickerCallback) -> None:
        """注册Ticker回调"""
        self._ticker_callbacks.append(callback)

    def on_trade(self, callback: TradeCallback) -> None:
        """注册Trade回调"""
        self._trade_callbacks.append(callback)

    def on_kline(self, callback: KlineCallback) -> None:
        """注册Kline回调"""
        self._kline_callbacks.append(callback)

    def on_depth(self, callback: DepthCallback) -> None:
        """注册Depth回调"""
        self._depth_callbacks.append(callback)

    def on_book_ticker(self, callback: BookTickerCallback) -> None:
        """注册BookTicker回调"""
        self._book_ticker_callbacks.append(callback)

    def on_mark_price(self, callback: MarkPriceCallback) -> None:
        """注册MarkPrice回调"""
        self._mark_price_callbacks.append(callback)

    def on_balance_update(self, callback: BalanceCallback) -> None:
        """注册余额更新回调"""
        self._balance_callbacks.append(callback)

    def on_position_update(self, callback: PositionCallback) -> None:
        """注册持仓更新回调"""
        self._position_callbacks.append(callback)

    def on_order_update(self, callback: OrderCallback) -> None:
        """注册订单更新回调"""
        self._order_callbacks.append(callback)

    def on_error(self, callback: ErrorCallback) -> None:
        """注册错误回调"""
        self._error_callbacks.append(callback)


# ==================== 工厂函数 ====================

def create_spot_websocket(
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
) -> BinanceWebSocket:
    """创建现货WebSocket客户端"""
    config = WebSocketConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        is_futures=False,
    )
    return BinanceWebSocket(config)


def create_futures_websocket(
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
) -> BinanceWebSocket:
    """创建合约WebSocket客户端"""
    config = WebSocketConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        is_futures=True,
    )
    return BinanceWebSocket(config)
