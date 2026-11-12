"""
Binance交易所适配器 (Binance Exchange Adapter)

支持Binance现货和合约交易:
- 现货交易
- USDT永续合约
- 币本位永续合约

API文档: https://binance-docs.github.io/apidocs/

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import hashlib
import hmac
import time
import logging

import aiohttp

from .base import (
    ExchangeAdapter,
    ExchangeConfig,
    ExchangeError,
    OrderError,
    InsufficientFundsError,
    RateLimitError,
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    Position,
    PositionSide,
    Balance,
    AccountInfo,
    Trade,
    Ticker,
)

logger = logging.getLogger(__name__)


# ==================== 配置类 ====================

@dataclass
class BinanceConfig(ExchangeConfig):
    """Binance配置"""
    # 现货API
    spot_base_url: str = "https://api.binance.com"
    spot_testnet_url: str = "https://testnet.binance.vision"

    # 合约API
    futures_base_url: str = "https://fapi.binance.com"
    futures_testnet_url: str = "https://testnet.binancefuture.com"

    # 币本位合约
    delivery_base_url: str = "https://dapi.binance.com"
    delivery_testnet_url: str = "https://testnet.binancedelivery.com"

    def __post_init__(self):
        self.name = "binance"


# ==================== 状态映射 ====================

STATUS_MAP = {
    'NEW': OrderStatus.NEW,
    'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
    'FILLED': OrderStatus.FILLED,
    'CANCELED': OrderStatus.CANCELED,
    'REJECTED': OrderStatus.REJECTED,
    'EXPIRED': OrderStatus.EXPIRED,
    'PENDING_CANCEL': OrderStatus.CANCELED,
}

ORDER_TYPE_MAP = {
    OrderType.MARKET: 'MARKET',
    OrderType.LIMIT: 'LIMIT',
    OrderType.STOP_MARKET: 'STOP_MARKET',
    OrderType.STOP_LIMIT: 'STOP',
    OrderType.TAKE_PROFIT_MARKET: 'TAKE_PROFIT_MARKET',
    OrderType.TAKE_PROFIT_LIMIT: 'TAKE_PROFIT',
    OrderType.TRAILING_STOP: 'TRAILING_STOP_MARKET',
}

ORDER_TYPE_MAP_REVERSE = {v: k for k, v in ORDER_TYPE_MAP.items()}

TIF_MAP = {
    TimeInForce.GTC: 'GTC',
    TimeInForce.IOC: 'IOC',
    TimeInForce.FOK: 'FOK',
    TimeInForce.GTX: 'GTX',
}


# ==================== Binance现货适配器 ====================

class BinanceAdapter(ExchangeAdapter):
    """Binance现货适配器"""

    def __init__(self, config: Optional[BinanceConfig] = None):
        self._config = config or BinanceConfig()
        super().__init__(self._config)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return "binance_spot"

    @property
    def base_url(self) -> str:
        if self._config.testnet:
            return self._config.spot_testnet_url
        return self._config.spot_base_url

    # ==================== 连接管理 ====================

    async def connect(self) -> bool:
        """连接"""
        try:
            self._session = aiohttp.ClientSession()

            # 测试连接
            result = await self._request('GET', '/api/v3/ping')
            self._connected = True
            logger.info(f"Binance现货连接成功 (testnet={self._config.testnet})")
            return True

        except Exception as e:
            logger.error(f"Binance连接失败: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """断开"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Binance现货断开连接")

    # ==================== API请求 ====================

    def _sign(self, params: Dict[str, Any]) -> str:
        """生成签名"""
        query_string = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self._config.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
    ) -> Any:
        """发送请求"""
        if not self._session:
            self._session = aiohttp.ClientSession()

        url = f"{self.base_url}{endpoint}"
        params = params or {}
        headers = {}

        if self._config.api_key:
            headers['X-MBX-APIKEY'] = self._config.api_key

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = self._config.recv_window
            params['signature'] = self._sign(params)

        try:
            if method == 'GET':
                async with self._session.get(url, params=params, headers=headers) as resp:
                    return await self._handle_response(resp)
            elif method == 'POST':
                async with self._session.post(url, params=params, headers=headers) as resp:
                    return await self._handle_response(resp)
            elif method == 'DELETE':
                async with self._session.delete(url, params=params, headers=headers) as resp:
                    return await self._handle_response(resp)

        except aiohttp.ClientError as e:
            raise ExchangeError(f"网络错误: {e}")

    async def _handle_response(self, resp: aiohttp.ClientResponse) -> Any:
        """处理响应"""
        data = await resp.json()

        if resp.status == 429:
            raise RateLimitError("速率限制", retry_after=60)

        if resp.status != 200:
            code = data.get('code', resp.status)
            msg = data.get('msg', str(data))

            if code == -2010:
                raise InsufficientFundsError(msg, str(code))
            elif code in [-1013, -1021, -1022]:
                raise OrderError(msg, str(code))
            else:
                raise ExchangeError(msg, str(code))

        return data

    # ==================== 账户相关 ====================

    async def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        data = await self._request('GET', '/api/v3/account', signed=True)

        balances = [
            Balance(
                asset=b['asset'],
                free=float(b['free']),
                locked=float(b['locked']),
            )
            for b in data.get('balances', [])
            if float(b['free']) > 0 or float(b['locked']) > 0
        ]

        return AccountInfo(
            account_id=str(data.get('accountType', '')),
            balances=balances,
            can_trade=data.get('canTrade', True),
            can_withdraw=data.get('canWithdraw', True),
            can_deposit=data.get('canDeposit', True),
            updated_at=datetime.now(),
        )

    async def get_balance(self, asset: str) -> Optional[Balance]:
        """获取资产余额"""
        account = await self.get_account_info()
        return account.get_balance(asset)

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取持仓（现货无持仓概念，返回余额）"""
        # 现货不支持持仓概念
        return []

    # ==================== 订单管理 ====================

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
        params = {
            'symbol': symbol,
            'side': side.value.upper(),
            'type': ORDER_TYPE_MAP.get(order_type, 'MARKET'),
            'quantity': str(quantity),
        }

        if client_order_id:
            params['newClientOrderId'] = client_order_id

        # 限价单需要价格和TIF
        if order_type == OrderType.LIMIT:
            if price is None:
                raise OrderError("限价单需要价格")
            params['price'] = str(price)
            params['timeInForce'] = TIF_MAP.get(time_in_force, 'GTC')

        # 止损单需要止损价
        if order_type in [OrderType.STOP_LIMIT, OrderType.STOP_MARKET]:
            if stop_price is None:
                raise OrderError("止损单需要止损价")
            params['stopPrice'] = str(stop_price)
            if price:
                params['price'] = str(price)
            params['timeInForce'] = TIF_MAP.get(time_in_force, 'GTC')

        data = await self._request('POST', '/api/v3/order', params, signed=True)

        return self._parse_order(data)

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """取消订单"""
        params = {
            'symbol': symbol,
            'orderId': order_id,
        }
        data = await self._request('DELETE', '/api/v3/order', params, signed=True)
        return self._parse_order(data)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """取消所有订单"""
        if not symbol:
            raise OrderError("现货取消所有订单需要指定symbol")

        params = {'symbol': symbol}
        data = await self._request('DELETE', '/api/v3/openOrders', params, signed=True)

        return [self._parse_order(o) for o in data]

    async def get_order(self, symbol: str, order_id: str) -> Order:
        """获取订单"""
        params = {
            'symbol': symbol,
            'orderId': order_id,
        }
        data = await self._request('GET', '/api/v3/order', params, signed=True)
        return self._parse_order(data)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未完成订单"""
        params = {}
        if symbol:
            params['symbol'] = symbol

        data = await self._request('GET', '/api/v3/openOrders', params, signed=True)
        return [self._parse_order(o) for o in data]

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """获取历史订单"""
        if not symbol:
            raise OrderError("现货历史订单需要指定symbol")

        params = {
            'symbol': symbol,
            'limit': min(limit, 1000),
        }
        data = await self._request('GET', '/api/v3/allOrders', params, signed=True)
        return [self._parse_order(o) for o in data]

    def _parse_order(self, data: Dict) -> Order:
        """解析订单"""
        return Order(
            order_id=str(data.get('orderId', '')),
            client_order_id=data.get('clientOrderId', ''),
            symbol=data.get('symbol', ''),
            side=OrderSide(data.get('side', 'BUY').lower()),
            order_type=ORDER_TYPE_MAP_REVERSE.get(
                data.get('type', 'MARKET'),
                OrderType.MARKET
            ),
            quantity=float(data.get('origQty', 0)),
            price=float(data.get('price', 0)) or None,
            stop_price=float(data.get('stopPrice', 0)) or None,
            status=STATUS_MAP.get(data.get('status', 'NEW'), OrderStatus.NEW),
            filled_quantity=float(data.get('executedQty', 0)),
            avg_fill_price=float(data.get('avgPrice', 0)) or float(data.get('price', 0)),
            time_in_force=TimeInForce.GTC,
            created_at=datetime.fromtimestamp(data.get('time', 0) / 1000) if data.get('time') else None,
            updated_at=datetime.fromtimestamp(data.get('updateTime', 0) / 1000) if data.get('updateTime') else None,
        )

    # ==================== 市场数据 ====================

    async def get_ticker(self, symbol: str) -> Ticker:
        """获取行情"""
        data = await self._request('GET', '/api/v3/ticker/24hr', {'symbol': symbol})
        return self._parse_ticker(data)

    async def get_tickers(self, symbols: Optional[List[str]] = None) -> List[Ticker]:
        """批量获取行情"""
        params = {}
        if symbols:
            params['symbols'] = str(symbols)

        data = await self._request('GET', '/api/v3/ticker/24hr', params)

        if isinstance(data, list):
            return [self._parse_ticker(t) for t in data]
        return [self._parse_ticker(data)]

    def _parse_ticker(self, data: Dict) -> Ticker:
        """解析行情"""
        return Ticker(
            symbol=data.get('symbol', ''),
            bid=float(data.get('bidPrice', 0)),
            ask=float(data.get('askPrice', 0)),
            last=float(data.get('lastPrice', 0)),
            volume_24h=float(data.get('volume', 0)),
            change_24h=float(data.get('priceChange', 0)),
            change_pct_24h=float(data.get('priceChangePercent', 0)),
            high_24h=float(data.get('highPrice', 0)),
            low_24h=float(data.get('lowPrice', 0)),
            timestamp=datetime.now(),
        )

    # ==================== 交易记录 ====================

    async def get_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Trade]:
        """获取成交记录"""
        params = {
            'symbol': symbol,
            'limit': min(limit, 1000),
        }
        data = await self._request('GET', '/api/v3/myTrades', params, signed=True)

        return [
            Trade(
                trade_id=str(t.get('id', '')),
                order_id=str(t.get('orderId', '')),
                symbol=t.get('symbol', ''),
                side=OrderSide.BUY if t.get('isBuyer') else OrderSide.SELL,
                price=float(t.get('price', 0)),
                quantity=float(t.get('qty', 0)),
                commission=float(t.get('commission', 0)),
                commission_asset=t.get('commissionAsset', ''),
                timestamp=datetime.fromtimestamp(t.get('time', 0) / 1000),
                is_maker=t.get('isMaker', False),
            )
            for t in data
        ]


# ==================== Binance合约适配器 ====================

class BinanceFuturesAdapter(ExchangeAdapter):
    """Binance USDT永续合约适配器"""

    def __init__(self, config: Optional[BinanceConfig] = None):
        self._config = config or BinanceConfig()
        super().__init__(self._config)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return "binance_futures"

    @property
    def base_url(self) -> str:
        if self._config.testnet:
            return self._config.futures_testnet_url
        return self._config.futures_base_url

    # ==================== 连接管理 ====================

    async def connect(self) -> bool:
        """连接"""
        try:
            self._session = aiohttp.ClientSession()
            result = await self._request('GET', '/fapi/v1/ping')
            self._connected = True
            logger.info(f"Binance合约连接成功 (testnet={self._config.testnet})")
            return True
        except Exception as e:
            logger.error(f"Binance合约连接失败: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """断开"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    # ==================== API请求 ====================

    def _sign(self, params: Dict[str, Any]) -> str:
        """生成签名"""
        query_string = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self._config.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
    ) -> Any:
        """发送请求"""
        if not self._session:
            self._session = aiohttp.ClientSession()

        url = f"{self.base_url}{endpoint}"
        params = params or {}
        headers = {}

        if self._config.api_key:
            headers['X-MBX-APIKEY'] = self._config.api_key

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = self._config.recv_window
            params['signature'] = self._sign(params)

        try:
            if method == 'GET':
                async with self._session.get(url, params=params, headers=headers) as resp:
                    return await self._handle_response(resp)
            elif method == 'POST':
                async with self._session.post(url, params=params, headers=headers) as resp:
                    return await self._handle_response(resp)
            elif method == 'DELETE':
                async with self._session.delete(url, params=params, headers=headers) as resp:
                    return await self._handle_response(resp)

        except aiohttp.ClientError as e:
            raise ExchangeError(f"网络错误: {e}")

    async def _handle_response(self, resp: aiohttp.ClientResponse) -> Any:
        """处理响应"""
        data = await resp.json()

        if resp.status == 429:
            raise RateLimitError("速率限制", retry_after=60)

        if resp.status != 200:
            code = data.get('code', resp.status)
            msg = data.get('msg', str(data))

            if code == -2019:
                raise InsufficientFundsError(msg, str(code))
            elif code in [-1013, -1021, -1022, -4028]:
                raise OrderError(msg, str(code))
            else:
                raise ExchangeError(msg, str(code))

        return data

    # ==================== 账户相关 ====================

    async def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        data = await self._request('GET', '/fapi/v2/account', signed=True)

        balances = [
            Balance(
                asset=a['asset'],
                free=float(a['availableBalance']),
                locked=float(a['balance']) - float(a['availableBalance']),
                total=float(a['balance']),
            )
            for a in data.get('assets', [])
            if float(a['balance']) > 0
        ]

        positions = []
        for p in data.get('positions', []):
            qty = float(p.get('positionAmt', 0))
            if qty != 0:
                positions.append(Position(
                    symbol=p['symbol'],
                    side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                    quantity=abs(qty),
                    entry_price=float(p.get('entryPrice', 0)),
                    mark_price=float(p.get('markPrice', 0)),
                    unrealized_pnl=float(p.get('unrealizedProfit', 0)),
                    margin=float(p.get('initialMargin', 0)),
                    leverage=int(p.get('leverage', 1)),
                    isolated=p.get('isolated', False),
                ))

        return AccountInfo(
            account_id="futures",
            balances=balances,
            positions=positions,
            total_wallet_balance=float(data.get('totalWalletBalance', 0)),
            total_unrealized_pnl=float(data.get('totalUnrealizedProfit', 0)),
            total_margin_balance=float(data.get('totalMarginBalance', 0)),
            available_balance=float(data.get('availableBalance', 0)),
            max_withdraw_amount=float(data.get('maxWithdrawAmount', 0)),
            can_trade=data.get('canTrade', True),
            updated_at=datetime.now(),
        )

    async def get_balance(self, asset: str) -> Optional[Balance]:
        """获取资产余额"""
        account = await self.get_account_info()
        return account.get_balance(asset)

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取持仓"""
        account = await self.get_account_info()
        positions = account.positions

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]

        return positions

    # ==================== 订单管理 ====================

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
        position_side: PositionSide = PositionSide.BOTH,
        **kwargs,
    ) -> Order:
        """创建订单"""
        params = {
            'symbol': symbol,
            'side': side.value.upper(),
            'type': ORDER_TYPE_MAP.get(order_type, 'MARKET'),
            'quantity': str(quantity),
        }

        if client_order_id:
            params['newClientOrderId'] = client_order_id

        if position_side != PositionSide.BOTH:
            params['positionSide'] = position_side.value.upper()

        if reduce_only:
            params['reduceOnly'] = 'true'

        if order_type == OrderType.LIMIT:
            if price is None:
                raise OrderError("限价单需要价格")
            params['price'] = str(price)
            params['timeInForce'] = TIF_MAP.get(time_in_force, 'GTC')

        if order_type in [OrderType.STOP_LIMIT, OrderType.STOP_MARKET,
                          OrderType.TAKE_PROFIT_LIMIT, OrderType.TAKE_PROFIT_MARKET]:
            if stop_price is None:
                raise OrderError("止损/止盈单需要触发价")
            params['stopPrice'] = str(stop_price)
            if price:
                params['price'] = str(price)

        data = await self._request('POST', '/fapi/v1/order', params, signed=True)

        return self._parse_order(data)

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """取消订单"""
        params = {
            'symbol': symbol,
            'orderId': order_id,
        }
        data = await self._request('DELETE', '/fapi/v1/order', params, signed=True)
        return self._parse_order(data)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """取消所有订单"""
        if not symbol:
            raise OrderError("取消所有订单需要指定symbol")

        params = {'symbol': symbol}
        data = await self._request('DELETE', '/fapi/v1/allOpenOrders', params, signed=True)

        # 返回成功消息
        return []

    async def get_order(self, symbol: str, order_id: str) -> Order:
        """获取订单"""
        params = {
            'symbol': symbol,
            'orderId': order_id,
        }
        data = await self._request('GET', '/fapi/v1/order', params, signed=True)
        return self._parse_order(data)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未完成订单"""
        params = {}
        if symbol:
            params['symbol'] = symbol

        data = await self._request('GET', '/fapi/v1/openOrders', params, signed=True)
        return [self._parse_order(o) for o in data]

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """获取历史订单"""
        params = {'limit': min(limit, 1000)}
        if symbol:
            params['symbol'] = symbol

        data = await self._request('GET', '/fapi/v1/allOrders', params, signed=True)
        return [self._parse_order(o) for o in data]

    def _parse_order(self, data: Dict) -> Order:
        """解析订单"""
        return Order(
            order_id=str(data.get('orderId', '')),
            client_order_id=data.get('clientOrderId', ''),
            symbol=data.get('symbol', ''),
            side=OrderSide(data.get('side', 'BUY').lower()),
            order_type=ORDER_TYPE_MAP_REVERSE.get(
                data.get('type', 'MARKET'),
                OrderType.MARKET
            ),
            quantity=float(data.get('origQty', 0)),
            price=float(data.get('price', 0)) or None,
            stop_price=float(data.get('stopPrice', 0)) or None,
            status=STATUS_MAP.get(data.get('status', 'NEW'), OrderStatus.NEW),
            filled_quantity=float(data.get('executedQty', 0)),
            avg_fill_price=float(data.get('avgPrice', 0)),
            reduce_only=data.get('reduceOnly', False),
            position_side=PositionSide(data.get('positionSide', 'BOTH').lower()),
            created_at=datetime.fromtimestamp(data.get('time', 0) / 1000) if data.get('time') else None,
            updated_at=datetime.fromtimestamp(data.get('updateTime', 0) / 1000) if data.get('updateTime') else None,
        )

    # ==================== 市场数据 ====================

    async def get_ticker(self, symbol: str) -> Ticker:
        """获取行情"""
        data = await self._request('GET', '/fapi/v1/ticker/24hr', {'symbol': symbol})
        return self._parse_ticker(data)

    async def get_tickers(self, symbols: Optional[List[str]] = None) -> List[Ticker]:
        """批量获取行情"""
        data = await self._request('GET', '/fapi/v1/ticker/24hr')

        tickers = [self._parse_ticker(t) for t in data]
        if symbols:
            tickers = [t for t in tickers if t.symbol in symbols]
        return tickers

    def _parse_ticker(self, data: Dict) -> Ticker:
        """解析行情"""
        return Ticker(
            symbol=data.get('symbol', ''),
            bid=float(data.get('bidPrice', 0)),
            ask=float(data.get('askPrice', 0)),
            last=float(data.get('lastPrice', 0)),
            volume_24h=float(data.get('volume', 0)),
            change_24h=float(data.get('priceChange', 0)),
            change_pct_24h=float(data.get('priceChangePercent', 0)),
            high_24h=float(data.get('highPrice', 0)),
            low_24h=float(data.get('lowPrice', 0)),
            timestamp=datetime.now(),
        )

    # ==================== 交易记录 ====================

    async def get_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Trade]:
        """获取成交记录"""
        params = {
            'symbol': symbol,
            'limit': min(limit, 1000),
        }
        data = await self._request('GET', '/fapi/v1/userTrades', params, signed=True)

        return [
            Trade(
                trade_id=str(t.get('id', '')),
                order_id=str(t.get('orderId', '')),
                symbol=t.get('symbol', ''),
                side=OrderSide(t.get('side', 'BUY').lower()),
                price=float(t.get('price', 0)),
                quantity=float(t.get('qty', 0)),
                commission=float(t.get('commission', 0)),
                commission_asset=t.get('commissionAsset', ''),
                timestamp=datetime.fromtimestamp(t.get('time', 0) / 1000),
                is_maker=t.get('maker', False),
                realized_pnl=float(t.get('realizedPnl', 0)),
            )
            for t in data
        ]

    # ==================== 合约特有功能 ====================

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """设置杠杆"""
        params = {
            'symbol': symbol,
            'leverage': leverage,
        }
        return await self._request('POST', '/fapi/v1/leverage', params, signed=True)

    async def set_margin_type(self, symbol: str, isolated: bool) -> Dict[str, Any]:
        """设置保证金类型"""
        params = {
            'symbol': symbol,
            'marginType': 'ISOLATED' if isolated else 'CROSSED',
        }
        return await self._request('POST', '/fapi/v1/marginType', params, signed=True)

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """获取资金费率"""
        data = await self._request('GET', '/fapi/v1/premiumIndex', {'symbol': symbol})
        return {
            'symbol': symbol,
            'mark_price': float(data.get('markPrice', 0)),
            'funding_rate': float(data.get('lastFundingRate', 0)),
            'next_funding_time': datetime.fromtimestamp(data.get('nextFundingTime', 0) / 1000),
        }


# ==================== 工厂函数 ====================

def create_binance_spot(
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
) -> BinanceAdapter:
    """创建Binance现货适配器"""
    config = BinanceConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    return BinanceAdapter(config)


def create_binance_futures(
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
) -> BinanceFuturesAdapter:
    """创建Binance合约适配器"""
    config = BinanceConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    return BinanceFuturesAdapter(config)
