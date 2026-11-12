"""
CCXT 加密货币数据源 (CCXT Cryptocurrency Data Source)

通过CCXT库提供100+加密货币交易所的统一接口:
- 现货/期货/永续行情
- 订单簿数据
- 交易历史
- 资金费率
- 账户信息(需API密钥)

完全免费，无需API密钥获取公开数据。

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

from .base import (
    DataSource,
    DataSourceConfig,
    DataSourceState,
    DataSourceError,
    DataNotFoundError,
    RateLimitError,
    MarketType,
    DataFrequency,
    OHLCV,
    Quote,
)

logger = logging.getLogger(__name__)

# 尝试导入ccxt
try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("ccxt未安装，请运行: pip install ccxt")


# ==================== 枚举类型 ====================

class ExchangeType(str, Enum):
    """交易所类型"""
    BINANCE = "binance"
    BINANCE_FUTURES = "binanceusdm"
    BYBIT = "bybit"
    OKX = "okx"
    HUOBI = "huobi"
    KUCOIN = "kucoin"
    GATE = "gate"
    BITGET = "bitget"
    MEXC = "mexc"
    COINBASE = "coinbase"
    KRAKEN = "kraken"


class MarketTypeEx(str, Enum):
    """市场类型扩展"""
    SPOT = "spot"
    FUTURES = "futures"  # 交割期货
    SWAP = "swap"        # 永续合约
    OPTION = "option"


# ==================== 配置类 ====================

@dataclass
class CCXTConfig(DataSourceConfig):
    """CCXT配置"""
    name: str = "ccxt"
    rate_limit_per_second: float = 10.0
    rate_limit_per_minute: float = 300.0
    cache_ttl: float = 10.0

    # 交易所配置
    exchange: ExchangeType = ExchangeType.BINANCE
    api_key: Optional[str] = None
    secret: Optional[str] = None
    password: Optional[str] = None  # 某些交易所需要
    sandbox: bool = False

    # 超时配置
    timeout: float = 30000  # ccxt使用毫秒

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'exchange': self.exchange.value,
            'sandbox': self.sandbox,
            'has_api_key': self.api_key is not None,
        })
        return base


# ==================== 数据类 ====================

@dataclass
class Ticker:
    """Ticker数据"""
    symbol: str
    exchange: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    change: float
    change_pct: float
    vwap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat(),
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'quote_volume': self.quote_volume,
            'change': self.change,
            'change_pct': self.change_pct,
            'vwap': self.vwap,
        }


@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    amount: float


@dataclass
class OrderBook:
    """订单簿"""
    symbol: str
    exchange: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat(),
            'bids': [[l.price, l.amount] for l in self.bids[:10]],
            'asks': [[l.price, l.amount] for l in self.asks[:10]],
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'spread': self.spread,
        }


@dataclass
class FundingRate:
    """资金费率"""
    symbol: str
    exchange: str
    timestamp: datetime
    funding_rate: float
    next_funding_time: Optional[datetime] = None
    predicted_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat(),
            'funding_rate': self.funding_rate,
            'funding_rate_pct': self.funding_rate * 100,
            'annualized_rate': self.funding_rate * 3 * 365 * 100,  # 8小时一次
            'next_funding_time': self.next_funding_time.isoformat() if self.next_funding_time else None,
            'predicted_rate': self.predicted_rate,
        }


# ==================== 主类 ====================

class CCXTDataSource(DataSource):
    """CCXT加密货币数据源"""

    def __init__(self, config: Optional[CCXTConfig] = None):
        super().__init__(config or CCXTConfig())
        self._exchange: Optional[Any] = None
        self._markets: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return f"ccxt_{self.config.exchange.value}"

    @property
    def supported_markets(self) -> List[MarketType]:
        return [MarketType.CRYPTO]

    async def connect(self) -> bool:
        """连接交易所"""
        if not CCXT_AVAILABLE:
            logger.error("ccxt库未安装")
            self.state = DataSourceState.ERROR
            return False

        try:
            exchange_class = getattr(ccxt_async, self.config.exchange.value)

            exchange_config = {
                'timeout': self.config.timeout,
                'enableRateLimit': True,
            }

            if self.config.api_key:
                exchange_config['apiKey'] = self.config.api_key
            if self.config.secret:
                exchange_config['secret'] = self.config.secret
            if self.config.password:
                exchange_config['password'] = self.config.password
            if self.config.sandbox:
                exchange_config['sandbox'] = True

            self._exchange = exchange_class(exchange_config)

            # 加载市场
            self._markets = await self._exchange.load_markets()

            self.state = DataSourceState.RUNNING
            logger.info(f"CCXT已连接到{self.config.exchange.value}，加载{len(self._markets)}个交易对")
            return True

        except Exception as e:
            logger.error(f"连接交易所失败: {e}")
            self.state = DataSourceState.ERROR
            return False

    async def disconnect(self) -> None:
        """断开连接"""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
        self.state = DataSourceState.IDLE
        logger.info("CCXT连接已断开")

    # ==================== OHLCV数据 ====================

    async def fetch_ohlcv(
        self,
        symbol: str,
        frequency: DataFrequency = DataFrequency.DAILY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """获取OHLCV数据"""
        if not self._exchange:
            raise DataSourceError("Exchange not connected")

        cache_key = f"ohlcv:{symbol}:{frequency}:{limit}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_ohlcv_impl,
                symbol, frequency, start_date, limit
            )
            await self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"获取{symbol} OHLCV失败: {e}")
            raise DataSourceError(f"Failed to fetch OHLCV for {symbol}: {e}")

    async def _fetch_ohlcv_impl(
        self,
        symbol: str,
        frequency: DataFrequency,
        start_date: Optional[datetime],
        limit: int,
    ) -> List[OHLCV]:
        """OHLCV实现"""
        timeframe_map = {
            DataFrequency.MINUTE_1: '1m',
            DataFrequency.MINUTE_5: '5m',
            DataFrequency.MINUTE_15: '15m',
            DataFrequency.MINUTE_30: '30m',
            DataFrequency.HOUR_1: '1h',
            DataFrequency.HOUR_4: '4h',
            DataFrequency.DAILY: '1d',
            DataFrequency.WEEKLY: '1w',
        }

        timeframe = timeframe_map.get(frequency, '1d')
        since = int(start_date.timestamp() * 1000) if start_date else None

        ohlcv_data = await self._exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=limit
        )

        results = []
        for candle in ohlcv_data:
            results.append(OHLCV(
                timestamp=datetime.fromtimestamp(candle[0] / 1000),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[5]),
                symbol=symbol,
                source=self.name,
            ))

        return results

    # ==================== 实时报价 ====================

    async def fetch_quote(self, symbol: str) -> Optional[Quote]:
        """获取实时报价"""
        if not self._exchange:
            return None

        cache_key = f"quote:{symbol}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_quote_impl, symbol
            )
            if result:
                await self._set_cached(cache_key, result, ttl=3.0)
            return result
        except Exception as e:
            logger.error(f"获取{symbol}报价失败: {e}")
            return None

    async def _fetch_quote_impl(self, symbol: str) -> Optional[Quote]:
        """报价实现"""
        ticker = await self._exchange.fetch_ticker(symbol)

        return Quote(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker.get('timestamp') else datetime.now(),
            bid=float(ticker.get('bid') or 0),
            ask=float(ticker.get('ask') or 0),
            last_price=float(ticker.get('last') or 0),
            volume=float(ticker.get('baseVolume') or 0),
            source=self.name,
        )

    # ==================== Ticker ====================

    async def fetch_ticker(self, symbol: str) -> Optional[Ticker]:
        """获取完整Ticker"""
        if not self._exchange:
            return None

        try:
            ticker = await self._exchange.fetch_ticker(symbol)

            return Ticker(
                symbol=symbol,
                exchange=self.config.exchange.value,
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker.get('timestamp') else datetime.now(),
                bid=float(ticker.get('bid') or 0),
                ask=float(ticker.get('ask') or 0),
                last=float(ticker.get('last') or 0),
                open=float(ticker.get('open') or 0),
                high=float(ticker.get('high') or 0),
                low=float(ticker.get('low') or 0),
                close=float(ticker.get('close') or ticker.get('last') or 0),
                volume=float(ticker.get('baseVolume') or 0),
                quote_volume=float(ticker.get('quoteVolume') or 0),
                change=float(ticker.get('change') or 0),
                change_pct=float(ticker.get('percentage') or 0),
                vwap=float(ticker.get('vwap')) if ticker.get('vwap') else None,
            )
        except Exception as e:
            logger.error(f"获取{symbol} Ticker失败: {e}")
            return None

    async def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Ticker]:
        """批量获取Tickers"""
        if not self._exchange:
            return {}

        try:
            tickers = await self._exchange.fetch_tickers(symbols)

            results = {}
            for symbol, ticker in tickers.items():
                results[symbol] = Ticker(
                    symbol=symbol,
                    exchange=self.config.exchange.value,
                    timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker.get('timestamp') else datetime.now(),
                    bid=float(ticker.get('bid') or 0),
                    ask=float(ticker.get('ask') or 0),
                    last=float(ticker.get('last') or 0),
                    open=float(ticker.get('open') or 0),
                    high=float(ticker.get('high') or 0),
                    low=float(ticker.get('low') or 0),
                    close=float(ticker.get('close') or ticker.get('last') or 0),
                    volume=float(ticker.get('baseVolume') or 0),
                    quote_volume=float(ticker.get('quoteVolume') or 0),
                    change=float(ticker.get('change') or 0),
                    change_pct=float(ticker.get('percentage') or 0),
                )
            return results
        except Exception as e:
            logger.error(f"批量获取Tickers失败: {e}")
            return {}

    # ==================== 订单簿 ====================

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        """获取订单簿"""
        if not self._exchange:
            return None

        try:
            book = await self._exchange.fetch_order_book(symbol, limit)

            return OrderBook(
                symbol=symbol,
                exchange=self.config.exchange.value,
                timestamp=datetime.fromtimestamp(book['timestamp'] / 1000) if book.get('timestamp') else datetime.now(),
                bids=[OrderBookLevel(price=float(b[0]), amount=float(b[1])) for b in book['bids']],
                asks=[OrderBookLevel(price=float(a[0]), amount=float(a[1])) for a in book['asks']],
            )
        except Exception as e:
            logger.error(f"获取{symbol}订单簿失败: {e}")
            return None

    # ==================== 资金费率 ====================

    async def fetch_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """获取资金费率(永续合约)"""
        if not self._exchange:
            return None

        try:
            # 不是所有交易所都支持
            if not hasattr(self._exchange, 'fetch_funding_rate'):
                return None

            rate = await self._exchange.fetch_funding_rate(symbol)

            return FundingRate(
                symbol=symbol,
                exchange=self.config.exchange.value,
                timestamp=datetime.fromtimestamp(rate['timestamp'] / 1000) if rate.get('timestamp') else datetime.now(),
                funding_rate=float(rate.get('fundingRate') or 0),
                next_funding_time=datetime.fromtimestamp(rate['fundingTimestamp'] / 1000) if rate.get('fundingTimestamp') else None,
                predicted_rate=float(rate.get('predictedFundingRate')) if rate.get('predictedFundingRate') else None,
            )
        except Exception as e:
            logger.error(f"获取{symbol}资金费率失败: {e}")
            return None

    async def fetch_funding_rates(self, symbols: Optional[List[str]] = None) -> Dict[str, FundingRate]:
        """批量获取资金费率"""
        if not self._exchange:
            return {}

        try:
            if not hasattr(self._exchange, 'fetch_funding_rates'):
                # 逐个获取
                results = {}
                for symbol in (symbols or []):
                    rate = await self.fetch_funding_rate(symbol)
                    if rate:
                        results[symbol] = rate
                return results

            rates = await self._exchange.fetch_funding_rates(symbols)

            results = {}
            for symbol, rate in rates.items():
                results[symbol] = FundingRate(
                    symbol=symbol,
                    exchange=self.config.exchange.value,
                    timestamp=datetime.fromtimestamp(rate['timestamp'] / 1000) if rate.get('timestamp') else datetime.now(),
                    funding_rate=float(rate.get('fundingRate') or 0),
                )
            return results
        except Exception as e:
            logger.error(f"批量获取资金费率失败: {e}")
            return {}

    # ==================== 交易对信息 ====================

    def get_markets(self) -> Dict[str, Any]:
        """获取所有交易对"""
        return self._markets

    def get_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取单个交易对信息"""
        return self._markets.get(symbol)

    def get_symbols(self, market_type: Optional[MarketTypeEx] = None) -> List[str]:
        """获取交易对列表"""
        symbols = []
        for symbol, market in self._markets.items():
            if market_type:
                if market_type.value == market.get('type'):
                    symbols.append(symbol)
            else:
                symbols.append(symbol)
        return symbols


# ==================== 工厂函数 ====================

def create_ccxt_source(
    exchange: ExchangeType = ExchangeType.BINANCE,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    sandbox: bool = False,
) -> CCXTDataSource:
    """创建CCXT数据源"""
    config = CCXTConfig(
        exchange=exchange,
        api_key=api_key,
        secret=secret,
        sandbox=sandbox,
    )
    return CCXTDataSource(config)
