"""
CoinGecko 加密货币数据源 (CoinGecko Cryptocurrency Data Source)

提供加密货币市场数据:
- 代币价格和市值
- 历史价格数据
- 市场概览
- 恐惧贪婪指数
- 交易所信息
- 趋势代币

完全免费，无需API密钥，50次/分钟。

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import aiohttp

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


# ==================== 常量 ====================

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_PRO_URL = "https://pro-api.coingecko.com/api/v3"


# ==================== 枚举类型 ====================

class CoinCategory(str, Enum):
    """代币类别"""
    DEFI = "decentralized-finance-defi"
    NFT = "non-fungible-tokens-nft"
    METAVERSE = "metaverse"
    GAMING = "gaming"
    LAYER1 = "layer-1"
    LAYER2 = "layer-2"
    MEME = "meme-token"
    STABLECOIN = "stablecoins"
    EXCHANGE_TOKEN = "exchange-based-tokens"


class VsCurrency(str, Enum):
    """计价货币"""
    USD = "usd"
    EUR = "eur"
    CNY = "cny"
    JPY = "jpy"
    BTC = "btc"
    ETH = "eth"


# ==================== 配置类 ====================

@dataclass
class CoinGeckoConfig(DataSourceConfig):
    """CoinGecko配置"""
    name: str = "coingecko"
    rate_limit_per_second: float = 0.8  # 50/分钟 ≈ 0.83/秒
    rate_limit_per_minute: float = 50.0
    cache_ttl: float = 60.0

    # API配置
    api_key: Optional[str] = None  # Pro版需要
    vs_currency: VsCurrency = VsCurrency.USD

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'vs_currency': self.vs_currency.value,
            'is_pro': self.api_key is not None,
        })
        return base


# ==================== 数据类 ====================

@dataclass
class CoinInfo:
    """代币信息"""
    id: str
    symbol: str
    name: str
    market_cap_rank: Optional[int] = None
    categories: List[str] = field(default_factory=list)
    description: str = ""
    homepage: str = ""
    github: str = ""
    twitter: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'name': self.name,
            'market_cap_rank': self.market_cap_rank,
            'categories': self.categories,
            'description': self.description[:200] if self.description else "",
            'homepage': self.homepage,
            'github': self.github,
            'twitter': self.twitter,
        }


@dataclass
class CoinMarketData:
    """代币市场数据"""
    id: str
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int
    total_volume: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    price_change_percentage_7d: float
    price_change_percentage_30d: float
    circulating_supply: float
    total_supply: Optional[float]
    max_supply: Optional[float]
    ath: float
    ath_change_percentage: float
    ath_date: Optional[datetime]
    atl: float
    atl_change_percentage: float
    atl_date: Optional[datetime]
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'name': self.name,
            'current_price': self.current_price,
            'market_cap': self.market_cap,
            'market_cap_rank': self.market_cap_rank,
            'total_volume': self.total_volume,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'price_change_24h': self.price_change_24h,
            'price_change_percentage_24h': self.price_change_percentage_24h,
            'price_change_percentage_7d': self.price_change_percentage_7d,
            'price_change_percentage_30d': self.price_change_percentage_30d,
            'circulating_supply': self.circulating_supply,
            'total_supply': self.total_supply,
            'max_supply': self.max_supply,
            'ath': self.ath,
            'ath_change_percentage': self.ath_change_percentage,
            'ath_date': self.ath_date.isoformat() if self.ath_date else None,
            'atl': self.atl,
            'atl_change_percentage': self.atl_change_percentage,
            'atl_date': self.atl_date.isoformat() if self.atl_date else None,
            'last_updated': self.last_updated.isoformat(),
        }


@dataclass
class GlobalMarketData:
    """全球市场数据"""
    total_market_cap: float
    total_volume: float
    market_cap_percentage: Dict[str, float]  # BTC, ETH占比
    market_cap_change_percentage_24h: float
    active_cryptocurrencies: int
    markets: int
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_market_cap': self.total_market_cap,
            'total_volume': self.total_volume,
            'market_cap_percentage': self.market_cap_percentage,
            'market_cap_change_percentage_24h': self.market_cap_change_percentage_24h,
            'active_cryptocurrencies': self.active_cryptocurrencies,
            'markets': self.markets,
            'updated_at': self.updated_at.isoformat(),
            'btc_dominance': self.market_cap_percentage.get('btc', 0),
            'eth_dominance': self.market_cap_percentage.get('eth', 0),
        }


@dataclass
class TrendingCoin:
    """趋势代币"""
    id: str
    symbol: str
    name: str
    market_cap_rank: int
    thumb: str
    score: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'name': self.name,
            'market_cap_rank': self.market_cap_rank,
            'thumb': self.thumb,
            'score': self.score,
        }


# ==================== 主类 ====================

class CoinGeckoDataSource(DataSource):
    """CoinGecko数据源"""

    def __init__(self, config: Optional[CoinGeckoConfig] = None):
        super().__init__(config or CoinGeckoConfig())
        self._session: Optional[aiohttp.ClientSession] = None
        self._coin_list_cache: Optional[List[Dict]] = None

    @property
    def name(self) -> str:
        return "coingecko"

    @property
    def supported_markets(self) -> List[MarketType]:
        return [MarketType.CRYPTO]

    @property
    def _base_url(self) -> str:
        if self.config.api_key:
            return COINGECKO_PRO_URL
        return COINGECKO_BASE_URL

    async def connect(self) -> bool:
        """连接"""
        try:
            headers = {}
            if self.config.api_key:
                headers['x-cg-pro-api-key'] = self.config.api_key

            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

            # 测试连接
            async with self._session.get(f"{self._base_url}/ping") as resp:
                if resp.status == 200:
                    self.state = DataSourceState.RUNNING
                    logger.info("CoinGecko数据源已连接")
                    return True
                else:
                    raise DataSourceError(f"Ping failed: {resp.status}")

        except Exception as e:
            logger.error(f"CoinGecko连接失败: {e}")
            self.state = DataSourceState.ERROR
            return False

    async def disconnect(self) -> None:
        """断开连接"""
        if self._session:
            await self._session.close()
            self._session = None
        self.state = DataSourceState.IDLE

    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """发送请求"""
        if not self._session:
            raise DataSourceError("Not connected")

        url = f"{self._base_url}{endpoint}"

        async with self._session.get(url, params=params) as resp:
            if resp.status == 429:
                raise RateLimitError("Rate limit exceeded", retry_after=60)
            if resp.status != 200:
                raise DataSourceError(f"Request failed: {resp.status}")
            return await resp.json()

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
        cache_key = f"ohlcv:{symbol}:{frequency}:{limit}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_ohlcv_impl,
                symbol, frequency, start_date, end_date, limit
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
        end_date: Optional[datetime],
        limit: int,
    ) -> List[OHLCV]:
        """OHLCV实现"""
        # CoinGecko使用coin id而不是symbol
        coin_id = await self._get_coin_id(symbol)
        if not coin_id:
            raise DataNotFoundError(f"Coin not found: {symbol}")

        # 计算天数
        days_map = {
            DataFrequency.DAILY: min(limit, 365),
            DataFrequency.HOUR_1: min(limit // 24, 90),
            DataFrequency.MINUTE_5: 1,
        }
        days = days_map.get(frequency, limit)

        data = await self._request(
            f"/coins/{coin_id}/ohlc",
            {'vs_currency': self.config.vs_currency.value, 'days': days}
        )

        results = []
        for candle in data:
            results.append(OHLCV(
                timestamp=datetime.fromtimestamp(candle[0] / 1000),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=0,  # CoinGecko OHLC不包含volume
                symbol=symbol,
                source=self.name,
            ))

        return results[-limit:]

    # ==================== 实时报价 ====================

    async def fetch_quote(self, symbol: str) -> Optional[Quote]:
        """获取实时报价"""
        cache_key = f"quote:{symbol}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_quote_impl, symbol
            )
            if result:
                await self._set_cached(cache_key, result, ttl=30.0)
            return result
        except Exception as e:
            logger.error(f"获取{symbol}报价失败: {e}")
            return None

    async def _fetch_quote_impl(self, symbol: str) -> Optional[Quote]:
        """报价实现"""
        coin_id = await self._get_coin_id(symbol)
        if not coin_id:
            return None

        data = await self._request(
            "/simple/price",
            {
                'ids': coin_id,
                'vs_currencies': self.config.vs_currency.value,
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
            }
        )

        if coin_id not in data:
            return None

        price_data = data[coin_id]
        vs = self.config.vs_currency.value

        return Quote(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=float(price_data.get(vs, 0)),
            ask=float(price_data.get(vs, 0)),
            last_price=float(price_data.get(vs, 0)),
            volume=float(price_data.get(f'{vs}_24h_vol', 0)),
            source=self.name,
        )

    # ==================== 市场数据 ====================

    async def fetch_market_data(
        self,
        per_page: int = 100,
        page: int = 1,
        category: Optional[CoinCategory] = None,
    ) -> List[CoinMarketData]:
        """获取市场数据列表"""
        params = {
            'vs_currency': self.config.vs_currency.value,
            'order': 'market_cap_desc',
            'per_page': per_page,
            'page': page,
            'sparkline': 'false',
            'price_change_percentage': '24h,7d,30d',
        }
        if category:
            params['category'] = category.value

        data = await self._request_with_retry(
            self._request, "/coins/markets", params
        )

        results = []
        for coin in data:
            try:
                results.append(CoinMarketData(
                    id=coin['id'],
                    symbol=coin['symbol'].upper(),
                    name=coin['name'],
                    current_price=float(coin.get('current_price') or 0),
                    market_cap=float(coin.get('market_cap') or 0),
                    market_cap_rank=int(coin.get('market_cap_rank') or 0),
                    total_volume=float(coin.get('total_volume') or 0),
                    high_24h=float(coin.get('high_24h') or 0),
                    low_24h=float(coin.get('low_24h') or 0),
                    price_change_24h=float(coin.get('price_change_24h') or 0),
                    price_change_percentage_24h=float(coin.get('price_change_percentage_24h') or 0),
                    price_change_percentage_7d=float(coin.get('price_change_percentage_7d_in_currency') or 0),
                    price_change_percentage_30d=float(coin.get('price_change_percentage_30d_in_currency') or 0),
                    circulating_supply=float(coin.get('circulating_supply') or 0),
                    total_supply=float(coin['total_supply']) if coin.get('total_supply') else None,
                    max_supply=float(coin['max_supply']) if coin.get('max_supply') else None,
                    ath=float(coin.get('ath') or 0),
                    ath_change_percentage=float(coin.get('ath_change_percentage') or 0),
                    ath_date=datetime.fromisoformat(coin['ath_date'].replace('Z', '+00:00')) if coin.get('ath_date') else None,
                    atl=float(coin.get('atl') or 0),
                    atl_change_percentage=float(coin.get('atl_change_percentage') or 0),
                    atl_date=datetime.fromisoformat(coin['atl_date'].replace('Z', '+00:00')) if coin.get('atl_date') else None,
                    last_updated=datetime.fromisoformat(coin['last_updated'].replace('Z', '+00:00')) if coin.get('last_updated') else datetime.now(),
                ))
            except Exception as e:
                logger.warning(f"解析代币数据失败: {e}")
                continue

        return results

    # ==================== 全球数据 ====================

    async def fetch_global_data(self) -> Optional[GlobalMarketData]:
        """获取全球市场数据"""
        try:
            data = await self._request_with_retry(self._request, "/global")
            global_data = data.get('data', {})

            return GlobalMarketData(
                total_market_cap=float(global_data.get('total_market_cap', {}).get(self.config.vs_currency.value, 0)),
                total_volume=float(global_data.get('total_volume', {}).get(self.config.vs_currency.value, 0)),
                market_cap_percentage=global_data.get('market_cap_percentage', {}),
                market_cap_change_percentage_24h=float(global_data.get('market_cap_change_percentage_24h_usd', 0)),
                active_cryptocurrencies=int(global_data.get('active_cryptocurrencies', 0)),
                markets=int(global_data.get('markets', 0)),
                updated_at=datetime.fromtimestamp(global_data.get('updated_at', 0)),
            )
        except Exception as e:
            logger.error(f"获取全球数据失败: {e}")
            return None

    # ==================== 趋势代币 ====================

    async def fetch_trending(self) -> List[TrendingCoin]:
        """获取趋势代币"""
        try:
            data = await self._request_with_retry(self._request, "/search/trending")

            results = []
            for item in data.get('coins', []):
                coin = item.get('item', {})
                results.append(TrendingCoin(
                    id=coin.get('id', ''),
                    symbol=coin.get('symbol', '').upper(),
                    name=coin.get('name', ''),
                    market_cap_rank=int(coin.get('market_cap_rank') or 0),
                    thumb=coin.get('thumb', ''),
                    score=int(coin.get('score', 0)),
                ))
            return results
        except Exception as e:
            logger.error(f"获取趋势代币失败: {e}")
            return []

    # ==================== 辅助方法 ====================

    async def _get_coin_id(self, symbol: str) -> Optional[str]:
        """通过symbol获取coin id"""
        # 常见映射
        common_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'USDT': 'tether',
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'XRP': 'ripple',
            'USDC': 'usd-coin',
            'ADA': 'cardano',
            'AVAX': 'avalanche-2',
            'DOGE': 'dogecoin',
            'DOT': 'polkadot',
            'TRX': 'tron',
            'LINK': 'chainlink',
            'MATIC': 'matic-network',
            'SHIB': 'shiba-inu',
            'LTC': 'litecoin',
            'UNI': 'uniswap',
            'ATOM': 'cosmos',
            'XLM': 'stellar',
            'FIL': 'filecoin',
        }

        symbol_upper = symbol.upper()
        if symbol_upper in common_mapping:
            return common_mapping[symbol_upper]

        # 从API获取
        if self._coin_list_cache is None:
            self._coin_list_cache = await self._request("/coins/list")

        for coin in self._coin_list_cache:
            if coin['symbol'].upper() == symbol_upper:
                return coin['id']

        return None

    async def get_coin_list(self) -> List[Dict]:
        """获取代币列表"""
        if self._coin_list_cache is None:
            self._coin_list_cache = await self._request("/coins/list")
        return self._coin_list_cache


# ==================== 工厂函数 ====================

def create_coingecko_source(
    api_key: Optional[str] = None,
    vs_currency: VsCurrency = VsCurrency.USD,
) -> CoinGeckoDataSource:
    """创建CoinGecko数据源"""
    config = CoinGeckoConfig(
        api_key=api_key,
        vs_currency=vs_currency,
    )
    return CoinGeckoDataSource(config)
