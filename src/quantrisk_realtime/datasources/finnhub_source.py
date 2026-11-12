"""
Finnhub数据源 (Finnhub Data Source)

提供美股实时数据，免费tier支持:
- 实时报价
- 公司基本面
- 财务报表
- 新闻
- SEC文件

API文档: https://finnhub.io/docs/api

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
    RateLimitError,
    DataNotFoundError,
    MarketType,
    DataFrequency,
    OHLCV,
    Quote,
)

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class FinnhubResolution(str, Enum):
    """K线周期"""
    MINUTE_1 = "1"
    MINUTE_5 = "5"
    MINUTE_15 = "15"
    MINUTE_30 = "30"
    MINUTE_60 = "60"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"


# ==================== 数据类 ====================

@dataclass
class CompanyProfile:
    """公司概况"""
    symbol: str
    name: str
    country: str
    currency: str
    exchange: str
    industry: str
    ipo_date: Optional[str] = None
    logo: Optional[str] = None
    market_cap: float = 0
    shares_outstanding: float = 0
    website: Optional[str] = None
    phone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'country': self.country,
            'currency': self.currency,
            'exchange': self.exchange,
            'industry': self.industry,
            'ipo_date': self.ipo_date,
            'logo': self.logo,
            'market_cap': self.market_cap,
            'shares_outstanding': self.shares_outstanding,
            'website': self.website,
        }


@dataclass
class BasicFinancials:
    """基本财务指标"""
    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'ps_ratio': self.ps_ratio,
            'dividend_yield': self.dividend_yield,
            'beta': self.beta,
            'roe': self.roe,
            'roa': self.roa,
            'profit_margin': self.profit_margin,
            'revenue_growth': self.revenue_growth,
            'earnings_growth': self.earnings_growth,
            'high_52w': self.high_52w,
            'low_52w': self.low_52w,
        }


@dataclass
class CompanyNews:
    """公司新闻"""
    id: int
    headline: str
    summary: str
    source: str
    url: str
    datetime: datetime
    category: str
    related: str
    image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'headline': self.headline,
            'summary': self.summary,
            'source': self.source,
            'url': self.url,
            'datetime': self.datetime.isoformat(),
            'category': self.category,
            'related': self.related,
            'image': self.image,
        }


@dataclass
class EarningsCalendar:
    """财报日历"""
    symbol: str
    date: str
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    revenue_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    hour: Optional[str] = None  # BMO (before market open) / AMC (after market close)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'date': self.date,
            'eps_actual': self.eps_actual,
            'eps_estimate': self.eps_estimate,
            'eps_surprise': self.eps_surprise,
            'eps_surprise_pct': self.eps_surprise_pct,
            'revenue_actual': self.revenue_actual,
            'revenue_estimate': self.revenue_estimate,
            'hour': self.hour,
        }


@dataclass
class RecommendationTrend:
    """分析师推荐趋势"""
    symbol: str
    period: str
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0

    @property
    def total(self) -> int:
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell

    @property
    def consensus(self) -> str:
        """获取共识评级"""
        if self.total == 0:
            return "N/A"
        score = (
            self.strong_buy * 5 + self.buy * 4 + self.hold * 3 +
            self.sell * 2 + self.strong_sell * 1
        ) / self.total
        if score >= 4.5:
            return "Strong Buy"
        elif score >= 3.5:
            return "Buy"
        elif score >= 2.5:
            return "Hold"
        elif score >= 1.5:
            return "Sell"
        else:
            return "Strong Sell"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'period': self.period,
            'strong_buy': self.strong_buy,
            'buy': self.buy,
            'hold': self.hold,
            'sell': self.sell,
            'strong_sell': self.strong_sell,
            'total': self.total,
            'consensus': self.consensus,
        }


@dataclass
class PriceTarget:
    """目标价"""
    symbol: str
    target_high: float
    target_low: float
    target_mean: float
    target_median: float
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'target_high': self.target_high,
            'target_low': self.target_low,
            'target_mean': self.target_mean,
            'target_median': self.target_median,
            'last_updated': self.last_updated.isoformat(),
        }


# ==================== 配置类 ====================

@dataclass
class FinnhubConfig(DataSourceConfig):
    """Finnhub配置"""
    api_key: str = ""
    base_url: str = "https://finnhub.io/api/v1"
    # 免费版限制: 60次/分钟
    rate_limit_per_minute: float = 60.0
    cache_ttl: float = 60.0  # 1分钟缓存

    def __post_init__(self):
        self.name = "finnhub"


# ==================== 主类 ====================

class FinnhubDataSource(DataSource):
    """Finnhub数据源"""

    def __init__(self, config: Optional[FinnhubConfig] = None):
        self._config = config or FinnhubConfig()
        super().__init__(self._config)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return "finnhub"

    @property
    def supported_markets(self) -> List[MarketType]:
        return [MarketType.STOCK_US]

    async def connect(self) -> bool:
        """连接"""
        try:
            # Close existing session first to prevent leaks
            if self._session:
                await self._session.close()
            self._session = aiohttp.ClientSession()
            self.state = DataSourceState.RUNNING

            # 测试连接
            test_result = await self._request("/stock/symbol", {"exchange": "US"})
            if test_result:
                logger.info("Finnhub连接成功")
                return True
            return False
        except Exception as e:
            logger.error(f"Finnhub连接失败: {e}")
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
            self._session = aiohttp.ClientSession()

        await self._rate_limiter.acquire()

        url = f"{self._config.base_url}{endpoint}"
        params = params or {}
        params["token"] = self._config.api_key

        self.stats.total_requests += 1

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 429:
                    self.stats.failed_requests += 1
                    raise RateLimitError("Finnhub rate limit exceeded", retry_after=60)
                elif resp.status == 401:
                    self.stats.failed_requests += 1
                    raise DataSourceError("Invalid Finnhub API key")
                elif resp.status != 200:
                    self.stats.failed_requests += 1
                    raise DataSourceError(f"Finnhub API error: {resp.status}")

                data = await resp.json()
                self.stats.successful_requests += 1
                return data

        except aiohttp.ClientError as e:
            self.stats.failed_requests += 1
            raise DataSourceError(f"Network error: {e}")

    def _convert_resolution(self, frequency: DataFrequency) -> str:
        """转换周期"""
        mapping = {
            DataFrequency.MINUTE_1: FinnhubResolution.MINUTE_1.value,
            DataFrequency.MINUTE_5: FinnhubResolution.MINUTE_5.value,
            DataFrequency.MINUTE_15: FinnhubResolution.MINUTE_15.value,
            DataFrequency.MINUTE_30: FinnhubResolution.MINUTE_30.value,
            DataFrequency.HOURLY: FinnhubResolution.MINUTE_60.value,
            DataFrequency.DAILY: FinnhubResolution.DAILY.value,
            DataFrequency.WEEKLY: FinnhubResolution.WEEKLY.value,
            DataFrequency.MONTHLY: FinnhubResolution.MONTHLY.value,
        }
        return mapping.get(frequency, FinnhubResolution.DAILY.value)

    # ==================== 行情数据 ====================

    async def fetch_ohlcv(
        self,
        symbol: str,
        frequency: DataFrequency = DataFrequency.DAILY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """获取K线数据"""
        cache_key = f"ohlcv:{symbol}:{frequency.value}:{limit}"
        cached = await self._cache.get(cache_key)
        if cached:
            self.stats.cache_hits += 1
            return cached
        self.stats.cache_misses += 1

        # 设置时间范围
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=limit)

        params = {
            "symbol": symbol,
            "resolution": self._convert_resolution(frequency),
            "from": int(start_date.timestamp()),
            "to": int(end_date.timestamp()),
        }

        data = await self._request("/stock/candle", params)

        if data.get("s") != "ok" or "c" not in data:
            return []

        result = []
        for i in range(len(data["c"])):
            result.append(OHLCV(
                timestamp=datetime.fromtimestamp(data["t"][i]),
                open=data["o"][i],
                high=data["h"][i],
                low=data["l"][i],
                close=data["c"][i],
                volume=data["v"][i],
                symbol=symbol,
                source=self.name,
            ))

        await self._cache.set(cache_key, result)
        return result

    async def fetch_quote(self, symbol: str) -> Optional[Quote]:
        """获取实时报价"""
        cache_key = f"quote:{symbol}"
        cached = await self._cache.get(cache_key)
        if cached:
            self.stats.cache_hits += 1
            return cached
        self.stats.cache_misses += 1

        data = await self._request("/quote", {"symbol": symbol})

        if not data or data.get("c", 0) == 0:
            return None

        quote = Quote(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(data.get("t", datetime.now().timestamp())),
            bid=data.get("c", 0),  # Finnhub免费版不提供bid/ask
            ask=data.get("c", 0),
            last_price=data.get("c", 0),
            volume=0,
            source=self.name,
            change=data.get("d", 0),
            change_pct=data.get("dp", 0),
            high=data.get("h", 0),
            low=data.get("l", 0),
            open=data.get("o", 0),
            prev_close=data.get("pc", 0),
        )

        await self._cache.set(cache_key, quote, ttl=15)  # 15秒缓存
        return quote

    # ==================== 公司数据 ====================

    async def fetch_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """获取公司概况"""
        cache_key = f"profile:{symbol}"
        cached = await self._cache.get(cache_key)
        if cached:
            self.stats.cache_hits += 1
            return cached
        self.stats.cache_misses += 1

        data = await self._request("/stock/profile2", {"symbol": symbol})

        if not data or "name" not in data:
            return None

        profile = CompanyProfile(
            symbol=symbol,
            name=data.get("name", ""),
            country=data.get("country", ""),
            currency=data.get("currency", ""),
            exchange=data.get("exchange", ""),
            industry=data.get("finnhubIndustry", ""),
            ipo_date=data.get("ipo"),
            logo=data.get("logo"),
            market_cap=data.get("marketCapitalization", 0),
            shares_outstanding=data.get("shareOutstanding", 0),
            website=data.get("weburl"),
            phone=data.get("phone"),
        )

        await self._cache.set(cache_key, profile, ttl=3600)  # 1小时缓存
        return profile

    async def fetch_basic_financials(self, symbol: str) -> Optional[BasicFinancials]:
        """获取基本财务指标"""
        cache_key = f"financials:{symbol}"
        cached = await self._cache.get(cache_key)
        if cached:
            self.stats.cache_hits += 1
            return cached
        self.stats.cache_misses += 1

        data = await self._request("/stock/metric", {"symbol": symbol, "metric": "all"})

        if not data or "metric" not in data:
            return None

        m = data["metric"]
        financials = BasicFinancials(
            symbol=symbol,
            pe_ratio=m.get("peBasicExclExtraTTM"),
            pb_ratio=m.get("pbQuarterly"),
            ps_ratio=m.get("psAnnual"),
            dividend_yield=m.get("dividendYieldIndicatedAnnual"),
            beta=m.get("beta"),
            roe=m.get("roeTTM"),
            roa=m.get("roaTTM"),
            profit_margin=m.get("netProfitMarginTTM"),
            revenue_growth=m.get("revenueGrowthQuarterlyYoy"),
            earnings_growth=m.get("epsGrowthQuarterlyYoy"),
            high_52w=m.get("52WeekHigh"),
            low_52w=m.get("52WeekLow"),
        )

        await self._cache.set(cache_key, financials, ttl=3600)
        return financials

    # ==================== 新闻数据 ====================

    async def fetch_company_news(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[CompanyNews]:
        """获取公司新闻"""
        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=7)

        params = {
            "symbol": symbol,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
        }

        data = await self._request("/company-news", params)

        if not data or not isinstance(data, list):
            return []

        result = []
        for item in data:
            result.append(CompanyNews(
                id=item.get("id", 0),
                headline=item.get("headline", ""),
                summary=item.get("summary", ""),
                source=item.get("source", ""),
                url=item.get("url", ""),
                datetime=datetime.fromtimestamp(item.get("datetime", 0)),
                category=item.get("category", ""),
                related=item.get("related", symbol),
                image=item.get("image"),
            ))

        return result

    async def fetch_market_news(self, category: str = "general") -> List[CompanyNews]:
        """获取市场新闻"""
        data = await self._request("/news", {"category": category})

        if not data or not isinstance(data, list):
            return []

        result = []
        for item in data:
            result.append(CompanyNews(
                id=item.get("id", 0),
                headline=item.get("headline", ""),
                summary=item.get("summary", ""),
                source=item.get("source", ""),
                url=item.get("url", ""),
                datetime=datetime.fromtimestamp(item.get("datetime", 0)),
                category=item.get("category", category),
                related=item.get("related", ""),
                image=item.get("image"),
            ))

        return result

    # ==================== 分析师数据 ====================

    async def fetch_recommendation_trends(self, symbol: str) -> List[RecommendationTrend]:
        """获取分析师推荐趋势"""
        cache_key = f"rec:{symbol}"
        cached = await self._cache.get(cache_key)
        if cached:
            self.stats.cache_hits += 1
            return cached
        self.stats.cache_misses += 1

        data = await self._request("/stock/recommendation", {"symbol": symbol})

        if not data or not isinstance(data, list):
            return []

        result = []
        for item in data:
            result.append(RecommendationTrend(
                symbol=symbol,
                period=item.get("period", ""),
                strong_buy=item.get("strongBuy", 0),
                buy=item.get("buy", 0),
                hold=item.get("hold", 0),
                sell=item.get("sell", 0),
                strong_sell=item.get("strongSell", 0),
            ))

        await self._cache.set(cache_key, result, ttl=3600)
        return result

    async def fetch_price_target(self, symbol: str) -> Optional[PriceTarget]:
        """获取目标价"""
        cache_key = f"target:{symbol}"
        cached = await self._cache.get(cache_key)
        if cached:
            self.stats.cache_hits += 1
            return cached
        self.stats.cache_misses += 1

        data = await self._request("/stock/price-target", {"symbol": symbol})

        if not data or "targetHigh" not in data:
            return None

        target = PriceTarget(
            symbol=symbol,
            target_high=data.get("targetHigh", 0),
            target_low=data.get("targetLow", 0),
            target_mean=data.get("targetMean", 0),
            target_median=data.get("targetMedian", 0),
            last_updated=datetime.fromisoformat(data.get("lastUpdated", datetime.now().isoformat())),
        )

        await self._cache.set(cache_key, target, ttl=3600)
        return target

    # ==================== 财报数据 ====================

    async def fetch_earnings_calendar(
        self,
        symbol: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[EarningsCalendar]:
        """获取财报日历"""
        if to_date is None:
            to_date = datetime.now() + timedelta(days=30)
        if from_date is None:
            from_date = datetime.now() - timedelta(days=7)

        params = {
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
        }
        if symbol:
            params["symbol"] = symbol

        data = await self._request("/calendar/earnings", params)

        if not data or "earningsCalendar" not in data:
            return []

        result = []
        for item in data["earningsCalendar"]:
            result.append(EarningsCalendar(
                symbol=item.get("symbol", ""),
                date=item.get("date", ""),
                eps_actual=item.get("epsActual"),
                eps_estimate=item.get("epsEstimate"),
                eps_surprise=item.get("epsSurprise"),
                eps_surprise_pct=item.get("epsSurprisePercent"),
                revenue_actual=item.get("revenueActual"),
                revenue_estimate=item.get("revenueEstimate"),
                hour=item.get("hour"),
            ))

        return result

    # ==================== 股票列表 ====================

    async def fetch_stock_symbols(self, exchange: str = "US") -> List[Dict[str, str]]:
        """获取股票列表"""
        cache_key = f"symbols:{exchange}"
        cached = await self._cache.get(cache_key)
        if cached:
            self.stats.cache_hits += 1
            return cached
        self.stats.cache_misses += 1

        data = await self._request("/stock/symbol", {"exchange": exchange})

        if not data or not isinstance(data, list):
            return []

        result = [
            {
                "symbol": item.get("symbol", ""),
                "description": item.get("description", ""),
                "type": item.get("type", ""),
                "currency": item.get("currency", "USD"),
            }
            for item in data
        ]

        await self._cache.set(cache_key, result, ttl=86400)  # 24小时缓存
        return result

    # ==================== 辅助方法 ====================

    async def fetch_stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """获取股票综合分析"""
        # 并行获取多种数据
        profile_task = self.fetch_company_profile(symbol)
        financials_task = self.fetch_basic_financials(symbol)
        quote_task = self.fetch_quote(symbol)
        recommendations_task = self.fetch_recommendation_trends(symbol)
        target_task = self.fetch_price_target(symbol)

        results = await asyncio.gather(
            profile_task,
            financials_task,
            quote_task,
            recommendations_task,
            target_task,
            return_exceptions=True,
        )

        profile, financials, quote, recommendations, target = results

        return {
            'symbol': symbol,
            'profile': profile.to_dict() if isinstance(profile, CompanyProfile) else None,
            'financials': financials.to_dict() if isinstance(financials, BasicFinancials) else None,
            'quote': quote.to_dict() if isinstance(quote, Quote) else None,
            'recommendations': [r.to_dict() for r in recommendations] if isinstance(recommendations, list) else [],
            'price_target': target.to_dict() if isinstance(target, PriceTarget) else None,
        }


# ==================== 工厂函数 ====================

def create_finnhub_source(api_key: str = "") -> FinnhubDataSource:
    """创建Finnhub数据源"""
    config = FinnhubConfig(api_key=api_key)
    return FinnhubDataSource(config)
