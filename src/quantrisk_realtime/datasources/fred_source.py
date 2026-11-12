"""
FRED 宏观经济数据源 (Federal Reserve Economic Data Source)

提供美联储经济数据库访问:
- GDP、通胀、就业数据
- 利率、国债收益率
- 货币供应量
- 美元指数
- 40万+经济指标

完全免费，需要API密钥(免费获取)。

Copyright (c) 2025 QuantRiskPro. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
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

FRED_BASE_URL = "https://api.stlouisfed.org/fred"


# ==================== 枚举类型 ====================

class EconomicIndicator(str, Enum):
    """常用经济指标"""
    # GDP相关
    GDP = "GDP"                              # 名义GDP
    GDPC1 = "GDPC1"                          # 实际GDP
    GDP_GROWTH = "A191RL1Q225SBEA"           # GDP增长率

    # 通胀相关
    CPI = "CPIAUCSL"                         # CPI
    CORE_CPI = "CPILFESL"                    # 核心CPI
    PCE = "PCEPI"                            # PCE价格指数
    CORE_PCE = "PCEPILFE"                    # 核心PCE
    PPI = "PPIACO"                           # PPI

    # 就业相关
    UNEMPLOYMENT = "UNRATE"                   # 失业率
    NONFARM_PAYROLL = "PAYEMS"               # 非农就业
    INITIAL_CLAIMS = "ICSA"                  # 首次申领失业金
    LABOR_FORCE = "CLF16OV"                  # 劳动力人数

    # 利率相关
    FED_FUNDS = "FEDFUNDS"                   # 联邦基金利率
    FED_FUNDS_TARGET = "DFEDTARU"            # 联邦基金目标利率上限
    PRIME_RATE = "DPRIME"                    # 最优惠利率

    # 国债收益率
    T1M = "DGS1MO"                           # 1月期国债
    T3M = "DGS3MO"                           # 3月期国债
    T6M = "DGS6MO"                           # 6月期国债
    T1Y = "DGS1"                             # 1年期国债
    T2Y = "DGS2"                             # 2年期国债
    T5Y = "DGS5"                             # 5年期国债
    T10Y = "DGS10"                           # 10年期国债
    T20Y = "DGS20"                           # 20年期国债
    T30Y = "DGS30"                           # 30年期国债

    # 利差
    T10Y2Y = "T10Y2Y"                        # 10Y-2Y利差
    T10Y3M = "T10Y3M"                        # 10Y-3M利差

    # 货币供应
    M1 = "M1SL"                              # M1货币供应
    M2 = "M2SL"                              # M2货币供应

    # 美元指数
    DXY = "DTWEXBGS"                         # 美元指数(广义)

    # 市场指标
    VIX = "VIXCLS"                           # VIX恐慌指数
    SP500 = "SP500"                          # 标普500

    # 房地产
    HOUSE_PRICE = "CSUSHPINSA"               # Case-Shiller房价指数
    HOUSING_STARTS = "HOUST"                 # 新屋开工

    # 消费者信心
    CONSUMER_SENTIMENT = "UMCSENT"           # 密歇根消费者信心

    # 制造业
    ISM_PMI = "MANEMP"                       # 制造业就业
    INDUSTRIAL_PROD = "INDPRO"               # 工业生产指数


class Frequency(str, Enum):
    """数据频率"""
    DAILY = "d"
    WEEKLY = "w"
    BIWEEKLY = "bw"
    MONTHLY = "m"
    QUARTERLY = "q"
    SEMIANNUAL = "sa"
    ANNUAL = "a"


class AggregationMethod(str, Enum):
    """聚合方法"""
    AVG = "avg"         # 平均
    SUM = "sum"         # 求和
    EOP = "eop"         # 期末值


# ==================== 配置类 ====================

@dataclass
class FREDConfig(DataSourceConfig):
    """FRED配置"""
    name: str = "fred"
    rate_limit_per_second: float = 2.0
    rate_limit_per_minute: float = 120.0
    cache_ttl: float = 3600.0  # 经济数据变化慢，缓存1小时

    # API配置
    api_key: str = ""  # 必需

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'has_api_key': bool(self.api_key),
        })
        return base


# ==================== 数据类 ====================

@dataclass
class SeriesInfo:
    """序列信息"""
    id: str
    title: str
    frequency: str
    units: str
    seasonal_adjustment: str
    last_updated: datetime
    observation_start: datetime
    observation_end: datetime
    popularity: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'frequency': self.frequency,
            'units': self.units,
            'seasonal_adjustment': self.seasonal_adjustment,
            'last_updated': self.last_updated.isoformat(),
            'observation_start': self.observation_start.isoformat(),
            'observation_end': self.observation_end.isoformat(),
            'popularity': self.popularity,
            'notes': self.notes[:200] if self.notes else "",
        }


@dataclass
class Observation:
    """观测值"""
    date: datetime
    value: Optional[float]
    series_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat(),
            'value': self.value,
            'series_id': self.series_id,
        }


@dataclass
class EconomicRelease:
    """经济数据发布"""
    id: int
    name: str
    press_release: bool
    link: str
    release_dates: List[datetime]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'press_release': self.press_release,
            'link': self.link,
            'release_dates': [d.isoformat() for d in self.release_dates[-5:]],
        }


@dataclass
class YieldCurve:
    """收益率曲线"""
    date: datetime
    t1m: Optional[float] = None
    t3m: Optional[float] = None
    t6m: Optional[float] = None
    t1y: Optional[float] = None
    t2y: Optional[float] = None
    t5y: Optional[float] = None
    t10y: Optional[float] = None
    t20y: Optional[float] = None
    t30y: Optional[float] = None

    @property
    def spread_10y_2y(self) -> Optional[float]:
        if self.t10y is not None and self.t2y is not None:
            return self.t10y - self.t2y
        return None

    @property
    def spread_10y_3m(self) -> Optional[float]:
        if self.t10y is not None and self.t3m is not None:
            return self.t10y - self.t3m
        return None

    @property
    def is_inverted(self) -> bool:
        spread = self.spread_10y_2y
        return spread is not None and spread < 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat(),
            't1m': self.t1m,
            't3m': self.t3m,
            't6m': self.t6m,
            't1y': self.t1y,
            't2y': self.t2y,
            't5y': self.t5y,
            't10y': self.t10y,
            't20y': self.t20y,
            't30y': self.t30y,
            'spread_10y_2y': self.spread_10y_2y,
            'spread_10y_3m': self.spread_10y_3m,
            'is_inverted': self.is_inverted,
        }


# ==================== 主类 ====================

class FREDDataSource(DataSource):
    """FRED宏观经济数据源"""

    def __init__(self, config: Optional[FREDConfig] = None):
        super().__init__(config or FREDConfig())
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return "fred"

    @property
    def supported_markets(self) -> List[MarketType]:
        return [MarketType.MACRO]

    async def connect(self) -> bool:
        """连接"""
        if not self.config.api_key:
            logger.error("FRED API密钥未配置")
            self.state = DataSourceState.ERROR
            return False

        try:
            # Close existing session first to prevent leaks
            if self._session:
                await self._session.close()
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

            # 测试连接
            test_data = await self._request("/series", {'series_id': 'GDP'})
            if test_data:
                self.state = DataSourceState.RUNNING
                logger.info("FRED数据源已连接")
                return True
            else:
                raise DataSourceError("Connection test failed")

        except Exception as e:
            logger.error(f"FRED连接失败: {e}")
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

        url = f"{FRED_BASE_URL}{endpoint}"

        request_params = {
            'api_key': self.config.api_key,
            'file_type': 'json',
        }
        if params:
            request_params.update(params)

        async with self._session.get(url, params=request_params) as resp:
            if resp.status == 429:
                raise RateLimitError("Rate limit exceeded")
            if resp.status != 200:
                raise DataSourceError(f"Request failed: {resp.status}")
            return await resp.json()

    # ==================== 核心数据方法 ====================

    async def fetch_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: Optional[Frequency] = None,
        aggregation: Optional[AggregationMethod] = None,
    ) -> List[Observation]:
        """获取时间序列数据"""
        cache_key = f"series:{series_id}:{start_date}:{end_date}:{frequency}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_series_impl,
                series_id, start_date, end_date, frequency, aggregation
            )
            await self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"获取{series_id}失败: {e}")
            raise DataSourceError(f"Failed to fetch series {series_id}: {e}")

    async def _fetch_series_impl(
        self,
        series_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: Optional[Frequency],
        aggregation: Optional[AggregationMethod],
    ) -> List[Observation]:
        """序列实现"""
        params = {'series_id': series_id}

        if start_date:
            params['observation_start'] = start_date.strftime('%Y-%m-%d')
        if end_date:
            params['observation_end'] = end_date.strftime('%Y-%m-%d')
        if frequency:
            params['frequency'] = frequency.value
        if aggregation:
            params['aggregation_method'] = aggregation.value

        data = await self._request("/series/observations", params)

        results = []
        for obs in data.get('observations', []):
            value = obs.get('value')
            if value == '.':  # FRED用.表示缺失值
                value = None
            else:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = None

            results.append(Observation(
                date=datetime.strptime(obs['date'], '%Y-%m-%d'),
                value=value,
                series_id=series_id,
            ))

        return results

    async def fetch_series_info(self, series_id: str) -> Optional[SeriesInfo]:
        """获取序列信息"""
        try:
            data = await self._request_with_retry(
                self._request, "/series", {'series_id': series_id}
            )

            series = data.get('seriess', [{}])[0]

            return SeriesInfo(
                id=series.get('id', ''),
                title=series.get('title', ''),
                frequency=series.get('frequency', ''),
                units=series.get('units', ''),
                seasonal_adjustment=series.get('seasonal_adjustment', ''),
                last_updated=datetime.strptime(series['last_updated'][:19], '%Y-%m-%d %H:%M:%S') if series.get('last_updated') else datetime.now(),
                observation_start=datetime.strptime(series['observation_start'], '%Y-%m-%d') if series.get('observation_start') else datetime.now(),
                observation_end=datetime.strptime(series['observation_end'], '%Y-%m-%d') if series.get('observation_end') else datetime.now(),
                popularity=int(series.get('popularity', 0)),
                notes=series.get('notes', ''),
            )
        except Exception as e:
            logger.error(f"获取序列信息失败: {e}")
            return None

    # ==================== 便捷方法 ====================

    async def fetch_indicator(
        self,
        indicator: EconomicIndicator,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Observation]:
        """获取经济指标"""
        return await self.fetch_series(indicator.value, start_date, end_date)

    async def fetch_latest(self, series_id: str) -> Optional[Observation]:
        """获取最新值"""
        data = await self.fetch_series(
            series_id,
            start_date=datetime.now() - timedelta(days=365),
        )
        return data[-1] if data else None

    async def fetch_yield_curve(self, date: Optional[datetime] = None) -> Optional[YieldCurve]:
        """获取收益率曲线"""
        date = date or datetime.now()
        start_date = date - timedelta(days=7)  # 获取最近一周数据确保有值

        # 并行获取所有期限
        series_ids = {
            't1m': EconomicIndicator.T1M.value,
            't3m': EconomicIndicator.T3M.value,
            't6m': EconomicIndicator.T6M.value,
            't1y': EconomicIndicator.T1Y.value,
            't2y': EconomicIndicator.T2Y.value,
            't5y': EconomicIndicator.T5Y.value,
            't10y': EconomicIndicator.T10Y.value,
            't20y': EconomicIndicator.T20Y.value,
            't30y': EconomicIndicator.T30Y.value,
        }

        tasks = {
            key: self.fetch_series(series_id, start_date, date)
            for key, series_id in series_ids.items()
        }

        results = {}
        for key, task in tasks.items():
            try:
                data = await task
                if data:
                    results[key] = data[-1].value
            except Exception:
                results[key] = None

        return YieldCurve(
            date=date,
            **results
        )

    async def fetch_macro_dashboard(self) -> Dict[str, Any]:
        """获取宏观经济仪表盘数据"""
        indicators = [
            (EconomicIndicator.UNEMPLOYMENT, "unemployment"),
            (EconomicIndicator.CPI, "cpi"),
            (EconomicIndicator.FED_FUNDS, "fed_funds"),
            (EconomicIndicator.T10Y2Y, "yield_spread"),
            (EconomicIndicator.VIX, "vix"),
            (EconomicIndicator.DXY, "dxy"),
        ]

        results = {}
        for indicator, key in indicators:
            try:
                obs = await self.fetch_latest(indicator.value)
                if obs and obs.value is not None:
                    results[key] = {
                        'value': obs.value,
                        'date': obs.date.isoformat(),
                        'series_id': indicator.value,
                    }
            except Exception as e:
                logger.warning(f"获取{key}失败: {e}")

        # 添加收益率曲线
        try:
            curve = await self.fetch_yield_curve()
            if curve:
                results['yield_curve'] = curve.to_dict()
        except Exception as e:
            logger.warning(f"获取收益率曲线失败: {e}")

        return results

    # ==================== 搜索 ====================

    async def search_series(
        self,
        query: str,
        limit: int = 10,
    ) -> List[SeriesInfo]:
        """搜索序列"""
        try:
            data = await self._request_with_retry(
                self._request,
                "/series/search",
                {
                    'search_text': query,
                    'limit': limit,
                    'order_by': 'popularity',
                    'sort_order': 'desc',
                }
            )

            results = []
            for series in data.get('seriess', []):
                try:
                    results.append(SeriesInfo(
                        id=series.get('id', ''),
                        title=series.get('title', ''),
                        frequency=series.get('frequency', ''),
                        units=series.get('units', ''),
                        seasonal_adjustment=series.get('seasonal_adjustment', ''),
                        last_updated=datetime.strptime(series['last_updated'][:19], '%Y-%m-%d %H:%M:%S') if series.get('last_updated') else datetime.now(),
                        observation_start=datetime.strptime(series['observation_start'], '%Y-%m-%d') if series.get('observation_start') else datetime.now(),
                        observation_end=datetime.strptime(series['observation_end'], '%Y-%m-%d') if series.get('observation_end') else datetime.now(),
                        popularity=int(series.get('popularity', 0)),
                    ))
                except Exception:
                    continue
            return results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    # ==================== 接口实现 ====================

    async def fetch_ohlcv(
        self,
        symbol: str,
        frequency: DataFrequency = DataFrequency.DAILY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """获取OHLCV（经济数据转换）"""
        data = await self.fetch_series(symbol, start_date, end_date)

        results = []
        for obs in data[-limit:]:
            if obs.value is not None:
                results.append(OHLCV(
                    timestamp=obs.date,
                    open=obs.value,
                    high=obs.value,
                    low=obs.value,
                    close=obs.value,
                    volume=0,
                    symbol=symbol,
                    source=self.name,
                ))
        return results

    async def fetch_quote(self, symbol: str) -> Optional[Quote]:
        """获取最新报价"""
        obs = await self.fetch_latest(symbol)
        if obs and obs.value is not None:
            return Quote(
                symbol=symbol,
                timestamp=obs.date,
                bid=obs.value,
                ask=obs.value,
                last_price=obs.value,
                source=self.name,
            )
        return None


# ==================== 工厂函数 ====================

def create_fred_source(api_key: str) -> FREDDataSource:
    """创建FRED数据源"""
    config = FREDConfig(api_key=api_key)
    return FREDDataSource(config)
