"""
恐惧贪婪指数数据源 (Fear & Greed Index Data Source)

提供市场情绪指标:
- 加密货币恐惧贪婪指数 (Alternative.me)
- CNN恐惧贪婪指数 (美股)
- 历史数据查询
- 情绪趋势分析

完全免费，无需API密钥。

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
    MarketType,
    DataFrequency,
    OHLCV,
    Quote,
)

logger = logging.getLogger(__name__)


# ==================== 常量 ====================

ALTERNATIVE_ME_URL = "https://api.alternative.me/fng/"
CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


# ==================== 枚举类型 ====================

class SentimentLevel(str, Enum):
    """情绪级别"""
    EXTREME_FEAR = "extreme_fear"       # 极度恐惧 (0-24)
    FEAR = "fear"                       # 恐惧 (25-44)
    NEUTRAL = "neutral"                 # 中性 (45-55)
    GREED = "greed"                     # 贪婪 (56-75)
    EXTREME_GREED = "extreme_greed"     # 极度贪婪 (76-100)


class MarketSentimentType(str, Enum):
    """市场情绪类型"""
    CRYPTO = "crypto"       # 加密货币
    STOCK_US = "stock_us"   # 美股


# ==================== 配置类 ====================

@dataclass
class FearGreedConfig(DataSourceConfig):
    """恐惧贪婪指数配置"""
    name: str = "fear_greed"
    rate_limit_per_second: float = 1.0
    rate_limit_per_minute: float = 30.0
    cache_ttl: float = 300.0  # 5分钟缓存

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()


# ==================== 数据类 ====================

@dataclass
class FearGreedIndex:
    """恐惧贪婪指数"""
    value: int
    classification: str
    timestamp: datetime
    market: MarketSentimentType
    time_until_update: Optional[int] = None  # 秒

    @property
    def level(self) -> SentimentLevel:
        if self.value <= 24:
            return SentimentLevel.EXTREME_FEAR
        elif self.value <= 44:
            return SentimentLevel.FEAR
        elif self.value <= 55:
            return SentimentLevel.NEUTRAL
        elif self.value <= 75:
            return SentimentLevel.GREED
        else:
            return SentimentLevel.EXTREME_GREED

    @property
    def signal(self) -> str:
        """交易信号建议"""
        if self.value <= 20:
            return "STRONG_BUY"
        elif self.value <= 35:
            return "BUY"
        elif self.value <= 65:
            return "HOLD"
        elif self.value <= 80:
            return "SELL"
        else:
            return "STRONG_SELL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'classification': self.classification,
            'level': self.level.value,
            'signal': self.signal,
            'timestamp': self.timestamp.isoformat(),
            'market': self.market.value,
            'time_until_update': self.time_until_update,
        }


@dataclass
class SentimentTrend:
    """情绪趋势"""
    current: FearGreedIndex
    yesterday: Optional[FearGreedIndex] = None
    last_week: Optional[FearGreedIndex] = None
    last_month: Optional[FearGreedIndex] = None

    @property
    def daily_change(self) -> Optional[int]:
        if self.yesterday:
            return self.current.value - self.yesterday.value
        return None

    @property
    def weekly_change(self) -> Optional[int]:
        if self.last_week:
            return self.current.value - self.last_week.value
        return None

    @property
    def monthly_change(self) -> Optional[int]:
        if self.last_month:
            return self.current.value - self.last_month.value
        return None

    @property
    def trend_direction(self) -> str:
        """趋势方向"""
        if self.daily_change is None:
            return "unknown"
        if self.daily_change > 5:
            return "improving"
        elif self.daily_change < -5:
            return "worsening"
        else:
            return "stable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'current': self.current.to_dict(),
            'yesterday': self.yesterday.to_dict() if self.yesterday else None,
            'last_week': self.last_week.to_dict() if self.last_week else None,
            'last_month': self.last_month.to_dict() if self.last_month else None,
            'daily_change': self.daily_change,
            'weekly_change': self.weekly_change,
            'monthly_change': self.monthly_change,
            'trend_direction': self.trend_direction,
        }


@dataclass
class SentimentHistory:
    """情绪历史"""
    data: List[FearGreedIndex]
    market: MarketSentimentType

    @property
    def average(self) -> float:
        if not self.data:
            return 50.0
        return sum(d.value for d in self.data) / len(self.data)

    @property
    def min_value(self) -> int:
        if not self.data:
            return 0
        return min(d.value for d in self.data)

    @property
    def max_value(self) -> int:
        if not self.data:
            return 100
        return max(d.value for d in self.data)

    @property
    def extreme_fear_days(self) -> int:
        return sum(1 for d in self.data if d.level == SentimentLevel.EXTREME_FEAR)

    @property
    def extreme_greed_days(self) -> int:
        return sum(1 for d in self.data if d.level == SentimentLevel.EXTREME_GREED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'market': self.market.value,
            'count': len(self.data),
            'average': self.average,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'extreme_fear_days': self.extreme_fear_days,
            'extreme_greed_days': self.extreme_greed_days,
            'data': [d.to_dict() for d in self.data[-30:]],  # 最近30天
        }


# ==================== 主类 ====================

class FearGreedDataSource(DataSource):
    """恐惧贪婪指数数据源"""

    def __init__(self, config: Optional[FearGreedConfig] = None):
        super().__init__(config or FearGreedConfig())
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return "fear_greed"

    @property
    def supported_markets(self) -> List[MarketType]:
        return [MarketType.CRYPTO, MarketType.STOCK_US]

    async def connect(self) -> bool:
        """连接"""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

            # 测试连接
            test = await self.fetch_crypto_fear_greed()
            if test:
                self.state = DataSourceState.RUNNING
                logger.info("Fear & Greed数据源已连接")
                return True
            else:
                raise DataSourceError("Connection test failed")

        except Exception as e:
            logger.error(f"Fear & Greed连接失败: {e}")
            self.state = DataSourceState.ERROR
            return False

    async def disconnect(self) -> None:
        """断开连接"""
        if self._session:
            await self._session.close()
            self._session = None
        self.state = DataSourceState.IDLE

    # ==================== 加密货币恐惧贪婪指数 ====================

    async def fetch_crypto_fear_greed(
        self,
        limit: int = 1,
    ) -> Optional[FearGreedIndex]:
        """获取加密货币恐惧贪婪指数"""
        cache_key = f"crypto_fng:{limit}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_crypto_fear_greed_impl, limit
            )
            if result:
                await self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"获取加密恐惧贪婪指数失败: {e}")
            return None

    async def _fetch_crypto_fear_greed_impl(self, limit: int) -> Optional[FearGreedIndex]:
        """实现"""
        if not self._session:
            raise DataSourceError("Not connected")

        async with self._session.get(
            ALTERNATIVE_ME_URL,
            params={'limit': limit, 'format': 'json'}
        ) as resp:
            if resp.status != 200:
                raise DataSourceError(f"Request failed: {resp.status}")

            data = await resp.json()

            if not data.get('data'):
                return None

            item = data['data'][0]

            return FearGreedIndex(
                value=int(item['value']),
                classification=item['value_classification'],
                timestamp=datetime.fromtimestamp(int(item['timestamp'])),
                market=MarketSentimentType.CRYPTO,
                time_until_update=int(item.get('time_until_update', 0)),
            )

    async def fetch_crypto_fear_greed_history(
        self,
        limit: int = 30,
    ) -> SentimentHistory:
        """获取加密货币恐惧贪婪指数历史"""
        cache_key = f"crypto_fng_history:{limit}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            if not self._session:
                raise DataSourceError("Not connected")

            async with self._session.get(
                ALTERNATIVE_ME_URL,
                params={'limit': limit, 'format': 'json'}
            ) as resp:
                if resp.status != 200:
                    raise DataSourceError(f"Request failed: {resp.status}")

                data = await resp.json()

                history = []
                for item in data.get('data', []):
                    history.append(FearGreedIndex(
                        value=int(item['value']),
                        classification=item['value_classification'],
                        timestamp=datetime.fromtimestamp(int(item['timestamp'])),
                        market=MarketSentimentType.CRYPTO,
                    ))

                result = SentimentHistory(
                    data=history,
                    market=MarketSentimentType.CRYPTO,
                )
                await self._set_cached(cache_key, result)
                return result

        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return SentimentHistory(data=[], market=MarketSentimentType.CRYPTO)

    async def fetch_crypto_sentiment_trend(self) -> Optional[SentimentTrend]:
        """获取加密货币情绪趋势"""
        try:
            history = await self.fetch_crypto_fear_greed_history(limit=31)

            if not history.data:
                return None

            current = history.data[0] if history.data else None
            yesterday = history.data[1] if len(history.data) > 1 else None
            last_week = history.data[7] if len(history.data) > 7 else None
            last_month = history.data[30] if len(history.data) > 30 else None

            if current:
                return SentimentTrend(
                    current=current,
                    yesterday=yesterday,
                    last_week=last_week,
                    last_month=last_month,
                )
            return None

        except Exception as e:
            logger.error(f"获取情绪趋势失败: {e}")
            return None

    # ==================== CNN恐惧贪婪指数 ====================

    async def fetch_cnn_fear_greed(self) -> Optional[FearGreedIndex]:
        """获取CNN恐惧贪婪指数(美股)"""
        cache_key = "cnn_fng"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        try:
            result = await self._request_with_retry(
                self._fetch_cnn_fear_greed_impl
            )
            if result:
                await self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"获取CNN恐惧贪婪指数失败: {e}")
            return None

    async def _fetch_cnn_fear_greed_impl(self) -> Optional[FearGreedIndex]:
        """CNN实现"""
        if not self._session:
            raise DataSourceError("Not connected")

        try:
            # CNN API需要特殊headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }

            async with self._session.get(CNN_FEAR_GREED_URL, headers=headers) as resp:
                if resp.status != 200:
                    # CNN API可能不稳定，返回None而不是抛异常
                    logger.warning(f"CNN API返回状态码: {resp.status}")
                    return None

                data = await resp.json()

                # 解析CNN数据格式
                fear_greed_data = data.get('fear_and_greed', {})
                score = fear_greed_data.get('score')

                if score is None:
                    return None

                value = int(float(score))

                # 确定分类
                if value <= 24:
                    classification = "Extreme Fear"
                elif value <= 44:
                    classification = "Fear"
                elif value <= 55:
                    classification = "Neutral"
                elif value <= 75:
                    classification = "Greed"
                else:
                    classification = "Extreme Greed"

                return FearGreedIndex(
                    value=value,
                    classification=classification,
                    timestamp=datetime.now(),
                    market=MarketSentimentType.STOCK_US,
                )

        except Exception as e:
            logger.warning(f"CNN API请求失败: {e}")
            return None

    # ==================== 综合方法 ====================

    async def fetch_all_sentiment(self) -> Dict[str, Any]:
        """获取所有市场情绪"""
        results = {}

        # 加密货币
        crypto = await self.fetch_crypto_fear_greed()
        if crypto:
            results['crypto'] = crypto.to_dict()

        # 美股
        stock = await self.fetch_cnn_fear_greed()
        if stock:
            results['stock_us'] = stock.to_dict()

        # 添加综合分析
        if crypto and stock:
            avg_value = (crypto.value + stock.value) // 2
            results['combined'] = {
                'average_value': avg_value,
                'crypto_value': crypto.value,
                'stock_value': stock.value,
                'divergence': abs(crypto.value - stock.value),
                'overall_signal': 'RISK_ON' if avg_value > 55 else 'RISK_OFF' if avg_value < 45 else 'NEUTRAL',
            }

        return results

    async def get_trading_signal(
        self,
        market: MarketSentimentType = MarketSentimentType.CRYPTO,
    ) -> Dict[str, Any]:
        """获取交易信号"""
        if market == MarketSentimentType.CRYPTO:
            trend = await self.fetch_crypto_sentiment_trend()
        else:
            # 美股只有当前值
            current = await self.fetch_cnn_fear_greed()
            if current:
                trend = SentimentTrend(current=current)
            else:
                trend = None

        if not trend:
            return {'signal': 'NO_DATA', 'confidence': 0}

        current = trend.current

        # 基础信号
        signal = current.signal
        confidence = 0.5

        # 根据趋势调整信心
        if trend.trend_direction == 'improving' and current.value < 50:
            confidence += 0.2  # 底部反弹信号更强
        elif trend.trend_direction == 'worsening' and current.value > 50:
            confidence += 0.2  # 顶部回落信号更强

        # 极端值提高信心
        if current.value <= 20 or current.value >= 80:
            confidence += 0.2

        return {
            'signal': signal,
            'value': current.value,
            'level': current.level.value,
            'confidence': min(1.0, confidence),
            'trend': trend.trend_direction,
            'daily_change': trend.daily_change,
            'recommendation': self._get_recommendation(current.value, trend.trend_direction),
        }

    def _get_recommendation(self, value: int, trend: str) -> str:
        """获取推荐"""
        if value <= 20:
            return "市场极度恐惧，可能是抄底机会，但需谨慎"
        elif value <= 35:
            if trend == 'improving':
                return "市场恐惧但有改善迹象，可以考虑逐步建仓"
            return "市场恐惧，观望或小仓位试探"
        elif value <= 55:
            return "市场情绪中性，按既定策略操作"
        elif value <= 75:
            if trend == 'worsening':
                return "市场贪婪但有降温迹象，可以考虑减仓"
            return "市场贪婪，注意风险控制"
        else:
            return "市场极度贪婪，高风险区域，考虑获利了结"

    # ==================== 接口实现 ====================

    async def fetch_ohlcv(
        self,
        symbol: str,
        frequency: DataFrequency = DataFrequency.DAILY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """获取OHLCV (情绪指数转换)"""
        history = await self.fetch_crypto_fear_greed_history(limit=limit)

        results = []
        for item in history.data:
            results.append(OHLCV(
                timestamp=item.timestamp,
                open=float(item.value),
                high=float(item.value),
                low=float(item.value),
                close=float(item.value),
                volume=0,
                symbol="FEAR_GREED",
                source=self.name,
            ))

        return results

    async def fetch_quote(self, symbol: str) -> Optional[Quote]:
        """获取报价"""
        if symbol.upper() in ['CRYPTO', 'BTC', 'ETH']:
            data = await self.fetch_crypto_fear_greed()
        else:
            data = await self.fetch_cnn_fear_greed()

        if data:
            return Quote(
                symbol="FEAR_GREED",
                timestamp=data.timestamp,
                bid=float(data.value),
                ask=float(data.value),
                last_price=float(data.value),
                source=self.name,
            )
        return None


# ==================== 工厂函数 ====================

def create_fear_greed_source() -> FearGreedDataSource:
    """创建恐惧贪婪指数数据源"""
    return FearGreedDataSource()
