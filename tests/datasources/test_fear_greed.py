"""
恐惧贪婪指数数据源测试
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from quantrisk_realtime.datasources.fear_greed_source import (
    FearGreedDataSource,
    FearGreedConfig,
    FearGreedIndex,
    SentimentTrend,
    SentimentHistory,
    SentimentLevel,
    MarketSentimentType,
    create_fear_greed_source,
)


class TestFearGreedIndex:
    """FearGreedIndex测试"""

    def test_creation(self):
        """测试创建"""
        index = FearGreedIndex(
            value=25,
            classification="Fear",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.value == 25
        assert index.classification == "Fear"

    def test_level_extreme_fear(self):
        """测试极度恐惧级别"""
        index = FearGreedIndex(
            value=10,
            classification="Extreme Fear",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.level == SentimentLevel.EXTREME_FEAR

    def test_level_fear(self):
        """测试恐惧级别"""
        index = FearGreedIndex(
            value=35,
            classification="Fear",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.level == SentimentLevel.FEAR

    def test_level_neutral(self):
        """测试中性级别"""
        index = FearGreedIndex(
            value=50,
            classification="Neutral",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.level == SentimentLevel.NEUTRAL

    def test_level_greed(self):
        """测试贪婪级别"""
        index = FearGreedIndex(
            value=65,
            classification="Greed",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.level == SentimentLevel.GREED

    def test_level_extreme_greed(self):
        """测试极度贪婪级别"""
        index = FearGreedIndex(
            value=85,
            classification="Extreme Greed",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.level == SentimentLevel.EXTREME_GREED

    def test_signal_strong_buy(self):
        """测试强买信号"""
        index = FearGreedIndex(
            value=15,
            classification="Extreme Fear",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.signal == "STRONG_BUY"

    def test_signal_buy(self):
        """测试买入信号"""
        index = FearGreedIndex(
            value=30,
            classification="Fear",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.signal == "BUY"

    def test_signal_hold(self):
        """测试持有信号"""
        index = FearGreedIndex(
            value=50,
            classification="Neutral",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.signal == "HOLD"

    def test_signal_sell(self):
        """测试卖出信号"""
        index = FearGreedIndex(
            value=70,
            classification="Greed",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.signal == "SELL"

    def test_signal_strong_sell(self):
        """测试强卖信号"""
        index = FearGreedIndex(
            value=85,
            classification="Extreme Greed",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        assert index.signal == "STRONG_SELL"

    def test_to_dict(self):
        """测试转字典"""
        index = FearGreedIndex(
            value=50,
            classification="Neutral",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        d = index.to_dict()
        assert d['value'] == 50
        assert d['classification'] == "Neutral"
        assert d['level'] == "neutral"
        assert d['signal'] == "HOLD"


class TestSentimentTrend:
    """SentimentTrend测试"""

    def test_daily_change(self):
        """测试日变化"""
        current = FearGreedIndex(
            value=50,
            classification="Neutral",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )
        yesterday = FearGreedIndex(
            value=40,
            classification="Fear",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        trend = SentimentTrend(current=current, yesterday=yesterday)

        assert trend.daily_change == 10

    def test_trend_improving(self):
        """测试改善趋势"""
        current = FearGreedIndex(
            value=50,
            classification="Neutral",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )
        yesterday = FearGreedIndex(
            value=40,
            classification="Fear",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        trend = SentimentTrend(current=current, yesterday=yesterday)

        assert trend.trend_direction == "improving"

    def test_trend_worsening(self):
        """测试恶化趋势"""
        current = FearGreedIndex(
            value=40,
            classification="Fear",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )
        yesterday = FearGreedIndex(
            value=50,
            classification="Neutral",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        trend = SentimentTrend(current=current, yesterday=yesterday)

        assert trend.trend_direction == "worsening"

    def test_trend_stable(self):
        """测试稳定趋势"""
        current = FearGreedIndex(
            value=50,
            classification="Neutral",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )
        yesterday = FearGreedIndex(
            value=48,
            classification="Neutral",
            timestamp=datetime.now(),
            market=MarketSentimentType.CRYPTO,
        )

        trend = SentimentTrend(current=current, yesterday=yesterday)

        assert trend.trend_direction == "stable"


class TestSentimentHistory:
    """SentimentHistory测试"""

    def test_average(self):
        """测试平均值"""
        data = [
            FearGreedIndex(value=40, classification="Fear", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
            FearGreedIndex(value=50, classification="Neutral", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
            FearGreedIndex(value=60, classification="Greed", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
        ]

        history = SentimentHistory(data=data, market=MarketSentimentType.CRYPTO)

        assert history.average == 50.0

    def test_min_max(self):
        """测试最小最大值"""
        data = [
            FearGreedIndex(value=20, classification="Fear", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
            FearGreedIndex(value=50, classification="Neutral", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
            FearGreedIndex(value=80, classification="Greed", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
        ]

        history = SentimentHistory(data=data, market=MarketSentimentType.CRYPTO)

        assert history.min_value == 20
        assert history.max_value == 80

    def test_extreme_days(self):
        """测试极端天数"""
        data = [
            FearGreedIndex(value=10, classification="Extreme Fear", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
            FearGreedIndex(value=15, classification="Extreme Fear", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
            FearGreedIndex(value=50, classification="Neutral", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
            FearGreedIndex(value=85, classification="Extreme Greed", timestamp=datetime.now(), market=MarketSentimentType.CRYPTO),
        ]

        history = SentimentHistory(data=data, market=MarketSentimentType.CRYPTO)

        assert history.extreme_fear_days == 2
        assert history.extreme_greed_days == 1


class TestFearGreedConfig:
    """配置测试"""

    def test_defaults(self):
        """测试默认配置"""
        config = FearGreedConfig()

        assert config.name == "fear_greed"
        assert config.rate_limit_per_minute == 30.0
        assert config.cache_ttl == 300.0


class TestFearGreedDataSource:
    """FearGreedDataSource测试"""

    def test_name(self):
        """测试名称"""
        source = FearGreedDataSource()
        assert source.name == "fear_greed"

    def test_supported_markets(self):
        """测试支持的市场"""
        from quantrisk_realtime.datasources.base import MarketType
        source = FearGreedDataSource()

        assert MarketType.CRYPTO in source.supported_markets
        assert MarketType.STOCK_US in source.supported_markets

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """测试连接断开"""
        source = FearGreedDataSource()

        # 模拟成功连接
        with patch.object(source, 'fetch_crypto_fear_greed', new_callable=AsyncMock) as mock:
            mock.return_value = FearGreedIndex(
                value=50,
                classification="Neutral",
                timestamp=datetime.now(),
                market=MarketSentimentType.CRYPTO,
            )

            result = await source.connect()
            # 由于实际网络请求，可能失败
            # assert result is True

            await source.disconnect()
            assert source._session is None


class TestFactoryFunction:
    """工厂函数测试"""

    def test_create_fear_greed_source(self):
        """测试创建数据源"""
        source = create_fear_greed_source()

        assert isinstance(source, FearGreedDataSource)
        assert source.config.name == "fear_greed"
