"""
Finnhub数据源测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from quantrisk_realtime.datasources.finnhub_source import (
    FinnhubDataSource,
    FinnhubConfig,
    FinnhubResolution,
    CompanyProfile,
    BasicFinancials,
    CompanyNews,
    EarningsCalendar,
    RecommendationTrend,
    PriceTarget,
    create_finnhub_source,
)
from quantrisk_realtime.datasources.base import MarketType


class TestCompanyProfile:
    """CompanyProfile测试"""

    def test_creation(self):
        """测试创建"""
        profile = CompanyProfile(
            symbol="AAPL",
            name="Apple Inc",
            country="US",
            currency="USD",
            exchange="NASDAQ",
            industry="Technology",
            market_cap=3000000000000,
        )

        assert profile.symbol == "AAPL"
        assert profile.name == "Apple Inc"
        assert profile.market_cap == 3000000000000

    def test_to_dict(self):
        """测试转字典"""
        profile = CompanyProfile(
            symbol="AAPL",
            name="Apple Inc",
            country="US",
            currency="USD",
            exchange="NASDAQ",
            industry="Technology",
        )

        d = profile.to_dict()
        assert d['symbol'] == "AAPL"
        assert d['name'] == "Apple Inc"


class TestBasicFinancials:
    """BasicFinancials测试"""

    def test_creation(self):
        """测试创建"""
        financials = BasicFinancials(
            symbol="AAPL",
            pe_ratio=30.5,
            pb_ratio=48.2,
            dividend_yield=0.5,
            beta=1.2,
        )

        assert financials.symbol == "AAPL"
        assert financials.pe_ratio == 30.5
        assert financials.beta == 1.2

    def test_to_dict(self):
        """测试转字典"""
        financials = BasicFinancials(
            symbol="AAPL",
            pe_ratio=30.5,
            roe=1.5,
        )

        d = financials.to_dict()
        assert d['pe_ratio'] == 30.5
        assert d['roe'] == 1.5


class TestCompanyNews:
    """CompanyNews测试"""

    def test_creation(self):
        """测试创建"""
        news = CompanyNews(
            id=12345,
            headline="Apple reports record earnings",
            summary="Apple Inc reported record quarterly earnings...",
            source="Reuters",
            url="https://example.com/news",
            datetime=datetime.now(),
            category="company",
            related="AAPL",
        )

        assert news.id == 12345
        assert news.headline == "Apple reports record earnings"
        assert news.source == "Reuters"

    def test_to_dict(self):
        """测试转字典"""
        news = CompanyNews(
            id=12345,
            headline="Test headline",
            summary="Test summary",
            source="Test",
            url="https://test.com",
            datetime=datetime.now(),
            category="company",
            related="AAPL",
        )

        d = news.to_dict()
        assert d['id'] == 12345
        assert d['headline'] == "Test headline"


class TestEarningsCalendar:
    """EarningsCalendar测试"""

    def test_creation(self):
        """测试创建"""
        earnings = EarningsCalendar(
            symbol="AAPL",
            date="2024-01-25",
            eps_actual=2.18,
            eps_estimate=2.10,
            eps_surprise=0.08,
            eps_surprise_pct=3.8,
            hour="AMC",
        )

        assert earnings.symbol == "AAPL"
        assert earnings.eps_actual == 2.18
        assert earnings.eps_surprise_pct == 3.8

    def test_to_dict(self):
        """测试转字典"""
        earnings = EarningsCalendar(
            symbol="AAPL",
            date="2024-01-25",
            eps_actual=2.18,
        )

        d = earnings.to_dict()
        assert d['symbol'] == "AAPL"
        assert d['eps_actual'] == 2.18


class TestRecommendationTrend:
    """RecommendationTrend测试"""

    def test_creation(self):
        """测试创建"""
        rec = RecommendationTrend(
            symbol="AAPL",
            period="2024-01",
            strong_buy=15,
            buy=20,
            hold=10,
            sell=2,
            strong_sell=1,
        )

        assert rec.symbol == "AAPL"
        assert rec.strong_buy == 15

    def test_total(self):
        """测试总数"""
        rec = RecommendationTrend(
            symbol="AAPL",
            period="2024-01",
            strong_buy=10,
            buy=20,
            hold=5,
            sell=2,
            strong_sell=1,
        )

        assert rec.total == 38

    def test_consensus_strong_buy(self):
        """测试强买共识"""
        rec = RecommendationTrend(
            symbol="AAPL",
            period="2024-01",
            strong_buy=20,
            buy=5,
            hold=2,
            sell=0,
            strong_sell=0,
        )

        assert rec.consensus == "Strong Buy"

    def test_consensus_buy(self):
        """测试买入共识"""
        rec = RecommendationTrend(
            symbol="AAPL",
            period="2024-01",
            strong_buy=10,
            buy=15,
            hold=5,
            sell=1,
            strong_sell=0,
        )

        assert rec.consensus == "Buy"

    def test_consensus_hold(self):
        """测试持有共识"""
        rec = RecommendationTrend(
            symbol="AAPL",
            period="2024-01",
            strong_buy=2,
            buy=5,
            hold=20,
            sell=3,
            strong_sell=1,
        )

        assert rec.consensus == "Hold"

    def test_consensus_na(self):
        """测试无共识"""
        rec = RecommendationTrend(
            symbol="AAPL",
            period="2024-01",
        )

        assert rec.consensus == "N/A"

    def test_to_dict(self):
        """测试转字典"""
        rec = RecommendationTrend(
            symbol="AAPL",
            period="2024-01",
            strong_buy=10,
            buy=20,
        )

        d = rec.to_dict()
        assert d['symbol'] == "AAPL"
        assert d['total'] == 30
        assert 'consensus' in d


class TestPriceTarget:
    """PriceTarget测试"""

    def test_creation(self):
        """测试创建"""
        target = PriceTarget(
            symbol="AAPL",
            target_high=250.0,
            target_low=180.0,
            target_mean=210.0,
            target_median=205.0,
            last_updated=datetime.now(),
        )

        assert target.symbol == "AAPL"
        assert target.target_high == 250.0
        assert target.target_mean == 210.0

    def test_to_dict(self):
        """测试转字典"""
        target = PriceTarget(
            symbol="AAPL",
            target_high=250.0,
            target_low=180.0,
            target_mean=210.0,
            target_median=205.0,
            last_updated=datetime.now(),
        )

        d = target.to_dict()
        assert d['target_high'] == 250.0
        assert 'last_updated' in d


class TestFinnhubConfig:
    """FinnhubConfig测试"""

    def test_defaults(self):
        """测试默认配置"""
        config = FinnhubConfig()

        assert config.name == "finnhub"
        assert config.rate_limit_per_minute == 60.0
        assert config.base_url == "https://finnhub.io/api/v1"

    def test_custom(self):
        """测试自定义配置"""
        config = FinnhubConfig(
            api_key="test_key",
            rate_limit_per_minute=30.0,
        )

        assert config.api_key == "test_key"
        assert config.rate_limit_per_minute == 30.0


class TestFinnhubDataSource:
    """FinnhubDataSource测试"""

    def test_name(self):
        """测试名称"""
        source = FinnhubDataSource()
        assert source.name == "finnhub"

    def test_supported_markets(self):
        """测试支持的市场"""
        source = FinnhubDataSource()
        assert MarketType.STOCK_US in source.supported_markets

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """测试断开连接"""
        source = FinnhubDataSource()
        await source.disconnect()
        assert source._session is None


class TestFactoryFunction:
    """工厂函数测试"""

    def test_create_finnhub_source(self):
        """测试创建数据源"""
        source = create_finnhub_source(api_key="test_key")

        assert isinstance(source, FinnhubDataSource)
        assert source._config.api_key == "test_key"

    def test_create_finnhub_source_no_key(self):
        """测试无API密钥创建"""
        source = create_finnhub_source()

        assert isinstance(source, FinnhubDataSource)
        assert source._config.api_key == ""
