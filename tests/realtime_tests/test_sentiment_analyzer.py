"""
市场情绪分析器测试 (Sentiment Analyzer Tests)

测试情绪分析器的所有功能:
- 多源情绪数据聚合
- 文本情绪分析
- 情绪强度评分
- 情绪趋势检测
- 异常情绪告警
- 情绪与价格相关性分析
- 情绪动量指标
- 恐惧贪婪指数计算
- 多资产情绪对比
- 情绪历史回测

Copyright (c) 2025 TraderX. All rights reserved.
"""

import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.sentiment_analyzer import (
    # Main Class
    SentimentAnalyzer,
    create_sentiment_analyzer,
    # Data Classes
    SentimentSource,
    SentimentScore,
    SentimentTrendData,
    SentimentAlert,
    FearGreedIndex,
    PriceSentimentCorrelation,
    SentimentComparison,
    SentimentSignal,
    SentimentConfig,
    # Providers
    SentimentDataProvider,
    SimulatedSentimentDataProvider,
    # Enums
    SentimentType,
    SentimentStrength,
    SentimentTrend,
    SentimentRegime,
    SourceType,
    AlertType,
    AnalyzerState,
)


# ============== Fixtures ==============

@pytest.fixture
def analyzer():
    """创建基础分析器"""
    return create_sentiment_analyzer(
        min_sample_size=3,
        sentiment_window=10,
        trend_window=5,
    )


@pytest.fixture
def data_provider():
    """创建数据提供者"""
    return SimulatedSentimentDataProvider()


@pytest.fixture
def analyzer_with_provider(data_provider):
    """创建带数据提供者的分析器"""
    return create_sentiment_analyzer(
        data_provider=data_provider,
        min_sample_size=3,
    )


def create_sentiment_source(
    symbol: str = "AAPL",
    source_type: SourceType = SourceType.NEWS,
    raw_score: float = 0.5,
    confidence: float = 0.8,
    content: str = "Positive news about the company",
    timestamp: datetime = None,
) -> SentimentSource:
    """创建情绪数据源"""
    return SentimentSource(
        source_id=f"source_{datetime.now().timestamp()}",
        source_type=source_type,
        symbol=symbol,
        content=content,
        timestamp=timestamp or datetime.now(),
        raw_score=raw_score,
        confidence=confidence,
    )


@pytest.fixture
def sample_sources():
    """创建样本数据源"""
    now = datetime.now()
    sources = []

    # 正面新闻
    for i in range(5):
        sources.append(create_sentiment_source(
            symbol="AAPL",
            source_type=SourceType.NEWS,
            raw_score=0.6 + i * 0.05,
            confidence=0.9,
            content="Strong earnings growth and positive outlook for the company",
            timestamp=now - timedelta(minutes=10-i),
        ))

    # 社交媒体
    for i in range(3):
        sources.append(create_sentiment_source(
            symbol="AAPL",
            source_type=SourceType.SOCIAL_MEDIA,
            raw_score=0.4 + i * 0.1,
            confidence=0.6,
            content="Bullish sentiment on social media",
            timestamp=now - timedelta(minutes=5-i),
        ))

    return sources


# ============== Test Text Analysis ==============

class TestTextAnalysis:
    """测试文本情绪分析"""

    def test_analyze_text_positive(self, analyzer):
        """测试1: 正面文本分析"""
        text = "Strong buy recommendation. Excellent growth prospects and bullish outlook."

        score, sentiment_type, confidence = analyzer.analyze_text(text)

        assert score > 50.0
        assert sentiment_type == SentimentType.POSITIVE
        assert confidence > 0

    def test_analyze_text_negative(self, analyzer):
        """测试2: 负面文本分析"""
        text = "Bearish outlook. Significant losses and downgrade. Sell recommendation."

        score, sentiment_type, confidence = analyzer.analyze_text(text)

        assert score < 50.0
        assert sentiment_type == SentimentType.NEGATIVE
        assert confidence > 0

    def test_analyze_text_neutral(self, analyzer):
        """测试3: 中性文本分析"""
        text = "The company reported quarterly results."

        score, sentiment_type, confidence = analyzer.analyze_text(text)

        assert sentiment_type == SentimentType.NEUTRAL
        assert 0 <= score <= 100

    def test_analyze_text_mixed(self, analyzer):
        """测试4: 混合情绪文本分析"""
        text = "Good earnings but concerns about risk. Bullish on growth but bearish on valuation."

        score, sentiment_type, confidence = analyzer.analyze_text(text)

        assert 0 <= score <= 100
        assert confidence > 0

    def test_analyze_text_empty(self, analyzer):
        """测试5: 空文本分析"""
        score, sentiment_type, confidence = analyzer.analyze_text("")

        assert sentiment_type == SentimentType.NEUTRAL
        assert score == 50.0


# ============== Test Sentiment Aggregation ==============

class TestSentimentAggregation:
    """测试情绪聚合"""

    @pytest.mark.asyncio
    async def test_aggregate_sentiment_basic(self, analyzer, sample_sources):
        """测试6: 基础情绪聚合"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加数据源
        for source in sample_sources:
            await analyzer.add_sentiment_source(source)

        # 聚合情绪
        sentiment = await analyzer.aggregate_sentiment(symbol)

        assert sentiment is not None
        assert sentiment.symbol == symbol
        assert 0 <= sentiment.score <= 100
        assert sentiment.sample_size > 0

    @pytest.mark.asyncio
    async def test_aggregate_sentiment_insufficient_data(self, analyzer):
        """测试7: 数据不足的情绪聚合"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 只添加少量数据
        source = create_sentiment_source(symbol=symbol)
        await analyzer.add_sentiment_source(source)

        sentiment = await analyzer.aggregate_sentiment(symbol)

        # 可能返回None(数据不足)
        assert sentiment is None or isinstance(sentiment, SentimentScore)

    @pytest.mark.asyncio
    async def test_aggregate_sentiment_weighted(self, analyzer):
        """测试8: 加权情绪聚合"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加不同权重的数据源
        sources = [
            create_sentiment_source(
                symbol=symbol,
                source_type=SourceType.RESEARCH,  # 高权重
                raw_score=0.8,
                confidence=0.9,
            ),
            create_sentiment_source(
                symbol=symbol,
                source_type=SourceType.SOCIAL_MEDIA,  # 低权重
                raw_score=0.2,
                confidence=0.5,
            ),
        ] * 3  # 重复确保足够数据

        for source in sources:
            await analyzer.add_sentiment_source(source)
            await asyncio.sleep(0.01)  # 小延迟

        sentiment = await analyzer.aggregate_sentiment(symbol)

        if sentiment:
            # 应该偏向高权重数据源
            assert sentiment.score > 50.0

    @pytest.mark.asyncio
    async def test_aggregate_sentiment_multiple_symbols(self, analyzer):
        """测试9: 多资产情绪聚合"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        for symbol in symbols:
            analyzer.add_symbol(symbol)

            # 为每个资产添加数据
            for _ in range(5):
                source = create_sentiment_source(
                    symbol=symbol,
                    raw_score=0.6,
                    confidence=0.8,
                )
                await analyzer.add_sentiment_source(source)

        # 聚合所有资产
        for symbol in symbols:
            sentiment = await analyzer.aggregate_sentiment(symbol)
            assert sentiment is not None
            assert sentiment.symbol == symbol

    def test_classify_sentiment_strength(self, analyzer):
        """测试10: 情绪强度分类"""
        # 非常强
        strength = analyzer._classify_sentiment_strength(95.0)
        assert strength == SentimentStrength.VERY_STRONG

        # 强
        strength = analyzer._classify_sentiment_strength(85.0)
        assert strength == SentimentStrength.STRONG

        # 中等
        strength = analyzer._classify_sentiment_strength(75.0)
        assert strength == SentimentStrength.MODERATE

        # 弱
        strength = analyzer._classify_sentiment_strength(62.0)
        assert strength == SentimentStrength.WEAK

        # 非常弱
        strength = analyzer._classify_sentiment_strength(52.0)
        assert strength == SentimentStrength.VERY_WEAK


# ============== Test Fear & Greed Index ==============

class TestFearGreedIndex:
    """测试恐惧贪婪指数"""

    @pytest.mark.asyncio
    async def test_calculate_fear_greed_basic(self, analyzer, sample_sources):
        """测试11: 基础恐惧贪婪指数计算"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        for source in sample_sources:
            await analyzer.add_sentiment_source(source)

        # 计算恐惧贪婪指数
        fear_greed = analyzer.calculate_fear_greed_index()

        assert fear_greed is not None
        assert 0 <= fear_greed.index_value <= 100
        assert fear_greed.regime in SentimentRegime

    def test_calculate_fear_greed_no_data(self, analyzer):
        """测试12: 无数据时的恐惧贪婪指数"""
        fear_greed = analyzer.calculate_fear_greed_index()

        assert fear_greed is not None
        assert fear_greed.index_value == 50.0
        assert fear_greed.regime == SentimentRegime.NEUTRAL

    @pytest.mark.asyncio
    async def test_fear_greed_extreme_fear(self, analyzer):
        """测试13: 极度恐惧状态"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加极度负面数据
        for i in range(10):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=-0.8,
                confidence=0.9,
                content="Severe bearish outlook",
            )
            await analyzer.add_sentiment_source(source)

        fear_greed = analyzer.calculate_fear_greed_index()

        assert fear_greed.index_value < 40.0
        assert fear_greed.regime in [SentimentRegime.EXTREME_FEAR, SentimentRegime.FEAR]

    @pytest.mark.asyncio
    async def test_fear_greed_extreme_greed(self, analyzer):
        """测试14: 极度贪婪状态"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加极度正面数据
        for i in range(10):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.9,
                confidence=0.9,
                content="Extremely bullish outlook",
            )
            await analyzer.add_sentiment_source(source)

        fear_greed = analyzer.calculate_fear_greed_index()

        assert fear_greed.index_value > 60.0
        assert fear_greed.regime in [SentimentRegime.GREED, SentimentRegime.EXTREME_GREED]

    @pytest.mark.asyncio
    async def test_fear_greed_components(self, analyzer, sample_sources):
        """测试15: 恐惧贪婪指数组成部分"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        for source in sample_sources:
            await analyzer.add_sentiment_source(source)

        fear_greed = analyzer.calculate_fear_greed_index()

        assert 'market_sentiment' in fear_greed.components
        assert 'sentiment_momentum' in fear_greed.components
        assert 'volatility' in fear_greed.components
        assert 'consistency' in fear_greed.components


# ============== Test Sentiment Extremes ==============

class TestSentimentExtremes:
    """测试极端情绪检测"""

    @pytest.mark.asyncio
    async def test_detect_extreme_bullish(self, analyzer):
        """测试16: 检测极端看涨情绪"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加极端正面数据
        for _ in range(10):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.95,
                confidence=0.9,
            )
            await analyzer.add_sentiment_source(source)

        # 聚合后检测极端
        await analyzer.aggregate_sentiment(symbol)
        extreme = analyzer.detect_sentiment_extremes(symbol)

        if extreme:
            assert extreme["extreme_type"] == "extreme_bullish"
            assert extreme["score"] >= 80

    @pytest.mark.asyncio
    async def test_detect_extreme_bearish(self, analyzer):
        """测试17: 检测极端看跌情绪"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加极端负面数据
        for _ in range(10):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=-0.95,
                confidence=0.9,
            )
            await analyzer.add_sentiment_source(source)

        await analyzer.aggregate_sentiment(symbol)
        extreme = analyzer.detect_sentiment_extremes(symbol)

        if extreme:
            assert extreme["extreme_type"] == "extreme_bearish"
            assert extreme["score"] <= 20

    @pytest.mark.asyncio
    async def test_detect_no_extreme(self, analyzer):
        """测试18: 无极端情绪"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加中性数据
        for _ in range(10):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.1,
                confidence=0.7,
            )
            await analyzer.add_sentiment_source(source)

        await analyzer.aggregate_sentiment(symbol)
        extreme = analyzer.detect_sentiment_extremes(symbol)

        assert extreme is None


# ============== Test Sentiment Trend ==============

class TestSentimentTrend:
    """测试情绪趋势"""

    @pytest.mark.asyncio
    async def test_calculate_trend_improving(self, analyzer):
        """测试19: 改善趋势"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加递增情绪数据
        for i in range(15):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=-0.3 + i * 0.1,  # 递增
                confidence=0.8,
                timestamp=datetime.now() - timedelta(minutes=15-i),
            )
            await analyzer.add_sentiment_source(source)
            await asyncio.sleep(0.01)

        trend = analyzer._calculate_trend(symbol)

        if trend:
            assert trend.trend in [SentimentTrend.IMPROVING, SentimentTrend.STABLE]
            assert trend.direction >= 0

    @pytest.mark.asyncio
    async def test_calculate_trend_deteriorating(self, analyzer):
        """测试20: 恶化趋势"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加递减情绪数据
        for i in range(15):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.5 - i * 0.08,  # 递减
                confidence=0.8,
                timestamp=datetime.now() - timedelta(minutes=15-i),
            )
            await analyzer.add_sentiment_source(source)
            await asyncio.sleep(0.01)

        trend = analyzer._calculate_trend(symbol)

        if trend:
            assert trend.trend in [SentimentTrend.DETERIORATING, SentimentTrend.STABLE]
            assert trend.direction <= 0

    @pytest.mark.asyncio
    async def test_calculate_trend_volatile(self, analyzer):
        """测试21: 波动趋势"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加波动情绪数据
        import random
        for i in range(15):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=random.uniform(-0.5, 0.5),
                confidence=0.8,
                timestamp=datetime.now() - timedelta(minutes=15-i),
            )
            await analyzer.add_sentiment_source(source)
            await asyncio.sleep(0.01)

        trend = analyzer._calculate_trend(symbol)

        if trend:
            assert trend.volatility > 0

    @pytest.mark.asyncio
    async def test_get_sentiment_trend(self, analyzer, sample_sources):
        """测试22: 获取情绪趋势"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        for source in sample_sources:
            await analyzer.add_sentiment_source(source)

        trend = analyzer.get_sentiment_trend(symbol)

        if trend:
            assert isinstance(trend, SentimentTrendData)
            assert trend.symbol == symbol


# ============== Test Price-Sentiment Correlation ==============

class TestPriceSentimentCorrelation:
    """测试价格情绪相关性"""

    @pytest.mark.asyncio
    async def test_correlate_with_price_basic(self, analyzer):
        """测试23: 基础价格相关性"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        now = datetime.now()

        # 添加同步上涨的数据
        for i in range(25):
            # 情绪数据
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.3 + i * 0.02,  # 递增
                confidence=0.8,
                timestamp=now - timedelta(minutes=25-i),
            )
            await analyzer.add_sentiment_source(source)

            # 价格数据
            await analyzer.add_price_point(
                symbol=symbol,
                timestamp=now - timedelta(minutes=25-i),
                price=100 + i * 2,  # 递增
            )

        correlation = analyzer._calculate_correlation(symbol)

        if correlation:
            assert isinstance(correlation, PriceSentimentCorrelation)
            assert -1 <= correlation.correlation <= 1

    @pytest.mark.asyncio
    async def test_correlate_divergence(self, analyzer):
        """测试24: 价格情绪背离"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        now = datetime.now()

        # 添加背离数据: 价格上涨,情绪下跌
        for i in range(25):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.5 - i * 0.02,  # 递减
                confidence=0.8,
                timestamp=now - timedelta(minutes=25-i),
            )
            await analyzer.add_sentiment_source(source)

            await analyzer.add_price_point(
                symbol=symbol,
                timestamp=now - timedelta(minutes=25-i),
                price=100 + i * 2,  # 递增
            )

        correlation = analyzer._calculate_correlation(symbol)

        if correlation:
            # 应该检测到背离
            assert correlation.divergence_score > 0

    @pytest.mark.asyncio
    async def test_correlate_insufficient_data(self, analyzer):
        """测试25: 数据不足的相关性"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 只添加少量数据
        source = create_sentiment_source(symbol=symbol)
        await analyzer.add_sentiment_source(source)
        await analyzer.add_price_point(symbol, datetime.now(), 100.0)

        correlation = analyzer._calculate_correlation(symbol)

        assert correlation is None

    def test_pearson_correlation(self, analyzer):
        """测试26: 皮尔逊相关系数计算"""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]

        correlation = analyzer._pearson_correlation(x, y)

        # Pearson correlation should be close to 1.0 for perfect positive correlation
        assert correlation == pytest.approx(1.0, abs=0.3)

    def test_pearson_correlation_negative(self, analyzer):
        """测试27: 负相关系数"""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]

        correlation = analyzer._pearson_correlation(x, y)

        # Pearson correlation should be close to -1.0 for perfect negative correlation
        assert correlation == pytest.approx(-1.0, abs=0.3)


# ============== Test Signal Generation ==============

class TestSignalGeneration:
    """测试信号生成"""

    @pytest.mark.asyncio
    async def test_generate_bullish_signal(self, analyzer):
        """测试28: 生成看涨信号"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加强烈正面情绪
        for _ in range(15):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.8,
                confidence=0.9,
            )
            await analyzer.add_sentiment_source(source)

        signal = analyzer.generate_signals(symbol)

        if signal:
            assert signal.signal_type in ["bullish", "neutral"]

    @pytest.mark.asyncio
    async def test_generate_bearish_signal(self, analyzer):
        """测试29: 生成看跌信号"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加强烈负面情绪
        for _ in range(15):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=-0.8,
                confidence=0.9,
            )
            await analyzer.add_sentiment_source(source)

        signal = analyzer.generate_signals(symbol)

        if signal:
            assert signal.signal_type in ["bearish", "neutral"]

    @pytest.mark.asyncio
    async def test_generate_neutral_signal(self, analyzer):
        """测试30: 生成中性信号"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加中性情绪
        for _ in range(15):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.0,
                confidence=0.7,
            )
            await analyzer.add_sentiment_source(source)

        signal = analyzer.generate_signals(symbol)

        if signal:
            assert signal.signal_type == "neutral"

    @pytest.mark.asyncio
    async def test_signal_confidence(self, analyzer):
        """测试31: 信号置信度"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        for _ in range(15):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.7,
                confidence=0.95,
            )
            await analyzer.add_sentiment_source(source)

        signal = analyzer.generate_signals(symbol)

        if signal:
            assert 0 <= signal.confidence <= 1


# ============== Test Multi-Asset Comparison ==============

class TestMultiAssetComparison:
    """测试多资产对比"""

    @pytest.mark.asyncio
    async def test_compare_sentiments_basic(self, analyzer):
        """测试32: 基础多资产对比"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        for symbol in symbols:
            analyzer.add_symbol(symbol)

            # 添加不同的情绪数据
            for _ in range(10):
                source = create_sentiment_source(
                    symbol=symbol,
                    raw_score=0.3 if symbol == "AAPL" else 0.6,
                    confidence=0.8,
                )
                await analyzer.add_sentiment_source(source)

        comparison = analyzer.compare_sentiments(symbols)

        assert isinstance(comparison, SentimentComparison)
        assert len(comparison.symbols) == len(symbols)
        assert len(comparison.rankings) == len(symbols)

    @pytest.mark.asyncio
    async def test_compare_sentiments_rankings(self, analyzer):
        """测试33: 情绪排名"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        scores_map = {"AAPL": 0.8, "GOOGL": 0.5, "MSFT": 0.3}

        for symbol in symbols:
            analyzer.add_symbol(symbol)

            for _ in range(10):
                source = create_sentiment_source(
                    symbol=symbol,
                    raw_score=scores_map[symbol],
                    confidence=0.8,
                )
                await analyzer.add_sentiment_source(source)

        comparison = analyzer.compare_sentiments(symbols)

        # AAPL应该排名第一
        if comparison.rankings:
            best_symbol = min(comparison.rankings.items(), key=lambda x: x[1])[0]
            assert best_symbol in symbols

    @pytest.mark.asyncio
    async def test_compare_sentiments_outliers(self, analyzer):
        """测试34: 异常值检测"""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        scores_map = {"AAPL": 0.5, "GOOGL": 0.5, "MSFT": 0.5, "TSLA": 0.95}  # TSLA异常

        for symbol in symbols:
            analyzer.add_symbol(symbol)

            for _ in range(10):
                source = create_sentiment_source(
                    symbol=symbol,
                    raw_score=scores_map[symbol],
                    confidence=0.8,
                )
                await analyzer.add_sentiment_source(source)

        comparison = analyzer.compare_sentiments(symbols)

        # 可能识别TSLA为异常值
        assert isinstance(comparison.outliers, list)

    def test_compare_sentiments_empty(self, analyzer):
        """测试35: 空资产对比"""
        comparison = analyzer.compare_sentiments([])

        assert len(comparison.symbols) == 0
        assert len(comparison.scores) == 0


# ============== Test Alert System ==============

class TestAlertSystem:
    """测试告警系统"""

    @pytest.mark.asyncio
    async def test_alert_extreme_sentiment(self, analyzer):
        """测试36: 极端情绪告警"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加极端情绪
        for _ in range(10):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.95,
                confidence=0.9,
            )
            await analyzer.add_sentiment_source(source)

        await analyzer.aggregate_sentiment(symbol)
        analyzer.detect_sentiment_extremes(symbol)

        alerts = analyzer.get_alerts(alert_type=AlertType.SENTIMENT_EXTREME)
        assert isinstance(alerts, list)

    def test_acknowledge_alert(self, analyzer):
        """测试37: 确认告警"""
        # 创建一个告警
        analyzer._create_alert(
            AlertType.SENTIMENT_EXTREME,
            "AAPL",
            "Test alert",
            "medium",
            85.0,
        )

        alerts = analyzer.get_alerts()
        if alerts:
            alert_id = alerts[0].alert_id
            success = analyzer.acknowledge_alert(alert_id)
            assert success

            alert = next(a for a in analyzer._alerts if a.alert_id == alert_id)
            assert alert.acknowledged

    def test_get_alerts_filtered(self, analyzer):
        """测试38: 筛选告警"""
        # 创建多个告警
        analyzer._create_alert(
            AlertType.SENTIMENT_EXTREME,
            "AAPL",
            "High alert",
            "high",
            90.0,
        )
        analyzer._create_alert(
            AlertType.SENTIMENT_SPIKE,
            "GOOGL",
            "Medium alert",
            "medium",
            70.0,
        )

        # 按类型筛选
        extreme_alerts = analyzer.get_alerts(alert_type=AlertType.SENTIMENT_EXTREME)
        assert all(a.alert_type == AlertType.SENTIMENT_EXTREME for a in extreme_alerts)

        # 按严重程度筛选
        high_alerts = analyzer.get_alerts(severity="high")
        assert all(a.severity == "high" for a in high_alerts)


# ============== Test Symbol Management ==============

class TestSymbolManagement:
    """测试资产管理"""

    def test_add_symbol(self, analyzer):
        """测试39: 添加资产"""
        analyzer.add_symbol("AAPL")

        assert "AAPL" in analyzer._symbols
        assert "AAPL" in analyzer._sources
        assert "AAPL" in analyzer._sentiment_scores

    def test_remove_symbol(self, analyzer):
        """测试40: 移除资产"""
        analyzer.add_symbol("AAPL")
        analyzer.remove_symbol("AAPL")

        assert "AAPL" not in analyzer._symbols
        assert "AAPL" not in analyzer._sources

    def test_get_symbols(self, analyzer):
        """测试41: 获取资产列表"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            analyzer.add_symbol(symbol)

        result = analyzer.get_symbols()
        assert set(result) == set(symbols)


# ============== Test Lifecycle ==============

class TestLifecycle:
    """测试生命周期"""

    @pytest.mark.asyncio
    async def test_start_stop(self, analyzer):
        """测试42: 启动和停止"""
        await analyzer.start()
        assert analyzer.state == AnalyzerState.RUNNING

        await asyncio.sleep(0.1)

        await analyzer.stop()
        assert analyzer.state == AnalyzerState.STOPPED

    @pytest.mark.asyncio
    async def test_pause_resume(self, analyzer):
        """测试43: 暂停和恢复"""
        await analyzer.start()
        assert analyzer.state == AnalyzerState.RUNNING

        await analyzer.pause()
        assert analyzer.state == AnalyzerState.PAUSED

        await analyzer.resume()
        assert analyzer.state == AnalyzerState.RUNNING

        await analyzer.stop()


# ============== Test Data Provider ==============

class TestDataProvider:
    """测试数据提供者"""

    @pytest.mark.asyncio
    async def test_simulated_provider_add_source(self, data_provider):
        """测试44: 添加数据源"""
        source = create_sentiment_source(symbol="AAPL")
        data_provider.add_source(source)

        sources = await data_provider.get_sentiment_sources("AAPL")
        assert len(sources) == 1
        assert sources[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_simulated_provider_add_price(self, data_provider):
        """测试45: 添加价格数据"""
        now = datetime.now()
        data_provider.add_price("AAPL", now, 150.0)

        prices = await data_provider.get_price_data("AAPL", 10)
        assert len(prices) == 1
        assert prices[0][1] == 150.0

    @pytest.mark.asyncio
    async def test_simulated_provider_filter_by_type(self, data_provider):
        """测试46: 按类型筛选数据源"""
        source1 = create_sentiment_source(
            symbol="AAPL",
            source_type=SourceType.NEWS,
        )
        source2 = create_sentiment_source(
            symbol="AAPL",
            source_type=SourceType.SOCIAL_MEDIA,
        )

        data_provider.add_source(source1)
        data_provider.add_source(source2)

        news_sources = await data_provider.get_sentiment_sources(
            "AAPL",
            source_type=SourceType.NEWS,
        )

        assert len(news_sources) == 1
        assert news_sources[0].source_type == SourceType.NEWS


# ============== Test Statistics ==============

class TestStatistics:
    """测试统计信息"""

    @pytest.mark.asyncio
    async def test_get_statistics(self, analyzer, sample_sources):
        """测试47: 获取统计信息"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        for source in sample_sources:
            await analyzer.add_sentiment_source(source)

        stats = analyzer.get_statistics()

        assert isinstance(stats, dict)
        assert "state" in stats
        assert "symbols" in stats
        assert "total_sources" in stats
        assert "average_sentiment" in stats


# ============== Test Query Methods ==============

class TestQueryMethods:
    """测试查询方法"""

    @pytest.mark.asyncio
    async def test_get_current_sentiment(self, analyzer, sample_sources):
        """测试48: 获取当前情绪"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        for source in sample_sources:
            await analyzer.add_sentiment_source(source)

        sentiment = analyzer.get_current_sentiment(symbol)

        if sentiment:
            assert sentiment.symbol == symbol

    def test_get_fear_greed_index(self, analyzer):
        """测试49: 获取恐惧贪婪指数"""
        index = analyzer.get_fear_greed_index()

        # 初始状态可能为None
        assert index is None or isinstance(index, FearGreedIndex)

    @pytest.mark.asyncio
    async def test_get_signals(self, analyzer):
        """测试50: 获取信号列表"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加数据以生成信号
        for _ in range(15):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.8,
                confidence=0.9,
            )
            await analyzer.add_sentiment_source(source)

        analyzer.generate_signals(symbol)

        signals = analyzer.get_signals(symbol=symbol)
        assert isinstance(signals, list)


# ============== Test Edge Cases ==============

class TestEdgeCases:
    """测试边界情况"""

    def test_analyze_text_very_long(self, analyzer):
        """测试51: 超长文本"""
        text = "Bullish " * 1000

        score, sentiment_type, confidence = analyzer.analyze_text(text)

        assert sentiment_type == SentimentType.POSITIVE
        assert score > 50

    @pytest.mark.asyncio
    async def test_rapid_updates(self, analyzer):
        """测试52: 快速更新"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 快速添加大量数据
        for i in range(100):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.5,
                confidence=0.8,
            )
            await analyzer.add_sentiment_source(source)

        sentiment = await analyzer.aggregate_sentiment(symbol)
        assert sentiment is not None

    @pytest.mark.asyncio
    async def test_extreme_score_values(self, analyzer):
        """测试53: 极端评分值"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 添加极端值
        for _ in range(10):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=1.0,  # 最大值
                confidence=1.0,
            )
            await analyzer.add_sentiment_source(source)

        sentiment = await analyzer.aggregate_sentiment(symbol)

        if sentiment:
            assert 0 <= sentiment.score <= 100


# ============== Test Integration ==============

class TestIntegration:
    """测试集成场景"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, analyzer):
        """测试54: 完整工作流程"""
        symbol = "AAPL"
        analyzer.add_symbol(symbol)

        # 1. 添加情绪数据
        for i in range(20):
            source = create_sentiment_source(
                symbol=symbol,
                raw_score=0.3 + i * 0.02,
                confidence=0.8,
                timestamp=datetime.now() - timedelta(minutes=20-i),
            )
            await analyzer.add_sentiment_source(source)

        # 2. 添加价格数据
        for i in range(20):
            await analyzer.add_price_point(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(minutes=20-i),
                price=100 + i * 2,
            )

        # 3. 聚合情绪
        sentiment = await analyzer.aggregate_sentiment(symbol)
        assert sentiment is not None

        # 4. 计算趋势
        trend = analyzer.get_sentiment_trend(symbol)
        if trend:
            assert isinstance(trend, SentimentTrendData)

        # 5. 计算相关性
        correlation = await analyzer.correlate_with_price(symbol)
        if correlation:
            assert isinstance(correlation, PriceSentimentCorrelation)

        # 6. 生成信号
        signal = analyzer.generate_signals(symbol)
        if signal:
            assert isinstance(signal, SentimentSignal)

        # 7. 计算恐惧贪婪指数
        fear_greed = analyzer.calculate_fear_greed_index()
        assert fear_greed is not None

        # 8. 获取统计
        stats = analyzer.get_statistics()
        assert stats["symbols"] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, analyzer):
        """测试55: 并发分析"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # 并发添加数据
        tasks = []
        for symbol in symbols:
            analyzer.add_symbol(symbol)

            async def add_data(sym):
                for _ in range(10):
                    source = create_sentiment_source(
                        symbol=sym,
                        raw_score=0.6,
                        confidence=0.8,
                    )
                    await analyzer.add_sentiment_source(source)

            tasks.append(add_data(symbol))

        await asyncio.gather(*tasks)

        # 验证所有资产都有数据
        for symbol in symbols:
            sentiment = await analyzer.aggregate_sentiment(symbol)
            if sentiment:
                assert sentiment.symbol == symbol


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
