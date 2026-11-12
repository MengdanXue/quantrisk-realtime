"""
市场情绪分析器 (Sentiment Analyzer)

实时追踪和分析市场情绪，包括：
- 多源情绪数据聚合（新闻、社交媒体、论坛）
- 文本情绪分析（正面/负面/中性）
- 情绪强度评分（0-100）
- 情绪趋势检测
- 异常情绪告警
- 情绪与价格相关性分析
- 情绪动量指标
- 恐惧贪婪指数计算
- 多资产情绪对比
- 情绪历史回测

Copyright (c) 2025 TraderX. All rights reserved.
"""

import asyncio
import logging
import math
import statistics
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Enums ============

class SentimentType(str, Enum):
    """情绪类型"""
    POSITIVE = "positive"       # 正面
    NEGATIVE = "negative"       # 负面
    NEUTRAL = "neutral"         # 中性


class SentimentStrength(str, Enum):
    """情绪强度"""
    VERY_STRONG = "very_strong"     # 非常强 (80-100)
    STRONG = "strong"               # 强 (60-80)
    MODERATE = "moderate"           # 中等 (40-60)
    WEAK = "weak"                   # 弱 (20-40)
    VERY_WEAK = "very_weak"         # 非常弱 (0-20)


class SentimentTrend(str, Enum):
    """情绪趋势"""
    IMPROVING = "improving"         # 改善中
    DETERIORATING = "deteriorating" # 恶化中
    STABLE = "stable"               # 稳定
    VOLATILE = "volatile"           # 波动


class SentimentRegime(str, Enum):
    """情绪状态"""
    EXTREME_FEAR = "extreme_fear"   # 极度恐惧 (0-20)
    FEAR = "fear"                   # 恐惧 (20-40)
    NEUTRAL = "neutral"             # 中性 (40-60)
    GREED = "greed"                 # 贪婪 (60-80)
    EXTREME_GREED = "extreme_greed" # 极度贪婪 (80-100)


class SourceType(str, Enum):
    """数据源类型"""
    NEWS = "news"                   # 新闻
    SOCIAL_MEDIA = "social_media"   # 社交媒体
    FORUM = "forum"                 # 论坛
    RESEARCH = "research"           # 研究报告
    ANALYST = "analyst"             # 分析师
    EARNINGS = "earnings"           # 财报
    MARKET_DATA = "market_data"     # 市场数据


class AlertType(str, Enum):
    """告警类型"""
    SENTIMENT_EXTREME = "sentiment_extreme"         # 极端情绪
    SENTIMENT_SHIFT = "sentiment_shift"             # 情绪转变
    SENTIMENT_DIVERGENCE = "sentiment_divergence"   # 情绪分歧
    FEAR_GREED_EXTREME = "fear_greed_extreme"       # 恐惧贪婪极端
    SENTIMENT_SPIKE = "sentiment_spike"             # 情绪飙升
    PRICE_SENTIMENT_DIVERGENCE = "price_sentiment_divergence"  # 价格情绪背离


class AnalyzerState(str, Enum):
    """分析器状态"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ Data Classes ============

@dataclass
class SentimentSource:
    """情绪数据源"""
    source_id: str
    source_type: SourceType
    symbol: str
    content: str
    timestamp: datetime
    raw_score: float = 0.0          # 原始情绪分数 (-1 to 1)
    confidence: float = 0.0         # 置信度 (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "symbol": self.symbol,
            "content": self.content[:200] if len(self.content) > 200 else self.content,
            "timestamp": self.timestamp.isoformat(),
            "raw_score": self.raw_score,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class SentimentScore:
    """情绪评分"""
    symbol: str
    timestamp: datetime
    score: float                    # 0-100 评分
    sentiment_type: SentimentType
    strength: SentimentStrength
    sample_size: int = 0           # 样本数量
    sources_breakdown: Dict[str, float] = field(default_factory=dict)  # 各来源评分
    confidence: float = 0.0         # 整体置信度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "sentiment_type": self.sentiment_type.value,
            "strength": self.strength.value,
            "sample_size": self.sample_size,
            "sources_breakdown": self.sources_breakdown,
            "confidence": self.confidence,
        }


@dataclass
class SentimentTrendData:
    """情绪趋势数据"""
    symbol: str
    timestamp: datetime
    current_score: float
    trend: SentimentTrend
    change_rate: float              # 变化率
    volatility: float               # 波动率
    momentum: float                 # 动量 (-100 to 100)
    direction: int = 0              # 方向 (-1, 0, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "current_score": self.current_score,
            "trend": self.trend.value,
            "change_rate": self.change_rate,
            "volatility": self.volatility,
            "momentum": self.momentum,
            "direction": self.direction,
        }


@dataclass
class SentimentAlert:
    """情绪告警"""
    alert_id: str
    alert_type: AlertType
    symbol: str
    message: str
    severity: str                   # "low", "medium", "high", "critical"
    sentiment_score: float
    timestamp: datetime
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "symbol": self.symbol,
            "message": self.message,
            "severity": self.severity,
            "sentiment_score": self.sentiment_score,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "metadata": self.metadata,
        }


@dataclass
class FearGreedIndex:
    """恐惧贪婪指数"""
    timestamp: datetime
    index_value: float              # 0-100
    regime: SentimentRegime
    components: Dict[str, float] = field(default_factory=dict)  # 各组成部分
    previous_value: float = 50.0
    change: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "index_value": self.index_value,
            "regime": self.regime.value,
            "components": self.components,
            "previous_value": self.previous_value,
            "change": self.change,
        }


@dataclass
class PriceSentimentCorrelation:
    """价格情绪相关性"""
    symbol: str
    timestamp: datetime
    correlation: float              # -1 to 1
    price_change: float            # 价格变化百分比
    sentiment_change: float        # 情绪变化
    divergence_score: float = 0.0  # 背离度 (0-100)
    signal_strength: float = 0.0   # 信号强度
    sample_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "correlation": self.correlation,
            "price_change": self.price_change,
            "sentiment_change": self.sentiment_change,
            "divergence_score": self.divergence_score,
            "signal_strength": self.signal_strength,
            "sample_size": self.sample_size,
        }


@dataclass
class SentimentComparison:
    """多资产情绪对比"""
    timestamp: datetime
    symbols: List[str]
    scores: Dict[str, float]
    rankings: Dict[str, int]        # 排名
    relative_scores: Dict[str, float]  # 相对评分
    outliers: List[str] = field(default_factory=list)  # 异常值

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbols": self.symbols,
            "scores": self.scores,
            "rankings": self.rankings,
            "relative_scores": self.relative_scores,
            "outliers": self.outliers,
        }


@dataclass
class SentimentSignal:
    """情绪交易信号"""
    signal_id: str
    symbol: str
    timestamp: datetime
    signal_type: str                # "bullish", "bearish", "neutral"
    strength: float                 # 0-100
    sentiment_score: float
    price_sentiment_alignment: float  # 价格情绪一致性
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type,
            "strength": self.strength,
            "sentiment_score": self.sentiment_score,
            "price_sentiment_alignment": self.price_sentiment_alignment,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


@dataclass
class SentimentConfig:
    """分析器配置"""
    # 分析参数
    min_sample_size: int = 5                    # 最小样本数
    sentiment_window: int = 20                  # 情绪计算窗口
    trend_window: int = 10                      # 趋势计算窗口

    # 权重配置
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        SourceType.NEWS.value: 1.0,
        SourceType.SOCIAL_MEDIA.value: 0.7,
        SourceType.FORUM.value: 0.5,
        SourceType.RESEARCH.value: 1.2,
        SourceType.ANALYST.value: 1.1,
        SourceType.EARNINGS.value: 1.5,
        SourceType.MARKET_DATA.value: 1.3,
    })

    # 阈值配置
    extreme_sentiment_threshold: float = 80.0   # 极端情绪阈值
    shift_threshold: float = 20.0               # 情绪转变阈值
    divergence_threshold: float = 30.0          # 背离阈值

    # 相关性分析
    correlation_window: int = 20                # 相关性计算窗口
    correlation_threshold: float = 0.3          # 相关性阈值

    # 信号生成
    signal_threshold: float = 60.0              # 信号阈值
    signal_confidence_min: float = 0.6          # 最小信号置信度

    # 监控参数
    update_interval: float = 60.0               # 更新间隔(秒)
    history_size: int = 1000                    # 历史记录大小

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_sample_size": self.min_sample_size,
            "sentiment_window": self.sentiment_window,
            "trend_window": self.trend_window,
            "source_weights": self.source_weights,
            "extreme_sentiment_threshold": self.extreme_sentiment_threshold,
            "shift_threshold": self.shift_threshold,
            "divergence_threshold": self.divergence_threshold,
            "correlation_window": self.correlation_window,
            "correlation_threshold": self.correlation_threshold,
            "signal_threshold": self.signal_threshold,
            "signal_confidence_min": self.signal_confidence_min,
            "update_interval": self.update_interval,
            "history_size": self.history_size,
        }


# ============ Providers ============

class SentimentDataProvider(ABC):
    """情绪数据提供者接口"""

    @abstractmethod
    async def get_sentiment_sources(
        self,
        symbol: str,
        source_type: Optional[SourceType] = None,
        limit: int = 100
    ) -> List[SentimentSource]:
        """获取情绪数据源"""
        pass

    @abstractmethod
    async def get_price_data(
        self,
        symbol: str,
        count: int
    ) -> List[Tuple[datetime, float]]:
        """获取价格数据"""
        pass


class SimulatedSentimentDataProvider(SentimentDataProvider):
    """模拟情绪数据提供者"""

    def __init__(self):
        self._sources: Dict[str, List[SentimentSource]] = defaultdict(list)
        self._prices: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    def add_source(self, source: SentimentSource) -> None:
        """添加情绪数据源"""
        self._sources[source.symbol].append(source)

    def add_price(self, symbol: str, timestamp: datetime, price: float) -> None:
        """添加价格数据"""
        self._prices[symbol].append((timestamp, price))

    async def get_sentiment_sources(
        self,
        symbol: str,
        source_type: Optional[SourceType] = None,
        limit: int = 100
    ) -> List[SentimentSource]:
        """获取情绪数据源"""
        sources = self._sources.get(symbol, [])

        if source_type:
            sources = [s for s in sources if s.source_type == source_type]

        return sources[-limit:]

    async def get_price_data(
        self,
        symbol: str,
        count: int
    ) -> List[Tuple[datetime, float]]:
        """获取价格数据"""
        prices = self._prices.get(symbol, [])
        return prices[-count:]


# ============ Main Class ============

class SentimentAnalyzer:
    """市场情绪分析器"""

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        data_provider: Optional[SentimentDataProvider] = None,
        alert_callback: Optional[Callable[[SentimentAlert], None]] = None,
        signal_callback: Optional[Callable[[SentimentSignal], None]] = None,
    ):
        self.config = config or SentimentConfig()
        self.data_provider = data_provider or simulated_default(SimulatedSentimentDataProvider, "data from SentimentAnalyzer's simulated data_provider (SimulatedSentimentDataProvider)")
        self.alert_callback = alert_callback
        self.signal_callback = signal_callback

        # 状态
        self._state = AnalyzerState.STOPPED
        self._monitor_task: Optional[asyncio.Task] = None

        # 数据存储
        self._symbols: Set[str] = set()
        self._sources: Dict[str, deque] = {}  # symbol -> deque[SentimentSource]
        self._sentiment_scores: Dict[str, deque] = {}  # symbol -> deque[SentimentScore]
        self._price_data: Dict[str, deque] = {}  # symbol -> deque[(timestamp, price)]

        # 分析结果
        self._current_scores: Dict[str, SentimentScore] = {}
        self._trends: Dict[str, SentimentTrendData] = {}
        self._correlations: Dict[str, PriceSentimentCorrelation] = {}
        self._fear_greed_index: Optional[FearGreedIndex] = None

        # 告警和信号
        self._alerts: deque = deque(maxlen=1000)
        self._signals: deque = deque(maxlen=500)

        # 历史记录
        self._score_history: deque = deque(maxlen=self.config.history_size)
        self._signal_history: deque = deque(maxlen=self.config.history_size)

    @property
    def state(self) -> AnalyzerState:
        return self._state

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动分析器"""
        if self._state == AnalyzerState.RUNNING:
            return

        logger.info("Starting Sentiment Analyzer")
        self._state = AnalyzerState.RUNNING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Sentiment Analyzer started")

    async def stop(self) -> None:
        """停止分析器"""
        if self._state == AnalyzerState.STOPPED:
            return

        logger.info("Stopping Sentiment Analyzer")
        self._state = AnalyzerState.STOPPED

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Sentiment Analyzer stopped")

    async def pause(self) -> None:
        """暂停分析器"""
        if self._state == AnalyzerState.RUNNING:
            self._state = AnalyzerState.PAUSED
            logger.info("Sentiment Analyzer paused")

    async def resume(self) -> None:
        """恢复分析器"""
        if self._state == AnalyzerState.PAUSED:
            self._state = AnalyzerState.RUNNING
            logger.info("Sentiment Analyzer resumed")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state in (AnalyzerState.RUNNING, AnalyzerState.PAUSED):
            try:
                if self._state == AnalyzerState.RUNNING:
                    await self._update_all_sentiments()
                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self._state = AnalyzerState.ERROR

    # ============ Symbol Management ============

    def add_symbol(self, symbol: str) -> None:
        """添加监控资产"""
        if symbol not in self._symbols:
            self._symbols.add(symbol)
            self._sources[symbol] = deque(maxlen=self.config.history_size)
            self._sentiment_scores[symbol] = deque(maxlen=self.config.history_size)
            self._price_data[symbol] = deque(maxlen=self.config.history_size)
            logger.info(f"Added symbol: {symbol}")

    def remove_symbol(self, symbol: str) -> None:
        """移除监控资产"""
        if symbol in self._symbols:
            self._symbols.remove(symbol)
            self._sources.pop(symbol, None)
            self._sentiment_scores.pop(symbol, None)
            self._price_data.pop(symbol, None)
            self._current_scores.pop(symbol, None)
            self._trends.pop(symbol, None)
            self._correlations.pop(symbol, None)
            logger.info(f"Removed symbol: {symbol}")

    def get_symbols(self) -> List[str]:
        """获取所有监控资产"""
        return list(self._symbols)

    # ============ Data Management ============

    async def add_sentiment_source(self, source: SentimentSource) -> None:
        """添加情绪数据源"""
        symbol = source.symbol

        # 自动添加资产
        if symbol not in self._symbols:
            self.add_symbol(symbol)

        self._sources[symbol].append(source)

        # 触发更新
        await self._update_sentiment_for_symbol(symbol)

    async def add_price_point(
        self,
        symbol: str,
        timestamp: datetime,
        price: float
    ) -> None:
        """添加价格数据点"""
        if symbol not in self._symbols:
            self.add_symbol(symbol)

        self._price_data[symbol].append((timestamp, price))

        # 更新相关性
        await self._update_correlation_for_symbol(symbol)

    # ============ Text Sentiment Analysis ============

    def analyze_text(self, text: str) -> Tuple[float, SentimentType, float]:
        """
        分析文本情绪

        Args:
            text: 文本内容

        Returns:
            (评分0-100, 情绪类型, 置信度)
        """
        # 简化的情绪分析算法
        # 实际应用中应该使用 NLP 模型如 BERT, FinBERT 等

        text_lower = text.lower()

        # 正面关键词
        positive_keywords = [
            'bullish', 'buy', 'growth', 'profit', 'gain', 'up', 'rally',
            'strong', 'good', 'excellent', 'positive', 'surge', 'boom',
            'outperform', 'upgrade', 'beat', 'exceed', 'record'
        ]

        # 负面关键词
        negative_keywords = [
            'bearish', 'sell', 'loss', 'down', 'crash', 'weak', 'bad',
            'negative', 'decline', 'plunge', 'fall', 'drop', 'underperform',
            'downgrade', 'miss', 'concern', 'risk', 'fear'
        ]

        # 计算关键词出现次数
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)

        total_keywords = positive_count + negative_count

        if total_keywords == 0:
            # 中性
            return 50.0, SentimentType.NEUTRAL, 0.3

        # 计算情绪分数
        sentiment_ratio = (positive_count - negative_count) / total_keywords
        score = 50 + (sentiment_ratio * 50)  # 0-100

        # 确定类型
        if score > 60:
            sentiment_type = SentimentType.POSITIVE
        elif score < 40:
            sentiment_type = SentimentType.NEGATIVE
        else:
            sentiment_type = SentimentType.NEUTRAL

        # 置信度基于关键词数量
        confidence = min(1.0, total_keywords / 10)

        return score, sentiment_type, confidence

    # ============ Sentiment Aggregation ============

    async def aggregate_sentiment(self, symbol: str) -> Optional[SentimentScore]:
        """
        聚合多源情绪

        Args:
            symbol: 资产代码

        Returns:
            聚合后的情绪评分
        """
        sources = list(self._sources.get(symbol, []))

        if len(sources) < self.config.min_sample_size:
            return None

        # 取最近的窗口数据
        recent_sources = sources[-self.config.sentiment_window:]

        # 按来源类型分组
        sources_by_type: Dict[str, List[SentimentSource]] = defaultdict(list)
        for source in recent_sources:
            sources_by_type[source.source_type.value].append(source)

        # 计算加权平均
        weighted_sum = 0.0
        total_weight = 0.0
        sources_breakdown = {}

        for source_type, type_sources in sources_by_type.items():
            # 转换 raw_score (-1 to 1) 到 (0 to 100)
            type_scores = [(s.raw_score + 1) * 50 * s.confidence for s in type_sources]
            type_avg = statistics.mean(type_scores) if type_scores else 50.0

            weight = self.config.source_weights.get(source_type, 1.0)
            weighted_sum += type_avg * weight
            total_weight += weight

            sources_breakdown[source_type] = type_avg

        # 最终评分
        final_score = weighted_sum / total_weight if total_weight > 0 else 50.0
        final_score = max(0.0, min(100.0, final_score))

        # 确定情绪类型
        if final_score > 60:
            sentiment_type = SentimentType.POSITIVE
        elif final_score < 40:
            sentiment_type = SentimentType.NEGATIVE
        else:
            sentiment_type = SentimentType.NEUTRAL

        # 确定强度
        strength = self._classify_sentiment_strength(final_score)

        # 计算整体置信度
        confidences = [s.confidence for s in recent_sources]
        overall_confidence = statistics.mean(confidences) if confidences else 0.5

        sentiment_score = SentimentScore(
            symbol=symbol,
            timestamp=datetime.now(),
            score=final_score,
            sentiment_type=sentiment_type,
            strength=strength,
            sample_size=len(recent_sources),
            sources_breakdown=sources_breakdown,
            confidence=overall_confidence,
        )

        return sentiment_score

    def _classify_sentiment_strength(self, score: float) -> SentimentStrength:
        """分类情绪强度"""
        abs_deviation = abs(score - 50)

        if abs_deviation >= 40:
            return SentimentStrength.VERY_STRONG
        elif abs_deviation >= 30:
            return SentimentStrength.STRONG
        elif abs_deviation >= 20:
            return SentimentStrength.MODERATE
        elif abs_deviation >= 10:
            return SentimentStrength.WEAK
        else:
            return SentimentStrength.VERY_WEAK

    # ============ Fear & Greed Index ============

    def calculate_fear_greed_index(self) -> FearGreedIndex:
        """
        计算恐惧贪婪指数

        Returns:
            恐惧贪婪指数
        """
        if not self._current_scores:
            return FearGreedIndex(
                timestamp=datetime.now(),
                index_value=50.0,
                regime=SentimentRegime.NEUTRAL,
            )

        # 组成部分
        components = {}

        # 1. 市场情绪 (40%)
        market_scores = [score.score for score in self._current_scores.values()]
        market_sentiment = statistics.mean(market_scores) if market_scores else 50.0
        components['market_sentiment'] = market_sentiment

        # 2. 情绪动量 (30%)
        momentum_values = []
        for symbol, trend in self._trends.items():
            momentum_values.append(50 + trend.momentum / 2)  # 转换到 0-100
        momentum_score = statistics.mean(momentum_values) if momentum_values else 50.0
        components['sentiment_momentum'] = momentum_score

        # 3. 市场波动性 (20%)
        volatilities = [trend.volatility for trend in self._trends.values()]
        avg_volatility = statistics.mean(volatilities) if volatilities else 0.0
        # 高波动性 = 恐惧，低波动性 = 贪婪
        volatility_score = max(0, min(100, 50 - avg_volatility * 200))
        components['volatility'] = volatility_score

        # 4. 情绪一致性 (10%)
        if len(market_scores) > 1:
            sentiment_std = statistics.stdev(market_scores)
            # 低标准差 = 一致性高 = 贪婪/恐惧更明显
            consistency_score = max(0, min(100, 100 - sentiment_std * 2))
        else:
            consistency_score = 50.0
        components['consistency'] = consistency_score

        # 加权平均
        index_value = (
            market_sentiment * 0.4 +
            momentum_score * 0.3 +
            volatility_score * 0.2 +
            consistency_score * 0.1
        )

        # 确定状态
        if index_value <= 20:
            regime = SentimentRegime.EXTREME_FEAR
        elif index_value <= 40:
            regime = SentimentRegime.FEAR
        elif index_value <= 60:
            regime = SentimentRegime.NEUTRAL
        elif index_value <= 80:
            regime = SentimentRegime.GREED
        else:
            regime = SentimentRegime.EXTREME_GREED

        # 计算变化
        previous_value = self._fear_greed_index.index_value if self._fear_greed_index else 50.0
        change = index_value - previous_value

        fear_greed = FearGreedIndex(
            timestamp=datetime.now(),
            index_value=index_value,
            regime=regime,
            components=components,
            previous_value=previous_value,
            change=change,
        )

        self._fear_greed_index = fear_greed
        return fear_greed

    # ============ Sentiment Extremes Detection ============

    def detect_sentiment_extremes(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        检测极端情绪

        Args:
            symbol: 资产代码

        Returns:
            极端情绪检测结果
        """
        score = self._current_scores.get(symbol)
        if not score:
            return None

        is_extreme = (
            score.score >= self.config.extreme_sentiment_threshold or
            score.score <= (100 - self.config.extreme_sentiment_threshold)
        )

        if not is_extreme:
            return None

        extreme_type = "extreme_bullish" if score.score >= 80 else "extreme_bearish"

        result = {
            "symbol": symbol,
            "extreme_type": extreme_type,
            "score": score.score,
            "strength": score.strength.value,
            "sample_size": score.sample_size,
            "confidence": score.confidence,
            "timestamp": datetime.now().isoformat(),
        }

        # 创建告警
        self._create_alert(
            AlertType.SENTIMENT_EXTREME,
            symbol,
            f"Extreme sentiment detected: {score.score:.1f}",
            "high" if score.score >= 90 or score.score <= 10 else "medium",
            score.score,
            {"extreme_type": extreme_type}
        )

        return result

    # ============ Sentiment Trend ============

    def get_sentiment_trend(self, symbol: str) -> Optional[SentimentTrendData]:
        """
        获取情绪趋势

        Args:
            symbol: 资产代码

        Returns:
            情绪趋势数据
        """
        return self._trends.get(symbol)

    def _calculate_trend(self, symbol: str) -> Optional[SentimentTrendData]:
        """计算情绪趋势"""
        scores = list(self._sentiment_scores.get(symbol, []))

        if len(scores) < self.config.trend_window:
            return None

        recent_scores = scores[-self.config.trend_window:]
        score_values = [s.score for s in recent_scores]

        current_score = score_values[-1]

        # 计算变化率
        if len(score_values) >= 2:
            change_rate = (score_values[-1] - score_values[0]) / self.config.trend_window
        else:
            change_rate = 0.0

        # 计算波动率
        volatility = statistics.stdev(score_values) if len(score_values) > 1 else 0.0

        # 计算动量
        if len(score_values) >= 3:
            recent_avg = statistics.mean(score_values[-3:])
            older_avg = statistics.mean(score_values[:-3])
            momentum = (recent_avg - older_avg) * 2  # -100 to 100
        else:
            momentum = 0.0

        # 确定趋势
        if abs(change_rate) < 0.5:
            if volatility > 10:
                trend = SentimentTrend.VOLATILE
            else:
                trend = SentimentTrend.STABLE
        elif change_rate > 0:
            trend = SentimentTrend.IMPROVING
        else:
            trend = SentimentTrend.DETERIORATING

        # 确定方向
        if change_rate > 1:
            direction = 1
        elif change_rate < -1:
            direction = -1
        else:
            direction = 0

        trend_data = SentimentTrendData(
            symbol=symbol,
            timestamp=datetime.now(),
            current_score=current_score,
            trend=trend,
            change_rate=change_rate,
            volatility=volatility,
            momentum=momentum,
            direction=direction,
        )

        return trend_data

    # ============ Price-Sentiment Correlation ============

    async def correlate_with_price(
        self,
        symbol: str
    ) -> Optional[PriceSentimentCorrelation]:
        """
        情绪价格相关性分析

        Args:
            symbol: 资产代码

        Returns:
            相关性分析结果
        """
        return self._correlations.get(symbol)

    def _calculate_correlation(self, symbol: str) -> Optional[PriceSentimentCorrelation]:
        """计算价格情绪相关性"""
        scores = list(self._sentiment_scores.get(symbol, []))
        prices = list(self._price_data.get(symbol, []))

        window = self.config.correlation_window

        if len(scores) < window or len(prices) < window:
            return None

        # 对齐数据
        recent_scores = scores[-window:]
        recent_prices = prices[-window:]

        # 计算价格变化
        price_changes = []
        for i in range(1, len(recent_prices)):
            prev_price = recent_prices[i-1][1]
            curr_price = recent_prices[i][1]
            if prev_price > 0:
                change = ((curr_price - prev_price) / prev_price) * 100
                price_changes.append(change)

        # 计算情绪变化
        sentiment_changes = []
        for i in range(1, len(recent_scores)):
            prev_score = recent_scores[i-1].score
            curr_score = recent_scores[i].score
            change = curr_score - prev_score
            sentiment_changes.append(change)

        # 对齐长度
        min_len = min(len(price_changes), len(sentiment_changes))
        if min_len < 5:
            return None

        price_changes = price_changes[-min_len:]
        sentiment_changes = sentiment_changes[-min_len:]

        # 计算相关系数
        if len(price_changes) >= 2 and len(sentiment_changes) >= 2:
            correlation = self._pearson_correlation(price_changes, sentiment_changes)
        else:
            correlation = 0.0

        # 计算总体变化
        total_price_change = sum(price_changes)
        total_sentiment_change = sum(sentiment_changes)

        # 计算背离度
        # 如果价格上涨但情绪下降，或价格下跌但情绪上升，则背离
        if (total_price_change > 0 and total_sentiment_change < 0) or \
           (total_price_change < 0 and total_sentiment_change > 0):
            divergence_score = abs(total_price_change) + abs(total_sentiment_change)
            divergence_score = min(100, divergence_score)
        else:
            divergence_score = 0.0

        # 信号强度
        signal_strength = abs(correlation) * 100

        corr_data = PriceSentimentCorrelation(
            symbol=symbol,
            timestamp=datetime.now(),
            correlation=correlation,
            price_change=total_price_change,
            sentiment_change=total_sentiment_change,
            divergence_score=divergence_score,
            signal_strength=signal_strength,
            sample_size=min_len,
        )

        # 检测背离告警
        if divergence_score >= self.config.divergence_threshold:
            self._create_alert(
                AlertType.PRICE_SENTIMENT_DIVERGENCE,
                symbol,
                f"Price-sentiment divergence detected: {divergence_score:.1f}",
                "high" if divergence_score >= 50 else "medium",
                recent_scores[-1].score,
                {
                    "divergence_score": divergence_score,
                    "price_change": total_price_change,
                    "sentiment_change": total_sentiment_change,
                }
            )

        return corr_data

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """计算皮尔逊相关系数"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n

        std_x = statistics.stdev(x)
        std_y = statistics.stdev(y)

        if std_x == 0 or std_y == 0:
            return 0.0

        correlation = covariance / (std_x * std_y)
        return max(-1.0, min(1.0, correlation))

    # ============ Signal Generation ============

    def generate_signals(self, symbol: str) -> Optional[SentimentSignal]:
        """
        生成交易信号

        Args:
            symbol: 资产代码

        Returns:
            情绪交易信号
        """
        score = self._current_scores.get(symbol)
        trend = self._trends.get(symbol)
        correlation = self._correlations.get(symbol)

        if not score or not trend:
            return None

        # 信号类型判断
        if score.score >= self.config.signal_threshold:
            if trend.momentum > 10:
                signal_type = "bullish"
                strength = min(100, score.score + trend.momentum / 2)
            else:
                signal_type = "neutral"
                strength = 50.0
        elif score.score <= (100 - self.config.signal_threshold):
            if trend.momentum < -10:
                signal_type = "bearish"
                strength = min(100, (100 - score.score) + abs(trend.momentum) / 2)
            else:
                signal_type = "neutral"
                strength = 50.0
        else:
            signal_type = "neutral"
            strength = 50.0

        # 价格情绪一致性
        if correlation:
            if correlation.correlation > self.config.correlation_threshold:
                alignment = min(100, abs(correlation.correlation) * 100)
            else:
                alignment = 50.0
        else:
            alignment = 50.0

        # 计算置信度
        confidence = (score.confidence + min(1.0, alignment / 100)) / 2

        # 生成理由
        reasoning_parts = [
            f"Sentiment score: {score.score:.1f} ({score.sentiment_type.value})",
            f"Trend: {trend.trend.value} (momentum: {trend.momentum:.1f})",
        ]

        if correlation:
            reasoning_parts.append(
                f"Price correlation: {correlation.correlation:.2f}"
            )

        reasoning = " | ".join(reasoning_parts)

        signal = SentimentSignal(
            signal_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            timestamp=datetime.now(),
            signal_type=signal_type,
            strength=strength,
            sentiment_score=score.score,
            price_sentiment_alignment=alignment,
            confidence=confidence,
            reasoning=reasoning,
        )

        # 如果信号强度和置信度足够，触发回调
        if strength >= self.config.signal_threshold and confidence >= self.config.signal_confidence_min:
            self._signals.append(signal)
            if self.signal_callback:
                self.signal_callback(signal)

        return signal

    # ============ Multi-Asset Comparison ============

    def compare_sentiments(self, symbols: Optional[List[str]] = None) -> SentimentComparison:
        """
        多资产情绪对比

        Args:
            symbols: 资产列表，None 则使用所有资产

        Returns:
            情绪对比结果
        """
        if symbols is None:
            symbols = list(self._symbols)

        # 收集评分
        scores = {}
        for symbol in symbols:
            score_obj = self._current_scores.get(symbol)
            if score_obj:
                scores[symbol] = score_obj.score

        if not scores:
            return SentimentComparison(
                timestamp=datetime.now(),
                symbols=[],
                scores={},
                rankings={},
                relative_scores={},
            )

        # 计算排名
        sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {symbol: rank + 1 for rank, (symbol, _) in enumerate(sorted_symbols)}

        # 计算相对评分 (Z-score)
        score_values = list(scores.values())
        mean_score = statistics.mean(score_values)
        std_score = statistics.stdev(score_values) if len(score_values) > 1 else 1.0

        relative_scores = {}
        outliers = []

        for symbol, score in scores.items():
            z_score = (score - mean_score) / std_score if std_score > 0 else 0
            relative_scores[symbol] = z_score

            # 识别异常值 (|Z| > 2)
            if abs(z_score) > 2:
                outliers.append(symbol)

        comparison = SentimentComparison(
            timestamp=datetime.now(),
            symbols=symbols,
            scores=scores,
            rankings=rankings,
            relative_scores=relative_scores,
            outliers=outliers,
        )

        return comparison

    # ============ Update Methods ============

    async def _update_all_sentiments(self) -> None:
        """更新所有资产的情绪分析"""
        for symbol in self._symbols:
            await self._update_sentiment_for_symbol(symbol)
            await self._update_correlation_for_symbol(symbol)

    async def _update_sentiment_for_symbol(self, symbol: str) -> None:
        """更新单个资产的情绪分析"""
        # 聚合情绪
        sentiment_score = await self.aggregate_sentiment(symbol)
        if sentiment_score:
            self._current_scores[symbol] = sentiment_score
            self._sentiment_scores[symbol].append(sentiment_score)
            self._score_history.append(sentiment_score)

        # 计算趋势
        trend = self._calculate_trend(symbol)
        if trend:
            self._trends[symbol] = trend

        # 检测极端情绪
        self.detect_sentiment_extremes(symbol)

        # 生成信号
        self.generate_signals(symbol)

    async def _update_correlation_for_symbol(self, symbol: str) -> None:
        """更新单个资产的相关性分析"""
        correlation = self._calculate_correlation(symbol)
        if correlation:
            self._correlations[symbol] = correlation

    # ============ Alert Management ============

    def _create_alert(
        self,
        alert_type: AlertType,
        symbol: str,
        message: str,
        severity: str,
        sentiment_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """创建告警"""
        alert = SentimentAlert(
            alert_id=str(uuid.uuid4())[:8],
            alert_type=alert_type,
            symbol=symbol,
            message=message,
            severity=severity,
            sentiment_score=sentiment_score,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        self._alerts.append(alert)

        if self.alert_callback:
            self.alert_callback(alert)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[SentimentAlert]:
        """获取告警"""
        alerts = list(self._alerts)

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-limit:]

    # ============ Query Methods ============

    def get_current_sentiment(self, symbol: str) -> Optional[SentimentScore]:
        """获取当前情绪评分"""
        return self._current_scores.get(symbol)

    def get_fear_greed_index(self) -> Optional[FearGreedIndex]:
        """获取恐惧贪婪指数"""
        return self._fear_greed_index

    def get_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[SentimentSignal]:
        """获取交易信号"""
        signals = list(self._signals)

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]

        return signals[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_sources = sum(len(sources) for sources in self._sources.values())

        avg_sentiment = 0.0
        if self._current_scores:
            scores = [s.score for s in self._current_scores.values()]
            avg_sentiment = statistics.mean(scores)

        return {
            "state": self._state.value,
            "symbols": len(self._symbols),
            "total_sources": total_sources,
            "total_scores": len(self._current_scores),
            "average_sentiment": avg_sentiment,
            "total_alerts": len(self._alerts),
            "unacknowledged_alerts": len([a for a in self._alerts if not a.acknowledged]),
            "total_signals": len(self._signals),
            "fear_greed_index": (
                self._fear_greed_index.index_value
                if self._fear_greed_index else None
            ),
            "config": self.config.to_dict(),
        }


# ============ Factory Functions ============

def create_sentiment_analyzer(
    data_provider: Optional[SentimentDataProvider] = None,
    **config_kwargs,
) -> SentimentAnalyzer:
    """创建情绪分析器"""
    config = SentimentConfig(**config_kwargs)
    return SentimentAnalyzer(
        config=config,
        data_provider=data_provider,
    )


# ============ Exports ============

__all__ = [
    # Enums
    "SentimentType",
    "SentimentStrength",
    "SentimentTrend",
    "SentimentRegime",
    "SourceType",
    "AlertType",
    "AnalyzerState",
    # Data Classes
    "SentimentSource",
    "SentimentScore",
    "SentimentTrendData",
    "SentimentAlert",
    "FearGreedIndex",
    "PriceSentimentCorrelation",
    "SentimentComparison",
    "SentimentSignal",
    "SentimentConfig",
    # Providers
    "SentimentDataProvider",
    "SimulatedSentimentDataProvider",
    # Main Class
    "SentimentAnalyzer",
    # Factory Functions
    "create_sentiment_analyzer",
]
