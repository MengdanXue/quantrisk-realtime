"""
Trade Signal Aggregator - 多策略信号聚合器

实时聚合和融合多个策略信号，包括：
- Multi-strategy Signal Fusion 多策略信号融合
- Weighted Voting 加权投票
- Confidence-based Aggregation 置信度聚合
- Conflict Resolution 冲突解决
- Signal Strength Normalization 信号强度归一化
- Historical Performance Weighting 历史表现加权
- Real-time Signal Streaming 实时信号流
- Signal Quality Metrics 信号质量指标

这是量化交易系统中策略整合和信号决策的核心模块。

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
import statistics
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Enums ============

class SignalDirection(str, Enum):
    """信号方向"""
    STRONG_LONG = "strong_long"      # 强多
    LONG = "long"                    # 多
    NEUTRAL = "neutral"              # 中性
    SHORT = "short"                  # 空
    STRONG_SHORT = "strong_short"    # 强空


class SignalType(str, Enum):
    """信号类型"""
    MOMENTUM = "momentum"            # 动量信号
    MEAN_REVERSION = "mean_reversion"  # 均值回归
    TREND_FOLLOWING = "trend_following"  # 趋势跟随
    BREAKOUT = "breakout"            # 突破信号
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"  # 统计套利
    SENTIMENT = "sentiment"          # 情绪信号
    FUNDAMENTAL = "fundamental"      # 基本面信号
    TECHNICAL = "technical"          # 技术面信号
    CUSTOM = "custom"                # 自定义


class AggregationMethod(str, Enum):
    """聚合方法"""
    WEIGHTED_AVERAGE = "weighted_average"        # 加权平均
    MAJORITY_VOTING = "majority_voting"          # 多数投票
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # 置信度加权
    PERFORMANCE_WEIGHTED = "performance_weighted"  # 表现加权
    BAYESIAN_FUSION = "bayesian_fusion"          # 贝叶斯融合
    ENSEMBLE = "ensemble"                        # 集成方法


class ConflictResolutionStrategy(str, Enum):
    """冲突解决策略"""
    HIGHEST_CONFIDENCE = "highest_confidence"    # 最高置信度
    HIGHEST_WEIGHT = "highest_weight"            # 最高权重
    CANCEL_OUT = "cancel_out"                    # 相互抵消
    NEUTRAL = "neutral"                          # 转为中性
    CUSTOM_LOGIC = "custom_logic"                # 自定义逻辑


class SignalQuality(str, Enum):
    """信号质量评级"""
    EXCELLENT = "excellent"          # 优秀 (> 0.8)
    GOOD = "good"                    # 良好 (0.6 - 0.8)
    FAIR = "fair"                    # 一般 (0.4 - 0.6)
    POOR = "poor"                    # 较差 (0.2 - 0.4)
    VERY_POOR = "very_poor"          # 很差 (< 0.2)


class AggregatorState(str, Enum):
    """聚合器状态"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ Data Classes ============

@dataclass
class StrategySignal:
    """策略信号"""
    signal_id: str
    strategy_name: str
    symbol: str
    direction: SignalDirection
    strength: float                   # 信号强度 [0, 1]
    confidence: float                 # 置信度 [0, 1]
    signal_type: SignalType
    timestamp: datetime
    weight: float = 1.0               # 策略权重
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "signal_type": self.signal_type.value,
            "timestamp": self.timestamp.isoformat(),
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class HistoricalPerformance:
    """历史表现记录"""
    strategy_name: str
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    win_rate: float = 0.0
    average_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def update_weight(self) -> float:
        """根据历史表现计算动态权重"""
        if self.total_signals == 0:
            return 1.0

        # 综合考虑胜率、平均收益和夏普比率
        win_rate_score = self.win_rate
        return_score = max(0.0, min(1.0, (self.average_return + 0.5) / 1.5))
        sharpe_score = max(0.0, min(1.0, (self.sharpe_ratio + 1.0) / 3.0))

        # 加权平均
        weight = 0.4 * win_rate_score + 0.3 * return_score + 0.3 * sharpe_score

        # 惩罚大回撤
        if self.max_drawdown > 0.3:
            weight *= (1.0 - self.max_drawdown)

        return max(0.1, min(2.0, weight))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_signals": self.total_signals,
            "successful_signals": self.successful_signals,
            "failed_signals": self.failed_signals,
            "win_rate": self.win_rate,
            "average_return": self.average_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class AggregatedSignal:
    """聚合后的信号"""
    aggregation_id: str
    symbol: str
    direction: SignalDirection
    strength: float                   # 聚合后的强度
    confidence: float                 # 聚合后的置信度
    quality: SignalQuality            # 信号质量
    timestamp: datetime

    # 参与聚合的信号
    input_signals: List[StrategySignal] = field(default_factory=list)
    signal_count: int = 0

    # 一致性指标
    agreement_ratio: float = 0.0      # 信号一致性比率
    conflict_level: float = 0.0       # 冲突水平

    # 方法和权重
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    total_weight: float = 0.0

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregation_id": self.aggregation_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "quality": self.quality.value,
            "timestamp": self.timestamp.isoformat(),
            "signal_count": self.signal_count,
            "agreement_ratio": self.agreement_ratio,
            "conflict_level": self.conflict_level,
            "aggregation_method": self.aggregation_method.value,
            "total_weight": self.total_weight,
            "input_signals": [s.to_dict() for s in self.input_signals],
            "metadata": self.metadata,
        }


@dataclass
class SignalQualityMetrics:
    """信号质量指标"""
    symbol: str
    timestamp: datetime

    # 综合指标
    overall_quality: float = 0.0      # 总体质量分数
    consistency_score: float = 0.0    # 一致性分数
    diversity_score: float = 0.0      # 多样性分数

    # 信号统计
    total_signals: int = 0
    long_signals: int = 0
    short_signals: int = 0
    neutral_signals: int = 0

    # 强度分布
    average_strength: float = 0.0
    max_strength: float = 0.0
    min_strength: float = 0.0

    # 置信度分布
    average_confidence: float = 0.0
    max_confidence: float = 0.0
    min_confidence: float = 0.0

    # 冲突指标
    conflict_count: int = 0
    conflict_severity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "overall_quality": self.overall_quality,
            "consistency_score": self.consistency_score,
            "diversity_score": self.diversity_score,
            "total_signals": self.total_signals,
            "long_signals": self.long_signals,
            "short_signals": self.short_signals,
            "neutral_signals": self.neutral_signals,
            "average_strength": self.average_strength,
            "max_strength": self.max_strength,
            "min_strength": self.min_strength,
            "average_confidence": self.average_confidence,
            "max_confidence": self.max_confidence,
            "min_confidence": self.min_confidence,
            "conflict_count": self.conflict_count,
            "conflict_severity": self.conflict_severity,
        }


@dataclass
class AggregatorConfig:
    """聚合器配置"""
    # 聚合方法
    aggregation_method: AggregationMethod = AggregationMethod.CONFIDENCE_WEIGHTED
    conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE

    # 信号过滤
    min_signal_strength: float = 0.3       # 最小信号强度
    min_signal_confidence: float = 0.4     # 最小置信度
    min_signals_required: int = 2          # 最小所需信号数

    # 归一化参数
    normalize_strength: bool = True        # 是否归一化强度
    normalize_confidence: bool = True      # 是否归一化置信度

    # 权重配置
    use_historical_weighting: bool = True  # 使用历史表现加权
    weight_decay_rate: float = 0.95        # 权重衰减率
    min_weight: float = 0.1                # 最小权重
    max_weight: float = 2.0                # 最大权重

    # 冲突处理
    conflict_threshold: float = 0.5        # 冲突阈值
    allow_neutral_on_conflict: bool = True # 冲突时允许中性

    # 质量评估
    quality_threshold: float = 0.5         # 质量阈值
    require_minimum_quality: bool = True   # 要求最低质量

    # 流式处理
    signal_buffer_size: int = 100          # 信号缓冲区大小
    aggregation_interval: float = 1.0      # 聚合间隔(秒)

    # 历史记录
    history_size: int = 1000               # 历史记录大小

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregation_method": self.aggregation_method.value,
            "conflict_resolution": self.conflict_resolution.value,
            "min_signal_strength": self.min_signal_strength,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signals_required": self.min_signals_required,
            "normalize_strength": self.normalize_strength,
            "normalize_confidence": self.normalize_confidence,
            "use_historical_weighting": self.use_historical_weighting,
            "weight_decay_rate": self.weight_decay_rate,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "conflict_threshold": self.conflict_threshold,
            "allow_neutral_on_conflict": self.allow_neutral_on_conflict,
            "quality_threshold": self.quality_threshold,
            "require_minimum_quality": self.require_minimum_quality,
            "signal_buffer_size": self.signal_buffer_size,
            "aggregation_interval": self.aggregation_interval,
            "history_size": self.history_size,
        }


# ============ Providers ============

class PerformanceProvider(ABC):
    """历史表现数据提供者接口"""

    @abstractmethod
    async def get_performance(self, strategy_name: str) -> Optional[HistoricalPerformance]:
        """获取策略历史表现"""
        pass

    @abstractmethod
    async def update_performance(self, strategy_name: str, success: bool, return_value: float) -> None:
        """更新策略表现"""
        pass


class SimulatedPerformanceProvider(PerformanceProvider):
    """模拟表现数据提供者"""

    def __init__(self):
        self._performances: Dict[str, HistoricalPerformance] = {}

    async def get_performance(self, strategy_name: str) -> Optional[HistoricalPerformance]:
        """获取策略历史表现"""
        return self._performances.get(strategy_name)

    async def update_performance(self, strategy_name: str, success: bool, return_value: float) -> None:
        """更新策略表现"""
        if strategy_name not in self._performances:
            self._performances[strategy_name] = HistoricalPerformance(strategy_name=strategy_name)

        perf = self._performances[strategy_name]
        perf.total_signals += 1

        if success:
            perf.successful_signals += 1
        else:
            perf.failed_signals += 1

        # 更新胜率
        perf.win_rate = perf.successful_signals / perf.total_signals if perf.total_signals > 0 else 0.0

        # 更新平均收益
        old_avg = perf.average_return
        n = perf.total_signals
        perf.average_return = (old_avg * (n - 1) + return_value) / n

        perf.last_updated = datetime.now()

    def set_performance(self, strategy_name: str, performance: HistoricalPerformance) -> None:
        """设置策略表现"""
        self._performances[strategy_name] = performance


# ============ Main Class ============

class TradeSignalAggregator:
    """多策略信号聚合器"""

    def __init__(
        self,
        config: Optional[AggregatorConfig] = None,
        performance_provider: Optional[PerformanceProvider] = None,
        aggregation_callback: Optional[Callable[[AggregatedSignal], None]] = None,
        quality_callback: Optional[Callable[[SignalQualityMetrics], None]] = None,
    ):
        self.config = config or AggregatorConfig()
        self.performance_provider = performance_provider or simulated_default(SimulatedPerformanceProvider, "data from TradeSignalAggregator's simulated performance_provider (SimulatedPerformanceProvider)")
        self.aggregation_callback = aggregation_callback
        self.quality_callback = quality_callback

        # 状态
        self._state = AggregatorState.STOPPED
        self._processing_task: Optional[asyncio.Task] = None

        # 信号缓冲区
        self._signal_buffer: Dict[str, deque] = {}  # symbol -> deque[StrategySignal]

        # 策略权重
        self._strategy_weights: Dict[str, float] = {}  # strategy_name -> weight

        # 历史记录
        self._aggregation_history: deque = deque(maxlen=self.config.history_size)
        self._signal_history: deque = deque(maxlen=self.config.history_size)

        # 质量指标
        self._quality_metrics: Dict[str, SignalQualityMetrics] = {}  # symbol -> metrics

        # 统计
        self._statistics = {
            "total_signals_received": 0,
            "total_aggregations": 0,
            "successful_aggregations": 0,
            "failed_aggregations": 0,
            "conflicts_resolved": 0,
        }

    @property
    def state(self) -> AggregatorState:
        return self._state

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动聚合器"""
        if self._state == AggregatorState.RUNNING:
            return

        logger.info("Starting Trade Signal Aggregator")
        self._state = AggregatorState.RUNNING
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Trade Signal Aggregator started")

    async def stop(self) -> None:
        """停止聚合器"""
        if self._state == AggregatorState.STOPPED:
            return

        logger.info("Stopping Trade Signal Aggregator")
        self._state = AggregatorState.STOPPED

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

        logger.info("Trade Signal Aggregator stopped")

    async def pause(self) -> None:
        """暂停聚合器"""
        if self._state == AggregatorState.RUNNING:
            self._state = AggregatorState.PAUSED
            logger.info("Trade Signal Aggregator paused")

    async def resume(self) -> None:
        """恢复聚合器"""
        if self._state == AggregatorState.PAUSED:
            self._state = AggregatorState.RUNNING
            logger.info("Trade Signal Aggregator resumed")

    async def _processing_loop(self) -> None:
        """处理循环"""
        while self._state in (AggregatorState.RUNNING, AggregatorState.PAUSED):
            try:
                if self._state == AggregatorState.RUNNING:
                    await self._process_all_buffers()
                await asyncio.sleep(self.config.aggregation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self._state = AggregatorState.ERROR

    # ============ Signal Management ============

    async def add_signal(self, signal: StrategySignal) -> None:
        """添加策略信号"""
        self._statistics["total_signals_received"] += 1

        # 验证信号
        if not self._validate_signal(signal):
            logger.warning(f"Invalid signal from {signal.strategy_name}: {signal.signal_id}")
            return

        # 过滤低质量信号
        if not self._filter_signal(signal):
            logger.debug(f"Signal filtered: {signal.strategy_name} - {signal.symbol}")
            return

        # 添加到缓冲区
        symbol = signal.symbol
        if symbol not in self._signal_buffer:
            self._signal_buffer[symbol] = deque(maxlen=self.config.signal_buffer_size)

        self._signal_buffer[symbol].append(signal)
        self._signal_history.append(signal)

        # 更新策略权重
        await self._update_strategy_weight(signal.strategy_name)

    async def add_signals_batch(self, signals: List[StrategySignal]) -> None:
        """批量添加信号"""
        for signal in signals:
            await self.add_signal(signal)

    def _validate_signal(self, signal: StrategySignal) -> bool:
        """验证信号有效性"""
        if not signal.symbol or not signal.strategy_name:
            return False

        if not (0.0 <= signal.strength <= 1.0):
            return False

        if not (0.0 <= signal.confidence <= 1.0):
            return False

        return True

    def _filter_signal(self, signal: StrategySignal) -> bool:
        """过滤信号"""
        if signal.strength < self.config.min_signal_strength:
            return False

        if signal.confidence < self.config.min_signal_confidence:
            return False

        return True

    # ============ Weight Management ============

    async def _update_strategy_weight(self, strategy_name: str) -> None:
        """更新策略权重"""
        if not self.config.use_historical_weighting:
            self._strategy_weights[strategy_name] = 1.0
            return

        # 获取历史表现
        perf = await self.performance_provider.get_performance(strategy_name)

        if perf:
            weight = perf.update_weight()
        else:
            weight = 1.0

        # 应用权重衰减
        if strategy_name in self._strategy_weights:
            old_weight = self._strategy_weights[strategy_name]
            weight = self.config.weight_decay_rate * old_weight + (1 - self.config.weight_decay_rate) * weight

        # 限制权重范围
        weight = max(self.config.min_weight, min(self.config.max_weight, weight))

        self._strategy_weights[strategy_name] = weight

    def set_strategy_weight(self, strategy_name: str, weight: float) -> None:
        """手动设置策略权重"""
        weight = max(self.config.min_weight, min(self.config.max_weight, weight))
        self._strategy_weights[strategy_name] = weight

    def get_strategy_weight(self, strategy_name: str) -> float:
        """获取策略权重"""
        return self._strategy_weights.get(strategy_name, 1.0)

    # ============ Signal Aggregation ============

    async def _process_all_buffers(self) -> None:
        """处理所有缓冲区"""
        for symbol in list(self._signal_buffer.keys()):
            if len(self._signal_buffer[symbol]) >= self.config.min_signals_required:
                await self.aggregate_signals(symbol)

    async def aggregate_signals(self, symbol: str) -> Optional[AggregatedSignal]:
        """聚合指定资产的信号"""
        if symbol not in self._signal_buffer or len(self._signal_buffer[symbol]) == 0:
            return None

        signals = list(self._signal_buffer[symbol])

        # 检查最小信号数
        if len(signals) < self.config.min_signals_required:
            return None

        try:
            # 应用策略权重
            self._apply_weights(signals)

            # 归一化
            if self.config.normalize_strength:
                self._normalize_strength(signals)
            if self.config.normalize_confidence:
                self._normalize_confidence(signals)

            # 执行聚合
            if self.config.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
                aggregated = self._aggregate_weighted_average(signals, symbol)
            elif self.config.aggregation_method == AggregationMethod.MAJORITY_VOTING:
                aggregated = self._aggregate_majority_voting(signals, symbol)
            elif self.config.aggregation_method == AggregationMethod.CONFIDENCE_WEIGHTED:
                aggregated = self._aggregate_confidence_weighted(signals, symbol)
            elif self.config.aggregation_method == AggregationMethod.PERFORMANCE_WEIGHTED:
                aggregated = self._aggregate_performance_weighted(signals, symbol)
            elif self.config.aggregation_method == AggregationMethod.BAYESIAN_FUSION:
                aggregated = self._aggregate_bayesian_fusion(signals, symbol)
            else:
                aggregated = self._aggregate_ensemble(signals, symbol)

            # 计算质量指标
            quality_metrics = self._calculate_quality_metrics(signals, aggregated)
            self._quality_metrics[symbol] = quality_metrics

            if self.quality_callback:
                self.quality_callback(quality_metrics)

            # 检查质量要求
            if self.config.require_minimum_quality:
                if aggregated.confidence < self.config.quality_threshold:
                    logger.debug(f"Aggregated signal quality too low for {symbol}")
                    self._statistics["failed_aggregations"] += 1
                    return None

            # 记录历史
            self._aggregation_history.append(aggregated)
            self._statistics["total_aggregations"] += 1
            self._statistics["successful_aggregations"] += 1

            # 清空缓冲区
            self._signal_buffer[symbol].clear()

            # 回调
            if self.aggregation_callback:
                self.aggregation_callback(aggregated)

            logger.info(f"Aggregated signal for {symbol}: {aggregated.direction.value} "
                       f"strength={aggregated.strength:.2f} confidence={aggregated.confidence:.2f}")

            return aggregated

        except Exception as e:
            logger.error(f"Error aggregating signals for {symbol}: {e}")
            self._statistics["failed_aggregations"] += 1
            return None

    def _apply_weights(self, signals: List[StrategySignal]) -> None:
        """应用策略权重"""
        for signal in signals:
            weight = self.get_strategy_weight(signal.strategy_name)
            signal.weight = weight

    def _normalize_strength(self, signals: List[StrategySignal]) -> None:
        """归一化信号强度"""
        if len(signals) <= 1:
            return

        strengths = [s.strength for s in signals]
        max_strength = max(strengths)
        min_strength = min(strengths)

        if max_strength - min_strength < 0.01:
            return

        for signal in signals:
            signal.strength = (signal.strength - min_strength) / (max_strength - min_strength)

    def _normalize_confidence(self, signals: List[StrategySignal]) -> None:
        """归一化置信度"""
        if len(signals) <= 1:
            return

        confidences = [s.confidence for s in signals]
        max_conf = max(confidences)
        min_conf = min(confidences)

        if max_conf - min_conf < 0.01:
            return

        for signal in signals:
            signal.confidence = (signal.confidence - min_conf) / (max_conf - min_conf)

    # ============ Aggregation Methods ============

    def _aggregate_weighted_average(
        self,
        signals: List[StrategySignal],
        symbol: str
    ) -> AggregatedSignal:
        """加权平均聚合"""
        # 计算方向得分
        direction_scores = {
            SignalDirection.STRONG_LONG: 0.0,
            SignalDirection.LONG: 0.0,
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.SHORT: 0.0,
            SignalDirection.STRONG_SHORT: 0.0,
        }

        total_weight = sum(s.weight for s in signals)

        for signal in signals:
            direction_scores[signal.direction] += signal.weight * signal.strength

        # 归一化得分
        if total_weight > 0:
            for direction in direction_scores:
                direction_scores[direction] /= total_weight

        # 选择最高得分的方向
        final_direction = max(direction_scores, key=direction_scores.get)
        final_strength = direction_scores[final_direction]

        # 加权平均置信度
        weighted_confidence = sum(s.confidence * s.weight for s in signals) / total_weight if total_weight > 0 else 0.0

        # 计算一致性和冲突
        agreement_ratio, conflict_level = self._calculate_agreement_metrics(signals)

        # 评估质量
        quality = self._assess_signal_quality(final_strength, weighted_confidence, agreement_ratio)

        return AggregatedSignal(
            aggregation_id=f"AGG-{uuid.uuid4().hex[:8].upper()}",
            symbol=symbol,
            direction=final_direction,
            strength=final_strength,
            confidence=weighted_confidence,
            quality=quality,
            timestamp=datetime.now(),
            input_signals=signals,
            signal_count=len(signals),
            agreement_ratio=agreement_ratio,
            conflict_level=conflict_level,
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            total_weight=total_weight,
        )

    def _aggregate_majority_voting(
        self,
        signals: List[StrategySignal],
        symbol: str
    ) -> AggregatedSignal:
        """多数投票聚合"""
        # 统计每个方向的票数
        direction_votes = {}
        for signal in signals:
            direction = signal.direction
            direction_votes[direction] = direction_votes.get(direction, 0) + 1

        # 找出最多票数的方向
        final_direction = max(direction_votes, key=direction_votes.get)
        vote_count = direction_votes[final_direction]

        # 计算强度和置信度 (来自该方向的信号的平均值)
        direction_signals = [s for s in signals if s.direction == final_direction]
        final_strength = statistics.mean([s.strength for s in direction_signals])
        final_confidence = statistics.mean([s.confidence for s in direction_signals])

        # 一致性指标
        agreement_ratio = vote_count / len(signals)
        conflict_level = 1.0 - agreement_ratio

        quality = self._assess_signal_quality(final_strength, final_confidence, agreement_ratio)

        return AggregatedSignal(
            aggregation_id=f"AGG-{uuid.uuid4().hex[:8].upper()}",
            symbol=symbol,
            direction=final_direction,
            strength=final_strength,
            confidence=final_confidence,
            quality=quality,
            timestamp=datetime.now(),
            input_signals=signals,
            signal_count=len(signals),
            agreement_ratio=agreement_ratio,
            conflict_level=conflict_level,
            aggregation_method=AggregationMethod.MAJORITY_VOTING,
            total_weight=sum(s.weight for s in signals),
        )

    def _aggregate_confidence_weighted(
        self,
        signals: List[StrategySignal],
        symbol: str
    ) -> AggregatedSignal:
        """置信度加权聚合"""
        # 使用置信度作为权重
        direction_scores = {}
        total_confidence = sum(s.confidence for s in signals)

        for signal in signals:
            direction = signal.direction
            score = signal.strength * signal.confidence
            direction_scores[direction] = direction_scores.get(direction, 0.0) + score

        # 归一化
        if total_confidence > 0:
            for direction in direction_scores:
                direction_scores[direction] /= total_confidence

        final_direction = max(direction_scores, key=direction_scores.get)
        final_strength = direction_scores[final_direction]
        final_confidence = statistics.mean([s.confidence for s in signals])

        agreement_ratio, conflict_level = self._calculate_agreement_metrics(signals)
        quality = self._assess_signal_quality(final_strength, final_confidence, agreement_ratio)

        return AggregatedSignal(
            aggregation_id=f"AGG-{uuid.uuid4().hex[:8].upper()}",
            symbol=symbol,
            direction=final_direction,
            strength=final_strength,
            confidence=final_confidence,
            quality=quality,
            timestamp=datetime.now(),
            input_signals=signals,
            signal_count=len(signals),
            agreement_ratio=agreement_ratio,
            conflict_level=conflict_level,
            aggregation_method=AggregationMethod.CONFIDENCE_WEIGHTED,
            total_weight=total_confidence,
        )

    def _aggregate_performance_weighted(
        self,
        signals: List[StrategySignal],
        symbol: str
    ) -> AggregatedSignal:
        """历史表现加权聚合"""
        # 类似加权平均，但权重已通过历史表现更新
        return self._aggregate_weighted_average(signals, symbol)

    def _aggregate_bayesian_fusion(
        self,
        signals: List[StrategySignal],
        symbol: str
    ) -> AggregatedSignal:
        """贝叶斯融合聚合"""
        # 简化的贝叶斯方法：组合各信号的置信度
        # P(direction | signals) ∝ P(signals | direction) * P(direction)

        direction_probs = {
            SignalDirection.STRONG_LONG: 0.2,
            SignalDirection.LONG: 0.2,
            SignalDirection.NEUTRAL: 0.2,
            SignalDirection.SHORT: 0.2,
            SignalDirection.STRONG_SHORT: 0.2,
        }

        # 更新先验概率
        for signal in signals:
            direction = signal.direction
            likelihood = signal.confidence * signal.strength
            direction_probs[direction] *= (1 + likelihood)

        # 归一化
        total_prob = sum(direction_probs.values())
        if total_prob > 0:
            for direction in direction_probs:
                direction_probs[direction] /= total_prob

        final_direction = max(direction_probs, key=direction_probs.get)
        final_strength = direction_probs[final_direction]
        final_confidence = statistics.mean([s.confidence for s in signals])

        agreement_ratio, conflict_level = self._calculate_agreement_metrics(signals)
        quality = self._assess_signal_quality(final_strength, final_confidence, agreement_ratio)

        return AggregatedSignal(
            aggregation_id=f"AGG-{uuid.uuid4().hex[:8].upper()}",
            symbol=symbol,
            direction=final_direction,
            strength=final_strength,
            confidence=final_confidence,
            quality=quality,
            timestamp=datetime.now(),
            input_signals=signals,
            signal_count=len(signals),
            agreement_ratio=agreement_ratio,
            conflict_level=conflict_level,
            aggregation_method=AggregationMethod.BAYESIAN_FUSION,
            total_weight=sum(s.weight for s in signals),
        )

    def _aggregate_ensemble(
        self,
        signals: List[StrategySignal],
        symbol: str
    ) -> AggregatedSignal:
        """集成方法聚合"""
        # 组合多种方法的结果
        results = []

        # 加权平均
        wa_result = self._aggregate_weighted_average(signals, symbol)
        results.append(wa_result)

        # 多数投票
        mv_result = self._aggregate_majority_voting(signals, symbol)
        results.append(mv_result)

        # 置信度加权
        cw_result = self._aggregate_confidence_weighted(signals, symbol)
        results.append(cw_result)

        # 选择置信度最高的结果
        final_result = max(results, key=lambda r: r.confidence)
        final_result.aggregation_method = AggregationMethod.ENSEMBLE
        final_result.aggregation_id = f"AGG-{uuid.uuid4().hex[:8].upper()}"

        return final_result

    # ============ Helper Methods ============

    def _calculate_agreement_metrics(
        self,
        signals: List[StrategySignal]
    ) -> Tuple[float, float]:
        """计算一致性指标"""
        if len(signals) <= 1:
            return 1.0, 0.0

        # 统计方向
        direction_counts = {}
        for signal in signals:
            direction = signal.direction
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        # 一致性比率 (最多的方向占比)
        max_count = max(direction_counts.values())
        agreement_ratio = max_count / len(signals)

        # 冲突水平 (多头和空头信号同时存在)
        long_count = (
            direction_counts.get(SignalDirection.STRONG_LONG, 0) +
            direction_counts.get(SignalDirection.LONG, 0)
        )
        short_count = (
            direction_counts.get(SignalDirection.STRONG_SHORT, 0) +
            direction_counts.get(SignalDirection.SHORT, 0)
        )

        if long_count > 0 and short_count > 0:
            conflict_level = min(long_count, short_count) / len(signals)
        else:
            conflict_level = 0.0

        return agreement_ratio, conflict_level

    def _assess_signal_quality(
        self,
        strength: float,
        confidence: float,
        agreement_ratio: float
    ) -> SignalQuality:
        """评估信号质量"""
        # 综合评分
        quality_score = 0.3 * strength + 0.4 * confidence + 0.3 * agreement_ratio

        if quality_score >= 0.8:
            return SignalQuality.EXCELLENT
        elif quality_score >= 0.6:
            return SignalQuality.GOOD
        elif quality_score >= 0.4:
            return SignalQuality.FAIR
        elif quality_score >= 0.2:
            return SignalQuality.POOR
        else:
            return SignalQuality.VERY_POOR

    def _calculate_quality_metrics(
        self,
        signals: List[StrategySignal],
        aggregated: AggregatedSignal
    ) -> SignalQualityMetrics:
        """计算信号质量指标"""
        # 方向统计
        long_signals = sum(1 for s in signals if s.direction in [SignalDirection.STRONG_LONG, SignalDirection.LONG])
        short_signals = sum(1 for s in signals if s.direction in [SignalDirection.STRONG_SHORT, SignalDirection.SHORT])
        neutral_signals = sum(1 for s in signals if s.direction == SignalDirection.NEUTRAL)

        # 强度统计
        strengths = [s.strength for s in signals]

        # 置信度统计
        confidences = [s.confidence for s in signals]

        # 冲突统计
        conflict_count = 1 if (long_signals > 0 and short_signals > 0) else 0
        conflict_severity = aggregated.conflict_level

        # 一致性分数
        consistency_score = aggregated.agreement_ratio

        # 多样性分数 (不同策略类型的数量)
        unique_types = len(set(s.signal_type for s in signals))
        diversity_score = min(1.0, unique_types / 5.0)

        # 综合质量
        overall_quality = 0.4 * consistency_score + 0.3 * statistics.mean(confidences) + 0.3 * diversity_score

        return SignalQualityMetrics(
            symbol=aggregated.symbol,
            timestamp=datetime.now(),
            overall_quality=overall_quality,
            consistency_score=consistency_score,
            diversity_score=diversity_score,
            total_signals=len(signals),
            long_signals=long_signals,
            short_signals=short_signals,
            neutral_signals=neutral_signals,
            average_strength=statistics.mean(strengths),
            max_strength=max(strengths),
            min_strength=min(strengths),
            average_confidence=statistics.mean(confidences),
            max_confidence=max(confidences),
            min_confidence=min(confidences),
            conflict_count=conflict_count,
            conflict_severity=conflict_severity,
        )

    # ============ Query Methods ============

    def get_aggregation_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[AggregatedSignal]:
        """获取聚合历史"""
        history = list(self._aggregation_history)

        if symbol:
            history = [a for a in history if a.symbol == symbol]

        return history[-limit:]

    def get_signal_history(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[StrategySignal]:
        """获取信号历史"""
        history = list(self._signal_history)

        if strategy_name:
            history = [s for s in history if s.strategy_name == strategy_name]

        if symbol:
            history = [s for s in history if s.symbol == symbol]

        return history[-limit:]

    def get_quality_metrics(self, symbol: str) -> Optional[SignalQualityMetrics]:
        """获取质量指标"""
        return self._quality_metrics.get(symbol)

    def get_all_quality_metrics(self) -> Dict[str, SignalQualityMetrics]:
        """获取所有质量指标"""
        return dict(self._quality_metrics)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = dict(self._statistics)
        stats["state"] = self._state.value
        stats["active_symbols"] = len(self._signal_buffer)
        stats["total_strategy_weights"] = len(self._strategy_weights)
        stats["config"] = self.config.to_dict()

        if stats["total_aggregations"] > 0:
            stats["success_rate"] = stats["successful_aggregations"] / stats["total_aggregations"]
        else:
            stats["success_rate"] = 0.0

        return stats

    def get_strategy_weights(self) -> Dict[str, float]:
        """获取所有策略权重"""
        return dict(self._strategy_weights)

    def clear_buffer(self, symbol: Optional[str] = None) -> None:
        """清空缓冲区"""
        if symbol:
            if symbol in self._signal_buffer:
                self._signal_buffer[symbol].clear()
        else:
            self._signal_buffer.clear()


# ============ Factory Functions ============

def create_trade_signal_aggregator(
    aggregation_method: AggregationMethod = AggregationMethod.CONFIDENCE_WEIGHTED,
    min_signals_required: int = 2,
    use_historical_weighting: bool = True,
    **kwargs
) -> TradeSignalAggregator:
    """
    创建交易信号聚合器

    Args:
        aggregation_method: 聚合方法
        min_signals_required: 最小所需信号数
        use_historical_weighting: 是否使用历史表现加权
        **kwargs: 其他配置参数

    Returns:
        TradeSignalAggregator实例
    """
    config = AggregatorConfig(
        aggregation_method=aggregation_method,
        min_signals_required=min_signals_required,
        use_historical_weighting=use_historical_weighting,
        **{k: v for k, v in kwargs.items() if hasattr(AggregatorConfig, k)}
    )

    return TradeSignalAggregator(config=config)


# ============ Exports ============

__all__ = [
    # Enums
    "SignalDirection",
    "SignalType",
    "AggregationMethod",
    "ConflictResolutionStrategy",
    "SignalQuality",
    "AggregatorState",
    # Data Classes
    "StrategySignal",
    "HistoricalPerformance",
    "AggregatedSignal",
    "SignalQualityMetrics",
    "AggregatorConfig",
    # Providers
    "PerformanceProvider",
    "SimulatedPerformanceProvider",
    # Main Class
    "TradeSignalAggregator",
    # Factory Functions
    "create_trade_signal_aggregator",
]
