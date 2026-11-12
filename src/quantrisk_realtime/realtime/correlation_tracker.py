"""
Correlation Tracker - 实时相关性追踪器

实时追踪和分析资产相关性，包括：
- 实时相关性矩阵计算 (Real-time Correlation Matrix)
- 滚动窗口相关性分析 (Rolling Window Correlation)
- 相关性变化检测 (Correlation Change Detection)
- 相关性突变警报 (Correlation Break Alert)
- 动态相关性模型 DCC-GARCH (Dynamic Conditional Correlation)
- 相关性聚类分析 (Correlation Clustering)
- 尾部相关性计算 (Tail Correlation)
- 相关性预测 (Correlation Forecasting)

这是投资组合风险管理、资产配置和对冲策略的核心模块。
"""

import asyncio
import logging
import math
import statistics
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Enums ============

class CorrelationStrength(str, Enum):
    """相关性强度分类"""
    VERY_STRONG_POSITIVE = "very_strong_positive"    # 0.8 ~ 1.0
    STRONG_POSITIVE = "strong_positive"              # 0.6 ~ 0.8
    MODERATE_POSITIVE = "moderate_positive"          # 0.4 ~ 0.6
    WEAK_POSITIVE = "weak_positive"                  # 0.2 ~ 0.4
    NEGLIGIBLE = "negligible"                        # -0.2 ~ 0.2
    WEAK_NEGATIVE = "weak_negative"                  # -0.4 ~ -0.2
    MODERATE_NEGATIVE = "moderate_negative"          # -0.6 ~ -0.4
    STRONG_NEGATIVE = "strong_negative"              # -0.8 ~ -0.6
    VERY_STRONG_NEGATIVE = "very_strong_negative"    # -1.0 ~ -0.8


class CorrelationTrend(str, Enum):
    """相关性趋势"""
    INCREASING = "increasing"        # 相关性增强
    DECREASING = "decreasing"        # 相关性减弱
    STABLE = "stable"                # 相关性稳定
    BREAKING = "breaking"            # 相关性突变


class CorrelationRegime(str, Enum):
    """相关性状态"""
    NORMAL = "normal"                # 正常状态
    HIGH_CORRELATION = "high_corr"   # 高相关性状态
    LOW_CORRELATION = "low_corr"     # 低相关性状态
    CRISIS = "crisis"                # 危机状态 (相关性趋向1)
    DIVERGENCE = "divergence"        # 分化状态


class AlertType(str, Enum):
    """告警类型"""
    CORRELATION_BREAK = "correlation_break"          # 相关性突变
    HIGH_CORRELATION = "high_correlation"            # 高相关性
    CORRELATION_SPIKE = "correlation_spike"          # 相关性飙升
    TAIL_CORRELATION = "tail_correlation"            # 尾部相关性
    REGIME_CHANGE = "regime_change"                  # 状态变化
    CLUSTER_CHANGE = "cluster_change"                # 聚类变化


class TrackerState(str, Enum):
    """追踪器状态"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ Data Classes ============

@dataclass
class ReturnData:
    """收益率数据"""
    symbol: str
    timestamp: datetime
    return_value: float
    price: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "return_value": self.return_value,
            "price": self.price,
        }


@dataclass
class CorrelationPair:
    """相关性配对"""
    symbol1: str
    symbol2: str
    correlation: float
    timestamp: datetime
    strength: CorrelationStrength
    sample_size: int = 0
    p_value: float = 0.0              # 统计显著性
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    @property
    def pair_key(self) -> str:
        """生成配对键 (保证顺序一致性)"""
        return f"{min(self.symbol1, self.symbol2)}-{max(self.symbol1, self.symbol2)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "correlation": self.correlation,
            "timestamp": self.timestamp.isoformat(),
            "strength": self.strength.value,
            "sample_size": self.sample_size,
            "p_value": self.p_value,
            "confidence_interval": list(self.confidence_interval),
            "pair_key": self.pair_key,
        }


@dataclass
class CorrelationMatrix:
    """相关性矩阵"""
    symbols: List[str]
    matrix: List[List[float]]
    timestamp: datetime
    sample_size: int = 0
    average_correlation: float = 0.0
    min_correlation: float = 0.0
    max_correlation: float = 0.0

    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """获取两个资产的相关性"""
        try:
            idx1 = self.symbols.index(symbol1)
            idx2 = self.symbols.index(symbol2)
            return self.matrix[idx1][idx2]
        except (ValueError, IndexError):
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "matrix": self.matrix,
            "timestamp": self.timestamp.isoformat(),
            "sample_size": self.sample_size,
            "average_correlation": self.average_correlation,
            "min_correlation": self.min_correlation,
            "max_correlation": self.max_correlation,
        }


@dataclass
class RollingCorrelation:
    """滚动相关性"""
    symbol1: str
    symbol2: str
    window_size: int
    correlations: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    current_correlation: float = 0.0
    trend: CorrelationTrend = CorrelationTrend.STABLE
    volatility: float = 0.0           # 相关性波动率

    def add_correlation(self, corr: float, timestamp: datetime) -> None:
        """添加新的相关性值"""
        self.correlations.append(corr)
        self.timestamps.append(timestamp)
        self.current_correlation = corr

        # 保持窗口大小
        if len(self.correlations) > self.window_size * 2:
            self.correlations = self.correlations[-self.window_size * 2:]
            self.timestamps = self.timestamps[-self.window_size * 2:]

        # 更新趋势和波动率
        self._update_trend()
        self._update_volatility()

    def _update_trend(self) -> None:
        """更新趋势"""
        if len(self.correlations) < 10:
            return

        recent = self.correlations[-10:]
        if len(recent) < 2:
            return

        # 简单线性趋势
        slope = (recent[-1] - recent[0]) / len(recent)

        if abs(slope) < 0.01:
            self.trend = CorrelationTrend.STABLE
        elif abs(slope) > 0.1:
            self.trend = CorrelationTrend.BREAKING
        elif slope > 0:
            self.trend = CorrelationTrend.INCREASING
        else:
            self.trend = CorrelationTrend.DECREASING

    def _update_volatility(self) -> None:
        """更新相关性波动率"""
        if len(self.correlations) < 20:
            return

        recent = self.correlations[-20:]
        self.volatility = statistics.stdev(recent) if len(recent) > 1 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "window_size": self.window_size,
            "current_correlation": self.current_correlation,
            "trend": self.trend.value,
            "volatility": self.volatility,
            "history_size": len(self.correlations),
        }


@dataclass
class DCCGarchState:
    """DCC-GARCH 动态条件相关性状态"""
    symbol1: str
    symbol2: str
    dynamic_correlation: float
    timestamp: datetime
    variance1: float = 0.0
    variance2: float = 0.0
    covariance: float = 0.0
    conditional_vol1: float = 0.0
    conditional_vol2: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "dynamic_correlation": self.dynamic_correlation,
            "timestamp": self.timestamp.isoformat(),
            "variance1": self.variance1,
            "variance2": self.variance2,
            "covariance": self.covariance,
            "conditional_vol1": self.conditional_vol1,
            "conditional_vol2": self.conditional_vol2,
        }


@dataclass
class TailCorrelation:
    """尾部相关性 (极端事件下的相关性)"""
    symbol1: str
    symbol2: str
    lower_tail_correlation: float    # 下尾相关性 (双跌)
    upper_tail_correlation: float    # 上尾相关性 (双涨)
    tail_threshold: float            # 尾部阈值 (百分位)
    timestamp: datetime
    lower_tail_events: int = 0
    upper_tail_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "lower_tail_correlation": self.lower_tail_correlation,
            "upper_tail_correlation": self.upper_tail_correlation,
            "tail_threshold": self.tail_threshold,
            "timestamp": self.timestamp.isoformat(),
            "lower_tail_events": self.lower_tail_events,
            "upper_tail_events": self.upper_tail_events,
        }


@dataclass
class CorrelationCluster:
    """相关性聚类"""
    cluster_id: str
    symbols: List[str]
    centroid_symbol: str              # 中心资产
    average_correlation: float        # 簇内平均相关性
    timestamp: datetime
    stability_score: float = 0.0      # 聚类稳定性

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "symbols": self.symbols,
            "centroid_symbol": self.centroid_symbol,
            "average_correlation": self.average_correlation,
            "timestamp": self.timestamp.isoformat(),
            "stability_score": self.stability_score,
        }


@dataclass
class CorrelationForecast:
    """相关性预测"""
    symbol1: str
    symbol2: str
    current_correlation: float
    forecast_correlation: float
    forecast_horizon: int             # 预测期数
    confidence: float                 # 预测置信度
    timestamp: datetime
    forecast_method: str = "arima"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "current_correlation": self.current_correlation,
            "forecast_correlation": self.forecast_correlation,
            "forecast_horizon": self.forecast_horizon,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "forecast_method": self.forecast_method,
        }


@dataclass
class CorrelationBreak:
    """相关性突变事件"""
    break_id: str
    symbol1: str
    symbol2: str
    old_correlation: float
    new_correlation: float
    change_magnitude: float
    timestamp: datetime
    detection_method: str
    significance: float = 0.0         # 显著性水平

    def to_dict(self) -> Dict[str, Any]:
        return {
            "break_id": self.break_id,
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "old_correlation": self.old_correlation,
            "new_correlation": self.new_correlation,
            "change_magnitude": self.change_magnitude,
            "timestamp": self.timestamp.isoformat(),
            "detection_method": self.detection_method,
            "significance": self.significance,
        }


@dataclass
class CorrelationAlert:
    """相关性告警"""
    alert_id: str
    alert_type: AlertType
    symbol1: str
    symbol2: str
    message: str
    severity: str                     # "low", "medium", "high", "critical"
    correlation_value: float
    timestamp: datetime
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "message": self.message,
            "severity": self.severity,
            "correlation_value": self.correlation_value,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "metadata": self.metadata,
        }


@dataclass
class TrackerConfig:
    """追踪器配置"""
    # 相关性计算参数
    min_correlation_window: int = 20        # 最小计算窗口
    default_correlation_window: int = 60    # 默认窗口
    rolling_window: int = 20                # 滚动窗口大小

    # 突变检测参数
    break_threshold: float = 0.3            # 突变阈值
    break_lookback: int = 10                # 突变检测回溯期

    # 尾部相关性参数
    tail_percentile: float = 0.05           # 尾部百分位 (5%)

    # DCC-GARCH 参数
    dcc_alpha: float = 0.05                 # DCC alpha 参数
    dcc_beta: float = 0.90                  # DCC beta 参数

    # 聚类参数
    cluster_threshold: float = 0.7          # 聚类相关性阈值
    min_cluster_size: int = 2               # 最小聚类大小

    # 预测参数
    forecast_horizon: int = 5               # 预测期数
    forecast_method: str = "ema"            # 预测方法: ema, arima, var

    # 告警参数
    high_correlation_threshold: float = 0.9  # 高相关性阈值
    spike_threshold: float = 0.5             # 相关性飙升阈值

    # 监控参数
    update_interval: float = 1.0             # 更新间隔(秒)
    max_pairs: int = 1000                    # 最大配对数
    history_size: int = 1000                 # 历史记录大小

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_correlation_window": self.min_correlation_window,
            "default_correlation_window": self.default_correlation_window,
            "rolling_window": self.rolling_window,
            "break_threshold": self.break_threshold,
            "break_lookback": self.break_lookback,
            "tail_percentile": self.tail_percentile,
            "dcc_alpha": self.dcc_alpha,
            "dcc_beta": self.dcc_beta,
            "cluster_threshold": self.cluster_threshold,
            "min_cluster_size": self.min_cluster_size,
            "forecast_horizon": self.forecast_horizon,
            "forecast_method": self.forecast_method,
            "high_correlation_threshold": self.high_correlation_threshold,
            "spike_threshold": self.spike_threshold,
            "update_interval": self.update_interval,
            "max_pairs": self.max_pairs,
            "history_size": self.history_size,
        }


# ============ Providers ============

class ReturnDataProvider(ABC):
    """收益率数据提供者接口"""

    @abstractmethod
    async def get_returns(
        self,
        symbols: List[str],
        count: int
    ) -> Dict[str, List[float]]:
        """获取多个资产的收益率数据"""
        pass

    @abstractmethod
    async def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """获取最新价格"""
        pass


class SimulatedReturnDataProvider(ReturnDataProvider):
    """模拟收益率数据提供者"""

    def __init__(self):
        self._returns: Dict[str, List[float]] = {}
        self._prices: Dict[str, List[float]] = {}

    def add_returns(self, symbol: str, returns: List[float]) -> None:
        """添加收益率数据"""
        self._returns[symbol] = returns

    def add_price(self, symbol: str, price: float) -> None:
        """添加价格数据"""
        if symbol not in self._prices:
            self._prices[symbol] = []
        self._prices[symbol].append(price)

    async def get_returns(
        self,
        symbols: List[str],
        count: int
    ) -> Dict[str, List[float]]:
        """获取多个资产的收益率数据"""
        result = {}
        for symbol in symbols:
            returns = self._returns.get(symbol, [])
            result[symbol] = returns[-count:] if len(returns) >= count else returns
        return result

    async def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """获取最新价格"""
        result = {}
        for symbol in symbols:
            prices = self._prices.get(symbol, [])
            if prices:
                result[symbol] = prices[-1]
        return result


# ============ Main Class ============

class CorrelationTracker:
    """实时相关性追踪器"""

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        data_provider: Optional[ReturnDataProvider] = None,
        correlation_callback: Optional[Callable[[CorrelationPair], None]] = None,
        alert_callback: Optional[Callable[[CorrelationAlert], None]] = None,
        break_callback: Optional[Callable[[CorrelationBreak], None]] = None,
    ):
        self.config = config or TrackerConfig()
        self.data_provider = data_provider or simulated_default(SimulatedReturnDataProvider, "data from CorrelationTracker's simulated data_provider (SimulatedReturnDataProvider)")
        self.correlation_callback = correlation_callback
        self.alert_callback = alert_callback
        self.break_callback = break_callback

        # 状态
        self._state = TrackerState.STOPPED
        self._monitor_task: Optional[asyncio.Task] = None

        # 数据存储
        self._symbols: Set[str] = set()
        self._return_data: Dict[str, deque] = {}  # symbol -> deque[ReturnData]

        # 相关性数据
        self._current_correlations: Dict[str, CorrelationPair] = {}  # pair_key -> CorrelationPair
        self._correlation_matrix: Optional[CorrelationMatrix] = None
        self._rolling_correlations: Dict[str, RollingCorrelation] = {}  # pair_key -> RollingCorrelation

        # DCC-GARCH 状态
        self._dcc_states: Dict[str, DCCGarchState] = {}  # pair_key -> DCCGarchState

        # 尾部相关性
        self._tail_correlations: Dict[str, TailCorrelation] = {}  # pair_key -> TailCorrelation

        # 聚类
        self._clusters: List[CorrelationCluster] = []
        self._cluster_history: deque = deque(maxlen=100)

        # 预测
        self._forecasts: Dict[str, CorrelationForecast] = {}  # pair_key -> CorrelationForecast

        # 突变检测
        self._breaks: deque = deque(maxlen=500)

        # 告警
        self._alerts: deque = deque(maxlen=1000)

        # 历史
        self._correlation_history: deque = deque(maxlen=self.config.history_size)

    @property
    def state(self) -> TrackerState:
        return self._state

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动追踪器"""
        if self._state == TrackerState.RUNNING:
            return

        logger.info("Starting Correlation Tracker")
        self._state = TrackerState.RUNNING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Correlation Tracker started")

    async def stop(self) -> None:
        """停止追踪器"""
        if self._state == TrackerState.STOPPED:
            return

        logger.info("Stopping Correlation Tracker")
        self._state = TrackerState.STOPPED

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Correlation Tracker stopped")

    async def pause(self) -> None:
        """暂停追踪器"""
        if self._state == TrackerState.RUNNING:
            self._state = TrackerState.PAUSED
            logger.info("Correlation Tracker paused")

    async def resume(self) -> None:
        """恢复追踪器"""
        if self._state == TrackerState.PAUSED:
            self._state = TrackerState.RUNNING
            logger.info("Correlation Tracker resumed")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state in (TrackerState.RUNNING, TrackerState.PAUSED):
            try:
                if self._state == TrackerState.RUNNING:
                    await self._update_all_correlations()
                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self._state = TrackerState.ERROR

    # ============ Symbol Management ============

    def add_symbol(self, symbol: str) -> None:
        """添加监控资产"""
        if symbol not in self._symbols:
            self._symbols.add(symbol)
            self._return_data[symbol] = deque(maxlen=self.config.history_size)
            logger.info(f"Added symbol: {symbol}")

    def remove_symbol(self, symbol: str) -> None:
        """移除监控资产"""
        if symbol in self._symbols:
            self._symbols.remove(symbol)
            self._return_data.pop(symbol, None)

            # 清理相关配对
            pairs_to_remove = [
                key for key in self._current_correlations.keys()
                if symbol in key
            ]
            for key in pairs_to_remove:
                self._current_correlations.pop(key, None)
                self._rolling_correlations.pop(key, None)
                self._dcc_states.pop(key, None)
                self._tail_correlations.pop(key, None)
                self._forecasts.pop(key, None)

            logger.info(f"Removed symbol: {symbol}")

    def get_symbols(self) -> List[str]:
        """获取所有监控资产"""
        return list(self._symbols)

    # ============ Return Data Management ============

    async def add_return_data(self, return_data: ReturnData) -> None:
        """添加收益率数据"""
        symbol = return_data.symbol

        # 自动添加资产
        if symbol not in self._symbols:
            self.add_symbol(symbol)

        self._return_data[symbol].append(return_data)

        # 触发更新
        if len(self._symbols) >= 2:
            await self._update_all_correlations()

    async def add_price(self, symbol: str, price: float, timestamp: datetime) -> None:
        """添加价格数据 (自动计算收益率)"""
        if symbol not in self._return_data or len(self._return_data[symbol]) == 0:
            # 首次添加，收益率为0
            return_data = ReturnData(
                symbol=symbol,
                timestamp=timestamp,
                return_value=0.0,
                price=price,
            )
            await self.add_return_data(return_data)
            return

        # 计算收益率
        last_data = self._return_data[symbol][-1]
        if last_data.price > 0:
            return_value = (price - last_data.price) / last_data.price
        else:
            return_value = 0.0

        return_data = ReturnData(
            symbol=symbol,
            timestamp=timestamp,
            return_value=return_value,
            price=price,
        )
        await self.add_return_data(return_data)

    # ============ Correlation Calculation ============

    async def _update_all_correlations(self) -> None:
        """更新所有相关性"""
        if len(self._symbols) < 2:
            return

        # 更新相关性矩阵
        await self._calculate_correlation_matrix()

        # 更新配对相关性
        await self._calculate_pairwise_correlations()

        # 更新滚动相关性
        self._update_rolling_correlations()

        # 更新 DCC-GARCH
        self._update_dcc_garch()

        # 更新尾部相关性
        self._calculate_tail_correlations()

        # 检测突变
        self._detect_correlation_breaks()

        # 更新聚类
        self._update_clusters()

        # 更新预测
        self._update_forecasts()

    async def _calculate_correlation_matrix(self) -> None:
        """计算相关性矩阵"""
        symbols = sorted(list(self._symbols))
        n = len(symbols)

        if n < 2:
            return

        # 获取收益率数据
        returns_dict = {}
        min_length = float('inf')

        for symbol in symbols:
            returns = [r.return_value for r in self._return_data[symbol]]
            if len(returns) < self.config.min_correlation_window:
                return
            returns_dict[symbol] = returns
            min_length = min(min_length, len(returns))

        # 构建矩阵
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    symbol1 = symbols[i]
                    symbol2 = symbols[j]

                    returns1 = returns_dict[symbol1][-min_length:]
                    returns2 = returns_dict[symbol2][-min_length:]

                    corr = self._calculate_correlation(returns1, returns2)
                    matrix[i][j] = corr

        # 计算统计信息
        correlations = [
            matrix[i][j]
            for i in range(n)
            for j in range(i + 1, n)
        ]

        self._correlation_matrix = CorrelationMatrix(
            symbols=symbols,
            matrix=matrix,
            timestamp=datetime.now(),
            sample_size=min_length,
            average_correlation=statistics.mean(correlations) if correlations else 0.0,
            min_correlation=min(correlations) if correlations else 0.0,
            max_correlation=max(correlations) if correlations else 0.0,
        )

    async def _calculate_pairwise_correlations(self) -> None:
        """计算配对相关性"""
        symbols = list(self._symbols)

        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i + 1:]:
                await self._calculate_pair_correlation(symbol1, symbol2)

    async def _calculate_pair_correlation(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[CorrelationPair]:
        """计算单个配对的相关性"""
        # 获取收益率数据
        returns1 = [r.return_value for r in self._return_data.get(symbol1, [])]
        returns2 = [r.return_value for r in self._return_data.get(symbol2, [])]

        min_length = min(len(returns1), len(returns2))
        if min_length < self.config.min_correlation_window:
            return None

        # 对齐数据
        returns1 = returns1[-min_length:]
        returns2 = returns2[-min_length:]

        # 计算相关性
        correlation = self._calculate_correlation(returns1, returns2)

        # 分类强度
        strength = self._classify_correlation_strength(correlation)

        # 计算统计显著性
        p_value = self._calculate_correlation_pvalue(correlation, min_length)

        # 计算置信区间
        ci = self._calculate_correlation_ci(correlation, min_length)

        # 创建配对对象
        pair = CorrelationPair(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=correlation,
            timestamp=datetime.now(),
            strength=strength,
            sample_size=min_length,
            p_value=p_value,
            confidence_interval=ci,
        )

        # 存储
        self._current_correlations[pair.pair_key] = pair

        # 回调
        if self.correlation_callback:
            self.correlation_callback(pair)

        return pair

    def _calculate_correlation(
        self,
        returns1: List[float],
        returns2: List[float]
    ) -> float:
        """计算皮尔逊相关系数"""
        if len(returns1) != len(returns2) or len(returns1) < 2:
            return 0.0

        n = len(returns1)
        mean1 = statistics.mean(returns1)
        mean2 = statistics.mean(returns2)

        # 计算协方差和标准差
        covariance = sum(
            (returns1[i] - mean1) * (returns2[i] - mean2)
            for i in range(n)
        ) / n

        std1 = statistics.stdev(returns1)
        std2 = statistics.stdev(returns2)

        if std1 == 0 or std2 == 0:
            return 0.0

        correlation = covariance / (std1 * std2)

        # 限制范围
        return max(-1.0, min(1.0, correlation))

    def _classify_correlation_strength(self, corr: float) -> CorrelationStrength:
        """分类相关性强度"""
        abs_corr = abs(corr)

        if abs_corr >= 0.8:
            strength_level = "very_strong"
        elif abs_corr >= 0.6:
            strength_level = "strong"
        elif abs_corr >= 0.4:
            strength_level = "moderate"
        elif abs_corr >= 0.2:
            strength_level = "weak"
        else:
            return CorrelationStrength.NEGLIGIBLE

        direction = "positive" if corr > 0 else "negative"
        return CorrelationStrength(f"{strength_level}_{direction}")

    def _calculate_correlation_pvalue(
        self,
        correlation: float,
        sample_size: int
    ) -> float:
        """计算相关性统计显著性 p-value"""
        if sample_size < 3:
            return 1.0

        # t-统计量
        t_stat = correlation * math.sqrt(sample_size - 2) / math.sqrt(1 - correlation ** 2)

        # 简化的 p-value 估算
        df = sample_size - 2
        p_value = 2 * (1 - self._t_distribution_cdf(abs(t_stat), df))

        return max(0.0, min(1.0, p_value))

    def _t_distribution_cdf(self, t: float, df: int) -> float:
        """t分布累积分布函数 (简化近似)"""
        # 使用正态分布近似
        if df > 30:
            return self._normal_cdf(t)

        # 简化的t分布近似
        x = t / math.sqrt(df)
        return 0.5 + 0.5 * math.erf(x / math.sqrt(2))

    def _normal_cdf(self, x: float) -> float:
        """标准正态分布累积分布函数"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _calculate_correlation_ci(
        self,
        correlation: float,
        sample_size: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """计算相关性置信区间 (Fisher Z变换)"""
        if sample_size < 4:
            return (correlation, correlation)

        # Fisher Z变换
        z = 0.5 * math.log((1 + correlation) / (1 - correlation))

        # 标准误差
        se = 1.0 / math.sqrt(sample_size - 3)

        # Z分数 (95%置信区间)
        z_score = 1.96 if confidence == 0.95 else 2.576

        # 置信区间
        z_lower = z - z_score * se
        z_upper = z + z_score * se

        # 逆变换
        ci_lower = (math.exp(2 * z_lower) - 1) / (math.exp(2 * z_lower) + 1)
        ci_upper = (math.exp(2 * z_upper) - 1) / (math.exp(2 * z_upper) + 1)

        return (ci_lower, ci_upper)

    # ============ Rolling Correlation ============

    def _update_rolling_correlations(self) -> None:
        """更新滚动相关性"""
        for pair_key, pair in self._current_correlations.items():
            if pair_key not in self._rolling_correlations:
                self._rolling_correlations[pair_key] = RollingCorrelation(
                    symbol1=pair.symbol1,
                    symbol2=pair.symbol2,
                    window_size=self.config.rolling_window,
                )

            rolling = self._rolling_correlations[pair_key]
            rolling.add_correlation(pair.correlation, pair.timestamp)

    # ============ DCC-GARCH ============

    def _update_dcc_garch(self) -> None:
        """更新 DCC-GARCH 动态条件相关性"""
        alpha = self.config.dcc_alpha
        beta = self.config.dcc_beta

        for pair_key, pair in self._current_correlations.items():
            symbol1, symbol2 = pair.symbol1, pair.symbol2

            # 获取收益率
            returns1 = [r.return_value for r in self._return_data.get(symbol1, [])]
            returns2 = [r.return_value for r in self._return_data.get(symbol2, [])]

            if len(returns1) < 20 or len(returns2) < 20:
                continue

            # 计算条件方差 (简化GARCH)
            var1 = statistics.variance(returns1[-20:])
            var2 = statistics.variance(returns2[-20:])

            # 计算协方差
            min_len = min(len(returns1), len(returns2))
            ret1 = returns1[-min_len:]
            ret2 = returns2[-min_len:]

            mean1 = statistics.mean(ret1)
            mean2 = statistics.mean(ret2)

            covariance = sum(
                (ret1[i] - mean1) * (ret2[i] - mean2)
                for i in range(len(ret1))
            ) / len(ret1)

            # 获取上一期状态
            old_state = self._dcc_states.get(pair_key)

            # DCC 更新
            if old_state:
                # Q_t = (1 - alpha - beta) * Q_bar + alpha * z_{t-1} * z'_{t-1} + beta * Q_{t-1}
                q11 = (1 - alpha - beta) * var1 + alpha * returns1[-1]**2 + beta * old_state.variance1
                q22 = (1 - alpha - beta) * var2 + alpha * returns2[-1]**2 + beta * old_state.variance2
                q12 = (1 - alpha - beta) * covariance + alpha * returns1[-1] * returns2[-1] + beta * old_state.covariance
            else:
                q11 = var1
                q22 = var2
                q12 = covariance

            # 计算动态相关性
            dynamic_corr = q12 / (math.sqrt(q11 * q22)) if q11 > 0 and q22 > 0 else 0.0
            dynamic_corr = max(-1.0, min(1.0, dynamic_corr))

            # 创建新状态
            dcc_state = DCCGarchState(
                symbol1=symbol1,
                symbol2=symbol2,
                dynamic_correlation=dynamic_corr,
                timestamp=datetime.now(),
                variance1=q11,
                variance2=q22,
                covariance=q12,
                conditional_vol1=math.sqrt(q11),
                conditional_vol2=math.sqrt(q22),
            )

            self._dcc_states[pair_key] = dcc_state

    # ============ Tail Correlation ============

    def _calculate_tail_correlations(self) -> None:
        """计算尾部相关性"""
        threshold = self.config.tail_percentile

        for pair_key, pair in self._current_correlations.items():
            symbol1, symbol2 = pair.symbol1, pair.symbol2

            # 获取收益率
            returns1 = [r.return_value for r in self._return_data.get(symbol1, [])]
            returns2 = [r.return_value for r in self._return_data.get(symbol2, [])]

            if len(returns1) < 50 or len(returns2) < 50:
                continue

            min_len = min(len(returns1), len(returns2))
            ret1 = returns1[-min_len:]
            ret2 = returns2[-min_len:]

            # 计算阈值
            sorted_ret1 = sorted(ret1)
            sorted_ret2 = sorted(ret2)

            lower_threshold1 = sorted_ret1[int(len(sorted_ret1) * threshold)]
            lower_threshold2 = sorted_ret2[int(len(sorted_ret2) * threshold)]

            upper_threshold1 = sorted_ret1[int(len(sorted_ret1) * (1 - threshold))]
            upper_threshold2 = sorted_ret2[int(len(sorted_ret2) * (1 - threshold))]

            # 识别尾部事件
            lower_tail_pairs = [
                (ret1[i], ret2[i])
                for i in range(len(ret1))
                if ret1[i] <= lower_threshold1 and ret2[i] <= lower_threshold2
            ]

            upper_tail_pairs = [
                (ret1[i], ret2[i])
                for i in range(len(ret1))
                if ret1[i] >= upper_threshold1 and ret2[i] >= upper_threshold2
            ]

            # 计算尾部相关性
            lower_tail_corr = 0.0
            if len(lower_tail_pairs) >= 3:
                lt1 = [p[0] for p in lower_tail_pairs]
                lt2 = [p[1] for p in lower_tail_pairs]
                lower_tail_corr = self._calculate_correlation(lt1, lt2)

            upper_tail_corr = 0.0
            if len(upper_tail_pairs) >= 3:
                ut1 = [p[0] for p in upper_tail_pairs]
                ut2 = [p[1] for p in upper_tail_pairs]
                upper_tail_corr = self._calculate_correlation(ut1, ut2)

            # 创建尾部相关性对象
            tail_corr = TailCorrelation(
                symbol1=symbol1,
                symbol2=symbol2,
                lower_tail_correlation=lower_tail_corr,
                upper_tail_correlation=upper_tail_corr,
                tail_threshold=threshold,
                timestamp=datetime.now(),
                lower_tail_events=len(lower_tail_pairs),
                upper_tail_events=len(upper_tail_pairs),
            )

            self._tail_correlations[pair_key] = tail_corr

            # 检查尾部相关性告警
            if lower_tail_corr > 0.8:
                self._create_alert(
                    AlertType.TAIL_CORRELATION,
                    symbol1,
                    symbol2,
                    f"High lower tail correlation ({lower_tail_corr:.2f}) detected",
                    "high",
                    lower_tail_corr,
                    {"tail_type": "lower", "events": len(lower_tail_pairs)}
                )

    # ============ Correlation Break Detection ============

    def _detect_correlation_breaks(self) -> None:
        """检测相关性突变"""
        lookback = self.config.break_lookback
        threshold = self.config.break_threshold

        for pair_key, rolling in self._rolling_correlations.items():
            if len(rolling.correlations) < lookback * 2:
                continue

            # 比较最近期和之前期的平均相关性
            recent_corr = statistics.mean(rolling.correlations[-lookback:])
            previous_corr = statistics.mean(
                rolling.correlations[-lookback * 2:-lookback]
            )

            change = abs(recent_corr - previous_corr)

            if change >= threshold:
                # 检测到突变
                pair = self._current_correlations.get(pair_key)
                if not pair:
                    continue

                correlation_break = CorrelationBreak(
                    break_id=str(uuid.uuid4())[:8],
                    symbol1=pair.symbol1,
                    symbol2=pair.symbol2,
                    old_correlation=previous_corr,
                    new_correlation=recent_corr,
                    change_magnitude=change,
                    timestamp=datetime.now(),
                    detection_method="rolling_average",
                    significance=change / threshold,
                )

                self._breaks.append(correlation_break)

                # 回调
                if self.break_callback:
                    self.break_callback(correlation_break)

                # 创建告警
                severity = "critical" if change > threshold * 1.5 else "high"
                self._create_alert(
                    AlertType.CORRELATION_BREAK,
                    pair.symbol1,
                    pair.symbol2,
                    f"Correlation break detected: {previous_corr:.2f} -> {recent_corr:.2f}",
                    severity,
                    recent_corr,
                    {"change": change, "break_id": correlation_break.break_id}
                )

    # ============ Correlation Clustering ============

    def _calculate_cluster_stability(self, current_symbols: frozenset) -> float:
        """
        计算聚类稳定性分数

        稳定性基于当前聚类与历史聚类的 Jaccard 相似度。
        分数范围 [0, 1]，1 表示完全稳定（历史上始终存在相同聚类）。

        Args:
            current_symbols: 当前聚类的资产集合

        Returns:
            稳定性分数 (0-1)
        """
        if not self._cluster_history or len(current_symbols) < 2:
            return 0.0

        # 计算与历史聚类的 Jaccard 相似度
        similarities = []
        for timestamp, historical_clusters in self._cluster_history:
            max_similarity = 0.0
            for hist_cluster in historical_clusters:
                hist_symbols = frozenset(hist_cluster.symbols)
                # Jaccard 相似度: |A ∩ B| / |A ∪ B|
                intersection = len(current_symbols & hist_symbols)
                union = len(current_symbols | hist_symbols)
                if union > 0:
                    similarity = intersection / union
                    max_similarity = max(max_similarity, similarity)
            similarities.append(max_similarity)

        if not similarities:
            return 0.0

        # 使用指数加权平均，近期历史权重更高
        n = len(similarities)
        if n == 1:
            return similarities[0]

        # 指数衰减权重 (最近的权重最高)
        weights = [0.9 ** (n - 1 - i) for i in range(n)]
        weight_sum = sum(weights)

        if weight_sum == 0:
            return statistics.mean(similarities)

        weighted_avg = sum(s * w for s, w in zip(similarities, weights)) / weight_sum
        return min(1.0, max(0.0, weighted_avg))

    def _update_clusters(self) -> None:
        """更新相关性聚类"""
        if not self._correlation_matrix:
            return

        symbols = self._correlation_matrix.symbols
        matrix = self._correlation_matrix.matrix
        threshold = self.config.cluster_threshold

        # 简单的层次聚类
        clusters: List[Set[str]] = []
        assigned: Set[str] = set()

        for i, symbol in enumerate(symbols):
            if symbol in assigned:
                continue

            # 创建新簇
            cluster: Set[str] = {symbol}
            assigned.add(symbol)

            # 找到高相关的资产
            for j, other_symbol in enumerate(symbols):
                if other_symbol in assigned:
                    continue

                if abs(matrix[i][j]) >= threshold:
                    cluster.add(other_symbol)
                    assigned.add(other_symbol)

            if len(cluster) >= self.config.min_cluster_size:
                clusters.append(cluster)

        # 创建聚类对象
        new_clusters = []
        for cluster_symbols in clusters:
            cluster_list = list(cluster_symbols)

            # 计算簇内平均相关性
            correlations = []
            for i, sym1 in enumerate(cluster_list):
                for sym2 in cluster_list[i + 1:]:
                    corr = self._correlation_matrix.get_correlation(sym1, sym2)
                    if corr is not None:
                        correlations.append(abs(corr))

            avg_corr = statistics.mean(correlations) if correlations else 0.0

            # 选择中心资产 (平均相关性最高)
            centroid = cluster_list[0]
            max_avg = 0.0
            for sym in cluster_list:
                sym_correlations = [
                    abs(self._correlation_matrix.get_correlation(sym, other) or 0.0)
                    for other in cluster_list
                    if other != sym
                ]
                sym_avg = statistics.mean(sym_correlations) if sym_correlations else 0.0
                if sym_avg > max_avg:
                    max_avg = sym_avg
                    centroid = sym

            # 计算聚类稳定性分数
            stability_score = self._calculate_cluster_stability(frozenset(cluster_list))

            cluster_obj = CorrelationCluster(
                cluster_id=str(uuid.uuid4())[:8],
                symbols=cluster_list,
                centroid_symbol=centroid,
                average_correlation=avg_corr,
                timestamp=datetime.now(),
                stability_score=stability_score,
            )

            new_clusters.append(cluster_obj)

        # 检测聚类变化
        if self._clusters:
            old_cluster_map = {
                frozenset(c.symbols): c
                for c in self._clusters
            }
            new_cluster_map = {
                frozenset(c.symbols): c
                for c in new_clusters
            }

            # 识别变化
            added = set(new_cluster_map.keys()) - set(old_cluster_map.keys())
            removed = set(old_cluster_map.keys()) - set(new_cluster_map.keys())

            if added or removed:
                # 聚类发生变化
                for cluster_symbols in added:
                    cluster = new_cluster_map[cluster_symbols]
                    self._create_alert(
                        AlertType.CLUSTER_CHANGE,
                        cluster.centroid_symbol,
                        "",
                        f"New correlation cluster formed with {len(cluster.symbols)} assets",
                        "medium",
                        cluster.average_correlation,
                        {"cluster_id": cluster.cluster_id, "symbols": cluster.symbols}
                    )

        self._clusters = new_clusters
        self._cluster_history.append((datetime.now(), new_clusters))

    # ============ Correlation Forecasting ============

    def _update_forecasts(self) -> None:
        """更新相关性预测"""
        method = self.config.forecast_method
        horizon = self.config.forecast_horizon

        for pair_key, rolling in self._rolling_correlations.items():
            if len(rolling.correlations) < 20:
                continue

            pair = self._current_correlations.get(pair_key)
            if not pair:
                continue

            # 使用不同方法预测
            if method == "ema":
                forecast_corr = self._forecast_ema(rolling.correlations, horizon)
            elif method == "arima":
                forecast_corr = self._forecast_arima(rolling.correlations, horizon)
            elif method == "var":
                forecast_corr = self._forecast_var(rolling.correlations, horizon)
            else:
                forecast_corr = rolling.current_correlation

            # 计算置信度
            confidence = self._calculate_forecast_confidence(rolling.correlations)

            forecast = CorrelationForecast(
                symbol1=pair.symbol1,
                symbol2=pair.symbol2,
                current_correlation=rolling.current_correlation,
                forecast_correlation=forecast_corr,
                forecast_horizon=horizon,
                confidence=confidence,
                timestamp=datetime.now(),
                forecast_method=method,
            )

            self._forecasts[pair_key] = forecast

    def _forecast_ema(self, series: List[float], horizon: int) -> float:
        """指数移动平均预测"""
        if len(series) < 5:
            return series[-1]

        # 使用EMA平滑
        alpha = 2 / (horizon + 1)
        ema = series[0]

        for value in series[1:]:
            ema = alpha * value + (1 - alpha) * ema

        return ema

    def _forecast_arima(self, series: List[float], horizon: int) -> float:
        """ARIMA 预测 (简化版 AR(1))"""
        if len(series) < 10:
            return series[-1]

        # 简化为 AR(1) 模型
        # x_t = phi * x_{t-1} + epsilon

        # 估计 phi
        x_lag = series[:-1]
        x_current = series[1:]

        mean_lag = statistics.mean(x_lag)
        mean_current = statistics.mean(x_current)

        numerator = sum(
            (x_lag[i] - mean_lag) * (x_current[i] - mean_current)
            for i in range(len(x_lag))
        )
        denominator = sum((x - mean_lag) ** 2 for x in x_lag)

        phi = numerator / denominator if denominator != 0 else 0.5
        phi = max(-0.99, min(0.99, phi))  # 确保平稳

        # 多步预测
        forecast = series[-1]
        for _ in range(horizon):
            forecast = phi * forecast

        return forecast

    def _forecast_var(self, series: List[float], horizon: int) -> float:
        """向量自回归预测 (简化版)"""
        # 由于是单变量，退化为 AR
        return self._forecast_arima(series, horizon)

    def _calculate_forecast_confidence(self, series: List[float]) -> float:
        """计算预测置信度"""
        if len(series) < 10:
            return 0.3

        # 基于序列稳定性
        volatility = statistics.stdev(series[-20:]) if len(series) >= 20 else statistics.stdev(series)

        # 波动率越低，置信度越高
        confidence = 1.0 / (1.0 + volatility * 10)

        return max(0.0, min(1.0, confidence))

    # ============ Alert Management ============

    def _create_alert(
        self,
        alert_type: AlertType,
        symbol1: str,
        symbol2: str,
        message: str,
        severity: str,
        correlation_value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """创建告警"""
        alert = CorrelationAlert(
            alert_id=str(uuid.uuid4())[:8],
            alert_type=alert_type,
            symbol1=symbol1,
            symbol2=symbol2,
            message=message,
            severity=severity,
            correlation_value=correlation_value,
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

    # ============ Query Methods ============

    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """获取两个资产的相关性"""
        symbols = sorted([symbol1, symbol2])
        pair_key = f"{symbols[0]}-{symbols[1]}"
        pair = self._current_correlations.get(pair_key)
        return pair.correlation if pair else None

    def get_correlation_pair(self, symbol1: str, symbol2: str) -> Optional[CorrelationPair]:
        """获取相关性配对对象"""
        symbols = sorted([symbol1, symbol2])
        pair_key = f"{symbols[0]}-{symbols[1]}"
        return self._current_correlations.get(pair_key)

    def get_correlation_matrix(self) -> Optional[CorrelationMatrix]:
        """获取相关性矩阵"""
        return self._correlation_matrix

    def get_rolling_correlation(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[RollingCorrelation]:
        """获取滚动相关性"""
        symbols = sorted([symbol1, symbol2])
        pair_key = f"{symbols[0]}-{symbols[1]}"
        return self._rolling_correlations.get(pair_key)

    def get_dcc_state(self, symbol1: str, symbol2: str) -> Optional[DCCGarchState]:
        """获取 DCC-GARCH 状态"""
        symbols = sorted([symbol1, symbol2])
        pair_key = f"{symbols[0]}-{symbols[1]}"
        return self._dcc_states.get(pair_key)

    def get_tail_correlation(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[TailCorrelation]:
        """获取尾部相关性"""
        symbols = sorted([symbol1, symbol2])
        pair_key = f"{symbols[0]}-{symbols[1]}"
        return self._tail_correlations.get(pair_key)

    def get_clusters(self) -> List[CorrelationCluster]:
        """获取当前聚类"""
        return list(self._clusters)

    def get_forecast(self, symbol1: str, symbol2: str) -> Optional[CorrelationForecast]:
        """获取相关性预测"""
        symbols = sorted([symbol1, symbol2])
        pair_key = f"{symbols[0]}-{symbols[1]}"
        return self._forecasts.get(pair_key)

    def get_breaks(
        self,
        symbol1: Optional[str] = None,
        symbol2: Optional[str] = None,
        limit: int = 100,
    ) -> List[CorrelationBreak]:
        """获取相关性突变记录"""
        breaks = list(self._breaks)

        if symbol1:
            breaks = [
                b for b in breaks
                if symbol1 in (b.symbol1, b.symbol2)
            ]
        if symbol2:
            breaks = [
                b for b in breaks
                if symbol2 in (b.symbol1, b.symbol2)
            ]

        return breaks[-limit:]

    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[CorrelationAlert]:
        """获取告警"""
        alerts = list(self._alerts)

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "symbols": len(self._symbols),
            "total_pairs": len(self._current_correlations),
            "total_breaks": len(self._breaks),
            "total_alerts": len(self._alerts),
            "unacknowledged_alerts": len([a for a in self._alerts if not a.acknowledged]),
            "clusters": len(self._clusters),
            "matrix_average_correlation": (
                self._correlation_matrix.average_correlation
                if self._correlation_matrix else 0.0
            ),
            "config": self.config.to_dict(),
        }


# ============ Factory Functions ============

def create_correlation_tracker(
    data_provider: Optional[ReturnDataProvider] = None,
    **config_kwargs,
) -> CorrelationTracker:
    """创建相关性追踪器"""
    config = TrackerConfig(**config_kwargs)
    return CorrelationTracker(
        config=config,
        data_provider=data_provider,
    )


# ============ Exports ============

__all__ = [
    # Enums
    "CorrelationStrength",
    "CorrelationTrend",
    "CorrelationRegime",
    "AlertType",
    "TrackerState",
    # Data Classes
    "ReturnData",
    "CorrelationPair",
    "CorrelationMatrix",
    "RollingCorrelation",
    "DCCGarchState",
    "TailCorrelation",
    "CorrelationCluster",
    "CorrelationForecast",
    "CorrelationBreak",
    "CorrelationAlert",
    "TrackerConfig",
    # Providers
    "ReturnDataProvider",
    "SimulatedReturnDataProvider",
    # Main Class
    "CorrelationTracker",
    # Factory Functions
    "create_correlation_tracker",
]
