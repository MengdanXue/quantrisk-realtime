"""
回撤监控器 (Drawdown Monitor)

提供实时回撤监控和分析功能:
- 实时回撤计算 (当前回撤、最大回撤)
- 回撤持续时间追踪
- 回撤恢复时间预估
- 多时间框架回撤 (日/周/月/年)
- 回撤分解 (按资产、策略、因子)
- 回撤警报系统 (阈值触发)
- 历史回撤统计分析
- 回撤风险预测

Copyright (c) 2025 TraderX. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import logging
import math
import statistics

logger = logging.getLogger(__name__)


# ============== Enums ==============

class TimeFrame(str, Enum):
    """时间框架"""
    INTRADAY = "intraday"       # 日内
    DAILY = "daily"             # 日
    WEEKLY = "weekly"           # 周
    MONTHLY = "monthly"         # 月
    QUARTERLY = "quarterly"     # 季
    YEARLY = "yearly"           # 年
    ALL_TIME = "all_time"       # 全部


class DrawdownState(str, Enum):
    """回撤状态"""
    NONE = "none"               # 无回撤(处于峰值)
    IN_DRAWDOWN = "in_drawdown" # 回撤中
    RECOVERING = "recovering"   # 恢复中
    RECOVERED = "recovered"     # 已恢复


class DrawdownSeverity(str, Enum):
    """回撤严重程度"""
    MINOR = "minor"             # 轻微 (<5%)
    MODERATE = "moderate"       # 中等 (5-10%)
    SIGNIFICANT = "significant" # 显著 (10-20%)
    SEVERE = "severe"           # 严重 (20-30%)
    EXTREME = "extreme"         # 极端 (>30%)


class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitorState(str, Enum):
    """监控状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class RecoveryModel(str, Enum):
    """恢复预测模型"""
    LINEAR = "linear"               # 线性恢复
    EXPONENTIAL = "exponential"     # 指数恢复
    MEAN_REVERSION = "mean_reversion"  # 均值回归
    HISTORICAL = "historical"       # 历史平均


# ============== Data Classes ==============

@dataclass
class DrawdownPeriod:
    """
    回撤期间数据

    记录一次完整的回撤周期(从峰值到谷底到恢复)
    """
    start_time: datetime            # 回撤开始时间(峰值时间)
    peak_value: float              # 峰值价值
    trough_time: Optional[datetime] = None  # 谷底时间
    trough_value: Optional[float] = None    # 谷底价值
    end_time: Optional[datetime] = None     # 回撤结束时间(恢复时间)
    recovery_value: Optional[float] = None  # 恢复时价值
    max_drawdown_pct: float = 0.0  # 最大回撤百分比
    duration_days: float = 0.0     # 回撤持续时间(天)
    recovery_days: Optional[float] = None   # 恢复时间(天)
    state: DrawdownState = DrawdownState.IN_DRAWDOWN
    severity: DrawdownSeverity = DrawdownSeverity.MINOR

    # 分解信息
    asset_contributions: Dict[str, float] = field(default_factory=dict)
    strategy_contributions: Dict[str, float] = field(default_factory=dict)
    factor_contributions: Dict[str, float] = field(default_factory=dict)

    # 统计信息
    underwater_time_pct: float = 0.0  # 水下时间百分比
    avg_recovery_rate: float = 0.0     # 平均恢复速率

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'peak_value': self.peak_value,
            'trough_time': self.trough_time.isoformat() if self.trough_time else None,
            'trough_value': self.trough_value,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'recovery_value': self.recovery_value,
            'max_drawdown_pct': self.max_drawdown_pct,
            'duration_days': self.duration_days,
            'recovery_days': self.recovery_days,
            'state': self.state.value,
            'severity': self.severity.value,
            'asset_contributions': self.asset_contributions,
            'strategy_contributions': self.strategy_contributions,
            'factor_contributions': self.factor_contributions,
            'underwater_time_pct': self.underwater_time_pct,
            'avg_recovery_rate': self.avg_recovery_rate,
        }


@dataclass
class DrawdownMetrics:
    """
    回撤指标

    当前时刻的回撤状态和指标
    """
    timestamp: datetime
    current_value: float
    peak_value: float
    current_drawdown_pct: float     # 当前回撤百分比
    current_drawdown_value: float   # 当前回撤金额
    max_drawdown_pct: float         # 历史最大回撤
    max_drawdown_value: float       # 历史最大回撤金额

    # 时间指标
    days_in_drawdown: float = 0.0   # 当前回撤持续天数
    days_since_peak: float = 0.0    # 距离峰值天数

    # 状态
    state: DrawdownState = DrawdownState.NONE
    severity: DrawdownSeverity = DrawdownSeverity.MINOR

    # 预测
    estimated_recovery_days: Optional[float] = None
    estimated_recovery_date: Optional[datetime] = None
    recovery_probability: float = 0.0  # 恢复概率

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'current_value': self.current_value,
            'peak_value': self.peak_value,
            'current_drawdown_pct': self.current_drawdown_pct,
            'current_drawdown_value': self.current_drawdown_value,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_drawdown_value': self.max_drawdown_value,
            'days_in_drawdown': self.days_in_drawdown,
            'days_since_peak': self.days_since_peak,
            'state': self.state.value,
            'severity': self.severity.value,
            'estimated_recovery_days': self.estimated_recovery_days,
            'estimated_recovery_date': self.estimated_recovery_date.isoformat() if self.estimated_recovery_date else None,
            'recovery_probability': self.recovery_probability,
        }


@dataclass
class TimeFrameDrawdown:
    """
    时间框架回撤数据

    特定时间框架内的回撤统计
    """
    timeframe: TimeFrame
    period_start: datetime
    period_end: datetime
    peak_value: float
    trough_value: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    recovery_pct: float = 0.0       # 恢复百分比(0-100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timeframe': self.timeframe.value,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'peak_value': self.peak_value,
            'trough_value': self.trough_value,
            'max_drawdown_pct': self.max_drawdown_pct,
            'current_drawdown_pct': self.current_drawdown_pct,
            'recovery_pct': self.recovery_pct,
        }


@dataclass
class DrawdownAlert:
    """回撤告警"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    message: str
    drawdown_pct: float
    threshold_pct: float
    current_value: float
    peak_value: float
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'drawdown_pct': self.drawdown_pct,
            'threshold_pct': self.threshold_pct,
            'current_value': self.current_value,
            'peak_value': self.peak_value,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
        }


@dataclass
class DrawdownDecomposition:
    """
    回撤分解

    将回撤分解到不同维度(资产、策略、因子)
    """
    timestamp: datetime
    total_drawdown_pct: float

    # 按资产分解
    asset_drawdowns: Dict[str, float] = field(default_factory=dict)
    asset_contributions_pct: Dict[str, float] = field(default_factory=dict)

    # 按策略分解
    strategy_drawdowns: Dict[str, float] = field(default_factory=dict)
    strategy_contributions_pct: Dict[str, float] = field(default_factory=dict)

    # 按因子分解
    factor_drawdowns: Dict[str, float] = field(default_factory=dict)
    factor_contributions_pct: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_drawdown_pct': self.total_drawdown_pct,
            'asset_drawdowns': self.asset_drawdowns,
            'asset_contributions_pct': self.asset_contributions_pct,
            'strategy_drawdowns': self.strategy_drawdowns,
            'strategy_contributions_pct': self.strategy_contributions_pct,
            'factor_drawdowns': self.factor_drawdowns,
            'factor_contributions_pct': self.factor_contributions_pct,
        }


@dataclass
class DrawdownStatistics:
    """
    回撤统计数据

    历史回撤的统计分析
    """
    total_drawdowns: int            # 总回撤次数
    avg_drawdown_pct: float         # 平均回撤
    median_drawdown_pct: float      # 中位数回撤
    max_drawdown_pct: float         # 最大回撤
    min_drawdown_pct: float         # 最小回撤
    std_drawdown_pct: float         # 回撤标准差

    # 持续时间统计
    avg_duration_days: float        # 平均持续时间
    max_duration_days: float        # 最长持续时间
    median_duration_days: float     # 中位数持续时间

    # 恢复时间统计
    avg_recovery_days: float        # 平均恢复时间
    max_recovery_days: float        # 最长恢复时间
    median_recovery_days: float     # 中位数恢复时间

    # 严重程度分布
    severity_distribution: Dict[str, int] = field(default_factory=dict)

    # 频率
    drawdowns_per_year: float = 0.0
    avg_time_underwater_pct: float = 0.0  # 平均水下时间百分比

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_drawdowns': self.total_drawdowns,
            'avg_drawdown_pct': self.avg_drawdown_pct,
            'median_drawdown_pct': self.median_drawdown_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'min_drawdown_pct': self.min_drawdown_pct,
            'std_drawdown_pct': self.std_drawdown_pct,
            'avg_duration_days': self.avg_duration_days,
            'max_duration_days': self.max_duration_days,
            'median_duration_days': self.median_duration_days,
            'avg_recovery_days': self.avg_recovery_days,
            'max_recovery_days': self.max_recovery_days,
            'median_recovery_days': self.median_recovery_days,
            'severity_distribution': self.severity_distribution,
            'drawdowns_per_year': self.drawdowns_per_year,
            'avg_time_underwater_pct': self.avg_time_underwater_pct,
        }


@dataclass
class DrawdownPrediction:
    """
    回撤预测

    预测未来回撤风险
    """
    timestamp: datetime
    prediction_horizon_days: int    # 预测时间范围

    # 回撤概率
    drawdown_5pct_probability: float    # 5%回撤概率
    drawdown_10pct_probability: float   # 10%回撤概率
    drawdown_20pct_probability: float   # 20%回撤概率

    # 预期回撤
    expected_drawdown_pct: float        # 预期最大回撤
    var_95_drawdown_pct: float          # 95% VaR回撤
    cvar_95_drawdown_pct: float         # 95% CVaR回撤

    # 恢复预测
    expected_recovery_days: float       # 预期恢复天数
    recovery_model: RecoveryModel       # 使用的恢复模型
    confidence_level: float = 0.0       # 预测置信度

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'prediction_horizon_days': self.prediction_horizon_days,
            'drawdown_5pct_probability': self.drawdown_5pct_probability,
            'drawdown_10pct_probability': self.drawdown_10pct_probability,
            'drawdown_20pct_probability': self.drawdown_20pct_probability,
            'expected_drawdown_pct': self.expected_drawdown_pct,
            'var_95_drawdown_pct': self.var_95_drawdown_pct,
            'cvar_95_drawdown_pct': self.cvar_95_drawdown_pct,
            'expected_recovery_days': self.expected_recovery_days,
            'recovery_model': self.recovery_model.value,
            'confidence_level': self.confidence_level,
        }


@dataclass
class MonitorConfig:
    """监控配置"""
    update_interval_sec: float = 60.0       # 更新间隔(秒)
    enable_alerts: bool = True              # 启用告警

    # 告警阈值
    warning_threshold_pct: float = 5.0      # 预警阈值
    critical_threshold_pct: float = 10.0    # 临界阈值
    emergency_threshold_pct: float = 20.0   # 紧急阈值

    # 历史数据
    max_history_days: int = 365             # 最大历史天数

    # 恢复预测
    recovery_model: RecoveryModel = RecoveryModel.HISTORICAL
    prediction_horizon_days: int = 30       # 预测时间范围

    # 分解配置
    enable_decomposition: bool = True       # 启用分解分析

    # 统计配置
    min_drawdown_pct: float = 1.0          # 最小回撤阈值(统计用)


# ============== Drawdown Calculator ==============

class DrawdownCalculator:
    """
    回撤计算器

    核心回撤计算逻辑
    """

    @staticmethod
    def calculate_drawdown(current_value: float, peak_value: float) -> Tuple[float, float]:
        """
        计算回撤

        Args:
            current_value: 当前价值
            peak_value: 峰值

        Returns:
            (回撤百分比, 回撤金额)
        """
        if peak_value <= 0:
            return 0.0, 0.0

        drawdown_value = peak_value - current_value
        drawdown_pct = (drawdown_value / peak_value) * 100

        return max(0.0, drawdown_pct), max(0.0, drawdown_value)

    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]:
        """
        计算最大回撤

        Args:
            values: 价值序列

        Returns:
            (最大回撤百分比, 峰值索引, 谷底索引)
        """
        if not values or len(values) < 2:
            return 0.0, 0, 0

        max_dd = 0.0
        peak_idx = 0
        trough_idx = 0
        peak_value = values[0]

        for i, value in enumerate(values):
            if value > peak_value:
                peak_value = value
                peak_idx = i

            dd_pct = ((peak_value - value) / peak_value * 100) if peak_value > 0 else 0
            if dd_pct > max_dd:
                max_dd = dd_pct
                trough_idx = i

        return max_dd, peak_idx, trough_idx

    @staticmethod
    def calculate_underwater_time(values: List[float]) -> float:
        """
        计算水下时间百分比

        Args:
            values: 价值序列

        Returns:
            水下时间百分比(0-100)
        """
        if not values:
            return 0.0

        peak = values[0]
        underwater_count = 0

        for value in values:
            if value > peak:
                peak = value
            elif value < peak:
                underwater_count += 1

        return (underwater_count / len(values)) * 100 if len(values) > 0 else 0.0

    @staticmethod
    def determine_severity(drawdown_pct: float) -> DrawdownSeverity:
        """
        确定回撤严重程度

        Args:
            drawdown_pct: 回撤百分比

        Returns:
            严重程度
        """
        if drawdown_pct >= 30:
            return DrawdownSeverity.EXTREME
        elif drawdown_pct >= 20:
            return DrawdownSeverity.SEVERE
        elif drawdown_pct >= 10:
            return DrawdownSeverity.SIGNIFICANT
        elif drawdown_pct >= 5:
            return DrawdownSeverity.MODERATE
        else:
            return DrawdownSeverity.MINOR


# ============== Recovery Estimator ==============

class RecoveryEstimator:
    """
    恢复时间预估器

    预测从回撤中恢复所需时间
    """

    def __init__(self, model: RecoveryModel = RecoveryModel.HISTORICAL):
        self.model = model
        self._historical_recoveries: List[Tuple[float, float]] = []  # (回撤%, 恢复天数)

    def add_historical_recovery(self, drawdown_pct: float, recovery_days: float):
        """添加历史恢复数据"""
        self._historical_recoveries.append((drawdown_pct, recovery_days))

    def estimate_recovery_time(
        self,
        current_drawdown_pct: float,
        current_value: float,
        peak_value: float,
        recent_returns: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        估算恢复时间

        Args:
            current_drawdown_pct: 当前回撤百分比
            current_value: 当前价值
            peak_value: 峰值
            recent_returns: 最近的收益率序列

        Returns:
            (预估天数, 置信度)
        """
        if current_drawdown_pct <= 0:
            return 0.0, 1.0

        if self.model == RecoveryModel.LINEAR:
            return self._linear_estimate(current_drawdown_pct, recent_returns)
        elif self.model == RecoveryModel.EXPONENTIAL:
            return self._exponential_estimate(current_drawdown_pct, recent_returns)
        elif self.model == RecoveryModel.MEAN_REVERSION:
            return self._mean_reversion_estimate(current_drawdown_pct, recent_returns)
        else:  # HISTORICAL
            return self._historical_estimate(current_drawdown_pct)

    def _linear_estimate(
        self,
        drawdown_pct: float,
        recent_returns: Optional[List[float]],
    ) -> Tuple[float, float]:
        """线性恢复估算"""
        data_points = 0
        if not recent_returns or len(recent_returns) < 5:
            # 默认假设年化10%收益
            avg_daily_return = 0.10 / 252
        else:
            avg_daily_return = statistics.mean(recent_returns[-20:])  # 最近20天平均
            data_points = len(recent_returns)

        if avg_daily_return <= 0:
            return float('inf'), 0.0

        # 需要的收益百分比
        required_return = drawdown_pct / 100

        # 预估天数
        days = required_return / avg_daily_return
        confidence = min(0.7, 0.3 + data_points / 100)  # 数据越多越可信

        return max(0, days), confidence

    def _exponential_estimate(
        self,
        drawdown_pct: float,
        recent_returns: Optional[List[float]],
    ) -> Tuple[float, float]:
        """指数恢复估算(考虑复利效应)"""
        if not recent_returns or len(recent_returns) < 5:
            avg_daily_return = 0.10 / 252
        else:
            avg_daily_return = statistics.mean(recent_returns[-20:])

        if avg_daily_return <= 0:
            return float('inf'), 0.0

        # 需要的复合收益率
        required_return = drawdown_pct / 100

        # 使用对数计算
        try:
            days = math.log(1 + required_return) / math.log(1 + avg_daily_return)
            confidence = min(0.75, 0.4 + len(recent_returns) / 100)
            return max(0, days), confidence
        except (ValueError, ZeroDivisionError):
            return float('inf'), 0.0

    def _mean_reversion_estimate(
        self,
        drawdown_pct: float,
        recent_returns: Optional[List[float]],
    ) -> Tuple[float, float]:
        """均值回归估算"""
        if not recent_returns or len(recent_returns) < 30:
            # 默认半衰期60天
            half_life = 60
        else:
            # 计算自相关性估算半衰期
            returns = recent_returns[-60:]
            if len(returns) >= 30:
                # 简化：使用标准差的倒数作为回归速度
                volatility = statistics.stdev(returns)
                half_life = max(30, int(1 / (volatility * 10)))  # 经验公式
            else:
                half_life = 60

        # 指数衰减模型
        decay_rate = math.log(2) / half_life
        days = math.log(1 / 0.05) / decay_rate  # 恢复到5%以内
        confidence = 0.65

        return days, confidence

    def _historical_estimate(self, drawdown_pct: float) -> Tuple[float, float]:
        """历史平均估算"""
        if not self._historical_recoveries:
            # 使用经验公式: 恢复时间约为回撤幅度的3-5倍
            days = drawdown_pct * 4
            return days, 0.3

        # 找到相似的历史回撤
        similar = [
            (dd, days) for dd, days in self._historical_recoveries
            if abs(dd - drawdown_pct) < drawdown_pct * 0.3  # 30%容差
        ]

        if similar:
            avg_days = statistics.mean([days for _, days in similar])
            confidence = min(0.9, 0.5 + len(similar) / 20)
            return avg_days, confidence
        else:
            # 使用所有历史数据的回归
            avg_days = statistics.mean([days for _, days in self._historical_recoveries])
            ratio = avg_days / statistics.mean([dd for dd, _ in self._historical_recoveries])
            estimated_days = drawdown_pct * ratio
            return estimated_days, 0.6


# ============== Drawdown Predictor ==============

class DrawdownPredictor:
    """
    回撤预测器

    预测未来回撤风险
    """

    def __init__(self):
        self._returns_history: List[float] = []
        self._drawdown_history: List[float] = []

    def add_return(self, daily_return: float):
        """添加收益率"""
        self._returns_history.append(daily_return)
        if len(self._returns_history) > 252:  # 保留1年数据
            self._returns_history.pop(0)

    def add_drawdown(self, drawdown_pct: float):
        """添加回撤记录"""
        self._drawdown_history.append(drawdown_pct)
        if len(self._drawdown_history) > 252:
            self._drawdown_history.pop(0)

    def predict(self, horizon_days: int = 30) -> DrawdownPrediction:
        """
        预测未来回撤

        Args:
            horizon_days: 预测时间范围(天)

        Returns:
            回撤预测结果
        """
        now = datetime.now()

        # 如果历史数据不足,返回保守估计
        if len(self._returns_history) < 30:
            return DrawdownPrediction(
                timestamp=now,
                prediction_horizon_days=horizon_days,
                drawdown_5pct_probability=0.3,
                drawdown_10pct_probability=0.15,
                drawdown_20pct_probability=0.05,
                expected_drawdown_pct=5.0,
                var_95_drawdown_pct=10.0,
                cvar_95_drawdown_pct=15.0,
                expected_recovery_days=30.0,
                recovery_model=RecoveryModel.HISTORICAL,
                confidence_level=0.3,
            )

        # 计算收益率统计
        mean_return = statistics.mean(self._returns_history)
        std_return = statistics.stdev(self._returns_history)

        # 使用蒙特卡洛模拟
        simulations = 1000
        max_drawdowns = []

        import random
        for _ in range(simulations):
            # 模拟未来路径
            value = 100.0
            peak = 100.0
            max_dd = 0.0

            for _ in range(horizon_days):
                ret = random.gauss(mean_return, std_return)
                value *= (1 + ret)
                peak = max(peak, value)
                dd = (peak - value) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)

            max_drawdowns.append(max_dd)

        # 计算概率
        max_drawdowns.sort()
        prob_5 = sum(1 for dd in max_drawdowns if dd >= 5) / simulations
        prob_10 = sum(1 for dd in max_drawdowns if dd >= 10) / simulations
        prob_20 = sum(1 for dd in max_drawdowns if dd >= 20) / simulations

        # 计算VaR/CVaR
        var_idx = int(0.95 * simulations)
        var_95 = max_drawdowns[var_idx] if var_idx < len(max_drawdowns) else max_drawdowns[-1]

        cvar_tail = max_drawdowns[var_idx:]
        cvar_95 = statistics.mean(cvar_tail) if cvar_tail else var_95

        expected_dd = statistics.mean(max_drawdowns)

        # 预期恢复时间(基于历史)
        expected_recovery = expected_dd * 3  # 经验公式

        return DrawdownPrediction(
            timestamp=now,
            prediction_horizon_days=horizon_days,
            drawdown_5pct_probability=prob_5,
            drawdown_10pct_probability=prob_10,
            drawdown_20pct_probability=prob_20,
            expected_drawdown_pct=expected_dd,
            var_95_drawdown_pct=var_95,
            cvar_95_drawdown_pct=cvar_95,
            expected_recovery_days=expected_recovery,
            recovery_model=RecoveryModel.HISTORICAL,
            confidence_level=min(0.9, 0.5 + len(self._returns_history) / 500),
        )


# ============== Drawdown Monitor ==============

class DrawdownMonitor:
    """
    回撤监控器

    功能:
    - 实时回撤计算
    - 回撤持续时间追踪
    - 回撤恢复时间预估
    - 多时间框架回撤
    - 回撤分解分析
    - 回撤警报系统
    - 历史回撤统计
    - 回撤风险预测
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()

        # 核心数据
        self._value_history: deque = deque(maxlen=self.config.max_history_days * 24)  # (时间, 价值)
        self._peak_value: float = 0.0
        self._peak_time: Optional[datetime] = None
        self._trough_value: float = 0.0
        self._trough_time: Optional[datetime] = None

        # 当前回撤期
        self._current_drawdown: Optional[DrawdownPeriod] = None

        # 历史回撤期
        self._historical_drawdowns: List[DrawdownPeriod] = []

        # 资产/策略/因子数据
        self._asset_values: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._strategy_values: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._factor_values: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

        # 收益率历史(用于预测)
        self._returns_history: List[float] = []

        # 计算器和预测器
        self._calculator = DrawdownCalculator()
        self._estimator = RecoveryEstimator(self.config.recovery_model)
        self._predictor = DrawdownPredictor()

        # 告警
        self._alerts: List[DrawdownAlert] = []
        self._alert_callbacks: List[Callable[[DrawdownAlert], None]] = []
        self._alert_counter: int = 0

        # 最新指标
        self._latest_metrics: Optional[DrawdownMetrics] = None

        # 监控状态
        self._state = MonitorState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        logger.info("DrawdownMonitor initialized")

    # ============== Data Update ==============

    def update_value(self, value: float, timestamp: Optional[datetime] = None):
        """
        更新组合价值

        Args:
            value: 当前价值
            timestamp: 时间戳(可选)
        """
        timestamp = timestamp or datetime.now()

        # 添加到历史
        self._value_history.append((timestamp, value))

        # 更新峰值
        if value >= self._peak_value:
            # 如果正在回撤中且恢复到峰值,结束当前回撤期
            if self._current_drawdown and self._current_drawdown.state == DrawdownState.IN_DRAWDOWN:
                self._current_drawdown.state = DrawdownState.RECOVERED
                self._current_drawdown.end_time = timestamp
                self._current_drawdown.recovery_value = value

                # 计算恢复时间
                if self._current_drawdown.trough_time:
                    recovery_time = (timestamp - self._current_drawdown.trough_time).total_seconds() / 86400
                    self._current_drawdown.recovery_days = recovery_time

                # 归档
                self._historical_drawdowns.append(self._current_drawdown)
                self._estimator.add_historical_recovery(
                    self._current_drawdown.max_drawdown_pct,
                    self._current_drawdown.recovery_days or 0,
                )

                self._current_drawdown = None

            # 更新峰值（无论是否有回撤恢复）
            if value > self._peak_value:
                self._peak_value = value
                self._peak_time = timestamp

        # 检查是否进入回撤
        elif value < self._peak_value:
            dd_pct, dd_value = self._calculator.calculate_drawdown(value, self._peak_value)

            # 创建新回撤期或更新现有回撤期
            if self._current_drawdown is None:
                self._current_drawdown = DrawdownPeriod(
                    start_time=self._peak_time or timestamp,
                    peak_value=self._peak_value,
                    trough_time=timestamp,
                    trough_value=value,
                    max_drawdown_pct=dd_pct,
                    state=DrawdownState.IN_DRAWDOWN,
                    severity=self._calculator.determine_severity(dd_pct),
                )
            else:
                # 更新谷底
                if value < (self._current_drawdown.trough_value or float('inf')):
                    self._current_drawdown.trough_time = timestamp
                    self._current_drawdown.trough_value = value
                    self._current_drawdown.max_drawdown_pct = dd_pct
                    self._current_drawdown.severity = self._calculator.determine_severity(dd_pct)

            # 更新持续时间
            if self._current_drawdown.start_time:
                duration = (timestamp - self._current_drawdown.start_time).total_seconds() / 86400
                self._current_drawdown.duration_days = duration

            # 检查告警
            if self.config.enable_alerts:
                self._check_alert_thresholds(dd_pct, value, timestamp)

        # 计算最新指标
        self._update_metrics(value, timestamp)

    def add_return(self, daily_return: float):
        """
        添加日收益率

        Args:
            daily_return: 日收益率
        """
        self._returns_history.append(daily_return)
        self._predictor.add_return(daily_return)

        # 限制历史长度
        if len(self._returns_history) > self.config.max_history_days:
            self._returns_history.pop(0)

    def update_asset_value(self, asset_name: str, value: float, timestamp: Optional[datetime] = None):
        """
        更新资产价值(用于分解)

        Args:
            asset_name: 资产名称
            value: 价值
            timestamp: 时间戳
        """
        timestamp = timestamp or datetime.now()
        self._asset_values[asset_name].append((timestamp, value))

        # 限制长度
        if len(self._asset_values[asset_name]) > self.config.max_history_days * 24:
            self._asset_values[asset_name].pop(0)

    def update_strategy_value(self, strategy_name: str, value: float, timestamp: Optional[datetime] = None):
        """
        更新策略价值(用于分解)

        Args:
            strategy_name: 策略名称
            value: 价值
            timestamp: 时间戳
        """
        timestamp = timestamp or datetime.now()
        self._strategy_values[strategy_name].append((timestamp, value))

        if len(self._strategy_values[strategy_name]) > self.config.max_history_days * 24:
            self._strategy_values[strategy_name].pop(0)

    def update_factor_value(self, factor_name: str, value: float, timestamp: Optional[datetime] = None):
        """
        更新因子价值(用于分解)

        Args:
            factor_name: 因子名称
            value: 价值
            timestamp: 时间戳
        """
        timestamp = timestamp or datetime.now()
        self._factor_values[factor_name].append((timestamp, value))

        if len(self._factor_values[factor_name]) > self.config.max_history_days * 24:
            self._factor_values[factor_name].pop(0)

    # ============== Metrics Calculation ==============

    def _update_metrics(self, current_value: float, timestamp: datetime):
        """更新当前指标"""
        # 计算当前回撤
        current_dd_pct, current_dd_value = self._calculator.calculate_drawdown(
            current_value, self._peak_value
        )

        # 计算最大回撤
        if len(self._value_history) >= 2:
            values = [v for _, v in self._value_history]
            max_dd_pct, _, _ = self._calculator.calculate_max_drawdown(values)
            max_dd_value = self._peak_value * (max_dd_pct / 100)
        else:
            max_dd_pct = 0.0
            max_dd_value = 0.0

        # 计算时间指标
        days_in_drawdown = 0.0
        days_since_peak = 0.0
        if self._peak_time:
            days_since_peak = (timestamp - self._peak_time).total_seconds() / 86400
            if self._current_drawdown:
                days_in_drawdown = self._current_drawdown.duration_days

        # 确定状态
        if current_dd_pct <= 0:
            state = DrawdownState.NONE
        else:
            state = DrawdownState.IN_DRAWDOWN

        severity = self._calculator.determine_severity(current_dd_pct)

        # 预估恢复时间
        estimated_days = None
        estimated_date = None
        recovery_prob = 0.0

        if current_dd_pct > 0:
            est_days, confidence = self._estimator.estimate_recovery_time(
                current_dd_pct,
                current_value,
                self._peak_value,
                self._returns_history,
            )
            if est_days != float('inf'):
                estimated_days = est_days
                estimated_date = timestamp + timedelta(days=est_days)
                recovery_prob = confidence

        self._latest_metrics = DrawdownMetrics(
            timestamp=timestamp,
            current_value=current_value,
            peak_value=self._peak_value,
            current_drawdown_pct=current_dd_pct,
            current_drawdown_value=current_dd_value,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_value=max_dd_value,
            days_in_drawdown=days_in_drawdown,
            days_since_peak=days_since_peak,
            state=state,
            severity=severity,
            estimated_recovery_days=estimated_days,
            estimated_recovery_date=estimated_date,
            recovery_probability=recovery_prob,
        )

        # 更新预测器
        self._predictor.add_drawdown(current_dd_pct)

    def get_current_metrics(self) -> Optional[DrawdownMetrics]:
        """获取当前回撤指标"""
        return self._latest_metrics

    def calculate_timeframe_drawdowns(self) -> Dict[TimeFrame, TimeFrameDrawdown]:
        """
        计算多时间框架回撤

        Returns:
            各时间框架的回撤数据
        """
        results = {}
        now = datetime.now()

        # 定义时间框架
        timeframes = {
            TimeFrame.DAILY: timedelta(days=1),
            TimeFrame.WEEKLY: timedelta(days=7),
            TimeFrame.MONTHLY: timedelta(days=30),
            TimeFrame.QUARTERLY: timedelta(days=90),
            TimeFrame.YEARLY: timedelta(days=365),
        }

        for tf, delta in timeframes.items():
            period_start = now - delta

            # 筛选时间范围内的数据
            period_values = [
                (t, v) for t, v in self._value_history
                if t >= period_start
            ]

            if not period_values:
                continue

            values = [v for _, v in period_values]
            max_dd_pct, peak_idx, trough_idx = self._calculator.calculate_max_drawdown(values)

            peak_value = values[peak_idx] if peak_idx < len(values) else values[0]
            trough_value = values[trough_idx] if trough_idx < len(values) else values[-1]
            current_value = values[-1]

            # 计算当前回撤
            current_dd_pct, _ = self._calculator.calculate_drawdown(current_value, peak_value)

            # 计算恢复百分比
            if max_dd_pct > 0:
                recovery_pct = max(0, (1 - current_dd_pct / max_dd_pct) * 100)
            else:
                recovery_pct = 100.0

            results[tf] = TimeFrameDrawdown(
                timeframe=tf,
                period_start=period_start,
                period_end=now,
                peak_value=peak_value,
                trough_value=trough_value,
                max_drawdown_pct=max_dd_pct,
                current_drawdown_pct=current_dd_pct,
                recovery_pct=recovery_pct,
            )

        # 全时间框架
        if len(self._value_history) >= 2:
            all_values = [v for _, v in self._value_history]
            max_dd_pct, peak_idx, trough_idx = self._calculator.calculate_max_drawdown(all_values)

            peak_value = all_values[peak_idx]
            trough_value = all_values[trough_idx]
            current_value = all_values[-1]
            current_dd_pct, _ = self._calculator.calculate_drawdown(current_value, peak_value)

            recovery_pct = max(0, (1 - current_dd_pct / max_dd_pct) * 100) if max_dd_pct > 0 else 100.0

            results[TimeFrame.ALL_TIME] = TimeFrameDrawdown(
                timeframe=TimeFrame.ALL_TIME,
                period_start=self._value_history[0][0],
                period_end=now,
                peak_value=peak_value,
                trough_value=trough_value,
                max_drawdown_pct=max_dd_pct,
                current_drawdown_pct=current_dd_pct,
                recovery_pct=recovery_pct,
            )

        return results

    # ============== Decomposition ==============

    def calculate_decomposition(self) -> Optional[DrawdownDecomposition]:
        """
        计算回撤分解

        Returns:
            回撤分解结果
        """
        if not self.config.enable_decomposition:
            return None

        if not self._latest_metrics or self._latest_metrics.current_drawdown_pct <= 0:
            return None

        now = datetime.now()
        total_dd = self._latest_metrics.current_drawdown_pct

        decomp = DrawdownDecomposition(
            timestamp=now,
            total_drawdown_pct=total_dd,
        )

        # 资产分解
        for asset_name, values in self._asset_values.items():
            if not values:
                continue

            asset_values = [v for _, v in values]
            asset_peak = max(asset_values)
            asset_current = asset_values[-1]
            asset_dd_pct, _ = self._calculator.calculate_drawdown(asset_current, asset_peak)

            decomp.asset_drawdowns[asset_name] = asset_dd_pct
            # 贡献度(简化计算)
            if total_dd > 0:
                decomp.asset_contributions_pct[asset_name] = (asset_dd_pct / total_dd) * 100

        # 策略分解
        for strategy_name, values in self._strategy_values.items():
            if not values:
                continue

            strategy_values = [v for _, v in values]
            strategy_peak = max(strategy_values)
            strategy_current = strategy_values[-1]
            strategy_dd_pct, _ = self._calculator.calculate_drawdown(strategy_current, strategy_peak)

            decomp.strategy_drawdowns[strategy_name] = strategy_dd_pct
            if total_dd > 0:
                decomp.strategy_contributions_pct[strategy_name] = (strategy_dd_pct / total_dd) * 100

        # 因子分解
        for factor_name, values in self._factor_values.items():
            if not values:
                continue

            factor_values = [v for _, v in values]
            factor_peak = max(factor_values)
            factor_current = factor_values[-1]
            factor_dd_pct, _ = self._calculator.calculate_drawdown(factor_current, factor_peak)

            decomp.factor_drawdowns[factor_name] = factor_dd_pct
            if total_dd > 0:
                decomp.factor_contributions_pct[factor_name] = (factor_dd_pct / total_dd) * 100

        return decomp

    # ============== Statistics ==============

    def calculate_statistics(self) -> Optional[DrawdownStatistics]:
        """
        计算历史回撤统计

        Returns:
            回撤统计结果
        """
        # 筛选有效回撤(超过最小阈值)
        valid_drawdowns = [
            dd for dd in self._historical_drawdowns
            if dd.max_drawdown_pct >= self.config.min_drawdown_pct
        ]

        if not valid_drawdowns:
            return None

        # 回撤百分比列表
        dd_pcts = [dd.max_drawdown_pct for dd in valid_drawdowns]
        durations = [dd.duration_days for dd in valid_drawdowns]
        recoveries = [dd.recovery_days for dd in valid_drawdowns if dd.recovery_days is not None]

        # 严重程度分布
        severity_dist = defaultdict(int)
        for dd in valid_drawdowns:
            severity_dist[dd.severity.value] += 1

        # 计算频率
        if len(self._value_history) >= 2:
            first_time = self._value_history[0][0]
            last_time = self._value_history[-1][0]
            total_years = (last_time - first_time).total_seconds() / (365 * 86400)
            drawdowns_per_year = len(valid_drawdowns) / total_years if total_years > 0 else 0
        else:
            drawdowns_per_year = 0

        # 水下时间
        if len(self._value_history) >= 2:
            values = [v for _, v in self._value_history]
            underwater_pct = self._calculator.calculate_underwater_time(values)
        else:
            underwater_pct = 0.0

        return DrawdownStatistics(
            total_drawdowns=len(valid_drawdowns),
            avg_drawdown_pct=statistics.mean(dd_pcts),
            median_drawdown_pct=statistics.median(dd_pcts),
            max_drawdown_pct=max(dd_pcts),
            min_drawdown_pct=min(dd_pcts),
            std_drawdown_pct=statistics.stdev(dd_pcts) if len(dd_pcts) >= 2 else 0.0,
            avg_duration_days=statistics.mean(durations),
            max_duration_days=max(durations),
            median_duration_days=statistics.median(durations),
            avg_recovery_days=statistics.mean(recoveries) if recoveries else 0.0,
            max_recovery_days=max(recoveries) if recoveries else 0.0,
            median_recovery_days=statistics.median(recoveries) if recoveries else 0.0,
            severity_distribution=dict(severity_dist),
            drawdowns_per_year=drawdowns_per_year,
            avg_time_underwater_pct=underwater_pct,
        )

    # ============== Prediction ==============

    def predict_drawdown(self, horizon_days: Optional[int] = None) -> DrawdownPrediction:
        """
        预测未来回撤风险

        Args:
            horizon_days: 预测时间范围(天)

        Returns:
            回撤预测结果
        """
        horizon = horizon_days or self.config.prediction_horizon_days
        return self._predictor.predict(horizon)

    # ============== Alerts ==============

    def _check_alert_thresholds(self, drawdown_pct: float, current_value: float, timestamp: datetime):
        """检查告警阈值"""
        # 确定告警级别
        level = None
        threshold = 0.0

        if drawdown_pct >= self.config.emergency_threshold_pct:
            level = AlertLevel.EMERGENCY
            threshold = self.config.emergency_threshold_pct
        elif drawdown_pct >= self.config.critical_threshold_pct:
            level = AlertLevel.CRITICAL
            threshold = self.config.critical_threshold_pct
        elif drawdown_pct >= self.config.warning_threshold_pct:
            level = AlertLevel.WARNING
            threshold = self.config.warning_threshold_pct

        if level is None:
            return

        # 检查是否已经有相同级别的未确认告警
        existing = [
            a for a in self._alerts
            if a.level == level and not a.acknowledged and not a.resolved
        ]

        if existing:
            return  # 避免重复告警

        # 创建告警
        self._alert_counter += 1
        alert = DrawdownAlert(
            alert_id=f"DD_ALERT_{self._alert_counter}",
            timestamp=timestamp,
            level=level,
            message=f"Drawdown {level.value}: {drawdown_pct:.2f}% (threshold: {threshold:.2f}%)",
            drawdown_pct=drawdown_pct,
            threshold_pct=threshold,
            current_value=current_value,
            peak_value=self._peak_value,
        )

        self._alerts.append(alert)

        # 触发回调
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: Callable[[DrawdownAlert], None]):
        """添加告警回调"""
        self._alert_callbacks.append(callback)

    def get_alerts(
        self,
        unacknowledged_only: bool = False,
        level: Optional[AlertLevel] = None,
    ) -> List[DrawdownAlert]:
        """
        获取告警列表

        Args:
            unacknowledged_only: 仅未确认的
            level: 指定级别

        Returns:
            告警列表
        """
        alerts = self._alerts

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        确认告警

        Args:
            alert_id: 告警ID

        Returns:
            是否成功
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        解决告警

        Args:
            alert_id: 告警ID

        Returns:
            是否成功
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False

    # ============== Historical Data ==============

    def get_historical_drawdowns(
        self,
        min_drawdown_pct: Optional[float] = None,
        severity: Optional[DrawdownSeverity] = None,
    ) -> List[DrawdownPeriod]:
        """
        获取历史回撤记录

        Args:
            min_drawdown_pct: 最小回撤阈值
            severity: 严重程度筛选

        Returns:
            回撤记录列表
        """
        drawdowns = self._historical_drawdowns

        if min_drawdown_pct is not None:
            drawdowns = [dd for dd in drawdowns if dd.max_drawdown_pct >= min_drawdown_pct]

        if severity:
            drawdowns = [dd for dd in drawdowns if dd.severity == severity]

        return drawdowns

    def get_current_drawdown_period(self) -> Optional[DrawdownPeriod]:
        """获取当前回撤期"""
        return self._current_drawdown

    # ============== Monitoring ==============

    async def start(self):
        """启动监控"""
        if self._state == MonitorState.RUNNING:
            return

        self._state = MonitorState.RUNNING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Drawdown monitor started")

    async def stop(self):
        """停止监控"""
        self._state = MonitorState.IDLE
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Drawdown monitor stopped")

    async def _monitor_loop(self):
        """监控循环"""
        while self._state == MonitorState.RUNNING:
            try:
                # 这里可以定期更新指标、检查告警等
                # 实际的价值更新由外部调用 update_value()

                await asyncio.sleep(self.config.update_interval_sec)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self._state = MonitorState.ERROR
                await asyncio.sleep(5)

    @property
    def state(self) -> MonitorState:
        """获取监控状态"""
        return self._state

    # ============== Summary ==============

    def get_summary(self) -> Dict[str, Any]:
        """
        获取监控摘要

        Returns:
            包含所有关键信息的字典
        """
        summary = {
            'state': self._state.value,
            'current_metrics': self._latest_metrics.to_dict() if self._latest_metrics else None,
            'current_drawdown_period': self._current_drawdown.to_dict() if self._current_drawdown else None,
            'historical_drawdowns_count': len(self._historical_drawdowns),
            'unacknowledged_alerts': len([a for a in self._alerts if not a.acknowledged]),
            'value_history_length': len(self._value_history),
            'peak_value': self._peak_value,
            'peak_time': self._peak_time.isoformat() if self._peak_time else None,
        }

        # 时间框架回撤
        tf_drawdowns = self.calculate_timeframe_drawdowns()
        summary['timeframe_drawdowns'] = {
            tf.value: dd.to_dict() for tf, dd in tf_drawdowns.items()
        }

        # 统计
        stats = self.calculate_statistics()
        if stats:
            summary['statistics'] = stats.to_dict()

        # 分解
        decomp = self.calculate_decomposition()
        if decomp:
            summary['decomposition'] = decomp.to_dict()

        # 预测
        prediction = self.predict_drawdown()
        summary['prediction'] = prediction.to_dict()

        return summary


# ============== Factory Functions ==============

def create_drawdown_monitor(
    update_interval_sec: float = 60.0,
    enable_alerts: bool = True,
    warning_threshold_pct: float = 5.0,
    critical_threshold_pct: float = 10.0,
    emergency_threshold_pct: float = 20.0,
    **kwargs
) -> DrawdownMonitor:
    """
    创建回撤监控器

    Args:
        update_interval_sec: 更新间隔(秒)
        enable_alerts: 启用告警
        warning_threshold_pct: 预警阈值
        critical_threshold_pct: 临界阈值
        emergency_threshold_pct: 紧急阈值
        **kwargs: 其他配置参数
    """
    config = MonitorConfig(
        update_interval_sec=update_interval_sec,
        enable_alerts=enable_alerts,
        warning_threshold_pct=warning_threshold_pct,
        critical_threshold_pct=critical_threshold_pct,
        emergency_threshold_pct=emergency_threshold_pct,
        **{k: v for k, v in kwargs.items() if hasattr(MonitorConfig, k)}
    )

    return DrawdownMonitor(config)


async def quick_drawdown_analysis(
    values: List[float],
    timestamps: Optional[List[datetime]] = None,
) -> Dict[str, Any]:
    """
    快速回撤分析

    Args:
        values: 价值序列
        timestamps: 时间戳序列(可选)

    Returns:
        分析结果
    """
    if not values:
        return {}

    # 生成时间戳(如果没有提供)
    if timestamps is None or len(timestamps) != len(values):
        now = datetime.now()
        timestamps = [now - timedelta(days=len(values)-i-1) for i in range(len(values))]

    monitor = create_drawdown_monitor()

    # 添加数据
    for ts, value in zip(timestamps, values):
        monitor.update_value(value, ts)

        # 计算收益率
        if len(monitor._value_history) >= 2:
            prev_value = list(monitor._value_history)[-2][1]
            if prev_value > 0:
                ret = (value - prev_value) / prev_value
                monitor.add_return(ret)

    # 返回摘要
    return monitor.get_summary()


__all__ = [
    # Enums
    'TimeFrame',
    'DrawdownState',
    'DrawdownSeverity',
    'AlertLevel',
    'MonitorState',
    'RecoveryModel',
    # Data Classes
    'DrawdownPeriod',
    'DrawdownMetrics',
    'TimeFrameDrawdown',
    'DrawdownAlert',
    'DrawdownDecomposition',
    'DrawdownStatistics',
    'DrawdownPrediction',
    'MonitorConfig',
    # Calculators & Predictors
    'DrawdownCalculator',
    'RecoveryEstimator',
    'DrawdownPredictor',
    # Monitor
    'DrawdownMonitor',
    # Factory Functions
    'create_drawdown_monitor',
    'quick_drawdown_analysis',
]
