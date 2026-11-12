"""
表现追踪器

追踪各信号源的实时表现，用于在线权重优化。
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


class PerformanceWindow(Enum):
    """表现计算窗口"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class SignalRecord:
    """信号记录"""
    timestamp: datetime
    source: str
    signal_value: float     # 信号强度 (-1 to 1)
    predicted_direction: int  # 预测方向 (-1, 0, 1)
    actual_return: Optional[float] = None  # 实际收益
    is_correct: Optional[bool] = None  # 方向是否正确


@dataclass
class SignalPerformance:
    """信号表现"""
    source: str
    window: PerformanceWindow
    accuracy: float         # 方向准确率
    hit_rate: float         # 胜率
    avg_return: float       # 平均收益
    sharpe_ratio: float     # 夏普比率
    ic: float               # 信息系数
    win_loss_ratio: float   # 盈亏比
    max_drawdown: float     # 最大回撤
    sample_count: int       # 样本数
    computed_at: datetime = field(default_factory=datetime.now)


class PerformanceTracker:
    """表现追踪器"""

    def __init__(
        self,
        sources: List[str],
        max_records: int = 10000
    ):
        self.sources = sources
        self.max_records = max_records

        # 信号记录 (按源分组)
        self._records: Dict[str, Deque[SignalRecord]] = {
            s: deque(maxlen=max_records) for s in sources
        }

        # 缓存的表现指标
        self._performance_cache: Dict[str, Dict[PerformanceWindow, SignalPerformance]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

    def record_signal(
        self,
        source: str,
        signal_value: float,
        predicted_direction: int,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        记录信号

        Args:
            source: 信号源
            signal_value: 信号值 (-1 to 1)
            predicted_direction: 预测方向
            timestamp: 时间戳
        """
        if source not in self.sources:
            return

        record = SignalRecord(
            timestamp=timestamp or datetime.now(),
            source=source,
            signal_value=signal_value,
            predicted_direction=predicted_direction
        )

        self._records[source].append(record)
        self._invalidate_cache()

    def update_outcome(
        self,
        source: str,
        timestamp: datetime,
        actual_return: float
    ) -> None:
        """
        更新结果

        Args:
            source: 信号源
            timestamp: 信号时间戳
            actual_return: 实际收益
        """
        if source not in self.sources:
            return

        # 查找匹配的记录
        for record in self._records[source]:
            if record.timestamp == timestamp:
                record.actual_return = actual_return
                record.is_correct = (
                    (record.predicted_direction > 0 and actual_return > 0) or
                    (record.predicted_direction < 0 and actual_return < 0) or
                    (record.predicted_direction == 0 and abs(actual_return) < 0.001)
                )
                break

        self._invalidate_cache()

    def get_performance(
        self,
        source: str,
        window: PerformanceWindow = PerformanceWindow.DAY
    ) -> SignalPerformance:
        """
        获取信号表现

        Args:
            source: 信号源
            window: 时间窗口

        Returns:
            SignalPerformance: 表现指标
        """
        if source not in self.sources:
            raise ValueError(f"Unknown source: {source}")

        # 检查缓存
        if self._is_cache_valid():
            cached = self._performance_cache.get(source, {}).get(window)
            if cached:
                return cached

        # 计算表现
        performance = self._compute_performance(source, window)

        # 更新缓存
        if source not in self._performance_cache:
            self._performance_cache[source] = {}
        self._performance_cache[source][window] = performance
        self._cache_timestamp = datetime.now()

        return performance

    def get_all_performances(
        self,
        window: PerformanceWindow = PerformanceWindow.DAY
    ) -> Dict[str, SignalPerformance]:
        """获取所有信号源的表现"""
        return {
            source: self.get_performance(source, window)
            for source in self.sources
        }

    def get_sharpe_scores(
        self,
        window: PerformanceWindow = PerformanceWindow.DAY
    ) -> Dict[str, float]:
        """
        获取所有信号源的Sharpe比率

        用于权重优化器
        """
        performances = self.get_all_performances(window)
        return {
            source: perf.sharpe_ratio
            for source, perf in performances.items()
        }

    def get_ranking(
        self,
        window: PerformanceWindow = PerformanceWindow.DAY,
        metric: str = "sharpe_ratio"
    ) -> List[Tuple[str, float]]:
        """获取排名"""
        performances = self.get_all_performances(window)

        ranking = []
        for source, perf in performances.items():
            value = getattr(perf, metric, 0)
            ranking.append((source, value))

        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """获取追踪器统计"""
        return {
            "sources": self.sources,
            "record_counts": {
                source: len(records)
                for source, records in self._records.items()
            },
            "cache_valid": self._is_cache_valid(),
            "max_records": self.max_records
        }

    def _compute_performance(
        self,
        source: str,
        window: PerformanceWindow
    ) -> SignalPerformance:
        """计算表现指标"""
        records = self._get_records_in_window(source, window)

        # 过滤有结果的记录
        completed = [r for r in records if r.actual_return is not None]

        if not completed:
            return SignalPerformance(
                source=source,
                window=window,
                accuracy=0,
                hit_rate=0,
                avg_return=0,
                sharpe_ratio=0,
                ic=0,
                win_loss_ratio=0,
                max_drawdown=0,
                sample_count=0
            )

        # 计算指标
        returns = [r.actual_return for r in completed]
        signals = [r.signal_value for r in completed]
        correct_count = sum(1 for r in completed if r.is_correct)

        # 准确率
        accuracy = correct_count / len(completed)

        # 胜率
        wins = [r for r in returns if r > 0]
        hit_rate = len(wins) / len(returns) if returns else 0

        # 平均收益
        avg_return = np.mean(returns)

        # 夏普比率 (假设日度数据)
        std_return = np.std(returns) if len(returns) > 1 else 0.001
        sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0

        # IC (信号与收益的相关性)
        if len(signals) > 1 and len(returns) > 1:
            ic = np.corrcoef(signals, returns)[0, 1]
            if np.isnan(ic):
                ic = 0
        else:
            ic = 0

        # 盈亏比
        losses = [abs(r) for r in returns if r < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0.001
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(returns)

        return SignalPerformance(
            source=source,
            window=window,
            accuracy=accuracy,
            hit_rate=hit_rate,
            avg_return=avg_return,
            sharpe_ratio=sharpe_ratio,
            ic=ic,
            win_loss_ratio=win_loss_ratio,
            max_drawdown=max_drawdown,
            sample_count=len(completed)
        )

    def _get_records_in_window(
        self,
        source: str,
        window: PerformanceWindow
    ) -> List[SignalRecord]:
        """获取窗口内的记录"""
        now = datetime.now()

        if window == PerformanceWindow.HOUR:
            cutoff = now - timedelta(hours=1)
        elif window == PerformanceWindow.DAY:
            cutoff = now - timedelta(days=1)
        elif window == PerformanceWindow.WEEK:
            cutoff = now - timedelta(weeks=1)
        elif window == PerformanceWindow.MONTH:
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(days=1)

        return [
            r for r in self._records[source]
            if r.timestamp >= cutoff
        ]

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤"""
        if not returns:
            return 0

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        if not self._cache_timestamp:
            return False
        return (datetime.now() - self._cache_timestamp) < self._cache_ttl

    def _invalidate_cache(self) -> None:
        """使缓存失效"""
        self._cache_timestamp = None


# 单例实例
_tracker_instance: Optional[PerformanceTracker] = None


def get_performance_tracker(
    sources: Optional[List[str]] = None
) -> PerformanceTracker:
    """获取表现追踪器单例"""
    global _tracker_instance
    if _tracker_instance is None:
        if sources is None:
            sources = ["technical", "fundamental", "sentiment"]
        _tracker_instance = PerformanceTracker(sources)
    return _tracker_instance
