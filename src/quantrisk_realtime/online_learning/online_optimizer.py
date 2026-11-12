"""
在线权重优化器

使用EMA和正则化在线调整信号权重。

特性:
- 指数移动平均 (EMA) 权重更新
- L1/L2正则化
- 学习率衰减
- 权重约束 (非负、归一化)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import math
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """优化器配置"""
    # 信号源列表
    sources: List[str] = field(default_factory=lambda: [
        "technical", "fundamental", "sentiment"
    ])

    # 初始权重 (默认等权)
    initial_weights: Optional[Dict[str, float]] = None

    # 学习率
    learning_rate: float = 0.01
    learning_rate_decay: float = 0.995  # 每次更新后衰减

    # EMA参数
    ema_alpha: float = 0.1  # EMA系数

    # 正则化
    l1_lambda: float = 0.001  # L1正则化系数
    l2_lambda: float = 0.001  # L2正则化系数

    # 权重约束
    min_weight: float = 0.05   # 最小权重
    max_weight: float = 0.8    # 最大权重

    # 更新阈值
    min_update_interval: int = 60  # 最小更新间隔 (秒)
    min_samples_for_update: int = 10  # 最小样本数


@dataclass
class WeightUpdate:
    """权重更新记录"""
    timestamp: datetime
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    performance_scores: Dict[str, float]
    learning_rate: float
    update_reason: str


class OnlineWeightOptimizer:
    """在线权重优化器"""

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.sources = self.config.sources

        # 初始化权重
        if self.config.initial_weights:
            self._weights = dict(self.config.initial_weights)
        else:
            # 等权初始化
            equal_weight = 1.0 / len(self.sources)
            self._weights = {s: equal_weight for s in self.sources}

        # 学习率 (可衰减)
        self._current_lr = self.config.learning_rate

        # 更新历史
        self._update_history: List[WeightUpdate] = []

        # 最后更新时间
        self._last_update: Optional[datetime] = None

        # 累积梯度 (用于EMA)
        self._gradient_ema: Dict[str, float] = {s: 0.0 for s in self.sources}

        # 更新计数
        self._update_count = 0

    @property
    def weights(self) -> Dict[str, float]:
        """当前权重"""
        return dict(self._weights)

    @property
    def learning_rate(self) -> float:
        """当前学习率"""
        return self._current_lr

    def update_weights(
        self,
        performance: Dict[str, float],
        force: bool = False
    ) -> Dict[str, float]:
        """
        更新权重

        Args:
            performance: 各信号源的表现得分 (如Sharpe比率)
            force: 是否强制更新 (忽略间隔检查)

        Returns:
            新权重
        """
        now = datetime.now()

        # 检查更新间隔
        if not force and self._last_update:
            elapsed = (now - self._last_update).total_seconds()
            if elapsed < self.config.min_update_interval:
                return self._weights

        old_weights = dict(self._weights)

        # 计算表现梯度
        gradient = self._compute_performance_gradient(performance)

        # 添加正则化梯度
        reg_gradient = self._compute_regularization_gradient()

        # 合并梯度
        total_gradient = {}
        for source in self.sources:
            total_gradient[source] = gradient.get(source, 0) - reg_gradient.get(source, 0)

        # 更新EMA梯度
        for source in self.sources:
            self._gradient_ema[source] = (
                (1 - self.config.ema_alpha) * self._gradient_ema[source] +
                self.config.ema_alpha * total_gradient[source]
            )

        # 应用梯度更新 (EMA平滑)
        new_weights = {}
        for source in self.sources:
            old_weight = self._weights[source]
            updated = old_weight + self._current_lr * self._gradient_ema[source]

            # EMA更新
            new_weights[source] = (
                (1 - self.config.ema_alpha) * old_weight +
                self.config.ema_alpha * updated
            )

        # 应用约束
        new_weights = self._apply_constraints(new_weights)

        # 更新状态
        self._weights = new_weights
        self._last_update = now
        self._update_count += 1

        # 衰减学习率
        self._current_lr *= self.config.learning_rate_decay

        # 记录更新
        update = WeightUpdate(
            timestamp=now,
            old_weights=old_weights,
            new_weights=new_weights,
            performance_scores=performance,
            learning_rate=self._current_lr,
            update_reason="performance_update"
        )
        self._update_history.append(update)

        # 限制历史长度
        if len(self._update_history) > 1000:
            self._update_history = self._update_history[-1000:]

        logger.debug(f"Weights updated: {old_weights} -> {new_weights}")

        return new_weights

    def reset(self) -> None:
        """重置优化器"""
        equal_weight = 1.0 / len(self.sources)
        self._weights = {s: equal_weight for s in self.sources}
        self._current_lr = self.config.learning_rate
        self._gradient_ema = {s: 0.0 for s in self.sources}
        self._update_count = 0
        self._last_update = None
        logger.info("Optimizer reset")

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        手动设置权重

        Args:
            weights: 新权重
        """
        # 验证所有源都有权重
        for source in self.sources:
            if source not in weights:
                raise ValueError(f"Missing weight for source: {source}")

        # 应用约束并设置
        self._weights = self._apply_constraints(weights)
        logger.info(f"Weights manually set to: {self._weights}")

    def get_update_history(
        self,
        limit: int = 100
    ) -> List[WeightUpdate]:
        """获取更新历史"""
        return self._update_history[-limit:]

    def get_stats(self) -> Dict:
        """获取优化器统计"""
        return {
            "current_weights": self._weights,
            "current_learning_rate": self._current_lr,
            "update_count": self._update_count,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "gradient_ema": self._gradient_ema,
            "sources": self.sources
        }

    def _compute_performance_gradient(
        self,
        performance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算表现梯度

        基于相对表现计算梯度:
        - 表现高于平均的信号源增加权重
        - 表现低于平均的信号源减少权重
        """
        if not performance:
            return {s: 0.0 for s in self.sources}

        # 计算平均表现
        perf_values = [performance.get(s, 0) for s in self.sources]
        avg_perf = np.mean(perf_values) if perf_values else 0

        # 计算标准差
        std_perf = np.std(perf_values) if len(perf_values) > 1 else 1

        gradient = {}
        for source in self.sources:
            perf = performance.get(source, 0)
            # 归一化表现差异作为梯度
            gradient[source] = (perf - avg_perf) / (std_perf + 1e-8)

        return gradient

    def _compute_regularization_gradient(self) -> Dict[str, float]:
        """
        计算正则化梯度

        L1: λ1 * sign(w)
        L2: λ2 * w
        """
        gradient = {}

        for source in self.sources:
            w = self._weights[source]

            # L1梯度
            l1_grad = self.config.l1_lambda * np.sign(w)

            # L2梯度
            l2_grad = self.config.l2_lambda * w

            gradient[source] = l1_grad + l2_grad

        return gradient

    def _apply_constraints(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        应用权重约束

        1. 非负约束
        2. 最小/最大权重约束
        3. 归一化 (和为1)
        """
        constrained = {}

        # 应用边界约束
        for source in self.sources:
            w = weights.get(source, 0)
            w = max(self.config.min_weight, w)
            w = min(self.config.max_weight, w)
            constrained[source] = w

        # 归一化
        total = sum(constrained.values())
        if total > 0:
            for source in self.sources:
                constrained[source] /= total
        else:
            # 如果总和为0，使用等权
            equal_weight = 1.0 / len(self.sources)
            constrained = {s: equal_weight for s in self.sources}

        return constrained


# 单例实例
_optimizer_instance: Optional[OnlineWeightOptimizer] = None


def get_weight_optimizer() -> OnlineWeightOptimizer:
    """获取权重优化器单例"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = OnlineWeightOptimizer()
    return _optimizer_instance
