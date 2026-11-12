"""
在线权重学习模块

基于实时表现自适应调整信号权重。

核心组件:
- OnlineWeightOptimizer: 在线权重优化器
- PerformanceTracker: 表现追踪器
"""

from .online_optimizer import (
    OnlineWeightOptimizer,
    OptimizerConfig,
    WeightUpdate,
)

from .performance_tracker import (
    PerformanceTracker,
    SignalPerformance,
    PerformanceWindow,
)

__all__ = [
    'OnlineWeightOptimizer',
    'OptimizerConfig',
    'WeightUpdate',
    'PerformanceTracker',
    'SignalPerformance',
    'PerformanceWindow',
]
