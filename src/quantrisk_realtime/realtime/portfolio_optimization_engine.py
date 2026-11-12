"""
组合优化引擎 (Portfolio Optimization Engine)

提供机构级投资组合优化功能:
- Mean-Variance Optimization 均值方差优化
- Black-Litterman Model Black-Litterman模型
- Robust Optimization 稳健优化
- CVaR Optimization CVaR优化
- Max Sharpe Ratio 最大夏普比率
- Min Volatility 最小波动率
- Efficient Frontier 有效前沿计算
- Risk Decomposition 风险分解
- Marginal Contribution 边际贡献分析
- What-if Analysis 情景分析

这是现代投资组合理论在量化交易中的核心实现。

作者: TraderX 量化团队
Copyright (c) 2025 TraderX. All rights reserved.
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

logger = logging.getLogger(__name__)


# ============ 枚举定义 ============

class OptimizationObjective(str, Enum):
    """优化目标"""
    MEAN_VARIANCE = "mean_variance"                     # 均值方差优化
    MAX_SHARPE = "max_sharpe"                           # 最大夏普比率
    MIN_VOLATILITY = "min_volatility"                   # 最小波动率
    MAX_RETURN = "max_return"                           # 最大收益
    MIN_CVAR = "min_cvar"                               # 最小CVaR
    RISK_PARITY = "risk_parity"                         # 风险平价
    BLACK_LITTERMAN = "black_litterman"                 # Black-Litterman
    ROBUST = "robust"                                   # 稳健优化
    MAX_DIVERSIFICATION = "max_diversification"         # 最大分散化


class OptimizationMethod(str, Enum):
    """优化方法枚举"""
    QUADRATIC_PROGRAMMING = "quadratic_programming"     # 二次规划
    GRADIENT_DESCENT = "gradient_descent"               # 梯度下降
    INTERIOR_POINT = "interior_point"                   # 内点法
    SEQUENTIAL_QUADRATIC = "sequential_quadratic"       # 序列二次规划
    GENETIC_ALGORITHM = "genetic_algorithm"             # 遗传算法
    PARTICLE_SWARM = "particle_swarm"                   # 粒子群优化


class ConstraintType(str, Enum):
    """约束类型"""
    WEIGHT_BOUNDS = "weight_bounds"                     # 权重界限
    SECTOR_LIMIT = "sector_limit"                       # 行业限制
    TURNOVER = "turnover"                               # 换手率
    LEVERAGE = "leverage"                               # 杠杆
    LONG_ONLY = "long_only"                             # 仅做多
    LONG_SHORT = "long_short"                           # 多空
    TRACKING_ERROR = "tracking_error"                   # 跟踪误差
    CONCENTRATION = "concentration"                     # 集中度


class RiskMetric(str, Enum):
    """风险度量"""
    VARIANCE = "variance"                               # 方差
    VOLATILITY = "volatility"                           # 波动率
    CVAR = "cvar"                                       # 条件风险价值
    VAR = "var"                                         # 风险价值
    SEMI_VARIANCE = "semi_variance"                     # 半方差
    MAX_DRAWDOWN = "max_drawdown"                       # 最大回撤
    TRACKING_ERROR = "tracking_error"                   # 跟踪误差


class EngineState(str, Enum):
    """引擎状态"""
    IDLE = "idle"
    OPTIMIZING = "optimizing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ 数据类 ============

@dataclass
class AssetData:
    """资产数据"""
    symbol: str

    # 收益数据
    returns: List[float] = field(default_factory=list)
    expected_return: float = 0.0                        # 期望收益率
    mean_return: float = 0.0                            # 平均收益率

    # 风险数据
    volatility: float = 0.0                             # 波动率
    beta: float = 1.0                                   # Beta系数

    # 价格数据
    prices: List[float] = field(default_factory=list)
    current_price: float = 0.0

    # 分类信息
    sector: Optional[str] = None                        # 行业
    asset_class: Optional[str] = None                   # 资产类别

    # 时间戳
    timestamps: List[datetime] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "expected_return": self.expected_return,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "beta": self.beta,
            "current_price": self.current_price,
            "sector": self.sector,
            "asset_class": self.asset_class,
            "data_points": len(self.returns),
        }


@dataclass
class PortfolioConstraint:
    """组合约束"""
    constraint_type: ConstraintType

    # 权重约束
    min_weight: Optional[float] = None                  # 最小权重
    max_weight: Optional[float] = None                  # 最大权重

    # 行业/板块约束
    sector: Optional[str] = None                        # 目标行业
    sector_min: Optional[float] = None                  # 行业最小权重
    sector_max: Optional[float] = None                  # 行业最大权重

    # 换手率约束
    max_turnover: Optional[float] = None                # 最大换手率

    # 杠杆约束
    min_leverage: Optional[float] = None                # 最小杠杆
    max_leverage: Optional[float] = None                # 最大杠杆

    # 其他约束
    value: Optional[float] = None                       # 通用数值

    # 约束属性
    is_hard_constraint: bool = True                     # 硬约束 vs 软约束
    description: str = ""

    def validate(self, weights: Dict[str, float], asset_data: Dict[str, AssetData]) -> Tuple[bool, str]:
        """验证约束是否满足"""
        if self.constraint_type == ConstraintType.WEIGHT_BOUNDS:
            for symbol, weight in weights.items():
                if self.min_weight is not None and weight < self.min_weight:
                    return False, f"{symbol} weight {weight:.4f} below min {self.min_weight}"
                if self.max_weight is not None and weight > self.max_weight:
                    return False, f"{symbol} weight {weight:.4f} above max {self.max_weight}"

        elif self.constraint_type == ConstraintType.SECTOR_LIMIT and self.sector:
            sector_weight = sum(
                weight for symbol, weight in weights.items()
                if asset_data.get(symbol) and asset_data[symbol].sector == self.sector
            )
            if self.sector_min is not None and sector_weight < self.sector_min:
                return False, f"Sector {self.sector} weight {sector_weight:.4f} below min"
            if self.sector_max is not None and sector_weight > self.sector_max:
                return False, f"Sector {self.sector} weight {sector_weight:.4f} above max"

        elif self.constraint_type == ConstraintType.LEVERAGE:
            total_leverage = sum(abs(w) for w in weights.values())
            if self.min_leverage is not None and total_leverage < self.min_leverage:
                return False, f"Leverage {total_leverage:.4f} below min"
            if self.max_leverage is not None and total_leverage > self.max_leverage:
                return False, f"Leverage {total_leverage:.4f} above max"

        return True, "Constraint satisfied"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type.value,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "sector": self.sector,
            "max_turnover": self.max_turnover,
            "is_hard_constraint": self.is_hard_constraint,
            "description": self.description,
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    optimization_id: str
    objective: OptimizationObjective

    # 权重配置
    weights: Dict[str, float]                           # 最优权重

    # 组合指标
    expected_return: float                              # 期望收益率
    expected_volatility: float                          # 期望波动率
    sharpe_ratio: float                                 # 夏普比率

    # 风险指标
    portfolio_var: float = 0.0                          # 组合VaR
    portfolio_cvar: float = 0.0                         # 组合CVaR
    max_drawdown: float = 0.0                           # 最大回撤

    # 分散化指标
    diversification_ratio: float = 0.0                  # 分散化比率
    effective_assets: float = 0.0                       # 有效资产数
    concentration_index: float = 0.0                    # 集中度指数(HHI)

    # 优化诊断
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    optimization_time: float = 0.0                      # 优化耗时(秒)
    iterations: int = 0                                 # 迭代次数
    success: bool = True
    message: str = ""

    # 约束状态
    constraints_satisfied: bool = True
    constraint_violations: List[str] = field(default_factory=list)

    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimization_id": self.optimization_id,
            "objective": self.objective.value,
            "weights": self.weights,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "portfolio_var": self.portfolio_var,
            "portfolio_cvar": self.portfolio_cvar,
            "diversification_ratio": self.diversification_ratio,
            "effective_assets": self.effective_assets,
            "concentration_index": self.concentration_index,
            "optimization_time": self.optimization_time,
            "iterations": self.iterations,
            "success": self.success,
            "message": self.message,
            "constraints_satisfied": self.constraints_satisfied,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EfficientFrontier:
    """有效前沿"""
    frontier_id: str

    # 前沿点
    returns: List[float]                                # 收益率序列
    volatilities: List[float]                           # 波动率序列
    sharpe_ratios: List[float]                          # 夏普比率序列
    weights_series: List[Dict[str, float]]              # 权重序列

    # 特殊点
    min_volatility_weights: Dict[str, float]            # 最小波动率组合
    max_sharpe_weights: Dict[str, float]                # 最大夏普组合

    # 统计
    num_points: int = 0
    risk_free_rate: float = 0.0

    # 时间戳
    calculated_at: datetime = field(default_factory=datetime.now)

    def get_portfolio_at_return(self, target_return: float) -> Optional[Dict[str, float]]:
        """获取目标收益率对应的组合权重"""
        if not self.returns or not self.weights_series:
            return None

        # 找到最接近目标收益的点
        min_diff = float('inf')
        best_idx = 0

        for i, ret in enumerate(self.returns):
            diff = abs(ret - target_return)
            if diff < min_diff:
                min_diff = diff
                best_idx = i

        return self.weights_series[best_idx]

    def get_portfolio_at_risk(self, target_volatility: float) -> Optional[Dict[str, float]]:
        """获取目标风险对应的组合权重"""
        if not self.volatilities or not self.weights_series:
            return None

        # 找到最接近目标波动率的点
        min_diff = float('inf')
        best_idx = 0

        for i, vol in enumerate(self.volatilities):
            diff = abs(vol - target_volatility)
            if diff < min_diff:
                min_diff = diff
                best_idx = i

        return self.weights_series[best_idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "returns": self.returns,
            "volatilities": self.volatilities,
            "sharpe_ratios": self.sharpe_ratios,
            "num_points": self.num_points,
            "min_volatility_weights": self.min_volatility_weights,
            "max_sharpe_weights": self.max_sharpe_weights,
            "risk_free_rate": self.risk_free_rate,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class RiskDecomposition:
    """风险分解"""
    symbol: str

    # 风险贡献
    marginal_risk: float                                # 边际风险
    component_risk: float                               # 成分风险
    pct_contribution: float                             # 百分比贡献

    # 分解指标
    systematic_risk: float = 0.0                        # 系统性风险
    specific_risk: float = 0.0                          # 特定风险

    # 相关性贡献
    correlation_contribution: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "marginal_risk": self.marginal_risk,
            "component_risk": self.component_risk,
            "pct_contribution": self.pct_contribution,
            "systematic_risk": self.systematic_risk,
            "specific_risk": self.specific_risk,
            "correlation_contribution": self.correlation_contribution,
        }


@dataclass
class WhatIfScenario:
    """情景分析"""
    scenario_id: str
    scenario_name: str

    # 情景假设
    asset_return_changes: Dict[str, float] = field(default_factory=dict)  # 收益率变化
    volatility_multipliers: Dict[str, float] = field(default_factory=dict)  # 波动率倍数
    correlation_shifts: Dict[Tuple[str, str], float] = field(default_factory=dict)  # 相关性变化

    # 情景结果
    scenario_return: float = 0.0                        # 情景收益率
    scenario_volatility: float = 0.0                    # 情景波动率
    scenario_var: float = 0.0                           # 情景VaR
    scenario_cvar: float = 0.0                          # 情景CVaR

    # 与基准对比
    return_change: float = 0.0                          # 收益变化
    volatility_change: float = 0.0                      # 波动率变化

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "scenario_return": self.scenario_return,
            "scenario_volatility": self.scenario_volatility,
            "scenario_var": self.scenario_var,
            "return_change": self.return_change,
            "volatility_change": self.volatility_change,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OptimizerConfig:
    """优化器配置"""
    # 优化参数
    objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE
    risk_free_rate: float = 0.02                        # 无风险利率
    target_return: Optional[float] = None               # 目标收益率
    target_volatility: Optional[float] = None           # 目标波动率

    # 约束参数
    min_weight: float = 0.0                             # 全局最小权重
    max_weight: float = 1.0                             # 全局最大权重
    allow_short: bool = False                           # 允许做空
    max_leverage: float = 1.0                           # 最大杠杆
    max_turnover: Optional[float] = None                # 最大换手率

    # 优化算法参数
    max_iterations: int = 1000                          # 最大迭代次数
    tolerance: float = 1e-6                             # 收敛容差
    learning_rate: float = 0.01                         # 学习率(梯度下降)

    # CVaR参数
    cvar_confidence: float = 0.95                       # CVaR置信水平

    # Black-Litterman参数
    bl_tau: float = 0.05                                # 不确定性参数
    bl_views: Dict[str, float] = field(default_factory=dict)  # 主观观点
    bl_view_confidences: Dict[str, float] = field(default_factory=dict)  # 观点置信度

    # 稳健优化参数
    robust_epsilon: float = 0.1                         # 不确定性半径

    # 数据参数
    min_history_length: int = 60                        # 最小历史长度
    lookback_window: int = 252                          # 回溯窗口

    # 有效前沿参数
    frontier_points: int = 50                           # 前沿点数

    # 实时更新参数
    update_interval: float = 60.0                       # 更新间隔(秒)
    enable_auto_rebalance: bool = False                 # 启用自动再平衡
    rebalance_threshold: float = 0.05                   # 再平衡阈值

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective.value,
            "risk_free_rate": self.risk_free_rate,
            "target_return": self.target_return,
            "target_volatility": self.target_volatility,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "allow_short": self.allow_short,
            "max_leverage": self.max_leverage,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
        }


# ============ 优化器基类 ============

class PortfolioOptimizer(ABC):
    """组合优化器基类"""

    def __init__(self, config: OptimizerConfig):
        self.config = config

    @abstractmethod
    def optimize(
        self,
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
        constraints: List[PortfolioConstraint],
        current_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """执行优化"""
        pass

    def _calculate_portfolio_return(
        self,
        weights: Dict[str, float],
        asset_data: Dict[str, AssetData],
    ) -> float:
        """计算组合期望收益率"""
        total_return = 0.0
        for symbol, weight in weights.items():
            if symbol in asset_data:
                total_return += weight * asset_data[symbol].expected_return
        return total_return

    def _calculate_portfolio_volatility(
        self,
        weights: Dict[str, float],
        symbols: List[str],
        covariance_matrix: np.ndarray,
    ) -> float:
        """计算组合波动率"""
        n = len(symbols)
        w = np.array([weights.get(s, 0.0) for s in symbols])

        variance = w.T @ covariance_matrix @ w
        return math.sqrt(max(0.0, variance))

    def _calculate_sharpe_ratio(
        self,
        portfolio_return: float,
        portfolio_volatility: float,
    ) -> float:
        """计算夏普比率"""
        if portfolio_volatility <= 0:
            return 0.0
        return (portfolio_return - self.config.risk_free_rate) / portfolio_volatility

    def _apply_constraints(
        self,
        weights: Dict[str, float],
        constraints: List[PortfolioConstraint],
        asset_data: Dict[str, AssetData],
    ) -> Dict[str, float]:
        """应用约束条件"""
        adjusted_weights = weights.copy()

        # 应用权重界限
        for symbol in adjusted_weights:
            adjusted_weights[symbol] = max(
                self.config.min_weight,
                min(self.config.max_weight, adjusted_weights[symbol])
            )

        # 归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {s: w / total for s, w in adjusted_weights.items()}

        # 应用杠杆约束
        total_leverage = sum(abs(w) for w in adjusted_weights.values())
        if total_leverage > self.config.max_leverage:
            scale = self.config.max_leverage / total_leverage
            adjusted_weights = {s: w * scale for s, w in adjusted_weights.items()}

        return adjusted_weights


# ============ 具体优化器实现 ============

class MeanVarianceOptimizer(PortfolioOptimizer):
    """均值方差优化器"""

    def optimize(
        self,
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
        constraints: List[PortfolioConstraint],
        current_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """均值方差优化"""
        import time
        start_time = time.time()

        symbols = sorted(list(asset_data.keys()))
        n = len(symbols)

        if n == 0:
            return self._create_empty_result()

        # 使用梯度下降求解
        weights = self._gradient_descent_optimization(
            symbols, asset_data, covariance_matrix
        )

        weights_dict = {symbols[i]: weights[i] for i in range(n)}
        weights_dict = self._apply_constraints(weights_dict, constraints, asset_data)

        # 计算组合指标
        portfolio_return = self._calculate_portfolio_return(weights_dict, asset_data)
        portfolio_vol = self._calculate_portfolio_volatility(weights_dict, symbols, covariance_matrix)
        sharpe = self._calculate_sharpe_ratio(portfolio_return, portfolio_vol)

        return OptimizationResult(
            optimization_id=str(uuid.uuid4()),
            objective=self.config.objective,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            optimization_time=time.time() - start_time,
            success=True,
            message="Mean-variance optimization completed",
        )

    def _gradient_descent_optimization(
        self,
        symbols: List[str],
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        """梯度下降优化"""
        n = len(symbols)

        # 初始化为等权重
        weights = np.ones(n) / n

        # 期望收益向量
        expected_returns = np.array([asset_data[s].expected_return for s in symbols])

        for iteration in range(self.config.max_iterations):
            old_weights = weights.copy()

            # 计算梯度
            portfolio_vol = math.sqrt(weights.T @ covariance_matrix @ weights)

            if portfolio_vol > 1e-8:
                # 夏普比率的梯度
                grad_return = expected_returns
                grad_vol = (covariance_matrix @ weights) / portfolio_vol

                excess_return = weights @ expected_returns - self.config.risk_free_rate
                grad_sharpe = (grad_return * portfolio_vol - excess_return * grad_vol) / (portfolio_vol ** 2)

                # 梯度上升(最大化夏普比率)
                weights += self.config.learning_rate * grad_sharpe

            # 投影到可行域
            weights = np.maximum(weights, self.config.min_weight)
            weights = np.minimum(weights, self.config.max_weight)

            # 归一化
            weights = weights / weights.sum()

            # 检查收敛
            if np.linalg.norm(weights - old_weights) < self.config.tolerance:
                break

        return weights


class MinVolatilityOptimizer(PortfolioOptimizer):
    """最小波动率优化器"""

    def optimize(
        self,
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
        constraints: List[PortfolioConstraint],
        current_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """最小波动率优化"""
        import time
        start_time = time.time()

        symbols = sorted(list(asset_data.keys()))
        n = len(symbols)

        if n == 0:
            return self._create_empty_result()

        # 最小方差解: w = (Σ^-1 * 1) / (1' * Σ^-1 * 1)
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
            ones = np.ones(n)

            weights = (cov_inv @ ones) / (ones.T @ cov_inv @ ones)

            # 确保非负
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()

        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用梯度下降
            weights = self._gradient_descent_min_vol(symbols, covariance_matrix)

        weights_dict = {symbols[i]: weights[i] for i in range(n)}
        weights_dict = self._apply_constraints(weights_dict, constraints, asset_data)

        portfolio_return = self._calculate_portfolio_return(weights_dict, asset_data)
        portfolio_vol = self._calculate_portfolio_volatility(weights_dict, symbols, covariance_matrix)
        sharpe = self._calculate_sharpe_ratio(portfolio_return, portfolio_vol)

        return OptimizationResult(
            optimization_id=str(uuid.uuid4()),
            objective=OptimizationObjective.MIN_VOLATILITY,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            optimization_time=time.time() - start_time,
            success=True,
            message="Minimum volatility optimization completed",
        )

    def _gradient_descent_min_vol(
        self,
        symbols: List[str],
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        """梯度下降最小化波动率"""
        n = len(symbols)
        weights = np.ones(n) / n

        for iteration in range(self.config.max_iterations):
            old_weights = weights.copy()

            # 计算梯度
            portfolio_vol = math.sqrt(weights.T @ covariance_matrix @ weights)

            if portfolio_vol > 1e-8:
                grad = (covariance_matrix @ weights) / portfolio_vol

                # 梯度下降
                weights -= self.config.learning_rate * grad

            # 投影到可行域
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()

            if np.linalg.norm(weights - old_weights) < self.config.tolerance:
                break

        return weights


class MaxSharpeOptimizer(PortfolioOptimizer):
    """最大夏普比率优化器"""

    def optimize(
        self,
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
        constraints: List[PortfolioConstraint],
        current_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """最大夏普比率优化"""
        import time
        start_time = time.time()

        symbols = sorted(list(asset_data.keys()))
        n = len(symbols)

        if n == 0:
            return self._create_empty_result()

        # 期望收益向量
        expected_returns = np.array([asset_data[s].expected_return for s in symbols])
        excess_returns = expected_returns - self.config.risk_free_rate

        # 使用二次规划求解最大夏普比率
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
            weights = cov_inv @ excess_returns
            weights = weights / weights.sum()
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
        except np.linalg.LinAlgError:
            # 回退到梯度下降
            weights = self._gradient_ascent_sharpe(
                symbols, asset_data, covariance_matrix
            )

        weights_dict = {symbols[i]: weights[i] for i in range(n)}
        weights_dict = self._apply_constraints(weights_dict, constraints, asset_data)

        portfolio_return = self._calculate_portfolio_return(weights_dict, asset_data)
        portfolio_vol = self._calculate_portfolio_volatility(weights_dict, symbols, covariance_matrix)
        sharpe = self._calculate_sharpe_ratio(portfolio_return, portfolio_vol)

        return OptimizationResult(
            optimization_id=str(uuid.uuid4()),
            objective=OptimizationObjective.MAX_SHARPE,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            optimization_time=time.time() - start_time,
            success=True,
            message="Maximum Sharpe ratio optimization completed",
        )

    def _gradient_ascent_sharpe(
        self,
        symbols: List[str],
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        """梯度上升求最大夏普"""
        n = len(symbols)
        weights = np.ones(n) / n

        expected_returns = np.array([asset_data[s].expected_return for s in symbols])

        for iteration in range(self.config.max_iterations):
            old_weights = weights.copy()

            portfolio_return = weights @ expected_returns
            portfolio_vol = math.sqrt(weights.T @ covariance_matrix @ weights)

            if portfolio_vol > 1e-8:
                grad_return = expected_returns
                grad_vol = (covariance_matrix @ weights) / portfolio_vol

                excess_return = portfolio_return - self.config.risk_free_rate
                grad_sharpe = (grad_return * portfolio_vol - excess_return * grad_vol) / (portfolio_vol ** 2)

                weights += self.config.learning_rate * grad_sharpe

            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()

            if np.linalg.norm(weights - old_weights) < self.config.tolerance:
                break

        return weights


class CVaROptimizer(PortfolioOptimizer):
    """CVaR优化器"""

    def optimize(
        self,
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
        constraints: List[PortfolioConstraint],
        current_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """CVaR优化 - 最小化条件风险价值"""
        import time
        start_time = time.time()

        symbols = sorted(list(asset_data.keys()))
        n = len(symbols)

        if n == 0:
            return self._create_empty_result()

        # 简化实现: 使用蒙特卡洛模拟
        weights = self._minimize_cvar_simulation(
            symbols, asset_data, covariance_matrix
        )

        weights_dict = {symbols[i]: weights[i] for i in range(n)}
        weights_dict = self._apply_constraints(weights_dict, constraints, asset_data)

        portfolio_return = self._calculate_portfolio_return(weights_dict, asset_data)
        portfolio_vol = self._calculate_portfolio_volatility(weights_dict, symbols, covariance_matrix)
        sharpe = self._calculate_sharpe_ratio(portfolio_return, portfolio_vol)

        # 计算CVaR
        cvar = self._calculate_cvar(weights_dict, symbols, asset_data, covariance_matrix)

        return OptimizationResult(
            optimization_id=str(uuid.uuid4()),
            objective=OptimizationObjective.MIN_CVAR,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            portfolio_cvar=cvar,
            optimization_time=time.time() - start_time,
            success=True,
            message="CVaR optimization completed",
        )

    def _minimize_cvar_simulation(
        self,
        symbols: List[str],
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        """使用模拟最小化CVaR"""
        n = len(symbols)

        # 生成场景
        num_scenarios = 10000
        expected_returns = np.array([asset_data[s].expected_return for s in symbols])

        # 多元正态分布抽样
        np.random.seed(42)
        scenarios = np.random.multivariate_normal(
            expected_returns, covariance_matrix, num_scenarios
        )

        # 优化CVaR
        best_weights = np.ones(n) / n
        best_cvar = float('inf')

        # 简单搜索
        for _ in range(100):
            # 随机生成权重
            weights = np.random.dirichlet(np.ones(n))

            # 计算CVaR
            portfolio_returns = scenarios @ weights
            var_threshold = np.percentile(portfolio_returns, (1 - self.config.cvar_confidence) * 100)
            cvar = -np.mean(portfolio_returns[portfolio_returns <= var_threshold])

            if cvar < best_cvar:
                best_cvar = cvar
                best_weights = weights

        return best_weights

    def _calculate_cvar(
        self,
        weights: Dict[str, float],
        symbols: List[str],
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
    ) -> float:
        """计算CVaR"""
        # 简化实现: 使用正态分布假设
        w = np.array([weights.get(s, 0.0) for s in symbols])
        expected_returns = np.array([asset_data[s].expected_return for s in symbols])

        portfolio_return = w @ expected_returns
        portfolio_vol = math.sqrt(w.T @ covariance_matrix @ w)

        # VaR (正态分布假设)
        from scipy import stats
        z_score = stats.norm.ppf(1 - self.config.cvar_confidence)
        var = portfolio_return + z_score * portfolio_vol

        # CVaR (正态分布假设)
        cvar = portfolio_return - portfolio_vol * stats.norm.pdf(z_score) / (1 - self.config.cvar_confidence)

        return -cvar  # 返回正值表示损失


class BlackLittermanOptimizer(PortfolioOptimizer):
    """Black-Litterman优化器"""

    def optimize(
        self,
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
        constraints: List[PortfolioConstraint],
        current_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """Black-Litterman优化"""
        import time
        start_time = time.time()

        symbols = sorted(list(asset_data.keys()))
        n = len(symbols)

        if n == 0:
            return self._create_empty_result()

        # 市场隐含收益率 (反向优化)
        market_weights = np.ones(n) / n  # 简化: 使用等权重作为市场权重
        risk_aversion = 2.5

        implied_returns = risk_aversion * covariance_matrix @ market_weights

        # 应用主观观点
        if self.config.bl_views:
            bl_returns = self._apply_black_litterman_views(
                symbols, implied_returns, covariance_matrix
            )
        else:
            bl_returns = implied_returns

        # 优化
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
            weights = (cov_inv @ bl_returns) / (np.ones(n).T @ cov_inv @ bl_returns)
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
        except Exception as exc:
            logger.warning(
                "Black-Litterman optimisation failed for %d assets; falling back to "
                "equal weights, which is not an optimised allocation: %s", n, exc
            )
            weights = np.ones(n) / n

        weights_dict = {symbols[i]: weights[i] for i in range(n)}
        weights_dict = self._apply_constraints(weights_dict, constraints, asset_data)

        # 更新资产数据的期望收益
        for i, symbol in enumerate(symbols):
            asset_data[symbol].expected_return = bl_returns[i]

        portfolio_return = self._calculate_portfolio_return(weights_dict, asset_data)
        portfolio_vol = self._calculate_portfolio_volatility(weights_dict, symbols, covariance_matrix)
        sharpe = self._calculate_sharpe_ratio(portfolio_return, portfolio_vol)

        return OptimizationResult(
            optimization_id=str(uuid.uuid4()),
            objective=OptimizationObjective.BLACK_LITTERMAN,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            optimization_time=time.time() - start_time,
            success=True,
            message="Black-Litterman optimization completed",
        )

    def _apply_black_litterman_views(
        self,
        symbols: List[str],
        implied_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        """应用Black-Litterman主观观点"""
        n = len(symbols)

        # 构建观点矩阵 P 和观点收益向量 Q
        views = []
        view_returns = []
        view_uncertainties = []

        for symbol, view_return in self.config.bl_views.items():
            if symbol in symbols:
                idx = symbols.index(symbol)
                p = np.zeros(n)
                p[idx] = 1.0
                views.append(p)
                view_returns.append(view_return)

                # 观点不确定性
                confidence = self.config.bl_view_confidences.get(symbol, 1.0)
                uncertainty = (1 - confidence) * covariance_matrix[idx, idx]
                view_uncertainties.append(uncertainty)

        if not views:
            return implied_returns

        P = np.array(views)
        Q = np.array(view_returns)
        Omega = np.diag(view_uncertainties)

        # Black-Litterman公式
        tau = self.config.bl_tau
        M_inv = np.linalg.inv(tau * covariance_matrix)

        try:
            bl_mean = np.linalg.inv(M_inv + P.T @ np.linalg.inv(Omega) @ P) @ (
                M_inv @ implied_returns + P.T @ np.linalg.inv(Omega) @ Q
            )
            return bl_mean
        except Exception as exc:
            # Returning the prior silently would let the caller believe their views
            # were incorporated when they were discarded.
            logger.warning(
                "Black-Litterman view blending failed; returning market-implied "
                "returns with the supplied views discarded: %s", exc
            )
            return implied_returns


class RobustOptimizer(PortfolioOptimizer):
    """稳健优化器"""

    def optimize(
        self,
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
        constraints: List[PortfolioConstraint],
        current_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """稳健优化 - 考虑参数不确定性"""
        import time
        start_time = time.time()

        symbols = sorted(list(asset_data.keys()))
        n = len(symbols)

        if n == 0:
            return self._create_empty_result()

        # 稳健优化: 最坏情况优化
        weights = self._worst_case_optimization(
            symbols, asset_data, covariance_matrix
        )

        weights_dict = {symbols[i]: weights[i] for i in range(n)}
        weights_dict = self._apply_constraints(weights_dict, constraints, asset_data)

        portfolio_return = self._calculate_portfolio_return(weights_dict, asset_data)
        portfolio_vol = self._calculate_portfolio_volatility(weights_dict, symbols, covariance_matrix)
        sharpe = self._calculate_sharpe_ratio(portfolio_return, portfolio_vol)

        return OptimizationResult(
            optimization_id=str(uuid.uuid4()),
            objective=OptimizationObjective.ROBUST,
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            optimization_time=time.time() - start_time,
            success=True,
            message="Robust optimization completed",
        )

    def _worst_case_optimization(
        self,
        symbols: List[str],
        asset_data: Dict[str, AssetData],
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        """最坏情况优化"""
        n = len(symbols)

        # 期望收益的不确定性椭球
        expected_returns = np.array([asset_data[s].expected_return for s in symbols])
        epsilon = self.config.robust_epsilon

        # 稳健优化: min_w max_{||u||<=epsilon} (-w'(mu - u) + lambda * w'Sigma*w)
        # 简化为: min_w (-w'mu + epsilon*||w||_2 + lambda * w'Sigma*w)

        weights = np.ones(n) / n
        lambda_risk = 1.0

        for iteration in range(self.config.max_iterations):
            old_weights = weights.copy()

            # 梯度
            grad_robust = -expected_returns + epsilon * weights / (np.linalg.norm(weights) + 1e-8)
            grad_risk = 2 * lambda_risk * covariance_matrix @ weights
            grad = grad_robust + grad_risk

            weights -= self.config.learning_rate * grad

            # 投影
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()

            if np.linalg.norm(weights - old_weights) < self.config.tolerance:
                break

        return weights

    def _create_empty_result(self) -> OptimizationResult:
        """创建空结果"""
        return OptimizationResult(
            optimization_id=str(uuid.uuid4()),
            objective=self.config.objective,
            weights={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            message="No assets to optimize",
        )


# ============ 组合优化引擎 ============

class PortfolioOptimizationEngine:
    """
    组合优化引擎

    功能:
    - 多种优化目标和方法
    - 约束条件管理
    - 有效前沿计算
    - 风险分解和边际贡献
    - 情景分析和What-if
    - 实时优化和再平衡
    """

    def __init__(
        self,
        config: Optional[OptimizerConfig] = None,
    ):
        self.config = config or OptimizerConfig()

        # 状态
        self._state = EngineState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        # 数据存储
        self._asset_data: Dict[str, AssetData] = {}
        self._symbols: Set[str] = set()
        self._constraints: List[PortfolioConstraint] = []

        # 优化结果
        self._current_result: Optional[OptimizationResult] = None
        self._optimization_history: deque = deque(maxlen=1000)

        # 协方差矩阵
        self._covariance_matrix: Optional[np.ndarray] = None
        self._last_covariance_update: Optional[datetime] = None

        # 优化器
        self._optimizers: Dict[OptimizationObjective, PortfolioOptimizer] = {
            OptimizationObjective.MEAN_VARIANCE: MeanVarianceOptimizer(self.config),
            OptimizationObjective.MIN_VOLATILITY: MinVolatilityOptimizer(self.config),
            OptimizationObjective.MAX_SHARPE: MaxSharpeOptimizer(self.config),
            OptimizationObjective.MIN_CVAR: CVaROptimizer(self.config),
            OptimizationObjective.BLACK_LITTERMAN: BlackLittermanOptimizer(self.config),
            OptimizationObjective.ROBUST: RobustOptimizer(self.config),
        }

        # 有效前沿
        self._current_frontier: Optional[EfficientFrontier] = None

        logger.info("Portfolio Optimization Engine initialized")

    @property
    def state(self) -> EngineState:
        return self._state

    # ============ 生命周期 ============

    async def start(self) -> None:
        """启动引擎"""
        if self._state == EngineState.RUNNING:
            return

        logger.info("Starting Portfolio Optimization Engine")
        self._state = EngineState.RUNNING

        if self.config.enable_auto_rebalance:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("Portfolio Optimization Engine started")

    async def stop(self) -> None:
        """停止引擎"""
        if self._state == EngineState.IDLE:
            return

        logger.info("Stopping Portfolio Optimization Engine")
        self._state = EngineState.IDLE

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Portfolio Optimization Engine stopped")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state == EngineState.RUNNING:
            try:
                # 定期重新优化
                if self._should_reoptimize():
                    await self.optimize_async()

                await asyncio.sleep(self.config.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self._state = EngineState.ERROR

    def _should_reoptimize(self) -> bool:
        """检查是否需要重新优化"""
        if self._current_result is None:
            return True

        # 检查距离上次优化的时间
        time_since_last = datetime.now() - self._current_result.timestamp
        if time_since_last.total_seconds() > self.config.update_interval:
            return True

        return False

    # ============ 资产管理 ============

    def add_asset(self, symbol: str, **kwargs) -> None:
        """添加资产"""
        if symbol not in self._symbols:
            self._symbols.add(symbol)
            self._asset_data[symbol] = AssetData(symbol=symbol, **kwargs)
            logger.info(f"Added asset: {symbol}")

    def remove_asset(self, symbol: str) -> None:
        """移除资产"""
        if symbol in self._symbols:
            self._symbols.remove(symbol)
            self._asset_data.pop(symbol, None)
            logger.info(f"Removed asset: {symbol}")

    def get_assets(self) -> List[str]:
        """获取所有资产"""
        return list(self._symbols)

    def update_asset_data(self, symbol: str, **kwargs) -> None:
        """更新资产数据"""
        if symbol in self._asset_data:
            asset = self._asset_data[symbol]
            for key, value in kwargs.items():
                if hasattr(asset, key):
                    setattr(asset, key, value)
            asset.last_updated = datetime.now()

    async def add_return(self, symbol: str, return_value: float, timestamp: datetime) -> None:
        """添加收益率数据"""
        if symbol not in self._symbols:
            self.add_asset(symbol)

        asset = self._asset_data[symbol]
        asset.returns.append(return_value)
        asset.timestamps.append(timestamp)

        # 保持窗口大小
        if len(asset.returns) > self.config.lookback_window:
            asset.returns = asset.returns[-self.config.lookback_window:]
            asset.timestamps = asset.timestamps[-self.config.lookback_window:]

        # 更新统计信息
        if len(asset.returns) >= 2:
            asset.volatility = statistics.stdev(asset.returns)
            asset.mean_return = statistics.mean(asset.returns)
            asset.expected_return = asset.mean_return  # 简化: 使用历史均值

    # ============ 约束管理 ============

    def add_constraint(self, constraint: PortfolioConstraint) -> None:
        """添加约束"""
        self._constraints.append(constraint)
        logger.info(f"Added constraint: {constraint.constraint_type.value}")

    def clear_constraints(self) -> None:
        """清除所有约束"""
        self._constraints.clear()
        logger.info("Cleared all constraints")

    def get_constraints(self) -> List[PortfolioConstraint]:
        """获取所有约束"""
        return self._constraints.copy()

    # ============ 协方差估计 ============

    def estimate_covariance(self) -> np.ndarray:
        """估计协方差矩阵"""
        symbols = sorted(list(self._symbols))
        n = len(symbols)

        if n == 0:
            return np.array([[]])

        # 获取收益率数据
        returns_data = []
        valid_symbols = []

        for symbol in symbols:
            asset = self._asset_data[symbol]
            if len(asset.returns) >= self.config.min_history_length:
                returns_data.append(asset.returns[-self.config.lookback_window:])
                valid_symbols.append(symbol)

        if not returns_data:
            return np.eye(n) * 0.01  # 默认协方差

        # 对齐数据长度
        min_length = min(len(r) for r in returns_data)
        aligned_returns = np.array([r[-min_length:] for r in returns_data])

        # 计算协方差矩阵
        cov_matrix = np.cov(aligned_returns)

        # 确保正定性
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])
        elif cov_matrix.ndim == 1:
            cov_matrix = np.diag(cov_matrix)

        self._covariance_matrix = cov_matrix
        self._last_covariance_update = datetime.now()

        return cov_matrix

    # ============ 优化 ============

    def optimize(
        self,
        objective: Optional[OptimizationObjective] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """执行优化"""
        self._state = EngineState.OPTIMIZING

        try:
            objective = objective or self.config.objective

            # 估计协方差矩阵
            cov_matrix = self.estimate_covariance()

            if cov_matrix.size == 0:
                return OptimizationResult(
                    optimization_id=str(uuid.uuid4()),
                    objective=objective,
                    weights={},
                    expected_return=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    success=False,
                    message="Insufficient data for optimization",
                )

            # 选择优化器
            optimizer = self._optimizers.get(objective)
            if optimizer is None:
                optimizer = self._optimizers[OptimizationObjective.MAX_SHARPE]

            # 执行优化
            result = optimizer.optimize(
                self._asset_data,
                cov_matrix,
                self._constraints,
                current_weights,
            )

            # 计算额外指标
            result = self._enhance_result(result)

            # 验证约束
            result = self._validate_constraints(result)

            # 保存结果
            self._current_result = result
            self._optimization_history.append(result)

            self._state = EngineState.RUNNING

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self._state = EngineState.ERROR

            return OptimizationResult(
                optimization_id=str(uuid.uuid4()),
                objective=objective or self.config.objective,
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                success=False,
                message=f"Optimization error: {str(e)}",
            )

    async def optimize_async(
        self,
        objective: Optional[OptimizationObjective] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """异步优化"""
        return await asyncio.to_thread(self.optimize, objective, current_weights)

    def _enhance_result(self, result: OptimizationResult) -> OptimizationResult:
        """增强优化结果 - 计算额外指标"""
        if not result.weights:
            return result

        # 分散化比率
        result.diversification_ratio = self._calculate_diversification_ratio(result.weights)

        # 有效资产数
        result.effective_assets = self._calculate_effective_assets(result.weights)

        # 集中度指数(HHI)
        result.concentration_index = self._calculate_concentration_index(result.weights)

        return result

    def _validate_constraints(self, result: OptimizationResult) -> OptimizationResult:
        """验证约束条件"""
        violations = []

        for constraint in self._constraints:
            is_valid, message = constraint.validate(result.weights, self._asset_data)
            if not is_valid:
                violations.append(message)
                if constraint.is_hard_constraint:
                    result.constraints_satisfied = False

        result.constraint_violations = violations

        return result

    def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """计算分散化比率"""
        if not weights or self._covariance_matrix is None:
            return 1.0

        symbols = sorted(list(weights.keys()))
        w = np.array([weights[s] for s in symbols])

        # 加权平均波动率
        weighted_vol = sum(
            weights[s] * self._asset_data[s].volatility
            for s in symbols
            if s in self._asset_data
        )

        # 组合波动率
        portfolio_vol = math.sqrt(w.T @ self._covariance_matrix @ w)

        if portfolio_vol > 1e-8:
            return weighted_vol / portfolio_vol

        return 1.0

    def _calculate_effective_assets(self, weights: Dict[str, float]) -> float:
        """计算有效资产数 (1 / HHI)"""
        if not weights:
            return 0.0

        # HHI (Herfindahl-Hirschman Index)
        hhi = sum(w ** 2 for w in weights.values())

        if hhi > 1e-8:
            return 1.0 / hhi

        return 0.0

    def _calculate_concentration_index(self, weights: Dict[str, float]) -> float:
        """计算集中度指数(HHI)"""
        if not weights:
            return 0.0

        return sum(w ** 2 for w in weights.values())

    # ============ 有效前沿 ============

    def calculate_efficient_frontier(
        self,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
    ) -> EfficientFrontier:
        """计算有效前沿"""
        symbols = sorted(list(self._asset_data.keys()))

        if not symbols:
            return EfficientFrontier(
                frontier_id=str(uuid.uuid4()),
                returns=[],
                volatilities=[],
                sharpe_ratios=[],
                weights_series=[],
                min_volatility_weights={},
                max_sharpe_weights={},
            )

        # 确定收益率范围
        if min_return is None:
            min_return = min(self._asset_data[s].expected_return for s in symbols)
        if max_return is None:
            max_return = max(self._asset_data[s].expected_return for s in symbols)

        # 生成目标收益率序列
        target_returns = np.linspace(min_return, max_return, self.config.frontier_points)

        returns_list = []
        volatilities_list = []
        sharpe_list = []
        weights_list = []

        cov_matrix = self.estimate_covariance()

        # 计算每个目标收益率对应的最小方差组合
        for target_return in target_returns:
            # 临时设置目标收益率
            old_target = self.config.target_return
            self.config.target_return = target_return

            # 优化
            result = self.optimize(OptimizationObjective.MIN_VOLATILITY)

            if result.success:
                returns_list.append(result.expected_return)
                volatilities_list.append(result.expected_volatility)
                sharpe_list.append(result.sharpe_ratio)
                weights_list.append(result.weights)

            self.config.target_return = old_target

        # 找到特殊点
        min_vol_result = self.optimize(OptimizationObjective.MIN_VOLATILITY)
        max_sharpe_result = self.optimize(OptimizationObjective.MAX_SHARPE)

        frontier = EfficientFrontier(
            frontier_id=str(uuid.uuid4()),
            returns=returns_list,
            volatilities=volatilities_list,
            sharpe_ratios=sharpe_list,
            weights_series=weights_list,
            min_volatility_weights=min_vol_result.weights if min_vol_result.success else {},
            max_sharpe_weights=max_sharpe_result.weights if max_sharpe_result.success else {},
            num_points=len(returns_list),
            risk_free_rate=self.config.risk_free_rate,
        )

        self._current_frontier = frontier

        return frontier

    # ============ 风险分解 ============

    def calculate_risk_decomposition(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[RiskDecomposition]:
        """计算风险分解"""
        if weights is None:
            if self._current_result is None:
                return []
            weights = self._current_result.weights

        if not weights or self._covariance_matrix is None:
            return []

        symbols = sorted(list(weights.keys()))
        w = np.array([weights[s] for s in symbols])

        # 组合波动率
        portfolio_vol = math.sqrt(w.T @ self._covariance_matrix @ w)

        decompositions = []

        for i, symbol in enumerate(symbols):
            # 边际风险贡献
            marginal_risk = (self._covariance_matrix[i, :] @ w) / portfolio_vol if portfolio_vol > 1e-8 else 0.0

            # 成分风险贡献
            component_risk = weights[symbol] * marginal_risk

            # 百分比贡献
            pct_contribution = component_risk / portfolio_vol if portfolio_vol > 1e-8 else 0.0

            decompositions.append(RiskDecomposition(
                symbol=symbol,
                marginal_risk=marginal_risk,
                component_risk=component_risk,
                pct_contribution=pct_contribution,
            ))

        return decompositions

    def calculate_marginal_contribution(
        self,
        symbol: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """计算单个资产的边际贡献"""
        if weights is None:
            if self._current_result is None:
                return 0.0
            weights = self._current_result.weights

        if symbol not in weights or self._covariance_matrix is None:
            return 0.0

        symbols = sorted(list(weights.keys()))

        if symbol not in symbols:
            return 0.0

        idx = symbols.index(symbol)
        w = np.array([weights[s] for s in symbols])

        # 组合波动率
        portfolio_vol = math.sqrt(w.T @ self._covariance_matrix @ w)

        if portfolio_vol > 1e-8:
            marginal_risk = (self._covariance_matrix[idx, :] @ w) / portfolio_vol
            return marginal_risk

        return 0.0

    # ============ 情景分析 ============

    def run_what_if_scenario(
        self,
        scenario_name: str,
        asset_return_changes: Optional[Dict[str, float]] = None,
        volatility_multipliers: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> WhatIfScenario:
        """运行What-if情景分析"""
        if weights is None:
            if self._current_result is None:
                weights = {}
            else:
                weights = self._current_result.weights

        # 创建情景
        scenario = WhatIfScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name=scenario_name,
            asset_return_changes=asset_return_changes or {},
            volatility_multipliers=volatility_multipliers or {},
        )

        if not weights:
            return scenario

        # 基准组合指标
        base_return = sum(
            weights.get(s, 0) * self._asset_data[s].expected_return
            for s in self._asset_data
            if s in weights
        )

        symbols = sorted(list(weights.keys()))
        w = np.array([weights[s] for s in symbols])

        if self._covariance_matrix is not None:
            base_vol = math.sqrt(w.T @ self._covariance_matrix @ w)
        else:
            base_vol = 0.0

        # 应用情景变化
        scenario_returns = {}
        for symbol in weights:
            if symbol in self._asset_data:
                base_ret = self._asset_data[symbol].expected_return
                change = asset_return_changes.get(symbol, 0.0) if asset_return_changes else 0.0
                scenario_returns[symbol] = base_ret + change

        # 计算情景组合收益
        scenario.scenario_return = sum(
            weights.get(s, 0) * scenario_returns.get(s, 0)
            for s in scenario_returns
        )

        # 计算情景波动率
        if volatility_multipliers and self._covariance_matrix is not None:
            # 调整协方差矩阵
            vol_mult_vector = np.array([
                volatility_multipliers.get(s, 1.0) for s in symbols
            ])
            adjusted_cov = self._covariance_matrix * np.outer(vol_mult_vector, vol_mult_vector)
            scenario.scenario_volatility = math.sqrt(w.T @ adjusted_cov @ w)
        else:
            scenario.scenario_volatility = base_vol

        # 计算变化
        scenario.return_change = scenario.scenario_return - base_return
        scenario.volatility_change = scenario.scenario_volatility - base_vol

        return scenario

    # ============ 查询方法 ============

    def get_current_result(self) -> Optional[OptimizationResult]:
        """获取当前优化结果"""
        return self._current_result

    def get_optimization_history(self, limit: int = 100) -> List[OptimizationResult]:
        """获取优化历史"""
        return list(self._optimization_history)[-limit:]

    def get_efficient_frontier(self) -> Optional[EfficientFrontier]:
        """获取有效前沿"""
        return self._current_frontier

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "num_assets": len(self._symbols),
            "num_constraints": len(self._constraints),
            "current_result": self._current_result.to_dict() if self._current_result else None,
            "optimization_count": len(self._optimization_history),
            "last_optimization": self._current_result.timestamp.isoformat() if self._current_result else None,
            "config": self.config.to_dict(),
        }


# ============ 工厂函数 ============

def create_optimization_engine(
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.02,
    **kwargs,
) -> PortfolioOptimizationEngine:
    """
    创建组合优化引擎

    Args:
        objective: 优化目标
        risk_free_rate: 无风险利率
        **kwargs: 其他配置参数
    """
    try:
        obj = OptimizationObjective(objective)
    except ValueError:
        obj = OptimizationObjective.MAX_SHARPE

    config = OptimizerConfig(
        objective=obj,
        risk_free_rate=risk_free_rate,
        **{k: v for k, v in kwargs.items() if hasattr(OptimizerConfig, k)}
    )

    return PortfolioOptimizationEngine(config=config)


def create_constraint(
    constraint_type: str,
    **kwargs,
) -> PortfolioConstraint:
    """创建约束"""
    try:
        ct = ConstraintType(constraint_type)
    except ValueError:
        ct = ConstraintType.WEIGHT_BOUNDS

    return PortfolioConstraint(
        constraint_type=ct,
        **{k: v for k, v in kwargs.items() if k in PortfolioConstraint.__dataclass_fields__}
    )


def quick_optimize(
    symbols: List[str],
    expected_returns: Dict[str, float],
    volatilities: Dict[str, float],
    objective: str = "max_sharpe",
    **kwargs,
) -> Dict[str, Any]:
    """
    快速优化

    Args:
        symbols: 资产列表
        expected_returns: 期望收益率字典
        volatilities: 波动率字典
        objective: 优化目标
        **kwargs: 其他参数
    """
    engine = create_optimization_engine(objective=objective, **kwargs)

    # 添加资产
    for symbol in symbols:
        engine.add_asset(
            symbol,
            expected_return=expected_returns.get(symbol, 0.0),
            volatility=volatilities.get(symbol, 0.0),
        )

    # 执行优化
    result = engine.optimize()

    return {
        "result": result.to_dict(),
        "statistics": engine.get_statistics(),
    }


# ============ 导出 ============

__all__ = [
    # 枚举
    "OptimizationObjective",
    "OptimizationMethod",
    "ConstraintType",
    "RiskMetric",
    "EngineState",
    # 数据类
    "AssetData",
    "PortfolioConstraint",
    "OptimizationResult",
    "EfficientFrontier",
    "RiskDecomposition",
    "WhatIfScenario",
    "OptimizerConfig",
    # 优化器
    "PortfolioOptimizer",
    "MeanVarianceOptimizer",
    "MinVolatilityOptimizer",
    "MaxSharpeOptimizer",
    "CVaROptimizer",
    "BlackLittermanOptimizer",
    "RobustOptimizer",
    # 核心引擎
    "PortfolioOptimizationEngine",
    # 工厂函数
    "create_optimization_engine",
    "create_constraint",
    "quick_optimize",
]
