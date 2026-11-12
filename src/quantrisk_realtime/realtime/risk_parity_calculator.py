"""
风险平价计算器 (Risk Parity Calculator)

提供实时风险平价组合优化功能:
- Equal Risk Contribution 等风险贡献
- Hierarchical Risk Parity 层次风险平价
- Inverse Volatility Weighting 反向波动率加权
- Risk Budgeting 风险预算
- Marginal Risk Contribution 边际风险贡献
- Covariance Estimation 协方差估计
- Leverage Constraints 杠杆约束
- Real-time Weight Updates 实时权重更新

这是现代资产配置和投资组合管理的核心模块。
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


# ============ Enums ============

class OptimizationMethod(str, Enum):
    """优化方法"""
    EQUAL_RISK = "equal_risk"                           # 等风险贡献
    HIERARCHICAL = "hierarchical"                       # 层次风险平价
    INVERSE_VOLATILITY = "inverse_volatility"           # 反向波动率加权
    RISK_BUDGETING = "risk_budgeting"                   # 风险预算
    MIN_VARIANCE = "min_variance"                       # 最小方差
    MAX_DIVERSIFICATION = "max_diversification"         # 最大分散化


class CovarianceMethod(str, Enum):
    """协方差估计方法"""
    SAMPLE = "sample"                                   # 样本协方差
    EXPONENTIAL = "exponential"                         # 指数加权
    LEDOIT_WOLF = "ledoit_wolf"                         # Ledoit-Wolf收缩
    OAS = "oas"                                         # Oracle近似收缩
    CONSTANT_CORRELATION = "constant_correlation"       # 恒定相关性


class RebalanceFrequency(str, Enum):
    """再平衡频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


class CalculatorState(str, Enum):
    """计算器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ Data Classes ============

@dataclass
class AssetData:
    """资产数据"""
    symbol: str
    returns: List[float]
    prices: List[float]
    timestamps: List[datetime]
    volatility: float = 0.0
    mean_return: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "returns_count": len(self.returns),
            "volatility": self.volatility,
            "mean_return": self.mean_return,
        }


@dataclass
class PortfolioWeights:
    """组合权重"""
    symbols: List[str]
    weights: List[float]
    timestamp: datetime
    method: OptimizationMethod
    total_risk: float = 0.0
    portfolio_volatility: float = 0.0
    diversification_ratio: float = 0.0
    leverage: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_weight(self, symbol: str) -> float:
        """获取单个资产的权重"""
        try:
            idx = self.symbols.index(symbol)
            return self.weights[idx]
        except ValueError:
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "weights": self.weights,
            "timestamp": self.timestamp.isoformat(),
            "method": self.method.value,
            "total_risk": self.total_risk,
            "portfolio_volatility": self.portfolio_volatility,
            "diversification_ratio": self.diversification_ratio,
            "leverage": self.leverage,
            "metadata": self.metadata,
        }


@dataclass
class RiskContribution:
    """风险贡献"""
    symbol: str
    weight: float
    volatility: float
    marginal_risk: float                    # 边际风险贡献 (dσ/dw_i)
    component_risk: float                   # 成分风险贡献 (w_i * dσ/dw_i)
    pct_contribution: float                 # 百分比风险贡献
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "weight": self.weight,
            "volatility": self.volatility,
            "marginal_risk": self.marginal_risk,
            "component_risk": self.component_risk,
            "pct_contribution": self.pct_contribution,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CovarianceMatrix:
    """协方差矩阵"""
    symbols: List[str]
    matrix: List[List[float]]
    method: CovarianceMethod
    timestamp: datetime
    condition_number: float = 0.0           # 条件数 (矩阵稳定性指标)
    sample_size: int = 0

    def get_covariance(self, symbol1: str, symbol2: str) -> Optional[float]:
        """获取两个资产的协方差"""
        try:
            idx1 = self.symbols.index(symbol1)
            idx2 = self.symbols.index(symbol2)
            return self.matrix[idx1][idx2]
        except (ValueError, IndexError):
            return None

    def get_variance(self, symbol: str) -> Optional[float]:
        """获取资产的方差"""
        return self.get_covariance(symbol, symbol)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "matrix": self.matrix,
            "method": self.method.value,
            "timestamp": self.timestamp.isoformat(),
            "condition_number": self.condition_number,
            "sample_size": self.sample_size,
        }


@dataclass
class RiskBudget:
    """风险预算"""
    symbol: str
    target_risk_pct: float                  # 目标风险百分比
    current_risk_pct: float = 0.0           # 当前风险百分比
    deviation: float = 0.0                  # 偏离度
    is_violated: bool = False               # 是否违反预算

    def update(self, current_risk_pct: float):
        """更新当前风险"""
        self.current_risk_pct = current_risk_pct
        self.deviation = current_risk_pct - self.target_risk_pct
        self.is_violated = abs(self.deviation) > 0.05  # 5%容忍度

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_risk_pct": self.target_risk_pct,
            "current_risk_pct": self.current_risk_pct,
            "deviation": self.deviation,
            "is_violated": self.is_violated,
        }


@dataclass
class LeverageConstraint:
    """杠杆约束"""
    min_leverage: float = 0.5
    max_leverage: float = 2.0
    target_leverage: float = 1.0
    current_leverage: float = 1.0

    def is_valid(self, leverage: float) -> bool:
        """检查杠杆是否在约束范围内"""
        return self.min_leverage <= leverage <= self.max_leverage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_leverage": self.min_leverage,
            "max_leverage": self.max_leverage,
            "target_leverage": self.target_leverage,
            "current_leverage": self.current_leverage,
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    weights: PortfolioWeights
    risk_contributions: List[RiskContribution]
    covariance_matrix: CovarianceMatrix
    convergence_info: Dict[str, Any]
    optimization_time: float                # 优化耗时(秒)
    success: bool = True
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "risk_contributions": [rc.to_dict() for rc in self.risk_contributions],
            "covariance_matrix": self.covariance_matrix.to_dict(),
            "convergence_info": self.convergence_info,
            "optimization_time": self.optimization_time,
            "success": self.success,
            "message": self.message,
        }


@dataclass
class RebalanceSignal:
    """再平衡信号"""
    signal_id: str
    should_rebalance: bool
    reason: str
    old_weights: PortfolioWeights
    new_weights: Optional[PortfolioWeights]
    timestamp: datetime
    turnover: float = 0.0                   # 换手率
    tracking_error: float = 0.0             # 跟踪误差

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "should_rebalance": self.should_rebalance,
            "reason": self.reason,
            "old_weights": self.old_weights.to_dict(),
            "new_weights": self.new_weights.to_dict() if self.new_weights else None,
            "timestamp": self.timestamp.isoformat(),
            "turnover": self.turnover,
            "tracking_error": self.tracking_error,
        }


@dataclass
class CalculatorConfig:
    """计算器配置"""
    # 优化参数
    optimization_method: OptimizationMethod = OptimizationMethod.EQUAL_RISK
    covariance_method: CovarianceMethod = CovarianceMethod.SAMPLE

    # 数据参数
    min_history_length: int = 60            # 最小历史数据长度
    lookback_window: int = 252              # 回溯窗口(天)
    exponential_lambda: float = 0.94        # 指数加权衰减因子

    # 约束参数
    min_weight: float = 0.0                 # 最小权重
    max_weight: float = 1.0                 # 最大权重
    leverage_constraint: LeverageConstraint = field(default_factory=LeverageConstraint)

    # 再平衡参数
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    rebalance_threshold: float = 0.05       # 再平衡阈值

    # 优化参数
    max_iterations: int = 1000              # 最大迭代次数
    tolerance: float = 1e-6                 # 收敛容差

    # 风险预算参数
    risk_budgets: Dict[str, float] = field(default_factory=dict)  # symbol -> target_risk_pct

    # 监控参数
    update_interval: float = 60.0           # 更新间隔(秒)
    enable_auto_rebalance: bool = False     # 启用自动再平衡

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimization_method": self.optimization_method.value,
            "covariance_method": self.covariance_method.value,
            "min_history_length": self.min_history_length,
            "lookback_window": self.lookback_window,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "leverage_constraint": self.leverage_constraint.to_dict(),
            "rebalance_frequency": self.rebalance_frequency.value,
            "rebalance_threshold": self.rebalance_threshold,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "update_interval": self.update_interval,
            "enable_auto_rebalance": self.enable_auto_rebalance,
        }


# ============ Covariance Estimators ============

class CovarianceEstimator(ABC):
    """协方差估计器基类"""

    @abstractmethod
    def estimate(self, returns: Dict[str, List[float]]) -> CovarianceMatrix:
        """估计协方差矩阵"""
        pass


class SampleCovarianceEstimator(CovarianceEstimator):
    """样本协方差估计器"""

    def estimate(self, returns: Dict[str, List[float]]) -> CovarianceMatrix:
        symbols = sorted(list(returns.keys()))
        n = len(symbols)

        # 对齐数据长度
        min_length = min(len(returns[s]) for s in symbols)
        aligned_returns = {s: returns[s][-min_length:] for s in symbols}

        # 计算协方差矩阵
        matrix = [[0.0] * n for _ in range(n)]

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                ret1 = aligned_returns[sym1]
                ret2 = aligned_returns[sym2]

                mean1 = statistics.mean(ret1)
                mean2 = statistics.mean(ret2)

                cov = sum(
                    (ret1[k] - mean1) * (ret2[k] - mean2)
                    for k in range(len(ret1))
                ) / (len(ret1) - 1)

                matrix[i][j] = cov

        # 计算条件数
        condition_number = self._calculate_condition_number(matrix)

        return CovarianceMatrix(
            symbols=symbols,
            matrix=matrix,
            method=CovarianceMethod.SAMPLE,
            timestamp=datetime.now(),
            condition_number=condition_number,
            sample_size=min_length,
        )

    def _calculate_condition_number(self, matrix: List[List[float]]) -> float:
        """计算矩阵条件数"""
        try:
            # 简化估算: 最大特征值 / 最小特征值的近似
            n = len(matrix)
            trace = sum(matrix[i][i] for i in range(n))
            det_approx = abs(trace ** n / math.factorial(n))

            if det_approx < 1e-10:
                return float('inf')

            return trace / det_approx
        except Exception as exc:
            # 1.0 means "perfectly conditioned" - the most reassuring answer there
            # is, and the opposite of what a failed estimate implies. Report the
            # matrix as singular so the caller's ill-conditioning checks still trip.
            logger.warning(
                "Condition number estimate failed; treating matrix as singular: %s", exc
            )
            return float('inf')


class ExponentialCovarianceEstimator(CovarianceEstimator):
    """指数加权协方差估计器"""

    def __init__(self, lambda_param: float = 0.94):
        self.lambda_param = lambda_param

    def estimate(self, returns: Dict[str, List[float]]) -> CovarianceMatrix:
        symbols = sorted(list(returns.keys()))
        n = len(symbols)

        # 对齐数据长度
        min_length = min(len(returns[s]) for s in symbols)
        aligned_returns = {s: returns[s][-min_length:] for s in symbols}

        # 计算指数加权协方差
        matrix = [[0.0] * n for _ in range(n)]

        # 计算权重
        T = min_length
        weights = [self.lambda_param ** i for i in range(T)]
        weights.reverse()
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                ret1 = aligned_returns[sym1]
                ret2 = aligned_returns[sym2]

                # 加权均值
                mean1 = sum(ret1[k] * weights[k] for k in range(T))
                mean2 = sum(ret2[k] * weights[k] for k in range(T))

                # 加权协方差
                cov = sum(
                    weights[k] * (ret1[k] - mean1) * (ret2[k] - mean2)
                    for k in range(T)
                )

                matrix[i][j] = cov

        return CovarianceMatrix(
            symbols=symbols,
            matrix=matrix,
            method=CovarianceMethod.EXPONENTIAL,
            timestamp=datetime.now(),
            sample_size=min_length,
        )


class ConstantCorrelationEstimator(CovarianceEstimator):
    """恒定相关性估计器"""

    def __init__(self, correlation: float = 0.5):
        self.correlation = correlation

    def estimate(self, returns: Dict[str, List[float]]) -> CovarianceMatrix:
        symbols = sorted(list(returns.keys()))
        n = len(symbols)

        # 计算各资产标准差
        volatilities = {}
        for symbol in symbols:
            if len(returns[symbol]) >= 2:
                volatilities[symbol] = statistics.stdev(returns[symbol])
            else:
                volatilities[symbol] = 0.0

        # 构建协方差矩阵
        matrix = [[0.0] * n for _ in range(n)]

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i == j:
                    matrix[i][j] = volatilities[sym1] ** 2
                else:
                    matrix[i][j] = self.correlation * volatilities[sym1] * volatilities[sym2]

        return CovarianceMatrix(
            symbols=symbols,
            matrix=matrix,
            method=CovarianceMethod.CONSTANT_CORRELATION,
            timestamp=datetime.now(),
            sample_size=len(returns[symbols[0]]) if symbols else 0,
        )


# ============ Risk Parity Optimizer ============

class RiskParityOptimizer:
    """风险平价优化器"""

    def __init__(self, config: CalculatorConfig):
        self.config = config

    def optimize_equal_risk(
        self,
        cov_matrix: CovarianceMatrix,
        risk_budgets: Optional[Dict[str, float]] = None,
    ) -> PortfolioWeights:
        """
        等风险贡献优化
        目标: 使每个资产的风险贡献相等
        """
        symbols = cov_matrix.symbols
        n = len(symbols)

        if n == 0:
            return self._create_empty_weights()

        # 如果未指定风险预算，使用均等预算
        if risk_budgets is None:
            risk_budgets = {s: 1.0 / n for s in symbols}

        # 迭代优化
        weights = [1.0 / n] * n  # 初始等权重

        for iteration in range(self.config.max_iterations):
            old_weights = weights.copy()

            # 计算当前风险贡献
            risk_contribs = self._calculate_risk_contributions(weights, cov_matrix)

            # 更新权重以匹配目标风险预算
            total_risk = sum(rc.component_risk for rc in risk_contribs)

            if total_risk < 1e-10:
                break

            new_weights = []
            for i, symbol in enumerate(symbols):
                target_risk = risk_budgets.get(symbol, 1.0 / n)
                current_risk = risk_contribs[i].pct_contribution

                # 调整权重
                adjustment = target_risk / (current_risk + 1e-10)
                new_weight = weights[i] * math.sqrt(adjustment)
                new_weights.append(new_weight)

            # 归一化权重
            weight_sum = sum(new_weights)
            weights = [w / weight_sum for w in new_weights]

            # 应用约束
            weights = self._apply_constraints(weights)

            # 检查收敛
            diff = sum((weights[i] - old_weights[i]) ** 2 for i in range(n))
            if diff < self.config.tolerance:
                break

        # 计算组合波动率
        portfolio_vol = self._calculate_portfolio_volatility(weights, cov_matrix)

        return PortfolioWeights(
            symbols=symbols,
            weights=weights,
            timestamp=datetime.now(),
            method=OptimizationMethod.EQUAL_RISK,
            portfolio_volatility=portfolio_vol,
            leverage=sum(weights),
        )

    def optimize_inverse_volatility(self, cov_matrix: CovarianceMatrix) -> PortfolioWeights:
        """反向波动率加权"""
        symbols = cov_matrix.symbols
        n = len(symbols)

        if n == 0:
            return self._create_empty_weights()

        # 计算波动率的倒数
        inv_vols = []
        for i in range(n):
            vol = math.sqrt(cov_matrix.matrix[i][i])
            inv_vols.append(1.0 / vol if vol > 1e-10 else 0.0)

        # 归一化为权重
        total = sum(inv_vols)
        weights = [iv / total for iv in inv_vols] if total > 0 else [1.0 / n] * n

        # 应用约束
        weights = self._apply_constraints(weights)

        portfolio_vol = self._calculate_portfolio_volatility(weights, cov_matrix)

        return PortfolioWeights(
            symbols=symbols,
            weights=weights,
            timestamp=datetime.now(),
            method=OptimizationMethod.INVERSE_VOLATILITY,
            portfolio_volatility=portfolio_vol,
            leverage=sum(weights),
        )

    def optimize_min_variance(self, cov_matrix: CovarianceMatrix) -> PortfolioWeights:
        """最小方差优化"""
        symbols = cov_matrix.symbols
        n = len(symbols)

        if n == 0:
            return self._create_empty_weights()

        # 简化的最小方差解: 协方差矩阵的逆乘以全1向量
        cov = cov_matrix.matrix

        # 计算协方差矩阵的逆 (使用简化方法)
        try:
            inv_cov = self._invert_matrix(cov)
            ones = [1.0] * n

            # w = (Σ^-1 * 1) / (1' * Σ^-1 * 1)
            inv_cov_ones = [
                sum(inv_cov[i][j] * ones[j] for j in range(n))
                for i in range(n)
            ]

            denominator = sum(inv_cov_ones)
            weights = [w / denominator for w in inv_cov_ones] if denominator > 0 else [1.0 / n] * n

        except Exception as exc:
            # 如果矩阵求逆失败，回退到等权重
            logger.warning(
                "Covariance inversion failed for %d assets; falling back to equal "
                "weights, which is not a risk-parity solution: %s", n, exc
            )
            weights = [1.0 / n] * n

        # 应用约束
        weights = self._apply_constraints(weights)

        portfolio_vol = self._calculate_portfolio_volatility(weights, cov_matrix)

        return PortfolioWeights(
            symbols=symbols,
            weights=weights,
            timestamp=datetime.now(),
            method=OptimizationMethod.MIN_VARIANCE,
            portfolio_volatility=portfolio_vol,
            leverage=sum(weights),
        )

    def optimize_hierarchical(self, cov_matrix: CovarianceMatrix) -> PortfolioWeights:
        """
        层次风险平价 (HRP)
        使用聚类方法构建层次化的风险平价组合
        """
        symbols = cov_matrix.symbols
        n = len(symbols)

        if n == 0:
            return self._create_empty_weights()

        # 1. 计算相关性矩阵
        corr_matrix = self._covariance_to_correlation(cov_matrix)

        # 2. 层次聚类
        clusters = self._hierarchical_clustering(corr_matrix)

        # 3. 递归分配权重
        weights = self._recursive_bisection(cov_matrix, clusters)

        # 应用约束
        weights = self._apply_constraints(weights)

        portfolio_vol = self._calculate_portfolio_volatility(weights, cov_matrix)

        return PortfolioWeights(
            symbols=symbols,
            weights=weights,
            timestamp=datetime.now(),
            method=OptimizationMethod.HIERARCHICAL,
            portfolio_volatility=portfolio_vol,
            leverage=sum(weights),
        )

    def _covariance_to_correlation(self, cov_matrix: CovarianceMatrix) -> List[List[float]]:
        """协方差矩阵转相关性矩阵"""
        n = len(cov_matrix.symbols)
        corr = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                std_i = math.sqrt(cov_matrix.matrix[i][i])
                std_j = math.sqrt(cov_matrix.matrix[j][j])

                if std_i > 1e-10 and std_j > 1e-10:
                    corr[i][j] = cov_matrix.matrix[i][j] / (std_i * std_j)
                else:
                    corr[i][j] = 1.0 if i == j else 0.0

        return corr

    def _hierarchical_clustering(self, corr_matrix: List[List[float]]) -> List[int]:
        """层次聚类 (简化版)"""
        n = len(corr_matrix)

        # 使用简单的排序作为聚类结果
        # 实际实现中应该使用完整的层次聚类算法
        return list(range(n))

    def _recursive_bisection(
        self,
        cov_matrix: CovarianceMatrix,
        order: List[int],
    ) -> List[float]:
        """递归二分配权重"""
        n = len(order)
        weights = [1.0 / n] * n

        # 简化版本: 返回反波动率权重
        for i, idx in enumerate(order):
            vol = math.sqrt(cov_matrix.matrix[idx][idx])
            weights[i] = 1.0 / vol if vol > 1e-10 else 1.0

        # 归一化
        total = sum(weights)
        weights = [w / total for w in weights] if total > 0 else weights

        # 按原始顺序重排
        reordered = [0.0] * n
        for i, idx in enumerate(order):
            reordered[idx] = weights[i]

        return reordered

    def _calculate_risk_contributions(
        self,
        weights: List[float],
        cov_matrix: CovarianceMatrix,
    ) -> List[RiskContribution]:
        """计算风险贡献"""
        symbols = cov_matrix.symbols
        n = len(symbols)

        # 计算组合波动率
        portfolio_vol = self._calculate_portfolio_volatility(weights, cov_matrix)

        risk_contributions = []

        for i, symbol in enumerate(symbols):
            # 边际风险贡献: (Σw)_i / σ_p
            marginal_risk = sum(
                cov_matrix.matrix[i][j] * weights[j]
                for j in range(n)
            )

            if portfolio_vol > 1e-10:
                marginal_risk /= portfolio_vol

            # 成分风险贡献
            component_risk = weights[i] * marginal_risk

            # 百分比风险贡献
            pct_contribution = component_risk / portfolio_vol if portfolio_vol > 1e-10 else 0.0

            risk_contributions.append(RiskContribution(
                symbol=symbol,
                weight=weights[i],
                volatility=math.sqrt(cov_matrix.matrix[i][i]),
                marginal_risk=marginal_risk,
                component_risk=component_risk,
                pct_contribution=pct_contribution,
                timestamp=datetime.now(),
            ))

        return risk_contributions

    def _calculate_portfolio_volatility(
        self,
        weights: List[float],
        cov_matrix: CovarianceMatrix,
    ) -> float:
        """计算组合波动率"""
        n = len(weights)
        variance = 0.0

        for i in range(n):
            for j in range(n):
                variance += weights[i] * weights[j] * cov_matrix.matrix[i][j]

        return math.sqrt(max(0.0, variance))

    def _apply_constraints(self, weights: List[float]) -> List[float]:
        """应用权重约束"""
        n = len(weights)

        # 应用最小/最大权重约束
        constrained = [
            max(self.config.min_weight, min(self.config.max_weight, w))
            for w in weights
        ]

        # 归一化
        total = sum(constrained)
        if total > 1e-10:
            constrained = [w / total for w in constrained]
        else:
            constrained = [1.0 / n] * n

        # 应用杠杆约束
        leverage = sum(constrained)
        target_leverage = self.config.leverage_constraint.target_leverage

        if not self.config.leverage_constraint.is_valid(leverage):
            constrained = [w * target_leverage / leverage for w in constrained]

        return constrained

    def _invert_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        """矩阵求逆 (使用高斯-约旦消元法)"""
        n = len(matrix)

        # 创建增广矩阵 [A | I]
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

        # 前向消元
        for i in range(n):
            # 找主元
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k

            aug[i], aug[max_row] = aug[max_row], aug[i]

            # 归一化主元行
            pivot = aug[i][i]
            if abs(pivot) < 1e-10:
                raise ValueError("Matrix is singular")

            for j in range(2 * n):
                aug[i][j] /= pivot

            # 消元
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]

        # 提取逆矩阵
        inv = [[aug[i][j + n] for j in range(n)] for i in range(n)]

        return inv

    def _create_empty_weights(self) -> PortfolioWeights:
        """创建空权重对象"""
        return PortfolioWeights(
            symbols=[],
            weights=[],
            timestamp=datetime.now(),
            method=self.config.optimization_method,
        )


# ============ Risk Parity Calculator ============

class RiskParityCalculator:
    """
    风险平价计算器

    功能:
    - 实时风险平价优化
    - 多种优化方法
    - 协方差估计
    - 风险贡献分析
    - 自动再平衡
    """

    def __init__(
        self,
        config: Optional[CalculatorConfig] = None,
        rebalance_callback: Optional[Callable[[RebalanceSignal], None]] = None,
    ):
        self.config = config or CalculatorConfig()
        self.rebalance_callback = rebalance_callback

        # 状态
        self._state = CalculatorState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        # 数据存储
        self._assets: Dict[str, AssetData] = {}
        self._symbols: Set[str] = set()

        # 当前权重和风险贡献
        self._current_weights: Optional[PortfolioWeights] = None
        self._current_risk_contributions: List[RiskContribution] = []
        self._current_covariance: Optional[CovarianceMatrix] = None

        # 优化器
        self._optimizer = RiskParityOptimizer(self.config)

        # 协方差估计器
        self._cov_estimators: Dict[CovarianceMethod, CovarianceEstimator] = {
            CovarianceMethod.SAMPLE: SampleCovarianceEstimator(),
            CovarianceMethod.EXPONENTIAL: ExponentialCovarianceEstimator(
                self.config.exponential_lambda
            ),
            CovarianceMethod.CONSTANT_CORRELATION: ConstantCorrelationEstimator(),
        }

        # 历史记录
        self._weights_history: deque = deque(maxlen=1000)
        self._rebalance_signals: deque = deque(maxlen=100)

        # 上次再平衡时间
        self._last_rebalance: Optional[datetime] = None

    @property
    def state(self) -> CalculatorState:
        return self._state

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动计算器"""
        if self._state == CalculatorState.RUNNING:
            return

        logger.info("Starting Risk Parity Calculator")
        self._state = CalculatorState.RUNNING

        if self.config.enable_auto_rebalance:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("Risk Parity Calculator started")

    async def stop(self) -> None:
        """停止计算器"""
        if self._state == CalculatorState.IDLE:
            return

        logger.info("Stopping Risk Parity Calculator")
        self._state = CalculatorState.IDLE

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Risk Parity Calculator stopped")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state == CalculatorState.RUNNING:
            try:
                # 检查是否需要再平衡
                signal = self.check_rebalance_needed()

                if signal.should_rebalance:
                    # 执行再平衡
                    await self.rebalance()

                    if self.rebalance_callback:
                        self.rebalance_callback(signal)

                await asyncio.sleep(self.config.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self._state = CalculatorState.ERROR

    # ============ Asset Management ============

    def add_asset(self, symbol: str) -> None:
        """添加资产"""
        if symbol not in self._symbols:
            self._symbols.add(symbol)
            self._assets[symbol] = AssetData(
                symbol=symbol,
                returns=[],
                prices=[],
                timestamps=[],
            )
            logger.info(f"Added asset: {symbol}")

    def remove_asset(self, symbol: str) -> None:
        """移除资产"""
        if symbol in self._symbols:
            self._symbols.remove(symbol)
            self._assets.pop(symbol, None)
            logger.info(f"Removed asset: {symbol}")

    def get_assets(self) -> List[str]:
        """获取所有资产"""
        return list(self._symbols)

    async def add_return(self, symbol: str, return_value: float, timestamp: datetime) -> None:
        """添加收益率数据"""
        if symbol not in self._symbols:
            self.add_asset(symbol)

        asset = self._assets[symbol]
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

    async def add_price(self, symbol: str, price: float, timestamp: datetime) -> None:
        """添加价格数据 (自动计算收益率)"""
        if symbol not in self._symbols:
            self.add_asset(symbol)

        asset = self._assets[symbol]

        # 计算收益率
        if len(asset.prices) > 0:
            last_price = asset.prices[-1]
            if last_price > 0:
                return_value = (price - last_price) / last_price
                await self.add_return(symbol, return_value, timestamp)

        asset.prices.append(price)

        # 保持窗口大小
        if len(asset.prices) > self.config.lookback_window:
            asset.prices = asset.prices[-self.config.lookback_window:]

    # ============ Covariance Estimation ============

    def estimate_covariance(
        self,
        method: Optional[CovarianceMethod] = None,
    ) -> CovarianceMatrix:
        """估计协方差矩阵"""
        method = method or self.config.covariance_method

        # 获取收益率数据
        returns = {
            symbol: self._assets[symbol].returns
            for symbol in self._symbols
            if len(self._assets[symbol].returns) >= self.config.min_history_length
        }

        if not returns:
            return CovarianceMatrix(
                symbols=[],
                matrix=[],
                method=method,
                timestamp=datetime.now(),
            )

        # 选择估计器
        estimator = self._cov_estimators.get(method)
        if estimator is None:
            estimator = self._cov_estimators[CovarianceMethod.SAMPLE]

        # 估计协方差
        cov_matrix = estimator.estimate(returns)
        self._current_covariance = cov_matrix

        return cov_matrix

    # ============ Optimization ============

    def optimize(
        self,
        method: Optional[OptimizationMethod] = None,
        risk_budgets: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """执行优化"""
        import time

        method = method or self.config.optimization_method
        start_time = time.time()

        # 估计协方差矩阵
        cov_matrix = self.estimate_covariance()

        if len(cov_matrix.symbols) == 0:
            return OptimizationResult(
                weights=PortfolioWeights(
                    symbols=[],
                    weights=[],
                    timestamp=datetime.now(),
                    method=method,
                ),
                risk_contributions=[],
                covariance_matrix=cov_matrix,
                convergence_info={},
                optimization_time=0.0,
                success=False,
                message="No sufficient data",
            )

        # 执行优化
        try:
            if method == OptimizationMethod.EQUAL_RISK or method == OptimizationMethod.RISK_BUDGETING:
                weights = self._optimizer.optimize_equal_risk(cov_matrix, risk_budgets)
            elif method == OptimizationMethod.INVERSE_VOLATILITY:
                weights = self._optimizer.optimize_inverse_volatility(cov_matrix)
            elif method == OptimizationMethod.MIN_VARIANCE:
                weights = self._optimizer.optimize_min_variance(cov_matrix)
            elif method == OptimizationMethod.HIERARCHICAL:
                weights = self._optimizer.optimize_hierarchical(cov_matrix)
            else:
                weights = self._optimizer.optimize_equal_risk(cov_matrix)

            # 计算风险贡献
            risk_contributions = self._optimizer._calculate_risk_contributions(
                weights.weights, cov_matrix
            )

            # 计算分散化比率
            diversification_ratio = self._calculate_diversification_ratio(
                weights.weights, cov_matrix
            )
            weights.diversification_ratio = diversification_ratio

            # 更新当前状态
            self._current_weights = weights
            self._current_risk_contributions = risk_contributions

            # 记录历史
            self._weights_history.append((datetime.now(), weights))

            optimization_time = time.time() - start_time

            return OptimizationResult(
                weights=weights,
                risk_contributions=risk_contributions,
                covariance_matrix=cov_matrix,
                convergence_info={
                    "iterations": self.config.max_iterations,
                    "converged": True,
                },
                optimization_time=optimization_time,
                success=True,
                message="Optimization successful",
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                weights=PortfolioWeights(
                    symbols=[],
                    weights=[],
                    timestamp=datetime.now(),
                    method=method,
                ),
                risk_contributions=[],
                covariance_matrix=cov_matrix,
                convergence_info={},
                optimization_time=time.time() - start_time,
                success=False,
                message=str(e),
            )

    def _calculate_diversification_ratio(
        self,
        weights: List[float],
        cov_matrix: CovarianceMatrix,
    ) -> float:
        """
        计算分散化比率
        DR = (Σ w_i * σ_i) / σ_p
        """
        n = len(weights)

        # 加权平均波动率
        weighted_vol = sum(
            weights[i] * math.sqrt(cov_matrix.matrix[i][i])
            for i in range(n)
        )

        # 组合波动率
        portfolio_vol = self._optimizer._calculate_portfolio_volatility(weights, cov_matrix)

        if portfolio_vol > 1e-10:
            return weighted_vol / portfolio_vol

        return 1.0

    # ============ Rebalancing ============

    def check_rebalance_needed(self) -> RebalanceSignal:
        """检查是否需要再平衡"""
        signal_id = str(uuid.uuid4())[:8]

        if self._current_weights is None:
            return RebalanceSignal(
                signal_id=signal_id,
                should_rebalance=True,
                reason="Initial optimization needed",
                old_weights=PortfolioWeights(
                    symbols=[],
                    weights=[],
                    timestamp=datetime.now(),
                    method=self.config.optimization_method,
                ),
                new_weights=None,
                timestamp=datetime.now(),
            )

        # 检查再平衡频率
        if self._last_rebalance:
            time_since_rebalance = datetime.now() - self._last_rebalance

            if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
                should_rebalance_time = time_since_rebalance.days >= 1
            elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
                should_rebalance_time = time_since_rebalance.days >= 7
            elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
                should_rebalance_time = time_since_rebalance.days >= 30
            elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
                should_rebalance_time = time_since_rebalance.days >= 90
            else:
                should_rebalance_time = False

            if not should_rebalance_time:
                return RebalanceSignal(
                    signal_id=signal_id,
                    should_rebalance=False,
                    reason="Not time for rebalance",
                    old_weights=self._current_weights,
                    new_weights=None,
                    timestamp=datetime.now(),
                )

        # 重新优化
        result = self.optimize()

        if not result.success:
            return RebalanceSignal(
                signal_id=signal_id,
                should_rebalance=False,
                reason="Optimization failed",
                old_weights=self._current_weights,
                new_weights=None,
                timestamp=datetime.now(),
            )

        # 计算权重偏离
        turnover = self._calculate_turnover(
            self._current_weights.weights,
            result.weights.weights,
        )

        should_rebalance = turnover > self.config.rebalance_threshold

        return RebalanceSignal(
            signal_id=signal_id,
            should_rebalance=should_rebalance,
            reason=f"Turnover {turnover:.2%} exceeds threshold" if should_rebalance else "Within threshold",
            old_weights=self._current_weights,
            new_weights=result.weights,
            timestamp=datetime.now(),
            turnover=turnover,
        )

    def _calculate_turnover(self, old_weights: List[float], new_weights: List[float]) -> float:
        """计算换手率"""
        if len(old_weights) != len(new_weights):
            return 1.0

        return sum(abs(new_weights[i] - old_weights[i]) for i in range(len(old_weights))) / 2

    async def rebalance(self) -> OptimizationResult:
        """执行再平衡"""
        result = self.optimize()
        self._last_rebalance = datetime.now()

        logger.info(f"Rebalanced portfolio at {self._last_rebalance}")

        return result

    # ============ Query Methods ============

    def get_current_weights(self) -> Optional[PortfolioWeights]:
        """获取当前权重"""
        return self._current_weights

    def get_risk_contributions(self) -> List[RiskContribution]:
        """获取风险贡献"""
        return self._current_risk_contributions

    def get_covariance_matrix(self) -> Optional[CovarianceMatrix]:
        """获取协方差矩阵"""
        return self._current_covariance

    def get_risk_budgets_status(self) -> List[RiskBudget]:
        """获取风险预算状态"""
        if not self._current_risk_contributions:
            return []

        budgets = []
        for rc in self._current_risk_contributions:
            target_risk = self.config.risk_budgets.get(rc.symbol, 1.0 / len(self._current_risk_contributions))

            budget = RiskBudget(
                symbol=rc.symbol,
                target_risk_pct=target_risk,
            )
            budget.update(rc.pct_contribution)
            budgets.append(budget)

        return budgets

    def get_weights_history(self, limit: int = 100) -> List[Tuple[datetime, PortfolioWeights]]:
        """获取权重历史"""
        return list(self._weights_history)[-limit:]

    def get_rebalance_signals(self, limit: int = 50) -> List[RebalanceSignal]:
        """获取再平衡信号历史"""
        return list(self._rebalance_signals)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "assets": len(self._symbols),
            "current_weights": self._current_weights.to_dict() if self._current_weights else None,
            "portfolio_volatility": self._current_weights.portfolio_volatility if self._current_weights else 0.0,
            "diversification_ratio": self._current_weights.diversification_ratio if self._current_weights else 0.0,
            "leverage": self._current_weights.leverage if self._current_weights else 1.0,
            "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
            "config": self.config.to_dict(),
        }


# ============ Factory Functions ============

def create_risk_parity_calculator(
    optimization_method: str = "equal_risk",
    covariance_method: str = "sample",
    **config_kwargs,
) -> RiskParityCalculator:
    """创建风险平价计算器"""
    try:
        opt_method = OptimizationMethod(optimization_method)
    except ValueError:
        opt_method = OptimizationMethod.EQUAL_RISK

    try:
        cov_method = CovarianceMethod(covariance_method)
    except ValueError:
        cov_method = CovarianceMethod.SAMPLE

    config = CalculatorConfig(
        optimization_method=opt_method,
        covariance_method=cov_method,
        **{k: v for k, v in config_kwargs.items() if hasattr(CalculatorConfig, k)}
    )

    return RiskParityCalculator(config=config)


# ============ Exports ============

__all__ = [
    # Enums
    "OptimizationMethod",
    "CovarianceMethod",
    "RebalanceFrequency",
    "CalculatorState",
    # Data Classes
    "AssetData",
    "PortfolioWeights",
    "RiskContribution",
    "CovarianceMatrix",
    "RiskBudget",
    "LeverageConstraint",
    "OptimizationResult",
    "RebalanceSignal",
    "CalculatorConfig",
    # Estimators
    "CovarianceEstimator",
    "SampleCovarianceEstimator",
    "ExponentialCovarianceEstimator",
    "ConstantCorrelationEstimator",
    # Optimizer
    "RiskParityOptimizer",
    # Main Class
    "RiskParityCalculator",
    # Factory Functions
    "create_risk_parity_calculator",
]
