"""
因子暴露分析模块
Factor Exposure Analyzer Module

提供:
- 多因子暴露计算（市场、规模、价值、动量、波动率、质量）
- 实时因子归因分析
- 因子贡献度追踪
- 因子风险预算
- 因子中性化检测
- 隐含因子暴露
- 因子暴露预测
- 风格漂移检测

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Deque
import numpy as np
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ============ Enums ============

class FactorCategory(Enum):
    """因子类别"""
    MARKET = "market"           # 市场因子
    SIZE = "size"               # 规模因子
    VALUE = "value"             # 价值因子
    MOMENTUM = "momentum"       # 动量因子
    VOLATILITY = "volatility"   # 波动率因子
    QUALITY = "quality"         # 质量因子
    LIQUIDITY = "liquidity"     # 流动性因子
    GROWTH = "growth"           # 成长因子
    CUSTOM = "custom"           # 自定义因子


class ExposureMethod(Enum):
    """暴露计算方法"""
    REGRESSION = "regression"           # 回归法
    FUNDAMENTAL = "fundamental"         # 基本面法
    STATISTICAL = "statistical"         # 统计法
    HYBRID = "hybrid"                   # 混合法
    IMPLIED = "implied"                 # 隐含法


class NeutralizationLevel(Enum):
    """中性化水平"""
    NONE = "none"               # 不中性化
    PARTIAL = "partial"         # 部分中性化
    FULL = "full"              # 完全中性化
    DOLLAR = "dollar"          # 金额中性
    BETA = "beta"              # Beta中性


class DriftSeverity(Enum):
    """漂移严重程度"""
    NONE = "none"              # 无漂移
    LOW = "low"                # 低度漂移
    MODERATE = "moderate"      # 中度漂移
    HIGH = "high"              # 高度漂移
    CRITICAL = "critical"      # 严重漂移


# ============ Data Classes ============

@dataclass
class FactorExposure:
    """因子暴露"""
    factor_name: str                          # 因子名称
    category: FactorCategory                  # 因子类别
    exposure: float                           # 暴露值
    standardized_exposure: float              # 标准化暴露
    percentile: float                         # 百分位数
    z_score: float                            # Z分数
    active_exposure: float = 0.0              # 主动暴露（相对基准）
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # 置信区间
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'category': self.category.value,
            'exposure': self.exposure,
            'standardized_exposure': self.standardized_exposure,
            'percentile': self.percentile,
            'z_score': self.z_score,
            'active_exposure': self.active_exposure,
            'confidence_interval': self.confidence_interval,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class FactorContribution:
    """因子贡献"""
    factor_name: str                    # 因子名称
    exposure: float                     # 暴露
    factor_return: float                # 因子收益
    contribution: float                 # 贡献值
    contribution_pct: float             # 贡献占比
    cumulative_contribution: float = 0.0  # 累积贡献
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'exposure': self.exposure,
            'factor_return': self.factor_return,
            'contribution': self.contribution,
            'contribution_pct': self.contribution_pct,
            'cumulative_contribution': self.cumulative_contribution,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class RiskBudget:
    """因子风险预算"""
    factor_name: str                    # 因子名称
    target_exposure: float              # 目标暴露
    current_exposure: float             # 当前暴露
    exposure_limit: Tuple[float, float]  # 暴露限制（下限，上限）
    risk_contribution: float            # 风险贡献
    risk_budget: float                  # 风险预算
    budget_utilization: float           # 预算使用率
    is_breached: bool                   # 是否违反限制
    breach_amount: float = 0.0          # 违反量
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'target_exposure': self.target_exposure,
            'current_exposure': self.current_exposure,
            'exposure_limit': self.exposure_limit,
            'risk_contribution': self.risk_contribution,
            'risk_budget': self.risk_budget,
            'budget_utilization': self.budget_utilization,
            'is_breached': self.is_breached,
            'breach_amount': self.breach_amount,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class NeutralizationResult:
    """中性化结果"""
    factor_name: str                           # 因子名称
    original_exposure: float                   # 原始暴露
    target_level: NeutralizationLevel          # 目标中性化水平
    neutralized_exposure: float                # 中性化后暴露
    adjustment_required: float                 # 需要调整量
    current_neutralization_pct: float          # 当前中性化百分比
    is_neutral: bool                           # 是否中性
    neutralization_cost: float = 0.0           # 中性化成本
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'original_exposure': self.original_exposure,
            'target_level': self.target_level.value,
            'neutralized_exposure': self.neutralized_exposure,
            'adjustment_required': self.adjustment_required,
            'current_neutralization_pct': self.current_neutralization_pct,
            'is_neutral': self.is_neutral,
            'neutralization_cost': self.neutralization_cost,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class ImpliedExposure:
    """隐含因子暴露"""
    factor_name: str                    # 因子名称
    implied_exposure: float             # 隐含暴露
    explicit_exposure: float            # 显性暴露
    implicit_exposure: float            # 隐性暴露
    correlation_based: float            # 基于相关性的暴露
    residual_exposure: float            # 残差暴露
    confidence: float                   # 置信度
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'implied_exposure': self.implied_exposure,
            'explicit_exposure': self.explicit_exposure,
            'implicit_exposure': self.implicit_exposure,
            'correlation_based': self.correlation_based,
            'residual_exposure': self.residual_exposure,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class ExposureForecast:
    """因子暴露预测"""
    factor_name: str                           # 因子名称
    current_exposure: float                    # 当前暴露
    forecast_exposure: float                   # 预测暴露
    forecast_horizon: int                      # 预测期限（天）
    confidence_interval: Tuple[float, float]   # 置信区间
    expected_change: float                     # 预期变化
    trend_strength: float                      # 趋势强度
    volatility: float                          # 波动率
    prediction_error: float = 0.0              # 预测误差
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'current_exposure': self.current_exposure,
            'forecast_exposure': self.forecast_exposure,
            'forecast_horizon': self.forecast_horizon,
            'confidence_interval': self.confidence_interval,
            'expected_change': self.expected_change,
            'trend_strength': self.trend_strength,
            'volatility': self.volatility,
            'prediction_error': self.prediction_error,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class StyleDrift:
    """风格漂移"""
    factor_name: str                    # 因子名称
    baseline_exposure: float            # 基线暴露
    current_exposure: float             # 当前暴露
    drift_amount: float                 # 漂移量
    drift_pct: float                    # 漂移百分比
    severity: DriftSeverity             # 严重程度
    drift_velocity: float               # 漂移速度
    time_in_drift: int                  # 漂移持续时间（天）
    is_systematic: bool                 # 是否系统性
    reversion_probability: float = 0.5  # 回归概率
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'baseline_exposure': self.baseline_exposure,
            'current_exposure': self.current_exposure,
            'drift_amount': self.drift_amount,
            'drift_pct': self.drift_pct,
            'severity': self.severity.value,
            'drift_velocity': self.drift_velocity,
            'time_in_drift': self.time_in_drift,
            'is_systematic': self.is_systematic,
            'reversion_probability': self.reversion_probability,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class PortfolioFactorProfile:
    """组合因子画像"""
    exposures: Dict[str, FactorExposure]              # 因子暴露
    contributions: Dict[str, FactorContribution]      # 因子贡献
    risk_budgets: Dict[str, RiskBudget]              # 风险预算
    neutralization: Dict[str, NeutralizationResult]  # 中性化结果
    implied: Dict[str, ImpliedExposure]              # 隐含暴露
    forecasts: Dict[str, ExposureForecast]           # 暴露预测
    drifts: Dict[str, StyleDrift]                    # 风格漂移
    total_risk: float = 0.0                          # 总风险
    diversification_score: float = 0.0               # 分散化得分
    factor_concentration: float = 0.0                # 因子集中度
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'exposures': {k: v.to_dict() for k, v in self.exposures.items()},
            'contributions': {k: v.to_dict() for k, v in self.contributions.items()},
            'risk_budgets': {k: v.to_dict() for k, v in self.risk_budgets.items()},
            'neutralization': {k: v.to_dict() for k, v in self.neutralization.items()},
            'implied': {k: v.to_dict() for k, v in self.implied.items()},
            'forecasts': {k: v.to_dict() for k, v in self.forecasts.items()},
            'drifts': {k: v.to_dict() for k, v in self.drifts.items()},
            'total_risk': self.total_risk,
            'diversification_score': self.diversification_score,
            'factor_concentration': self.factor_concentration,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class AnalyzerConfig:
    """分析器配置"""
    method: ExposureMethod = ExposureMethod.HYBRID
    factors: List[str] = field(default_factory=lambda: [
        'market', 'size', 'value', 'momentum', 'volatility', 'quality'
    ])
    lookback_period: int = 252                        # 回溯期（天）
    neutralization_threshold: float = 0.05            # 中性化阈值
    drift_threshold: float = 0.1                      # 漂移阈值
    confidence_level: float = 0.95                    # 置信水平
    rebalance_frequency: int = 20                     # 再平衡频率（天）
    use_robust_estimation: bool = True                # 使用稳健估计
    min_observations: int = 60                        # 最小观测数
    max_factor_exposure: float = 3.0                  # 最大因子暴露
    risk_budget_method: str = 'equal'                 # 风险预算方法


# ============ Factor Exposure Calculator ============

class FactorExposureCalculator:
    """
    因子暴露计算器

    计算组合对各因子的暴露度，支持多种计算方法：
    - 回归法：通过回归分析计算因子暴露
    - 基本面法：基于基本面数据直接计算
    - 统计法：基于统计特征计算
    - 混合法：综合多种方法
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self._factor_returns_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.lookback_period)
        )
        self._portfolio_returns_history: Deque[float] = deque(
            maxlen=self.config.lookback_period
        )

    def calculate_exposures(
        self,
        weights: Dict[str, float],
        asset_factors: Dict[str, Dict[str, float]],
        benchmark_weights: Optional[Dict[str, float]] = None,
        benchmark_factors: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, FactorExposure]:
        """
        计算因子暴露

        Args:
            weights: 资产权重字典 {symbol: weight}
            asset_factors: 资产因子数据 {symbol: {factor: value}}
            benchmark_weights: 基准权重（可选）
            benchmark_factors: 基准因子数据（可选）

        Returns:
            因子暴露字典 {factor_name: FactorExposure}
        """
        exposures = {}

        # 计算组合级别的因子暴露（权重加权）
        factor_values = defaultdict(float)
        total_weight = sum(weights.values())

        if total_weight == 0:
            logger.warning("Total weight is zero, cannot calculate exposures")
            return exposures

        # 加权求和计算暴露
        for symbol, weight in weights.items():
            if symbol not in asset_factors:
                continue

            normalized_weight = weight / total_weight
            for factor_name, factor_value in asset_factors[symbol].items():
                factor_values[factor_name] += normalized_weight * factor_value

        # 计算基准暴露（如果提供）
        benchmark_exposures = {}
        if benchmark_weights and benchmark_factors:
            benchmark_total = sum(benchmark_weights.values())
            if benchmark_total > 0:
                for symbol, weight in benchmark_weights.items():
                    if symbol not in benchmark_factors:
                        continue
                    normalized_weight = weight / benchmark_total
                    for factor_name, factor_value in benchmark_factors[symbol].items():
                        if factor_name not in benchmark_exposures:
                            benchmark_exposures[factor_name] = 0.0
                        benchmark_exposures[factor_name] += normalized_weight * factor_value

        # 计算统计量
        all_factor_values = defaultdict(list)
        for asset_factor_dict in asset_factors.values():
            for factor_name, factor_value in asset_factor_dict.items():
                all_factor_values[factor_name].append(factor_value)

        # 创建FactorExposure对象
        for factor_name, exposure_value in factor_values.items():
            factor_data = all_factor_values.get(factor_name, [])

            if len(factor_data) > 0:
                mean = np.mean(factor_data)
                std = np.std(factor_data) if len(factor_data) > 1 else 1.0

                # 标准化暴露（Z-score）
                z_score = (exposure_value - mean) / std if std > 0 else 0.0

                # 百分位数
                if len(factor_data) > 1:
                    percentile = stats.percentileofscore(factor_data, exposure_value) / 100.0
                else:
                    percentile = 0.5

                # 置信区间（使用t分布）
                if len(factor_data) > 2:
                    confidence_level = self.config.confidence_level
                    t_stat = stats.t.ppf((1 + confidence_level) / 2, len(factor_data) - 1)
                    margin = t_stat * std / np.sqrt(len(factor_data))
                    ci = (exposure_value - margin, exposure_value + margin)
                else:
                    ci = (exposure_value, exposure_value)
            else:
                z_score = 0.0
                percentile = 0.5
                ci = (exposure_value, exposure_value)

            # 主动暴露
            benchmark_exp = benchmark_exposures.get(factor_name, 0.0)
            active_exposure = exposure_value - benchmark_exp

            # 确定因子类别
            category = self._determine_category(factor_name)

            exposures[factor_name] = FactorExposure(
                factor_name=factor_name,
                category=category,
                exposure=exposure_value,
                standardized_exposure=z_score,
                percentile=percentile,
                z_score=z_score,
                active_exposure=active_exposure,
                confidence_interval=ci,
            )

        return exposures

    def calculate_regression_exposures(
        self,
        portfolio_returns: np.ndarray,
        factor_returns: Dict[str, np.ndarray],
    ) -> Dict[str, FactorExposure]:
        """
        基于回归法计算因子暴露

        使用多元线性回归: r_p = α + β₁f₁ + β₂f₂ + ... + βₙfₙ + ε

        Args:
            portfolio_returns: 组合收益序列
            factor_returns: 因子收益序列字典

        Returns:
            因子暴露字典
        """
        if len(portfolio_returns) < self.config.min_observations:
            logger.warning(f"Insufficient data for regression: {len(portfolio_returns)} < {self.config.min_observations}")
            return {}

        exposures = {}

        # 构建因子矩阵
        factor_names = sorted(factor_returns.keys())
        n_obs = len(portfolio_returns)
        X = np.column_stack([factor_returns[f][:n_obs] for f in factor_names])
        y = portfolio_returns[:n_obs]

        # 添加截距项
        X = np.column_stack([np.ones(n_obs), X])

        try:
            if self.config.use_robust_estimation:
                # 使用稳健回归（加权最小二乘）
                from scipy.stats import t

                # 初始OLS估计
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta

                # Huber权重
                median_abs_residual = np.median(np.abs(residuals))
                if median_abs_residual > 0:
                    k = 1.345  # Huber常数
                    normalized_residuals = residuals / median_abs_residual
                    weights = np.where(
                        np.abs(normalized_residuals) <= k,
                        1.0,
                        k / np.abs(normalized_residuals)
                    )

                    # 加权最小二乘
                    W = np.diag(weights)
                    beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
            else:
                # 普通最小二乘
                beta = np.linalg.lstsq(X, y, rcond=None)[0]

            # 计算标准误和置信区间
            residuals = y - X @ beta
            mse = np.sum(residuals ** 2) / (n_obs - len(beta))
            var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
            se_beta = np.sqrt(var_beta)

            t_stat = stats.t.ppf((1 + self.config.confidence_level) / 2, n_obs - len(beta))

            # 提取因子暴露（跳过截距）
            for i, factor_name in enumerate(factor_names):
                beta_i = beta[i + 1]  # +1 跳过截距
                se_i = se_beta[i + 1]
                ci = (beta_i - t_stat * se_i, beta_i + t_stat * se_i)

                category = self._determine_category(factor_name)

                exposures[factor_name] = FactorExposure(
                    factor_name=factor_name,
                    category=category,
                    exposure=beta_i,
                    standardized_exposure=beta_i / se_i if se_i > 0 else 0.0,
                    percentile=0.5,  # 回归系数无法直接计算百分位
                    z_score=beta_i / se_i if se_i > 0 else 0.0,
                    active_exposure=beta_i,  # 回归系数即为相对无暴露的主动暴露
                    confidence_interval=ci,
                )

        except np.linalg.LinAlgError as e:
            logger.error(f"Regression failed: {e}")
            return {}

        return exposures

    def update_history(
        self,
        portfolio_return: float,
        factor_returns: Dict[str, float],
    ):
        """
        更新历史数据

        Args:
            portfolio_return: 组合收益
            factor_returns: 因子收益字典
        """
        self._portfolio_returns_history.append(portfolio_return)

        for factor_name, factor_return in factor_returns.items():
            self._factor_returns_history[factor_name].append(factor_return)

    def _determine_category(self, factor_name: str) -> FactorCategory:
        """确定因子类别"""
        factor_name_lower = factor_name.lower()

        # 优先匹配更具体的关键词，避免误判
        # 需要按照优先级顺序判断

        # Value因子：book, pe, pb, btm等
        if 'book' in factor_name_lower or 'pe' in factor_name_lower or 'pb' in factor_name_lower or 'btm' in factor_name_lower:
            return FactorCategory.VALUE
        # Size因子：size, cap（但不包括单独的market）
        elif 'size' in factor_name_lower or ('cap' in factor_name_lower and 'market' not in factor_name_lower):
            return FactorCategory.SIZE
        # Market因子：market, beta
        # 但如果同时包含cap（如market_cap），应该是SIZE
        elif 'market' in factor_name_lower or 'beta' in factor_name_lower:
            if 'cap' in factor_name_lower:
                return FactorCategory.SIZE
            return FactorCategory.MARKET
        # Value因子：包含value关键词
        elif 'value' in factor_name_lower:
            return FactorCategory.VALUE
        # Momentum因子
        elif 'momentum' in factor_name_lower or 'trend' in factor_name_lower:
            return FactorCategory.MOMENTUM
        # Volatility因子
        elif 'vol' in factor_name_lower or 'volatility' in factor_name_lower:
            return FactorCategory.VOLATILITY
        # Quality因子
        elif 'quality' in factor_name_lower or 'profit' in factor_name_lower or 'roe' in factor_name_lower:
            return FactorCategory.QUALITY
        # Liquidity因子
        elif 'liquid' in factor_name_lower or 'volume' in factor_name_lower:
            return FactorCategory.LIQUIDITY
        # Growth因子
        elif 'growth' in factor_name_lower or 'earnings' in factor_name_lower:
            return FactorCategory.GROWTH
        else:
            return FactorCategory.CUSTOM


# ============ Factor Attribution Tracker ============

class FactorAttributionTracker:
    """
    因子归因追踪器

    追踪因子对组合收益的贡献
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self._contribution_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.lookback_period)
        )

    def calculate_contributions(
        self,
        exposures: Dict[str, FactorExposure],
        factor_returns: Dict[str, float],
        total_return: float,
    ) -> Dict[str, FactorContribution]:
        """
        计算因子贡献

        贡献 = 因子暴露 × 因子收益

        Args:
            exposures: 因子暴露字典
            factor_returns: 因子收益字典
            total_return: 总收益

        Returns:
            因子贡献字典
        """
        contributions = {}
        total_factor_contribution = 0.0

        # 计算各因子贡献
        for factor_name, exposure_obj in exposures.items():
            factor_return = factor_returns.get(factor_name, 0.0)
            contribution = exposure_obj.exposure * factor_return

            total_factor_contribution += contribution

            # 更新历史
            self._contribution_history[factor_name].append(contribution)

            contributions[factor_name] = FactorContribution(
                factor_name=factor_name,
                exposure=exposure_obj.exposure,
                factor_return=factor_return,
                contribution=contribution,
                contribution_pct=0.0,  # 稍后计算
                cumulative_contribution=sum(self._contribution_history[factor_name]),
            )

        # 计算贡献占比
        if abs(total_factor_contribution) > 1e-10:
            for contrib in contributions.values():
                contrib.contribution_pct = contrib.contribution / total_factor_contribution
        else:
            for contrib in contributions.values():
                contrib.contribution_pct = 0.0

        return contributions

    def decompose_return(
        self,
        total_return: float,
        contributions: Dict[str, FactorContribution],
    ) -> Tuple[float, float]:
        """
        分解收益

        总收益 = 因子收益 + 特异性收益

        Args:
            total_return: 总收益
            contributions: 因子贡献字典

        Returns:
            (因子收益总和, 特异性收益)
        """
        factor_return = sum(c.contribution for c in contributions.values())
        specific_return = total_return - factor_return

        return factor_return, specific_return


# ============ Factor Risk Budget Manager ============

class FactorRiskBudgetManager:
    """
    因子风险预算管理器

    管理各因子的风险预算和暴露限制
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self._risk_budgets: Dict[str, float] = {}
        self._exposure_limits: Dict[str, Tuple[float, float]] = {}

    def set_risk_budget(
        self,
        factor_name: str,
        budget: float,
        exposure_limit: Optional[Tuple[float, float]] = None,
    ):
        """
        设置因子风险预算

        Args:
            factor_name: 因子名称
            budget: 风险预算（占总风险的比例）
            exposure_limit: 暴露限制（下限，上限）
        """
        self._risk_budgets[factor_name] = budget

        if exposure_limit:
            self._exposure_limits[factor_name] = exposure_limit
        else:
            # 默认限制
            max_exp = self.config.max_factor_exposure
            self._exposure_limits[factor_name] = (-max_exp, max_exp)

    def set_equal_risk_budgets(self, factor_names: List[str]):
        """设置等风险预算"""
        n_factors = len(factor_names)
        if n_factors == 0:
            return

        equal_budget = 1.0 / n_factors
        for factor_name in factor_names:
            self.set_risk_budget(factor_name, equal_budget)

    def calculate_risk_budgets(
        self,
        exposures: Dict[str, FactorExposure],
        factor_covariance: np.ndarray,
        factor_names: List[str],
    ) -> Dict[str, RiskBudget]:
        """
        计算风险预算状态

        Args:
            exposures: 因子暴露字典
            factor_covariance: 因子协方差矩阵
            factor_names: 因子名称列表（与协方差矩阵对应）

        Returns:
            风险预算状态字典
        """
        budgets = {}

        # 构建暴露向量
        exposure_vector = np.array([
            exposures.get(f, FactorExposure(f, FactorCategory.CUSTOM, 0.0, 0.0, 0.5, 0.0)).exposure
            for f in factor_names
        ])

        # 组合风险
        portfolio_variance = exposure_vector @ factor_covariance @ exposure_vector
        portfolio_risk = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0.0

        # 边际风险贡献
        if portfolio_risk > 0:
            marginal_risk = (factor_covariance @ exposure_vector) / portfolio_risk

            # 风险贡献
            risk_contributions = exposure_vector * marginal_risk
        else:
            marginal_risk = np.zeros(len(factor_names))
            risk_contributions = np.zeros(len(factor_names))

        total_risk_contribution = np.sum(np.abs(risk_contributions))

        # 创建RiskBudget对象
        for i, factor_name in enumerate(factor_names):
            exposure_obj = exposures.get(factor_name)
            if not exposure_obj:
                continue

            current_exposure = exposure_obj.exposure
            target_exposure = 0.0  # 可以根据目标设置

            risk_contribution = risk_contributions[i]
            risk_budget = self._risk_budgets.get(factor_name, 1.0 / len(factor_names))

            # 预算使用率
            if risk_budget > 0 and total_risk_contribution > 0:
                budget_utilization = abs(risk_contribution) / total_risk_contribution / risk_budget
            else:
                budget_utilization = 0.0

            # 检查限制违反
            limit = self._exposure_limits.get(
                factor_name,
                (-self.config.max_factor_exposure, self.config.max_factor_exposure)
            )

            is_breached = current_exposure < limit[0] or current_exposure > limit[1]
            breach_amount = 0.0

            if current_exposure < limit[0]:
                breach_amount = limit[0] - current_exposure
            elif current_exposure > limit[1]:
                breach_amount = current_exposure - limit[1]

            budgets[factor_name] = RiskBudget(
                factor_name=factor_name,
                target_exposure=target_exposure,
                current_exposure=current_exposure,
                exposure_limit=limit,
                risk_contribution=risk_contribution,
                risk_budget=risk_budget,
                budget_utilization=budget_utilization,
                is_breached=is_breached,
                breach_amount=breach_amount,
            )

        return budgets


# ============ Factor Neutralization Detector ============

class FactorNeutralizationDetector:
    """
    因子中性化检测器

    检测组合的因子中性化程度
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()

    def detect_neutralization(
        self,
        exposures: Dict[str, FactorExposure],
        target_levels: Optional[Dict[str, NeutralizationLevel]] = None,
    ) -> Dict[str, NeutralizationResult]:
        """
        检测因子中性化

        Args:
            exposures: 因子暴露字典
            target_levels: 目标中性化水平字典

        Returns:
            中性化结果字典
        """
        results = {}

        if target_levels is None:
            target_levels = {f: NeutralizationLevel.PARTIAL for f in exposures.keys()}

        for factor_name, exposure_obj in exposures.items():
            target_level = target_levels.get(factor_name, NeutralizationLevel.PARTIAL)

            original_exposure = exposure_obj.exposure

            # 根据目标水平确定中性化阈值
            if target_level == NeutralizationLevel.NONE:
                neutralization_threshold = float('inf')
                neutralized_exposure = original_exposure
            elif target_level == NeutralizationLevel.PARTIAL:
                neutralization_threshold = self.config.neutralization_threshold
                neutralized_exposure = self._partial_neutralize(original_exposure)
            elif target_level == NeutralizationLevel.FULL:
                neutralization_threshold = self.config.neutralization_threshold / 2
                neutralized_exposure = 0.0
            elif target_level == NeutralizationLevel.DOLLAR:
                # Dollar neutral: 多头和空头金额相等
                neutralization_threshold = 0.01
                neutralized_exposure = 0.0
            elif target_level == NeutralizationLevel.BETA:
                # Beta neutral: 市场beta为0
                if exposure_obj.category == FactorCategory.MARKET:
                    neutralization_threshold = 0.05
                    neutralized_exposure = 0.0
                else:
                    neutralization_threshold = self.config.neutralization_threshold
                    neutralized_exposure = original_exposure
            else:
                neutralization_threshold = self.config.neutralization_threshold
                neutralized_exposure = original_exposure

            # 调整量
            adjustment_required = neutralized_exposure - original_exposure

            # 当前中性化百分比
            if abs(original_exposure) > 1e-10:
                current_neutralization_pct = 1.0 - abs(neutralized_exposure) / abs(original_exposure)
            else:
                current_neutralization_pct = 1.0

            # 是否中性
            is_neutral = abs(original_exposure) <= neutralization_threshold

            # 估计中性化成本（基于市场冲击和交易成本）
            neutralization_cost = self._estimate_neutralization_cost(
                abs(adjustment_required),
                exposure_obj.category
            )

            results[factor_name] = NeutralizationResult(
                factor_name=factor_name,
                original_exposure=original_exposure,
                target_level=target_level,
                neutralized_exposure=neutralized_exposure,
                adjustment_required=adjustment_required,
                current_neutralization_pct=current_neutralization_pct,
                is_neutral=is_neutral,
                neutralization_cost=neutralization_cost,
            )

        return results

    def _partial_neutralize(self, exposure: float) -> float:
        """部分中性化：将暴露降低50%"""
        return exposure * 0.5

    def _estimate_neutralization_cost(
        self,
        adjustment_amount: float,
        category: FactorCategory
    ) -> float:
        """
        估计中性化成本

        成本 = 调整量 × (交易成本率 + 市场冲击率)
        """
        # 不同因子类别的成本率不同
        base_cost_rates = {
            FactorCategory.MARKET: 0.001,      # 0.1%
            FactorCategory.SIZE: 0.002,        # 0.2%
            FactorCategory.VALUE: 0.0015,      # 0.15%
            FactorCategory.MOMENTUM: 0.0025,   # 0.25%
            FactorCategory.VOLATILITY: 0.002,  # 0.2%
            FactorCategory.QUALITY: 0.0015,    # 0.15%
            FactorCategory.LIQUIDITY: 0.003,   # 0.3%
            FactorCategory.GROWTH: 0.002,      # 0.2%
            FactorCategory.CUSTOM: 0.002,      # 0.2%
        }

        cost_rate = base_cost_rates.get(category, 0.002)
        return adjustment_amount * cost_rate


# ============ Implied Exposure Analyzer ============

class ImpliedExposureAnalyzer:
    """
    隐含因子暴露分析器

    分析隐含的因子暴露，包括：
    - 显性暴露：直接持仓带来的暴露
    - 隐性暴露：通过相关性间接产生的暴露
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()

    def analyze_implied_exposures(
        self,
        explicit_exposures: Dict[str, FactorExposure],
        factor_correlation: np.ndarray,
        factor_names: List[str],
        returns_history: Optional[np.ndarray] = None,
    ) -> Dict[str, ImpliedExposure]:
        """
        分析隐含因子暴露

        Args:
            explicit_exposures: 显性暴露字典
            factor_correlation: 因子相关性矩阵
            factor_names: 因子名称列表
            returns_history: 历史收益序列（用于计算残差暴露）

        Returns:
            隐含暴露字典
        """
        implied = {}

        # 构建显性暴露向量
        explicit_vector = np.array([
            explicit_exposures.get(f, FactorExposure(f, FactorCategory.CUSTOM, 0.0, 0.0, 0.5, 0.0)).exposure
            for f in factor_names
        ])

        # 计算隐含暴露：通过因子相关性传导
        # implied = correlation × explicit
        implied_vector = factor_correlation @ explicit_vector

        # 计算残差暴露
        residual_exposures = {}
        if returns_history is not None and len(returns_history) >= self.config.min_observations:
            residual_exposures = self._calculate_residual_exposures(
                returns_history,
                explicit_vector,
                factor_names
            )

        # 创建ImpliedExposure对象
        for i, factor_name in enumerate(factor_names):
            explicit_exp = explicit_vector[i]
            implied_exp = implied_vector[i]

            # 隐性暴露 = 隐含暴露 - 显性暴露
            implicit_exp = implied_exp - explicit_exp

            # 基于相关性的暴露
            correlation_based = sum(
                factor_correlation[i, j] * explicit_vector[j]
                for j in range(len(factor_names))
                if j != i
            )

            # 残差暴露
            residual_exp = residual_exposures.get(factor_name, 0.0)

            # 置信度：基于相关性的显著性
            confidence = self._calculate_confidence(
                factor_correlation[i, :],
                len(factor_names)
            )

            implied[factor_name] = ImpliedExposure(
                factor_name=factor_name,
                implied_exposure=implied_exp,
                explicit_exposure=explicit_exp,
                implicit_exposure=implicit_exp,
                correlation_based=correlation_based,
                residual_exposure=residual_exp,
                confidence=confidence,
            )

        return implied

    def _calculate_residual_exposures(
        self,
        returns: np.ndarray,
        exposures: np.ndarray,
        factor_names: List[str],
    ) -> Dict[str, float]:
        """计算残差暴露"""
        # 简化版：假设残差与某些隐藏因子相关
        residuals = {}

        # 这里可以实现更复杂的残差分析
        # 目前返回零暴露
        for factor_name in factor_names:
            residuals[factor_name] = 0.0

        return residuals

    def _calculate_confidence(self, correlations: np.ndarray, n: int) -> float:
        """
        计算置信度

        基于相关性的统计显著性
        """
        if n <= 2:
            return 0.0

        # 平均绝对相关性
        mean_abs_corr = np.mean(np.abs(correlations))

        # 转换为置信度（简化）
        confidence = min(mean_abs_corr * 2, 1.0)

        return confidence


# ============ Exposure Forecaster ============

class ExposureForecaster:
    """
    因子暴露预测器

    预测未来的因子暴露
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self._exposure_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.lookback_period)
        )

    def update_history(self, exposures: Dict[str, FactorExposure]):
        """更新暴露历史"""
        for factor_name, exposure_obj in exposures.items():
            self._exposure_history[factor_name].append(exposure_obj.exposure)

    def forecast_exposures(
        self,
        current_exposures: Dict[str, FactorExposure],
        forecast_horizon: int = 5,
    ) -> Dict[str, ExposureForecast]:
        """
        预测因子暴露

        使用时间序列方法预测未来暴露

        Args:
            current_exposures: 当前暴露字典
            forecast_horizon: 预测期限（天）

        Returns:
            暴露预测字典
        """
        forecasts = {}

        for factor_name, exposure_obj in current_exposures.items():
            history = list(self._exposure_history.get(factor_name, []))

            if len(history) < 10:  # 最少需要10个历史点
                # 数据不足，使用当前值
                forecasts[factor_name] = ExposureForecast(
                    factor_name=factor_name,
                    current_exposure=exposure_obj.exposure,
                    forecast_exposure=exposure_obj.exposure,
                    forecast_horizon=forecast_horizon,
                    confidence_interval=(exposure_obj.exposure, exposure_obj.exposure),
                    expected_change=0.0,
                    trend_strength=0.0,
                    volatility=0.0,
                )
                continue

            current_exposure = exposure_obj.exposure

            # 简单趋势预测：使用线性回归
            x = np.arange(len(history))
            y = np.array(history)

            # 拟合线性趋势
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)

                # 预测
                forecast_x = len(history) + forecast_horizon - 1
                forecast_exposure = slope * forecast_x + intercept

                # 计算趋势强度（R²）
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)

                if ss_tot > 0:
                    trend_strength = 1 - ss_res / ss_tot
                else:
                    trend_strength = 0.0

                # 计算波动率
                volatility = np.std(y - y_pred) if len(y) > 1 else 0.0

                # 置信区间
                if volatility > 0:
                    z_score = stats.norm.ppf((1 + self.config.confidence_level) / 2)
                    margin = z_score * volatility * np.sqrt(forecast_horizon)
                    ci = (forecast_exposure - margin, forecast_exposure + margin)
                else:
                    ci = (forecast_exposure, forecast_exposure)

                expected_change = forecast_exposure - current_exposure
            else:
                forecast_exposure = current_exposure
                trend_strength = 0.0
                volatility = 0.0
                ci = (current_exposure, current_exposure)
                expected_change = 0.0

            forecasts[factor_name] = ExposureForecast(
                factor_name=factor_name,
                current_exposure=current_exposure,
                forecast_exposure=forecast_exposure,
                forecast_horizon=forecast_horizon,
                confidence_interval=ci,
                expected_change=expected_change,
                trend_strength=trend_strength,
                volatility=volatility,
            )

        return forecasts


# ============ Style Drift Detector ============

class StyleDriftDetector:
    """
    风格漂移检测器

    检测组合的风格漂移
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self._baseline_exposures: Dict[str, float] = {}
        self._drift_start_dates: Dict[str, datetime] = {}
        self._exposure_history: Dict[str, Deque[Tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=self.config.lookback_period)
        )

    def set_baseline(self, exposures: Dict[str, FactorExposure]):
        """设置基线暴露"""
        self._baseline_exposures = {
            factor_name: exp.exposure
            for factor_name, exp in exposures.items()
        }

    def update_history(self, exposures: Dict[str, FactorExposure], timestamp: datetime):
        """更新暴露历史"""
        for factor_name, exposure_obj in exposures.items():
            self._exposure_history[factor_name].append((timestamp, exposure_obj.exposure))

    def detect_drift(
        self,
        current_exposures: Dict[str, FactorExposure],
    ) -> Dict[str, StyleDrift]:
        """
        检测风格漂移

        Args:
            current_exposures: 当前暴露字典

        Returns:
            风格漂移字典
        """
        drifts = {}

        if not self._baseline_exposures:
            # 如果没有设置基线，使用第一次的暴露作为基线
            self.set_baseline(current_exposures)

        current_time = datetime.now()

        for factor_name, exposure_obj in current_exposures.items():
            baseline = self._baseline_exposures.get(factor_name, 0.0)
            current = exposure_obj.exposure

            # 漂移量
            drift_amount = current - baseline

            # 漂移百分比
            if abs(baseline) > 1e-10:
                drift_pct = drift_amount / baseline
            else:
                drift_pct = drift_amount if abs(drift_amount) > 1e-10 else 0.0

            # 判断严重程度
            abs_drift_pct = abs(drift_pct)
            threshold = self.config.drift_threshold

            if abs_drift_pct < threshold:
                severity = DriftSeverity.NONE
            elif abs_drift_pct < threshold * 2:
                severity = DriftSeverity.LOW
            elif abs_drift_pct < threshold * 3:
                severity = DriftSeverity.MODERATE
            elif abs_drift_pct < threshold * 5:
                severity = DriftSeverity.HIGH
            else:
                severity = DriftSeverity.CRITICAL

            # 计算漂移速度
            history = list(self._exposure_history.get(factor_name, []))
            drift_velocity = self._calculate_drift_velocity(history)

            # 漂移持续时间
            if severity != DriftSeverity.NONE:
                if factor_name not in self._drift_start_dates:
                    self._drift_start_dates[factor_name] = current_time

                time_in_drift = (current_time - self._drift_start_dates[factor_name]).days
            else:
                if factor_name in self._drift_start_dates:
                    del self._drift_start_dates[factor_name]
                time_in_drift = 0

            # 是否系统性漂移
            is_systematic = self._is_systematic_drift(history, baseline)

            # 回归概率（基于均值回归假设）
            reversion_probability = self._estimate_reversion_probability(
                drift_amount,
                drift_velocity,
                time_in_drift
            )

            drifts[factor_name] = StyleDrift(
                factor_name=factor_name,
                baseline_exposure=baseline,
                current_exposure=current,
                drift_amount=drift_amount,
                drift_pct=drift_pct,
                severity=severity,
                drift_velocity=drift_velocity,
                time_in_drift=time_in_drift,
                is_systematic=is_systematic,
                reversion_probability=reversion_probability,
            )

        return drifts

    def _calculate_drift_velocity(
        self,
        history: List[Tuple[datetime, float]]
    ) -> float:
        """
        计算漂移速度

        速度 = 暴露变化 / 时间变化
        """
        if len(history) < 2:
            return 0.0

        # 使用最近的数据点计算速度
        recent_window = min(20, len(history))
        recent_history = history[-recent_window:]

        times = np.array([(t - recent_history[0][0]).total_seconds() / 86400 for t, _ in recent_history])
        exposures = np.array([exp for _, exp in recent_history])

        if len(times) > 1:
            # 线性回归计算斜率
            slope, _ = np.polyfit(times, exposures, 1)
            return slope

        return 0.0

    def _is_systematic_drift(
        self,
        history: List[Tuple[datetime, float]],
        baseline: float
    ) -> bool:
        """
        判断是否为系统性漂移

        系统性漂移：持续向一个方向移动
        """
        if len(history) < 10:
            return False

        recent_window = min(20, len(history))
        recent_exposures = [exp for _, exp in history[-recent_window:]]

        # 计算与基线的偏离
        deviations = [exp - baseline for exp in recent_exposures]

        # 检查是否同号（同方向）
        if len(deviations) == 0:
            return False

        positive_count = sum(1 for d in deviations if d > 0)
        negative_count = sum(1 for d in deviations if d < 0)

        # 如果80%以上同向，认为是系统性的
        max_count = max(positive_count, negative_count)
        return max_count / len(deviations) >= 0.8

    def _estimate_reversion_probability(
        self,
        drift_amount: float,
        drift_velocity: float,
        time_in_drift: int
    ) -> float:
        """
        估计回归概率

        基于均值回归理论，漂移越大、速度越快、时间越长，回归概率越高
        """
        # 基于漂移量
        drift_factor = 1.0 / (1.0 + np.exp(-abs(drift_amount)))

        # 基于时间
        time_factor = min(time_in_drift / 60.0, 1.0)  # 60天归一化

        # 基于速度（速度快可能继续，速度慢可能回归）
        velocity_factor = 1.0 / (1.0 + abs(drift_velocity) * 10)

        # 综合概率
        probability = 0.4 * drift_factor + 0.3 * time_factor + 0.3 * velocity_factor

        return min(max(probability, 0.0), 1.0)


# ============ Main Factor Exposure Analyzer ============

class FactorExposureAnalyzer:
    """
    因子暴露分析器主类

    整合所有因子暴露分析功能
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()

        self.exposure_calculator = FactorExposureCalculator(config)
        self.attribution_tracker = FactorAttributionTracker(config)
        self.risk_budget_manager = FactorRiskBudgetManager(config)
        self.neutralization_detector = FactorNeutralizationDetector(config)
        self.implied_analyzer = ImpliedExposureAnalyzer(config)
        self.forecaster = ExposureForecaster(config)
        self.drift_detector = StyleDriftDetector(config)

        self._profile_history: Deque[PortfolioFactorProfile] = deque(maxlen=252)

    async def analyze_portfolio(
        self,
        weights: Dict[str, float],
        asset_factors: Dict[str, Dict[str, float]],
        factor_returns: Dict[str, float],
        portfolio_return: float,
        benchmark_weights: Optional[Dict[str, float]] = None,
        benchmark_factors: Optional[Dict[str, Dict[str, float]]] = None,
        factor_covariance: Optional[np.ndarray] = None,
        factor_correlation: Optional[np.ndarray] = None,
        target_neutralization: Optional[Dict[str, NeutralizationLevel]] = None,
        forecast_horizon: int = 5,
    ) -> PortfolioFactorProfile:
        """
        执行完整的组合因子分析

        Args:
            weights: 资产权重字典
            asset_factors: 资产因子数据
            factor_returns: 因子收益字典
            portfolio_return: 组合收益
            benchmark_weights: 基准权重（可选）
            benchmark_factors: 基准因子数据（可选）
            factor_covariance: 因子协方差矩阵（可选）
            factor_correlation: 因子相关性矩阵（可选）
            target_neutralization: 目标中性化水平（可选）
            forecast_horizon: 预测期限

        Returns:
            PortfolioFactorProfile: 组合因子画像
        """
        # 1. 计算因子暴露
        exposures = await asyncio.get_event_loop().run_in_executor(
            None,
            self.exposure_calculator.calculate_exposures,
            weights,
            asset_factors,
            benchmark_weights,
            benchmark_factors,
        )

        # 2. 计算因子贡献
        contributions = await asyncio.get_event_loop().run_in_executor(
            None,
            self.attribution_tracker.calculate_contributions,
            exposures,
            factor_returns,
            portfolio_return,
        )

        # 3. 计算风险预算
        risk_budgets = {}
        if factor_covariance is not None:
            factor_names = sorted(exposures.keys())
            risk_budgets = await asyncio.get_event_loop().run_in_executor(
                None,
                self.risk_budget_manager.calculate_risk_budgets,
                exposures,
                factor_covariance,
                factor_names,
            )

        # 4. 检测中性化
        neutralization = await asyncio.get_event_loop().run_in_executor(
            None,
            self.neutralization_detector.detect_neutralization,
            exposures,
            target_neutralization,
        )

        # 5. 分析隐含暴露
        implied = {}
        if factor_correlation is not None:
            factor_names = sorted(exposures.keys())
            implied = await asyncio.get_event_loop().run_in_executor(
                None,
                self.implied_analyzer.analyze_implied_exposures,
                exposures,
                factor_correlation,
                factor_names,
                None,
            )

        # 6. 预测暴露
        self.forecaster.update_history(exposures)
        forecasts = await asyncio.get_event_loop().run_in_executor(
            None,
            self.forecaster.forecast_exposures,
            exposures,
            forecast_horizon,
        )

        # 7. 检测风格漂移
        current_time = datetime.now()
        self.drift_detector.update_history(exposures, current_time)
        drifts = await asyncio.get_event_loop().run_in_executor(
            None,
            self.drift_detector.detect_drift,
            exposures,
        )

        # 8. 计算综合指标
        total_risk = self._calculate_total_risk(exposures, factor_covariance)
        diversification_score = self._calculate_diversification_score(exposures)
        factor_concentration = self._calculate_factor_concentration(exposures)

        # 创建组合因子画像
        profile = PortfolioFactorProfile(
            exposures=exposures,
            contributions=contributions,
            risk_budgets=risk_budgets,
            neutralization=neutralization,
            implied=implied,
            forecasts=forecasts,
            drifts=drifts,
            total_risk=total_risk,
            diversification_score=diversification_score,
            factor_concentration=factor_concentration,
        )

        # 保存历史
        self._profile_history.append(profile)

        # 更新历史数据
        self.exposure_calculator.update_history(portfolio_return, factor_returns)

        return profile

    def analyze_portfolio_sync(self, **kwargs) -> PortfolioFactorProfile:
        """同步版本的分析方法"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.analyze_portfolio(**kwargs))
        finally:
            loop.close()

    def set_risk_budgets(
        self,
        budgets: Dict[str, float],
        limits: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        设置风险预算

        Args:
            budgets: 风险预算字典 {factor_name: budget}
            limits: 暴露限制字典 {factor_name: (lower, upper)}
        """
        for factor_name, budget in budgets.items():
            limit = limits.get(factor_name) if limits else None
            self.risk_budget_manager.set_risk_budget(factor_name, budget, limit)

    def set_baseline_style(self, exposures: Dict[str, FactorExposure]):
        """设置基线风格"""
        self.drift_detector.set_baseline(exposures)

    def get_profile_history(self) -> List[PortfolioFactorProfile]:
        """获取历史画像"""
        return list(self._profile_history)

    def _calculate_total_risk(
        self,
        exposures: Dict[str, FactorExposure],
        factor_covariance: Optional[np.ndarray]
    ) -> float:
        """计算总风险"""
        if factor_covariance is None:
            return 0.0

        factor_names = sorted(exposures.keys())
        exposure_vector = np.array([exposures[f].exposure for f in factor_names])

        variance = exposure_vector @ factor_covariance @ exposure_vector
        return np.sqrt(variance) if variance > 0 else 0.0

    def _calculate_diversification_score(
        self,
        exposures: Dict[str, FactorExposure]
    ) -> float:
        """
        计算分散化得分

        基于因子暴露的分散程度，使用Herfindahl指数
        """
        if not exposures:
            return 0.0

        # 使用绝对值暴露
        abs_exposures = [abs(exp.exposure) for exp in exposures.values()]
        total = sum(abs_exposures)

        if total == 0:
            return 1.0

        # 归一化
        weights = [e / total for e in abs_exposures]

        # Herfindahl指数
        herfindahl = sum(w ** 2 for w in weights)

        # 转换为分散化得分（1 - H）
        # 完全集中：H=1, score=0
        # 完全分散：H=1/n, score=(n-1)/n
        n = len(exposures)
        max_diversification = (n - 1) / n if n > 1 else 0.0

        if max_diversification > 0:
            score = (1 - herfindahl) / max_diversification
        else:
            score = 0.0

        return min(max(score, 0.0), 1.0)

    def _calculate_factor_concentration(
        self,
        exposures: Dict[str, FactorExposure]
    ) -> float:
        """
        计算因子集中度

        测量暴露在少数因子上的集中程度
        """
        if not exposures:
            return 0.0

        abs_exposures = sorted([abs(exp.exposure) for exp in exposures.values()], reverse=True)
        total = sum(abs_exposures)

        if total == 0:
            return 0.0

        # 计算前3个因子的占比
        top_3 = sum(abs_exposures[:3]) / total if len(abs_exposures) >= 3 else sum(abs_exposures) / total

        return top_3


# ============ Factory Functions ============

def create_analyzer(
    method: str = 'hybrid',
    factors: Optional[List[str]] = None,
    lookback_period: int = 252,
    **kwargs
) -> FactorExposureAnalyzer:
    """
    创建因子暴露分析器

    Args:
        method: 计算方法 (regression/fundamental/statistical/hybrid/implied)
        factors: 因子列表
        lookback_period: 回溯期
        **kwargs: 其他配置参数

    Returns:
        FactorExposureAnalyzer实例
    """
    method_map = {
        'regression': ExposureMethod.REGRESSION,
        'fundamental': ExposureMethod.FUNDAMENTAL,
        'statistical': ExposureMethod.STATISTICAL,
        'hybrid': ExposureMethod.HYBRID,
        'implied': ExposureMethod.IMPLIED,
    }

    config = AnalyzerConfig(
        method=method_map.get(method, ExposureMethod.HYBRID),
        factors=factors or ['market', 'size', 'value', 'momentum', 'volatility', 'quality'],
        lookback_period=lookback_period,
        **kwargs
    )

    return FactorExposureAnalyzer(config)


async def quick_exposure_analysis(
    portfolio: Dict[str, float],
    factors: Dict[str, Dict[str, float]],
    factor_returns: Dict[str, float],
    portfolio_return: float,
) -> PortfolioFactorProfile:
    """
    快速因子暴露分析

    Args:
        portfolio: 组合权重 {symbol: weight}
        factors: 因子数据 {symbol: {factor: value}}
        factor_returns: 因子收益 {factor: return}
        portfolio_return: 组合收益

    Returns:
        组合因子画像
    """
    analyzer = create_analyzer()

    return await analyzer.analyze_portfolio(
        weights=portfolio,
        asset_factors=factors,
        factor_returns=factor_returns,
        portfolio_return=portfolio_return,
    )


__all__ = [
    # Enums
    'FactorCategory',
    'ExposureMethod',
    'NeutralizationLevel',
    'DriftSeverity',
    # Data Classes
    'FactorExposure',
    'FactorContribution',
    'RiskBudget',
    'NeutralizationResult',
    'ImpliedExposure',
    'ExposureForecast',
    'StyleDrift',
    'PortfolioFactorProfile',
    'AnalyzerConfig',
    # Components
    'FactorExposureCalculator',
    'FactorAttributionTracker',
    'FactorRiskBudgetManager',
    'FactorNeutralizationDetector',
    'ImpliedExposureAnalyzer',
    'ExposureForecaster',
    'StyleDriftDetector',
    # Main Class
    'FactorExposureAnalyzer',
    # Factory Functions
    'create_analyzer',
    'quick_exposure_analysis',
]
