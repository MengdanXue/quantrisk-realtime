"""
市场冲击估算器模块 (Market Impact Estimator)

提供全面的市场冲击估算和分析功能:
- 多种冲击模型 (Linear, Square Root, Almgren-Chriss, Kyle Lambda)
- 临时/永久冲击分解
- 预交易冲击预测
- 实时冲击跟踪
- 最优执行路径规划
- 历史参数校准
- 成本/风险权衡分析
"""

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


# ==================== 枚举类型 ====================

class ImpactModelType(str, Enum):
    """冲击模型类型"""
    LINEAR = "linear"                          # 线性冲击模型
    SQUARE_ROOT = "square_root"               # 平方根模型
    POWER_LAW = "power_law"                   # 幂律模型
    ALMGREN_CHRISS = "almgren_chriss"         # Almgren-Chriss模型
    KYLE_LAMBDA = "kyle_lambda"               # Kyle Lambda模型


class ImpactType(str, Enum):
    """冲击类型"""
    TEMPORARY = "temporary"                    # 临时冲击
    PERMANENT = "permanent"                    # 永久冲击
    TOTAL = "total"                            # 总冲击


class ExecutionStrategy(str, Enum):
    """执行策略"""
    AGGRESSIVE = "aggressive"                  # 激进执行
    PASSIVE = "passive"                        # 被动执行
    TWAP = "twap"                             # 时间加权
    VWAP = "vwap"                             # 成交量加权
    POV = "pov"                               # 参与率策略
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"  # 执行差价


class TradeDirection(str, Enum):
    """交易方向"""
    BUY = "buy"
    SELL = "sell"


class MarketCondition(str, Enum):
    """市场状态"""
    NORMAL = "normal"                          # 正常
    VOLATILE = "volatile"                      # 高波动
    ILLIQUID = "illiquid"                     # 低流动性
    STRESSED = "stressed"                      # 压力状态


# ==================== 数据类 ====================

@dataclass
class ImpactParameters:
    """冲击参数"""
    symbol: str

    # 市场参数
    price: float                               # 当前价格
    daily_volume: float                        # 日均成交量
    volatility: float                          # 波动率(年化)
    spread_bps: float                          # 买卖价差(bps)

    # 模型参数 - Linear Model
    linear_coefficient: float = 0.1

    # 模型参数 - Square Root Model
    sqrt_coefficient: float = 0.1
    sqrt_exponent: float = 0.5

    # 模型参数 - Almgren-Chriss Model
    eta: float = 0.001                         # 临时冲击参数
    gamma: float = 0.0001                      # 永久冲击参数
    lambda_risk: float = 2e-6                  # 风险厌恶参数

    # 模型参数 - Kyle Lambda Model
    kyle_lambda: float = 0.0001                # Kyle's Lambda

    # 流动性因子
    liquidity_factor: float = 1.0              # 流动性调整因子(1.0=正常)

    # 市场状态
    market_condition: MarketCondition = MarketCondition.NORMAL

    # 校准信息
    calibrated_at: Optional[datetime] = None
    calibration_period_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'daily_volume': self.daily_volume,
            'volatility': self.volatility,
            'spread_bps': self.spread_bps,
            'linear_coefficient': self.linear_coefficient,
            'sqrt_coefficient': self.sqrt_coefficient,
            'eta': self.eta,
            'gamma': self.gamma,
            'kyle_lambda': self.kyle_lambda,
            'liquidity_factor': self.liquidity_factor,
            'market_condition': self.market_condition.value,
            'calibrated_at': self.calibrated_at.isoformat() if self.calibrated_at else None,
        }


@dataclass
class ImpactDecomposition:
    """冲击分解"""
    temporary_impact: float                    # 临时冲击(bps)
    permanent_impact: float                    # 永久冲击(bps)
    total_impact: float                        # 总冲击(bps)

    # 分解详情
    spread_component: float = 0.0              # 价差成分
    liquidity_component: float = 0.0           # 流动性成分
    information_component: float = 0.0         # 信息成分
    volatility_component: float = 0.0          # 波动性成分

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'temporary_impact': round(self.temporary_impact, 2),
            'permanent_impact': round(self.permanent_impact, 2),
            'total_impact': round(self.total_impact, 2),
            'spread_component': round(self.spread_component, 2),
            'liquidity_component': round(self.liquidity_component, 2),
            'information_component': round(self.information_component, 2),
            'volatility_component': round(self.volatility_component, 2),
        }


@dataclass
class ImpactEstimate:
    """冲击估计"""
    symbol: str
    direction: TradeDirection
    quantity: float
    current_price: float

    # 冲击估计(bps)
    estimated_impact: float                    # 预估总冲击
    temporary_impact: float                    # 临时冲击
    permanent_impact: float                    # 永久冲击

    # 置信区间
    confidence: float                          # 置信度(0-1)
    impact_low: float                          # 最低估计
    impact_high: float                         # 最高估计

    # 模型信息
    model_type: ImpactModelType
    participation_rate: float                  # 参与率

    # 成本估计
    estimated_cost_bps: float                  # 总成本(bps)
    estimated_cost_currency: float             # 总成本(货币)

    # 分解
    decomposition: Optional[ImpactDecomposition] = None

    # 时间戳
    estimated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'quantity': self.quantity,
            'current_price': self.current_price,
            'estimated_impact': round(self.estimated_impact, 2),
            'temporary_impact': round(self.temporary_impact, 2),
            'permanent_impact': round(self.permanent_impact, 2),
            'confidence': round(self.confidence, 2),
            'impact_low': round(self.impact_low, 2),
            'impact_high': round(self.impact_high, 2),
            'model_type': self.model_type.value,
            'participation_rate': round(self.participation_rate, 4),
            'estimated_cost_bps': round(self.estimated_cost_bps, 2),
            'estimated_cost_currency': round(self.estimated_cost_currency, 2),
            'decomposition': self.decomposition.to_dict() if self.decomposition else None,
            'estimated_at': self.estimated_at.isoformat(),
        }


@dataclass
class ExecutionSlice:
    """执行切片"""
    slice_number: int
    quantity: float
    start_time: float                          # 相对时间(秒)
    end_time: float                            # 相对时间(秒)
    expected_price: float                      # 预期价格
    expected_impact: float                     # 预期冲击(bps)
    expected_cost: float                       # 预期成本(货币)


@dataclass
class ExecutionPath:
    """执行路径"""
    symbol: str
    direction: TradeDirection
    total_quantity: float

    # 路径策略
    strategy: ExecutionStrategy

    # 执行切片
    slices: List[ExecutionSlice]

    # 总体指标
    total_duration_seconds: float              # 总执行时间(秒)
    total_expected_impact: float               # 总预期冲击(bps)
    total_expected_cost: float                 # 总预期成本(货币)

    # 风险指标
    execution_risk: float                      # 执行风险
    timing_risk: float                         # 时机风险

    # 优化目标
    optimization_objective: str = "minimize_cost"  # 优化目标

    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'total_quantity': self.total_quantity,
            'strategy': self.strategy.value,
            'slices': [
                {
                    'slice_number': s.slice_number,
                    'quantity': s.quantity,
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'expected_price': s.expected_price,
                    'expected_impact': round(s.expected_impact, 2),
                    'expected_cost': round(s.expected_cost, 2),
                }
                for s in self.slices
            ],
            'total_duration_seconds': self.total_duration_seconds,
            'total_expected_impact': round(self.total_expected_impact, 2),
            'total_expected_cost': round(self.total_expected_cost, 2),
            'execution_risk': round(self.execution_risk, 2),
            'timing_risk': round(self.timing_risk, 2),
            'optimization_objective': self.optimization_objective,
            'generated_at': self.generated_at.isoformat(),
        }


@dataclass
class RealTimeImpact:
    """实时冲击跟踪"""
    symbol: str
    direction: TradeDirection

    # 执行信息
    target_quantity: float
    executed_quantity: float
    remaining_quantity: float

    # 价格信息
    initial_price: float
    current_price: float
    average_execution_price: float

    # 冲击测量
    realized_impact: float                     # 已实现冲击(bps)
    estimated_remaining_impact: float          # 剩余预估冲击(bps)
    total_projected_impact: float              # 总预估冲击(bps)

    # 执行进度
    execution_progress: float                  # 执行进度(0-1)
    elapsed_time_seconds: float                # 已用时间(秒)

    # 成本
    realized_cost: float                       # 已实现成本(货币)
    estimated_total_cost: float                # 总预估成本(货币)

    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'target_quantity': self.target_quantity,
            'executed_quantity': self.executed_quantity,
            'remaining_quantity': self.remaining_quantity,
            'initial_price': self.initial_price,
            'current_price': self.current_price,
            'average_execution_price': self.average_execution_price,
            'realized_impact': round(self.realized_impact, 2),
            'estimated_remaining_impact': round(self.estimated_remaining_impact, 2),
            'total_projected_impact': round(self.total_projected_impact, 2),
            'execution_progress': round(self.execution_progress, 2),
            'elapsed_time_seconds': round(self.elapsed_time_seconds, 1),
            'realized_cost': round(self.realized_cost, 2),
            'estimated_total_cost': round(self.estimated_total_cost, 2),
            'updated_at': self.updated_at.isoformat(),
        }


@dataclass
class ScenarioAnalysis:
    """情景分析"""
    symbol: str
    base_quantity: float

    # 情景定义
    scenarios: List[Dict[str, Any]]

    # 结果
    scenario_results: List[Dict[str, float]]

    # 推荐
    recommended_scenario: str
    recommendation_reason: str

    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'base_quantity': self.base_quantity,
            'scenarios': self.scenarios,
            'scenario_results': self.scenario_results,
            'recommended_scenario': self.recommended_scenario,
            'recommendation_reason': self.recommendation_reason,
            'analyzed_at': self.analyzed_at.isoformat(),
        }


@dataclass
class EstimatorConfig:
    """估算器配置"""
    # 默认模型
    default_model: ImpactModelType = ImpactModelType.SQUARE_ROOT

    # 参数校准
    auto_calibrate: bool = True
    calibration_window_days: int = 30
    min_trades_for_calibration: int = 20

    # 风险参数
    default_risk_aversion: float = 2e-6
    max_participation_rate: float = 0.25       # 最大参与率

    # 执行优化
    default_execution_horizon_minutes: int = 60  # 默认执行时间窗口
    min_slice_size: float = 100.0              # 最小切片大小
    max_slices: int = 20                       # 最大切片数

    # 置信度参数
    confidence_level: float = 0.95             # 置信水平

    # 市场状态调整因子
    market_condition_factors: Dict[MarketCondition, float] = field(default_factory=lambda: {
        MarketCondition.NORMAL: 1.0,
        MarketCondition.VOLATILE: 1.5,
        MarketCondition.ILLIQUID: 2.0,
        MarketCondition.STRESSED: 2.5,
    })


# ==================== 冲击模型接口 ====================

class ImpactModel(ABC):
    """市场冲击模型基类"""

    @abstractmethod
    def estimate_impact(
        self,
        quantity: float,
        parameters: ImpactParameters,
        direction: TradeDirection
    ) -> Tuple[float, float, float]:
        """
        估计市场冲击

        Args:
            quantity: 交易数量
            parameters: 冲击参数
            direction: 交易方向

        Returns:
            (总冲击, 临时冲击, 永久冲击) in bps
        """
        pass


class LinearImpactModel(ImpactModel):
    """线性冲击模型"""

    def estimate_impact(
        self,
        quantity: float,
        parameters: ImpactParameters,
        direction: TradeDirection
    ) -> Tuple[float, float, float]:
        """Impact = coefficient * participation_rate * volatility"""
        if parameters.daily_volume <= 0:
            return 0.0, 0.0, 0.0

        participation_rate = quantity / parameters.daily_volume
        daily_vol = parameters.volatility / math.sqrt(252)

        # 线性冲击
        impact = parameters.linear_coefficient * participation_rate * daily_vol * 10000

        # 流动性调整
        impact *= parameters.liquidity_factor

        # 简化: 60%临时, 40%永久
        temporary = impact * 0.6
        permanent = impact * 0.4

        return impact, temporary, permanent


class SquareRootImpactModel(ImpactModel):
    """平方根冲击模型"""

    def estimate_impact(
        self,
        quantity: float,
        parameters: ImpactParameters,
        direction: TradeDirection
    ) -> Tuple[float, float, float]:
        """Impact = coefficient * sqrt(participation_rate) * volatility"""
        if parameters.daily_volume <= 0:
            return 0.0, 0.0, 0.0

        participation_rate = quantity / parameters.daily_volume
        daily_vol = parameters.volatility / math.sqrt(252)

        # 平方根冲击
        impact = (
            parameters.sqrt_coefficient *
            (participation_rate ** parameters.sqrt_exponent) *
            daily_vol * 10000
        )

        # 流动性调整
        impact *= parameters.liquidity_factor

        # 简化: 70%临时, 30%永久
        temporary = impact * 0.7
        permanent = impact * 0.3

        return impact, temporary, permanent


class AlmgrenChrissImpactModel(ImpactModel):
    """Almgren-Chriss冲击模型"""

    def estimate_impact(
        self,
        quantity: float,
        parameters: ImpactParameters,
        direction: TradeDirection
    ) -> Tuple[float, float, float]:
        """
        Almgren-Chriss模型
        临时冲击: eta * trading_rate
        永久冲击: gamma * quantity
        """
        if parameters.daily_volume <= 0:
            return 0.0, 0.0, 0.0

        daily_vol = parameters.volatility / math.sqrt(252)

        # 临时冲击 (假设在1小时内执行)
        trading_rate = quantity  # 简化
        temp_impact_raw = parameters.eta * trading_rate

        # 永久冲击
        perm_impact_raw = parameters.gamma * quantity

        # 转换为bps
        temporary = temp_impact_raw * daily_vol * 10000
        permanent = perm_impact_raw * daily_vol * 10000

        # 流动性调整
        temporary *= parameters.liquidity_factor
        permanent *= parameters.liquidity_factor

        total = temporary + permanent

        return total, temporary, permanent


class KyleLambdaImpactModel(ImpactModel):
    """Kyle Lambda冲击模型"""

    def estimate_impact(
        self,
        quantity: float,
        parameters: ImpactParameters,
        direction: TradeDirection
    ) -> Tuple[float, float, float]:
        """
        Kyle's Lambda Model
        Impact = lambda * quantity
        """
        if parameters.daily_volume <= 0:
            return 0.0, 0.0, 0.0

        # Kyle's Lambda冲击
        impact_raw = parameters.kyle_lambda * quantity

        # 转换为bps
        if parameters.price > 0:
            impact = (impact_raw / parameters.price) * 10000
        else:
            impact = 0.0

        # 流动性调整
        impact *= parameters.liquidity_factor

        # Kyle模型主要是永久冲击
        temporary = impact * 0.2
        permanent = impact * 0.8

        return impact, temporary, permanent


# ==================== 市场冲击估算器 ====================

class MarketImpactEstimator:
    """市场冲击估算器"""

    def __init__(
        self,
        config: Optional[EstimatorConfig] = None
    ):
        self.config = config or EstimatorConfig()

        # 冲击模型
        self._impact_models: Dict[ImpactModelType, ImpactModel] = {
            ImpactModelType.LINEAR: LinearImpactModel(),
            ImpactModelType.SQUARE_ROOT: SquareRootImpactModel(),
            ImpactModelType.ALMGREN_CHRISS: AlmgrenChrissImpactModel(),
            ImpactModelType.KYLE_LAMBDA: KyleLambdaImpactModel(),
        }

        # 参数存储
        self._parameters: Dict[str, ImpactParameters] = {}

        # 历史数据
        self._historical_impacts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # 实时跟踪
        self._active_tracks: Dict[str, RealTimeImpact] = {}

        logger.info("MarketImpactEstimator initialized")

    async def estimate_impact(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: float,
        current_price: float,
        parameters: Optional[ImpactParameters] = None,
        model_type: Optional[ImpactModelType] = None
    ) -> ImpactEstimate:
        """
        估计市场冲击

        Args:
            symbol: 交易品种
            direction: 交易方向
            quantity: 交易数量
            current_price: 当前价格
            parameters: 冲击参数
            model_type: 模型类型

        Returns:
            冲击估计
        """
        # 获取或创建参数
        if parameters is None:
            parameters = await self._get_parameters(symbol)

        # 选择模型
        model_type = model_type or self.config.default_model
        model = self._impact_models[model_type]

        # 估计冲击
        total_impact, temp_impact, perm_impact = model.estimate_impact(
            quantity, parameters, direction
        )

        # 市场状态调整
        condition_factor = self.config.market_condition_factors.get(
            parameters.market_condition, 1.0
        )
        total_impact *= condition_factor
        temp_impact *= condition_factor
        perm_impact *= condition_factor

        # 计算参与率
        participation_rate = (
            quantity / parameters.daily_volume
            if parameters.daily_volume > 0 else 0
        )

        # 置信区间
        confidence = self._calculate_confidence(participation_rate, parameters)
        impact_std = total_impact * 0.3  # 30%标准差
        impact_low = max(0, total_impact - impact_std)
        impact_high = total_impact + impact_std

        # 成本估计
        notional = quantity * current_price
        estimated_cost_bps = total_impact
        estimated_cost_currency = notional * estimated_cost_bps / 10000

        # 冲击分解
        decomposition = self._decompose_impact(
            total_impact, parameters, participation_rate
        )

        estimate = ImpactEstimate(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            current_price=current_price,
            estimated_impact=total_impact,
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            confidence=confidence,
            impact_low=impact_low,
            impact_high=impact_high,
            model_type=model_type,
            participation_rate=participation_rate,
            estimated_cost_bps=estimated_cost_bps,
            estimated_cost_currency=estimated_cost_currency,
            decomposition=decomposition,
        )

        # 记录历史
        self._historical_impacts[symbol].append({
            'timestamp': datetime.now(),
            'estimate': estimate,
        })

        return estimate

    async def optimize_execution_path(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: float,
        current_price: float,
        strategy: ExecutionStrategy = ExecutionStrategy.TWAP,
        duration_minutes: Optional[int] = None,
        parameters: Optional[ImpactParameters] = None
    ) -> ExecutionPath:
        """
        优化执行路径

        Args:
            symbol: 交易品种
            direction: 交易方向
            quantity: 总数量
            current_price: 当前价格
            strategy: 执行策略
            duration_minutes: 执行时长(分钟)
            parameters: 冲击参数

        Returns:
            执行路径
        """
        # 获取参数
        if parameters is None:
            parameters = await self._get_parameters(symbol)

        duration_minutes = duration_minutes or self.config.default_execution_horizon_minutes
        duration_seconds = duration_minutes * 60

        # 根据策略生成切片
        if strategy == ExecutionStrategy.TWAP:
            slices = self._generate_twap_slices(
                quantity, duration_seconds, current_price, parameters, direction
            )
        elif strategy == ExecutionStrategy.VWAP:
            slices = self._generate_vwap_slices(
                quantity, duration_seconds, current_price, parameters, direction
            )
        elif strategy == ExecutionStrategy.POV:
            slices = self._generate_pov_slices(
                quantity, duration_seconds, current_price, parameters, direction
            )
        elif strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL:
            slices = self._generate_is_slices(
                quantity, duration_seconds, current_price, parameters, direction
            )
        else:
            # 默认TWAP
            slices = self._generate_twap_slices(
                quantity, duration_seconds, current_price, parameters, direction
            )

        # 计算总体指标
        total_expected_impact = sum(s.expected_impact for s in slices) / len(slices)
        total_expected_cost = sum(s.expected_cost for s in slices)

        # 风险估计
        execution_risk = self._estimate_execution_risk(slices, parameters)
        timing_risk = self._estimate_timing_risk(duration_seconds, parameters)

        path = ExecutionPath(
            symbol=symbol,
            direction=direction,
            total_quantity=quantity,
            strategy=strategy,
            slices=slices,
            total_duration_seconds=duration_seconds,
            total_expected_impact=total_expected_impact,
            total_expected_cost=total_expected_cost,
            execution_risk=execution_risk,
            timing_risk=timing_risk,
        )

        return path

    async def track_realtime_impact(
        self,
        track_id: str,
        symbol: str,
        direction: TradeDirection,
        target_quantity: float,
        executed_quantity: float,
        initial_price: float,
        current_price: float,
        average_execution_price: float,
        elapsed_time_seconds: float,
        parameters: Optional[ImpactParameters] = None
    ) -> RealTimeImpact:
        """
        实时跟踪冲击

        Args:
            track_id: 跟踪ID
            symbol: 品种
            direction: 方向
            target_quantity: 目标数量
            executed_quantity: 已执行数量
            initial_price: 初始价格
            current_price: 当前价格
            average_execution_price: 平均执行价格
            elapsed_time_seconds: 已用时间
            parameters: 冲击参数

        Returns:
            实时冲击
        """
        # 获取参数
        if parameters is None:
            parameters = await self._get_parameters(symbol)

        remaining_quantity = target_quantity - executed_quantity
        execution_progress = executed_quantity / target_quantity if target_quantity > 0 else 0

        # 计算已实现冲击
        if direction == TradeDirection.BUY:
            realized_impact = ((average_execution_price - initial_price) / initial_price) * 10000
        else:
            realized_impact = ((initial_price - average_execution_price) / initial_price) * 10000

        # 估计剩余冲击
        if remaining_quantity > 0:
            estimate = await self.estimate_impact(
                symbol, direction, remaining_quantity, current_price, parameters
            )
            estimated_remaining_impact = estimate.estimated_impact
        else:
            estimated_remaining_impact = 0.0

        # 总预估冲击
        total_projected_impact = (
            realized_impact * execution_progress +
            estimated_remaining_impact * (1 - execution_progress)
        )

        # 成本
        realized_cost = executed_quantity * abs(average_execution_price - initial_price)
        estimated_total_cost = (
            realized_cost +
            remaining_quantity * current_price * estimated_remaining_impact / 10000
        )

        impact = RealTimeImpact(
            symbol=symbol,
            direction=direction,
            target_quantity=target_quantity,
            executed_quantity=executed_quantity,
            remaining_quantity=remaining_quantity,
            initial_price=initial_price,
            current_price=current_price,
            average_execution_price=average_execution_price,
            realized_impact=realized_impact,
            estimated_remaining_impact=estimated_remaining_impact,
            total_projected_impact=total_projected_impact,
            execution_progress=execution_progress,
            elapsed_time_seconds=elapsed_time_seconds,
            realized_cost=realized_cost,
            estimated_total_cost=estimated_total_cost,
        )

        # 缓存
        self._active_tracks[track_id] = impact

        return impact

    async def calibrate_parameters(
        self,
        symbol: str,
        historical_trades: List[Dict[str, Any]],
        model_type: Optional[ImpactModelType] = None
    ) -> ImpactParameters:
        """
        校准冲击参数

        Args:
            symbol: 品种
            historical_trades: 历史交易数据
            model_type: 模型类型

        Returns:
            校准后的参数
        """
        if len(historical_trades) < self.config.min_trades_for_calibration:
            logger.warning(
                f"Insufficient trades for calibration: {len(historical_trades)} < "
                f"{self.config.min_trades_for_calibration}"
            )
            # 返回默认参数
            return await self._get_default_parameters(symbol)

        # 提取数据
        impacts = []
        participation_rates = []
        volatilities = []

        for trade in historical_trades:
            if 'realized_impact' in trade and 'participation_rate' in trade:
                impacts.append(trade['realized_impact'])
                participation_rates.append(trade['participation_rate'])
                if 'volatility' in trade:
                    volatilities.append(trade['volatility'])

        if not impacts:
            return await self._get_default_parameters(symbol)

        # 校准模型参数 (简化的回归方法)
        avg_impact = np.mean(impacts)
        avg_participation = np.mean(participation_rates)
        avg_volatility = np.mean(volatilities) if volatilities else 0.20

        # 获取当前参数或创建新参数
        if symbol in self._parameters:
            params = self._parameters[symbol]
        else:
            params = await self._get_default_parameters(symbol)

        # 校准系数
        daily_vol = avg_volatility / math.sqrt(252)

        if model_type == ImpactModelType.LINEAR or model_type is None:
            if avg_participation > 0 and daily_vol > 0:
                params.linear_coefficient = avg_impact / (avg_participation * daily_vol * 10000)

        if model_type == ImpactModelType.SQUARE_ROOT or model_type is None:
            if avg_participation > 0 and daily_vol > 0:
                params.sqrt_coefficient = avg_impact / (
                    (avg_participation ** 0.5) * daily_vol * 10000
                )

        params.calibrated_at = datetime.now()

        # 保存参数
        self._parameters[symbol] = params

        logger.info(f"Calibrated parameters for {symbol}")

        return params

    async def analyze_scenarios(
        self,
        symbol: str,
        direction: TradeDirection,
        base_quantity: float,
        current_price: float,
        parameters: Optional[ImpactParameters] = None
    ) -> ScenarioAnalysis:
        """
        情景分析

        Args:
            symbol: 品种
            direction: 方向
            base_quantity: 基础数量
            current_price: 当前价格
            parameters: 冲击参数

        Returns:
            情景分析结果
        """
        # 获取参数
        if parameters is None:
            parameters = await self._get_parameters(symbol)

        # 定义情景
        scenarios = [
            {'name': 'aggressive', 'quantity': base_quantity, 'duration_minutes': 15},
            {'name': 'moderate', 'quantity': base_quantity, 'duration_minutes': 60},
            {'name': 'passive', 'quantity': base_quantity, 'duration_minutes': 240},
            {'name': 'split_half', 'quantity': base_quantity / 2, 'duration_minutes': 30},
            {'name': 'split_quarter', 'quantity': base_quantity / 4, 'duration_minutes': 15},
        ]

        # 分析每个情景
        results = []
        for scenario in scenarios:
            estimate = await self.estimate_impact(
                symbol,
                direction,
                scenario['quantity'],
                current_price,
                parameters
            )

            results.append({
                'name': scenario['name'],
                'quantity': scenario['quantity'],
                'duration_minutes': scenario['duration_minutes'],
                'estimated_impact': estimate.estimated_impact,
                'estimated_cost': estimate.estimated_cost_currency,
                'participation_rate': estimate.participation_rate,
                'confidence': estimate.confidence,
            })

        # 推荐最佳情景
        # 简单规则: 选择成本最低且置信度足够的情景
        valid_scenarios = [
            r for r in results
            if r['confidence'] >= 0.7 and r['participation_rate'] <= self.config.max_participation_rate
        ]

        if valid_scenarios:
            best = min(valid_scenarios, key=lambda x: x['estimated_cost'])
            recommended = best['name']
            reason = f"Lowest cost ({best['estimated_cost']:.2f}) with confidence {best['confidence']:.2f}"
        else:
            # 如果没有有效情景，选择最被动的
            best = max(results, key=lambda x: x['duration_minutes'])
            recommended = best['name']
            reason = "Most passive execution to ensure feasibility"

        analysis = ScenarioAnalysis(
            symbol=symbol,
            base_quantity=base_quantity,
            scenarios=scenarios,
            scenario_results=results,
            recommended_scenario=recommended,
            recommendation_reason=reason,
        )

        return analysis

    def set_parameters(self, symbol: str, parameters: ImpactParameters):
        """设置品种参数"""
        self._parameters[symbol] = parameters
        logger.info(f"Set parameters for {symbol}")

    def get_parameters(self, symbol: str) -> Optional[ImpactParameters]:
        """获取品种参数"""
        return self._parameters.get(symbol)

    def get_realtime_track(self, track_id: str) -> Optional[RealTimeImpact]:
        """获取实时跟踪"""
        return self._active_tracks.get(track_id)

    def clear_history(self, symbol: Optional[str] = None):
        """清除历史数据"""
        if symbol:
            self._historical_impacts[symbol].clear()
        else:
            self._historical_impacts.clear()
        logger.info(f"Cleared history for {symbol if symbol else 'all symbols'}")

    # ==================== 私有方法 ====================

    async def _get_parameters(self, symbol: str) -> ImpactParameters:
        """获取参数(如不存在则创建默认)"""
        if symbol in self._parameters:
            return self._parameters[symbol]
        return await self._get_default_parameters(symbol)

    async def _get_default_parameters(self, symbol: str) -> ImpactParameters:
        """获取默认参数"""
        params = ImpactParameters(
            symbol=symbol,
            price=100.0,
            daily_volume=1000000,
            volatility=0.30,
            spread_bps=5.0,
            linear_coefficient=0.1,
            sqrt_coefficient=0.1,
            sqrt_exponent=0.5,
            eta=0.001,
            gamma=0.0001,
            lambda_risk=2e-6,
            kyle_lambda=0.0001,
            liquidity_factor=1.0,
            market_condition=MarketCondition.NORMAL,
        )
        return params

    def _calculate_confidence(
        self,
        participation_rate: float,
        parameters: ImpactParameters
    ) -> float:
        """计算置信度"""
        # 置信度随参与率下降
        confidence = 1.0 - min(participation_rate * 2, 0.5)

        # 根据是否校准调整
        if parameters.calibrated_at:
            days_since_calibration = (
                datetime.now() - parameters.calibrated_at
            ).days
            if days_since_calibration <= 7:
                confidence = min(confidence + 0.1, 1.0)
        else:
            confidence *= 0.8  # 未校准的参数置信度降低

        return max(0.5, min(1.0, confidence))

    def _decompose_impact(
        self,
        total_impact: float,
        parameters: ImpactParameters,
        participation_rate: float
    ) -> ImpactDecomposition:
        """分解冲击"""
        # 价差成分 (约20-30%)
        spread_component = min(total_impact * 0.25, parameters.spread_bps)

        # 流动性成分 (与参与率相关, 约30-40%)
        liquidity_component = total_impact * 0.35 * min(participation_rate * 10, 1.0)

        # 信息成分 (约10-20%)
        information_component = total_impact * 0.15

        # 波动性成分 (剩余)
        volatility_component = total_impact - (
            spread_component + liquidity_component + information_component
        )

        # 临时/永久分解 (70/30)
        temporary = total_impact * 0.7
        permanent = total_impact * 0.3

        return ImpactDecomposition(
            temporary_impact=temporary,
            permanent_impact=permanent,
            total_impact=total_impact,
            spread_component=spread_component,
            liquidity_component=liquidity_component,
            information_component=information_component,
            volatility_component=volatility_component,
        )

    def _generate_twap_slices(
        self,
        quantity: float,
        duration_seconds: float,
        current_price: float,
        parameters: ImpactParameters,
        direction: TradeDirection
    ) -> List[ExecutionSlice]:
        """生成TWAP切片"""
        # 计算切片数
        n_slices = min(
            max(int(duration_seconds / 300), 1),  # 每5分钟一片
            self.config.max_slices
        )

        slice_quantity = quantity / n_slices
        slice_duration = duration_seconds / n_slices

        slices = []
        for i in range(n_slices):
            # 估计该切片的冲击
            model = self._impact_models[self.config.default_model]
            total_impact, temp, perm = model.estimate_impact(
                slice_quantity, parameters, direction
            )

            expected_cost = slice_quantity * current_price * total_impact / 10000

            slices.append(ExecutionSlice(
                slice_number=i + 1,
                quantity=slice_quantity,
                start_time=i * slice_duration,
                end_time=(i + 1) * slice_duration,
                expected_price=current_price,
                expected_impact=total_impact,
                expected_cost=expected_cost,
            ))

        return slices

    def _generate_vwap_slices(
        self,
        quantity: float,
        duration_seconds: float,
        current_price: float,
        parameters: ImpactParameters,
        direction: TradeDirection
    ) -> List[ExecutionSlice]:
        """生成VWAP切片 (简化: 与TWAP相同)"""
        return self._generate_twap_slices(
            quantity, duration_seconds, current_price, parameters, direction
        )

    def _generate_pov_slices(
        self,
        quantity: float,
        duration_seconds: float,
        current_price: float,
        parameters: ImpactParameters,
        direction: TradeDirection
    ) -> List[ExecutionSlice]:
        """生成POV切片"""
        # POV: 保持固定参与率
        target_pov = 0.05  # 5%参与率

        # 估计需要的时间
        estimated_duration = quantity / (parameters.daily_volume * target_pov)
        n_slices = min(
            max(int(estimated_duration * 10), 1),
            self.config.max_slices
        )

        slice_quantity = quantity / n_slices
        slice_duration = duration_seconds / n_slices

        slices = []
        for i in range(n_slices):
            model = self._impact_models[self.config.default_model]
            total_impact, temp, perm = model.estimate_impact(
                slice_quantity, parameters, direction
            )

            expected_cost = slice_quantity * current_price * total_impact / 10000

            slices.append(ExecutionSlice(
                slice_number=i + 1,
                quantity=slice_quantity,
                start_time=i * slice_duration,
                end_time=(i + 1) * slice_duration,
                expected_price=current_price,
                expected_impact=total_impact,
                expected_cost=expected_cost,
            ))

        return slices

    def _generate_is_slices(
        self,
        quantity: float,
        duration_seconds: float,
        current_price: float,
        parameters: ImpactParameters,
        direction: TradeDirection
    ) -> List[ExecutionSlice]:
        """生成Implementation Shortfall切片 (前端加载)"""
        # IS策略: 前端执行更多
        n_slices = min(
            max(int(duration_seconds / 180), 3),  # 每3分钟一片
            self.config.max_slices
        )

        # 前端权重更高
        weights = [1.0 / (i + 1) for i in range(n_slices)]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        slices = []
        current_time = 0
        slice_duration = duration_seconds / n_slices

        for i in range(n_slices):
            slice_quantity = quantity * weights[i]

            model = self._impact_models[self.config.default_model]
            total_impact, temp, perm = model.estimate_impact(
                slice_quantity, parameters, direction
            )

            expected_cost = slice_quantity * current_price * total_impact / 10000

            slices.append(ExecutionSlice(
                slice_number=i + 1,
                quantity=slice_quantity,
                start_time=current_time,
                end_time=current_time + slice_duration,
                expected_price=current_price,
                expected_impact=total_impact,
                expected_cost=expected_cost,
            ))

            current_time += slice_duration

        return slices

    def _estimate_execution_risk(
        self,
        slices: List[ExecutionSlice],
        parameters: ImpactParameters
    ) -> float:
        """估计执行风险"""
        # 执行风险 = 方差 * 波动率
        total_quantity = sum(s.quantity for s in slices)
        quantity_variance = np.var([s.quantity for s in slices])

        daily_vol = parameters.volatility / math.sqrt(252)

        risk = math.sqrt(quantity_variance) * daily_vol * 10000

        return risk

    def _estimate_timing_risk(
        self,
        duration_seconds: float,
        parameters: ImpactParameters
    ) -> float:
        """估计时机风险"""
        # 时机风险 = sqrt(duration) * volatility
        duration_days = duration_seconds / 86400
        daily_vol = parameters.volatility / math.sqrt(252)

        risk = math.sqrt(duration_days) * daily_vol * 10000

        return risk


# ==================== 工厂函数 ====================

def create_impact_estimator(
    default_model: str = "square_root",
    auto_calibrate: bool = True,
    **kwargs
) -> MarketImpactEstimator:
    """
    创建市场冲击估算器

    Args:
        default_model: 默认模型类型
        auto_calibrate: 自动校准
        **kwargs: 其他配置参数

    Returns:
        MarketImpactEstimator实例
    """
    config = EstimatorConfig(
        default_model=ImpactModelType(default_model),
        auto_calibrate=auto_calibrate,
        **{k: v for k, v in kwargs.items() if hasattr(EstimatorConfig, k)}
    )

    return MarketImpactEstimator(config=config)


def create_impact_parameters(
    symbol: str,
    price: float,
    daily_volume: float,
    volatility: float,
    spread_bps: float = 5.0,
    **kwargs
) -> ImpactParameters:
    """
    创建冲击参数

    Args:
        symbol: 品种
        price: 当前价格
        daily_volume: 日均成交量
        volatility: 波动率
        spread_bps: 买卖价差
        **kwargs: 其他参数

    Returns:
        ImpactParameters实例
    """
    return ImpactParameters(
        symbol=symbol,
        price=price,
        daily_volume=daily_volume,
        volatility=volatility,
        spread_bps=spread_bps,
        **{k: v for k, v in kwargs.items() if hasattr(ImpactParameters, k)}
    )


async def quick_impact_estimate(
    symbol: str,
    direction: str,
    quantity: float,
    price: float,
    daily_volume: float,
    volatility: float,
    model: str = "square_root"
) -> ImpactEstimate:
    """
    快速冲击估算

    Args:
        symbol: 品种
        direction: 方向 ('buy' or 'sell')
        quantity: 数量
        price: 价格
        daily_volume: 日均成交量
        volatility: 波动率
        model: 模型类型

    Returns:
        冲击估计
    """
    estimator = create_impact_estimator(default_model=model)

    parameters = create_impact_parameters(
        symbol=symbol,
        price=price,
        daily_volume=daily_volume,
        volatility=volatility,
    )

    return await estimator.estimate_impact(
        symbol=symbol,
        direction=TradeDirection(direction),
        quantity=quantity,
        current_price=price,
        parameters=parameters,
    )
