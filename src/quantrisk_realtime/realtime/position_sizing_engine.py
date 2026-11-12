"""
仓位规模计算引擎 (Position Sizing Engine)

机构级仓位管理和风险控制系统，提供:
- Kelly Criterion 凯利公式计算
- Fixed Fractional 固定比例法
- Volatility-based sizing 波动率调整
- Risk parity sizing 风险平价
- Maximum drawdown constraint 最大回撤约束
- Position limits management 仓位限制管理
- Portfolio-level sizing 组合级别计算
- Real-time size adjustment 实时调整

作者: TraderX 量化团队
Copyright (c) 2025 TraderX. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import logging
import math
import statistics
import uuid

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class SizingMethod(str, Enum):
    """仓位规模计算方法"""
    KELLY = "kelly"                         # 凯利公式
    FIXED_FRACTIONAL = "fixed_fractional"   # 固定比例
    VOLATILITY = "volatility"               # 波动率调整
    RISK_PARITY = "risk_parity"             # 风险平价
    EQUAL_WEIGHT = "equal_weight"           # 等权重
    OPTIMAL_F = "optimal_f"                 # 最优f值
    ATR_BASED = "atr_based"                 # ATR基准
    DRAWDOWN_CONSTRAINED = "drawdown"       # 回撤约束


class PositionType(str, Enum):
    """持仓类型"""
    LONG = "long"           # 多头
    SHORT = "short"         # 空头
    NEUTRAL = "neutral"     # 中性


class RiskLevel(str, Enum):
    """风险等级"""
    CONSERVATIVE = "conservative"   # 保守
    MODERATE = "moderate"           # 适中
    AGGRESSIVE = "aggressive"       # 激进
    CUSTOM = "custom"               # 自定义


class ConstraintType(str, Enum):
    """约束类型"""
    MAX_POSITION = "max_position"           # 最大仓位
    MIN_POSITION = "min_position"           # 最小仓位
    MAX_LEVERAGE = "max_leverage"           # 最大杠杆
    MAX_PORTFOLIO = "max_portfolio"         # 最大组合占比
    MAX_SECTOR = "max_sector"               # 最大行业占比
    MAX_DRAWDOWN = "max_drawdown"           # 最大回撤
    POSITION_CORRELATION = "correlation"    # 相关性限制


class AdjustmentTrigger(str, Enum):
    """调整触发条件"""
    VOLATILITY_CHANGE = "volatility_change"     # 波动率变化
    PRICE_CHANGE = "price_change"               # 价格变化
    DRAWDOWN = "drawdown"                       # 回撤
    PROFIT_TARGET = "profit_target"             # 盈利目标
    TIME_BASED = "time_based"                   # 时间基准
    RISK_CHANGE = "risk_change"                 # 风险变化
    CORRELATION_SHIFT = "correlation_shift"     # 相关性变化


class EngineState(str, Enum):
    """引擎状态"""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


# ==================== 数据类 ====================

@dataclass
class AssetMetrics:
    """资产指标"""
    symbol: str

    # 价格数据
    current_price: float
    entry_price: Optional[float] = None

    # 收益统计
    expected_return: float = 0.0        # 期望收益率
    win_rate: float = 0.5               # 胜率
    avg_win: float = 0.0                # 平均盈利
    avg_loss: float = 0.0               # 平均亏损
    profit_factor: float = 1.0          # 盈亏比

    # 风险指标
    volatility: float = 0.0             # 波动率(年化)
    daily_volatility: float = 0.0       # 日波动率
    max_drawdown: float = 0.0           # 最大回撤
    beta: float = 1.0                   # Beta系数
    sharpe_ratio: float = 0.0           # 夏普比率

    # ATR数据
    atr: float = 0.0                    # 平均真实波幅
    atr_percent: float = 0.0            # ATR百分比

    # 相关性
    correlation_to_portfolio: float = 0.0

    # 元数据
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'entry_price': self.entry_price,
            'expected_return': self.expected_return,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'volatility': self.volatility,
            'daily_volatility': self.daily_volatility,
            'max_drawdown': self.max_drawdown,
            'beta': self.beta,
            'sharpe_ratio': self.sharpe_ratio,
            'atr': self.atr,
            'atr_percent': self.atr_percent,
            'correlation_to_portfolio': self.correlation_to_portfolio,
            'last_updated': self.last_updated.isoformat(),
        }


@dataclass
class SizingResult:
    """仓位规模计算结果"""
    calculation_id: str
    symbol: str
    method: SizingMethod

    # 仓位建议
    suggested_size: float               # 建议仓位大小(合约数/股数)
    suggested_value: float              # 建议仓位价值
    suggested_weight: float             # 建议权重(占组合比例)

    # 风险参数
    risk_per_trade: float               # 单笔风险金额
    risk_percent: float                 # 风险百分比
    stop_loss_price: Optional[float] = None
    stop_loss_percent: float = 0.0

    # Kelly相关
    kelly_fraction: Optional[float] = None
    half_kelly: Optional[float] = None
    quarter_kelly: Optional[float] = None

    # 约束检查
    constraints_applied: List[str] = field(default_factory=list)
    is_constrained: bool = False
    original_size: Optional[float] = None

    # 置信度
    confidence: float = 1.0             # 计算置信度(0-1)

    # 计算时间
    calculated_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'calculation_id': self.calculation_id,
            'symbol': self.symbol,
            'method': self.method.value,
            'suggested_size': self.suggested_size,
            'suggested_value': self.suggested_value,
            'suggested_weight': self.suggested_weight,
            'risk_per_trade': self.risk_per_trade,
            'risk_percent': self.risk_percent,
            'stop_loss_price': self.stop_loss_price,
            'stop_loss_percent': self.stop_loss_percent,
            'kelly_fraction': self.kelly_fraction,
            'half_kelly': self.half_kelly,
            'quarter_kelly': self.quarter_kelly,
            'constraints_applied': self.constraints_applied,
            'is_constrained': self.is_constrained,
            'original_size': self.original_size,
            'confidence': self.confidence,
            'calculated_at': self.calculated_at.isoformat(),
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
        }


@dataclass
class PortfolioSizing:
    """组合级别仓位配置"""
    sizing_id: str

    # 资产配置
    allocations: Dict[str, float]       # symbol -> weight
    position_sizes: Dict[str, float]    # symbol -> size
    position_values: Dict[str, float]   # symbol -> value

    # 组合指标
    total_exposure: float               # 总敞口
    net_exposure: float                 # 净敞口
    leverage: float                     # 杠杆
    diversification_score: float = 0.0  # 分散化评分

    # 风险平价相关
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    target_risk: Optional[float] = None

    # 调整建议
    rebalance_required: bool = False
    rebalance_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    # 时间戳
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sizing_id': self.sizing_id,
            'allocations': self.allocations,
            'position_sizes': self.position_sizes,
            'position_values': self.position_values,
            'total_exposure': self.total_exposure,
            'net_exposure': self.net_exposure,
            'leverage': self.leverage,
            'diversification_score': self.diversification_score,
            'risk_contributions': self.risk_contributions,
            'target_risk': self.target_risk,
            'rebalance_required': self.rebalance_required,
            'rebalance_suggestions': self.rebalance_suggestions,
            'calculated_at': self.calculated_at.isoformat(),
        }


@dataclass
class PositionConstraint:
    """仓位约束"""
    constraint_type: ConstraintType
    value: float
    is_hard_limit: bool = True          # 硬限制(违反时拒绝) vs 软限制(警告)
    description: str = ""

    def check(self, current_value: float) -> Tuple[bool, str]:
        """检查是否违反约束"""
        violated = False
        message = ""

        if self.constraint_type == ConstraintType.MAX_POSITION:
            if current_value > self.value:
                violated = True
                message = f"Position exceeds max limit: {current_value:.2f} > {self.value:.2f}"

        elif self.constraint_type == ConstraintType.MIN_POSITION:
            if current_value < self.value:
                violated = True
                message = f"Position below min limit: {current_value:.2f} < {self.value:.2f}"

        elif self.constraint_type == ConstraintType.MAX_LEVERAGE:
            if current_value > self.value:
                violated = True
                message = f"Leverage exceeds max: {current_value:.2f} > {self.value:.2f}"

        elif self.constraint_type == ConstraintType.MAX_DRAWDOWN:
            if current_value > self.value:
                violated = True
                message = f"Drawdown exceeds max: {current_value:.2%} > {self.value:.2%}"

        return violated, message

    def to_dict(self) -> Dict[str, Any]:
        return {
            'constraint_type': self.constraint_type.value,
            'value': self.value,
            'is_hard_limit': self.is_hard_limit,
            'description': self.description,
        }


@dataclass
class AdjustmentSignal:
    """仓位调整信号"""
    signal_id: str
    symbol: str
    trigger: AdjustmentTrigger

    # 当前状态
    current_size: float
    current_value: float

    # 建议调整
    suggested_size: float
    suggested_value: float
    adjustment_pct: float               # 调整百分比

    # 触发原因
    reason: str
    trigger_value: float
    threshold_value: float

    # 紧急程度
    urgency: str = "normal"             # low/normal/high/critical

    # 时间戳
    triggered_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'trigger': self.trigger.value,
            'current_size': self.current_size,
            'current_value': self.current_value,
            'suggested_size': self.suggested_size,
            'suggested_value': self.suggested_value,
            'adjustment_pct': self.adjustment_pct,
            'reason': self.reason,
            'trigger_value': self.trigger_value,
            'threshold_value': self.threshold_value,
            'urgency': self.urgency,
            'triggered_at': self.triggered_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class SizingConfig:
    """仓位计算配置"""
    # 基础参数
    account_size: float = 100000.0      # 账户规模
    risk_per_trade: float = 0.02        # 单笔风险(2%)
    max_position_size: float = 0.25     # 最大单个仓位(25%)
    max_portfolio_heat: float = 0.06    # 最大组合风险(6%)

    # Kelly参数
    kelly_multiplier: float = 0.25      # Kelly乘数(通常用1/4 Kelly)
    kelly_min_trades: int = 20          # 最小交易次数

    # 波动率参数
    volatility_target: float = 0.15     # 目标波动率(15%)
    volatility_lookback: int = 20       # 波动率计算周期

    # 风险平价参数
    risk_parity_enabled: bool = False
    target_risk_contribution: float = 0.1  # 目标风险贡献

    # 回撤约束
    max_drawdown_pct: float = 0.20      # 最大回撤(20%)
    drawdown_scale_factor: float = 0.5  # 回撤缩减系数

    # ATR参数
    atr_multiplier: float = 2.0         # ATR止损倍数
    atr_period: int = 14                # ATR计算周期

    # 调整阈值
    rebalance_threshold: float = 0.05   # 再平衡阈值(5%)
    volatility_adjust_threshold: float = 0.20  # 波动率调整阈值(20%)

    # 风险等级
    risk_level: RiskLevel = RiskLevel.MODERATE

    # 其他设置
    allow_fractional_shares: bool = True
    round_to_lot_size: bool = False
    lot_size: int = 100
    min_position_value: float = 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'account_size': self.account_size,
            'risk_per_trade': self.risk_per_trade,
            'max_position_size': self.max_position_size,
            'max_portfolio_heat': self.max_portfolio_heat,
            'kelly_multiplier': self.kelly_multiplier,
            'volatility_target': self.volatility_target,
            'max_drawdown_pct': self.max_drawdown_pct,
            'risk_level': self.risk_level.value,
        }


# ==================== 计算器基类 ====================

class PositionSizer:
    """仓位计算器基类"""

    def __init__(self, config: SizingConfig):
        self.config = config

    def calculate(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_position: float = 0.0
    ) -> SizingResult:
        """计算仓位规模"""
        raise NotImplementedError

    def _apply_constraints(
        self,
        size: float,
        constraints: List[PositionConstraint]
    ) -> Tuple[float, List[str]]:
        """应用约束条件"""
        original_size = size
        applied = []

        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.MAX_POSITION:
                if size > constraint.value:
                    size = constraint.value
                    applied.append(f"max_position:{constraint.value}")

            elif constraint.constraint_type == ConstraintType.MIN_POSITION:
                if 0 < size < constraint.value:
                    size = constraint.value
                    applied.append(f"min_position:{constraint.value}")

        return size, applied


# ==================== 具体计算器实现 ====================

class KellyCriterionSizer(PositionSizer):
    """凯利公式仓位计算器"""

    def calculate(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_position: float = 0.0
    ) -> SizingResult:
        """
        Kelly公式: f* = (p*W - (1-p)*L) / (W*L)
        其中: p=胜率, W=平均盈利, L=平均亏损
        """
        # 检查数据有效性
        if metrics.win_rate <= 0 or metrics.win_rate >= 1:
            metrics.win_rate = 0.5

        if metrics.avg_win <= 0:
            metrics.avg_win = 0.02

        if metrics.avg_loss <= 0:
            metrics.avg_loss = 0.01

        # Kelly公式计算
        p = metrics.win_rate
        W = abs(metrics.avg_win)
        L = abs(metrics.avg_loss)

        kelly_fraction = (p * W - (1 - p) * L) / (W * L) if W * L > 0 else 0.0

        # 限制Kelly值范围
        kelly_fraction = max(-0.5, min(kelly_fraction, 1.0))

        # 应用Kelly乘数(通常使用1/4或1/2 Kelly)
        adjusted_kelly = kelly_fraction * self.config.kelly_multiplier

        # 计算仓位
        position_value = self.config.account_size * adjusted_kelly
        position_size = position_value / metrics.current_price if metrics.current_price > 0 else 0

        # 风险计算
        risk_per_trade = self.config.account_size * self.config.risk_per_trade
        risk_percent = adjusted_kelly

        return SizingResult(
            calculation_id=str(uuid.uuid4()),
            symbol=symbol,
            method=SizingMethod.KELLY,
            suggested_size=position_size,
            suggested_value=position_value,
            suggested_weight=adjusted_kelly,
            risk_per_trade=risk_per_trade,
            risk_percent=risk_percent,
            kelly_fraction=kelly_fraction,
            half_kelly=kelly_fraction * 0.5,
            quarter_kelly=kelly_fraction * 0.25,
            confidence=0.7 if kelly_fraction > 0 else 0.3,
        )


class FixedFractionalSizer(PositionSizer):
    """固定比例仓位计算器"""

    def calculate(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_position: float = 0.0
    ) -> SizingResult:
        """固定比例法: 每笔交易风险固定账户的X%"""

        # 计算风险金额
        risk_amount = self.config.account_size * self.config.risk_per_trade

        # 计算止损距离
        if metrics.entry_price and metrics.current_price > 0:
            stop_loss_distance = abs(metrics.current_price - metrics.entry_price)
        elif metrics.atr > 0:
            stop_loss_distance = metrics.atr * self.config.atr_multiplier
        else:
            # 使用波动率估算
            stop_loss_distance = metrics.current_price * metrics.daily_volatility * 2

        if stop_loss_distance <= 0:
            stop_loss_distance = metrics.current_price * 0.02  # 默认2%

        # 仓位大小 = 风险金额 / 止损距离
        position_size = risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0
        position_value = position_size * metrics.current_price

        # 限制最大仓位
        max_value = self.config.account_size * self.config.max_position_size
        if position_value > max_value:
            position_value = max_value
            position_size = position_value / metrics.current_price

        # 计算止损价格
        stop_loss_price = metrics.current_price - stop_loss_distance
        stop_loss_percent = stop_loss_distance / metrics.current_price if metrics.current_price > 0 else 0

        return SizingResult(
            calculation_id=str(uuid.uuid4()),
            symbol=symbol,
            method=SizingMethod.FIXED_FRACTIONAL,
            suggested_size=position_size,
            suggested_value=position_value,
            suggested_weight=position_value / self.config.account_size if self.config.account_size > 0 else 0,
            risk_per_trade=risk_amount,
            risk_percent=self.config.risk_per_trade,
            stop_loss_price=stop_loss_price,
            stop_loss_percent=stop_loss_percent,
            confidence=0.9,
        )


class VolatilityBasedSizer(PositionSizer):
    """波动率调整仓位计算器"""

    def calculate(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_position: float = 0.0
    ) -> SizingResult:
        """根据波动率调整仓位: 波动率越高，仓位越小"""

        # 获取波动率
        volatility = metrics.daily_volatility if metrics.daily_volatility > 0 else metrics.volatility / math.sqrt(252)

        if volatility <= 0:
            volatility = 0.01  # 默认1%日波动

        # 目标波动率
        target_vol = self.config.volatility_target / math.sqrt(252)  # 转换为日波动率

        # 仓位缩放因子
        scale_factor = target_vol / volatility
        scale_factor = min(scale_factor, 3.0)  # 限制最大3倍

        # 基础仓位(使用固定比例作为基础)
        base_weight = self.config.risk_per_trade * 5  # 基础权重
        adjusted_weight = base_weight * scale_factor

        # 限制在最大仓位内
        adjusted_weight = min(adjusted_weight, self.config.max_position_size)

        # 计算仓位
        position_value = self.config.account_size * adjusted_weight
        position_size = position_value / metrics.current_price if metrics.current_price > 0 else 0

        # 风险计算
        position_volatility = volatility * math.sqrt(252)
        risk_percent = adjusted_weight * position_volatility

        return SizingResult(
            calculation_id=str(uuid.uuid4()),
            symbol=symbol,
            method=SizingMethod.VOLATILITY,
            suggested_size=position_size,
            suggested_value=position_value,
            suggested_weight=adjusted_weight,
            risk_per_trade=self.config.account_size * risk_percent,
            risk_percent=risk_percent,
            confidence=0.8,
        )


class RiskParitySizer(PositionSizer):
    """风险平价仓位计算器"""

    def calculate(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_position: float = 0.0
    ) -> SizingResult:
        """风险平价: 每个资产对组合的风险贡献相等"""

        # 获取波动率
        volatility = metrics.volatility if metrics.volatility > 0 else 0.15

        # 目标风险贡献
        target_risk = self.config.target_risk_contribution

        # 根据风险平价原理: weight = target_risk / volatility
        weight = target_risk / volatility if volatility > 0 else 0

        # 限制权重
        weight = min(weight, self.config.max_position_size)

        # 计算仓位
        position_value = self.config.account_size * weight
        position_size = position_value / metrics.current_price if metrics.current_price > 0 else 0

        # 风险贡献
        risk_contribution = weight * volatility

        return SizingResult(
            calculation_id=str(uuid.uuid4()),
            symbol=symbol,
            method=SizingMethod.RISK_PARITY,
            suggested_size=position_size,
            suggested_value=position_value,
            suggested_weight=weight,
            risk_per_trade=self.config.account_size * risk_contribution,
            risk_percent=risk_contribution,
            confidence=0.75,
        )


class EqualWeightSizer(PositionSizer):
    """等权重仓位计算器"""

    def __init__(self, config: SizingConfig, num_positions: int = 10):
        super().__init__(config)
        self.num_positions = num_positions

    def calculate(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_position: float = 0.0
    ) -> SizingResult:
        """等权重: 每个持仓占组合的相同比例"""

        # 等权重
        weight = 1.0 / self.num_positions if self.num_positions > 0 else 0
        weight = min(weight, self.config.max_position_size)

        # 计算仓位
        position_value = self.config.account_size * weight
        position_size = position_value / metrics.current_price if metrics.current_price > 0 else 0

        return SizingResult(
            calculation_id=str(uuid.uuid4()),
            symbol=symbol,
            method=SizingMethod.EQUAL_WEIGHT,
            suggested_size=position_size,
            suggested_value=position_value,
            suggested_weight=weight,
            risk_per_trade=position_value * metrics.volatility,
            risk_percent=weight * metrics.volatility,
            confidence=1.0,
        )


class ATRBasedSizer(PositionSizer):
    """ATR基准仓位计算器"""

    def calculate(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_position: float = 0.0
    ) -> SizingResult:
        """基于ATR的仓位计算"""

        # 获取ATR
        atr = metrics.atr if metrics.atr > 0 else metrics.current_price * 0.02

        # 风险金额
        risk_amount = self.config.account_size * self.config.risk_per_trade

        # 止损距离 = ATR * 倍数
        stop_distance = atr * self.config.atr_multiplier

        # 仓位大小 = 风险金额 / 止损距离
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        position_value = position_size * metrics.current_price

        # 限制最大仓位
        max_value = self.config.account_size * self.config.max_position_size
        if position_value > max_value:
            position_value = max_value
            position_size = position_value / metrics.current_price

        # 止损价格
        stop_loss_price = metrics.current_price - stop_distance
        stop_loss_percent = stop_distance / metrics.current_price if metrics.current_price > 0 else 0

        return SizingResult(
            calculation_id=str(uuid.uuid4()),
            symbol=symbol,
            method=SizingMethod.ATR_BASED,
            suggested_size=position_size,
            suggested_value=position_value,
            suggested_weight=position_value / self.config.account_size if self.config.account_size > 0 else 0,
            risk_per_trade=risk_amount,
            risk_percent=self.config.risk_per_trade,
            stop_loss_price=stop_loss_price,
            stop_loss_percent=stop_loss_percent,
            confidence=0.85,
        )


class DrawdownConstrainedSizer(PositionSizer):
    """回撤约束仓位计算器"""

    def calculate(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_position: float = 0.0
    ) -> SizingResult:
        """根据回撤情况动态调整仓位"""

        # 基础仓位(使用固定比例)
        base_sizer = FixedFractionalSizer(self.config)
        base_result = base_sizer.calculate(symbol, metrics, current_position)

        # 回撤调整因子
        if metrics.max_drawdown > self.config.max_drawdown_pct:
            # 超过最大回撤，大幅减仓
            drawdown_factor = self.config.drawdown_scale_factor
        elif metrics.max_drawdown > self.config.max_drawdown_pct * 0.5:
            # 接近最大回撤，适度减仓
            drawdown_factor = 1.0 - (metrics.max_drawdown / self.config.max_drawdown_pct) * 0.5
        else:
            # 回撤可接受
            drawdown_factor = 1.0

        # 调整仓位
        adjusted_size = base_result.suggested_size * drawdown_factor
        adjusted_value = base_result.suggested_value * drawdown_factor
        adjusted_weight = base_result.suggested_weight * drawdown_factor

        result = SizingResult(
            calculation_id=str(uuid.uuid4()),
            symbol=symbol,
            method=SizingMethod.DRAWDOWN_CONSTRAINED,
            suggested_size=adjusted_size,
            suggested_value=adjusted_value,
            suggested_weight=adjusted_weight,
            risk_per_trade=base_result.risk_per_trade * drawdown_factor,
            risk_percent=base_result.risk_percent,
            stop_loss_price=base_result.stop_loss_price,
            stop_loss_percent=base_result.stop_loss_percent,
            original_size=base_result.suggested_size,
            is_constrained=drawdown_factor < 1.0,
            constraints_applied=[f"drawdown_factor:{drawdown_factor:.2f}"],
            confidence=0.75,
        )

        return result


# ==================== 核心引擎 ====================

class PositionSizingEngine:
    """
    仓位规模计算引擎

    功能:
    - 多种仓位计算方法
    - 约束条件管理
    - 组合级别计算
    - 实时调整监控
    """

    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig()

        # 计算器
        self._sizers: Dict[SizingMethod, PositionSizer] = {
            SizingMethod.KELLY: KellyCriterionSizer(self.config),
            SizingMethod.FIXED_FRACTIONAL: FixedFractionalSizer(self.config),
            SizingMethod.VOLATILITY: VolatilityBasedSizer(self.config),
            SizingMethod.RISK_PARITY: RiskParitySizer(self.config),
            SizingMethod.EQUAL_WEIGHT: EqualWeightSizer(self.config),
            SizingMethod.ATR_BASED: ATRBasedSizer(self.config),
            SizingMethod.DRAWDOWN_CONSTRAINED: DrawdownConstrainedSizer(self.config),
        }

        # 数据存储
        self._asset_metrics: Dict[str, AssetMetrics] = {}
        self._current_positions: Dict[str, float] = {}
        self._constraints: Dict[str, List[PositionConstraint]] = {}
        self._sizing_history: List[SizingResult] = []
        self._adjustment_signals: List[AdjustmentSignal] = []

        # 组合数据
        self._portfolio_value: float = self.config.account_size
        self._peak_value: float = self.config.account_size
        self._current_drawdown: float = 0.0

        # 状态
        self._state = EngineState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        # 回调
        self._on_sizing_calculated: Optional[Callable[[SizingResult], None]] = None
        self._on_adjustment_signal: Optional[Callable[[AdjustmentSignal], None]] = None

        # 统计
        self.stats = {
            'calculations_count': 0,
            'adjustments_triggered': 0,
            'constraints_violations': 0,
        }

        logger.info("PositionSizingEngine initialized")

    # ==================== 资产管理 ====================

    def update_asset_metrics(self, metrics: AssetMetrics):
        """更新资产指标"""
        self._asset_metrics[metrics.symbol] = metrics

    def get_asset_metrics(self, symbol: str) -> Optional[AssetMetrics]:
        """获取资产指标"""
        return self._asset_metrics.get(symbol)

    def update_position(self, symbol: str, size: float):
        """更新当前持仓"""
        self._current_positions[symbol] = size

    def get_position(self, symbol: str) -> float:
        """获取当前持仓"""
        return self._current_positions.get(symbol, 0.0)

    def update_portfolio_value(self, value: float):
        """更新组合价值"""
        self._portfolio_value = value

        # 更新峰值和回撤
        if value > self._peak_value:
            self._peak_value = value

        if self._peak_value > 0:
            self._current_drawdown = (self._peak_value - value) / self._peak_value

    # ==================== 约束管理 ====================

    def add_constraint(self, symbol: str, constraint: PositionConstraint):
        """添加约束"""
        if symbol not in self._constraints:
            self._constraints[symbol] = []
        self._constraints[symbol].append(constraint)

    def remove_constraint(self, symbol: str, constraint_type: ConstraintType):
        """移除约束"""
        if symbol in self._constraints:
            self._constraints[symbol] = [
                c for c in self._constraints[symbol]
                if c.constraint_type != constraint_type
            ]

    def get_constraints(self, symbol: str) -> List[PositionConstraint]:
        """获取约束"""
        return self._constraints.get(symbol, [])

    def check_constraints(
        self,
        symbol: str,
        proposed_size: float
    ) -> Tuple[bool, List[str]]:
        """检查约束条件"""
        constraints = self.get_constraints(symbol)
        violations = []

        for constraint in constraints:
            violated, message = constraint.check(proposed_size)
            if violated:
                violations.append(message)
                if constraint.is_hard_limit:
                    self.stats['constraints_violations'] += 1

        return len(violations) == 0, violations

    # ==================== 仓位计算 ====================

    def calculate_size(
        self,
        symbol: str,
        method: Optional[SizingMethod] = None,
        metrics: Optional[AssetMetrics] = None
    ) -> SizingResult:
        """计算仓位规模"""

        # 获取资产指标
        if metrics is None:
            metrics = self.get_asset_metrics(symbol)

        if metrics is None:
            raise ValueError(f"No metrics available for {symbol}")

        # 选择计算方法
        if method is None:
            method = self._select_best_method(metrics)

        # 获取计算器
        sizer = self._sizers.get(method)
        if sizer is None:
            raise ValueError(f"Unknown sizing method: {method}")

        # 计算基础仓位
        current_position = self.get_position(symbol)
        result = sizer.calculate(symbol, metrics, current_position)

        # 应用约束
        constraints = self.get_constraints(symbol)
        if constraints:
            original_size = result.suggested_size
            adjusted_size, applied = sizer._apply_constraints(result.suggested_size, constraints)

            if adjusted_size != original_size:
                result.original_size = original_size
                result.suggested_size = adjusted_size
                result.suggested_value = adjusted_size * metrics.current_price
                result.suggested_weight = result.suggested_value / self.config.account_size
                result.is_constrained = True
                result.constraints_applied = applied

        # 应用回撤约束
        if self._current_drawdown > 0:
            drawdown_factor = self._calculate_drawdown_factor()
            if drawdown_factor < 1.0:
                if result.original_size is None:
                    result.original_size = result.suggested_size
                result.suggested_size *= drawdown_factor
                result.suggested_value *= drawdown_factor
                result.suggested_weight *= drawdown_factor
                result.is_constrained = True
                result.constraints_applied.append(f"drawdown_factor:{drawdown_factor:.2f}")

        # 圆整
        if not self.config.allow_fractional_shares:
            result.suggested_size = math.floor(result.suggested_size)

        if self.config.round_to_lot_size:
            lot_size = self.config.lot_size
            result.suggested_size = math.floor(result.suggested_size / lot_size) * lot_size

        # 检查最小仓位价值
        if result.suggested_value < self.config.min_position_value:
            result.suggested_size = 0
            result.suggested_value = 0
            result.suggested_weight = 0

        # 记录
        self._sizing_history.append(result)
        self.stats['calculations_count'] += 1

        # 回调
        if self._on_sizing_calculated:
            self._on_sizing_calculated(result)

        return result

    def calculate_multiple(
        self,
        symbols: List[str],
        method: Optional[SizingMethod] = None
    ) -> Dict[str, SizingResult]:
        """批量计算仓位"""
        results = {}
        for symbol in symbols:
            try:
                result = self.calculate_size(symbol, method)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Failed to calculate size for {symbol}: {e}")
        return results

    def _select_best_method(self, metrics: AssetMetrics) -> SizingMethod:
        """选择最佳计算方法"""
        # 根据风险等级选择
        if self.config.risk_level == RiskLevel.CONSERVATIVE:
            return SizingMethod.FIXED_FRACTIONAL
        elif self.config.risk_level == RiskLevel.AGGRESSIVE:
            if metrics.win_rate > 0.5 and metrics.profit_factor > 1.5:
                return SizingMethod.KELLY
            return SizingMethod.VOLATILITY
        else:  # MODERATE
            if metrics.atr > 0:
                return SizingMethod.ATR_BASED
            return SizingMethod.FIXED_FRACTIONAL

    def _calculate_drawdown_factor(self) -> float:
        """计算回撤调整因子"""
        if self._current_drawdown <= 0:
            return 1.0

        if self._current_drawdown >= self.config.max_drawdown_pct:
            return self.config.drawdown_scale_factor

        # 线性插值
        factor = 1.0 - (self._current_drawdown / self.config.max_drawdown_pct) * (1.0 - self.config.drawdown_scale_factor)
        return max(factor, self.config.drawdown_scale_factor)

    # ==================== 组合级别计算 ====================

    def calculate_portfolio_sizing(
        self,
        symbols: List[str],
        method: SizingMethod = SizingMethod.RISK_PARITY
    ) -> PortfolioSizing:
        """计算组合级别配置"""

        # 计算各个资产的仓位
        sizing_results = self.calculate_multiple(symbols, method)

        # 汇总数据
        allocations = {}
        position_sizes = {}
        position_values = {}
        risk_contributions = {}

        total_weight = 0.0
        total_value = 0.0

        for symbol, result in sizing_results.items():
            allocations[symbol] = result.suggested_weight
            position_sizes[symbol] = result.suggested_size
            position_values[symbol] = result.suggested_value

            total_weight += result.suggested_weight
            total_value += result.suggested_value

            # 风险贡献
            metrics = self.get_asset_metrics(symbol)
            if metrics:
                risk_contributions[symbol] = result.suggested_weight * metrics.volatility

        # 归一化权重(如果总权重超过1)
        if total_weight > 1.0:
            for symbol in allocations:
                allocations[symbol] /= total_weight
                position_sizes[symbol] /= total_weight
                position_values[symbol] /= total_weight

        # 计算组合指标
        total_exposure = sum(position_values.values())
        net_exposure = total_exposure  # 简化：假设都是多头
        leverage = total_exposure / self._portfolio_value if self._portfolio_value > 0 else 1.0

        # 分散化评分(越接近等权重，分数越高)
        n = len(symbols)
        if n > 0:
            ideal_weight = 1.0 / n
            weight_variance = sum((w - ideal_weight) ** 2 for w in allocations.values()) / n
            diversification_score = 1.0 - min(weight_variance * n, 1.0)
        else:
            diversification_score = 0.0

        # 再平衡检查
        rebalance_required = False
        rebalance_suggestions = []

        for symbol, target_weight in allocations.items():
            current_position = self.get_position(symbol)
            metrics = self.get_asset_metrics(symbol)
            if metrics:
                current_value = current_position * metrics.current_price
                current_weight = current_value / self._portfolio_value if self._portfolio_value > 0 else 0

                weight_diff = abs(target_weight - current_weight)
                if weight_diff > self.config.rebalance_threshold:
                    rebalance_required = True
                    rebalance_suggestions.append({
                        'symbol': symbol,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'action': 'increase' if target_weight > current_weight else 'decrease',
                        'amount': weight_diff * self._portfolio_value,
                    })

        return PortfolioSizing(
            sizing_id=str(uuid.uuid4()),
            allocations=allocations,
            position_sizes=position_sizes,
            position_values=position_values,
            total_exposure=total_exposure,
            net_exposure=net_exposure,
            leverage=leverage,
            diversification_score=diversification_score,
            risk_contributions=risk_contributions,
            rebalance_required=rebalance_required,
            rebalance_suggestions=rebalance_suggestions,
        )

    # ==================== 实时调整 ====================

    def check_adjustment_triggers(self, symbol: str) -> List[AdjustmentSignal]:
        """检查是否需要调整仓位"""
        signals = []

        metrics = self.get_asset_metrics(symbol)
        if not metrics:
            return signals

        current_size = self.get_position(symbol)
        current_value = current_size * metrics.current_price

        # 检查波动率变化
        signal = self._check_volatility_trigger(symbol, metrics, current_size, current_value)
        if signal:
            signals.append(signal)

        # 检查回撤触发
        signal = self._check_drawdown_trigger(symbol, metrics, current_size, current_value)
        if signal:
            signals.append(signal)

        # 检查价格变化
        if metrics.entry_price:
            signal = self._check_price_trigger(symbol, metrics, current_size, current_value)
            if signal:
                signals.append(signal)

        # 记录信号
        self._adjustment_signals.extend(signals)
        self.stats['adjustments_triggered'] += len(signals)

        # 回调
        for signal in signals:
            if self._on_adjustment_signal:
                self._on_adjustment_signal(signal)

        return signals

    def _check_volatility_trigger(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_size: float,
        current_value: float
    ) -> Optional[AdjustmentSignal]:
        """检查波动率触发"""

        # 计算当前应有的仓位(基于波动率)
        sizer = VolatilityBasedSizer(self.config)
        target_result = sizer.calculate(symbol, metrics, current_size)

        # 检查偏差
        if current_value > 0:
            deviation = abs(target_result.suggested_value - current_value) / current_value

            if deviation > self.config.volatility_adjust_threshold:
                return AdjustmentSignal(
                    signal_id=str(uuid.uuid4()),
                    symbol=symbol,
                    trigger=AdjustmentTrigger.VOLATILITY_CHANGE,
                    current_size=current_size,
                    current_value=current_value,
                    suggested_size=target_result.suggested_size,
                    suggested_value=target_result.suggested_value,
                    adjustment_pct=deviation,
                    reason=f"Volatility adjustment required: {deviation:.1%} deviation",
                    trigger_value=metrics.volatility,
                    threshold_value=self.config.volatility_target,
                    urgency='normal' if deviation < 0.3 else 'high',
                )

        return None

    def _check_drawdown_trigger(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_size: float,
        current_value: float
    ) -> Optional[AdjustmentSignal]:
        """检查回撤触发"""

        if self._current_drawdown > self.config.max_drawdown_pct * 0.5:
            # 计算应有的仓位(应用回撤约束)
            drawdown_factor = self._calculate_drawdown_factor()
            target_value = current_value * drawdown_factor
            target_size = current_size * drawdown_factor

            return AdjustmentSignal(
                signal_id=str(uuid.uuid4()),
                symbol=symbol,
                trigger=AdjustmentTrigger.DRAWDOWN,
                current_size=current_size,
                current_value=current_value,
                suggested_size=target_size,
                suggested_value=target_value,
                adjustment_pct=-(1.0 - drawdown_factor),
                reason=f"Drawdown reduction: {self._current_drawdown:.1%} current drawdown",
                trigger_value=self._current_drawdown,
                threshold_value=self.config.max_drawdown_pct,
                urgency='high' if self._current_drawdown > self.config.max_drawdown_pct * 0.8 else 'normal',
            )

        return None

    def _check_price_trigger(
        self,
        symbol: str,
        metrics: AssetMetrics,
        current_size: float,
        current_value: float
    ) -> Optional[AdjustmentSignal]:
        """检查价格变化触发"""

        if not metrics.entry_price or metrics.entry_price <= 0:
            return None

        price_change = (metrics.current_price - metrics.entry_price) / metrics.entry_price

        # 大幅盈利时可以考虑加仓
        if price_change > 0.20:  # 盈利20%
            target_size = current_size * 1.2  # 加仓20%
            target_value = target_size * metrics.current_price

            return AdjustmentSignal(
                signal_id=str(uuid.uuid4()),
                symbol=symbol,
                trigger=AdjustmentTrigger.PROFIT_TARGET,
                current_size=current_size,
                current_value=current_value,
                suggested_size=target_size,
                suggested_value=target_value,
                adjustment_pct=0.2,
                reason=f"Profit target hit: {price_change:.1%} gain",
                trigger_value=price_change,
                threshold_value=0.20,
                urgency='low',
            )

        return None

    # ==================== 监控控制 ====================

    def on_sizing_calculated(self, callback: Callable[[SizingResult], None]):
        """注册仓位计算回调"""
        self._on_sizing_calculated = callback

    def on_adjustment_signal(self, callback: Callable[[AdjustmentSignal], None]):
        """注册调整信号回调"""
        self._on_adjustment_signal = callback

    async def start_monitoring(self, symbols: List[str]):
        """启动监控"""
        if self._state == EngineState.ACTIVE:
            return

        self._state = EngineState.ACTIVE
        self._monitor_task = asyncio.create_task(self._monitor_loop(symbols))
        logger.info("Position sizing monitoring started")

    async def stop_monitoring(self):
        """停止监控"""
        self._state = EngineState.IDLE
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Position sizing monitoring stopped")

    async def _monitor_loop(self, symbols: List[str]):
        """监控循环"""
        while self._state == EngineState.ACTIVE:
            try:
                # 检查所有持仓的调整触发
                for symbol in symbols:
                    if self.get_position(symbol) > 0:
                        self.check_adjustment_triggers(symbol)

                await asyncio.sleep(60)  # 每分钟检查一次

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self._state = EngineState.ERROR
                await asyncio.sleep(5)

    # ==================== 统计和分析 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'state': self._state.value,
            'calculations_count': self.stats['calculations_count'],
            'adjustments_triggered': self.stats['adjustments_triggered'],
            'constraints_violations': self.stats['constraints_violations'],
            'tracked_assets': len(self._asset_metrics),
            'active_positions': len([s for s in self._current_positions.values() if s > 0]),
            'portfolio_value': self._portfolio_value,
            'current_drawdown': self._current_drawdown,
            'config': self.config.to_dict(),
        }

    def get_sizing_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[SizingResult]:
        """获取历史计算记录"""
        history = self._sizing_history

        if symbol:
            history = [r for r in history if r.symbol == symbol]

        return history[-limit:]

    def get_adjustment_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> List[AdjustmentSignal]:
        """获取调整信号"""
        signals = self._adjustment_signals

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        return signals[-limit:]

    @property
    def state(self) -> EngineState:
        """获取引擎状态"""
        return self._state


# ==================== 工厂函数 ====================

def create_sizing_engine(
    account_size: float = 100000.0,
    risk_per_trade: float = 0.02,
    risk_level: str = 'moderate',
    **kwargs
) -> PositionSizingEngine:
    """
    创建仓位规模计算引擎

    Args:
        account_size: 账户规模
        risk_per_trade: 单笔风险比例
        risk_level: 风险等级
        **kwargs: 其他配置参数
    """
    try:
        level = RiskLevel(risk_level)
    except ValueError:
        level = RiskLevel.MODERATE

    config = SizingConfig(
        account_size=account_size,
        risk_per_trade=risk_per_trade,
        risk_level=level,
        **{k: v for k, v in kwargs.items() if hasattr(SizingConfig, k)}
    )

    return PositionSizingEngine(config)


def create_asset_metrics(
    symbol: str,
    current_price: float,
    **kwargs
) -> AssetMetrics:
    """创建资产指标"""
    return AssetMetrics(
        symbol=symbol,
        current_price=current_price,
        **{k: v for k, v in kwargs.items() if k in AssetMetrics.__dataclass_fields__}
    )


def create_position_constraint(
    constraint_type: str,
    value: float,
    is_hard_limit: bool = True,
    description: str = ""
) -> PositionConstraint:
    """创建仓位约束"""
    try:
        ct = ConstraintType(constraint_type)
    except ValueError:
        ct = ConstraintType.MAX_POSITION

    return PositionConstraint(
        constraint_type=ct,
        value=value,
        is_hard_limit=is_hard_limit,
        description=description,
    )


def quick_position_sizing(
    symbol: str,
    current_price: float,
    account_size: float,
    method: str = 'fixed_fractional',
    **kwargs
) -> Dict[str, Any]:
    """
    快速仓位计算

    Args:
        symbol: 交易品种
        current_price: 当前价格
        account_size: 账户规模
        method: 计算方法
        **kwargs: 资产指标参数
    """
    engine = create_sizing_engine(account_size=account_size)

    # 创建资产指标
    metrics = create_asset_metrics(
        symbol=symbol,
        current_price=current_price,
        **kwargs
    )
    engine.update_asset_metrics(metrics)

    # 计算仓位
    try:
        sizing_method = SizingMethod(method)
    except ValueError:
        sizing_method = SizingMethod.FIXED_FRACTIONAL

    result = engine.calculate_size(symbol, sizing_method)

    return {
        'symbol': symbol,
        'method': method,
        'sizing_result': result.to_dict(),
        'statistics': engine.get_statistics(),
    }


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'SizingMethod',
    'PositionType',
    'RiskLevel',
    'ConstraintType',
    'AdjustmentTrigger',
    'EngineState',
    # 数据类
    'AssetMetrics',
    'SizingResult',
    'PortfolioSizing',
    'PositionConstraint',
    'AdjustmentSignal',
    'SizingConfig',
    # 计算器
    'PositionSizer',
    'KellyCriterionSizer',
    'FixedFractionalSizer',
    'VolatilityBasedSizer',
    'RiskParitySizer',
    'EqualWeightSizer',
    'ATRBasedSizer',
    'DrawdownConstrainedSizer',
    # 核心引擎
    'PositionSizingEngine',
    # 工厂函数
    'create_sizing_engine',
    'create_asset_metrics',
    'create_position_constraint',
    'quick_position_sizing',
]
