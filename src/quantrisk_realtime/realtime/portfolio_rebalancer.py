"""
组合再平衡器 (Portfolio Rebalancer)

机构级组合再平衡系统，提供:
- Calendar rebalancing (定期再平衡)
- Threshold rebalancing (阈值触发再平衡)
- Tactical rebalancing (战术再平衡)
- Tax-aware rebalancing (税务感知再平衡)
- Transaction cost optimization (交易成本优化)
- Drift monitoring (漂移监控)
- Rebalance order generation (再平衡订单生成)
- What-if analysis (假设分析)

作者: TraderX 量化团队
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import uuid
import math
import logging
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class RebalanceStrategy(Enum):
    """再平衡策略"""
    CALENDAR = "calendar"               # 定期再平衡
    THRESHOLD = "threshold"             # 阈值触发
    TACTICAL = "tactical"               # 战术再平衡
    TAX_AWARE = "tax_aware"            # 税务感知
    HYBRID = "hybrid"                   # 混合策略
    OPPORTUNISTIC = "opportunistic"     # 机会型


class RebalanceFrequency(Enum):
    """再平衡频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    ON_DEMAND = "on_demand"


class DriftType(Enum):
    """漂移类型"""
    ABSOLUTE = "absolute"               # 绝对漂移
    RELATIVE = "relative"               # 相对漂移
    TRACKING_ERROR = "tracking_error"   # 跟踪误差
    RISK_BASED = "risk_based"           # 基于风险
    COMBINED = "combined"               # 综合


class TaxLotMethod(Enum):
    """税务批次方法"""
    FIFO = "fifo"                       # 先进先出
    LIFO = "lifo"                       # 后进先出
    HIFO = "hifo"                       # 最高成本先出
    SPEC_ID = "specific_identification" # 特定标识
    MIN_TAX = "minimize_tax"            # 最小化税务


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ADAPTIVE = "adaptive"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RebalanceState(Enum):
    """再平衡状态"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"


class AlertType(Enum):
    """告警类型"""
    DRIFT_THRESHOLD = "drift_threshold"
    CALENDAR_DUE = "calendar_due"
    TAX_OPPORTUNITY = "tax_opportunity"
    COST_ALERT = "cost_alert"
    EXECUTION_ISSUE = "execution_issue"


class TaxEventType(Enum):
    """税务事件类型"""
    SHORT_TERM_GAIN = "short_term_gain"
    LONG_TERM_GAIN = "long_term_gain"
    SHORT_TERM_LOSS = "short_term_loss"
    LONG_TERM_LOSS = "long_term_loss"
    WASH_SALE = "wash_sale"


# ==================== 数据类 ====================

@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: float
    market_value: float
    weight: float                       # 当前权重 (0-1)
    target_weight: float                # 目标权重 (0-1)

    # 成本信息
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # 税务信息
    holding_period_days: int = 0
    tax_lots: List['TaxLot'] = field(default_factory=list)

    # 时间戳
    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "market_value": self.market_value,
            "weight": self.weight,
            "target_weight": self.target_weight,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "holding_period_days": self.holding_period_days,
            "tax_lots_count": len(self.tax_lots),
            "as_of": self.as_of.isoformat(),
        }


@dataclass
class TaxLot:
    """税务批次"""
    lot_id: str
    symbol: str
    quantity: float
    cost_basis: float
    acquisition_date: datetime

    # 税务属性
    holding_period_days: int = 0
    is_long_term: bool = False
    unrealized_gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lot_id": self.lot_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "cost_basis": self.cost_basis,
            "acquisition_date": self.acquisition_date.isoformat(),
            "holding_period_days": self.holding_period_days,
            "is_long_term": self.is_long_term,
            "unrealized_gain": self.unrealized_gain,
        }


@dataclass
class PositionDrift:
    """持仓漂移"""
    symbol: str
    current_weight: float
    target_weight: float
    drift: float                        # 漂移 = current - target
    drift_pct: float                    # 漂移百分比
    drift_type: DriftType

    # 风险度量
    tracking_error_contribution: float = 0.0
    risk_contribution: float = 0.0

    # 再平衡建议
    recommended_trade_value: float = 0.0
    recommended_trade_shares: float = 0.0
    urgency_score: float = 0.0          # 0-1, 越高越紧急

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "drift": self.drift,
            "drift_pct": self.drift_pct,
            "drift_type": self.drift_type.value,
            "tracking_error_contribution": self.tracking_error_contribution,
            "risk_contribution": self.risk_contribution,
            "recommended_trade_value": self.recommended_trade_value,
            "recommended_trade_shares": self.recommended_trade_shares,
            "urgency_score": self.urgency_score,
        }


@dataclass
class RebalanceOrder:
    """再平衡订单"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET

    # 价格
    limit_price: Optional[float] = None
    estimated_price: float = 0.0
    executed_price: float = 0.0

    # 权重目标
    current_weight: float = 0.0
    target_weight: float = 0.0

    # 税务
    tax_lots_used: List[str] = field(default_factory=list)
    estimated_tax_impact: float = 0.0

    # 成本
    estimated_cost: float = 0.0
    actual_cost: float = 0.0

    # 状态
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "estimated_price": self.estimated_price,
            "executed_price": self.executed_price,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "tax_lots_used": self.tax_lots_used,
            "estimated_tax_impact": self.estimated_tax_impact,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }


@dataclass
class RebalancePlan:
    """再平衡计划"""
    plan_id: str
    portfolio_id: str
    strategy: RebalanceStrategy

    # 订单
    orders: List[RebalanceOrder] = field(default_factory=list)

    # 统计
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    total_turnover: float = 0.0
    turnover_rate: float = 0.0          # 相对于组合总值

    # 成本与税务
    estimated_commission: float = 0.0
    estimated_slippage: float = 0.0
    estimated_market_impact: float = 0.0
    estimated_total_cost: float = 0.0
    estimated_tax_impact: float = 0.0
    after_tax_cost: float = 0.0

    # 预期效果
    pre_drift_max: float = 0.0
    post_drift_max: float = 0.0
    drift_reduction: float = 0.0
    pre_tracking_error: float = 0.0
    post_tracking_error: float = 0.0

    # 风险度量
    pre_risk: float = 0.0
    post_risk: float = 0.0
    risk_reduction: float = 0.0

    # 状态
    state: RebalanceState = RebalanceState.IDLE

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "portfolio_id": self.portfolio_id,
            "strategy": self.strategy.value,
            "orders": [o.to_dict() for o in self.orders],
            "total_buy_value": self.total_buy_value,
            "total_sell_value": self.total_sell_value,
            "total_turnover": self.total_turnover,
            "turnover_rate": self.turnover_rate,
            "estimated_commission": self.estimated_commission,
            "estimated_slippage": self.estimated_slippage,
            "estimated_market_impact": self.estimated_market_impact,
            "estimated_total_cost": self.estimated_total_cost,
            "estimated_tax_impact": self.estimated_tax_impact,
            "after_tax_cost": self.after_tax_cost,
            "pre_drift_max": self.pre_drift_max,
            "post_drift_max": self.post_drift_max,
            "drift_reduction": self.drift_reduction,
            "pre_tracking_error": self.pre_tracking_error,
            "post_tracking_error": self.post_tracking_error,
            "pre_risk": self.pre_risk,
            "post_risk": self.post_risk,
            "risk_reduction": self.risk_reduction,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class WhatIfScenario:
    """假设分析场景"""
    scenario_id: str
    scenario_name: str

    # 场景参数
    weight_changes: Dict[str, float] = field(default_factory=dict)  # symbol -> new_weight
    price_changes: Dict[str, float] = field(default_factory=dict)   # symbol -> price_change_pct

    # 再平衡计划
    rebalance_plan: Optional[RebalancePlan] = None

    # 预测结果
    predicted_drift: float = 0.0
    predicted_tracking_error: float = 0.0
    predicted_risk: float = 0.0
    predicted_return: float = 0.0
    predicted_cost: float = 0.0
    predicted_tax: float = 0.0

    # 对比基准
    baseline_scenario: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "weight_changes": self.weight_changes,
            "price_changes": self.price_changes,
            "rebalance_plan": self.rebalance_plan.to_dict() if self.rebalance_plan else None,
            "predicted_drift": self.predicted_drift,
            "predicted_tracking_error": self.predicted_tracking_error,
            "predicted_risk": self.predicted_risk,
            "predicted_return": self.predicted_return,
            "predicted_cost": self.predicted_cost,
            "predicted_tax": self.predicted_tax,
            "baseline_scenario": self.baseline_scenario,
        }


@dataclass
class TaxHarvestingOpportunity:
    """税收收割机会"""
    opportunity_id: str
    symbol: str

    # 税务批次
    tax_lot: TaxLot

    # 损失/收益
    unrealized_loss: float              # 未实现损失（正数表示损失）
    tax_benefit: float                  # 税收优惠

    # 建议
    recommended_quantity: float
    recommended_action: str             # "harvest", "defer"

    # 限制
    wash_sale_risk: bool = False
    holding_period_remaining: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "symbol": self.symbol,
            "tax_lot": self.tax_lot.to_dict(),
            "unrealized_loss": self.unrealized_loss,
            "tax_benefit": self.tax_benefit,
            "recommended_quantity": self.recommended_quantity,
            "recommended_action": self.recommended_action,
            "wash_sale_risk": self.wash_sale_risk,
            "holding_period_remaining": self.holding_period_remaining,
        }


@dataclass
class RebalanceAlert:
    """再平衡告警"""
    alert_id: str
    alert_type: AlertType
    portfolio_id: str

    # 告警详情
    message: str
    severity: str = "warning"           # info/warning/critical
    details: Dict[str, Any] = field(default_factory=dict)

    # 建议
    recommendation: str = ""

    # 状态
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "portfolio_id": self.portfolio_id,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
            "recommendation": self.recommendation,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class RebalancerConfig:
    """再平衡器配置"""
    # 漂移阈值
    absolute_drift_threshold: float = 0.02      # 2% 绝对漂移
    relative_drift_threshold: float = 0.10      # 10% 相对漂移
    tracking_error_threshold: float = 0.02      # 2% 跟踪误差

    # 交易限制
    min_trade_value: float = 100.0              # 最小交易金额
    max_turnover_rate: float = 0.20             # 最大换手率 20%
    max_orders_per_rebalance: int = 50

    # 成本参数
    commission_rate: float = 0.001              # 0.1% 佣金
    slippage_bps: float = 5.0                   # 5 bps 滑点
    market_impact_factor: float = 0.001         # 市场冲击因子

    # 税务参数
    short_term_tax_rate: float = 0.37           # 短期资本利得税率
    long_term_tax_rate: float = 0.20            # 长期资本利得税率
    long_term_holding_days: int = 365           # 长期持有天数
    wash_sale_period_days: int = 30             # 洗售规则期限

    # 日历再平衡
    calendar_frequency: RebalanceFrequency = RebalanceFrequency.QUARTERLY
    next_calendar_rebalance: Optional[datetime] = None

    # 监控设置
    monitor_interval_seconds: float = 60.0
    enable_drift_monitoring: bool = True
    enable_tax_harvesting: bool = True

    # 优化设置
    optimize_for_tax: bool = True
    optimize_for_cost: bool = True
    use_limit_orders: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "absolute_drift_threshold": self.absolute_drift_threshold,
            "relative_drift_threshold": self.relative_drift_threshold,
            "tracking_error_threshold": self.tracking_error_threshold,
            "min_trade_value": self.min_trade_value,
            "max_turnover_rate": self.max_turnover_rate,
            "max_orders_per_rebalance": self.max_orders_per_rebalance,
            "commission_rate": self.commission_rate,
            "slippage_bps": self.slippage_bps,
            "market_impact_factor": self.market_impact_factor,
            "short_term_tax_rate": self.short_term_tax_rate,
            "long_term_tax_rate": self.long_term_tax_rate,
            "long_term_holding_days": self.long_term_holding_days,
            "calendar_frequency": self.calendar_frequency.value,
            "next_calendar_rebalance": self.next_calendar_rebalance.isoformat() if self.next_calendar_rebalance else None,
            "monitor_interval_seconds": self.monitor_interval_seconds,
            "enable_drift_monitoring": self.enable_drift_monitoring,
            "enable_tax_harvesting": self.enable_tax_harvesting,
            "optimize_for_tax": self.optimize_for_tax,
            "optimize_for_cost": self.optimize_for_cost,
            "use_limit_orders": self.use_limit_orders,
        }


# ==================== 接口定义 ====================

class PortfolioProvider(ABC):
    """组合数据提供者接口"""

    @abstractmethod
    def get_positions(self, portfolio_id: str) -> List[Position]:
        """获取持仓"""
        pass

    @abstractmethod
    def get_total_value(self, portfolio_id: str) -> float:
        """获取组合总值"""
        pass

    @abstractmethod
    def get_cash(self, portfolio_id: str) -> float:
        """获取现金"""
        pass


class PriceProvider(ABC):
    """价格数据提供者接口"""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """获取当前价格"""
        pass

    @abstractmethod
    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """批量获取价格"""
        pass


class TargetProvider(ABC):
    """目标权重提供者接口"""

    @abstractmethod
    def get_target_weights(self, portfolio_id: str) -> Dict[str, float]:
        """获取目标权重"""
        pass


class OrderExecutor(ABC):
    """订单执行器接口"""

    @abstractmethod
    def execute_order(self, order: RebalanceOrder) -> bool:
        """执行订单"""
        pass


# ==================== 模拟实现 ====================

class SimulatedPortfolioProvider(PortfolioProvider):
    """模拟组合提供者"""

    def __init__(self):
        self.portfolios: Dict[str, List[Position]] = {}
        self.total_values: Dict[str, float] = {}
        self.cash: Dict[str, float] = {}

    def set_positions(self, portfolio_id: str, positions: List[Position]):
        self.portfolios[portfolio_id] = positions

    def set_total_value(self, portfolio_id: str, value: float):
        self.total_values[portfolio_id] = value

    def set_cash(self, portfolio_id: str, cash: float):
        self.cash[portfolio_id] = cash

    def get_positions(self, portfolio_id: str) -> List[Position]:
        return self.portfolios.get(portfolio_id, [])

    def get_total_value(self, portfolio_id: str) -> float:
        return self.total_values.get(portfolio_id, 1000000.0)

    def get_cash(self, portfolio_id: str) -> float:
        return self.cash.get(portfolio_id, 0.0)


class SimulatedPriceProvider(PriceProvider):
    """模拟价格提供者"""

    def __init__(self):
        self.prices: Dict[str, float] = {}

    def set_price(self, symbol: str, price: float):
        self.prices[symbol] = price

    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 100.0)

    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        return {s: self.get_price(s) for s in symbols}


class SimulatedTargetProvider(TargetProvider):
    """模拟目标提供者"""

    def __init__(self):
        self.targets: Dict[str, Dict[str, float]] = {}

    def set_targets(self, portfolio_id: str, targets: Dict[str, float]):
        self.targets[portfolio_id] = targets

    def get_target_weights(self, portfolio_id: str) -> Dict[str, float]:
        return self.targets.get(portfolio_id, {})


class SimulatedOrderExecutor(OrderExecutor):
    """模拟订单执行器"""

    def __init__(self):
        self.executed_orders: List[RebalanceOrder] = []

    def execute_order(self, order: RebalanceOrder) -> bool:
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.executed_price = order.estimated_price
        order.executed_at = datetime.now()
        self.executed_orders.append(order)
        return True


# ==================== 核心引擎 ====================

class PortfolioRebalancer:
    """组合再平衡器"""

    def __init__(
        self,
        config: Optional[RebalancerConfig] = None,
        portfolio_provider: Optional[PortfolioProvider] = None,
        price_provider: Optional[PriceProvider] = None,
        target_provider: Optional[TargetProvider] = None,
        order_executor: Optional[OrderExecutor] = None,
    ):
        self.config = config or RebalancerConfig()
        self.portfolio_provider = portfolio_provider or simulated_default(SimulatedPortfolioProvider, "data from PortfolioRebalancer's simulated portfolio_provider (SimulatedPortfolioProvider)")
        self.price_provider = price_provider or simulated_default(SimulatedPriceProvider, "data from PortfolioRebalancer's simulated price_provider (SimulatedPriceProvider)")
        self.target_provider = target_provider or simulated_default(SimulatedTargetProvider, "data from PortfolioRebalancer's simulated target_provider (SimulatedTargetProvider)")
        self.order_executor = order_executor or simulated_default(SimulatedOrderExecutor, "data from PortfolioRebalancer's simulated order_executor (SimulatedOrderExecutor)")

        # 再平衡历史
        self.rebalance_plans: List[RebalancePlan] = []

        # 告警
        self.alerts: List[RebalanceAlert] = []

        # 假设分析场景
        self.scenarios: Dict[str, WhatIfScenario] = {}

        # 状态
        self.state = RebalanceState.IDLE
        self._monitor_task: Optional[asyncio.Task] = None

        # 回调
        self._on_drift_alert: Optional[Callable[[PositionDrift], None]] = None
        self._on_rebalance_complete: Optional[Callable[[RebalancePlan], None]] = None
        self._on_alert: Optional[Callable[[RebalanceAlert], None]] = None

        # 统计
        self.stats = {
            "total_rebalances": 0,
            "calendar_rebalances": 0,
            "threshold_rebalances": 0,
            "tactical_rebalances": 0,
            "total_orders": 0,
            "total_turnover": 0.0,
            "total_cost": 0.0,
            "total_tax": 0.0,
            "alerts_created": 0,
        }

    # ==================== Drift Monitoring (漂移监控) ====================

    def calculate_drift(
        self,
        portfolio_id: str,
        drift_type: DriftType = DriftType.ABSOLUTE
    ) -> List[PositionDrift]:
        """计算持仓漂移"""
        positions = self.portfolio_provider.get_positions(portfolio_id)
        total_value = self.portfolio_provider.get_total_value(portfolio_id)

        if total_value <= 0:
            return []

        drifts = []
        for position in positions:
            current_weight = position.weight
            target_weight = position.target_weight

            # 计算漂移
            if drift_type == DriftType.ABSOLUTE:
                drift = current_weight - target_weight
                drift_pct = (drift / target_weight * 100) if target_weight > 0 else 0
            elif drift_type == DriftType.RELATIVE:
                drift = (current_weight - target_weight) / target_weight if target_weight > 0 else 0
                drift_pct = drift * 100
            else:
                drift = current_weight - target_weight
                drift_pct = (drift / target_weight * 100) if target_weight > 0 else 0

            # 计算再平衡建议
            trade_value = -drift * total_value
            price = self.price_provider.get_price(position.symbol)
            trade_shares = trade_value / price if price > 0 else 0

            # 计算紧急程度
            urgency = min(abs(drift) / self.config.absolute_drift_threshold, 1.0)

            position_drift = PositionDrift(
                symbol=position.symbol,
                current_weight=current_weight,
                target_weight=target_weight,
                drift=drift,
                drift_pct=drift_pct,
                drift_type=drift_type,
                recommended_trade_value=trade_value,
                recommended_trade_shares=trade_shares,
                urgency_score=urgency,
            )
            drifts.append(position_drift)

        # 按漂移绝对值排序
        drifts.sort(key=lambda d: abs(d.drift), reverse=True)
        return drifts

    def check_drift_threshold(self, portfolio_id: str) -> bool:
        """检查是否超过漂移阈值"""
        drifts = self.calculate_drift(portfolio_id)

        if not drifts:
            return False

        max_drift = max(abs(d.drift) for d in drifts)

        if max_drift > self.config.absolute_drift_threshold:
            # 创建告警
            self._create_alert(
                AlertType.DRIFT_THRESHOLD,
                portfolio_id,
                f"Maximum drift ({max_drift:.2%}) exceeds threshold ({self.config.absolute_drift_threshold:.2%})",
                severity="warning",
                details={
                    "max_drift": max_drift,
                    "threshold": self.config.absolute_drift_threshold,
                    "positions_affected": len([d for d in drifts if abs(d.drift) > self.config.absolute_drift_threshold]),
                }
            )
            return True

        return False

    # ==================== Calendar Rebalancing (定期再平衡) ====================

    def schedule_calendar_rebalance(self):
        """调度下次日历再平衡"""
        now = datetime.now()
        freq = self.config.calendar_frequency

        if freq == RebalanceFrequency.DAILY:
            self.config.next_calendar_rebalance = now + timedelta(days=1)
        elif freq == RebalanceFrequency.WEEKLY:
            self.config.next_calendar_rebalance = now + timedelta(weeks=1)
        elif freq == RebalanceFrequency.MONTHLY:
            self.config.next_calendar_rebalance = now + timedelta(days=30)
        elif freq == RebalanceFrequency.QUARTERLY:
            self.config.next_calendar_rebalance = now + timedelta(days=90)
        elif freq == RebalanceFrequency.SEMI_ANNUAL:
            self.config.next_calendar_rebalance = now + timedelta(days=180)
        elif freq == RebalanceFrequency.ANNUAL:
            self.config.next_calendar_rebalance = now + timedelta(days=365)

    def is_calendar_rebalance_due(self) -> bool:
        """检查是否到期日历再平衡"""
        if not self.config.next_calendar_rebalance:
            return False
        return datetime.now() >= self.config.next_calendar_rebalance

    # ==================== Threshold Rebalancing (阈值触发再平衡) ====================

    def should_threshold_rebalance(self, portfolio_id: str) -> bool:
        """判断是否应触发阈值再平衡"""
        return self.check_drift_threshold(portfolio_id)

    # ==================== Tax-Aware Rebalancing (税务感知再平衡) ====================

    def find_tax_harvesting_opportunities(
        self,
        portfolio_id: str,
        min_loss_threshold: float = 100.0
    ) -> List[TaxHarvestingOpportunity]:
        """寻找税收收割机会"""
        positions = self.portfolio_provider.get_positions(portfolio_id)
        opportunities = []

        for position in positions:
            if position.unrealized_pnl >= 0:
                continue  # 只关注损失

            for tax_lot in position.tax_lots:
                if tax_lot.unrealized_gain >= 0:
                    continue  # 损失批次

                unrealized_loss = -tax_lot.unrealized_gain

                if unrealized_loss < min_loss_threshold:
                    continue

                # 计算税收优惠
                tax_rate = (
                    self.config.short_term_tax_rate
                    if not tax_lot.is_long_term
                    else self.config.long_term_tax_rate
                )
                tax_benefit = unrealized_loss * tax_rate

                # 检查洗售风险
                wash_sale_risk = tax_lot.holding_period_days < self.config.wash_sale_period_days

                opportunity = TaxHarvestingOpportunity(
                    opportunity_id=str(uuid.uuid4()),
                    symbol=position.symbol,
                    tax_lot=tax_lot,
                    unrealized_loss=unrealized_loss,
                    tax_benefit=tax_benefit,
                    recommended_quantity=tax_lot.quantity,
                    recommended_action="harvest" if not wash_sale_risk else "defer",
                    wash_sale_risk=wash_sale_risk,
                    holding_period_remaining=max(0, self.config.wash_sale_period_days - tax_lot.holding_period_days),
                )
                opportunities.append(opportunity)

        # 按税收优惠排序
        opportunities.sort(key=lambda o: o.tax_benefit, reverse=True)
        return opportunities

    def select_tax_lots(
        self,
        position: Position,
        quantity: float,
        method: TaxLotMethod = TaxLotMethod.MIN_TAX
    ) -> List[TaxLot]:
        """选择税务批次"""
        if not position.tax_lots:
            return []

        lots = position.tax_lots.copy()

        if method == TaxLotMethod.FIFO:
            lots.sort(key=lambda l: l.acquisition_date)
        elif method == TaxLotMethod.LIFO:
            lots.sort(key=lambda l: l.acquisition_date, reverse=True)
        elif method == TaxLotMethod.HIFO:
            lots.sort(key=lambda l: l.cost_basis, reverse=True)
        elif method == TaxLotMethod.MIN_TAX:
            # 优先选择损失批次，然后是长期持有的收益批次
            lots.sort(
                key=lambda l: (
                    -l.unrealized_gain if l.unrealized_gain < 0 else 0,  # 损失优先
                    l.is_long_term,                                        # 长期持有优先
                    -l.unrealized_gain                                     # 收益小的优先
                )
            )

        # 选择批次直到满足数量
        selected = []
        remaining = quantity

        for lot in lots:
            if remaining <= 0:
                break

            use_quantity = min(lot.quantity, remaining)

            # 创建部分批次
            selected_lot = TaxLot(
                lot_id=lot.lot_id,
                symbol=lot.symbol,
                quantity=use_quantity,
                cost_basis=lot.cost_basis * (use_quantity / lot.quantity),
                acquisition_date=lot.acquisition_date,
                holding_period_days=lot.holding_period_days,
                is_long_term=lot.is_long_term,
                unrealized_gain=lot.unrealized_gain * (use_quantity / lot.quantity),
            )
            selected.append(selected_lot)
            remaining -= use_quantity

        return selected

    def calculate_tax_impact(
        self,
        position: Position,
        sell_quantity: float,
        tax_lot_method: TaxLotMethod = TaxLotMethod.MIN_TAX
    ) -> float:
        """计算税务影响"""
        if sell_quantity <= 0:
            return 0.0

        selected_lots = self.select_tax_lots(position, sell_quantity, tax_lot_method)

        total_tax = 0.0
        for lot in selected_lots:
            if lot.unrealized_gain > 0:
                # 资本利得
                tax_rate = (
                    self.config.short_term_tax_rate
                    if not lot.is_long_term
                    else self.config.long_term_tax_rate
                )
                total_tax += lot.unrealized_gain * tax_rate
            else:
                # 资本损失可以抵税
                tax_rate = (
                    self.config.short_term_tax_rate
                    if not lot.is_long_term
                    else self.config.long_term_tax_rate
                )
                total_tax += lot.unrealized_gain * tax_rate  # 负数，减少税务

        return total_tax

    # ==================== Transaction Cost Optimization (交易成本优化) ====================

    def estimate_transaction_cost(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
        total_value: float
    ) -> float:
        """估算交易成本"""
        price = self.price_provider.get_price(symbol)
        trade_value = abs(quantity * price)

        # 佣金
        commission = trade_value * self.config.commission_rate

        # 滑点
        slippage = trade_value * (self.config.slippage_bps / 10000.0)

        # 市场冲击
        participation_rate = trade_value / total_value if total_value > 0 else 0
        market_impact = trade_value * self.config.market_impact_factor * math.sqrt(participation_rate)

        return commission + slippage + market_impact

    def optimize_order_size(
        self,
        orders: List[RebalanceOrder],
        total_value: float
    ) -> List[RebalanceOrder]:
        """优化订单大小以降低成本"""
        optimized = []

        for order in orders:
            trade_value = abs(order.quantity * order.estimated_price)

            # 检查最小交易金额
            if trade_value < self.config.min_trade_value:
                continue  # 跳过小额交易

            optimized.append(order)

        return optimized

    # ==================== Rebalance Order Generation (再平衡订单生成) ====================

    def generate_rebalance_plan(
        self,
        portfolio_id: str,
        strategy: RebalanceStrategy = RebalanceStrategy.THRESHOLD,
        max_turnover_override: Optional[float] = None,
    ) -> RebalancePlan:
        """生成再平衡计划"""
        self.state = RebalanceState.GENERATING

        # 获取数据
        positions = self.portfolio_provider.get_positions(portfolio_id)
        total_value = self.portfolio_provider.get_total_value(portfolio_id)

        # 计算漂移
        drifts = self.calculate_drift(portfolio_id)

        # 生成订单
        orders = []
        total_buy = 0.0
        total_sell = 0.0

        for drift in drifts:
            # 跳过小漂移
            if abs(drift.drift) < self.config.absolute_drift_threshold / 2:
                continue

            # 获取持仓
            position = next((p for p in positions if p.symbol == drift.symbol), None)
            if not position:
                continue

            trade_value = abs(drift.recommended_trade_value)
            if trade_value < self.config.min_trade_value:
                continue

            side = OrderSide.BUY if drift.drift < 0 else OrderSide.SELL
            quantity = abs(drift.recommended_trade_shares)
            price = self.price_provider.get_price(drift.symbol)

            # 估算成本
            cost = self.estimate_transaction_cost(
                drift.symbol, quantity, side, total_value
            )

            # 计算税务影响
            tax_impact = 0.0
            tax_lots_used = []
            if side == OrderSide.SELL and self.config.optimize_for_tax:
                tax_impact = self.calculate_tax_impact(position, quantity)
                selected_lots = self.select_tax_lots(position, quantity)
                tax_lots_used = [lot.lot_id for lot in selected_lots]

            order = RebalanceOrder(
                order_id=str(uuid.uuid4()),
                symbol=drift.symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.LIMIT if self.config.use_limit_orders else OrderType.MARKET,
                limit_price=price * 0.999 if side == OrderSide.BUY else price * 1.001,
                estimated_price=price,
                current_weight=drift.current_weight,
                target_weight=drift.target_weight,
                tax_lots_used=tax_lots_used,
                estimated_tax_impact=tax_impact,
                estimated_cost=cost,
            )
            orders.append(order)

            if side == OrderSide.BUY:
                total_buy += trade_value
            else:
                total_sell += trade_value

        # 优化订单
        if self.config.optimize_for_cost:
            orders = self.optimize_order_size(orders, total_value)

        # 检查换手率限制
        max_turnover = max_turnover_override or self.config.max_turnover_rate
        total_turnover = (total_buy + total_sell) / total_value if total_value > 0 else 0

        if total_turnover > max_turnover:
            # 按紧急程度排序，保留最重要的订单
            drifts_map = {d.symbol: d for d in drifts}
            orders.sort(key=lambda o: drifts_map.get(o.symbol, PositionDrift("", 0, 0, 0, 0, DriftType.ABSOLUTE)).urgency_score, reverse=True)

            # 限制订单数量
            scale_factor = max_turnover / total_turnover
            filtered_orders = []
            running_turnover = 0.0

            for order in orders:
                order_turnover = abs(order.quantity * order.estimated_price) / total_value
                if running_turnover + order_turnover <= max_turnover:
                    filtered_orders.append(order)
                    running_turnover += order_turnover

            orders = filtered_orders

            # 重新计算
            total_buy = sum(o.quantity * o.estimated_price for o in orders if o.side == OrderSide.BUY)
            total_sell = sum(o.quantity * o.estimated_price for o in orders if o.side == OrderSide.SELL)
            total_turnover = (total_buy + total_sell) / total_value if total_value > 0 else 0

        # 计算成本和税务
        estimated_commission = sum(
            abs(o.quantity * o.estimated_price) * self.config.commission_rate
            for o in orders
        )
        estimated_slippage = sum(
            abs(o.quantity * o.estimated_price) * (self.config.slippage_bps / 10000.0)
            for o in orders
        )
        estimated_market_impact = sum(o.estimated_cost for o in orders) - estimated_commission - estimated_slippage
        estimated_tax_impact = sum(o.estimated_tax_impact for o in orders)

        # 计算预期效果
        pre_drift_max = max(abs(d.drift) for d in drifts) if drifts else 0.0

        # 计算再平衡后的漂移
        remaining_drifts = [
            d for d in drifts
            if not any(o.symbol == d.symbol for o in orders)
        ]
        post_drift_max = max(abs(d.drift) for d in remaining_drifts) if remaining_drifts else 0.0

        plan = RebalancePlan(
            plan_id=str(uuid.uuid4()),
            portfolio_id=portfolio_id,
            strategy=strategy,
            orders=orders,
            total_buy_value=total_buy,
            total_sell_value=total_sell,
            total_turnover=total_buy + total_sell,
            turnover_rate=total_turnover,
            estimated_commission=estimated_commission,
            estimated_slippage=estimated_slippage,
            estimated_market_impact=estimated_market_impact,
            estimated_total_cost=estimated_commission + estimated_slippage + estimated_market_impact,
            estimated_tax_impact=estimated_tax_impact,
            after_tax_cost=estimated_commission + estimated_slippage + estimated_market_impact + estimated_tax_impact,
            pre_drift_max=pre_drift_max,
            post_drift_max=post_drift_max,
            drift_reduction=pre_drift_max - post_drift_max,
        )

        self.rebalance_plans.append(plan)
        self.state = RebalanceState.IDLE

        return plan

    def execute_rebalance_plan(self, plan: RebalancePlan) -> bool:
        """执行再平衡计划"""
        if plan.state not in [RebalanceState.IDLE, RebalanceState.GENERATING]:
            return False

        self.state = RebalanceState.EXECUTING
        plan.state = RebalanceState.EXECUTING
        plan.executed_at = datetime.now()

        success = True
        for order in plan.orders:
            if not self.order_executor.execute_order(order):
                success = False
                break

        if success:
            plan.state = RebalanceState.COMPLETED
            plan.completed_at = datetime.now()

            # 更新统计
            self.stats["total_rebalances"] += 1
            if plan.strategy == RebalanceStrategy.CALENDAR:
                self.stats["calendar_rebalances"] += 1
            elif plan.strategy == RebalanceStrategy.THRESHOLD:
                self.stats["threshold_rebalances"] += 1
            elif plan.strategy == RebalanceStrategy.TACTICAL:
                self.stats["tactical_rebalances"] += 1

            self.stats["total_orders"] += len(plan.orders)
            self.stats["total_turnover"] += plan.total_turnover
            self.stats["total_cost"] += plan.estimated_total_cost
            self.stats["total_tax"] += plan.estimated_tax_impact

            # 调度下次日历再平衡
            if plan.strategy == RebalanceStrategy.CALENDAR:
                self.schedule_calendar_rebalance()

            # 触发回调
            if self._on_rebalance_complete:
                self._on_rebalance_complete(plan)
        else:
            plan.state = RebalanceState.FAILED

        self.state = RebalanceState.IDLE
        return success

    # ==================== What-If Analysis (假设分析) ====================

    def create_what_if_scenario(
        self,
        scenario_name: str,
        portfolio_id: str,
        weight_changes: Optional[Dict[str, float]] = None,
        price_changes: Optional[Dict[str, float]] = None,
    ) -> WhatIfScenario:
        """创建假设分析场景"""
        scenario = WhatIfScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name=scenario_name,
            weight_changes=weight_changes or {},
            price_changes=price_changes or {},
        )

        self.scenarios[scenario.scenario_id] = scenario
        return scenario

    def analyze_what_if_scenario(
        self,
        scenario: WhatIfScenario,
        portfolio_id: str,
    ) -> WhatIfScenario:
        """分析假设场景"""
        # 备份当前配置
        original_positions = self.portfolio_provider.get_positions(portfolio_id)

        # 应用场景变化（这里简化，实际需要修改持仓）
        # ... 应用 weight_changes 和 price_changes

        # 生成再平衡计划
        plan = self.generate_rebalance_plan(portfolio_id, RebalanceStrategy.TACTICAL)
        scenario.rebalance_plan = plan

        # 预测结果
        scenario.predicted_drift = plan.pre_drift_max
        scenario.predicted_cost = plan.estimated_total_cost
        scenario.predicted_tax = plan.estimated_tax_impact

        # 恢复原始配置
        # ...

        return scenario

    def compare_scenarios(
        self,
        scenario_ids: List[str]
    ) -> Dict[str, Any]:
        """比较多个场景"""
        scenarios = [self.scenarios[sid] for sid in scenario_ids if sid in self.scenarios]

        if not scenarios:
            return {}

        comparison = {
            "scenarios": [s.to_dict() for s in scenarios],
            "best_for_cost": min(scenarios, key=lambda s: s.predicted_cost).scenario_id,
            "best_for_tax": min(scenarios, key=lambda s: s.predicted_tax).scenario_id,
            "best_for_drift": min(scenarios, key=lambda s: s.predicted_drift).scenario_id,
        }

        return comparison

    # ==================== Tactical Rebalancing (战术再平衡) ====================

    def tactical_rebalance(
        self,
        portfolio_id: str,
        tactical_adjustments: Dict[str, float],  # symbol -> weight_adjustment
    ) -> RebalancePlan:
        """战术再平衡"""
        # 获取当前目标
        current_targets = self.target_provider.get_target_weights(portfolio_id)

        # 应用战术调整
        adjusted_targets = current_targets.copy()
        for symbol, adjustment in tactical_adjustments.items():
            if symbol in adjusted_targets:
                adjusted_targets[symbol] += adjustment

        # 标准化权重
        total_weight = sum(adjusted_targets.values())
        if total_weight > 0:
            adjusted_targets = {k: v / total_weight for k, v in adjusted_targets.items()}

        # 暂时更新目标（实际应该通过 target_provider）
        # ...

        # 生成并执行再平衡计划
        plan = self.generate_rebalance_plan(portfolio_id, RebalanceStrategy.TACTICAL)

        return plan

    # ==================== 告警管理 ====================

    def _create_alert(
        self,
        alert_type: AlertType,
        portfolio_id: str,
        message: str,
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None,
        recommendation: str = ""
    ) -> RebalanceAlert:
        """创建告警"""
        alert = RebalanceAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            portfolio_id=portfolio_id,
            message=message,
            severity=severity,
            details=details or {},
            recommendation=recommendation,
        )
        self.alerts.append(alert)
        self.stats["alerts_created"] += 1

        if self._on_alert:
            self._on_alert(alert)

        return alert

    def get_alerts(
        self,
        portfolio_id: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100
    ) -> List[RebalanceAlert]:
        """获取告警"""
        alerts = self.alerts

        if portfolio_id:
            alerts = [a for a in alerts if a.portfolio_id == portfolio_id]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                return True
        return False

    # ==================== 监控控制 ====================

    def on_drift_alert(self, callback: Callable[[PositionDrift], None]):
        """注册漂移告警回调"""
        self._on_drift_alert = callback

    def on_rebalance_complete(self, callback: Callable[[RebalancePlan], None]):
        """注册再平衡完成回调"""
        self._on_rebalance_complete = callback

    def on_alert(self, callback: Callable[[RebalanceAlert], None]):
        """注册告警回调"""
        self._on_alert = callback

    async def start_monitoring(self, portfolio_ids: List[str]):
        """启动监控"""
        if self.state == RebalanceState.MONITORING:
            return

        self.state = RebalanceState.MONITORING
        self._monitor_task = asyncio.create_task(self._monitor_loop(portfolio_ids))

    async def stop_monitoring(self):
        """停止监控"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        self.state = RebalanceState.IDLE

    async def _monitor_loop(self, portfolio_ids: List[str]):
        """监控循环"""
        while self.state == RebalanceState.MONITORING:
            try:
                for portfolio_id in portfolio_ids:
                    # 检查漂移
                    if self.config.enable_drift_monitoring:
                        if self.should_threshold_rebalance(portfolio_id):
                            # 自动生成再平衡计划（不自动执行）
                            self.generate_rebalance_plan(portfolio_id, RebalanceStrategy.THRESHOLD)

                    # 检查日历再平衡
                    if self.is_calendar_rebalance_due():
                        self._create_alert(
                            AlertType.CALENDAR_DUE,
                            portfolio_id,
                            f"Calendar rebalance is due ({self.config.calendar_frequency.value})",
                            severity="info",
                            recommendation="Execute calendar rebalance"
                        )

                    # 检查税收收割机会
                    if self.config.enable_tax_harvesting:
                        opportunities = self.find_tax_harvesting_opportunities(portfolio_id)
                        if opportunities:
                            best_opp = opportunities[0]
                            self._create_alert(
                                AlertType.TAX_OPPORTUNITY,
                                portfolio_id,
                                f"Tax harvesting opportunity: ${best_opp.tax_benefit:.2f} benefit",
                                severity="info",
                                details={"opportunities_count": len(opportunities)},
                                recommendation="Review tax harvesting opportunities"
                            )

                await asyncio.sleep(self.config.monitor_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Rebalancer monitoring loop iteration failed; continuing")
                # Back off before retrying so a failure raised before the sleep
                # above cannot spin this loop hot.
                await asyncio.sleep(self.config.monitor_interval_seconds)

    # ==================== 查询 ====================

    def get_rebalance_plan(self, plan_id: str) -> Optional[RebalancePlan]:
        """获取再平衡计划"""
        for plan in self.rebalance_plans:
            if plan.plan_id == plan_id:
                return plan
        return None

    def get_rebalance_history(
        self,
        portfolio_id: Optional[str] = None,
        limit: int = 100
    ) -> List[RebalancePlan]:
        """获取再平衡历史"""
        plans = self.rebalance_plans

        if portfolio_id:
            plans = [p for p in plans if p.portfolio_id == portfolio_id]

        return plans[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self.state.value,
            "total_plans": len(self.rebalance_plans),
            "active_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "scenarios_count": len(self.scenarios),
            **self.stats,
        }


# ==================== 工厂函数 ====================

def create_portfolio_rebalancer(
    config: Optional[RebalancerConfig] = None,
    portfolio_provider: Optional[PortfolioProvider] = None,
    price_provider: Optional[PriceProvider] = None,
    target_provider: Optional[TargetProvider] = None,
    order_executor: Optional[OrderExecutor] = None,
) -> PortfolioRebalancer:
    """创建组合再平衡器"""
    return PortfolioRebalancer(
        config=config,
        portfolio_provider=portfolio_provider,
        price_provider=price_provider,
        target_provider=target_provider,
        order_executor=order_executor,
    )


def create_position(
    symbol: str,
    quantity: float,
    market_value: float,
    weight: float,
    target_weight: float,
    **kwargs
) -> Position:
    """创建持仓"""
    return Position(
        symbol=symbol,
        quantity=quantity,
        market_value=market_value,
        weight=weight,
        target_weight=target_weight,
        **kwargs
    )


def create_tax_lot(
    symbol: str,
    quantity: float,
    cost_basis: float,
    acquisition_date: datetime,
    **kwargs
) -> TaxLot:
    """创建税务批次"""
    lot_id = kwargs.pop("lot_id", str(uuid.uuid4()))
    return TaxLot(
        lot_id=lot_id,
        symbol=symbol,
        quantity=quantity,
        cost_basis=cost_basis,
        acquisition_date=acquisition_date,
        **kwargs
    )


def quick_rebalance_analysis(
    positions: List[Dict[str, Any]],
    target_weights: Dict[str, float],
    total_value: float,
) -> Dict[str, Any]:
    """快速再平衡分析"""
    rebalancer = create_portfolio_rebalancer()

    # 设置数据
    portfolio_id = "quick_portfolio"

    position_objects = [
        create_position(
            symbol=p["symbol"],
            quantity=p.get("quantity", 0),
            market_value=p.get("market_value", 0),
            weight=p.get("weight", 0),
            target_weight=target_weights.get(p["symbol"], 0),
        )
        for p in positions
    ]

    rebalancer.portfolio_provider.set_positions(portfolio_id, position_objects)
    rebalancer.portfolio_provider.set_total_value(portfolio_id, total_value)
    rebalancer.target_provider.set_targets(portfolio_id, target_weights)

    # 分析
    drifts = rebalancer.calculate_drift(portfolio_id)
    plan = rebalancer.generate_rebalance_plan(portfolio_id)

    return {
        "drifts": [d.to_dict() for d in drifts],
        "plan": plan.to_dict(),
        "statistics": rebalancer.get_statistics(),
    }


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "RebalanceStrategy",
    "RebalanceFrequency",
    "DriftType",
    "TaxLotMethod",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "RebalanceState",
    "AlertType",
    "TaxEventType",
    # 数据类
    "Position",
    "TaxLot",
    "PositionDrift",
    "RebalanceOrder",
    "RebalancePlan",
    "WhatIfScenario",
    "TaxHarvestingOpportunity",
    "RebalanceAlert",
    "RebalancerConfig",
    # 接口
    "PortfolioProvider",
    "PriceProvider",
    "TargetProvider",
    "OrderExecutor",
    # 模拟实现
    "SimulatedPortfolioProvider",
    "SimulatedPriceProvider",
    "SimulatedTargetProvider",
    "SimulatedOrderExecutor",
    # 核心引擎
    "PortfolioRebalancer",
    # 工厂函数
    "create_portfolio_rebalancer",
    "create_position",
    "create_tax_lot",
    "quick_rebalance_analysis",
]
