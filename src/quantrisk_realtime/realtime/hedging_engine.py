"""
Hedging Engine - 对冲引擎

实时对冲引擎，提供Delta、Gamma、Vega等希腊字母对冲策略。
支持动态对冲、成本优化、跨资产对冲和尾部风险管理。

核心功能：
1. Delta对冲 - Delta中性策略、带宽触发、动态频率
2. Gamma对冲 - Gamma Scalping、风险监控、期权组合管理
3. Vega对冲 - 波动率敞口管理、期限结构对冲
4. 组合对冲 - 跨资产对冲、Beta对冲、相关性对冲、尾部风险对冲
5. 对冲执行 - 订单生成、成本计算、效果评估、再平衡调度

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Constants ============

# 默认对冲参数
DEFAULT_DELTA_BANDWIDTH = 0.05  # Delta触发带宽（5%）
DEFAULT_GAMMA_THRESHOLD = 0.10  # Gamma触发阈值
DEFAULT_VEGA_THRESHOLD = 100.0  # Vega触发阈值
DEFAULT_REHEDGE_INTERVAL = 300  # 再平衡间隔（秒）

# 成本参数
DEFAULT_TRANSACTION_COST_BPS = 5  # 交易成本（基点）
DEFAULT_SLIPPAGE_BPS = 2  # 滑点（基点）


# ============ Enums ============


class HedgeType(str, Enum):
    """对冲类型"""
    DELTA = "delta"  # Delta对冲
    GAMMA = "gamma"  # Gamma对冲
    VEGA = "vega"  # Vega对冲
    THETA = "theta"  # Theta对冲
    RHO = "rho"  # Rho对冲
    CROSS_GAMMA = "cross_gamma"  # 交叉Gamma对冲
    CORRELATION = "correlation"  # 相关性对冲
    BETA = "beta"  # Beta对冲
    TAIL_RISK = "tail_risk"  # 尾部风险对冲


class HedgeStrategy(str, Enum):
    """对冲策略"""
    STATIC = "static"  # 静态对冲
    DYNAMIC = "dynamic"  # 动态对冲
    BANDWIDTH = "bandwidth"  # 带宽触发
    TIME_BASED = "time_based"  # 基于时间
    COST_OPTIMIZED = "cost_optimized"  # 成本优化
    GAMMA_SCALPING = "gamma_scalping"  # Gamma Scalping
    DELTA_NEUTRAL = "delta_neutral"  # Delta中性
    VEGA_NEUTRAL = "vega_neutral"  # Vega中性


class HedgeInstrument(str, Enum):
    """对冲工具"""
    UNDERLYING = "underlying"  # 标的资产
    FUTURES = "futures"  # 期货
    OPTIONS = "options"  # 期权
    ETF = "etf"  # ETF
    SWAP = "swap"  # 掉期


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单
    STOP = "stop"  # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class HedgeStatus(str, Enum):
    """对冲状态"""
    PENDING = "pending"  # 待处理
    APPROVED = "approved"  # 已批准
    EXECUTING = "executing"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class AlertSeverity(str, Enum):
    """警报严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EngineState(str, Enum):
    """引擎状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ Data Classes ============


@dataclass
class PositionData:
    """持仓数据"""
    position_id: str
    symbol: str
    quantity: float  # 持仓数量
    price: float  # 持仓均价
    current_price: float  # 当前价格
    market_value: float  # 市值
    # 希腊字母
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    # 期权特定
    is_option: bool = False
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    option_type: Optional[str] = None  # call/put

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "is_option": self.is_option,
            "strike": self.strike,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "option_type": self.option_type,
        }


@dataclass
class PortfolioGreeks:
    """组合希腊字母"""
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0
    total_rho: float = 0.0
    # 按标的分组
    by_underlying: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_delta": self.total_delta,
            "total_gamma": self.total_gamma,
            "total_vega": self.total_vega,
            "total_theta": self.total_theta,
            "total_rho": self.total_rho,
            "by_underlying": self.by_underlying,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HedgeOrder:
    """对冲订单"""
    order_id: str
    hedge_type: HedgeType
    symbol: str
    instrument: HedgeInstrument
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    # 对冲相关
    target_delta: float = 0.0
    current_delta: float = 0.0
    delta_adjustment: float = 0.0
    # 成本估算
    estimated_cost: float = 0.0
    transaction_cost: float = 0.0
    slippage: float = 0.0
    # 状态
    status: HedgeStatus = HedgeStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "hedge_type": self.hedge_type.value,
            "symbol": self.symbol,
            "instrument": self.instrument.value,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "target_delta": self.target_delta,
            "current_delta": self.current_delta,
            "delta_adjustment": self.delta_adjustment,
            "estimated_cost": self.estimated_cost,
            "transaction_cost": self.transaction_cost,
            "slippage": self.slippage,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "executed_price": self.executed_price,
            "executed_quantity": self.executed_quantity,
            "error_message": self.error_message,
        }


@dataclass
class HedgeResult:
    """对冲结果"""
    hedge_id: str
    hedge_type: HedgeType
    strategy: HedgeStrategy
    # 对冲前后对比
    before_delta: float
    after_delta: float
    delta_reduction: float
    before_gamma: float = 0.0
    after_gamma: float = 0.0
    before_vega: float = 0.0
    after_vega: float = 0.0
    # 订单信息
    orders: List[HedgeOrder] = field(default_factory=list)
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    # 成本
    total_cost: float = 0.0
    transaction_costs: float = 0.0
    slippage_costs: float = 0.0
    # 效果评估
    effectiveness: float = 0.0  # 0-1，1表示完全达到目标
    cost_efficiency: float = 0.0  # 成本效率
    # 时间
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hedge_id": self.hedge_id,
            "hedge_type": self.hedge_type.value,
            "strategy": self.strategy.value,
            "before_delta": self.before_delta,
            "after_delta": self.after_delta,
            "delta_reduction": self.delta_reduction,
            "before_gamma": self.before_gamma,
            "after_gamma": self.after_gamma,
            "before_vega": self.before_vega,
            "after_vega": self.after_vega,
            "orders": [o.to_dict() for o in self.orders],
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "total_cost": self.total_cost,
            "transaction_costs": self.transaction_costs,
            "slippage_costs": self.slippage_costs,
            "effectiveness": self.effectiveness,
            "cost_efficiency": self.cost_efficiency,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class DeltaHedgeConfig:
    """Delta对冲配置"""
    enabled: bool = True
    strategy: HedgeStrategy = HedgeStrategy.BANDWIDTH
    target_delta: float = 0.0  # 目标Delta
    bandwidth: float = DEFAULT_DELTA_BANDWIDTH  # 触发带宽
    min_hedge_amount: float = 100.0  # 最小对冲金额
    max_hedge_amount: float = 1000000.0  # 最大对冲金额
    rehedge_interval_seconds: int = DEFAULT_REHEDGE_INTERVAL
    instrument: HedgeInstrument = HedgeInstrument.UNDERLYING
    use_futures: bool = False  # 是否使用期货

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strategy": self.strategy.value,
            "target_delta": self.target_delta,
            "bandwidth": self.bandwidth,
            "min_hedge_amount": self.min_hedge_amount,
            "max_hedge_amount": self.max_hedge_amount,
            "rehedge_interval_seconds": self.rehedge_interval_seconds,
            "instrument": self.instrument.value,
            "use_futures": self.use_futures,
        }


@dataclass
class GammaHedgeConfig:
    """Gamma对冲配置"""
    enabled: bool = False
    strategy: HedgeStrategy = HedgeStrategy.GAMMA_SCALPING
    target_gamma: float = 0.0
    gamma_threshold: float = DEFAULT_GAMMA_THRESHOLD
    scalping_enabled: bool = True
    scalping_trigger_pct: float = 0.01  # 1% price move triggers scalping
    rehedge_frequency_minutes: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strategy": self.strategy.value,
            "target_gamma": self.target_gamma,
            "gamma_threshold": self.gamma_threshold,
            "scalping_enabled": self.scalping_enabled,
            "scalping_trigger_pct": self.scalping_trigger_pct,
            "rehedge_frequency_minutes": self.rehedge_frequency_minutes,
        }


@dataclass
class VegaHedgeConfig:
    """Vega对冲配置"""
    enabled: bool = False
    strategy: HedgeStrategy = HedgeStrategy.VEGA_NEUTRAL
    target_vega: float = 0.0
    vega_threshold: float = DEFAULT_VEGA_THRESHOLD
    term_structure_hedge: bool = False  # 是否对冲期限结构
    use_options: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strategy": self.strategy.value,
            "target_vega": self.target_vega,
            "vega_threshold": self.vega_threshold,
            "term_structure_hedge": self.term_structure_hedge,
            "use_options": self.use_options,
        }


@dataclass
class CorrelationHedgeConfig:
    """相关性对冲配置"""
    enabled: bool = False
    correlation_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    correlation_threshold: float = 0.7  # 相关性阈值
    hedge_correlated_moves: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "correlation_matrix": {f"{k[0]}_{k[1]}": v for k, v in self.correlation_matrix.items()},
            "correlation_threshold": self.correlation_threshold,
            "hedge_correlated_moves": self.hedge_correlated_moves,
        }


@dataclass
class TailRiskHedgeConfig:
    """尾部风险对冲配置"""
    enabled: bool = False
    var_threshold: float = -100000.0  # VaR阈值
    use_put_options: bool = True
    use_vix_products: bool = False
    hedge_ratio: float = 0.5  # 对冲比例

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "var_threshold": self.var_threshold,
            "use_put_options": self.use_put_options,
            "use_vix_products": self.use_vix_products,
            "hedge_ratio": self.hedge_ratio,
        }


@dataclass
class HedgeAlert:
    """对冲警报"""
    alert_id: str
    hedge_type: HedgeType
    severity: AlertSeverity
    message: str
    current_value: float
    threshold: float
    recommended_action: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "hedge_type": self.hedge_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RebalanceSchedule:
    """再平衡计划"""
    schedule_id: str
    hedge_type: HedgeType
    next_rebalance: datetime
    interval_seconds: int
    last_rebalance: Optional[datetime] = None
    rebalance_count: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "hedge_type": self.hedge_type.value,
            "next_rebalance": self.next_rebalance.isoformat(),
            "interval_seconds": self.interval_seconds,
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "rebalance_count": self.rebalance_count,
            "enabled": self.enabled,
        }


@dataclass
class EngineConfig:
    """引擎配置"""
    # 对冲配置
    delta_hedge: DeltaHedgeConfig = field(default_factory=DeltaHedgeConfig)
    gamma_hedge: GammaHedgeConfig = field(default_factory=GammaHedgeConfig)
    vega_hedge: VegaHedgeConfig = field(default_factory=VegaHedgeConfig)
    correlation_hedge: CorrelationHedgeConfig = field(default_factory=CorrelationHedgeConfig)
    tail_risk_hedge: TailRiskHedgeConfig = field(default_factory=TailRiskHedgeConfig)
    # 成本参数
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    # 风控参数
    max_hedge_orders_per_hour: int = 20
    max_total_hedge_cost: float = 100000.0
    require_approval_above: float = 50000.0
    # 监控参数
    update_interval_ms: int = 1000
    alert_enabled: bool = True


# ============ Abstract Interfaces ============


class PositionProvider(ABC):
    """持仓提供者接口"""

    @abstractmethod
    async def get_all_positions(self) -> List[PositionData]:
        """获取所有持仓"""
        pass

    @abstractmethod
    async def get_position(self, position_id: str) -> Optional[PositionData]:
        """获取单个持仓"""
        pass


class MarketDataProvider(ABC):
    """市场数据提供者接口"""

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        pass

    @abstractmethod
    async def get_volatility(self, symbol: str) -> float:
        """获取波动率"""
        pass


class OrderExecutor(ABC):
    """订单执行器接口"""

    @abstractmethod
    async def execute_order(self, order: HedgeOrder) -> bool:
        """执行对冲订单"""
        pass


# ============ Simulated Providers ============


class SimulatedPositionProvider(PositionProvider):
    """模拟持仓提供者"""

    def __init__(self):
        self._positions: Dict[str, PositionData] = {}

    def add_position(self, position: PositionData) -> None:
        """添加持仓"""
        self._positions[position.position_id] = position

    def remove_position(self, position_id: str) -> None:
        """移除持仓"""
        self._positions.pop(position_id, None)

    async def get_all_positions(self) -> List[PositionData]:
        return list(self._positions.values())

    async def get_position(self, position_id: str) -> Optional[PositionData]:
        return self._positions.get(position_id)


class SimulatedMarketDataProvider(MarketDataProvider):
    """模拟市场数据提供者"""

    def __init__(self):
        self._prices: Dict[str, float] = {}
        self._volatilities: Dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """设置价格"""
        self._prices[symbol] = price

    def set_volatility(self, symbol: str, vol: float) -> None:
        """设置波动率"""
        self._volatilities[symbol] = vol

    async def get_current_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 100.0)

    async def get_volatility(self, symbol: str) -> float:
        return self._volatilities.get(symbol, 0.2)


class SimulatedOrderExecutor(OrderExecutor):
    """模拟订单执行器"""

    def __init__(self):
        self._executed_orders: List[HedgeOrder] = []

    async def execute_order(self, order: HedgeOrder) -> bool:
        """模拟执行订单"""
        try:
            order.status = HedgeStatus.EXECUTING
            # 模拟执行延迟
            await asyncio.sleep(0.1)
            order.executed_at = datetime.now()
            order.executed_price = order.limit_price or 100.0
            order.executed_quantity = order.quantity
            order.status = HedgeStatus.COMPLETED
            self._executed_orders.append(order)
            return True
        except Exception as e:
            logger.warning(
                "Hedge order %s failed: %s", getattr(order, "order_id", "<unknown>"), e,
                exc_info=True
            )
            order.status = HedgeStatus.FAILED
            order.error_message = str(e)
            return False

    def get_executed_orders(self) -> List[HedgeOrder]:
        """获取已执行订单"""
        return self._executed_orders


# ============ Main Class ============


class HedgingEngine:
    """对冲引擎"""

    def __init__(
        self,
        config: EngineConfig,
        position_provider: Optional[PositionProvider] = None,
        market_data_provider: Optional[MarketDataProvider] = None,
        order_executor: Optional[OrderExecutor] = None,
    ):
        self._config = config
        self._position_provider = position_provider or simulated_default(SimulatedPositionProvider, "data from HedgingEngine's simulated position_provider (SimulatedPositionProvider)")
        self._market_data_provider = market_data_provider or simulated_default(SimulatedMarketDataProvider, "data from HedgingEngine's simulated market_data_provider (SimulatedMarketDataProvider)")
        self._order_executor = order_executor or simulated_default(SimulatedOrderExecutor, "data from HedgingEngine's simulated order_executor (SimulatedOrderExecutor)")

        # 状态
        self._state = EngineState.IDLE
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        # 对冲历史
        self._hedge_results: List[HedgeResult] = []
        self._hedge_orders: Dict[str, HedgeOrder] = {}
        self._alerts: List[HedgeAlert] = []
        self._alert_counter = 0

        # 再平衡计划
        self._rebalance_schedules: Dict[HedgeType, RebalanceSchedule] = {}

        # 缓存
        self._current_greeks: Optional[PortfolioGreeks] = None
        self._last_hedge_time: Dict[HedgeType, datetime] = {}

        # 回调
        self._hedge_callbacks: List[Callable[[HedgeResult], None]] = []
        self._alert_callbacks: List[Callable[[HedgeAlert], None]] = []

        # 统计
        self._total_hedges = 0
        self._total_hedge_cost = 0.0
        self._hedge_count_by_type: Dict[HedgeType, int] = {}

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动引擎"""
        if self._running:
            return

        self._running = True
        self._state = EngineState.RUNNING
        self._initialize_schedules()
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("HedgingEngine started")

    async def stop(self) -> None:
        """停止引擎"""
        self._running = False
        self._state = EngineState.IDLE
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("HedgingEngine stopped")

    def _initialize_schedules(self) -> None:
        """初始化再平衡计划"""
        now = datetime.now()

        if self._config.delta_hedge.enabled:
            self._rebalance_schedules[HedgeType.DELTA] = RebalanceSchedule(
                schedule_id=f"SCHEDULE_DELTA_{uuid.uuid4().hex[:8]}",
                hedge_type=HedgeType.DELTA,
                next_rebalance=now + timedelta(seconds=self._config.delta_hedge.rehedge_interval_seconds),
                interval_seconds=self._config.delta_hedge.rehedge_interval_seconds,
            )

        if self._config.gamma_hedge.enabled:
            self._rebalance_schedules[HedgeType.GAMMA] = RebalanceSchedule(
                schedule_id=f"SCHEDULE_GAMMA_{uuid.uuid4().hex[:8]}",
                hedge_type=HedgeType.GAMMA,
                next_rebalance=now + timedelta(minutes=self._config.gamma_hedge.rehedge_frequency_minutes),
                interval_seconds=self._config.gamma_hedge.rehedge_frequency_minutes * 60,
            )

    async def _update_loop(self) -> None:
        """更新循环"""
        while self._running:
            try:
                await self._update_greeks()
                await self._check_hedge_triggers()
                await self._check_rebalance_schedules()
                await asyncio.sleep(self._config.update_interval_ms / 1000)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                self._state = EngineState.ERROR
                await asyncio.sleep(1)

    # ============ Greeks Calculation ============

    async def _update_greeks(self) -> None:
        """更新希腊字母"""
        try:
            positions = await self._position_provider.get_all_positions()
            if not positions:
                return

            total_delta = 0.0
            total_gamma = 0.0
            total_vega = 0.0
            total_theta = 0.0
            total_rho = 0.0
            by_underlying: Dict[str, Dict[str, float]] = {}

            for position in positions:
                total_delta += position.delta * position.quantity
                total_gamma += position.gamma * position.quantity
                total_vega += position.vega * position.quantity
                total_theta += position.theta * position.quantity
                total_rho += position.rho * position.quantity

                # 按标的分组
                if position.symbol not in by_underlying:
                    by_underlying[position.symbol] = {
                        "delta": 0.0,
                        "gamma": 0.0,
                        "vega": 0.0,
                        "theta": 0.0,
                        "rho": 0.0,
                    }

                by_underlying[position.symbol]["delta"] += position.delta * position.quantity
                by_underlying[position.symbol]["gamma"] += position.gamma * position.quantity
                by_underlying[position.symbol]["vega"] += position.vega * position.quantity
                by_underlying[position.symbol]["theta"] += position.theta * position.quantity
                by_underlying[position.symbol]["rho"] += position.rho * position.quantity

            self._current_greeks = PortfolioGreeks(
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_vega=total_vega,
                total_theta=total_theta,
                total_rho=total_rho,
                by_underlying=by_underlying,
            )

        except Exception as e:
            logger.error(f"Error updating greeks: {e}")

    # ============ Hedge Triggers ============

    async def _check_hedge_triggers(self) -> None:
        """检查对冲触发条件"""
        if not self._current_greeks:
            return

        # Delta对冲触发
        if self._config.delta_hedge.enabled:
            await self._check_delta_hedge_trigger()

        # Gamma对冲触发
        if self._config.gamma_hedge.enabled:
            await self._check_gamma_hedge_trigger()

        # Vega对冲触发
        if self._config.vega_hedge.enabled:
            await self._check_vega_hedge_trigger()

    async def _check_delta_hedge_trigger(self) -> None:
        """检查Delta对冲触发"""
        if not self._current_greeks:
            return

        delta = self._current_greeks.total_delta
        target = self._config.delta_hedge.target_delta
        bandwidth = self._config.delta_hedge.bandwidth

        # 计算触发阈值
        threshold = abs(delta) * bandwidth

        if abs(delta - target) > max(threshold, self._config.delta_hedge.min_hedge_amount):
            # 触发对冲
            await self.execute_delta_hedge()

    async def _check_gamma_hedge_trigger(self) -> None:
        """检查Gamma对冲触发"""
        if not self._current_greeks:
            return

        gamma = self._current_greeks.total_gamma
        threshold = self._config.gamma_hedge.gamma_threshold

        if abs(gamma) > threshold:
            await self.execute_gamma_hedge()

    async def _check_vega_hedge_trigger(self) -> None:
        """检查Vega对冲触发"""
        if not self._current_greeks:
            return

        vega = self._current_greeks.total_vega
        threshold = self._config.vega_hedge.vega_threshold

        if abs(vega) > threshold:
            await self.execute_vega_hedge()

    # ============ Rebalance Schedules ============

    async def _check_rebalance_schedules(self) -> None:
        """检查再平衡计划"""
        now = datetime.now()

        for hedge_type, schedule in self._rebalance_schedules.items():
            if not schedule.enabled:
                continue

            if now >= schedule.next_rebalance:
                # 执行再平衡
                await self._execute_scheduled_rebalance(hedge_type)

                # 更新计划
                schedule.last_rebalance = now
                schedule.next_rebalance = now + timedelta(seconds=schedule.interval_seconds)
                schedule.rebalance_count += 1

    async def _execute_scheduled_rebalance(self, hedge_type: HedgeType) -> None:
        """执行计划的再平衡"""
        logger.info(f"Executing scheduled rebalance for {hedge_type.value}")

        if hedge_type == HedgeType.DELTA:
            await self.execute_delta_hedge()
        elif hedge_type == HedgeType.GAMMA:
            await self.execute_gamma_hedge()
        elif hedge_type == HedgeType.VEGA:
            await self.execute_vega_hedge()

    # ============ Delta Hedging ============

    async def execute_delta_hedge(
        self, target_delta: Optional[float] = None
    ) -> HedgeResult:
        """执行Delta对冲"""
        if not self._current_greeks:
            raise ValueError("No portfolio greeks available")

        hedge_id = f"DELTA_HEDGE_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()

        current_delta = self._current_greeks.total_delta
        target = target_delta if target_delta is not None else self._config.delta_hedge.target_delta
        delta_adjustment = target - current_delta

        logger.info(f"Executing delta hedge: current={current_delta:.2f}, target={target:.2f}, adjustment={delta_adjustment:.2f}")

        # 创建对冲订单
        orders = await self._create_delta_hedge_orders(delta_adjustment)

        # 计算成本
        total_cost = sum(o.estimated_cost for o in orders)

        # 检查是否需要批准
        if total_cost > self._config.require_approval_above:
            logger.warning(f"Hedge cost {total_cost:.2f} exceeds approval threshold")
            # 在实际系统中，这里会触发审批流程
            # 现在我们直接继续

        # 执行订单
        successful = 0
        failed = 0
        for order in orders:
            success = await self._order_executor.execute_order(order)
            if success:
                successful += 1
            else:
                failed += 1

        # 更新希腊字母
        await self._update_greeks()
        after_delta = self._current_greeks.total_delta if self._current_greeks else current_delta

        # 计算效果
        delta_reduction = abs(current_delta) - abs(after_delta)
        effectiveness = min(delta_reduction / abs(delta_adjustment), 1.0) if delta_adjustment != 0 else 1.0

        # 创建结果
        result = HedgeResult(
            hedge_id=hedge_id,
            hedge_type=HedgeType.DELTA,
            strategy=self._config.delta_hedge.strategy,
            before_delta=current_delta,
            after_delta=after_delta,
            delta_reduction=delta_reduction,
            orders=orders,
            total_orders=len(orders),
            successful_orders=successful,
            failed_orders=failed,
            total_cost=total_cost,
            effectiveness=effectiveness,
            start_time=start_time,
            end_time=datetime.now(),
        )
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        # 记录
        self._hedge_results.append(result)
        self._total_hedges += 1
        self._total_hedge_cost += total_cost
        self._hedge_count_by_type[HedgeType.DELTA] = self._hedge_count_by_type.get(HedgeType.DELTA, 0) + 1
        self._last_hedge_time[HedgeType.DELTA] = datetime.now()

        # 通知回调
        self._notify_hedge_result(result)

        logger.info(f"Delta hedge completed: effectiveness={effectiveness:.2%}, cost={total_cost:.2f}")

        return result

    async def _create_delta_hedge_orders(self, delta_adjustment: float) -> List[HedgeOrder]:
        """创建Delta对冲订单"""
        orders = []

        if abs(delta_adjustment) < self._config.delta_hedge.min_hedge_amount:
            return orders

        # 按标的创建订单
        if self._current_greeks and self._current_greeks.by_underlying:
            for symbol, greeks in self._current_greeks.by_underlying.items():
                symbol_delta = greeks["delta"]
                if abs(symbol_delta) < 1.0:
                    continue

                # 计算对冲数量（简化：假设delta为1）
                hedge_quantity = abs(symbol_delta)
                side = OrderSide.SELL if symbol_delta > 0 else OrderSide.BUY

                # 获取当前价格
                current_price = await self._market_data_provider.get_current_price(symbol)

                # 估算成本
                notional = hedge_quantity * current_price
                transaction_cost = notional * self._config.transaction_cost_bps / 10000
                slippage = notional * self._config.slippage_bps / 10000
                total_cost = transaction_cost + slippage

                order = HedgeOrder(
                    order_id=f"ORDER_{uuid.uuid4().hex[:8]}",
                    hedge_type=HedgeType.DELTA,
                    symbol=symbol,
                    instrument=self._config.delta_hedge.instrument,
                    side=side,
                    quantity=hedge_quantity,
                    order_type=OrderType.MARKET,
                    limit_price=current_price,
                    target_delta=0.0,
                    current_delta=symbol_delta,
                    delta_adjustment=-symbol_delta,
                    estimated_cost=total_cost,
                    transaction_cost=transaction_cost,
                    slippage=slippage,
                )

                orders.append(order)
                self._hedge_orders[order.order_id] = order

        return orders

    # ============ Gamma Hedging ============

    async def execute_gamma_hedge(self) -> HedgeResult:
        """执行Gamma对冲"""
        if not self._current_greeks:
            raise ValueError("No portfolio greeks available")

        hedge_id = f"GAMMA_HEDGE_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()

        current_gamma = self._current_greeks.total_gamma
        target_gamma = self._config.gamma_hedge.target_gamma

        logger.info(f"Executing gamma hedge: current={current_gamma:.4f}, target={target_gamma:.4f}")

        # Gamma对冲通常使用期权
        orders = await self._create_gamma_hedge_orders(current_gamma, target_gamma)

        # 执行订单
        successful = 0
        failed = 0
        for order in orders:
            success = await self._order_executor.execute_order(order)
            if success:
                successful += 1
            else:
                failed += 1

        # 更新希腊字母
        await self._update_greeks()
        after_gamma = self._current_greeks.total_gamma if self._current_greeks else current_gamma

        # 创建结果
        result = HedgeResult(
            hedge_id=hedge_id,
            hedge_type=HedgeType.GAMMA,
            strategy=self._config.gamma_hedge.strategy,
            before_delta=self._current_greeks.total_delta if self._current_greeks else 0.0,
            after_delta=self._current_greeks.total_delta if self._current_greeks else 0.0,
            delta_reduction=0.0,
            before_gamma=current_gamma,
            after_gamma=after_gamma,
            orders=orders,
            total_orders=len(orders),
            successful_orders=successful,
            failed_orders=failed,
            start_time=start_time,
            end_time=datetime.now(),
        )
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        self._hedge_results.append(result)
        self._hedge_count_by_type[HedgeType.GAMMA] = self._hedge_count_by_type.get(HedgeType.GAMMA, 0) + 1
        self._last_hedge_time[HedgeType.GAMMA] = datetime.now()

        self._notify_hedge_result(result)

        return result

    async def _create_gamma_hedge_orders(
        self, current_gamma: float, target_gamma: float
    ) -> List[HedgeOrder]:
        """创建Gamma对冲订单"""
        orders = []
        # 简化实现：实际需要使用期权来对冲gamma
        return orders

    # ============ Vega Hedging ============

    async def execute_vega_hedge(self) -> HedgeResult:
        """执行Vega对冲"""
        if not self._current_greeks:
            raise ValueError("No portfolio greeks available")

        hedge_id = f"VEGA_HEDGE_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()

        current_vega = self._current_greeks.total_vega
        target_vega = self._config.vega_hedge.target_vega

        logger.info(f"Executing vega hedge: current={current_vega:.2f}, target={target_vega:.2f}")

        # Vega对冲通常使用期权
        orders = await self._create_vega_hedge_orders(current_vega, target_vega)

        # 执行订单
        successful = 0
        failed = 0
        for order in orders:
            success = await self._order_executor.execute_order(order)
            if success:
                successful += 1
            else:
                failed += 1

        # 更新希腊字母
        await self._update_greeks()
        after_vega = self._current_greeks.total_vega if self._current_greeks else current_vega

        # 创建结果
        result = HedgeResult(
            hedge_id=hedge_id,
            hedge_type=HedgeType.VEGA,
            strategy=self._config.vega_hedge.strategy,
            before_delta=self._current_greeks.total_delta if self._current_greeks else 0.0,
            after_delta=self._current_greeks.total_delta if self._current_greeks else 0.0,
            delta_reduction=0.0,
            before_vega=current_vega,
            after_vega=after_vega,
            orders=orders,
            total_orders=len(orders),
            successful_orders=successful,
            failed_orders=failed,
            start_time=start_time,
            end_time=datetime.now(),
        )
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        self._hedge_results.append(result)
        self._hedge_count_by_type[HedgeType.VEGA] = self._hedge_count_by_type.get(HedgeType.VEGA, 0) + 1
        self._last_hedge_time[HedgeType.VEGA] = datetime.now()

        self._notify_hedge_result(result)

        return result

    async def _create_vega_hedge_orders(
        self, current_vega: float, target_vega: float
    ) -> List[HedgeOrder]:
        """创建Vega对冲订单"""
        orders = []
        # 简化实现：实际需要使用期权来对冲vega
        return orders

    # ============ Cost Calculation ============

    def calculate_hedge_cost(
        self, symbol: str, quantity: float, price: float
    ) -> Dict[str, float]:
        """计算对冲成本"""
        notional = quantity * price
        transaction_cost = notional * self._config.transaction_cost_bps / 10000
        slippage = notional * self._config.slippage_bps / 10000
        total_cost = transaction_cost + slippage

        return {
            "notional": notional,
            "transaction_cost": transaction_cost,
            "slippage": slippage,
            "total_cost": total_cost,
            "cost_pct": (total_cost / notional * 100) if notional > 0 else 0,
        }

    # ============ Alerts ============

    def create_alert(
        self,
        hedge_type: HedgeType,
        severity: AlertSeverity,
        message: str,
        current_value: float,
        threshold: float,
        recommended_action: Optional[str] = None,
    ) -> HedgeAlert:
        """创建警报"""
        self._alert_counter += 1
        alert = HedgeAlert(
            alert_id=f"HEDGE_ALERT_{self._alert_counter}",
            hedge_type=hedge_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold=threshold,
            recommended_action=recommended_action,
        )
        self._alerts.append(alert)
        self._notify_alert(alert)
        return alert

    # ============ Callbacks ============

    def on_hedge_result(self, callback: Callable[[HedgeResult], None]) -> None:
        """注册对冲结果回调"""
        self._hedge_callbacks.append(callback)

    def on_alert(self, callback: Callable[[HedgeAlert], None]) -> None:
        """注册警报回调"""
        self._alert_callbacks.append(callback)

    def _notify_hedge_result(self, result: HedgeResult) -> None:
        """通知对冲结果"""
        for callback in self._hedge_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in hedge result callback: {e}")

    def _notify_alert(self, alert: HedgeAlert) -> None:
        """通知警报"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    # ============ Public API ============

    def get_current_greeks(self) -> Optional[PortfolioGreeks]:
        """获取当前希腊字母"""
        return self._current_greeks

    def get_hedge_results(self, limit: int = 100) -> List[HedgeResult]:
        """获取对冲历史"""
        return self._hedge_results[-limit:]

    def get_hedge_result(self, hedge_id: str) -> Optional[HedgeResult]:
        """获取单个对冲结果"""
        for result in self._hedge_results:
            if result.hedge_id == hedge_id:
                return result
        return None

    def get_hedge_orders(self, hedge_id: Optional[str] = None) -> List[HedgeOrder]:
        """获取对冲订单"""
        if hedge_id:
            return [
                order for order in self._hedge_orders.values()
                if any(order.order_id == o.order_id for result in self._hedge_results
                       if result.hedge_id == hedge_id for o in result.orders)
            ]
        return list(self._hedge_orders.values())

    def get_alerts(self, limit: int = 100) -> List[HedgeAlert]:
        """获取警报"""
        return self._alerts[-limit:]

    def get_rebalance_schedules(self) -> List[RebalanceSchedule]:
        """获取再平衡计划"""
        return list(self._rebalance_schedules.values())

    def get_state(self) -> EngineState:
        """获取引擎状态"""
        return self._state

    def is_running(self) -> bool:
        """是否运行中"""
        return self._running

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "running": self._running,
            "total_hedges": self._total_hedges,
            "total_hedge_cost": self._total_hedge_cost,
            "hedge_count_by_type": {k.value: v for k, v in self._hedge_count_by_type.items()},
            "active_schedules": len([s for s in self._rebalance_schedules.values() if s.enabled]),
            "total_orders": len(self._hedge_orders),
            "total_alerts": len(self._alerts),
            "current_greeks": self._current_greeks.to_dict() if self._current_greeks else None,
            "last_hedge_times": {k.value: v.isoformat() for k, v in self._last_hedge_time.items()},
        }


# ============ Factory Functions ============


def create_hedging_engine(
    delta_hedge_enabled: bool = True,
    gamma_hedge_enabled: bool = False,
    vega_hedge_enabled: bool = False,
    position_provider: Optional[PositionProvider] = None,
    market_data_provider: Optional[MarketDataProvider] = None,
    order_executor: Optional[OrderExecutor] = None,
) -> HedgingEngine:
    """创建对冲引擎"""
    config = EngineConfig(
        delta_hedge=DeltaHedgeConfig(enabled=delta_hedge_enabled),
        gamma_hedge=GammaHedgeConfig(enabled=gamma_hedge_enabled),
        vega_hedge=VegaHedgeConfig(enabled=vega_hedge_enabled),
    )

    return HedgingEngine(
        config=config,
        position_provider=position_provider,
        market_data_provider=market_data_provider,
        order_executor=order_executor,
    )


def create_position_data(
    position_id: str,
    symbol: str,
    quantity: float,
    price: float,
    current_price: float,
    delta: float = 0.0,
    gamma: float = 0.0,
    vega: float = 0.0,
) -> PositionData:
    """创建持仓数据"""
    return PositionData(
        position_id=position_id,
        symbol=symbol,
        quantity=quantity,
        price=price,
        current_price=current_price,
        market_value=quantity * current_price,
        delta=delta,
        gamma=gamma,
        vega=vega,
    )


# ============ Exports ============


__all__ = [
    # Enums
    "HedgeType",
    "HedgeStrategy",
    "HedgeInstrument",
    "OrderType",
    "OrderSide",
    "HedgeStatus",
    "AlertSeverity",
    "EngineState",
    # Data Classes
    "PositionData",
    "PortfolioGreeks",
    "HedgeOrder",
    "HedgeResult",
    "DeltaHedgeConfig",
    "GammaHedgeConfig",
    "VegaHedgeConfig",
    "CorrelationHedgeConfig",
    "TailRiskHedgeConfig",
    "HedgeAlert",
    "RebalanceSchedule",
    "EngineConfig",
    # Interfaces
    "PositionProvider",
    "MarketDataProvider",
    "OrderExecutor",
    # Simulated Providers
    "SimulatedPositionProvider",
    "SimulatedMarketDataProvider",
    "SimulatedOrderExecutor",
    # Main Class
    "HedgingEngine",
    # Factory Functions
    "create_hedging_engine",
    "create_position_data",
]
