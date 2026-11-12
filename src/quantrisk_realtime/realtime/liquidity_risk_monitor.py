"""
Liquidity Risk Monitor - 流动性风险监控器

实时监控市场流动性风险，包括：
- 买卖价差监控
- 市场深度分析
- 流动性成本估算
- 流动性枯竭预警
- 分批退出策略
- 流动性覆盖率 (LCR) 计算

这是风险管理的关键模块，用于预防流动性危机。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
import statistics
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Enums ============

class LiquidityLevel(str, Enum):
    """流动性等级"""
    EXCELLENT = "excellent"      # 优秀 - 可以大量交易
    GOOD = "good"                # 良好 - 正常交易
    MODERATE = "moderate"        # 中等 - 需注意
    LOW = "low"                  # 低 - 限制交易
    CRITICAL = "critical"        # 危险 - 暂停交易
    FROZEN = "frozen"            # 冻结 - 无流动性


class AlertSeverity(str, Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitorState(str, Enum):
    """监控器状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class ExitStrategy(str, Enum):
    """退出策略"""
    IMMEDIATE = "immediate"      # 立即全部退出
    STAGED = "staged"            # 分批退出
    TWAP = "twap"                # 时间加权
    VWAP = "vwap"                # 成交量加权
    ADAPTIVE = "adaptive"        # 自适应


class SpreadType(str, Enum):
    """价差类型"""
    BID_ASK = "bid_ask"          # 买卖价差
    EFFECTIVE = "effective"      # 有效价差
    REALIZED = "realized"        # 实现价差
    QUOTED = "quoted"            # 报价价差


# ============ Data Classes ============

@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    quantity: float
    num_orders: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": self.price,
            "quantity": self.quantity,
            "num_orders": self.num_orders,
        }


@dataclass
class OrderBook:
    """订单簿"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)  # 买单 (价格从高到低)
    asks: List[OrderBookLevel] = field(default_factory=list)  # 卖单 (价格从低到高)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """价差（基点）"""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 10000
        return None

    def get_depth(self, side: str, levels: int = 5) -> float:
        """获取某一侧的深度"""
        book = self.bids if side == "bid" else self.asks
        return sum(level.quantity for level in book[:levels])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bid_depth_5": self.get_depth("bid", 5),
            "ask_depth_5": self.get_depth("ask", 5),
            "bids": [b.to_dict() for b in self.bids[:10]],
            "asks": [a.to_dict() for a in self.asks[:10]],
        }


@dataclass
class LiquidityMetrics:
    """流动性指标"""
    symbol: str
    timestamp: datetime

    # 价差指标
    bid_ask_spread: float = 0.0           # 买卖价差
    spread_bps: float = 0.0               # 价差（基点）
    effective_spread: float = 0.0          # 有效价差

    # 深度指标
    bid_depth: float = 0.0                 # 买方深度
    ask_depth: float = 0.0                 # 卖方深度
    total_depth: float = 0.0               # 总深度
    depth_imbalance: float = 0.0           # 深度不平衡 (-1 到 1)

    # 成交量指标
    adv: float = 0.0                       # 平均日成交量
    current_volume: float = 0.0            # 当前成交量
    volume_ratio: float = 0.0              # 成交量比率

    # 流动性评分
    liquidity_score: float = 0.0           # 流动性综合评分 (0-100)
    liquidity_level: LiquidityLevel = LiquidityLevel.MODERATE

    # 市场影响估算
    estimated_impact_1pct: float = 0.0     # 1%仓位的预估影响
    estimated_impact_5pct: float = 0.0     # 5%仓位的预估影响
    estimated_impact_10pct: float = 0.0    # 10%仓位的预估影响

    # Amihud非流动性指标
    amihud_illiquidity: float = 0.0

    # Kyle's Lambda (价格影响系数)
    kyle_lambda: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid_ask_spread": self.bid_ask_spread,
            "spread_bps": self.spread_bps,
            "effective_spread": self.effective_spread,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "total_depth": self.total_depth,
            "depth_imbalance": self.depth_imbalance,
            "adv": self.adv,
            "current_volume": self.current_volume,
            "volume_ratio": self.volume_ratio,
            "liquidity_score": self.liquidity_score,
            "liquidity_level": self.liquidity_level.value,
            "estimated_impact_1pct": self.estimated_impact_1pct,
            "estimated_impact_5pct": self.estimated_impact_5pct,
            "estimated_impact_10pct": self.estimated_impact_10pct,
            "amihud_illiquidity": self.amihud_illiquidity,
            "kyle_lambda": self.kyle_lambda,
        }


@dataclass
class LiquidityAlert:
    """流动性告警"""
    alert_id: str
    symbol: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: str
    message: str
    current_value: float
    threshold: float
    metrics: Optional[LiquidityMetrics] = None
    recommended_action: str = ""
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "alert_type": self.alert_type,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "recommended_action": self.recommended_action,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class LiquidationCostEstimate:
    """清算成本估算"""
    symbol: str
    timestamp: datetime
    position_size: float
    position_value: float

    # 成本组成
    spread_cost: float = 0.0               # 价差成本
    market_impact_cost: float = 0.0        # 市场影响成本
    timing_cost: float = 0.0               # 时机成本
    total_cost: float = 0.0                # 总成本
    total_cost_bps: float = 0.0            # 总成本（基点）

    # 时间估算
    estimated_time_minutes: float = 0.0    # 预估完成时间

    # 推荐策略
    recommended_strategy: ExitStrategy = ExitStrategy.STAGED
    recommended_slices: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "position_size": self.position_size,
            "position_value": self.position_value,
            "spread_cost": self.spread_cost,
            "market_impact_cost": self.market_impact_cost,
            "timing_cost": self.timing_cost,
            "total_cost": self.total_cost,
            "total_cost_bps": self.total_cost_bps,
            "estimated_time_minutes": self.estimated_time_minutes,
            "recommended_strategy": self.recommended_strategy.value,
            "recommended_slices": self.recommended_slices,
        }


@dataclass
class ExitPlan:
    """退出计划"""
    plan_id: str
    symbol: str
    created_at: datetime
    position_size: float
    strategy: ExitStrategy

    # 计划详情
    slices: List[Dict[str, Any]] = field(default_factory=list)
    total_slices: int = 1
    completed_slices: int = 0

    # 状态
    status: str = "pending"  # pending, executing, completed, cancelled

    # 执行结果
    executed_quantity: float = 0.0
    average_price: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "symbol": self.symbol,
            "created_at": self.created_at.isoformat(),
            "position_size": self.position_size,
            "strategy": self.strategy.value,
            "slices": self.slices,
            "total_slices": self.total_slices,
            "completed_slices": self.completed_slices,
            "status": self.status,
            "executed_quantity": self.executed_quantity,
            "average_price": self.average_price,
            "total_cost": self.total_cost,
        }


@dataclass
class LiquidityCoverageRatio:
    """流动性覆盖率 (LCR)"""
    timestamp: datetime

    # 高质量流动性资产 (HQLA)
    hqla_level1: float = 0.0              # 一级资产（现金、央行存款）
    hqla_level2a: float = 0.0             # 二级A资产（政府债券）
    hqla_level2b: float = 0.0             # 二级B资产（公司债券）
    total_hqla: float = 0.0

    # 净现金流出
    cash_outflows: float = 0.0
    cash_inflows: float = 0.0
    net_cash_outflows: float = 0.0

    # LCR
    lcr: float = 0.0                       # 流动性覆盖率
    lcr_requirement: float = 100.0         # 监管要求 (100%)
    is_compliant: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "hqla_level1": self.hqla_level1,
            "hqla_level2a": self.hqla_level2a,
            "hqla_level2b": self.hqla_level2b,
            "total_hqla": self.total_hqla,
            "cash_outflows": self.cash_outflows,
            "cash_inflows": self.cash_inflows,
            "net_cash_outflows": self.net_cash_outflows,
            "lcr": self.lcr,
            "lcr_requirement": self.lcr_requirement,
            "is_compliant": self.is_compliant,
        }


@dataclass
class LiquidityStressScenario:
    """流动性压力情景"""
    scenario_id: str
    name: str
    description: str

    # 压力因子
    spread_multiplier: float = 1.0         # 价差放大倍数
    depth_reduction: float = 0.0           # 深度减少比例
    volume_reduction: float = 0.0          # 成交量减少比例
    impact_multiplier: float = 1.0         # 影响放大倍数

    # 历史参考
    historical_event: Optional[str] = None  # 如 "2008金融危机"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "spread_multiplier": self.spread_multiplier,
            "depth_reduction": self.depth_reduction,
            "volume_reduction": self.volume_reduction,
            "impact_multiplier": self.impact_multiplier,
            "historical_event": self.historical_event,
        }


@dataclass
class LiquidityStressResult:
    """流动性压力测试结果"""
    scenario: LiquidityStressScenario
    timestamp: datetime

    # 压力后指标
    stressed_spread_bps: float = 0.0
    stressed_depth: float = 0.0
    stressed_impact: float = 0.0
    stressed_liquidation_cost: float = 0.0

    # 影响评估
    additional_cost_bps: float = 0.0
    liquidation_time_increase: float = 0.0

    # 风险评级
    risk_level: str = "moderate"  # low, moderate, high, severe

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "stressed_spread_bps": self.stressed_spread_bps,
            "stressed_depth": self.stressed_depth,
            "stressed_impact": self.stressed_impact,
            "stressed_liquidation_cost": self.stressed_liquidation_cost,
            "additional_cost_bps": self.additional_cost_bps,
            "liquidation_time_increase": self.liquidation_time_increase,
            "risk_level": self.risk_level,
        }


@dataclass
class LiquidityMonitorConfig:
    """流动性监控配置"""
    # 监控间隔
    update_interval_sec: float = 5.0

    # 价差阈值（基点）
    spread_warning_bps: float = 50.0
    spread_critical_bps: float = 100.0
    spread_emergency_bps: float = 200.0

    # 深度阈值
    min_depth_ratio: float = 0.1           # 最小深度比率
    depth_warning_ratio: float = 0.3

    # 流动性评分阈值
    liquidity_warning_score: float = 40.0
    liquidity_critical_score: float = 20.0

    # 市场影响阈值（基点）
    impact_warning_bps: float = 50.0
    impact_critical_bps: float = 100.0

    # 告警冷却时间（秒）
    alert_cooldown_sec: float = 60.0

    # 历史数据保留
    max_history_size: int = 1000

    # ADV计算窗口
    adv_window_days: int = 20

    # 退出策略参数
    max_participation_rate: float = 0.1    # 最大参与率 (10%)
    min_slice_interval_sec: float = 30.0   # 最小切片间隔

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_interval_sec": self.update_interval_sec,
            "spread_warning_bps": self.spread_warning_bps,
            "spread_critical_bps": self.spread_critical_bps,
            "spread_emergency_bps": self.spread_emergency_bps,
            "min_depth_ratio": self.min_depth_ratio,
            "depth_warning_ratio": self.depth_warning_ratio,
            "liquidity_warning_score": self.liquidity_warning_score,
            "liquidity_critical_score": self.liquidity_critical_score,
            "impact_warning_bps": self.impact_warning_bps,
            "impact_critical_bps": self.impact_critical_bps,
            "alert_cooldown_sec": self.alert_cooldown_sec,
            "max_history_size": self.max_history_size,
            "adv_window_days": self.adv_window_days,
            "max_participation_rate": self.max_participation_rate,
            "min_slice_interval_sec": self.min_slice_interval_sec,
        }


# ============ Abstract Data Providers ============

class OrderBookProvider(ABC):
    """订单簿数据提供者接口"""

    @abstractmethod
    async def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """获取订单簿"""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str], callback: Callable[[OrderBook], None]) -> None:
        """订阅订单簿更新"""
        pass


class VolumeDataProvider(ABC):
    """成交量数据提供者接口"""

    @abstractmethod
    async def get_adv(self, symbol: str, days: int = 20) -> float:
        """获取平均日成交量"""
        pass

    @abstractmethod
    async def get_current_volume(self, symbol: str) -> float:
        """获取当前成交量"""
        pass

    @abstractmethod
    async def get_historical_volumes(self, symbol: str, days: int) -> List[float]:
        """获取历史成交量"""
        pass


class PositionDataProvider(ABC):
    """持仓数据提供者接口"""

    @abstractmethod
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有持仓"""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取单个持仓"""
        pass


# ============ Simulated Providers ============

class SimulatedOrderBookProvider(OrderBookProvider):
    """模拟订单簿提供者"""

    def __init__(self):
        self._order_books: Dict[str, OrderBook] = {}
        self._base_prices: Dict[str, float] = {}

    def set_base_price(self, symbol: str, price: float) -> None:
        self._base_prices[symbol] = price

    def set_order_book(self, symbol: str, order_book: OrderBook) -> None:
        self._order_books[symbol] = order_book

    async def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        if symbol in self._order_books:
            return self._order_books[symbol]

        # 生成模拟订单簿
        base_price = self._base_prices.get(symbol, 100.0)
        spread = base_price * 0.001  # 0.1% 价差

        bids = [
            OrderBookLevel(base_price - spread/2 - i * 0.01, 100 * (10 - i), 5)
            for i in range(10)
        ]
        asks = [
            OrderBookLevel(base_price + spread/2 + i * 0.01, 100 * (10 - i), 5)
            for i in range(10)
        ]

        return OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
        )

    async def subscribe(self, symbols: List[str], callback: Callable[[OrderBook], None]) -> None:
        pass


class SimulatedVolumeDataProvider(VolumeDataProvider):
    """模拟成交量数据提供者"""

    def __init__(self):
        self._adv_data: Dict[str, float] = {}
        self._current_volumes: Dict[str, float] = {}

    def set_adv(self, symbol: str, adv: float) -> None:
        self._adv_data[symbol] = adv

    def set_current_volume(self, symbol: str, volume: float) -> None:
        self._current_volumes[symbol] = volume

    async def get_adv(self, symbol: str, days: int = 20) -> float:
        return self._adv_data.get(symbol, 1000000.0)

    async def get_current_volume(self, symbol: str) -> float:
        return self._current_volumes.get(symbol, 500000.0)

    async def get_historical_volumes(self, symbol: str, days: int) -> List[float]:
        adv = await self.get_adv(symbol)
        import random
        return [adv * random.uniform(0.7, 1.3) for _ in range(days)]


class SimulatedPositionDataProvider(PositionDataProvider):
    """模拟持仓数据提供者"""

    def __init__(self):
        self._positions: Dict[str, Dict[str, Any]] = {}

    def set_position(self, symbol: str, quantity: float, avg_price: float) -> None:
        self._positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "avg_price": avg_price,
            "market_value": quantity * avg_price,
        }

    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        return self._positions.copy()

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._positions.get(symbol)


# ============ Main Monitor Class ============

class LiquidityRiskMonitor:
    """流动性风险监控器"""

    def __init__(
        self,
        config: Optional[LiquidityMonitorConfig] = None,
        order_book_provider: Optional[OrderBookProvider] = None,
        volume_provider: Optional[VolumeDataProvider] = None,
        position_provider: Optional[PositionDataProvider] = None,
        on_alert: Optional[Callable[[LiquidityAlert], None]] = None,
        on_metrics_update: Optional[Callable[[str, LiquidityMetrics], None]] = None,
    ):
        self.config = config or LiquidityMonitorConfig()
        self._order_book_provider = order_book_provider or simulated_default(SimulatedOrderBookProvider, "data from LiquidityRiskMonitor's simulated order_book_provider (SimulatedOrderBookProvider)")
        self._volume_provider = volume_provider or simulated_default(SimulatedVolumeDataProvider, "data from LiquidityRiskMonitor's simulated volume_provider (SimulatedVolumeDataProvider)")
        self._position_provider = position_provider or simulated_default(SimulatedPositionDataProvider, "data from LiquidityRiskMonitor's simulated position_provider (SimulatedPositionDataProvider)")

        self._on_alert = on_alert
        self._on_metrics_update = on_metrics_update

        # 状态
        self._state = MonitorState.STOPPED
        self._monitor_task: Optional[asyncio.Task] = None

        # 监控的标的
        self._watched_symbols: set = set()

        # 数据存储
        self._metrics_history: Dict[str, deque] = {}
        self._alerts: List[LiquidityAlert] = []
        self._exit_plans: Dict[str, ExitPlan] = {}
        self._last_alert_time: Dict[str, datetime] = {}

        # 预定义压力情景
        self._stress_scenarios = self._init_stress_scenarios()

        # 计数器
        self._alert_counter = 0
        self._plan_counter = 0

    def _init_stress_scenarios(self) -> Dict[str, LiquidityStressScenario]:
        """初始化预定义压力情景"""
        return {
            "normal_stress": LiquidityStressScenario(
                scenario_id="normal_stress",
                name="Normal Stress",
                description="Normal market stress conditions",
                spread_multiplier=2.0,
                depth_reduction=0.3,
                volume_reduction=0.2,
                impact_multiplier=1.5,
            ),
            "severe_stress": LiquidityStressScenario(
                scenario_id="severe_stress",
                name="Severe Stress",
                description="Severe market stress (similar to 2020 COVID crash)",
                spread_multiplier=5.0,
                depth_reduction=0.6,
                volume_reduction=0.4,
                impact_multiplier=3.0,
                historical_event="2020 COVID-19 Market Crash",
            ),
            "extreme_stress": LiquidityStressScenario(
                scenario_id="extreme_stress",
                name="Extreme Stress",
                description="Extreme market stress (similar to 2008 financial crisis)",
                spread_multiplier=10.0,
                depth_reduction=0.8,
                volume_reduction=0.6,
                impact_multiplier=5.0,
                historical_event="2008 Financial Crisis",
            ),
            "flash_crash": LiquidityStressScenario(
                scenario_id="flash_crash",
                name="Flash Crash",
                description="Flash crash scenario (sudden liquidity evaporation)",
                spread_multiplier=20.0,
                depth_reduction=0.95,
                volume_reduction=0.8,
                impact_multiplier=10.0,
                historical_event="2010 Flash Crash",
            ),
        }

    @property
    def state(self) -> MonitorState:
        return self._state

    # ============ 监控控制 ============

    async def start(self) -> None:
        """启动监控"""
        if self._state == MonitorState.RUNNING:
            return

        self._state = MonitorState.STARTING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._state = MonitorState.RUNNING
        logger.info("Liquidity Risk Monitor started")

    async def stop(self) -> None:
        """停止监控"""
        self._state = MonitorState.STOPPED
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Liquidity Risk Monitor stopped")

    async def pause(self) -> None:
        """暂停监控"""
        self._state = MonitorState.PAUSED
        logger.info("Liquidity Risk Monitor paused")

    async def resume(self) -> None:
        """恢复监控"""
        if self._state == MonitorState.PAUSED:
            self._state = MonitorState.RUNNING
            logger.info("Liquidity Risk Monitor resumed")

    async def _monitor_loop(self) -> None:
        """监控主循环"""
        while self._state in [MonitorState.RUNNING, MonitorState.PAUSED]:
            try:
                if self._state == MonitorState.RUNNING:
                    await self._update_all_metrics()
                await asyncio.sleep(self.config.update_interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self._state = MonitorState.ERROR

    # ============ 标的管理 ============

    def add_symbol(self, symbol: str) -> None:
        """添加监控标的"""
        self._watched_symbols.add(symbol)
        if symbol not in self._metrics_history:
            self._metrics_history[symbol] = deque(maxlen=self.config.max_history_size)

    def remove_symbol(self, symbol: str) -> None:
        """移除监控标的"""
        self._watched_symbols.discard(symbol)

    def get_watched_symbols(self) -> List[str]:
        """获取所有监控标的"""
        return list(self._watched_symbols)

    # ============ 指标计算 ============

    async def _update_all_metrics(self) -> None:
        """更新所有标的的指标"""
        for symbol in self._watched_symbols:
            try:
                metrics = await self.calculate_metrics(symbol)
                if metrics:
                    self._store_metrics(symbol, metrics)
                    await self._check_alerts(metrics)
                    if self._on_metrics_update:
                        self._on_metrics_update(symbol, metrics)
            except Exception as e:
                logger.error(f"Error updating metrics for {symbol}: {e}")

    async def calculate_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """计算流动性指标"""
        order_book = await self._order_book_provider.get_order_book(symbol)
        if not order_book:
            return None

        adv = await self._volume_provider.get_adv(symbol, self.config.adv_window_days)
        current_volume = await self._volume_provider.get_current_volume(symbol)

        # 计算价差指标
        spread = order_book.spread or 0
        spread_bps = order_book.spread_bps or 0

        # 计算深度指标
        bid_depth = order_book.get_depth("bid", 5)
        ask_depth = order_book.get_depth("ask", 5)
        total_depth = bid_depth + ask_depth

        # 深度不平衡 (-1 卖方强, +1 买方强)
        depth_imbalance = 0.0
        if total_depth > 0:
            depth_imbalance = (bid_depth - ask_depth) / total_depth

        # 计算成交量比率
        volume_ratio = current_volume / adv if adv > 0 else 0

        # 计算Amihud非流动性指标
        amihud = self._calculate_amihud(symbol, order_book.mid_price or 0, current_volume)

        # 计算Kyle's Lambda
        kyle_lambda = self._calculate_kyle_lambda(order_book, current_volume)

        # 计算市场影响
        impact_1pct = self._estimate_market_impact(order_book, adv, 0.01)
        impact_5pct = self._estimate_market_impact(order_book, adv, 0.05)
        impact_10pct = self._estimate_market_impact(order_book, adv, 0.10)

        # 计算流动性综合评分
        liquidity_score = self._calculate_liquidity_score(
            spread_bps, total_depth, adv, volume_ratio, depth_imbalance
        )

        # 确定流动性等级
        liquidity_level = self._determine_liquidity_level(liquidity_score)

        return LiquidityMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_ask_spread=spread,
            spread_bps=spread_bps,
            effective_spread=spread * 1.1,  # 简化的有效价差
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_depth=total_depth,
            depth_imbalance=depth_imbalance,
            adv=adv,
            current_volume=current_volume,
            volume_ratio=volume_ratio,
            liquidity_score=liquidity_score,
            liquidity_level=liquidity_level,
            estimated_impact_1pct=impact_1pct,
            estimated_impact_5pct=impact_5pct,
            estimated_impact_10pct=impact_10pct,
            amihud_illiquidity=amihud,
            kyle_lambda=kyle_lambda,
        )

    def _calculate_amihud(self, symbol: str, price: float, volume: float) -> float:
        """计算Amihud非流动性指标"""
        if volume == 0 or price == 0:
            return 0.0

        # 获取历史数据计算
        history = list(self._metrics_history.get(symbol, []))
        if len(history) < 2:
            return 0.0

        # 计算平均 |return| / volume
        returns = []
        for i in range(1, len(history)):
            prev_price = history[i-1].bid_ask_spread  # 使用spread作为price proxy
            curr_price = history[i].bid_ask_spread
            if prev_price > 0:
                ret = abs(curr_price - prev_price) / prev_price
                vol = history[i].current_volume
                if vol > 0:
                    returns.append(ret / vol)

        return statistics.mean(returns) * 1e6 if returns else 0.0

    def _calculate_kyle_lambda(self, order_book: OrderBook, volume: float) -> float:
        """计算Kyle's Lambda (价格影响系数)"""
        if volume == 0:
            return 0.0

        # 简化计算: 价差 / (2 * 深度)
        spread = order_book.spread or 0
        total_depth = order_book.get_depth("bid", 5) + order_book.get_depth("ask", 5)

        if total_depth == 0:
            return float('inf')

        return spread / (2 * total_depth)

    def _estimate_market_impact(
        self,
        order_book: OrderBook,
        adv: float,
        position_pct: float,
    ) -> float:
        """
        估算市场影响 (基于Almgren-Chriss模型简化版)

        Impact = spread/2 + k * sigma * sqrt(Q/V)

        其中:
        - k: 影响系数 (约0.1-0.5)
        - sigma: 波动率 (假设)
        - Q: 订单量
        - V: 日成交量
        """
        if adv == 0:
            return float('inf')

        spread_bps = order_book.spread_bps or 0
        k = 0.3  # 影响系数
        sigma = 0.02  # 假设日波动率2%

        # 计算影响 (基点)
        permanent_impact = spread_bps / 2
        temporary_impact = k * sigma * 10000 * (position_pct ** 0.5)

        return permanent_impact + temporary_impact

    def _calculate_liquidity_score(
        self,
        spread_bps: float,
        total_depth: float,
        adv: float,
        volume_ratio: float,
        depth_imbalance: float,
    ) -> float:
        """
        计算流动性综合评分 (0-100)

        评分维度:
        - 价差评分 (30%): 越小越好
        - 深度评分 (30%): 越大越好
        - 成交量评分 (25%): ADV越大越好
        - 平衡性评分 (15%): 越平衡越好
        """
        # 价差评分 (0-30)
        if spread_bps <= 5:
            spread_score = 30
        elif spread_bps <= 20:
            spread_score = 30 - (spread_bps - 5) * 1.0
        elif spread_bps <= 100:
            spread_score = 15 - (spread_bps - 20) * 0.15
        else:
            spread_score = max(0, 5 - (spread_bps - 100) * 0.05)

        # 深度评分 (0-30)
        # 假设正常深度为 ADV 的 1%
        expected_depth = adv * 0.01
        if expected_depth > 0:
            depth_ratio = total_depth / expected_depth
            depth_score = min(30, depth_ratio * 15)
        else:
            depth_score = 15 if total_depth > 0 else 0

        # 成交量评分 (0-25)
        if adv >= 10000000:  # 1000万+
            volume_score = 25
        elif adv >= 1000000:  # 100万+
            volume_score = 20 + (adv - 1000000) / 9000000 * 5
        elif adv >= 100000:  # 10万+
            volume_score = 10 + (adv - 100000) / 900000 * 10
        else:
            volume_score = adv / 100000 * 10

        # 平衡性评分 (0-15)
        balance_score = 15 * (1 - abs(depth_imbalance))

        return min(100, max(0, spread_score + depth_score + volume_score + balance_score))

    def _determine_liquidity_level(self, score: float) -> LiquidityLevel:
        """根据评分确定流动性等级"""
        if score >= 80:
            return LiquidityLevel.EXCELLENT
        elif score >= 60:
            return LiquidityLevel.GOOD
        elif score >= 40:
            return LiquidityLevel.MODERATE
        elif score >= 20:
            return LiquidityLevel.LOW
        elif score >= 5:
            return LiquidityLevel.CRITICAL
        else:
            return LiquidityLevel.FROZEN

    def _store_metrics(self, symbol: str, metrics: LiquidityMetrics) -> None:
        """存储指标历史"""
        if symbol not in self._metrics_history:
            self._metrics_history[symbol] = deque(maxlen=self.config.max_history_size)
        self._metrics_history[symbol].append(metrics)

    # ============ 告警管理 ============

    async def _check_alerts(self, metrics: LiquidityMetrics) -> None:
        """检查并生成告警"""
        symbol = metrics.symbol
        now = datetime.now()

        # 检查冷却时间
        last_alert = self._last_alert_time.get(symbol)
        if last_alert and (now - last_alert).total_seconds() < self.config.alert_cooldown_sec:
            return

        alerts_to_send = []

        # 价差告警
        if metrics.spread_bps >= self.config.spread_emergency_bps:
            alerts_to_send.append(self._create_alert(
                symbol, AlertSeverity.EMERGENCY, "spread_emergency",
                f"Spread extremely wide: {metrics.spread_bps:.1f} bps",
                metrics.spread_bps, self.config.spread_emergency_bps,
                metrics, "Halt trading immediately. Consider staged exit."
            ))
        elif metrics.spread_bps >= self.config.spread_critical_bps:
            alerts_to_send.append(self._create_alert(
                symbol, AlertSeverity.CRITICAL, "spread_critical",
                f"Spread critically wide: {metrics.spread_bps:.1f} bps",
                metrics.spread_bps, self.config.spread_critical_bps,
                metrics, "Reduce position size. Use limit orders only."
            ))
        elif metrics.spread_bps >= self.config.spread_warning_bps:
            alerts_to_send.append(self._create_alert(
                symbol, AlertSeverity.WARNING, "spread_warning",
                f"Spread widening: {metrics.spread_bps:.1f} bps",
                metrics.spread_bps, self.config.spread_warning_bps,
                metrics, "Monitor closely. Avoid market orders."
            ))

        # 流动性评分告警
        if metrics.liquidity_score <= self.config.liquidity_critical_score:
            alerts_to_send.append(self._create_alert(
                symbol, AlertSeverity.CRITICAL, "liquidity_critical",
                f"Critical liquidity: score {metrics.liquidity_score:.1f}",
                metrics.liquidity_score, self.config.liquidity_critical_score,
                metrics, "Prepare exit plan. Reduce exposure."
            ))
        elif metrics.liquidity_score <= self.config.liquidity_warning_score:
            alerts_to_send.append(self._create_alert(
                symbol, AlertSeverity.WARNING, "liquidity_warning",
                f"Low liquidity: score {metrics.liquidity_score:.1f}",
                metrics.liquidity_score, self.config.liquidity_warning_score,
                metrics, "Review position sizing."
            ))

        # 深度不平衡告警
        if abs(metrics.depth_imbalance) > 0.7:
            alerts_to_send.append(self._create_alert(
                symbol, AlertSeverity.WARNING, "depth_imbalance",
                f"Significant depth imbalance: {metrics.depth_imbalance:.2f}",
                metrics.depth_imbalance, 0.7,
                metrics, "Market may be one-sided. Trade cautiously."
            ))

        # 发送告警
        for alert in alerts_to_send:
            self._alerts.append(alert)
            self._last_alert_time[symbol] = now
            if self._on_alert:
                self._on_alert(alert)

    def _create_alert(
        self,
        symbol: str,
        severity: AlertSeverity,
        alert_type: str,
        message: str,
        current_value: float,
        threshold: float,
        metrics: LiquidityMetrics,
        recommended_action: str,
    ) -> LiquidityAlert:
        """创建告警"""
        self._alert_counter += 1
        return LiquidityAlert(
            alert_id=f"LIQ-{self._alert_counter:06d}",
            symbol=symbol,
            timestamp=datetime.now(),
            severity=severity,
            alert_type=alert_type,
            message=message,
            current_value=current_value,
            threshold=threshold,
            metrics=metrics,
            recommended_action=recommended_action,
        )

    def get_alerts(
        self,
        symbol: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[LiquidityAlert]:
        """获取告警列表"""
        alerts = self._alerts.copy()

        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                return True
        return False

    # ============ 清算成本估算 ============

    async def estimate_liquidation_cost(
        self,
        symbol: str,
        position_size: float,
        urgency: str = "normal",  # normal, high, emergency
    ) -> LiquidationCostEstimate:
        """估算清算成本"""
        order_book = await self._order_book_provider.get_order_book(symbol)
        adv = await self._volume_provider.get_adv(symbol)

        if not order_book or not order_book.mid_price:
            raise ValueError(f"Cannot get order book for {symbol}")

        position_value = position_size * order_book.mid_price
        position_pct_adv = position_size / adv if adv > 0 else float('inf')

        # 价差成本
        spread_cost = (order_book.spread_bps or 0) / 2 * position_value / 10000

        # 市场影响成本 (基于Almgren-Chriss)
        urgency_factor = {"normal": 1.0, "high": 1.5, "emergency": 3.0}.get(urgency, 1.0)
        impact_bps = self._estimate_market_impact(order_book, adv, position_pct_adv)
        market_impact_cost = impact_bps * position_value / 10000 * urgency_factor

        # 时机成本 (仅在非紧急情况下有意义)
        timing_cost = 0 if urgency == "emergency" else position_value * 0.0005  # 5bps

        # 总成本
        total_cost = spread_cost + market_impact_cost + timing_cost
        total_cost_bps = total_cost / position_value * 10000 if position_value > 0 else 0

        # 估算完成时间
        max_rate = self.config.max_participation_rate
        if urgency == "emergency":
            max_rate = 0.25  # 紧急情况允许更高参与率

        daily_capacity = adv * max_rate
        days_to_liquidate = position_size / daily_capacity if daily_capacity > 0 else float('inf')
        estimated_minutes = days_to_liquidate * 390  # 假设6.5小时交易日

        # 推荐策略和切片数
        if urgency == "emergency":
            strategy = ExitStrategy.IMMEDIATE
            slices = 1
        elif position_pct_adv > 0.5:
            strategy = ExitStrategy.TWAP
            slices = max(10, int(position_pct_adv * 20))
        elif position_pct_adv > 0.1:
            strategy = ExitStrategy.STAGED
            slices = max(5, int(position_pct_adv * 10))
        else:
            strategy = ExitStrategy.ADAPTIVE
            slices = max(3, int(position_pct_adv * 5))

        return LiquidationCostEstimate(
            symbol=symbol,
            timestamp=datetime.now(),
            position_size=position_size,
            position_value=position_value,
            spread_cost=spread_cost,
            market_impact_cost=market_impact_cost,
            timing_cost=timing_cost,
            total_cost=total_cost,
            total_cost_bps=total_cost_bps,
            estimated_time_minutes=estimated_minutes,
            recommended_strategy=strategy,
            recommended_slices=slices,
        )

    # ============ 退出计划 ============

    async def create_exit_plan(
        self,
        symbol: str,
        position_size: float,
        strategy: Optional[ExitStrategy] = None,
    ) -> ExitPlan:
        """创建退出计划"""
        cost_estimate = await self.estimate_liquidation_cost(symbol, position_size)

        if strategy is None:
            strategy = cost_estimate.recommended_strategy

        self._plan_counter += 1
        plan_id = f"EXIT-{self._plan_counter:06d}"

        # 生成切片
        num_slices = cost_estimate.recommended_slices
        slice_size = position_size / num_slices

        slices = []
        for i in range(num_slices):
            slices.append({
                "slice_id": i + 1,
                "quantity": slice_size,
                "status": "pending",
                "scheduled_time": None,
                "executed_time": None,
                "executed_price": None,
            })

        plan = ExitPlan(
            plan_id=plan_id,
            symbol=symbol,
            created_at=datetime.now(),
            position_size=position_size,
            strategy=strategy,
            slices=slices,
            total_slices=num_slices,
        )

        self._exit_plans[plan_id] = plan
        return plan

    def get_exit_plan(self, plan_id: str) -> Optional[ExitPlan]:
        """获取退出计划"""
        return self._exit_plans.get(plan_id)

    def get_all_exit_plans(self, status: Optional[str] = None) -> List[ExitPlan]:
        """获取所有退出计划"""
        plans = list(self._exit_plans.values())
        if status:
            plans = [p for p in plans if p.status == status]
        return plans

    def cancel_exit_plan(self, plan_id: str) -> bool:
        """取消退出计划"""
        plan = self._exit_plans.get(plan_id)
        if plan and plan.status in ["pending", "executing"]:
            plan.status = "cancelled"
            return True
        return False

    # ============ 压力测试 ============

    async def run_stress_test(
        self,
        symbol: str,
        scenario_id: str,
    ) -> LiquidityStressResult:
        """运行流动性压力测试"""
        scenario = self._stress_scenarios.get(scenario_id)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_id}")

        # 获取当前指标
        metrics = await self.calculate_metrics(symbol)
        if not metrics:
            raise ValueError(f"Cannot get metrics for {symbol}")

        # 应用压力因子
        stressed_spread_bps = metrics.spread_bps * scenario.spread_multiplier
        stressed_depth = metrics.total_depth * (1 - scenario.depth_reduction)
        stressed_impact = metrics.estimated_impact_5pct * scenario.impact_multiplier

        # 计算压力后清算成本
        position = await self._position_provider.get_position(symbol)
        position_size = position.get("quantity", 0) if position else 0

        if position_size > 0:
            normal_cost = await self.estimate_liquidation_cost(symbol, position_size)
            stressed_cost = normal_cost.total_cost_bps * scenario.impact_multiplier
        else:
            stressed_cost = 0

        # 计算额外成本
        additional_cost_bps = stressed_spread_bps - metrics.spread_bps + (stressed_impact - metrics.estimated_impact_5pct)

        # 确定风险等级
        if stressed_spread_bps > 500 or stressed_depth < metrics.total_depth * 0.1:
            risk_level = "severe"
        elif stressed_spread_bps > 200 or stressed_depth < metrics.total_depth * 0.3:
            risk_level = "high"
        elif stressed_spread_bps > 100 or stressed_depth < metrics.total_depth * 0.5:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return LiquidityStressResult(
            scenario=scenario,
            timestamp=datetime.now(),
            stressed_spread_bps=stressed_spread_bps,
            stressed_depth=stressed_depth,
            stressed_impact=stressed_impact,
            stressed_liquidation_cost=stressed_cost,
            additional_cost_bps=additional_cost_bps,
            liquidation_time_increase=scenario.impact_multiplier,
            risk_level=risk_level,
        )

    async def run_all_stress_tests(self, symbol: str) -> List[LiquidityStressResult]:
        """运行所有压力测试情景"""
        results = []
        for scenario_id in self._stress_scenarios:
            try:
                result = await self.run_stress_test(symbol, scenario_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Stress test {scenario_id} failed for {symbol}: {e}")
        return results

    def get_stress_scenarios(self) -> Dict[str, LiquidityStressScenario]:
        """获取所有压力情景"""
        return self._stress_scenarios.copy()

    def add_stress_scenario(self, scenario: LiquidityStressScenario) -> None:
        """添加自定义压力情景"""
        self._stress_scenarios[scenario.scenario_id] = scenario

    # ============ LCR计算 ============

    async def calculate_lcr(
        self,
        hqla_level1: float,
        hqla_level2a: float,
        hqla_level2b: float,
        cash_outflows: float,
        cash_inflows: float,
    ) -> LiquidityCoverageRatio:
        """计算流动性覆盖率"""
        # 计算HQLA总额 (二级资产有折扣)
        total_hqla = hqla_level1 + hqla_level2a * 0.85 + hqla_level2b * 0.50

        # 计算净现金流出
        # 现金流入最多抵扣75%的流出
        max_inflows = cash_outflows * 0.75
        adjusted_inflows = min(cash_inflows, max_inflows)
        net_cash_outflows = cash_outflows - adjusted_inflows

        # 计算LCR
        lcr = (total_hqla / net_cash_outflows * 100) if net_cash_outflows > 0 else float('inf')

        return LiquidityCoverageRatio(
            timestamp=datetime.now(),
            hqla_level1=hqla_level1,
            hqla_level2a=hqla_level2a,
            hqla_level2b=hqla_level2b,
            total_hqla=total_hqla,
            cash_outflows=cash_outflows,
            cash_inflows=cash_inflows,
            net_cash_outflows=net_cash_outflows,
            lcr=lcr,
            lcr_requirement=100.0,
            is_compliant=lcr >= 100.0,
        )

    # ============ 数据查询 ============

    def get_latest_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """获取最新指标"""
        history = self._metrics_history.get(symbol)
        return history[-1] if history else None

    def get_metrics_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[LiquidityMetrics]:
        """获取指标历史"""
        history = self._metrics_history.get(symbol, [])
        return list(history)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "watched_symbols": len(self._watched_symbols),
            "total_alerts": len(self._alerts),
            "unacknowledged_alerts": len([a for a in self._alerts if not a.acknowledged]),
            "active_exit_plans": len([p for p in self._exit_plans.values() if p.status in ["pending", "executing"]]),
            "metrics_history_sizes": {
                symbol: len(history) for symbol, history in self._metrics_history.items()
            },
        }


# ============ Factory Functions ============

def create_liquidity_monitor(
    config: Optional[LiquidityMonitorConfig] = None,
    order_book_provider: Optional[OrderBookProvider] = None,
    volume_provider: Optional[VolumeDataProvider] = None,
    position_provider: Optional[PositionDataProvider] = None,
    on_alert: Optional[Callable[[LiquidityAlert], None]] = None,
    on_metrics_update: Optional[Callable[[str, LiquidityMetrics], None]] = None,
) -> LiquidityRiskMonitor:
    """创建流动性风险监控器"""
    return LiquidityRiskMonitor(
        config=config,
        order_book_provider=order_book_provider,
        volume_provider=volume_provider,
        position_provider=position_provider,
        on_alert=on_alert,
        on_metrics_update=on_metrics_update,
    )


def create_order_book(
    symbol: str,
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
) -> OrderBook:
    """快速创建订单簿"""
    return OrderBook(
        symbol=symbol,
        timestamp=datetime.now(),
        bids=[OrderBookLevel(price=p, quantity=q) for p, q in bids],
        asks=[OrderBookLevel(price=p, quantity=q) for p, q in asks],
    )


async def quick_liquidity_check(
    symbol: str,
    order_book_provider: Optional[OrderBookProvider] = None,
    volume_provider: Optional[VolumeDataProvider] = None,
) -> Optional[LiquidityMetrics]:
    """快速流动性检查"""
    monitor = LiquidityRiskMonitor(
        order_book_provider=order_book_provider,
        volume_provider=volume_provider,
    )
    return await monitor.calculate_metrics(symbol)


# ============ Exports ============

__all__ = [
    # Enums
    "LiquidityLevel",
    "AlertSeverity",
    "MonitorState",
    "ExitStrategy",
    "SpreadType",
    # Data Classes
    "OrderBookLevel",
    "OrderBook",
    "LiquidityMetrics",
    "LiquidityAlert",
    "LiquidationCostEstimate",
    "ExitPlan",
    "LiquidityCoverageRatio",
    "LiquidityStressScenario",
    "LiquidityStressResult",
    "LiquidityMonitorConfig",
    # Providers
    "OrderBookProvider",
    "VolumeDataProvider",
    "PositionDataProvider",
    "SimulatedOrderBookProvider",
    "SimulatedVolumeDataProvider",
    "SimulatedPositionDataProvider",
    # Main Class
    "LiquidityRiskMonitor",
    # Factory Functions
    "create_liquidity_monitor",
    "create_order_book",
    "quick_liquidity_check",
]
