"""
基准追踪器 (Benchmark Tracker)

机构级指数追踪和组合管理系统，提供:
- 基准指数管理 (Benchmark Index Management)
- 跟踪误差计算 (Tracking Error Calculation)
- 组合偏离分析 (Portfolio Deviation Analysis)
- 再平衡信号生成 (Rebalancing Signal Generation)
- 成分股管理 (Constituent Management)
- 历史回测追踪 (Historical Tracking)
- 实时偏离监控 (Real-time Deviation Monitoring)

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

class BenchmarkType(Enum):
    """基准类型"""
    INDEX = "index"                 # 市场指数
    ETF = "etf"                     # ETF
    CUSTOM = "custom"               # 自定义组合
    SECTOR = "sector"               # 行业基准
    FACTOR = "factor"               # 因子基准
    BLENDED = "blended"             # 混合基准


class WeightingMethod(Enum):
    """权重方法"""
    MARKET_CAP = "market_cap"       # 市值加权
    EQUAL_WEIGHT = "equal_weight"   # 等权重
    PRICE_WEIGHT = "price_weight"   # 价格加权
    FUNDAMENTAL = "fundamental"     # 基本面加权
    CUSTOM = "custom"               # 自定义权重


class RebalanceFrequency(Enum):
    """再平衡频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    ON_DEMAND = "on_demand"


class DeviationType(Enum):
    """偏离类型"""
    UNDERWEIGHT = "underweight"     # 低配
    OVERWEIGHT = "overweight"       # 超配
    MISSING = "missing"             # 缺失
    EXTRA = "extra"                 # 多余
    WITHIN_TOLERANCE = "within_tolerance"  # 容差内


class TrackerState(Enum):
    """追踪器状态"""
    IDLE = "idle"
    TRACKING = "tracking"
    REBALANCING = "rebalancing"
    PAUSED = "paused"
    STOPPED = "stopped"


class AlertType(Enum):
    """告警类型"""
    TRACKING_ERROR = "tracking_error"
    WEIGHT_DEVIATION = "weight_deviation"
    MISSING_CONSTITUENT = "missing_constituent"
    REBALANCE_NEEDED = "rebalance_needed"
    DATA_STALE = "data_stale"


# ==================== 数据类 ====================

@dataclass
class BenchmarkConstituent:
    """基准成分股"""
    symbol: str
    name: str
    weight: float               # 目标权重 (0-1)

    # 市场数据
    price: float = 0.0
    market_cap: float = 0.0
    shares_outstanding: float = 0.0

    # 分类信息
    sector: str = ""
    industry: str = ""
    country: str = ""

    # 元数据
    added_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "weight": self.weight,
            "price": self.price,
            "market_cap": self.market_cap,
            "shares_outstanding": self.shares_outstanding,
            "sector": self.sector,
            "industry": self.industry,
            "country": self.country,
            "added_date": self.added_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class Benchmark:
    """基准定义"""
    benchmark_id: str
    name: str
    benchmark_type: BenchmarkType

    # 成分股
    constituents: List[BenchmarkConstituent] = field(default_factory=list)

    # 权重方法
    weighting_method: WeightingMethod = WeightingMethod.MARKET_CAP

    # 再平衡设置
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.QUARTERLY
    last_rebalance: Optional[datetime] = None
    next_rebalance: Optional[datetime] = None

    # 基准值
    base_value: float = 1000.0
    current_value: float = 1000.0

    # 历史数据
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)

    # 元数据
    description: str = ""
    currency: str = "USD"
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_weight(self) -> float:
        """总权重"""
        return sum(c.weight for c in self.constituents)

    @property
    def constituent_count(self) -> int:
        """成分股数量"""
        return len(self.constituents)

    def get_constituent(self, symbol: str) -> Optional[BenchmarkConstituent]:
        """获取成分股"""
        for c in self.constituents:
            if c.symbol == symbol:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "benchmark_type": self.benchmark_type.value,
            "constituents": [c.to_dict() for c in self.constituents],
            "weighting_method": self.weighting_method.value,
            "rebalance_frequency": self.rebalance_frequency.value,
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "next_rebalance": self.next_rebalance.isoformat() if self.next_rebalance else None,
            "base_value": self.base_value,
            "current_value": self.current_value,
            "total_weight": self.total_weight,
            "constituent_count": self.constituent_count,
            "description": self.description,
            "currency": self.currency,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class PortfolioHolding:
    """组合持仓"""
    symbol: str
    quantity: float
    market_value: float
    weight: float               # 实际权重 (0-1)

    # 成本信息
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0

    # 时间戳
    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "market_value": self.market_value,
            "weight": self.weight,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "as_of": self.as_of.isoformat(),
        }


@dataclass
class WeightDeviation:
    """权重偏离"""
    symbol: str
    benchmark_weight: float     # 基准权重
    portfolio_weight: float     # 组合权重
    deviation: float            # 偏离 = 组合 - 基准
    deviation_pct: float        # 偏离百分比
    deviation_type: DeviationType

    # 交易建议
    suggested_trade_value: float = 0.0  # 建议交易金额
    suggested_trade_shares: float = 0.0  # 建议交易股数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "benchmark_weight": self.benchmark_weight,
            "portfolio_weight": self.portfolio_weight,
            "deviation": self.deviation,
            "deviation_pct": self.deviation_pct,
            "deviation_type": self.deviation_type.value,
            "suggested_trade_value": self.suggested_trade_value,
            "suggested_trade_shares": self.suggested_trade_shares,
        }


@dataclass
class TrackingMetrics:
    """跟踪指标"""
    # 跟踪误差
    tracking_error: float = 0.0         # 年化跟踪误差
    tracking_error_daily: float = 0.0   # 日跟踪误差

    # 相关性
    correlation: float = 0.0            # 与基准相关性
    beta: float = 1.0                   # Beta

    # 收益差异
    excess_return: float = 0.0          # 超额收益
    information_ratio: float = 0.0      # 信息比率

    # 偏离统计
    max_deviation: float = 0.0          # 最大单一偏离
    avg_deviation: float = 0.0          # 平均偏离
    total_active_share: float = 0.0     # 主动份额

    # 计算日期
    calculated_at: datetime = field(default_factory=datetime.now)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracking_error": self.tracking_error,
            "tracking_error_daily": self.tracking_error_daily,
            "correlation": self.correlation,
            "beta": self.beta,
            "excess_return": self.excess_return,
            "information_ratio": self.information_ratio,
            "max_deviation": self.max_deviation,
            "avg_deviation": self.avg_deviation,
            "total_active_share": self.total_active_share,
            "calculated_at": self.calculated_at.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }


@dataclass
class RebalanceOrder:
    """再平衡订单"""
    order_id: str
    symbol: str
    side: str                   # buy/sell
    quantity: float
    target_weight: float
    current_weight: float

    # 订单属性
    order_type: str = "market"  # market/limit
    limit_price: Optional[float] = None
    urgency: str = "normal"     # low/normal/high

    # 状态
    status: str = "pending"     # pending/submitted/filled/cancelled
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "urgency": self.urgency,
            "status": self.status,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }


@dataclass
class RebalanceResult:
    """再平衡结果"""
    rebalance_id: str
    benchmark_id: str

    # 再平衡订单
    orders: List[RebalanceOrder] = field(default_factory=list)

    # 统计
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    total_turnover: float = 0.0
    estimated_cost: float = 0.0

    # 预期效果
    pre_tracking_error: float = 0.0
    post_tracking_error: float = 0.0
    tracking_error_reduction: float = 0.0

    # 状态
    status: str = "generated"   # generated/executing/completed/cancelled

    # 时间戳
    generated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rebalance_id": self.rebalance_id,
            "benchmark_id": self.benchmark_id,
            "orders": [o.to_dict() for o in self.orders],
            "total_buy_value": self.total_buy_value,
            "total_sell_value": self.total_sell_value,
            "total_turnover": self.total_turnover,
            "estimated_cost": self.estimated_cost,
            "pre_tracking_error": self.pre_tracking_error,
            "post_tracking_error": self.post_tracking_error,
            "tracking_error_reduction": self.tracking_error_reduction,
            "status": self.status,
            "generated_at": self.generated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class TrackerAlert:
    """追踪器告警"""
    alert_id: str
    alert_type: AlertType
    benchmark_id: str

    # 告警详情
    message: str
    severity: str = "warning"   # info/warning/critical
    details: Dict[str, Any] = field(default_factory=dict)

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
            "benchmark_id": self.benchmark_id,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TrackerConfig:
    """追踪器配置"""
    # 容差设置
    weight_tolerance: float = 0.01          # 权重容差 (1%)
    tracking_error_threshold: float = 0.02  # 跟踪误差阈值 (2%)

    # 再平衡设置
    min_trade_value: float = 1000.0         # 最小交易金额
    max_turnover: float = 0.5               # 最大换手率

    # 交易成本
    commission_rate: float = 0.001          # 佣金率
    market_impact: float = 0.001            # 市场冲击

    # 监控设置
    update_interval_seconds: float = 60.0   # 更新间隔
    alert_on_deviation: bool = True
    alert_on_rebalance: bool = True

    # 历史数据
    max_history_days: int = 365

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_tolerance": self.weight_tolerance,
            "tracking_error_threshold": self.tracking_error_threshold,
            "min_trade_value": self.min_trade_value,
            "max_turnover": self.max_turnover,
            "commission_rate": self.commission_rate,
            "market_impact": self.market_impact,
            "update_interval_seconds": self.update_interval_seconds,
            "alert_on_deviation": self.alert_on_deviation,
            "alert_on_rebalance": self.alert_on_rebalance,
            "max_history_days": self.max_history_days,
        }


# ==================== 接口定义 ====================

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

    @abstractmethod
    def get_historical_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, float]]:
        """获取历史价格"""
        pass


class PortfolioProvider(ABC):
    """组合数据提供者接口"""

    @abstractmethod
    def get_holdings(self, portfolio_id: str) -> List[PortfolioHolding]:
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


# ==================== 模拟实现 ====================

class SimulatedPriceProvider(PriceProvider):
    """模拟价格提供者"""

    def __init__(self):
        self.prices: Dict[str, float] = {}
        self.historical: Dict[str, List[Tuple[datetime, float]]] = {}

    def set_price(self, symbol: str, price: float):
        self.prices[symbol] = price

    def set_historical(self, symbol: str, data: List[Tuple[datetime, float]]):
        self.historical[symbol] = data

    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 100.0)

    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        return {s: self.get_price(s) for s in symbols}

    def get_historical_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, float]]:
        if symbol in self.historical:
            return [
                (dt, p) for dt, p in self.historical[symbol]
                if start_date <= dt <= end_date
            ]
        # 生成模拟数据
        result = []
        current = start_date
        price = self.get_price(symbol)
        while current <= end_date:
            result.append((current, price * (1 + (hash(str(current)) % 100 - 50) / 1000)))
            current += timedelta(days=1)
        return result


class SimulatedPortfolioProvider(PortfolioProvider):
    """模拟组合提供者"""

    def __init__(self):
        self.holdings: Dict[str, List[PortfolioHolding]] = {}
        self.total_values: Dict[str, float] = {}
        self.cash: Dict[str, float] = {}

    def set_holdings(self, portfolio_id: str, holdings: List[PortfolioHolding]):
        self.holdings[portfolio_id] = holdings

    def set_total_value(self, portfolio_id: str, value: float):
        self.total_values[portfolio_id] = value

    def set_cash(self, portfolio_id: str, cash: float):
        self.cash[portfolio_id] = cash

    def get_holdings(self, portfolio_id: str) -> List[PortfolioHolding]:
        return self.holdings.get(portfolio_id, [])

    def get_total_value(self, portfolio_id: str) -> float:
        return self.total_values.get(portfolio_id, 1000000.0)

    def get_cash(self, portfolio_id: str) -> float:
        return self.cash.get(portfolio_id, 0.0)


# ==================== 核心引擎 ====================

class BenchmarkTracker:
    """基准追踪器"""

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        price_provider: Optional[PriceProvider] = None,
        portfolio_provider: Optional[PortfolioProvider] = None,
    ):
        self.config = config or TrackerConfig()
        self.price_provider = price_provider or simulated_default(SimulatedPriceProvider, "data from BenchmarkTracker's simulated price_provider (SimulatedPriceProvider)")
        self.portfolio_provider = portfolio_provider or simulated_default(SimulatedPortfolioProvider, "data from BenchmarkTracker's simulated portfolio_provider (SimulatedPortfolioProvider)")

        # 基准存储
        self.benchmarks: Dict[str, Benchmark] = {}

        # 组合-基准映射
        self.portfolio_benchmark_map: Dict[str, str] = {}

        # 告警存储
        self.alerts: List[TrackerAlert] = []

        # 再平衡历史
        self.rebalance_history: List[RebalanceResult] = []

        # 状态
        self.state = TrackerState.IDLE
        self._tracking_task: Optional[asyncio.Task] = None

        # 回调
        self._on_deviation: Optional[Callable[[str, List[WeightDeviation]], None]] = None
        self._on_alert: Optional[Callable[[TrackerAlert], None]] = None
        self._on_rebalance: Optional[Callable[[RebalanceResult], None]] = None

        # 统计
        self.stats = {
            "benchmarks_tracked": 0,
            "total_deviations_detected": 0,
            "rebalances_generated": 0,
            "alerts_created": 0,
        }

    # ==================== 基准管理 ====================

    def add_benchmark(self, benchmark: Benchmark) -> bool:
        """添加基准"""
        if benchmark.benchmark_id in self.benchmarks:
            return False
        self.benchmarks[benchmark.benchmark_id] = benchmark
        self.stats["benchmarks_tracked"] = len(self.benchmarks)
        return True

    def update_benchmark(self, benchmark_id: str, updates: Dict[str, Any]) -> bool:
        """更新基准"""
        if benchmark_id not in self.benchmarks:
            return False

        benchmark = self.benchmarks[benchmark_id]
        for key, value in updates.items():
            if hasattr(benchmark, key):
                setattr(benchmark, key, value)
        return True

    def remove_benchmark(self, benchmark_id: str) -> bool:
        """移除基准"""
        if benchmark_id not in self.benchmarks:
            return False
        del self.benchmarks[benchmark_id]
        self.stats["benchmarks_tracked"] = len(self.benchmarks)
        return True

    def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """获取基准"""
        return self.benchmarks.get(benchmark_id)

    def get_all_benchmarks(self) -> List[Benchmark]:
        """获取所有基准"""
        return list(self.benchmarks.values())

    # ==================== 成分股管理 ====================

    def add_constituent(
        self,
        benchmark_id: str,
        constituent: BenchmarkConstituent
    ) -> bool:
        """添加成分股"""
        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return False

        # 检查是否已存在
        if benchmark.get_constituent(constituent.symbol):
            return False

        benchmark.constituents.append(constituent)
        return True

    def update_constituent(
        self,
        benchmark_id: str,
        symbol: str,
        updates: Dict[str, Any]
    ) -> bool:
        """更新成分股"""
        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return False

        constituent = benchmark.get_constituent(symbol)
        if not constituent:
            return False

        for key, value in updates.items():
            if hasattr(constituent, key):
                setattr(constituent, key, value)
        constituent.last_updated = datetime.now()
        return True

    def remove_constituent(self, benchmark_id: str, symbol: str) -> bool:
        """移除成分股"""
        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return False

        benchmark.constituents = [
            c for c in benchmark.constituents if c.symbol != symbol
        ]
        return True

    def update_constituent_prices(self, benchmark_id: str) -> bool:
        """更新成分股价格"""
        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return False

        symbols = [c.symbol for c in benchmark.constituents]
        prices = self.price_provider.get_prices(symbols)

        for constituent in benchmark.constituents:
            if constituent.symbol in prices:
                constituent.price = prices[constituent.symbol]
                constituent.last_updated = datetime.now()

        # 更新基准值
        self._update_benchmark_value(benchmark)
        return True

    def _update_benchmark_value(self, benchmark: Benchmark):
        """更新基准值"""
        if not benchmark.constituents:
            return

        # 计算加权价格变化
        total_value = 0.0
        for constituent in benchmark.constituents:
            total_value += constituent.weight * constituent.price

        # 更新历史
        now = datetime.now()
        benchmark.historical_values.append((now, total_value))

        # 限制历史大小
        max_points = self.config.max_history_days * 24  # 按小时
        if len(benchmark.historical_values) > max_points:
            benchmark.historical_values = benchmark.historical_values[-max_points:]

        benchmark.current_value = total_value

    # ==================== 组合-基准映射 ====================

    def assign_benchmark(self, portfolio_id: str, benchmark_id: str) -> bool:
        """分配基准给组合"""
        if benchmark_id not in self.benchmarks:
            return False
        self.portfolio_benchmark_map[portfolio_id] = benchmark_id
        return True

    def unassign_benchmark(self, portfolio_id: str) -> bool:
        """取消组合基准分配"""
        if portfolio_id in self.portfolio_benchmark_map:
            del self.portfolio_benchmark_map[portfolio_id]
            return True
        return False

    def get_portfolio_benchmark(self, portfolio_id: str) -> Optional[str]:
        """获取组合的基准"""
        return self.portfolio_benchmark_map.get(portfolio_id)

    # ==================== 偏离分析 ====================

    def analyze_deviations(
        self,
        portfolio_id: str,
        benchmark_id: Optional[str] = None
    ) -> List[WeightDeviation]:
        """分析组合与基准的偏离"""
        # 确定基准
        if not benchmark_id:
            benchmark_id = self.portfolio_benchmark_map.get(portfolio_id)
        if not benchmark_id:
            return []

        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return []

        # 获取组合持仓
        holdings = self.portfolio_provider.get_holdings(portfolio_id)
        total_value = self.portfolio_provider.get_total_value(portfolio_id)

        if total_value <= 0:
            return []

        # 构建组合权重映射
        portfolio_weights: Dict[str, float] = {}
        for holding in holdings:
            portfolio_weights[holding.symbol] = holding.market_value / total_value

        # 计算偏离
        deviations = []
        all_symbols = set(c.symbol for c in benchmark.constituents) | set(portfolio_weights.keys())

        for symbol in all_symbols:
            benchmark_weight = 0.0
            constituent = benchmark.get_constituent(symbol)
            if constituent:
                benchmark_weight = constituent.weight

            portfolio_weight = portfolio_weights.get(symbol, 0.0)
            deviation = portfolio_weight - benchmark_weight

            # 确定偏离类型
            if benchmark_weight == 0 and portfolio_weight > 0:
                deviation_type = DeviationType.EXTRA
            elif benchmark_weight > 0 and portfolio_weight == 0:
                deviation_type = DeviationType.MISSING
            elif abs(deviation) <= self.config.weight_tolerance:
                deviation_type = DeviationType.WITHIN_TOLERANCE
            elif deviation > 0:
                deviation_type = DeviationType.OVERWEIGHT
            else:
                deviation_type = DeviationType.UNDERWEIGHT

            # 计算偏离百分比
            deviation_pct = (deviation / benchmark_weight * 100) if benchmark_weight > 0 else (
                100.0 if portfolio_weight > 0 else 0.0
            )

            # 计算建议交易
            suggested_trade_value = -deviation * total_value
            price = self.price_provider.get_price(symbol)
            suggested_trade_shares = suggested_trade_value / price if price > 0 else 0

            deviations.append(WeightDeviation(
                symbol=symbol,
                benchmark_weight=benchmark_weight,
                portfolio_weight=portfolio_weight,
                deviation=deviation,
                deviation_pct=deviation_pct,
                deviation_type=deviation_type,
                suggested_trade_value=suggested_trade_value,
                suggested_trade_shares=suggested_trade_shares,
            ))

        # 排序：偏离最大的在前
        deviations.sort(key=lambda d: abs(d.deviation), reverse=True)

        # 统计
        self.stats["total_deviations_detected"] += len([
            d for d in deviations if d.deviation_type != DeviationType.WITHIN_TOLERANCE
        ])

        # 触发回调
        if self._on_deviation:
            self._on_deviation(portfolio_id, deviations)

        return deviations

    # ==================== 跟踪指标计算 ====================

    def calculate_tracking_metrics(
        self,
        portfolio_id: str,
        benchmark_id: Optional[str] = None,
        period_days: int = 252
    ) -> TrackingMetrics:
        """计算跟踪指标"""
        if not benchmark_id:
            benchmark_id = self.portfolio_benchmark_map.get(portfolio_id)
        if not benchmark_id:
            return TrackingMetrics()

        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return TrackingMetrics()

        # 分析当前偏离
        deviations = self.analyze_deviations(portfolio_id, benchmark_id)

        # 计算主动份额 (Active Share)
        active_share = sum(abs(d.deviation) for d in deviations) / 2

        # 计算最大和平均偏离
        abs_deviations = [abs(d.deviation) for d in deviations]
        max_deviation = max(abs_deviations) if abs_deviations else 0.0
        avg_deviation = sum(abs_deviations) / len(abs_deviations) if abs_deviations else 0.0

        # 获取历史数据计算跟踪误差
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        # 简化的跟踪误差计算（实际应使用日收益率）
        tracking_error_daily = avg_deviation * 0.1  # 简化估算
        tracking_error_annual = tracking_error_daily * math.sqrt(252)

        # 简化的信息比率
        excess_return = 0.0  # 需要实际收益数据
        information_ratio = excess_return / tracking_error_annual if tracking_error_annual > 0 else 0.0

        return TrackingMetrics(
            tracking_error=tracking_error_annual,
            tracking_error_daily=tracking_error_daily,
            correlation=1 - active_share,  # 简化估算
            beta=1.0,
            excess_return=excess_return,
            information_ratio=information_ratio,
            max_deviation=max_deviation,
            avg_deviation=avg_deviation,
            total_active_share=active_share,
            period_start=start_date,
            period_end=end_date,
        )

    # ==================== 再平衡 ====================

    def generate_rebalance(
        self,
        portfolio_id: str,
        benchmark_id: Optional[str] = None,
        max_turnover: Optional[float] = None
    ) -> RebalanceResult:
        """生成再平衡方案"""
        if not benchmark_id:
            benchmark_id = self.portfolio_benchmark_map.get(portfolio_id)
        if not benchmark_id:
            raise ValueError(f"No benchmark assigned for portfolio {portfolio_id}")

        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            raise ValueError(f"Benchmark not found: {benchmark_id}")

        max_turnover = max_turnover or self.config.max_turnover

        # 分析偏离
        deviations = self.analyze_deviations(portfolio_id, benchmark_id)
        total_value = self.portfolio_provider.get_total_value(portfolio_id)

        # 计算当前跟踪指标
        pre_metrics = self.calculate_tracking_metrics(portfolio_id, benchmark_id)

        # 生成订单
        orders = []
        total_buy = 0.0
        total_sell = 0.0

        for deviation in deviations:
            # 跳过容差内的偏离
            if deviation.deviation_type == DeviationType.WITHIN_TOLERANCE:
                continue

            trade_value = abs(deviation.suggested_trade_value)

            # 检查最小交易金额
            if trade_value < self.config.min_trade_value:
                continue

            side = "sell" if deviation.deviation > 0 else "buy"
            quantity = abs(deviation.suggested_trade_shares)

            order = RebalanceOrder(
                order_id=str(uuid.uuid4()),
                symbol=deviation.symbol,
                side=side,
                quantity=quantity,
                target_weight=deviation.benchmark_weight,
                current_weight=deviation.portfolio_weight,
            )
            orders.append(order)

            if side == "buy":
                total_buy += trade_value
            else:
                total_sell += trade_value

        # 检查换手率限制
        total_turnover = (total_buy + total_sell) / total_value if total_value > 0 else 0
        if total_turnover > max_turnover:
            # 按偏离大小排序，只保留部分订单
            scale_factor = max_turnover / total_turnover
            for order in orders:
                order.quantity *= scale_factor
            total_buy *= scale_factor
            total_sell *= scale_factor
            total_turnover = max_turnover

        # 估算交易成本
        estimated_cost = (total_buy + total_sell) * (
            self.config.commission_rate + self.config.market_impact
        )

        result = RebalanceResult(
            rebalance_id=str(uuid.uuid4()),
            benchmark_id=benchmark_id,
            orders=orders,
            total_buy_value=total_buy,
            total_sell_value=total_sell,
            total_turnover=total_turnover,
            estimated_cost=estimated_cost,
            pre_tracking_error=pre_metrics.tracking_error,
            post_tracking_error=pre_metrics.tracking_error * (1 - total_turnover),  # 简化估算
            tracking_error_reduction=pre_metrics.tracking_error * total_turnover,
        )

        # 记录历史
        self.rebalance_history.append(result)
        self.stats["rebalances_generated"] += 1

        # 触发回调
        if self._on_rebalance:
            self._on_rebalance(result)

        # 更新基准再平衡时间
        benchmark.last_rebalance = datetime.now()
        self._schedule_next_rebalance(benchmark)

        return result

    def _schedule_next_rebalance(self, benchmark: Benchmark):
        """调度下次再平衡"""
        now = datetime.now()
        freq = benchmark.rebalance_frequency

        if freq == RebalanceFrequency.DAILY:
            benchmark.next_rebalance = now + timedelta(days=1)
        elif freq == RebalanceFrequency.WEEKLY:
            benchmark.next_rebalance = now + timedelta(weeks=1)
        elif freq == RebalanceFrequency.MONTHLY:
            benchmark.next_rebalance = now + timedelta(days=30)
        elif freq == RebalanceFrequency.QUARTERLY:
            benchmark.next_rebalance = now + timedelta(days=90)
        elif freq == RebalanceFrequency.SEMI_ANNUAL:
            benchmark.next_rebalance = now + timedelta(days=180)
        elif freq == RebalanceFrequency.ANNUAL:
            benchmark.next_rebalance = now + timedelta(days=365)

    def get_rebalance_history(
        self,
        benchmark_id: Optional[str] = None,
        limit: int = 100
    ) -> List[RebalanceResult]:
        """获取再平衡历史"""
        results = self.rebalance_history
        if benchmark_id:
            results = [r for r in results if r.benchmark_id == benchmark_id]
        return results[-limit:]

    # ==================== 告警管理 ====================

    def _create_alert(
        self,
        alert_type: AlertType,
        benchmark_id: str,
        message: str,
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None
    ) -> TrackerAlert:
        """创建告警"""
        alert = TrackerAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            benchmark_id=benchmark_id,
            message=message,
            severity=severity,
            details=details or {},
        )
        self.alerts.append(alert)
        self.stats["alerts_created"] += 1

        if self._on_alert:
            self._on_alert(alert)

        return alert

    def get_alerts(
        self,
        benchmark_id: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100
    ) -> List[TrackerAlert]:
        """获取告警"""
        alerts = self.alerts

        if benchmark_id:
            alerts = [a for a in alerts if a.benchmark_id == benchmark_id]

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

    def check_and_alert(self, portfolio_id: str):
        """检查并生成告警"""
        benchmark_id = self.portfolio_benchmark_map.get(portfolio_id)
        if not benchmark_id:
            return

        benchmark = self.benchmarks.get(benchmark_id)
        if not benchmark:
            return

        # 计算跟踪指标
        metrics = self.calculate_tracking_metrics(portfolio_id, benchmark_id)

        # 检查跟踪误差
        if metrics.tracking_error > self.config.tracking_error_threshold:
            if self.config.alert_on_deviation:
                self._create_alert(
                    AlertType.TRACKING_ERROR,
                    benchmark_id,
                    f"Tracking error ({metrics.tracking_error:.2%}) exceeds threshold ({self.config.tracking_error_threshold:.2%})",
                    severity="warning",
                    details={"tracking_error": metrics.tracking_error},
                )

        # 检查权重偏离
        deviations = self.analyze_deviations(portfolio_id, benchmark_id)
        significant_deviations = [
            d for d in deviations
            if d.deviation_type not in [DeviationType.WITHIN_TOLERANCE]
        ]

        if significant_deviations and self.config.alert_on_deviation:
            max_dev = max(significant_deviations, key=lambda d: abs(d.deviation))
            self._create_alert(
                AlertType.WEIGHT_DEVIATION,
                benchmark_id,
                f"Significant weight deviation detected: {max_dev.symbol} ({max_dev.deviation:.2%})",
                severity="warning",
                details={"symbol": max_dev.symbol, "deviation": max_dev.deviation},
            )

        # 检查是否需要再平衡
        if benchmark.next_rebalance and datetime.now() >= benchmark.next_rebalance:
            if self.config.alert_on_rebalance:
                self._create_alert(
                    AlertType.REBALANCE_NEEDED,
                    benchmark_id,
                    f"Scheduled rebalance is due for benchmark {benchmark.name}",
                    severity="info",
                )

    # ==================== 监控控制 ====================

    def on_deviation(self, callback: Callable[[str, List[WeightDeviation]], None]):
        """注册偏离回调"""
        self._on_deviation = callback

    def on_alert(self, callback: Callable[[TrackerAlert], None]):
        """注册告警回调"""
        self._on_alert = callback

    def on_rebalance(self, callback: Callable[[RebalanceResult], None]):
        """注册再平衡回调"""
        self._on_rebalance = callback

    async def start_tracking(self):
        """启动追踪"""
        if self.state == TrackerState.TRACKING:
            return

        self.state = TrackerState.TRACKING
        self._tracking_task = asyncio.create_task(self._tracking_loop())

    async def stop_tracking(self):
        """停止追踪"""
        self.state = TrackerState.STOPPED
        if self._tracking_task:
            self._tracking_task.cancel()
            try:
                await self._tracking_task
            except asyncio.CancelledError:
                pass
            self._tracking_task = None

    async def _tracking_loop(self):
        """追踪循环"""
        while self.state == TrackerState.TRACKING:
            try:
                # 更新所有基准价格
                for benchmark_id in self.benchmarks:
                    self.update_constituent_prices(benchmark_id)

                # 检查所有组合
                for portfolio_id in self.portfolio_benchmark_map:
                    self.check_and_alert(portfolio_id)

                await asyncio.sleep(self.config.update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Benchmark tracking loop iteration failed; continuing")
                # Back off before retrying so a failure raised before the sleep
                # above cannot spin this loop hot.
                await asyncio.sleep(self.config.update_interval_seconds)

    # ==================== 统计 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self.state.value,
            "benchmarks_count": len(self.benchmarks),
            "portfolios_tracked": len(self.portfolio_benchmark_map),
            "alerts_count": len(self.alerts),
            "unacknowledged_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "rebalance_history_count": len(self.rebalance_history),
            **self.stats,
        }


# ==================== 工厂函数 ====================

def create_benchmark_tracker(
    config: Optional[TrackerConfig] = None,
    price_provider: Optional[PriceProvider] = None,
    portfolio_provider: Optional[PortfolioProvider] = None,
) -> BenchmarkTracker:
    """创建基准追踪器"""
    return BenchmarkTracker(
        config=config,
        price_provider=price_provider,
        portfolio_provider=portfolio_provider,
    )


def create_benchmark(
    name: str,
    benchmark_type: BenchmarkType = BenchmarkType.INDEX,
    benchmark_id: Optional[str] = None,
    **kwargs
) -> Benchmark:
    """创建基准"""
    return Benchmark(
        benchmark_id=benchmark_id or str(uuid.uuid4()),
        name=name,
        benchmark_type=benchmark_type,
        **kwargs
    )


def create_constituent(
    symbol: str,
    name: str,
    weight: float,
    **kwargs
) -> BenchmarkConstituent:
    """创建成分股"""
    return BenchmarkConstituent(
        symbol=symbol,
        name=name,
        weight=weight,
        **kwargs
    )


def create_portfolio_holding(
    symbol: str,
    quantity: float,
    market_value: float,
    total_portfolio_value: float,
    **kwargs
) -> PortfolioHolding:
    """创建组合持仓"""
    weight = market_value / total_portfolio_value if total_portfolio_value > 0 else 0
    return PortfolioHolding(
        symbol=symbol,
        quantity=quantity,
        market_value=market_value,
        weight=weight,
        **kwargs
    )


def quick_deviation_analysis(
    portfolio_holdings: List[Dict[str, Any]],
    benchmark_weights: Dict[str, float],
    total_portfolio_value: float,
) -> List[WeightDeviation]:
    """快速偏离分析"""
    tracker = create_benchmark_tracker()

    # 创建基准
    benchmark = create_benchmark("QuickBenchmark")
    for symbol, weight in benchmark_weights.items():
        benchmark.constituents.append(create_constituent(symbol, symbol, weight))
    tracker.add_benchmark(benchmark)

    # 设置组合
    holdings = [
        create_portfolio_holding(
            h["symbol"],
            h.get("quantity", 0),
            h.get("market_value", 0),
            total_portfolio_value
        )
        for h in portfolio_holdings
    ]
    tracker.portfolio_provider.set_holdings("quick_portfolio", holdings)
    tracker.portfolio_provider.set_total_value("quick_portfolio", total_portfolio_value)

    # 分配基准
    tracker.assign_benchmark("quick_portfolio", benchmark.benchmark_id)

    # 分析偏离
    return tracker.analyze_deviations("quick_portfolio")


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "BenchmarkType",
    "WeightingMethod",
    "RebalanceFrequency",
    "DeviationType",
    "TrackerState",
    "AlertType",
    # 数据类
    "BenchmarkConstituent",
    "Benchmark",
    "PortfolioHolding",
    "WeightDeviation",
    "TrackingMetrics",
    "RebalanceOrder",
    "RebalanceResult",
    "TrackerAlert",
    "TrackerConfig",
    # 接口
    "PriceProvider",
    "PortfolioProvider",
    # 模拟实现
    "SimulatedPriceProvider",
    "SimulatedPortfolioProvider",
    # 核心引擎
    "BenchmarkTracker",
    # 工厂函数
    "create_benchmark_tracker",
    "create_benchmark",
    "create_constituent",
    "create_portfolio_holding",
    "quick_deviation_analysis",
]
