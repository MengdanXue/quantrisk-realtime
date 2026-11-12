"""
资金费率监控器 (Funding Rate Monitor)

机构级资金费率监控和套利系统，提供:
- 实时资金费率跟踪 (Real-time Funding Rate Tracking)
- 历史费率记录管理 (Historical Rate Management)
- 预测费率计算 (Predicted Rate Calculation)
- 跨交易所费率比较 (Cross-Exchange Rate Comparison)
- 费率趋势分析 (Rate Trend Analysis)
- 费率波动分析 (Rate Volatility Analysis)
- 现货-永续套利机会 (Spot-Perpetual Arbitrage)
- 跨所套利机会 (Cross-Exchange Arbitrage)
- 持仓成本估算 (Position Cost Estimation)
- 资金费率警报系统 (Rate Alert System)

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
import statistics
import logging
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class RateType(Enum):
    """资金费率类型"""
    PERPETUAL = "perpetual"         # 永续合约
    SWAP = "swap"                   # 掉期
    FUTURES = "futures"             # 期货
    OPTION = "option"               # 期权


class PaymentFrequency(Enum):
    """支付频率"""
    HOURLY = "hourly"               # 每小时
    FOUR_HOURLY = "4hourly"         # 每4小时
    EIGHT_HOURLY = "8hourly"        # 每8小时
    DAILY = "daily"                 # 每日
    WEEKLY = "weekly"               # 每周


class ExchangeType(Enum):
    """交易所类型"""
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    BITMEX = "bitmex"
    DERIBIT = "deribit"
    FTX = "ftx"
    HUOBI = "huobi"
    KRAKEN = "kraken"
    COINBASE = "coinbase"
    BITFINEX = "bitfinex"


class AlertLevel(Enum):
    """警报级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """警报类型"""
    HIGH_RATE = "high_rate"                     # 高费率
    LOW_RATE = "low_rate"                       # 低费率
    RATE_REVERSAL = "rate_reversal"             # 费率反转
    ARBITRAGE_OPPORTUNITY = "arbitrage_opp"     # 套利机会
    COST_LIMIT_EXCEEDED = "cost_limit"          # 成本超限
    RATE_SPIKE = "rate_spike"                   # 费率激增
    CROSS_EXCHANGE_DIVERGENCE = "cross_diverge" # 跨所差异


class ArbitrageType(Enum):
    """套利类型"""
    SPOT_PERP = "spot_perpetual"        # 现货-永续
    CROSS_EXCHANGE = "cross_exchange"   # 跨交易所
    CALENDAR_SPREAD = "calendar_spread" # 跨期套利
    TRIANGULAR = "triangular"           # 三角套利


class TrendDirection(Enum):
    """趋势方向"""
    BULLISH = "bullish"         # 看涨
    BEARISH = "bearish"         # 看跌
    NEUTRAL = "neutral"         # 中性
    VOLATILE = "volatile"       # 震荡


class MonitorState(Enum):
    """监控器状态"""
    IDLE = "idle"
    MONITORING = "monitoring"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# ==================== 数据类 ====================

@dataclass
class FundingRateData:
    """资金费率数据"""
    symbol: str
    exchange: ExchangeType
    rate: float                     # 当前费率
    timestamp: datetime
    next_payment_time: datetime
    payment_frequency: PaymentFrequency

    # 预测费率
    predicted_rate: Optional[float] = None

    # 市场数据
    mark_price: float = 0.0
    index_price: float = 0.0
    spot_price: float = 0.0

    # 费率统计
    rate_24h_avg: float = 0.0
    rate_7d_avg: float = 0.0
    rate_30d_avg: float = 0.0

    # 费率变化
    rate_change_1h: float = 0.0
    rate_change_4h: float = 0.0
    rate_change_24h: float = 0.0

    # 波动性
    rate_volatility: float = 0.0

    # 元数据
    data_source: str = "api"
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "rate": self.rate,
            "predicted_rate": self.predicted_rate,
            "timestamp": self.timestamp.isoformat(),
            "next_payment_time": self.next_payment_time.isoformat(),
            "payment_frequency": self.payment_frequency.value,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "spot_price": self.spot_price,
            "rate_24h_avg": self.rate_24h_avg,
            "rate_7d_avg": self.rate_7d_avg,
            "rate_30d_avg": self.rate_30d_avg,
            "rate_change_1h": self.rate_change_1h,
            "rate_change_4h": self.rate_change_4h,
            "rate_change_24h": self.rate_change_24h,
            "rate_volatility": self.rate_volatility,
            "data_source": self.data_source,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class HistoricalRate:
    """历史费率记录"""
    symbol: str
    exchange: ExchangeType
    rate: float
    mark_price: float
    index_price: float
    timestamp: datetime
    payment_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "rate": self.rate,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "timestamp": self.timestamp.isoformat(),
            "payment_time": self.payment_time.isoformat(),
        }


@dataclass
class RateTrend:
    """费率趋势分析"""
    symbol: str
    exchange: ExchangeType

    # 趋势方向
    trend_direction: TrendDirection
    trend_strength: float           # 趋势强度 0-1

    # 统计数据
    current_rate: float
    avg_rate: float
    median_rate: float
    std_dev: float
    min_rate: float
    max_rate: float

    # 移动平均
    ma_7: float = 0.0
    ma_30: float = 0.0
    ma_90: float = 0.0

    # 趋势线
    slope: float = 0.0              # 斜率
    r_squared: float = 0.0          # 拟合度

    # 周期性分析
    is_cyclical: bool = False
    cycle_period_hours: Optional[float] = None

    # 分析时间
    period_start: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=30))
    period_end: datetime = field(default_factory=datetime.now)
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "trend_direction": self.trend_direction.value,
            "trend_strength": self.trend_strength,
            "current_rate": self.current_rate,
            "avg_rate": self.avg_rate,
            "median_rate": self.median_rate,
            "std_dev": self.std_dev,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "ma_7": self.ma_7,
            "ma_30": self.ma_30,
            "ma_90": self.ma_90,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "is_cyclical": self.is_cyclical,
            "cycle_period_hours": self.cycle_period_hours,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class RateCorrelation:
    """费率相关性分析"""
    symbol: str

    # 与价格相关性
    price_correlation: float        # 与价格变化相关性
    volume_correlation: float       # 与交易量相关性
    volatility_correlation: float   # 与波动率相关性

    # 与其他交易所相关性
    cross_exchange_correlations: Dict[str, float] = field(default_factory=dict)

    # 滞后相关性
    lag_1h_correlation: float = 0.0
    lag_4h_correlation: float = 0.0
    lag_24h_correlation: float = 0.0

    # 计算时间
    period_days: int = 30
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price_correlation": self.price_correlation,
            "volume_correlation": self.volume_correlation,
            "volatility_correlation": self.volatility_correlation,
            "cross_exchange_correlations": self.cross_exchange_correlations,
            "lag_1h_correlation": self.lag_1h_correlation,
            "lag_4h_correlation": self.lag_4h_correlation,
            "lag_24h_correlation": self.lag_24h_correlation,
            "period_days": self.period_days,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class ArbitrageOpportunity:
    """套利机会"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    symbol: str

    # 交易所信息
    long_exchange: ExchangeType
    short_exchange: ExchangeType

    # 费率差异
    rate_difference: float          # 费率差
    rate_difference_pct: float      # 费率差百分比

    # 价格差异
    price_difference: float = 0.0
    price_difference_pct: float = 0.0

    # 预期收益
    expected_profit_rate: float = 0.0       # 预期收益率
    expected_profit_value: float = 0.0      # 预期收益金额
    annualized_return: float = 0.0          # 年化收益率

    # 成本估算
    transaction_cost: float = 0.0
    slippage_cost: float = 0.0
    funding_cost_long: float = 0.0
    funding_cost_short: float = 0.0
    total_cost: float = 0.0

    # 风险评估
    risk_level: str = "medium"              # low/medium/high
    liquidity_score: float = 0.5
    stability_score: float = 0.5

    # 建议
    suggested_position_size: float = 0.0
    suggested_duration_hours: float = 0.0
    min_holding_period: float = 0.0

    # 状态
    is_active: bool = True
    confidence: float = 0.5

    # 时间戳
    detected_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "arbitrage_type": self.arbitrage_type.value,
            "symbol": self.symbol,
            "long_exchange": self.long_exchange.value,
            "short_exchange": self.short_exchange.value,
            "rate_difference": self.rate_difference,
            "rate_difference_pct": self.rate_difference_pct,
            "price_difference": self.price_difference,
            "price_difference_pct": self.price_difference_pct,
            "expected_profit_rate": self.expected_profit_rate,
            "expected_profit_value": self.expected_profit_value,
            "annualized_return": self.annualized_return,
            "transaction_cost": self.transaction_cost,
            "slippage_cost": self.slippage_cost,
            "funding_cost_long": self.funding_cost_long,
            "funding_cost_short": self.funding_cost_short,
            "total_cost": self.total_cost,
            "risk_level": self.risk_level,
            "liquidity_score": self.liquidity_score,
            "stability_score": self.stability_score,
            "suggested_position_size": self.suggested_position_size,
            "suggested_duration_hours": self.suggested_duration_hours,
            "min_holding_period": self.min_holding_period,
            "is_active": self.is_active,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class PositionCost:
    """持仓成本分析"""
    symbol: str
    exchange: ExchangeType

    # 持仓信息
    position_size: float            # 仓位大小
    entry_price: float              # 入场价格
    current_price: float            # 当前价格
    position_value: float           # 仓位价值

    # 累计费用
    cumulative_funding: float = 0.0         # 累计资金费用
    paid_funding_count: int = 0             # 已支付次数

    # 预期费用
    expected_next_funding: float = 0.0      # 下次预期费用
    expected_daily_funding: float = 0.0     # 预期日费用
    expected_weekly_funding: float = 0.0    # 预期周费用
    expected_monthly_funding: float = 0.0   # 预期月费用

    # 成本率
    funding_cost_rate: float = 0.0          # 资金费用率
    total_cost_rate: float = 0.0            # 总成本率

    # 盈亏分析
    unrealized_pnl: float = 0.0
    break_even_price: float = 0.0
    break_even_days: float = 0.0            # 盈亏平衡天数

    # 最优持仓时间
    optimal_holding_hours: Optional[float] = None
    max_recommended_hours: Optional[float] = None

    # 时间信息
    position_opened_at: datetime = field(default_factory=datetime.now)
    last_funding_payment: Optional[datetime] = None
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "position_value": self.position_value,
            "cumulative_funding": self.cumulative_funding,
            "paid_funding_count": self.paid_funding_count,
            "expected_next_funding": self.expected_next_funding,
            "expected_daily_funding": self.expected_daily_funding,
            "expected_weekly_funding": self.expected_weekly_funding,
            "expected_monthly_funding": self.expected_monthly_funding,
            "funding_cost_rate": self.funding_cost_rate,
            "total_cost_rate": self.total_cost_rate,
            "unrealized_pnl": self.unrealized_pnl,
            "break_even_price": self.break_even_price,
            "break_even_days": self.break_even_days,
            "optimal_holding_hours": self.optimal_holding_hours,
            "max_recommended_hours": self.max_recommended_hours,
            "position_opened_at": self.position_opened_at.isoformat(),
            "last_funding_payment": self.last_funding_payment.isoformat() if self.last_funding_payment else None,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class FundingAlert:
    """资金费率警报"""
    alert_id: str
    alert_type: AlertType
    alert_level: AlertLevel

    symbol: str
    exchange: ExchangeType

    # 警报详情
    message: str
    current_rate: float
    threshold_rate: Optional[float] = None

    # 额外数据
    details: Dict[str, Any] = field(default_factory=dict)

    # 建议操作
    suggested_action: str = ""

    # 状态
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "alert_level": self.alert_level.value,
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "message": self.message,
            "current_rate": self.current_rate,
            "threshold_rate": self.threshold_rate,
            "details": self.details,
            "suggested_action": self.suggested_action,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class MonitorConfig:
    """监控器配置"""
    # 费率阈值
    high_rate_threshold: float = 0.001          # 高费率阈值 (0.1%)
    low_rate_threshold: float = -0.001          # 低费率阈值 (-0.1%)
    extreme_rate_threshold: float = 0.005       # 极端费率阈值 (0.5%)

    # 费率变化阈值
    rate_change_threshold: float = 0.0005       # 费率变化阈值
    rate_spike_threshold: float = 0.002         # 费率激增阈值

    # 套利参数
    min_arbitrage_profit_rate: float = 0.0005   # 最小套利收益率 (0.05%)
    min_cross_exchange_diff: float = 0.0003     # 最小跨所差异 (0.03%)

    # 成本警告
    max_daily_cost_rate: float = 0.01           # 最大日成本率 (1%)
    max_monthly_cost_rate: float = 0.10         # 最大月成本率 (10%)

    # 监控设置
    update_interval_seconds: float = 60.0       # 更新间隔
    historical_data_days: int = 90              # 历史数据保留天数

    # 警报设置
    alert_cooldown_seconds: float = 300.0       # 警报冷却时间
    enable_high_rate_alert: bool = True
    enable_arbitrage_alert: bool = True
    enable_cost_alert: bool = True

    # 分析设置
    trend_analysis_enabled: bool = True
    correlation_analysis_enabled: bool = True
    prediction_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_rate_threshold": self.high_rate_threshold,
            "low_rate_threshold": self.low_rate_threshold,
            "extreme_rate_threshold": self.extreme_rate_threshold,
            "rate_change_threshold": self.rate_change_threshold,
            "rate_spike_threshold": self.rate_spike_threshold,
            "min_arbitrage_profit_rate": self.min_arbitrage_profit_rate,
            "min_cross_exchange_diff": self.min_cross_exchange_diff,
            "max_daily_cost_rate": self.max_daily_cost_rate,
            "max_monthly_cost_rate": self.max_monthly_cost_rate,
            "update_interval_seconds": self.update_interval_seconds,
            "historical_data_days": self.historical_data_days,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
            "enable_high_rate_alert": self.enable_high_rate_alert,
            "enable_arbitrage_alert": self.enable_arbitrage_alert,
            "enable_cost_alert": self.enable_cost_alert,
            "trend_analysis_enabled": self.trend_analysis_enabled,
            "correlation_analysis_enabled": self.correlation_analysis_enabled,
            "prediction_enabled": self.prediction_enabled,
        }


# ==================== 接口定义 ====================

class FundingRateProvider(ABC):
    """资金费率数据提供者接口"""

    @abstractmethod
    def get_funding_rate(
        self,
        symbol: str,
        exchange: ExchangeType
    ) -> Optional[FundingRateData]:
        """获取当前资金费率"""
        pass

    @abstractmethod
    def get_funding_rates(
        self,
        symbols: List[str],
        exchange: ExchangeType
    ) -> Dict[str, FundingRateData]:
        """批量获取资金费率"""
        pass

    @abstractmethod
    def get_historical_rates(
        self,
        symbol: str,
        exchange: ExchangeType,
        start_time: datetime,
        end_time: datetime
    ) -> List[HistoricalRate]:
        """获取历史资金费率"""
        pass

    @abstractmethod
    def get_all_exchanges_rates(
        self,
        symbol: str
    ) -> Dict[ExchangeType, FundingRateData]:
        """获取所有交易所的资金费率"""
        pass


class PriceProvider(ABC):
    """价格数据提供者接口"""

    @abstractmethod
    def get_spot_price(self, symbol: str, exchange: ExchangeType) -> float:
        """获取现货价格"""
        pass

    @abstractmethod
    def get_mark_price(self, symbol: str, exchange: ExchangeType) -> float:
        """获取标记价格"""
        pass

    @abstractmethod
    def get_index_price(self, symbol: str, exchange: ExchangeType) -> float:
        """获取指数价格"""
        pass


class PositionProvider(ABC):
    """持仓数据提供者接口"""

    @abstractmethod
    def get_position(
        self,
        symbol: str,
        exchange: ExchangeType
    ) -> Optional[Dict[str, Any]]:
        """获取持仓信息"""
        pass

    @abstractmethod
    def get_all_positions(
        self,
        exchange: Optional[ExchangeType] = None
    ) -> List[Dict[str, Any]]:
        """获取所有持仓"""
        pass


# ==================== 模拟实现 ====================

class SimulatedFundingRateProvider(FundingRateProvider):
    """模拟资金费率提供者"""

    def __init__(self):
        self.rates: Dict[Tuple[str, ExchangeType], FundingRateData] = {}
        self.historical: Dict[Tuple[str, ExchangeType], List[HistoricalRate]] = {}

    def set_rate(self, symbol: str, exchange: ExchangeType, rate_data: FundingRateData):
        self.rates[(symbol, exchange)] = rate_data

    def set_historical(
        self,
        symbol: str,
        exchange: ExchangeType,
        data: List[HistoricalRate]
    ):
        self.historical[(symbol, exchange)] = data

    def get_funding_rate(
        self,
        symbol: str,
        exchange: ExchangeType
    ) -> Optional[FundingRateData]:
        return self.rates.get((symbol, exchange))

    def get_funding_rates(
        self,
        symbols: List[str],
        exchange: ExchangeType
    ) -> Dict[str, FundingRateData]:
        result = {}
        for symbol in symbols:
            rate_data = self.get_funding_rate(symbol, exchange)
            if rate_data:
                result[symbol] = rate_data
        return result

    def get_historical_rates(
        self,
        symbol: str,
        exchange: ExchangeType,
        start_time: datetime,
        end_time: datetime
    ) -> List[HistoricalRate]:
        key = (symbol, exchange)
        if key not in self.historical:
            return []

        return [
            h for h in self.historical[key]
            if start_time <= h.timestamp <= end_time
        ]

    def get_all_exchanges_rates(
        self,
        symbol: str
    ) -> Dict[ExchangeType, FundingRateData]:
        result = {}
        for exchange in ExchangeType:
            rate_data = self.get_funding_rate(symbol, exchange)
            if rate_data:
                result[exchange] = rate_data
        return result


class SimulatedPriceProvider(PriceProvider):
    """模拟价格提供者"""

    def __init__(self):
        self.spot_prices: Dict[Tuple[str, ExchangeType], float] = {}
        self.mark_prices: Dict[Tuple[str, ExchangeType], float] = {}
        self.index_prices: Dict[Tuple[str, ExchangeType], float] = {}

    def set_prices(
        self,
        symbol: str,
        exchange: ExchangeType,
        spot: float,
        mark: float,
        index: float
    ):
        key = (symbol, exchange)
        self.spot_prices[key] = spot
        self.mark_prices[key] = mark
        self.index_prices[key] = index

    def get_spot_price(self, symbol: str, exchange: ExchangeType) -> float:
        return self.spot_prices.get((symbol, exchange), 50000.0)

    def get_mark_price(self, symbol: str, exchange: ExchangeType) -> float:
        return self.mark_prices.get((symbol, exchange), 50000.0)

    def get_index_price(self, symbol: str, exchange: ExchangeType) -> float:
        return self.index_prices.get((symbol, exchange), 50000.0)


class SimulatedPositionProvider(PositionProvider):
    """模拟持仓提供者"""

    def __init__(self):
        self.positions: Dict[Tuple[str, ExchangeType], Dict[str, Any]] = {}

    def set_position(
        self,
        symbol: str,
        exchange: ExchangeType,
        position: Dict[str, Any]
    ):
        self.positions[(symbol, exchange)] = position

    def get_position(
        self,
        symbol: str,
        exchange: ExchangeType
    ) -> Optional[Dict[str, Any]]:
        return self.positions.get((symbol, exchange))

    def get_all_positions(
        self,
        exchange: Optional[ExchangeType] = None
    ) -> List[Dict[str, Any]]:
        if exchange:
            return [
                pos for (sym, ex), pos in self.positions.items()
                if ex == exchange
            ]
        return list(self.positions.values())


# ==================== 核心引擎 ====================

class FundingRateMonitor:
    """资金费率监控器"""

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        funding_rate_provider: Optional[FundingRateProvider] = None,
        price_provider: Optional[PriceProvider] = None,
        position_provider: Optional[PositionProvider] = None,
    ):
        self.config = config or MonitorConfig()
        self.funding_rate_provider = funding_rate_provider or simulated_default(SimulatedFundingRateProvider, "data from FundingRateMonitor's simulated funding_rate_provider (SimulatedFundingRateProvider)")
        self.price_provider = price_provider or simulated_default(SimulatedPriceProvider, "data from FundingRateMonitor's simulated price_provider (SimulatedPriceProvider)")
        self.position_provider = position_provider or simulated_default(SimulatedPositionProvider, "data from FundingRateMonitor's simulated position_provider (SimulatedPositionProvider)")

        # 数据存储
        self.current_rates: Dict[Tuple[str, ExchangeType], FundingRateData] = {}
        self.historical_rates: Dict[Tuple[str, ExchangeType], List[HistoricalRate]] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        self.alerts: List[FundingAlert] = []

        # 分析缓存
        self.trend_cache: Dict[Tuple[str, ExchangeType], RateTrend] = {}
        self.correlation_cache: Dict[str, RateCorrelation] = {}

        # 状态
        self.state = MonitorState.IDLE
        self._monitoring_task: Optional[asyncio.Task] = None

        # 回调
        self._on_rate_update: Optional[Callable[[FundingRateData], None]] = None
        self._on_alert: Optional[Callable[[FundingAlert], None]] = None
        self._on_arbitrage: Optional[Callable[[ArbitrageOpportunity], None]] = None

        # 统计
        self.stats = {
            "symbols_monitored": 0,
            "total_alerts": 0,
            "arbitrage_opportunities_found": 0,
            "rate_updates": 0,
        }

    # ==================== 费率跟踪 ====================

    def update_funding_rate(
        self,
        symbol: str,
        exchange: ExchangeType
    ) -> Optional[FundingRateData]:
        """更新资金费率"""
        rate_data = self.funding_rate_provider.get_funding_rate(symbol, exchange)
        if not rate_data:
            return None

        # 存储当前费率
        key = (symbol, exchange)
        old_rate = self.current_rates.get(key)
        self.current_rates[key] = rate_data

        # 添加到历史记录
        if key not in self.historical_rates:
            self.historical_rates[key] = []

        historical_entry = HistoricalRate(
            symbol=symbol,
            exchange=exchange,
            rate=rate_data.rate,
            mark_price=rate_data.mark_price,
            index_price=rate_data.index_price,
            timestamp=rate_data.timestamp,
            payment_time=rate_data.next_payment_time,
        )
        self.historical_rates[key].append(historical_entry)

        # 限制历史数据大小
        max_records = self.config.historical_data_days * 24  # 按小时
        if len(self.historical_rates[key]) > max_records:
            self.historical_rates[key] = self.historical_rates[key][-max_records:]

        # 检查费率变化
        if old_rate:
            self._check_rate_changes(old_rate, rate_data)

        # 统计
        self.stats["rate_updates"] += 1

        # 触发回调
        if self._on_rate_update:
            self._on_rate_update(rate_data)

        return rate_data

    def update_all_rates(
        self,
        symbols: List[str],
        exchanges: Optional[List[ExchangeType]] = None
    ) -> Dict[Tuple[str, ExchangeType], FundingRateData]:
        """批量更新费率"""
        if exchanges is None:
            exchanges = list(ExchangeType)

        results = {}
        for symbol in symbols:
            for exchange in exchanges:
                rate_data = self.update_funding_rate(symbol, exchange)
                if rate_data:
                    results[(symbol, exchange)] = rate_data

        self.stats["symbols_monitored"] = len(set(k[0] for k in results.keys()))
        return results

    def get_current_rate(
        self,
        symbol: str,
        exchange: ExchangeType
    ) -> Optional[FundingRateData]:
        """获取当前费率"""
        return self.current_rates.get((symbol, exchange))

    def get_all_current_rates(self) -> Dict[Tuple[str, ExchangeType], FundingRateData]:
        """获取所有当前费率"""
        return dict(self.current_rates)

    def get_historical_rates(
        self,
        symbol: str,
        exchange: ExchangeType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[HistoricalRate]:
        """获取历史费率"""
        key = (symbol, exchange)
        if key not in self.historical_rates:
            return []

        rates = self.historical_rates[key]

        if start_time:
            rates = [r for r in rates if r.timestamp >= start_time]
        if end_time:
            rates = [r for r in rates if r.timestamp <= end_time]

        return rates[-limit:]

    # ==================== 费率预测 ====================

    def predict_next_rate(
        self,
        symbol: str,
        exchange: ExchangeType,
        method: str = "ma"
    ) -> Optional[float]:
        """预测下次费率"""
        historical = self.get_historical_rates(symbol, exchange, limit=100)
        if len(historical) < 3:
            return None

        rates = [h.rate for h in historical]

        if method == "ma":
            # 移动平均
            return statistics.mean(rates[-10:])
        elif method == "ema":
            # 指数移动平均
            alpha = 0.3
            ema = rates[-1]
            for rate in rates[-20:]:
                ema = alpha * rate + (1 - alpha) * ema
            return ema
        elif method == "linear":
            # 线性趋势
            n = len(rates)
            x_mean = (n - 1) / 2
            y_mean = statistics.mean(rates)

            numerator = sum((i - x_mean) * (rates[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                return rates[-1]

            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            return slope * n + intercept
        else:
            return rates[-1]

    # ==================== 费率分析 ====================

    def analyze_rate_trend(
        self,
        symbol: str,
        exchange: ExchangeType,
        period_days: int = 30
    ) -> RateTrend:
        """分析费率趋势"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=period_days)

        historical = self.get_historical_rates(symbol, exchange, start_time, end_time)

        if not historical:
            current = self.get_current_rate(symbol, exchange)
            current_rate = current.rate if current else 0.0

            return RateTrend(
                symbol=symbol,
                exchange=exchange,
                trend_direction=TrendDirection.NEUTRAL,
                trend_strength=0.0,
                current_rate=current_rate,
                avg_rate=current_rate,
                median_rate=current_rate,
                std_dev=0.0,
                min_rate=current_rate,
                max_rate=current_rate,
            )

        rates = [h.rate for h in historical]
        current_rate = rates[-1]

        # 基本统计
        avg_rate = statistics.mean(rates)
        median_rate = statistics.median(rates)
        std_dev = statistics.stdev(rates) if len(rates) > 1 else 0.0
        min_rate = min(rates)
        max_rate = max(rates)

        # 移动平均
        ma_7 = statistics.mean(rates[-min(7, len(rates)):])
        ma_30 = statistics.mean(rates[-min(30, len(rates)):])
        ma_90 = statistics.mean(rates[-min(90, len(rates)):]) if len(rates) >= 90 else avg_rate

        # 计算趋势线
        n = len(rates)
        x_mean = (n - 1) / 2
        y_mean = avg_rate

        numerator = sum((i - x_mean) * (rates[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0.0

        # 计算R²
        y_pred = [slope * i + (y_mean - slope * x_mean) for i in range(n)]
        ss_tot = sum((rates[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((rates[i] - y_pred[i]) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # 确定趋势方向
        if slope > 0.0001 and r_squared > 0.5:
            trend_direction = TrendDirection.BULLISH
            trend_strength = min(abs(slope) * 1000, 1.0) * r_squared
        elif slope < -0.0001 and r_squared > 0.5:
            trend_direction = TrendDirection.BEARISH
            trend_strength = min(abs(slope) * 1000, 1.0) * r_squared
        elif std_dev > abs(avg_rate) * 0.5:
            trend_direction = TrendDirection.VOLATILE
            trend_strength = min(std_dev / abs(avg_rate), 1.0)
        else:
            trend_direction = TrendDirection.NEUTRAL
            trend_strength = 0.3

        result = RateTrend(
            symbol=symbol,
            exchange=exchange,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            current_rate=current_rate,
            avg_rate=avg_rate,
            median_rate=median_rate,
            std_dev=std_dev,
            min_rate=min_rate,
            max_rate=max_rate,
            ma_7=ma_7,
            ma_30=ma_30,
            ma_90=ma_90,
            slope=slope,
            r_squared=r_squared,
            period_start=start_time,
            period_end=end_time,
        )

        # 缓存结果
        self.trend_cache[(symbol, exchange)] = result

        return result

    def analyze_rate_correlation(
        self,
        symbol: str,
        period_days: int = 30
    ) -> RateCorrelation:
        """分析费率相关性"""
        # 简化实现 - 实际应使用真实价格和交易量数据
        result = RateCorrelation(
            symbol=symbol,
            price_correlation=0.6,      # 模拟值
            volume_correlation=0.4,     # 模拟值
            volatility_correlation=0.7, # 模拟值
            period_days=period_days,
        )

        # 跨交易所相关性
        exchanges = list(ExchangeType)
        for i, ex1 in enumerate(exchanges):
            for ex2 in exchanges[i+1:]:
                corr = 0.8 + (hash(f"{ex1.value}{ex2.value}") % 20 - 10) / 100
                result.cross_exchange_correlations[f"{ex1.value}_{ex2.value}"] = corr

        self.correlation_cache[symbol] = result
        return result

    # ==================== 套利分析 ====================

    def find_arbitrage_opportunities(
        self,
        symbols: Optional[List[str]] = None
    ) -> List[ArbitrageOpportunity]:
        """寻找套利机会"""
        if symbols is None:
            symbols = list(set(k[0] for k in self.current_rates.keys()))

        opportunities = []

        for symbol in symbols:
            # 跨交易所套利
            cross_ex_opps = self._find_cross_exchange_arbitrage(symbol)
            opportunities.extend(cross_ex_opps)

            # 现货-永续套利
            spot_perp_opps = self._find_spot_perpetual_arbitrage(symbol)
            opportunities.extend(spot_perp_opps)

        # 过滤和排序
        opportunities = [
            opp for opp in opportunities
            if opp.expected_profit_rate >= self.config.min_arbitrage_profit_rate
        ]
        opportunities.sort(key=lambda x: x.expected_profit_rate, reverse=True)

        # 存储
        self.arbitrage_opportunities = opportunities
        self.stats["arbitrage_opportunities_found"] = len(opportunities)

        # 触发回调
        for opp in opportunities:
            if self._on_arbitrage:
                self._on_arbitrage(opp)

        return opportunities

    def _find_cross_exchange_arbitrage(
        self,
        symbol: str
    ) -> List[ArbitrageOpportunity]:
        """寻找跨交易所套利机会"""
        opportunities = []

        # 获取所有交易所的费率
        rates = {
            ex: self.current_rates.get((symbol, ex))
            for ex in ExchangeType
        }
        rates = {ex: r for ex, r in rates.items() if r is not None}

        if len(rates) < 2:
            return opportunities

        # 比较每对交易所
        exchanges = list(rates.keys())
        for i, ex1 in enumerate(exchanges):
            for ex2 in exchanges[i+1:]:
                rate1 = rates[ex1].rate
                rate2 = rates[ex2].rate

                rate_diff = abs(rate1 - rate2)

                if rate_diff < self.config.min_cross_exchange_diff:
                    continue

                # 确定多空方向
                if rate1 < rate2:
                    long_ex, short_ex = ex1, ex2
                    long_rate, short_rate = rate1, rate2
                else:
                    long_ex, short_ex = ex2, ex1
                    long_rate, short_rate = rate2, rate1

                # 估算收益
                expected_profit = short_rate - long_rate
                annualized = expected_profit * 365 * 3  # 假设每天3次支付

                # 估算成本
                tx_cost = 0.001  # 0.1%交易成本

                opportunity = ArbitrageOpportunity(
                    opportunity_id=str(uuid.uuid4()),
                    arbitrage_type=ArbitrageType.CROSS_EXCHANGE,
                    symbol=symbol,
                    long_exchange=long_ex,
                    short_exchange=short_ex,
                    rate_difference=expected_profit,
                    rate_difference_pct=expected_profit / abs(long_rate) * 100 if long_rate != 0 else 0,
                    expected_profit_rate=expected_profit,
                    annualized_return=annualized,
                    transaction_cost=tx_cost,
                    total_cost=tx_cost,
                    risk_level="medium",
                    confidence=0.7,
                )

                opportunities.append(opportunity)

        return opportunities

    def _find_spot_perpetual_arbitrage(
        self,
        symbol: str
    ) -> List[ArbitrageOpportunity]:
        """寻找现货-永续套利机会"""
        opportunities = []

        for exchange in ExchangeType:
            rate_data = self.current_rates.get((symbol, exchange))
            if not rate_data:
                continue

            # 获取价格
            spot_price = self.price_provider.get_spot_price(symbol, exchange)
            mark_price = rate_data.mark_price

            if spot_price <= 0 or mark_price <= 0:
                continue

            # 价格差异
            price_diff = mark_price - spot_price
            price_diff_pct = (price_diff / spot_price) * 100

            # 如果永续合约溢价 + 正费率 = 做空永续+做多现货
            if rate_data.rate > 0 and price_diff > 0:
                expected_profit = rate_data.rate + (price_diff_pct / 100)

                opportunity = ArbitrageOpportunity(
                    opportunity_id=str(uuid.uuid4()),
                    arbitrage_type=ArbitrageType.SPOT_PERP,
                    symbol=symbol,
                    long_exchange=exchange,  # 做多现货
                    short_exchange=exchange, # 做空永续
                    rate_difference=rate_data.rate,
                    rate_difference_pct=rate_data.rate * 100,
                    price_difference=price_diff,
                    price_difference_pct=price_diff_pct,
                    expected_profit_rate=expected_profit,
                    annualized_return=expected_profit * 365 * 3,
                    risk_level="low",
                    confidence=0.8,
                )

                opportunities.append(opportunity)

        return opportunities

    # ==================== 持仓成本 ====================

    def calculate_position_cost(
        self,
        symbol: str,
        exchange: ExchangeType,
        position_size: Optional[float] = None,
        entry_price: Optional[float] = None
    ) -> PositionCost:
        """计算持仓成本"""
        # 获取持仓信息
        if position_size is None or entry_price is None:
            position = self.position_provider.get_position(symbol, exchange)
            if position:
                position_size = position.get("size", 0)
                entry_price = position.get("entry_price", 0)
            else:
                position_size = 0
                entry_price = 0

        # 获取当前费率和价格
        rate_data = self.get_current_rate(symbol, exchange)
        current_price = self.price_provider.get_mark_price(symbol, exchange)

        if not rate_data or position_size == 0:
            return PositionCost(
                symbol=symbol,
                exchange=exchange,
                position_size=position_size,
                entry_price=entry_price,
                current_price=current_price,
                position_value=position_size * current_price,
            )

        position_value = abs(position_size) * current_price

        # 计算预期费用
        payments_per_day = self._get_payments_per_day(rate_data.payment_frequency)
        expected_daily = position_value * rate_data.rate * payments_per_day
        expected_weekly = expected_daily * 7
        expected_monthly = expected_daily * 30

        # 计算下次费用
        expected_next = position_value * rate_data.rate

        # 盈亏分析
        unrealized_pnl = (current_price - entry_price) * position_size

        # 盈亏平衡价格（简化）
        break_even_price = entry_price
        if position_size > 0:  # 多头
            break_even_price = entry_price + (expected_daily / abs(position_size))
        else:  # 空头
            break_even_price = entry_price - (expected_daily / abs(position_size))

        result = PositionCost(
            symbol=symbol,
            exchange=exchange,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            position_value=position_value,
            expected_next_funding=expected_next,
            expected_daily_funding=expected_daily,
            expected_weekly_funding=expected_weekly,
            expected_monthly_funding=expected_monthly,
            funding_cost_rate=rate_data.rate * payments_per_day,
            unrealized_pnl=unrealized_pnl,
            break_even_price=break_even_price,
        )

        return result

    def _get_payments_per_day(self, frequency: PaymentFrequency) -> int:
        """获取每日支付次数"""
        mapping = {
            PaymentFrequency.HOURLY: 24,
            PaymentFrequency.FOUR_HOURLY: 6,
            PaymentFrequency.EIGHT_HOURLY: 3,
            PaymentFrequency.DAILY: 1,
            PaymentFrequency.WEEKLY: 1/7,
        }
        return mapping.get(frequency, 3)

    # ==================== 警报系统 ====================

    def _check_rate_changes(
        self,
        old_rate: FundingRateData,
        new_rate: FundingRateData
    ):
        """检查费率变化并生成警报"""
        rate_change = abs(new_rate.rate - old_rate.rate)

        # 检查费率激增
        if rate_change >= self.config.rate_spike_threshold:
            self._create_alert(
                AlertType.RATE_SPIKE,
                AlertLevel.WARNING,
                new_rate.symbol,
                new_rate.exchange,
                f"Funding rate spiked by {rate_change*100:.3f}%",
                new_rate.rate,
                details={"old_rate": old_rate.rate, "change": rate_change},
            )

        # 检查费率反转
        if (old_rate.rate > 0 and new_rate.rate < 0) or (old_rate.rate < 0 and new_rate.rate > 0):
            self._create_alert(
                AlertType.RATE_REVERSAL,
                AlertLevel.INFO,
                new_rate.symbol,
                new_rate.exchange,
                f"Funding rate reversed from {old_rate.rate*100:.3f}% to {new_rate.rate*100:.3f}%",
                new_rate.rate,
            )

        # 检查高/低费率
        if new_rate.rate >= self.config.high_rate_threshold and self.config.enable_high_rate_alert:
            self._create_alert(
                AlertType.HIGH_RATE,
                AlertLevel.WARNING if new_rate.rate < self.config.extreme_rate_threshold else AlertLevel.CRITICAL,
                new_rate.symbol,
                new_rate.exchange,
                f"High funding rate: {new_rate.rate*100:.3f}%",
                new_rate.rate,
                threshold_rate=self.config.high_rate_threshold,
            )

        if new_rate.rate <= self.config.low_rate_threshold:
            self._create_alert(
                AlertType.LOW_RATE,
                AlertLevel.INFO,
                new_rate.symbol,
                new_rate.exchange,
                f"Low funding rate: {new_rate.rate*100:.3f}%",
                new_rate.rate,
                threshold_rate=self.config.low_rate_threshold,
            )

    def _create_alert(
        self,
        alert_type: AlertType,
        alert_level: AlertLevel,
        symbol: str,
        exchange: ExchangeType,
        message: str,
        current_rate: float,
        threshold_rate: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> FundingAlert:
        """创建警报"""
        alert = FundingAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            alert_level=alert_level,
            symbol=symbol,
            exchange=exchange,
            message=message,
            current_rate=current_rate,
            threshold_rate=threshold_rate,
            details=details or {},
        )

        self.alerts.append(alert)
        self.stats["total_alerts"] += 1

        if self._on_alert:
            self._on_alert(alert)

        return alert

    def get_alerts(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[ExchangeType] = None,
        alert_type: Optional[AlertType] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100
    ) -> List[FundingAlert]:
        """获取警报"""
        alerts = self.alerts

        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        if exchange:
            alerts = [a for a in alerts if a.exchange == exchange]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认警报"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                return True
        return False

    # ==================== 监控控制 ====================

    def on_rate_update(self, callback: Callable[[FundingRateData], None]):
        """注册费率更新回调"""
        self._on_rate_update = callback

    def on_alert(self, callback: Callable[[FundingAlert], None]):
        """注册警报回调"""
        self._on_alert = callback

    def on_arbitrage(self, callback: Callable[[ArbitrageOpportunity], None]):
        """注册套利机会回调"""
        self._on_arbitrage = callback

    async def start_monitoring(
        self,
        symbols: List[str],
        exchanges: Optional[List[ExchangeType]] = None
    ):
        """启动监控"""
        if self.state == MonitorState.MONITORING:
            return

        self.state = MonitorState.MONITORING
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(symbols, exchanges)
        )

    async def stop_monitoring(self):
        """停止监控"""
        self.state = MonitorState.STOPPED
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

    async def _monitoring_loop(
        self,
        symbols: List[str],
        exchanges: Optional[List[ExchangeType]]
    ):
        """监控循环"""
        while self.state == MonitorState.MONITORING:
            try:
                # 更新费率
                self.update_all_rates(symbols, exchanges)

                # 寻找套利机会
                if self.config.enable_arbitrage_alert:
                    self.find_arbitrage_opportunities(symbols)

                # 分析趋势
                if self.config.trend_analysis_enabled:
                    for symbol in symbols:
                        for exchange in (exchanges or list(ExchangeType)):
                            if (symbol, exchange) in self.current_rates:
                                self.analyze_rate_trend(symbol, exchange)

                await asyncio.sleep(self.config.update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                # The monitor stops here, so this is the only record of why.
                logger.exception("Funding rate monitoring loop failed; monitor stopping")
                self.state = MonitorState.ERROR
                break

    # ==================== 统计 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self.state.value,
            "symbols_monitored": self.stats["symbols_monitored"],
            "total_rates_tracked": len(self.current_rates),
            "total_alerts": self.stats["total_alerts"],
            "unacknowledged_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "arbitrage_opportunities": len(self.arbitrage_opportunities),
            "active_opportunities": len([o for o in self.arbitrage_opportunities if o.is_active]),
            "rate_updates": self.stats["rate_updates"],
        }


# ==================== 工厂函数 ====================

def create_funding_rate_monitor(
    config: Optional[MonitorConfig] = None,
    funding_rate_provider: Optional[FundingRateProvider] = None,
    price_provider: Optional[PriceProvider] = None,
    position_provider: Optional[PositionProvider] = None,
) -> FundingRateMonitor:
    """创建资金费率监控器"""
    return FundingRateMonitor(
        config=config,
        funding_rate_provider=funding_rate_provider,
        price_provider=price_provider,
        position_provider=position_provider,
    )


def create_funding_rate_data(
    symbol: str,
    exchange: ExchangeType,
    rate: float,
    **kwargs
) -> FundingRateData:
    """创建资金费率数据"""
    return FundingRateData(
        symbol=symbol,
        exchange=exchange,
        rate=rate,
        timestamp=kwargs.get("timestamp", datetime.now()),
        next_payment_time=kwargs.get("next_payment_time", datetime.now() + timedelta(hours=8)),
        payment_frequency=kwargs.get("payment_frequency", PaymentFrequency.EIGHT_HOURLY),
        **{k: v for k, v in kwargs.items() if k not in ["timestamp", "next_payment_time", "payment_frequency"]}
    )


def quick_funding_check(
    symbol: str,
    exchanges: List[ExchangeType],
    rates: Dict[ExchangeType, float]
) -> Dict[str, Any]:
    """快速资金费率检查"""
    monitor = create_funding_rate_monitor()

    # 设置模拟数据
    provider = monitor.funding_rate_provider
    if isinstance(provider, SimulatedFundingRateProvider):
        for exchange, rate in rates.items():
            rate_data = create_funding_rate_data(symbol, exchange, rate)
            provider.set_rate(symbol, exchange, rate_data)

    # 更新费率
    monitor.update_all_rates([symbol], exchanges)

    # 寻找套利机会
    opportunities = monitor.find_arbitrage_opportunities([symbol])

    return {
        "symbol": symbol,
        "rates": {ex.value: rates.get(ex, 0) for ex in exchanges},
        "arbitrage_opportunities": len(opportunities),
        "best_opportunity": opportunities[0].to_dict() if opportunities else None,
        "statistics": monitor.get_statistics(),
    }


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "RateType",
    "PaymentFrequency",
    "ExchangeType",
    "AlertLevel",
    "AlertType",
    "ArbitrageType",
    "TrendDirection",
    "MonitorState",
    # 数据类
    "FundingRateData",
    "HistoricalRate",
    "RateTrend",
    "RateCorrelation",
    "ArbitrageOpportunity",
    "PositionCost",
    "FundingAlert",
    "MonitorConfig",
    # 接口
    "FundingRateProvider",
    "PriceProvider",
    "PositionProvider",
    # 模拟实现
    "SimulatedFundingRateProvider",
    "SimulatedPriceProvider",
    "SimulatedPositionProvider",
    # 核心引擎
    "FundingRateMonitor",
    # 工厂函数
    "create_funding_rate_monitor",
    "create_funding_rate_data",
    "quick_funding_check",
]
