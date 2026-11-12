"""
Counterparty Risk Monitor - 交易对手风险监控器

实时监控交易对手风险，包括：
- 交易对手信用评级跟踪
- 敞口集中度监控
- 信用VaR计算
- 交易对手风险调整价值(CVA)
- 交易对手限额管理
- 信用事件预警
- 违约概率(PD)与损失给定违约(LGD)评估

符合监管要求: Dodd-Frank, EMIR, Basel III
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
import math
import uuid
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ============ Enums ============

class CreditRating(str, Enum):
    """信用评级 (标普/穆迪对照)"""
    AAA = "AAA"      # Aaa - 最高信用
    AA_PLUS = "AA+"  # Aa1
    AA = "AA"        # Aa2
    AA_MINUS = "AA-" # Aa3
    A_PLUS = "A+"    # A1
    A = "A"          # A2
    A_MINUS = "A-"   # A3
    BBB_PLUS = "BBB+"  # Baa1
    BBB = "BBB"      # Baa2 - 投资级下限
    BBB_MINUS = "BBB-" # Baa3
    BB_PLUS = "BB+"  # Ba1 - 投机级开始
    BB = "BB"        # Ba2
    BB_MINUS = "BB-" # Ba3
    B_PLUS = "B+"    # B1
    B = "B"          # B2
    B_MINUS = "B-"   # B3
    CCC = "CCC"      # Caa
    CC = "CC"        # Ca
    C = "C"          # C
    D = "D"          # D - 已违约
    NR = "NR"        # 未评级


class CounterpartyType(str, Enum):
    """交易对手类型"""
    BANK = "bank"                    # 银行
    BROKER_DEALER = "broker_dealer"  # 券商
    HEDGE_FUND = "hedge_fund"        # 对冲基金
    ASSET_MANAGER = "asset_manager"  # 资产管理公司
    CORPORATE = "corporate"          # 企业
    SOVEREIGN = "sovereign"          # 主权国家
    CLEARING_HOUSE = "clearing_house" # 清算所
    EXCHANGE = "exchange"            # 交易所
    OTHER = "other"


class ExposureType(str, Enum):
    """敞口类型"""
    CURRENT = "current"              # 当前敞口
    POTENTIAL = "potential"          # 潜在敞口
    SETTLEMENT = "settlement"        # 结算敞口
    PRE_SETTLEMENT = "pre_settlement" # 结算前敞口
    CREDIT = "credit"                # 信用敞口
    WRONG_WAY = "wrong_way"          # 错向风险敞口


class CreditEventType(str, Enum):
    """信用事件类型"""
    RATING_DOWNGRADE = "rating_downgrade"  # 评级下调
    RATING_UPGRADE = "rating_upgrade"      # 评级上调
    RATING_WATCH = "rating_watch"          # 评级观察
    CDS_SPIKE = "cds_spike"                # CDS价差飙升
    DEFAULT = "default"                    # 违约
    BANKRUPTCY = "bankruptcy"              # 破产
    RESTRUCTURING = "restructuring"        # 债务重组
    CROSS_DEFAULT = "cross_default"        # 交叉违约
    LIMIT_BREACH = "limit_breach"          # 限额突破
    CONCENTRATION_WARNING = "concentration_warning"  # 集中度警告


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


# ============ Data Classes ============

@dataclass
class CounterpartyProfile:
    """交易对手档案"""
    counterparty_id: str
    name: str
    counterparty_type: CounterpartyType
    credit_rating: CreditRating
    internal_rating: Optional[str] = None  # 内部评级
    lei: Optional[str] = None              # 法人实体识别码
    country: str = "US"
    sector: str = "Financial"
    parent_company: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterparty_id": self.counterparty_id,
            "name": self.name,
            "counterparty_type": self.counterparty_type.value,
            "credit_rating": self.credit_rating.value,
            "internal_rating": self.internal_rating,
            "lei": self.lei,
            "country": self.country,
            "sector": self.sector,
            "parent_company": self.parent_company,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


@dataclass
class DefaultProbability:
    """违约概率"""
    counterparty_id: str
    pd_1y: float           # 1年违约概率
    pd_5y: float           # 5年违约概率
    pd_cumulative: float   # 累计违约概率
    lgd: float             # 损失给定违约 (0-1)
    recovery_rate: float   # 回收率 (1 - lgd)
    expected_loss: float   # 预期损失 = PD * LGD * EAD
    calculated_at: datetime = field(default_factory=datetime.now)
    source: str = "internal"  # internal/external/model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterparty_id": self.counterparty_id,
            "pd_1y": self.pd_1y,
            "pd_5y": self.pd_5y,
            "pd_cumulative": self.pd_cumulative,
            "lgd": self.lgd,
            "recovery_rate": self.recovery_rate,
            "expected_loss": self.expected_loss,
            "calculated_at": self.calculated_at.isoformat(),
            "source": self.source,
        }


@dataclass
class Exposure:
    """敞口"""
    exposure_id: str
    counterparty_id: str
    exposure_type: ExposureType
    gross_exposure: float          # 总敞口
    net_exposure: float            # 净敞口 (考虑抵消)
    collateral: float = 0.0        # 抵押品
    netting_benefit: float = 0.0   # 净额结算收益
    currency: str = "USD"
    positions: List[Dict[str, Any]] = field(default_factory=list)
    calculated_at: datetime = field(default_factory=datetime.now)

    @property
    def collateralized_exposure(self) -> float:
        """抵押后敞口"""
        return max(0, self.net_exposure - self.collateral)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exposure_id": self.exposure_id,
            "counterparty_id": self.counterparty_id,
            "exposure_type": self.exposure_type.value,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "collateral": self.collateral,
            "netting_benefit": self.netting_benefit,
            "collateralized_exposure": self.collateralized_exposure,
            "currency": self.currency,
            "position_count": len(self.positions),
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class CounterpartyLimit:
    """交易对手限额"""
    limit_id: str
    counterparty_id: str
    limit_type: str                # gross/net/settlement
    limit_value: float
    current_usage: float = 0.0
    currency: str = "USD"
    effective_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None
    approved_by: str = "system"
    warning_threshold: float = 0.8  # 80% 预警
    critical_threshold: float = 0.95  # 95% 严重

    @property
    def utilization(self) -> float:
        """使用率"""
        if self.limit_value <= 0:
            return 0.0
        return self.current_usage / self.limit_value

    @property
    def available(self) -> float:
        """可用额度"""
        return max(0, self.limit_value - self.current_usage)

    @property
    def status(self) -> str:
        """限额状态"""
        util = self.utilization
        if util >= 1.0:
            return "breached"
        elif util >= self.critical_threshold:
            return "critical"
        elif util >= self.warning_threshold:
            return "warning"
        return "normal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "limit_id": self.limit_id,
            "counterparty_id": self.counterparty_id,
            "limit_type": self.limit_type,
            "limit_value": self.limit_value,
            "current_usage": self.current_usage,
            "utilization": self.utilization,
            "available": self.available,
            "status": self.status,
            "currency": self.currency,
            "effective_date": self.effective_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "approved_by": self.approved_by,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
        }


@dataclass
class CreditVaR:
    """信用VaR"""
    counterparty_id: str
    var_95: float                  # 95% VaR
    var_99: float                  # 99% VaR
    expected_shortfall: float      # Expected Shortfall (CVaR)
    time_horizon_days: int = 10    # 时间窗口
    confidence_level: float = 0.99
    calculated_at: datetime = field(default_factory=datetime.now)
    methodology: str = "monte_carlo"  # monte_carlo/historical/parametric

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterparty_id": self.counterparty_id,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "expected_shortfall": self.expected_shortfall,
            "time_horizon_days": self.time_horizon_days,
            "confidence_level": self.confidence_level,
            "calculated_at": self.calculated_at.isoformat(),
            "methodology": self.methodology,
        }


@dataclass
class CVAResult:
    """信用估值调整结果"""
    counterparty_id: str
    cva: float                     # Credit Valuation Adjustment
    dva: float                     # Debit Valuation Adjustment
    bilateral_cva: float           # 双边CVA
    fva: float                     # Funding Valuation Adjustment
    total_xva: float               # 总XVA
    exposure_profile: List[float] = field(default_factory=list)  # 敞口曲线
    discount_factors: List[float] = field(default_factory=list)  # 折现因子
    survival_probs: List[float] = field(default_factory=list)    # 存活概率
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterparty_id": self.counterparty_id,
            "cva": self.cva,
            "dva": self.dva,
            "bilateral_cva": self.bilateral_cva,
            "fva": self.fva,
            "total_xva": self.total_xva,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class CreditEvent:
    """信用事件"""
    event_id: str
    counterparty_id: str
    event_type: CreditEventType
    severity: AlertSeverity
    description: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    source: str = "internal"
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "counterparty_id": self.counterparty_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "actions_taken": self.actions_taken,
        }


@dataclass
class ConcentrationMetrics:
    """集中度指标"""
    total_exposure: float
    top_counterparty_exposure: float
    top_5_exposure: float
    top_10_exposure: float
    herfindahl_index: float        # 赫芬达尔指数 (0-1, 越高越集中)
    largest_counterparty_id: str
    largest_counterparty_pct: float
    counterparty_count: int
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_exposure": self.total_exposure,
            "top_counterparty_exposure": self.top_counterparty_exposure,
            "top_5_exposure": self.top_5_exposure,
            "top_10_exposure": self.top_10_exposure,
            "herfindahl_index": self.herfindahl_index,
            "largest_counterparty_id": self.largest_counterparty_id,
            "largest_counterparty_pct": self.largest_counterparty_pct,
            "counterparty_count": self.counterparty_count,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class CounterpartyRiskReport:
    """交易对手风险报告"""
    report_id: str
    counterparty_id: str
    profile: CounterpartyProfile
    current_exposure: Optional[Exposure]
    default_probability: Optional[DefaultProbability]
    credit_var: Optional[CreditVaR]
    cva_result: Optional[CVAResult]
    limits: List[CounterpartyLimit]
    recent_events: List[CreditEvent]
    risk_score: float              # 综合风险评分 0-100
    risk_level: str                # low/medium/high/critical
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "counterparty_id": self.counterparty_id,
            "profile": self.profile.to_dict(),
            "current_exposure": self.current_exposure.to_dict() if self.current_exposure else None,
            "default_probability": self.default_probability.to_dict() if self.default_probability else None,
            "credit_var": self.credit_var.to_dict() if self.credit_var else None,
            "cva_result": self.cva_result.to_dict() if self.cva_result else None,
            "limits": [l.to_dict() for l in self.limits],
            "recent_events": [e.to_dict() for e in self.recent_events],
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class CounterpartyRiskConfig:
    """交易对手风险监控配置"""
    update_interval: float = 60.0        # 更新间隔(秒)
    var_confidence: float = 0.99         # VaR置信度
    var_horizon_days: int = 10           # VaR时间窗口
    concentration_warning: float = 0.25  # 单一对手25%集中度警告
    concentration_critical: float = 0.35 # 单一对手35%集中度严重
    cds_spike_threshold: float = 0.5     # CDS价差变动50%触发警告
    rating_watch_days: int = 30          # 评级观察期
    history_size: int = 1000             # 历史记录大小
    enable_cva: bool = True              # 启用CVA计算
    enable_var: bool = True              # 启用VaR计算

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_interval": self.update_interval,
            "var_confidence": self.var_confidence,
            "var_horizon_days": self.var_horizon_days,
            "concentration_warning": self.concentration_warning,
            "concentration_critical": self.concentration_critical,
            "cds_spike_threshold": self.cds_spike_threshold,
            "rating_watch_days": self.rating_watch_days,
            "history_size": self.history_size,
            "enable_cva": self.enable_cva,
            "enable_var": self.enable_var,
        }


# ============ Rating to PD Mapping ============

# 基于历史违约率的评级-PD映射表 (1年期)
RATING_TO_PD: Dict[CreditRating, float] = {
    CreditRating.AAA: 0.0001,      # 0.01%
    CreditRating.AA_PLUS: 0.0002,
    CreditRating.AA: 0.0003,
    CreditRating.AA_MINUS: 0.0004,
    CreditRating.A_PLUS: 0.0006,
    CreditRating.A: 0.0008,
    CreditRating.A_MINUS: 0.0012,
    CreditRating.BBB_PLUS: 0.0020,
    CreditRating.BBB: 0.0030,
    CreditRating.BBB_MINUS: 0.0050,
    CreditRating.BB_PLUS: 0.0080,
    CreditRating.BB: 0.0120,
    CreditRating.BB_MINUS: 0.0180,
    CreditRating.B_PLUS: 0.0280,
    CreditRating.B: 0.0420,
    CreditRating.B_MINUS: 0.0650,
    CreditRating.CCC: 0.1500,
    CreditRating.CC: 0.2500,
    CreditRating.C: 0.4000,
    CreditRating.D: 1.0000,        # 已违约
    CreditRating.NR: 0.0500,       # 未评级假设5%
}

# 评级对应的LGD (损失给定违约)
RATING_TO_LGD: Dict[CounterpartyType, float] = {
    CounterpartyType.BANK: 0.45,
    CounterpartyType.BROKER_DEALER: 0.50,
    CounterpartyType.HEDGE_FUND: 0.60,
    CounterpartyType.ASSET_MANAGER: 0.55,
    CounterpartyType.CORPORATE: 0.55,
    CounterpartyType.SOVEREIGN: 0.40,
    CounterpartyType.CLEARING_HOUSE: 0.30,
    CounterpartyType.EXCHANGE: 0.35,
    CounterpartyType.OTHER: 0.55,
}


# ============ Providers ============

class ExposureDataProvider(ABC):
    """敞口数据提供者接口"""

    @abstractmethod
    async def get_positions(self, counterparty_id: str) -> List[Dict[str, Any]]:
        """获取与交易对手相关的持仓"""
        pass

    @abstractmethod
    async def get_collateral(self, counterparty_id: str) -> float:
        """获取抵押品价值"""
        pass

    @abstractmethod
    async def get_netting_sets(self, counterparty_id: str) -> List[Dict[str, Any]]:
        """获取净额结算集"""
        pass


class MarketDataProvider(ABC):
    """市场数据提供者接口"""

    @abstractmethod
    async def get_cds_spread(self, counterparty_id: str) -> Optional[float]:
        """获取CDS价差(基点)"""
        pass

    @abstractmethod
    async def get_credit_rating(self, counterparty_id: str) -> Optional[CreditRating]:
        """获取最新信用评级"""
        pass


class SimulatedExposureProvider(ExposureDataProvider):
    """模拟敞口数据提供者"""

    def __init__(self):
        self._positions: Dict[str, List[Dict[str, Any]]] = {}
        self._collateral: Dict[str, float] = {}
        self._netting_sets: Dict[str, List[Dict[str, Any]]] = {}

    def set_positions(self, counterparty_id: str, positions: List[Dict[str, Any]]) -> None:
        self._positions[counterparty_id] = positions

    def set_collateral(self, counterparty_id: str, collateral: float) -> None:
        self._collateral[counterparty_id] = collateral

    def set_netting_sets(self, counterparty_id: str, netting_sets: List[Dict[str, Any]]) -> None:
        self._netting_sets[counterparty_id] = netting_sets

    async def get_positions(self, counterparty_id: str) -> List[Dict[str, Any]]:
        return self._positions.get(counterparty_id, [])

    async def get_collateral(self, counterparty_id: str) -> float:
        return self._collateral.get(counterparty_id, 0.0)

    async def get_netting_sets(self, counterparty_id: str) -> List[Dict[str, Any]]:
        return self._netting_sets.get(counterparty_id, [])


class SimulatedMarketDataProvider(MarketDataProvider):
    """模拟市场数据提供者"""

    def __init__(self):
        self._cds_spreads: Dict[str, float] = {}
        self._ratings: Dict[str, CreditRating] = {}

    def set_cds_spread(self, counterparty_id: str, spread: float) -> None:
        self._cds_spreads[counterparty_id] = spread

    def set_rating(self, counterparty_id: str, rating: CreditRating) -> None:
        self._ratings[counterparty_id] = rating

    async def get_cds_spread(self, counterparty_id: str) -> Optional[float]:
        return self._cds_spreads.get(counterparty_id)

    async def get_credit_rating(self, counterparty_id: str) -> Optional[CreditRating]:
        return self._ratings.get(counterparty_id)


# ============ Main Class ============

class CounterpartyRiskMonitor:
    """交易对手风险监控器"""

    def __init__(
        self,
        config: Optional[CounterpartyRiskConfig] = None,
        exposure_provider: Optional[ExposureDataProvider] = None,
        market_data_provider: Optional[MarketDataProvider] = None,
        event_callback: Optional[Callable[[CreditEvent], None]] = None,
        limit_callback: Optional[Callable[[CounterpartyLimit], None]] = None,
    ):
        self.config = config or CounterpartyRiskConfig()
        self.exposure_provider = exposure_provider or simulated_default(SimulatedExposureProvider, "data from CounterpartyRiskMonitor's simulated exposure_provider (SimulatedExposureProvider)")
        self.market_data_provider = market_data_provider or simulated_default(SimulatedMarketDataProvider, "data from CounterpartyRiskMonitor's simulated market_data_provider (SimulatedMarketDataProvider)")
        self.event_callback = event_callback
        self.limit_callback = limit_callback

        # 状态
        self._state = MonitorState.STOPPED
        self._monitor_task: Optional[asyncio.Task] = None

        # 数据存储
        self._counterparties: Dict[str, CounterpartyProfile] = {}
        self._exposures: Dict[str, Exposure] = {}
        self._limits: Dict[str, List[CounterpartyLimit]] = {}
        self._default_probs: Dict[str, DefaultProbability] = {}
        self._credit_vars: Dict[str, CreditVaR] = {}
        self._cva_results: Dict[str, CVAResult] = {}
        self._events: deque = deque(maxlen=self.config.history_size)
        self._cds_history: Dict[str, deque] = {}

    @property
    def state(self) -> MonitorState:
        return self._state

    # ============ Lifecycle ============

    async def start(self) -> None:
        """启动监控器"""
        if self._state == MonitorState.RUNNING:
            return

        self._state = MonitorState.STARTING
        logger.info("Starting Counterparty Risk Monitor")

        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._state = MonitorState.RUNNING
        logger.info("Counterparty Risk Monitor started")

    async def stop(self) -> None:
        """停止监控器"""
        if self._state == MonitorState.STOPPED:
            return

        logger.info("Stopping Counterparty Risk Monitor")
        self._state = MonitorState.STOPPED

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Counterparty Risk Monitor stopped")

    async def pause(self) -> None:
        """暂停监控器"""
        if self._state == MonitorState.RUNNING:
            self._state = MonitorState.PAUSED
            logger.info("Counterparty Risk Monitor paused")

    async def resume(self) -> None:
        """恢复监控器"""
        if self._state == MonitorState.PAUSED:
            self._state = MonitorState.RUNNING
            logger.info("Counterparty Risk Monitor resumed")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._state in (MonitorState.RUNNING, MonitorState.PAUSED):
            try:
                if self._state == MonitorState.RUNNING:
                    await self._update_all_metrics()
                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self._state = MonitorState.ERROR

    async def _update_all_metrics(self) -> None:
        """更新所有指标"""
        for cp_id in list(self._counterparties.keys()):
            try:
                await self.calculate_exposure(cp_id)
                await self.calculate_default_probability(cp_id)
                if self.config.enable_var:
                    await self.calculate_credit_var(cp_id)
                if self.config.enable_cva:
                    await self.calculate_cva(cp_id)
                await self._check_limits(cp_id)
                await self._check_credit_events(cp_id)
            except Exception as e:
                logger.error(f"Error updating metrics for {cp_id}: {e}")

    # ============ Counterparty Management ============

    def add_counterparty(self, profile: CounterpartyProfile) -> None:
        """添加交易对手"""
        self._counterparties[profile.counterparty_id] = profile
        self._cds_history[profile.counterparty_id] = deque(maxlen=100)
        logger.info(f"Added counterparty: {profile.counterparty_id}")

    def update_counterparty(self, profile: CounterpartyProfile) -> bool:
        """更新交易对手信息"""
        if profile.counterparty_id not in self._counterparties:
            return False

        old_profile = self._counterparties[profile.counterparty_id]

        # 检查评级变化
        if old_profile.credit_rating != profile.credit_rating:
            self._create_rating_event(
                profile.counterparty_id,
                old_profile.credit_rating,
                profile.credit_rating,
            )

        profile.updated_at = datetime.now()
        self._counterparties[profile.counterparty_id] = profile
        return True

    def remove_counterparty(self, counterparty_id: str) -> bool:
        """移除交易对手"""
        if counterparty_id not in self._counterparties:
            return False

        del self._counterparties[counterparty_id]
        self._exposures.pop(counterparty_id, None)
        self._limits.pop(counterparty_id, None)
        self._default_probs.pop(counterparty_id, None)
        self._credit_vars.pop(counterparty_id, None)
        self._cva_results.pop(counterparty_id, None)
        self._cds_history.pop(counterparty_id, None)
        logger.info(f"Removed counterparty: {counterparty_id}")
        return True

    def get_counterparty(self, counterparty_id: str) -> Optional[CounterpartyProfile]:
        """获取交易对手信息"""
        return self._counterparties.get(counterparty_id)

    def get_all_counterparties(self) -> List[CounterpartyProfile]:
        """获取所有交易对手"""
        return list(self._counterparties.values())

    # ============ Limit Management ============

    def set_limit(self, limit: CounterpartyLimit) -> None:
        """设置限额"""
        if limit.counterparty_id not in self._limits:
            self._limits[limit.counterparty_id] = []

        # 更新或添加限额
        limits = self._limits[limit.counterparty_id]
        for i, existing in enumerate(limits):
            if existing.limit_type == limit.limit_type:
                limits[i] = limit
                return
        limits.append(limit)
        logger.info(f"Set limit for {limit.counterparty_id}: {limit.limit_type} = {limit.limit_value}")

    def get_limits(self, counterparty_id: str) -> List[CounterpartyLimit]:
        """获取交易对手限额"""
        return self._limits.get(counterparty_id, [])

    def update_limit_usage(self, counterparty_id: str, limit_type: str, usage: float) -> bool:
        """更新限额使用量"""
        limits = self._limits.get(counterparty_id, [])
        for limit in limits:
            if limit.limit_type == limit_type:
                old_status = limit.status
                limit.current_usage = usage

                # 检查状态变化
                new_status = limit.status
                if new_status != old_status and new_status in ("warning", "critical", "breached"):
                    self._create_limit_event(limit)

                return True
        return False

    async def _check_limits(self, counterparty_id: str) -> None:
        """检查限额"""
        limits = self._limits.get(counterparty_id, [])
        exposure = self._exposures.get(counterparty_id)

        if not exposure:
            return

        for limit in limits:
            if limit.limit_type == "gross":
                limit.current_usage = exposure.gross_exposure
            elif limit.limit_type == "net":
                limit.current_usage = exposure.net_exposure
            elif limit.limit_type == "collateralized":
                limit.current_usage = exposure.collateralized_exposure

            if limit.status in ("warning", "critical", "breached"):
                self._create_limit_event(limit)

    # ============ Exposure Calculation ============

    async def calculate_exposure(self, counterparty_id: str) -> Optional[Exposure]:
        """计算交易对手敞口"""
        profile = self._counterparties.get(counterparty_id)
        if not profile:
            return None

        positions = await self.exposure_provider.get_positions(counterparty_id)
        collateral = await self.exposure_provider.get_collateral(counterparty_id)
        netting_sets = await self.exposure_provider.get_netting_sets(counterparty_id)

        # 计算总敞口
        gross_exposure = sum(
            abs(p.get("market_value", 0)) for p in positions
        )

        # 计算净额结算收益
        netting_benefit = 0.0
        for ns in netting_sets:
            long_value = sum(p.get("market_value", 0) for p in positions
                           if p.get("netting_set") == ns.get("set_id") and p.get("market_value", 0) > 0)
            short_value = abs(sum(p.get("market_value", 0) for p in positions
                                if p.get("netting_set") == ns.get("set_id") and p.get("market_value", 0) < 0))
            netting_benefit += min(long_value, short_value)

        net_exposure = gross_exposure - netting_benefit

        exposure = Exposure(
            exposure_id=str(uuid.uuid4())[:8],
            counterparty_id=counterparty_id,
            exposure_type=ExposureType.CURRENT,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            collateral=collateral,
            netting_benefit=netting_benefit,
            positions=positions,
        )

        self._exposures[counterparty_id] = exposure
        return exposure

    def get_exposure(self, counterparty_id: str) -> Optional[Exposure]:
        """获取交易对手敞口"""
        return self._exposures.get(counterparty_id)

    # ============ Default Probability ============

    async def calculate_default_probability(self, counterparty_id: str) -> Optional[DefaultProbability]:
        """计算违约概率"""
        profile = self._counterparties.get(counterparty_id)
        if not profile:
            return None

        # 从评级映射获取PD
        pd_1y = RATING_TO_PD.get(profile.credit_rating, 0.05)

        # 5年期PD (简化计算：累计)
        pd_5y = 1 - (1 - pd_1y) ** 5

        # 累计PD
        pd_cumulative = pd_5y  # 使用5年期

        # LGD
        lgd = RATING_TO_LGD.get(profile.counterparty_type, 0.55)
        recovery_rate = 1 - lgd

        # 预期损失
        exposure = self._exposures.get(counterparty_id)
        ead = exposure.net_exposure if exposure else 0
        expected_loss = pd_1y * lgd * ead

        dp = DefaultProbability(
            counterparty_id=counterparty_id,
            pd_1y=pd_1y,
            pd_5y=pd_5y,
            pd_cumulative=pd_cumulative,
            lgd=lgd,
            recovery_rate=recovery_rate,
            expected_loss=expected_loss,
        )

        self._default_probs[counterparty_id] = dp
        return dp

    def get_default_probability(self, counterparty_id: str) -> Optional[DefaultProbability]:
        """获取违约概率"""
        return self._default_probs.get(counterparty_id)

    # ============ Credit VaR ============

    async def calculate_credit_var(self, counterparty_id: str) -> Optional[CreditVaR]:
        """计算信用VaR"""
        profile = self._counterparties.get(counterparty_id)
        dp = self._default_probs.get(counterparty_id)
        exposure = self._exposures.get(counterparty_id)

        if not profile or not dp or not exposure:
            return None

        # 简化Monte Carlo VaR计算
        # 实际应用中应使用完整的Monte Carlo模拟
        ead = exposure.net_exposure
        pd = dp.pd_1y
        lgd = dp.lgd

        # 基于正态分布近似
        # VaR = EAD * LGD * N^-1(confidence) * sqrt(PD * (1-PD))
        # 使用预计算的标准正态分位数，避免依赖scipy
        z_95 = 1.645  # 95% confidence
        z_99 = 2.326  # 99% confidence

        volatility = math.sqrt(pd * (1 - pd))

        var_95 = ead * lgd * z_95 * volatility
        var_99 = ead * lgd * z_99 * volatility

        # Expected Shortfall (简化)
        es = var_99 * 1.1  # 近似

        credit_var = CreditVaR(
            counterparty_id=counterparty_id,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=es,
            time_horizon_days=self.config.var_horizon_days,
            confidence_level=self.config.var_confidence,
        )

        self._credit_vars[counterparty_id] = credit_var
        return credit_var

    def get_credit_var(self, counterparty_id: str) -> Optional[CreditVaR]:
        """获取信用VaR"""
        return self._credit_vars.get(counterparty_id)

    # ============ CVA Calculation ============

    async def calculate_cva(self, counterparty_id: str) -> Optional[CVAResult]:
        """计算CVA (Credit Valuation Adjustment)"""
        profile = self._counterparties.get(counterparty_id)
        dp = self._default_probs.get(counterparty_id)
        exposure = self._exposures.get(counterparty_id)

        if not profile or not dp or not exposure:
            return None

        # 简化CVA计算
        # CVA = (1 - R) * sum(DF_i * EE_i * PD_i)
        # 其中 R = 回收率, DF = 折现因子, EE = 预期敞口, PD = 边际违约概率

        lgd = dp.lgd
        pd_1y = dp.pd_1y
        ead = exposure.net_exposure

        # 简化：假设5年期，年度时间点
        time_points = [1, 2, 3, 4, 5]
        discount_rate = 0.03  # 假设3%折现率

        discount_factors = [1 / (1 + discount_rate) ** t for t in time_points]
        survival_probs = [(1 - pd_1y) ** t for t in time_points]
        marginal_pds = [
            survival_probs[i-1] - survival_probs[i] if i > 0
            else 1 - survival_probs[0]
            for i in range(len(time_points))
        ]

        # 简化：假设敞口曲线平坦
        exposure_profile = [ead * 0.9 ** t for t in time_points]  # 逐年递减

        # CVA计算
        cva = lgd * sum(
            df * ee * mpd
            for df, ee, mpd in zip(discount_factors, exposure_profile, marginal_pds)
        )

        # DVA (Debit Valuation Adjustment) - 己方信用风险
        # 简化：假设己方评级为A
        own_pd = 0.0008
        own_lgd = 0.55
        dva = own_lgd * sum(
            df * ee * own_pd
            for df, ee in zip(discount_factors, exposure_profile)
        )

        # FVA (Funding Valuation Adjustment)
        funding_spread = 0.01  # 1% funding spread
        fva = funding_spread * sum(
            df * ee for df, ee in zip(discount_factors, exposure_profile)
        ) / len(time_points)

        bilateral_cva = cva - dva
        total_xva = cva + fva  # 实际应用可能还有MVA, KVA等

        cva_result = CVAResult(
            counterparty_id=counterparty_id,
            cva=cva,
            dva=dva,
            bilateral_cva=bilateral_cva,
            fva=fva,
            total_xva=total_xva,
            exposure_profile=exposure_profile,
            discount_factors=discount_factors,
            survival_probs=survival_probs,
        )

        self._cva_results[counterparty_id] = cva_result
        return cva_result

    def get_cva(self, counterparty_id: str) -> Optional[CVAResult]:
        """获取CVA结果"""
        return self._cva_results.get(counterparty_id)

    # ============ Concentration Analysis ============

    def calculate_concentration(self) -> ConcentrationMetrics:
        """计算集中度指标"""
        exposures = []
        for cp_id, exp in self._exposures.items():
            exposures.append((cp_id, exp.net_exposure))

        if not exposures:
            return ConcentrationMetrics(
                total_exposure=0,
                top_counterparty_exposure=0,
                top_5_exposure=0,
                top_10_exposure=0,
                herfindahl_index=0,
                largest_counterparty_id="",
                largest_counterparty_pct=0,
                counterparty_count=0,
            )

        # 按敞口排序
        exposures.sort(key=lambda x: x[1], reverse=True)

        total = sum(e[1] for e in exposures)

        if total == 0:
            return ConcentrationMetrics(
                total_exposure=0,
                top_counterparty_exposure=exposures[0][1] if exposures else 0,
                top_5_exposure=0,
                top_10_exposure=0,
                herfindahl_index=0,
                largest_counterparty_id=exposures[0][0] if exposures else "",
                largest_counterparty_pct=0,
                counterparty_count=len(exposures),
            )

        top_5 = sum(e[1] for e in exposures[:5])
        top_10 = sum(e[1] for e in exposures[:10])

        # Herfindahl Index = sum((exposure_i / total)^2)
        hhi = sum((e[1] / total) ** 2 for e in exposures)

        return ConcentrationMetrics(
            total_exposure=total,
            top_counterparty_exposure=exposures[0][1],
            top_5_exposure=top_5,
            top_10_exposure=top_10,
            herfindahl_index=hhi,
            largest_counterparty_id=exposures[0][0],
            largest_counterparty_pct=exposures[0][1] / total if total > 0 else 0,
            counterparty_count=len(exposures),
        )

    # ============ Event Management ============

    def _create_rating_event(
        self,
        counterparty_id: str,
        old_rating: CreditRating,
        new_rating: CreditRating,
    ) -> None:
        """创建评级变动事件"""
        # 判断是升级还是降级
        ratings_order = list(CreditRating)
        old_idx = ratings_order.index(old_rating)
        new_idx = ratings_order.index(new_rating)

        if new_idx > old_idx:  # 降级 (评级越往后越差)
            event_type = CreditEventType.RATING_DOWNGRADE
            severity = AlertSeverity.CRITICAL if new_idx - old_idx >= 3 else AlertSeverity.WARNING
            description = f"Rating downgraded from {old_rating.value} to {new_rating.value}"
        else:
            event_type = CreditEventType.RATING_UPGRADE
            severity = AlertSeverity.INFO
            description = f"Rating upgraded from {old_rating.value} to {new_rating.value}"

        event = CreditEvent(
            event_id=str(uuid.uuid4())[:8],
            counterparty_id=counterparty_id,
            event_type=event_type,
            severity=severity,
            description=description,
            old_value=old_rating.value,
            new_value=new_rating.value,
        )

        self._events.append(event)

        if self.event_callback:
            self.event_callback(event)

    def _create_limit_event(self, limit: CounterpartyLimit) -> None:
        """创建限额事件"""
        if limit.status == "breached":
            severity = AlertSeverity.EMERGENCY
        elif limit.status == "critical":
            severity = AlertSeverity.CRITICAL
        else:
            severity = AlertSeverity.WARNING

        event = CreditEvent(
            event_id=str(uuid.uuid4())[:8],
            counterparty_id=limit.counterparty_id,
            event_type=CreditEventType.LIMIT_BREACH,
            severity=severity,
            description=f"Limit {limit.limit_type} at {limit.utilization:.1%} (status: {limit.status})",
            old_value=str(limit.limit_value),
            new_value=str(limit.current_usage),
        )

        self._events.append(event)

        if self.limit_callback:
            self.limit_callback(limit)

    async def _check_credit_events(self, counterparty_id: str) -> None:
        """检查信用事件"""
        # 检查CDS价差变动
        cds_spread = await self.market_data_provider.get_cds_spread(counterparty_id)
        if cds_spread is not None:
            history = self._cds_history.get(counterparty_id, deque(maxlen=100))

            if history:
                avg_spread = statistics.mean(history)
                if avg_spread > 0:
                    change_pct = (cds_spread - avg_spread) / avg_spread
                    if change_pct > self.config.cds_spike_threshold:
                        self._create_cds_spike_event(counterparty_id, avg_spread, cds_spread)

            history.append(cds_spread)
            self._cds_history[counterparty_id] = history

        # 检查集中度
        concentration = self.calculate_concentration()
        if concentration.largest_counterparty_pct > self.config.concentration_critical:
            self._create_concentration_event(counterparty_id, concentration, "critical")
        elif concentration.largest_counterparty_pct > self.config.concentration_warning:
            self._create_concentration_event(counterparty_id, concentration, "warning")

    def _create_cds_spike_event(
        self,
        counterparty_id: str,
        old_spread: float,
        new_spread: float,
    ) -> None:
        """创建CDS价差飙升事件"""
        event = CreditEvent(
            event_id=str(uuid.uuid4())[:8],
            counterparty_id=counterparty_id,
            event_type=CreditEventType.CDS_SPIKE,
            severity=AlertSeverity.WARNING,
            description=f"CDS spread spiked from {old_spread:.0f}bps to {new_spread:.0f}bps",
            old_value=f"{old_spread:.0f}",
            new_value=f"{new_spread:.0f}",
        )

        self._events.append(event)

        if self.event_callback:
            self.event_callback(event)

    def _create_concentration_event(
        self,
        counterparty_id: str,
        concentration: ConcentrationMetrics,
        level: str,
    ) -> None:
        """创建集中度警告事件"""
        severity = AlertSeverity.CRITICAL if level == "critical" else AlertSeverity.WARNING

        event = CreditEvent(
            event_id=str(uuid.uuid4())[:8],
            counterparty_id=concentration.largest_counterparty_id,
            event_type=CreditEventType.CONCENTRATION_WARNING,
            severity=severity,
            description=f"Concentration warning: {concentration.largest_counterparty_id} at {concentration.largest_counterparty_pct:.1%}",
            new_value=f"{concentration.largest_counterparty_pct:.1%}",
        )

        self._events.append(event)

        if self.event_callback:
            self.event_callback(event)

    def get_events(
        self,
        counterparty_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[CreditEvent]:
        """获取信用事件"""
        events = list(self._events)

        if counterparty_id:
            events = [e for e in events if e.counterparty_id == counterparty_id]
        if event_type:
            events = [e for e in events if e.event_type.value == event_type]
        if severity:
            events = [e for e in events if e.severity.value == severity]
        if acknowledged is not None:
            events = [e for e in events if e.acknowledged == acknowledged]

        return events[-limit:]

    def acknowledge_event(self, event_id: str, user: str) -> bool:
        """确认事件"""
        for event in self._events:
            if event.event_id == event_id:
                event.acknowledged = True
                event.acknowledged_by = user
                event.acknowledged_at = datetime.now()
                return True
        return False

    # ============ Risk Report ============

    async def generate_report(self, counterparty_id: str) -> Optional[CounterpartyRiskReport]:
        """生成交易对手风险报告"""
        profile = self._counterparties.get(counterparty_id)
        if not profile:
            return None

        # 确保最新数据
        exposure = await self.calculate_exposure(counterparty_id)
        dp = await self.calculate_default_probability(counterparty_id)
        credit_var = await self.calculate_credit_var(counterparty_id) if self.config.enable_var else None
        cva = await self.calculate_cva(counterparty_id) if self.config.enable_cva else None

        limits = self.get_limits(counterparty_id)
        events = self.get_events(counterparty_id=counterparty_id, limit=10)

        # 计算风险评分
        risk_score = self._calculate_risk_score(profile, exposure, dp, limits)
        risk_level = self._determine_risk_level(risk_score)
        recommendations = self._generate_recommendations(profile, exposure, dp, limits, risk_level)

        report = CounterpartyRiskReport(
            report_id=str(uuid.uuid4())[:8],
            counterparty_id=counterparty_id,
            profile=profile,
            current_exposure=exposure,
            default_probability=dp,
            credit_var=credit_var,
            cva_result=cva,
            limits=limits,
            recent_events=events,
            risk_score=risk_score,
            risk_level=risk_level,
            recommendations=recommendations,
        )

        return report

    def _calculate_risk_score(
        self,
        profile: CounterpartyProfile,
        exposure: Optional[Exposure],
        dp: Optional[DefaultProbability],
        limits: List[CounterpartyLimit],
    ) -> float:
        """计算风险评分 (0-100, 越高风险越大)"""
        score = 0.0

        # 评级得分 (0-40)
        rating_scores = {
            CreditRating.AAA: 0, CreditRating.AA_PLUS: 2, CreditRating.AA: 4,
            CreditRating.AA_MINUS: 6, CreditRating.A_PLUS: 8, CreditRating.A: 10,
            CreditRating.A_MINUS: 12, CreditRating.BBB_PLUS: 15, CreditRating.BBB: 18,
            CreditRating.BBB_MINUS: 22, CreditRating.BB_PLUS: 26, CreditRating.BB: 30,
            CreditRating.BB_MINUS: 34, CreditRating.B_PLUS: 38, CreditRating.B: 42,
            CreditRating.B_MINUS: 50, CreditRating.CCC: 65, CreditRating.CC: 80,
            CreditRating.C: 90, CreditRating.D: 100, CreditRating.NR: 40,
        }
        score += rating_scores.get(profile.credit_rating, 40) * 0.4

        # 敞口得分 (0-30)
        if exposure and dp and dp.expected_loss > 0:
            # 预期损失占敞口比例
            el_ratio = dp.expected_loss / max(exposure.net_exposure, 1)
            score += min(30, el_ratio * 300)

        # 限额使用得分 (0-30)
        max_utilization = 0.0
        for limit in limits:
            max_utilization = max(max_utilization, limit.utilization)
        score += max_utilization * 30

        return min(100, score)

    def _determine_risk_level(self, score: float) -> str:
        """确定风险等级"""
        if score >= 75:
            return "critical"
        elif score >= 50:
            return "high"
        elif score >= 25:
            return "medium"
        return "low"

    def _generate_recommendations(
        self,
        profile: CounterpartyProfile,
        exposure: Optional[Exposure],
        dp: Optional[DefaultProbability],
        limits: List[CounterpartyLimit],
        risk_level: str,
    ) -> List[str]:
        """生成建议"""
        recommendations = []

        if risk_level == "critical":
            recommendations.append("Consider reducing exposure immediately")
            recommendations.append("Review collateral requirements")
            recommendations.append("Escalate to senior management")

        if profile.credit_rating.value.startswith("B") or profile.credit_rating in (CreditRating.CCC, CreditRating.CC, CreditRating.C):
            recommendations.append("Counterparty is below investment grade - review risk appetite")

        if exposure and exposure.collateral < exposure.net_exposure * 0.3:
            recommendations.append("Consider requesting additional collateral")

        for limit in limits:
            if limit.status == "breached":
                recommendations.append(f"Limit {limit.limit_type} breached - immediate action required")
            elif limit.status == "critical":
                recommendations.append(f"Limit {limit.limit_type} near breach - monitor closely")

        if not recommendations:
            recommendations.append("No immediate actions required - continue monitoring")

        return recommendations

    # ============ Statistics ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        concentration = self.calculate_concentration()

        return {
            "state": self._state.value,
            "counterparty_count": len(self._counterparties),
            "total_exposure": concentration.total_exposure,
            "total_events": len(self._events),
            "unacknowledged_events": len([e for e in self._events if not e.acknowledged]),
            "concentration": concentration.to_dict(),
            "config": self.config.to_dict(),
        }


# ============ Factory Functions ============

def create_counterparty_monitor(
    exposure_provider: Optional[ExposureDataProvider] = None,
    market_data_provider: Optional[MarketDataProvider] = None,
    **config_kwargs,
) -> CounterpartyRiskMonitor:
    """创建交易对手风险监控器"""
    config = CounterpartyRiskConfig(**config_kwargs)
    return CounterpartyRiskMonitor(
        config=config,
        exposure_provider=exposure_provider,
        market_data_provider=market_data_provider,
    )


def create_counterparty_profile(
    counterparty_id: str,
    name: str,
    counterparty_type: str,
    credit_rating: str,
    **kwargs,
) -> CounterpartyProfile:
    """创建交易对手档案"""
    return CounterpartyProfile(
        counterparty_id=counterparty_id,
        name=name,
        counterparty_type=CounterpartyType(counterparty_type),
        credit_rating=CreditRating(credit_rating),
        **kwargs,
    )


def create_counterparty_limit(
    counterparty_id: str,
    limit_type: str,
    limit_value: float,
    **kwargs,
) -> CounterpartyLimit:
    """创建交易对手限额"""
    return CounterpartyLimit(
        limit_id=str(uuid.uuid4())[:8],
        counterparty_id=counterparty_id,
        limit_type=limit_type,
        limit_value=limit_value,
        **kwargs,
    )


async def quick_counterparty_check(
    monitor: CounterpartyRiskMonitor,
    counterparty_id: str,
) -> Dict[str, Any]:
    """快速交易对手风险检查"""
    profile = monitor.get_counterparty(counterparty_id)
    if not profile:
        return {"error": f"Counterparty not found: {counterparty_id}"}

    exposure = await monitor.calculate_exposure(counterparty_id)
    dp = await monitor.calculate_default_probability(counterparty_id)
    limits = monitor.get_limits(counterparty_id)

    return {
        "counterparty_id": counterparty_id,
        "name": profile.name,
        "rating": profile.credit_rating.value,
        "exposure": exposure.net_exposure if exposure else 0,
        "pd_1y": dp.pd_1y if dp else 0,
        "expected_loss": dp.expected_loss if dp else 0,
        "limit_status": [
            {"type": l.limit_type, "status": l.status, "utilization": l.utilization}
            for l in limits
        ],
    }


# ============ Exports ============

__all__ = [
    # Enums
    "CreditRating",
    "CounterpartyType",
    "ExposureType",
    "CreditEventType",
    "AlertSeverity",
    "MonitorState",
    # Data Classes
    "CounterpartyProfile",
    "DefaultProbability",
    "Exposure",
    "CounterpartyLimit",
    "CreditVaR",
    "CVAResult",
    "CreditEvent",
    "ConcentrationMetrics",
    "CounterpartyRiskReport",
    "CounterpartyRiskConfig",
    # Providers
    "ExposureDataProvider",
    "MarketDataProvider",
    "SimulatedExposureProvider",
    "SimulatedMarketDataProvider",
    # Main Class
    "CounterpartyRiskMonitor",
    # Factory Functions
    "create_counterparty_monitor",
    "create_counterparty_profile",
    "create_counterparty_limit",
    "quick_counterparty_check",
    # Constants
    "RATING_TO_PD",
    "RATING_TO_LGD",
]
