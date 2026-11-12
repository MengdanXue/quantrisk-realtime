"""
风控限额引擎模块
Risk Limits Engine

提供多层次风控限额管理、实时监控和事前拦截功能。

功能特性:
- 多层次限额体系 (交易员/策略/账户/公司级)
- 多维度限额类型 (敞口/VaR/集中度/杠杆/日亏损)
- 实时限额监控与预警
- Pre-Trade风险检查
- 限额动态调整
- 限额突破审批流程
- 完整审计日志

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class LimitType(Enum):
    """限额类型"""
    # 敞口限额
    GROSS_EXPOSURE = "gross_exposure"  # 总敞口
    NET_EXPOSURE = "net_exposure"  # 净敞口
    LONG_EXPOSURE = "long_exposure"  # 多头敞口
    SHORT_EXPOSURE = "short_exposure"  # 空头敞口

    # 风险限额
    VAR_LIMIT = "var_limit"  # VaR限额
    CVAR_LIMIT = "cvar_limit"  # CVaR限额
    DRAWDOWN_LIMIT = "drawdown_limit"  # 回撤限额

    # 集中度限额
    SINGLE_POSITION = "single_position"  # 单一持仓
    SECTOR_CONCENTRATION = "sector_concentration"  # 行业集中度
    ASSET_CLASS_CONCENTRATION = "asset_class_concentration"  # 资产类别集中度
    COUNTRY_CONCENTRATION = "country_concentration"  # 国家集中度

    # 杠杆限额
    LEVERAGE_RATIO = "leverage_ratio"  # 杠杆比率
    MARGIN_UTILIZATION = "margin_utilization"  # 保证金利用率

    # 交易限额
    DAILY_LOSS = "daily_loss"  # 日亏损限额
    DAILY_TURNOVER = "daily_turnover"  # 日交易额限额
    ORDER_SIZE = "order_size"  # 单笔订单大小
    DAILY_TRADES = "daily_trades"  # 日交易次数

    # 流动性限额
    ADV_LIMIT = "adv_limit"  # 日均成交量限额
    MARKET_IMPACT = "market_impact"  # 市场冲击限额


class LimitLevel(Enum):
    """限额级别"""
    TRADER = "trader"  # 交易员级
    STRATEGY = "strategy"  # 策略级
    PORTFOLIO = "portfolio"  # 组合级
    ACCOUNT = "account"  # 账户级
    DESK = "desk"  # 交易台级
    FIRM = "firm"  # 公司级


class LimitStatus(Enum):
    """限额状态"""
    NORMAL = "normal"  # 正常
    WARNING = "warning"  # 预警 (80%)
    CRITICAL = "critical"  # 严重 (90%)
    BREACHED = "breached"  # 突破 (100%)
    SUSPENDED = "suspended"  # 暂停
    EXPIRED = "expired"  # 过期


class CheckResult(Enum):
    """检查结果"""
    PASSED = "passed"  # 通过
    SOFT_REJECT = "soft_reject"  # 软拒绝（需审批）
    HARD_REJECT = "hard_reject"  # 硬拒绝


class ApprovalStatus(Enum):
    """审批状态"""
    PENDING = "pending"  # 待审批
    APPROVED = "approved"  # 已批准
    REJECTED = "rejected"  # 已拒绝
    EXPIRED = "expired"  # 已过期


class EngineState(Enum):
    """引擎状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ==================== 数据结构 ====================

@dataclass
class RiskLimit:
    """风控限额"""
    limit_id: str
    limit_type: LimitType
    limit_level: LimitLevel
    entity_id: str  # 交易员ID/策略ID/账户ID等
    limit_value: Decimal
    warning_threshold: Decimal = Decimal("0.80")  # 80%预警
    critical_threshold: Decimal = Decimal("0.90")  # 90%严重
    current_usage: Decimal = Decimal("0")
    currency: str = "USD"
    effective_date: date = field(default_factory=date.today)
    expiry_date: Optional[date] = None
    is_hard_limit: bool = True  # 硬限制不可突破
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def usage_ratio(self) -> Decimal:
        """使用率"""
        if self.limit_value == 0:
            return Decimal("0")
        return self.current_usage / self.limit_value

    @property
    def status(self) -> LimitStatus:
        """限额状态"""
        if self.expiry_date and date.today() > self.expiry_date:
            return LimitStatus.EXPIRED
        ratio = self.usage_ratio
        if ratio >= Decimal("1"):
            return LimitStatus.BREACHED
        elif ratio >= self.critical_threshold:
            return LimitStatus.CRITICAL
        elif ratio >= self.warning_threshold:
            return LimitStatus.WARNING
        return LimitStatus.NORMAL

    @property
    def available(self) -> Decimal:
        """可用额度"""
        return max(Decimal("0"), self.limit_value - self.current_usage)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "limit_id": self.limit_id,
            "limit_type": self.limit_type.value,
            "limit_level": self.limit_level.value,
            "entity_id": self.entity_id,
            "limit_value": float(self.limit_value),
            "warning_threshold": float(self.warning_threshold),
            "critical_threshold": float(self.critical_threshold),
            "current_usage": float(self.current_usage),
            "usage_ratio": float(self.usage_ratio),
            "available": float(self.available),
            "status": self.status.value,
            "currency": self.currency,
            "effective_date": self.effective_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "is_hard_limit": self.is_hard_limit,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class LimitCheckRequest:
    """限额检查请求"""
    order_id: str
    entity_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    price: Decimal
    order_type: str = "limit"
    account_id: Optional[str] = None
    strategy_id: Optional[str] = None
    trader_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> Decimal:
        """名义金额"""
        return self.quantity * self.price

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "order_id": self.order_id,
            "entity_id": self.entity_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "price": float(self.price),
            "notional": float(self.notional),
            "order_type": self.order_type,
            "account_id": self.account_id,
            "strategy_id": self.strategy_id,
            "trader_id": self.trader_id,
            "portfolio_id": self.portfolio_id,
        }


@dataclass
class LimitCheckResult:
    """限额检查结果"""
    check_id: str
    order_id: str
    result: CheckResult
    checked_limits: List[Dict[str, Any]]
    violated_limits: List[Dict[str, Any]]
    warnings: List[str]
    approval_required: bool = False
    approval_id: Optional[str] = None
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "check_id": self.check_id,
            "order_id": self.order_id,
            "result": self.result.value,
            "checked_limits": self.checked_limits,
            "violated_limits": self.violated_limits,
            "warnings": self.warnings,
            "approval_required": self.approval_required,
            "approval_id": self.approval_id,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LimitBreach:
    """限额突破记录"""
    breach_id: str
    limit_id: str
    limit_type: LimitType
    entity_id: str
    limit_value: Decimal
    actual_value: Decimal
    breach_ratio: Decimal
    order_id: Optional[str]
    breach_time: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_time: Optional[datetime] = None
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "breach_id": self.breach_id,
            "limit_id": self.limit_id,
            "limit_type": self.limit_type.value,
            "entity_id": self.entity_id,
            "limit_value": float(self.limit_value),
            "actual_value": float(self.actual_value),
            "breach_ratio": float(self.breach_ratio),
            "order_id": self.order_id,
            "breach_time": self.breach_time.isoformat(),
            "resolved": self.resolved,
            "resolved_time": self.resolved_time.isoformat() if self.resolved_time else None,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class ApprovalRequest:
    """审批请求"""
    approval_id: str
    order_id: str
    entity_id: str
    violated_limits: List[str]
    requested_by: str
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approval_notes: str = ""
    requested_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "approval_id": self.approval_id,
            "order_id": self.order_id,
            "entity_id": self.entity_id,
            "violated_limits": self.violated_limits,
            "requested_by": self.requested_by,
            "reason": self.reason,
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approval_notes": self.approval_notes,
            "requested_at": self.requested_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class LimitAlert:
    """限额预警"""
    alert_id: str
    limit_id: str
    limit_type: LimitType
    entity_id: str
    alert_level: str  # "warning", "critical", "breach"
    usage_ratio: Decimal
    message: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "alert_id": self.alert_id,
            "limit_id": self.limit_id,
            "limit_type": self.limit_type.value,
            "entity_id": self.entity_id,
            "alert_level": self.alert_level,
            "usage_ratio": float(self.usage_ratio),
            "message": self.message,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EngineConfig:
    """引擎配置"""
    check_interval: float = 60.0  # 检查间隔（秒）
    enable_pre_trade_check: bool = True
    enable_real_time_monitoring: bool = True
    alert_cooldown: float = 300.0  # 告警冷却时间（秒）
    max_approval_wait: float = 3600.0  # 最大审批等待时间（秒）
    auto_suspend_on_breach: bool = True  # 突破时自动暂停
    cascade_limits: bool = True  # 级联限额检查


# ==================== 数据提供者接口 ====================

class PositionDataProvider(ABC):
    """持仓数据提供者接口"""

    @abstractmethod
    async def get_positions(self, entity_id: str) -> List[Dict[str, Any]]:
        """获取持仓"""
        pass

    @abstractmethod
    async def get_gross_exposure(self, entity_id: str) -> Decimal:
        """获取总敞口"""
        pass

    @abstractmethod
    async def get_net_exposure(self, entity_id: str) -> Decimal:
        """获取净敞口"""
        pass


class RiskMetricsProvider(ABC):
    """风险指标提供者接口"""

    @abstractmethod
    async def get_var(self, entity_id: str, confidence: float = 0.95) -> Decimal:
        """获取VaR"""
        pass

    @abstractmethod
    async def get_cvar(self, entity_id: str, confidence: float = 0.95) -> Decimal:
        """获取CVaR"""
        pass

    @abstractmethod
    async def get_drawdown(self, entity_id: str) -> Decimal:
        """获取回撤"""
        pass


# ==================== 模拟实现 ====================

class SimulatedPositionProvider(PositionDataProvider):
    """模拟持仓数据提供者"""

    def __init__(self):
        self.positions: Dict[str, List[Dict[str, Any]]] = {}
        self.exposures: Dict[str, Dict[str, Decimal]] = {}

    async def get_positions(self, entity_id: str) -> List[Dict[str, Any]]:
        return self.positions.get(entity_id, [])

    async def get_gross_exposure(self, entity_id: str) -> Decimal:
        return self.exposures.get(entity_id, {}).get("gross", Decimal("0"))

    async def get_net_exposure(self, entity_id: str) -> Decimal:
        return self.exposures.get(entity_id, {}).get("net", Decimal("0"))

    def set_exposure(self, entity_id: str, gross: Decimal, net: Decimal):
        """设置敞口（测试用）"""
        self.exposures[entity_id] = {"gross": gross, "net": net}

    def add_position(self, entity_id: str, position: Dict[str, Any]):
        """添加持仓（测试用）"""
        if entity_id not in self.positions:
            self.positions[entity_id] = []
        self.positions[entity_id].append(position)


class SimulatedRiskMetricsProvider(RiskMetricsProvider):
    """模拟风险指标提供者"""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Decimal]] = {}

    async def get_var(self, entity_id: str, confidence: float = 0.95) -> Decimal:
        return self.metrics.get(entity_id, {}).get("var", Decimal("0"))

    async def get_cvar(self, entity_id: str, confidence: float = 0.95) -> Decimal:
        return self.metrics.get(entity_id, {}).get("cvar", Decimal("0"))

    async def get_drawdown(self, entity_id: str) -> Decimal:
        return self.metrics.get(entity_id, {}).get("drawdown", Decimal("0"))

    def set_metrics(self, entity_id: str, var: Decimal, cvar: Decimal, drawdown: Decimal):
        """设置指标（测试用）"""
        self.metrics[entity_id] = {"var": var, "cvar": cvar, "drawdown": drawdown}


# ==================== 核心引擎 ====================

class RiskLimitsEngine:
    """风控限额引擎"""

    def __init__(
        self,
        position_provider: Optional[PositionDataProvider] = None,
        risk_provider: Optional[RiskMetricsProvider] = None,
        config: Optional[EngineConfig] = None,
    ):
        self.position_provider = position_provider or simulated_default(SimulatedPositionProvider, "data from RiskLimitsEngine's simulated position_provider (SimulatedPositionProvider)")
        self.risk_provider = risk_provider or simulated_default(SimulatedRiskMetricsProvider, "data from RiskLimitsEngine's simulated risk_provider (SimulatedRiskMetricsProvider)")
        self.config = config or EngineConfig()

        # 限额存储
        self.limits: Dict[str, RiskLimit] = {}
        self.limits_by_entity: Dict[str, List[str]] = {}
        self.limits_by_type: Dict[LimitType, List[str]] = {}

        # 状态
        self.state = EngineState.IDLE
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # 历史记录
        self.breaches: List[LimitBreach] = []
        self.alerts: List[LimitAlert] = []
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.check_history: List[LimitCheckResult] = []

        # 回调
        self._on_alert: Optional[Callable[[LimitAlert], None]] = None
        self._on_breach: Optional[Callable[[LimitBreach], None]] = None
        self._on_approval_required: Optional[Callable[[ApprovalRequest], None]] = None

        # 统计
        self._stats = {
            "total_checks": 0,
            "passed_checks": 0,
            "rejected_checks": 0,
            "total_breaches": 0,
            "total_alerts": 0,
            "pending_approvals": 0,
        }

        # 告警冷却
        self._alert_cooldown: Dict[str, datetime] = {}

        # 日内计数器
        self._daily_counters: Dict[str, Dict[str, Any]] = {}
        self._counter_date: date = date.today()

    # ==================== 回调注册 ====================

    def on_alert(self, callback: Callable[[LimitAlert], None]):
        """注册告警回调"""
        self._on_alert = callback

    def on_breach(self, callback: Callable[[LimitBreach], None]):
        """注册突破回调"""
        self._on_breach = callback

    def on_approval_required(self, callback: Callable[[ApprovalRequest], None]):
        """注册审批回调"""
        self._on_approval_required = callback

    # ==================== 限额管理 ====================

    def add_limit(self, limit: RiskLimit) -> str:
        """添加限额"""
        self.limits[limit.limit_id] = limit

        # 按实体索引
        if limit.entity_id not in self.limits_by_entity:
            self.limits_by_entity[limit.entity_id] = []
        self.limits_by_entity[limit.entity_id].append(limit.limit_id)

        # 按类型索引
        if limit.limit_type not in self.limits_by_type:
            self.limits_by_type[limit.limit_type] = []
        self.limits_by_type[limit.limit_type].append(limit.limit_id)

        logger.info(f"Added limit: {limit.limit_id} - {limit.limit_type.value} for {limit.entity_id}")
        return limit.limit_id

    def remove_limit(self, limit_id: str) -> bool:
        """移除限额"""
        if limit_id not in self.limits:
            return False

        limit = self.limits[limit_id]

        # 从索引中移除
        if limit.entity_id in self.limits_by_entity:
            self.limits_by_entity[limit.entity_id].remove(limit_id)
        if limit.limit_type in self.limits_by_type:
            self.limits_by_type[limit.limit_type].remove(limit_id)

        del self.limits[limit_id]
        logger.info(f"Removed limit: {limit_id}")
        return True

    def get_limit(self, limit_id: str) -> Optional[RiskLimit]:
        """获取限额"""
        return self.limits.get(limit_id)

    def get_limits_by_entity(self, entity_id: str) -> List[RiskLimit]:
        """获取实体的所有限额"""
        limit_ids = self.limits_by_entity.get(entity_id, [])
        return [self.limits[lid] for lid in limit_ids if lid in self.limits]

    def get_limits_by_type(self, limit_type: LimitType) -> List[RiskLimit]:
        """获取某类型的所有限额"""
        limit_ids = self.limits_by_type.get(limit_type, [])
        return [self.limits[lid] for lid in limit_ids if lid in self.limits]

    def update_limit(
        self,
        limit_id: str,
        limit_value: Optional[Decimal] = None,
        warning_threshold: Optional[Decimal] = None,
        critical_threshold: Optional[Decimal] = None,
    ) -> bool:
        """更新限额"""
        if limit_id not in self.limits:
            return False

        limit = self.limits[limit_id]
        if limit_value is not None:
            limit.limit_value = limit_value
        if warning_threshold is not None:
            limit.warning_threshold = warning_threshold
        if critical_threshold is not None:
            limit.critical_threshold = critical_threshold
        limit.updated_at = datetime.now()

        logger.info(f"Updated limit: {limit_id}")
        return True

    def update_usage(self, limit_id: str, usage: Decimal) -> bool:
        """更新使用量"""
        if limit_id not in self.limits:
            return False

        limit = self.limits[limit_id]
        old_status = limit.status
        limit.current_usage = usage
        limit.updated_at = datetime.now()

        new_status = limit.status

        # 状态变化检测
        if new_status != old_status:
            self._handle_status_change(limit, old_status, new_status)

        return True

    def _handle_status_change(
        self,
        limit: RiskLimit,
        old_status: LimitStatus,
        new_status: LimitStatus,
    ):
        """处理状态变化"""
        # 生成告警
        if new_status in [LimitStatus.WARNING, LimitStatus.CRITICAL, LimitStatus.BREACHED]:
            self._generate_alert(limit, new_status.value)

        # 记录突破
        if new_status == LimitStatus.BREACHED:
            self._record_breach(limit, None)

    # ==================== Pre-Trade 检查 ====================

    async def check_order(self, request: LimitCheckRequest) -> LimitCheckResult:
        """检查订单"""
        check_id = str(uuid.uuid4())[:8]
        checked_limits = []
        violated_limits = []
        warnings = []

        self._stats["total_checks"] += 1

        # 重置日内计数器
        self._reset_daily_counters_if_needed()

        # 获取所有相关限额
        entities_to_check = self._get_entities_to_check(request)

        for entity_id in entities_to_check:
            limits = self.get_limits_by_entity(entity_id)

            for limit in limits:
                check_result = await self._check_single_limit(limit, request)
                checked_limits.append({
                    "limit_id": limit.limit_id,
                    "limit_type": limit.limit_type.value,
                    "limit_value": float(limit.limit_value),
                    "current_usage": float(limit.current_usage),
                    "projected_usage": float(check_result["projected_usage"]),
                    "status": check_result["status"],
                })

                if check_result["status"] == "violated":
                    violated_limits.append({
                        "limit_id": limit.limit_id,
                        "limit_type": limit.limit_type.value,
                        "is_hard_limit": limit.is_hard_limit,
                        "excess": float(check_result["excess"]),
                    })
                elif check_result["status"] == "warning":
                    warnings.append(
                        f"Limit {limit.limit_id} ({limit.limit_type.value}) at "
                        f"{check_result['projected_ratio']:.1%} utilization"
                    )

        # 确定结果
        result = CheckResult.PASSED
        approval_required = False
        approval_id = None

        if violated_limits:
            hard_violations = [v for v in violated_limits if v["is_hard_limit"]]
            soft_violations = [v for v in violated_limits if not v["is_hard_limit"]]

            if hard_violations:
                result = CheckResult.HARD_REJECT
                self._stats["rejected_checks"] += 1
            elif soft_violations:
                result = CheckResult.SOFT_REJECT
                approval_required = True
                approval_id = self._create_approval_request(request, soft_violations)
        else:
            self._stats["passed_checks"] += 1

        message = self._generate_check_message(result, violated_limits, warnings)

        check_result = LimitCheckResult(
            check_id=check_id,
            order_id=request.order_id,
            result=result,
            checked_limits=checked_limits,
            violated_limits=violated_limits,
            warnings=warnings,
            approval_required=approval_required,
            approval_id=approval_id,
            message=message,
        )

        self.check_history.append(check_result)
        return check_result

    def _get_entities_to_check(self, request: LimitCheckRequest) -> List[str]:
        """获取需要检查的实体列表"""
        entities = [request.entity_id]

        if self.config.cascade_limits:
            if request.trader_id:
                entities.append(request.trader_id)
            if request.strategy_id:
                entities.append(request.strategy_id)
            if request.portfolio_id:
                entities.append(request.portfolio_id)
            if request.account_id:
                entities.append(request.account_id)

        return list(set(entities))

    async def _check_single_limit(
        self,
        limit: RiskLimit,
        request: LimitCheckRequest,
    ) -> Dict[str, Any]:
        """检查单个限额"""
        projected_usage = limit.current_usage
        status = "ok"

        # 根据限额类型计算预期使用量
        if limit.limit_type == LimitType.GROSS_EXPOSURE:
            projected_usage = limit.current_usage + request.notional

        elif limit.limit_type == LimitType.NET_EXPOSURE:
            if request.side == "buy":
                projected_usage = limit.current_usage + request.notional
            else:
                projected_usage = limit.current_usage - request.notional

        elif limit.limit_type == LimitType.SINGLE_POSITION:
            projected_usage = request.notional

        elif limit.limit_type == LimitType.ORDER_SIZE:
            projected_usage = request.notional

        elif limit.limit_type == LimitType.DAILY_TRADES:
            counter = self._get_daily_counter(request.entity_id)
            projected_usage = Decimal(str(counter.get("trades", 0) + 1))

        elif limit.limit_type == LimitType.DAILY_TURNOVER:
            counter = self._get_daily_counter(request.entity_id)
            projected_usage = counter.get("turnover", Decimal("0")) + request.notional

        elif limit.limit_type == LimitType.DAILY_LOSS:
            counter = self._get_daily_counter(request.entity_id)
            projected_usage = counter.get("loss", Decimal("0"))

        # 计算比率
        projected_ratio = (
            projected_usage / limit.limit_value
            if limit.limit_value > 0
            else Decimal("0")
        )

        # 判断状态
        if projected_ratio >= Decimal("1"):
            status = "violated"
        elif projected_ratio >= limit.critical_threshold:
            status = "critical"
        elif projected_ratio >= limit.warning_threshold:
            status = "warning"

        excess = max(Decimal("0"), projected_usage - limit.limit_value)

        return {
            "projected_usage": projected_usage,
            "projected_ratio": float(projected_ratio),
            "status": status,
            "excess": excess,
        }

    def _generate_check_message(
        self,
        result: CheckResult,
        violated_limits: List[Dict],
        warnings: List[str],
    ) -> str:
        """生成检查消息"""
        if result == CheckResult.PASSED:
            if warnings:
                return f"Order passed with {len(warnings)} warning(s)"
            return "Order passed all limit checks"

        elif result == CheckResult.SOFT_REJECT:
            return f"Order requires approval: {len(violated_limits)} soft limit(s) violated"

        else:  # HARD_REJECT
            return f"Order rejected: {len(violated_limits)} hard limit(s) violated"

    # ==================== 审批管理 ====================

    def _create_approval_request(
        self,
        request: LimitCheckRequest,
        violated_limits: List[Dict],
    ) -> str:
        """创建审批请求"""
        approval_id = str(uuid.uuid4())[:8]

        approval = ApprovalRequest(
            approval_id=approval_id,
            order_id=request.order_id,
            entity_id=request.entity_id,
            violated_limits=[v["limit_id"] for v in violated_limits],
            requested_by=request.trader_id or request.entity_id,
            reason=f"Order {request.order_id} violated {len(violated_limits)} soft limit(s)",
        )

        self.approval_requests[approval_id] = approval
        self._stats["pending_approvals"] += 1

        if self._on_approval_required:
            self._on_approval_required(approval)

        return approval_id

    def approve_request(
        self,
        approval_id: str,
        approved_by: str,
        notes: str = "",
    ) -> bool:
        """批准请求"""
        if approval_id not in self.approval_requests:
            return False

        approval = self.approval_requests[approval_id]
        if approval.status != ApprovalStatus.PENDING:
            return False

        if datetime.now() > approval.expires_at:
            approval.status = ApprovalStatus.EXPIRED
            return False

        approval.status = ApprovalStatus.APPROVED
        approval.approved_by = approved_by
        approval.approval_notes = notes
        approval.resolved_at = datetime.now()

        self._stats["pending_approvals"] -= 1
        logger.info(f"Approval {approval_id} approved by {approved_by}")
        return True

    def reject_request(
        self,
        approval_id: str,
        rejected_by: str,
        notes: str = "",
    ) -> bool:
        """拒绝请求"""
        if approval_id not in self.approval_requests:
            return False

        approval = self.approval_requests[approval_id]
        if approval.status != ApprovalStatus.PENDING:
            return False

        approval.status = ApprovalStatus.REJECTED
        approval.approved_by = rejected_by
        approval.approval_notes = notes
        approval.resolved_at = datetime.now()

        self._stats["pending_approvals"] -= 1
        logger.info(f"Approval {approval_id} rejected by {rejected_by}")
        return True

    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """获取待审批列表"""
        return [
            a for a in self.approval_requests.values()
            if a.status == ApprovalStatus.PENDING
        ]

    # ==================== 告警管理 ====================

    def _generate_alert(self, limit: RiskLimit, alert_level: str):
        """生成告警"""
        # 检查冷却
        cooldown_key = f"{limit.limit_id}_{alert_level}"
        if cooldown_key in self._alert_cooldown:
            if datetime.now() < self._alert_cooldown[cooldown_key]:
                return

        alert_id = str(uuid.uuid4())[:8]
        alert = LimitAlert(
            alert_id=alert_id,
            limit_id=limit.limit_id,
            limit_type=limit.limit_type,
            entity_id=limit.entity_id,
            alert_level=alert_level,
            usage_ratio=limit.usage_ratio,
            message=f"Limit {limit.limit_id} ({limit.limit_type.value}) at {limit.usage_ratio:.1%} - {alert_level}",
        )

        self.alerts.append(alert)
        self._stats["total_alerts"] += 1

        # 设置冷却
        self._alert_cooldown[cooldown_key] = (
            datetime.now() + timedelta(seconds=self.config.alert_cooldown)
        )

        if self._on_alert:
            self._on_alert(alert)

        logger.warning(f"Limit alert: {alert.message}")

    def _record_breach(self, limit: RiskLimit, order_id: Optional[str]):
        """记录突破"""
        breach_id = str(uuid.uuid4())[:8]
        breach = LimitBreach(
            breach_id=breach_id,
            limit_id=limit.limit_id,
            limit_type=limit.limit_type,
            entity_id=limit.entity_id,
            limit_value=limit.limit_value,
            actual_value=limit.current_usage,
            breach_ratio=limit.usage_ratio,
            order_id=order_id,
        )

        self.breaches.append(breach)
        self._stats["total_breaches"] += 1

        if self._on_breach:
            self._on_breach(breach)

        logger.error(f"Limit breach: {limit.limit_id} - {limit.usage_ratio:.1%}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def get_unacknowledged_alerts(self) -> List[LimitAlert]:
        """获取未确认告警"""
        return [a for a in self.alerts if not a.acknowledged]

    # ==================== 日内计数器 ====================

    def _reset_daily_counters_if_needed(self):
        """重置日内计数器"""
        today = date.today()
        if self._counter_date != today:
            self._daily_counters = {}
            self._counter_date = today

    def _get_daily_counter(self, entity_id: str) -> Dict[str, Any]:
        """获取日内计数器"""
        if entity_id not in self._daily_counters:
            self._daily_counters[entity_id] = {
                "trades": 0,
                "turnover": Decimal("0"),
                "loss": Decimal("0"),
            }
        return self._daily_counters[entity_id]

    def update_daily_counter(
        self,
        entity_id: str,
        trades: int = 0,
        turnover: Decimal = Decimal("0"),
        loss: Decimal = Decimal("0"),
    ):
        """更新日内计数器"""
        self._reset_daily_counters_if_needed()
        counter = self._get_daily_counter(entity_id)
        counter["trades"] += trades
        counter["turnover"] += turnover
        counter["loss"] += loss

    # ==================== 实时监控 ====================

    async def start(self):
        """启动实时监控"""
        if self._running:
            return

        self._running = True
        self.state = EngineState.RUNNING
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Risk limits engine started")

    async def stop(self):
        """停止实时监控"""
        self._running = False
        self.state = EngineState.IDLE
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Risk limits engine stopped")

    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                await self._update_all_limits()
                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self.state = EngineState.ERROR
                await asyncio.sleep(60)

    async def _update_all_limits(self):
        """更新所有限额"""
        for limit_id, limit in self.limits.items():
            try:
                await self._update_limit_usage(limit)
            except Exception as e:
                logger.error(f"Error updating limit {limit_id}: {e}")

    async def _update_limit_usage(self, limit: RiskLimit):
        """更新限额使用量"""
        usage = Decimal("0")

        if limit.limit_type == LimitType.GROSS_EXPOSURE:
            usage = await self.position_provider.get_gross_exposure(limit.entity_id)

        elif limit.limit_type == LimitType.NET_EXPOSURE:
            usage = await self.position_provider.get_net_exposure(limit.entity_id)

        elif limit.limit_type == LimitType.VAR_LIMIT:
            usage = await self.risk_provider.get_var(limit.entity_id)

        elif limit.limit_type == LimitType.CVAR_LIMIT:
            usage = await self.risk_provider.get_cvar(limit.entity_id)

        elif limit.limit_type == LimitType.DRAWDOWN_LIMIT:
            usage = await self.risk_provider.get_drawdown(limit.entity_id)

        elif limit.limit_type == LimitType.DAILY_TRADES:
            counter = self._get_daily_counter(limit.entity_id)
            usage = Decimal(str(counter.get("trades", 0)))

        elif limit.limit_type == LimitType.DAILY_TURNOVER:
            counter = self._get_daily_counter(limit.entity_id)
            usage = counter.get("turnover", Decimal("0"))

        elif limit.limit_type == LimitType.DAILY_LOSS:
            counter = self._get_daily_counter(limit.entity_id)
            usage = counter.get("loss", Decimal("0"))

        self.update_usage(limit.limit_id, usage)

    # ==================== 查询接口 ====================

    def get_all_limits(self) -> List[RiskLimit]:
        """获取所有限额"""
        return list(self.limits.values())

    def get_breaches(
        self,
        entity_id: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> List[LimitBreach]:
        """获取突破记录"""
        breaches = self.breaches
        if entity_id:
            breaches = [b for b in breaches if b.entity_id == entity_id]
        if resolved is not None:
            breaches = [b for b in breaches if b.resolved == resolved]
        return breaches[-limit:]

    def get_alerts(
        self,
        entity_id: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[LimitAlert]:
        """获取告警"""
        alerts = self.alerts
        if entity_id:
            alerts = [a for a in alerts if a.entity_id == entity_id]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        return alerts[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "state": self.state.value,
            "total_limits": len(self.limits),
            "limits_by_status": self._count_limits_by_status(),
            "unacknowledged_alerts": len(self.get_unacknowledged_alerts()),
        }

    def _count_limits_by_status(self) -> Dict[str, int]:
        """统计各状态限额数量"""
        counts = {}
        for limit in self.limits.values():
            status = limit.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts

    def get_limit_summary(self, entity_id: str) -> Dict[str, Any]:
        """获取限额摘要"""
        limits = self.get_limits_by_entity(entity_id)
        return {
            "entity_id": entity_id,
            "total_limits": len(limits),
            "limits": [l.to_dict() for l in limits],
            "breached": sum(1 for l in limits if l.status == LimitStatus.BREACHED),
            "critical": sum(1 for l in limits if l.status == LimitStatus.CRITICAL),
            "warning": sum(1 for l in limits if l.status == LimitStatus.WARNING),
            "normal": sum(1 for l in limits if l.status == LimitStatus.NORMAL),
        }


# ==================== 工厂函数 ====================

def create_risk_limits_engine(
    position_provider: Optional[PositionDataProvider] = None,
    risk_provider: Optional[RiskMetricsProvider] = None,
    check_interval: float = 60.0,
    enable_pre_trade: bool = True,
    enable_monitoring: bool = True,
) -> RiskLimitsEngine:
    """创建风控限额引擎"""
    config = EngineConfig(
        check_interval=check_interval,
        enable_pre_trade_check=enable_pre_trade,
        enable_real_time_monitoring=enable_monitoring,
    )

    return RiskLimitsEngine(
        position_provider=position_provider,
        risk_provider=risk_provider,
        config=config,
    )


def create_risk_limit(
    limit_type: str,
    entity_id: str,
    limit_value: float,
    limit_level: str = "account",
    warning_threshold: float = 0.80,
    critical_threshold: float = 0.90,
    is_hard_limit: bool = True,
    description: str = "",
) -> RiskLimit:
    """创建风控限额"""
    type_map = {
        "gross_exposure": LimitType.GROSS_EXPOSURE,
        "net_exposure": LimitType.NET_EXPOSURE,
        "long_exposure": LimitType.LONG_EXPOSURE,
        "short_exposure": LimitType.SHORT_EXPOSURE,
        "var_limit": LimitType.VAR_LIMIT,
        "cvar_limit": LimitType.CVAR_LIMIT,
        "drawdown_limit": LimitType.DRAWDOWN_LIMIT,
        "single_position": LimitType.SINGLE_POSITION,
        "sector_concentration": LimitType.SECTOR_CONCENTRATION,
        "leverage_ratio": LimitType.LEVERAGE_RATIO,
        "margin_utilization": LimitType.MARGIN_UTILIZATION,
        "daily_loss": LimitType.DAILY_LOSS,
        "daily_turnover": LimitType.DAILY_TURNOVER,
        "order_size": LimitType.ORDER_SIZE,
        "daily_trades": LimitType.DAILY_TRADES,
        "adv_limit": LimitType.ADV_LIMIT,
        "market_impact": LimitType.MARKET_IMPACT,
    }

    level_map = {
        "trader": LimitLevel.TRADER,
        "strategy": LimitLevel.STRATEGY,
        "portfolio": LimitLevel.PORTFOLIO,
        "account": LimitLevel.ACCOUNT,
        "desk": LimitLevel.DESK,
        "firm": LimitLevel.FIRM,
    }

    return RiskLimit(
        limit_id=str(uuid.uuid4())[:8],
        limit_type=type_map.get(limit_type, LimitType.GROSS_EXPOSURE),
        limit_level=level_map.get(limit_level, LimitLevel.ACCOUNT),
        entity_id=entity_id,
        limit_value=Decimal(str(limit_value)),
        warning_threshold=Decimal(str(warning_threshold)),
        critical_threshold=Decimal(str(critical_threshold)),
        is_hard_limit=is_hard_limit,
        description=description,
    )


def create_limit_check_request(
    order_id: str,
    entity_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    **kwargs,
) -> LimitCheckRequest:
    """创建限额检查请求"""
    return LimitCheckRequest(
        order_id=order_id,
        entity_id=entity_id,
        symbol=symbol,
        side=side,
        quantity=Decimal(str(quantity)),
        price=Decimal(str(price)),
        **kwargs,
    )


async def quick_limit_check(
    engine: RiskLimitsEngine,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    entity_id: str = "default",
) -> Dict[str, Any]:
    """快速限额检查"""
    request = create_limit_check_request(
        order_id=str(uuid.uuid4())[:8],
        entity_id=entity_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
    )

    result = await engine.check_order(request)
    return result.to_dict()


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "LimitType",
    "LimitLevel",
    "LimitStatus",
    "CheckResult",
    "ApprovalStatus",
    "EngineState",
    # 数据类
    "RiskLimit",
    "LimitCheckRequest",
    "LimitCheckResult",
    "LimitBreach",
    "ApprovalRequest",
    "LimitAlert",
    "EngineConfig",
    # 接口
    "PositionDataProvider",
    "RiskMetricsProvider",
    # 模拟实现
    "SimulatedPositionProvider",
    "SimulatedRiskMetricsProvider",
    # 核心引擎
    "RiskLimitsEngine",
    # 工厂函数
    "create_risk_limits_engine",
    "create_risk_limit",
    "create_limit_check_request",
    "quick_limit_check",
]
