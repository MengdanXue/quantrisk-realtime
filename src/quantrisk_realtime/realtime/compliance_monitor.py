"""
合规监控引擎 (Compliance Monitor)

机构级交易合规监控系统，提供:
- 交易规则检查 (Trading Rule Checks)
- 持仓限制监控 (Position Limit Monitoring)
- 交易频率监控 (Trading Frequency Monitoring)
- 黑名单/白名单管理 (Restricted List Management)
- 洗售检测 (Wash Trading Detection)
- 内幕交易检测 (Insider Trading Detection)
- 合规报告生成 (Compliance Reporting)
- 审计日志 (Audit Trail)

作者: TraderX 量化团队
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import uuid
import logging
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class RuleType(Enum):
    """规则类型"""
    POSITION_LIMIT = "position_limit"           # 持仓限制
    TRADING_FREQUENCY = "trading_frequency"     # 交易频率
    ORDER_SIZE = "order_size"                   # 订单大小
    CONCENTRATION = "concentration"             # 集中度
    RESTRICTED_SECURITY = "restricted_security" # 受限证券
    WASH_TRADING = "wash_trading"               # 洗售
    INSIDER_TRADING = "insider_trading"         # 内幕交易
    MARKET_MANIPULATION = "market_manipulation" # 市场操纵
    BEST_EXECUTION = "best_execution"           # 最优执行
    CROSS_TRADING = "cross_trading"             # 对敲交易


class RuleSeverity(Enum):
    """规则严重性"""
    INFO = "info"           # 信息
    WARNING = "warning"     # 警告
    VIOLATION = "violation" # 违规
    CRITICAL = "critical"   # 严重违规


class ViolationStatus(Enum):
    """违规状态"""
    OPEN = "open"                   # 待处理
    UNDER_REVIEW = "under_review"   # 审核中
    RESOLVED = "resolved"           # 已解决
    ESCALATED = "escalated"         # 已升级
    DISMISSED = "dismissed"         # 已驳回


class RestrictionType(Enum):
    """限制类型"""
    BLACKLIST = "blacklist"         # 黑名单 - 禁止交易
    WHITELIST = "whitelist"         # 白名单 - 仅可交易
    WATCHLIST = "watchlist"         # 观察名单 - 需审批
    GREYLIST = "greylist"           # 灰名单 - 限制交易


class MonitorState(Enum):
    """监控状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


# ==================== 数据类 ====================

@dataclass
class ComplianceRule:
    """合规规则"""
    rule_id: str
    rule_type: RuleType
    name: str
    description: str

    # 规则条件
    condition: Dict[str, Any] = field(default_factory=dict)

    # 规则属性
    severity: RuleSeverity = RuleSeverity.WARNING
    enabled: bool = True

    # 适用范围
    applies_to_accounts: List[str] = field(default_factory=list)  # 空=所有账户
    applies_to_symbols: List[str] = field(default_factory=list)   # 空=所有标的

    # 时间窗口
    time_window_minutes: int = 0  # 0=不限

    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "applies_to_accounts": self.applies_to_accounts,
            "applies_to_symbols": self.applies_to_symbols,
            "time_window_minutes": self.time_window_minutes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
        }


@dataclass
class TradeForCompliance:
    """待合规检查的交易"""
    trade_id: str
    account_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float

    # 交易属性
    order_type: str = "market"
    venue: str = ""
    strategy_id: str = ""
    trader_id: str = ""

    # 时间
    timestamp: datetime = field(default_factory=datetime.now)

    # 计算属性
    @property
    def notional_value(self) -> float:
        return self.quantity * self.price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type,
            "venue": self.venue,
            "strategy_id": self.strategy_id,
            "trader_id": self.trader_id,
            "timestamp": self.timestamp.isoformat(),
            "notional_value": self.notional_value,
        }


@dataclass
class ComplianceViolation:
    """合规违规"""
    violation_id: str
    rule_id: str
    rule_name: str
    rule_type: RuleType
    severity: RuleSeverity

    # 违规详情
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    # 关联信息
    trade_id: Optional[str] = None
    account_id: Optional[str] = None
    symbol: Optional[str] = None
    trader_id: Optional[str] = None

    # 状态
    status: ViolationStatus = ViolationStatus.OPEN

    # 处理信息
    assigned_to: Optional[str] = None
    resolution_notes: str = ""
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    # 时间
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "details": self.details,
            "trade_id": self.trade_id,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "trader_id": self.trader_id,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "resolution_notes": self.resolution_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class RestrictedSecurity:
    """受限证券"""
    symbol: str
    restriction_type: RestrictionType
    reason: str

    # 限制期间
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None  # None=永久

    # 限制详情
    max_position: Optional[float] = None  # 最大持仓
    requires_approval: bool = False       # 是否需要审批
    approved_accounts: List[str] = field(default_factory=list)  # 豁免账户

    # 元数据
    added_by: str = ""
    added_at: datetime = field(default_factory=datetime.now)
    notes: str = ""

    def is_active(self, check_time: Optional[datetime] = None) -> bool:
        """检查限制是否有效"""
        check_time = check_time or datetime.now()
        if check_time < self.start_date:
            return False
        if self.end_date and check_time > self.end_date:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "restriction_type": self.restriction_type.value,
            "reason": self.reason,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "max_position": self.max_position,
            "requires_approval": self.requires_approval,
            "approved_accounts": self.approved_accounts,
            "added_by": self.added_by,
            "added_at": self.added_at.isoformat(),
            "notes": self.notes,
            "is_active": self.is_active(),
        }


@dataclass
class AuditEntry:
    """审计日志条目"""
    entry_id: str
    action: str
    actor: str
    target_type: str  # rule, violation, restriction, trade
    target_id: str

    # 变更详情
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None

    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: str = ""
    user_agent: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "action": self.action,
            "actor": self.actor,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "notes": self.notes,
        }


@dataclass
class ComplianceCheckResult:
    """合规检查结果"""
    passed: bool
    rule_id: str
    rule_name: str
    rule_type: RuleType

    # 检查详情
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # 违规信息（如果未通过）
    violation: Optional[ComplianceViolation] = None

    # 时间
    checked_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "message": self.message,
            "details": self.details,
            "violation": self.violation.to_dict() if self.violation else None,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class ComplianceReport:
    """合规报告"""
    report_id: str
    report_type: str  # daily, weekly, monthly, ad-hoc
    period_start: datetime
    period_end: datetime

    # 统计数据
    total_trades_checked: int = 0
    violations_detected: int = 0
    violations_resolved: int = 0
    violations_open: int = 0

    # 按类型统计
    violations_by_type: Dict[str, int] = field(default_factory=dict)
    violations_by_severity: Dict[str, int] = field(default_factory=dict)
    violations_by_account: Dict[str, int] = field(default_factory=dict)

    # 合规率
    compliance_rate: float = 100.0

    # 详细违规列表
    violations: List[ComplianceViolation] = field(default_factory=list)

    # 元数据
    generated_at: datetime = field(default_factory=datetime.now)
    generated_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_trades_checked": self.total_trades_checked,
            "violations_detected": self.violations_detected,
            "violations_resolved": self.violations_resolved,
            "violations_open": self.violations_open,
            "violations_by_type": self.violations_by_type,
            "violations_by_severity": self.violations_by_severity,
            "violations_by_account": self.violations_by_account,
            "compliance_rate": self.compliance_rate,
            "violations": [v.to_dict() for v in self.violations],
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
        }


@dataclass
class ComplianceConfig:
    """合规监控配置"""
    # 检查频率
    real_time_check: bool = True
    batch_check_interval_minutes: int = 60

    # 洗售检测参数
    wash_trade_time_window_seconds: int = 300  # 5分钟
    wash_trade_price_tolerance: float = 0.001  # 0.1%

    # 交易频率限制
    max_trades_per_minute: int = 100
    max_trades_per_hour: int = 1000
    max_trades_per_day: int = 10000

    # 持仓限制
    max_position_pct_of_adv: float = 0.1  # ADV的10%
    max_concentration_pct: float = 0.2    # 单一标的最大20%

    # 告警设置
    alert_on_warning: bool = True
    alert_on_violation: bool = True
    alert_on_critical: bool = True

    # 报告设置
    auto_generate_daily_report: bool = True
    report_recipients: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "real_time_check": self.real_time_check,
            "batch_check_interval_minutes": self.batch_check_interval_minutes,
            "wash_trade_time_window_seconds": self.wash_trade_time_window_seconds,
            "wash_trade_price_tolerance": self.wash_trade_price_tolerance,
            "max_trades_per_minute": self.max_trades_per_minute,
            "max_trades_per_hour": self.max_trades_per_hour,
            "max_trades_per_day": self.max_trades_per_day,
            "max_position_pct_of_adv": self.max_position_pct_of_adv,
            "max_concentration_pct": self.max_concentration_pct,
            "alert_on_warning": self.alert_on_warning,
            "alert_on_violation": self.alert_on_violation,
            "alert_on_critical": self.alert_on_critical,
            "auto_generate_daily_report": self.auto_generate_daily_report,
            "report_recipients": self.report_recipients,
        }


# ==================== 接口定义 ====================

class PositionProvider(ABC):
    """持仓数据提供者接口"""

    @abstractmethod
    def get_position(self, account_id: str, symbol: str) -> float:
        """获取持仓数量"""
        pass

    @abstractmethod
    def get_all_positions(self, account_id: str) -> Dict[str, float]:
        """获取所有持仓"""
        pass

    @abstractmethod
    def get_portfolio_value(self, account_id: str) -> float:
        """获取组合总值"""
        pass


class MarketDataProvider(ABC):
    """市场数据提供者接口"""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """获取当前价格"""
        pass

    @abstractmethod
    def get_adv(self, symbol: str) -> float:
        """获取平均日成交量"""
        pass


# ==================== 模拟实现 ====================

class SimulatedPositionProvider(PositionProvider):
    """模拟持仓提供者"""

    def __init__(self):
        self.positions: Dict[str, Dict[str, float]] = {}
        self.portfolio_values: Dict[str, float] = {}

    def set_position(self, account_id: str, symbol: str, quantity: float):
        if account_id not in self.positions:
            self.positions[account_id] = {}
        self.positions[account_id][symbol] = quantity

    def set_portfolio_value(self, account_id: str, value: float):
        self.portfolio_values[account_id] = value

    def get_position(self, account_id: str, symbol: str) -> float:
        return self.positions.get(account_id, {}).get(symbol, 0.0)

    def get_all_positions(self, account_id: str) -> Dict[str, float]:
        return self.positions.get(account_id, {}).copy()

    def get_portfolio_value(self, account_id: str) -> float:
        return self.portfolio_values.get(account_id, 1000000.0)


class SimulatedMarketDataProvider(MarketDataProvider):
    """模拟市场数据提供者"""

    def __init__(self):
        self.prices: Dict[str, float] = {}
        self.advs: Dict[str, float] = {}

    def set_price(self, symbol: str, price: float):
        self.prices[symbol] = price

    def set_adv(self, symbol: str, adv: float):
        self.advs[symbol] = adv

    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 100.0)

    def get_adv(self, symbol: str) -> float:
        return self.advs.get(symbol, 1000000.0)


# ==================== 核心引擎 ====================

class ComplianceMonitor:
    """合规监控引擎"""

    def __init__(
        self,
        config: Optional[ComplianceConfig] = None,
        position_provider: Optional[PositionProvider] = None,
        market_data_provider: Optional[MarketDataProvider] = None,
    ):
        self.config = config or ComplianceConfig()
        self.position_provider = position_provider or simulated_default(SimulatedPositionProvider, "data from ComplianceMonitor's simulated position_provider (SimulatedPositionProvider)")
        self.market_data_provider = market_data_provider or simulated_default(SimulatedMarketDataProvider, "data from ComplianceMonitor's simulated market_data_provider (SimulatedMarketDataProvider)")

        # 规则存储
        self.rules: Dict[str, ComplianceRule] = {}

        # 违规存储
        self.violations: Dict[str, ComplianceViolation] = {}

        # 受限证券
        self.restricted_securities: Dict[str, RestrictedSecurity] = {}

        # 审计日志
        self.audit_log: List[AuditEntry] = []

        # 交易历史（用于洗售检测等）
        self.trade_history: List[TradeForCompliance] = []
        self.trade_history_max_size: int = 100000

        # 交易计数（用于频率检测）
        self.trade_counts: Dict[str, Dict[str, int]] = {}  # account_id -> {minute: count}

        # 状态
        self.state = MonitorState.IDLE
        self._monitoring_task: Optional[asyncio.Task] = None

        # 回调
        self._on_violation: Optional[Callable[[ComplianceViolation], None]] = None
        self._on_alert: Optional[Callable[[str, Dict[str, Any]], None]] = None

        # 统计
        self.stats = {
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "violations_created": 0,
            "violations_resolved": 0,
        }

        # 初始化默认规则
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认合规规则"""
        # 持仓限制规则
        self.add_rule(ComplianceRule(
            rule_id="position_limit_adv",
            rule_type=RuleType.POSITION_LIMIT,
            name="ADV持仓限制",
            description="单一标的持仓不得超过ADV的指定比例",
            condition={"max_pct_of_adv": self.config.max_position_pct_of_adv},
            severity=RuleSeverity.VIOLATION,
        ))

        # 集中度限制规则
        self.add_rule(ComplianceRule(
            rule_id="concentration_limit",
            rule_type=RuleType.CONCENTRATION,
            name="持仓集中度限制",
            description="单一标的持仓不得超过组合的指定比例",
            condition={"max_concentration_pct": self.config.max_concentration_pct},
            severity=RuleSeverity.WARNING,
        ))

        # 交易频率规则
        self.add_rule(ComplianceRule(
            rule_id="trading_frequency_minute",
            rule_type=RuleType.TRADING_FREQUENCY,
            name="分钟交易频率限制",
            description="每分钟交易次数限制",
            condition={"max_trades": self.config.max_trades_per_minute, "period": "minute"},
            severity=RuleSeverity.WARNING,
            time_window_minutes=1,
        ))

        # 洗售检测规则
        self.add_rule(ComplianceRule(
            rule_id="wash_trading_detection",
            rule_type=RuleType.WASH_TRADING,
            name="洗售交易检测",
            description="检测短时间内相反方向的交易",
            condition={
                "time_window_seconds": self.config.wash_trade_time_window_seconds,
                "price_tolerance": self.config.wash_trade_price_tolerance,
            },
            severity=RuleSeverity.CRITICAL,
        ))

        # 受限证券规则
        self.add_rule(ComplianceRule(
            rule_id="restricted_security_check",
            rule_type=RuleType.RESTRICTED_SECURITY,
            name="受限证券检查",
            description="检查交易标的是否在受限名单中",
            condition={},
            severity=RuleSeverity.VIOLATION,
        ))

    # ==================== 规则管理 ====================

    def add_rule(self, rule: ComplianceRule, actor: str = "system") -> bool:
        """添加合规规则"""
        if rule.rule_id in self.rules:
            return False

        self.rules[rule.rule_id] = rule
        self._add_audit_entry(
            action="add_rule",
            actor=actor,
            target_type="rule",
            target_id=rule.rule_id,
            new_value=rule.to_dict(),
        )
        return True

    def update_rule(self, rule_id: str, updates: Dict[str, Any], actor: str = "system") -> bool:
        """更新规则"""
        if rule_id not in self.rules:
            return False

        rule = self.rules[rule_id]
        old_value = rule.to_dict()

        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        rule.updated_at = datetime.now()

        self._add_audit_entry(
            action="update_rule",
            actor=actor,
            target_type="rule",
            target_id=rule_id,
            old_value=old_value,
            new_value=rule.to_dict(),
        )
        return True

    def remove_rule(self, rule_id: str, actor: str = "system") -> bool:
        """移除规则"""
        if rule_id not in self.rules:
            return False

        rule = self.rules.pop(rule_id)
        self._add_audit_entry(
            action="remove_rule",
            actor=actor,
            target_type="rule",
            target_id=rule_id,
            old_value=rule.to_dict(),
        )
        return True

    def get_rule(self, rule_id: str) -> Optional[ComplianceRule]:
        """获取规则"""
        return self.rules.get(rule_id)

    def get_all_rules(self, enabled_only: bool = False) -> List[ComplianceRule]:
        """获取所有规则"""
        rules = list(self.rules.values())
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        return rules

    def enable_rule(self, rule_id: str, actor: str = "system") -> bool:
        """启用规则"""
        return self.update_rule(rule_id, {"enabled": True}, actor)

    def disable_rule(self, rule_id: str, actor: str = "system") -> bool:
        """禁用规则"""
        return self.update_rule(rule_id, {"enabled": False}, actor)

    # ==================== 受限证券管理 ====================

    def add_restricted_security(
        self,
        security: RestrictedSecurity,
        actor: str = "system"
    ) -> bool:
        """添加受限证券"""
        self.restricted_securities[security.symbol] = security
        self._add_audit_entry(
            action="add_restricted_security",
            actor=actor,
            target_type="restriction",
            target_id=security.symbol,
            new_value=security.to_dict(),
        )
        return True

    def remove_restricted_security(self, symbol: str, actor: str = "system") -> bool:
        """移除受限证券"""
        if symbol not in self.restricted_securities:
            return False

        security = self.restricted_securities.pop(symbol)
        self._add_audit_entry(
            action="remove_restricted_security",
            actor=actor,
            target_type="restriction",
            target_id=symbol,
            old_value=security.to_dict(),
        )
        return True

    def get_restricted_security(self, symbol: str) -> Optional[RestrictedSecurity]:
        """获取受限证券信息"""
        return self.restricted_securities.get(symbol)

    def get_all_restricted_securities(
        self,
        restriction_type: Optional[RestrictionType] = None,
        active_only: bool = True
    ) -> List[RestrictedSecurity]:
        """获取所有受限证券"""
        securities = list(self.restricted_securities.values())

        if restriction_type:
            securities = [s for s in securities if s.restriction_type == restriction_type]

        if active_only:
            securities = [s for s in securities if s.is_active()]

        return securities

    def is_restricted(self, symbol: str, account_id: Optional[str] = None) -> bool:
        """检查证券是否受限"""
        security = self.restricted_securities.get(symbol)
        if not security or not security.is_active():
            return False

        # 检查豁免账户
        if account_id and account_id in security.approved_accounts:
            return False

        return security.restriction_type == RestrictionType.BLACKLIST

    def requires_approval(self, symbol: str) -> bool:
        """检查交易是否需要审批"""
        security = self.restricted_securities.get(symbol)
        if not security or not security.is_active():
            return False

        return security.requires_approval or security.restriction_type == RestrictionType.WATCHLIST

    # ==================== 合规检查 ====================

    def check_trade(self, trade: TradeForCompliance) -> List[ComplianceCheckResult]:
        """检查交易合规性"""
        results = []

        for rule in self.get_all_rules(enabled_only=True):
            # 检查规则是否适用
            if not self._rule_applies(rule, trade):
                continue

            # 执行检查
            result = self._check_rule(rule, trade)
            results.append(result)

            # 统计
            self.stats["total_checks"] += 1
            if result.passed:
                self.stats["passed_checks"] += 1
            else:
                self.stats["failed_checks"] += 1

                # 创建违规记录
                if result.violation:
                    self.violations[result.violation.violation_id] = result.violation
                    self.stats["violations_created"] += 1

                    # 触发回调
                    if self._on_violation:
                        self._on_violation(result.violation)

                    # 触发告警
                    self._trigger_alert(result.violation)

        # 记录交易历史
        self._add_trade_to_history(trade)

        return results

    def _rule_applies(self, rule: ComplianceRule, trade: TradeForCompliance) -> bool:
        """检查规则是否适用于此交易"""
        # 检查账户限制
        if rule.applies_to_accounts and trade.account_id not in rule.applies_to_accounts:
            return False

        # 检查标的限制
        if rule.applies_to_symbols and trade.symbol not in rule.applies_to_symbols:
            return False

        return True

    def _check_rule(self, rule: ComplianceRule, trade: TradeForCompliance) -> ComplianceCheckResult:
        """执行单条规则检查"""
        check_methods = {
            RuleType.POSITION_LIMIT: self._check_position_limit,
            RuleType.CONCENTRATION: self._check_concentration,
            RuleType.TRADING_FREQUENCY: self._check_trading_frequency,
            RuleType.WASH_TRADING: self._check_wash_trading,
            RuleType.RESTRICTED_SECURITY: self._check_restricted_security,
            RuleType.ORDER_SIZE: self._check_order_size,
        }

        check_method = check_methods.get(rule.rule_type)
        if check_method:
            return check_method(rule, trade)

        # 默认通过
        return ComplianceCheckResult(
            passed=True,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            rule_type=rule.rule_type,
            message="规则类型未实现检查逻辑",
        )

    def _check_position_limit(
        self,
        rule: ComplianceRule,
        trade: TradeForCompliance
    ) -> ComplianceCheckResult:
        """检查持仓限制"""
        max_pct_of_adv = rule.condition.get("max_pct_of_adv", 0.1)

        # 获取当前持仓和ADV
        current_position = self.position_provider.get_position(trade.account_id, trade.symbol)
        adv = self.market_data_provider.get_adv(trade.symbol)

        # 计算交易后持仓
        if trade.side.lower() == "buy":
            new_position = current_position + trade.quantity
        else:
            new_position = current_position - trade.quantity

        # 检查是否超限
        position_pct = abs(new_position) / adv if adv > 0 else 0

        if position_pct > max_pct_of_adv:
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                description=f"持仓超过ADV限制: {position_pct:.2%} > {max_pct_of_adv:.2%}",
                details={
                    "current_position": current_position,
                    "trade_quantity": trade.quantity,
                    "new_position": new_position,
                    "adv": adv,
                    "position_pct": position_pct,
                    "limit_pct": max_pct_of_adv,
                },
                trade_id=trade.trade_id,
                account_id=trade.account_id,
                symbol=trade.symbol,
                trader_id=trade.trader_id,
            )

            return ComplianceCheckResult(
                passed=False,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=violation.description,
                details=violation.details,
                violation=violation,
            )

        return ComplianceCheckResult(
            passed=True,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            rule_type=rule.rule_type,
            message=f"持仓在ADV限制内: {position_pct:.2%} <= {max_pct_of_adv:.2%}",
            details={"position_pct": position_pct, "limit_pct": max_pct_of_adv},
        )

    def _check_concentration(
        self,
        rule: ComplianceRule,
        trade: TradeForCompliance
    ) -> ComplianceCheckResult:
        """检查持仓集中度"""
        max_concentration_pct = rule.condition.get("max_concentration_pct", 0.2)

        # 获取当前持仓和组合总值
        current_position = self.position_provider.get_position(trade.account_id, trade.symbol)
        portfolio_value = self.position_provider.get_portfolio_value(trade.account_id)
        price = self.market_data_provider.get_price(trade.symbol)

        # 计算交易后持仓价值
        if trade.side.lower() == "buy":
            new_position = current_position + trade.quantity
        else:
            new_position = current_position - trade.quantity

        position_value = abs(new_position) * price
        concentration_pct = position_value / portfolio_value if portfolio_value > 0 else 0

        if concentration_pct > max_concentration_pct:
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                description=f"持仓集中度超限: {concentration_pct:.2%} > {max_concentration_pct:.2%}",
                details={
                    "position_value": position_value,
                    "portfolio_value": portfolio_value,
                    "concentration_pct": concentration_pct,
                    "limit_pct": max_concentration_pct,
                },
                trade_id=trade.trade_id,
                account_id=trade.account_id,
                symbol=trade.symbol,
                trader_id=trade.trader_id,
            )

            return ComplianceCheckResult(
                passed=False,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=violation.description,
                details=violation.details,
                violation=violation,
            )

        return ComplianceCheckResult(
            passed=True,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            rule_type=rule.rule_type,
            message=f"持仓集中度在限制内: {concentration_pct:.2%} <= {max_concentration_pct:.2%}",
            details={"concentration_pct": concentration_pct, "limit_pct": max_concentration_pct},
        )

    def _check_trading_frequency(
        self,
        rule: ComplianceRule,
        trade: TradeForCompliance
    ) -> ComplianceCheckResult:
        """检查交易频率"""
        max_trades = rule.condition.get("max_trades", 100)
        period = rule.condition.get("period", "minute")

        # 获取当前时间窗口的交易计数
        account_id = trade.account_id
        if account_id not in self.trade_counts:
            self.trade_counts[account_id] = {}

        # 时间窗口键
        if period == "minute":
            window_key = trade.timestamp.strftime("%Y%m%d%H%M")
        elif period == "hour":
            window_key = trade.timestamp.strftime("%Y%m%d%H")
        else:  # day
            window_key = trade.timestamp.strftime("%Y%m%d")

        # 增加计数
        current_count = self.trade_counts[account_id].get(window_key, 0) + 1
        self.trade_counts[account_id][window_key] = current_count

        if current_count > max_trades:
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                description=f"交易频率超限: {current_count} > {max_trades}/{period}",
                details={
                    "current_count": current_count,
                    "max_trades": max_trades,
                    "period": period,
                    "window_key": window_key,
                },
                trade_id=trade.trade_id,
                account_id=trade.account_id,
                symbol=trade.symbol,
                trader_id=trade.trader_id,
            )

            return ComplianceCheckResult(
                passed=False,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=violation.description,
                details=violation.details,
                violation=violation,
            )

        return ComplianceCheckResult(
            passed=True,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            rule_type=rule.rule_type,
            message=f"交易频率在限制内: {current_count}/{max_trades} {period}",
            details={"current_count": current_count, "max_trades": max_trades},
        )

    def _check_wash_trading(
        self,
        rule: ComplianceRule,
        trade: TradeForCompliance
    ) -> ComplianceCheckResult:
        """检查洗售交易"""
        time_window_seconds = rule.condition.get("time_window_seconds", 300)
        price_tolerance = rule.condition.get("price_tolerance", 0.001)

        # 查找最近的相反方向交易
        cutoff_time = trade.timestamp - timedelta(seconds=time_window_seconds)
        opposite_side = "sell" if trade.side.lower() == "buy" else "buy"

        suspicious_trades = []
        for hist_trade in reversed(self.trade_history):
            if hist_trade.timestamp < cutoff_time:
                break

            # 检查是否是相同账户、标的、相反方向
            if (hist_trade.account_id == trade.account_id and
                hist_trade.symbol == trade.symbol and
                hist_trade.side.lower() == opposite_side):

                # 检查价格是否接近
                price_diff = abs(hist_trade.price - trade.price) / hist_trade.price
                if price_diff <= price_tolerance:
                    suspicious_trades.append(hist_trade)

        if suspicious_trades:
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                description=f"疑似洗售交易: {len(suspicious_trades)}笔相反方向交易在{time_window_seconds}秒内",
                details={
                    "suspicious_trade_count": len(suspicious_trades),
                    "suspicious_trade_ids": [t.trade_id for t in suspicious_trades],
                    "time_window_seconds": time_window_seconds,
                    "price_tolerance": price_tolerance,
                },
                trade_id=trade.trade_id,
                account_id=trade.account_id,
                symbol=trade.symbol,
                trader_id=trade.trader_id,
            )

            return ComplianceCheckResult(
                passed=False,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=violation.description,
                details=violation.details,
                violation=violation,
            )

        return ComplianceCheckResult(
            passed=True,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            rule_type=rule.rule_type,
            message="未检测到洗售交易模式",
        )

    def _check_restricted_security(
        self,
        rule: ComplianceRule,
        trade: TradeForCompliance
    ) -> ComplianceCheckResult:
        """检查受限证券"""
        security = self.restricted_securities.get(trade.symbol)

        if not security or not security.is_active():
            return ComplianceCheckResult(
                passed=True,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=f"{trade.symbol}不在受限名单中",
            )

        # 检查是否是豁免账户
        if trade.account_id in security.approved_accounts:
            return ComplianceCheckResult(
                passed=True,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=f"账户{trade.account_id}已获得{trade.symbol}的交易豁免",
            )

        # 黑名单 - 直接违规
        if security.restriction_type == RestrictionType.BLACKLIST:
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=RuleSeverity.CRITICAL,
                description=f"交易黑名单证券: {trade.symbol}",
                details={
                    "restriction_type": security.restriction_type.value,
                    "reason": security.reason,
                },
                trade_id=trade.trade_id,
                account_id=trade.account_id,
                symbol=trade.symbol,
                trader_id=trade.trader_id,
            )

            return ComplianceCheckResult(
                passed=False,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=violation.description,
                details=violation.details,
                violation=violation,
            )

        # 观察名单 - 需要审批
        if security.restriction_type == RestrictionType.WATCHLIST:
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=RuleSeverity.WARNING,
                description=f"交易观察名单证券需要审批: {trade.symbol}",
                details={
                    "restriction_type": security.restriction_type.value,
                    "reason": security.reason,
                    "requires_approval": True,
                },
                trade_id=trade.trade_id,
                account_id=trade.account_id,
                symbol=trade.symbol,
                trader_id=trade.trader_id,
            )

            return ComplianceCheckResult(
                passed=False,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=violation.description,
                details=violation.details,
                violation=violation,
            )

        return ComplianceCheckResult(
            passed=True,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            rule_type=rule.rule_type,
            message=f"{trade.symbol}在受限名单中但允许交易",
        )

    def _check_order_size(
        self,
        rule: ComplianceRule,
        trade: TradeForCompliance
    ) -> ComplianceCheckResult:
        """检查订单大小"""
        max_order_value = rule.condition.get("max_order_value", float("inf"))
        max_order_quantity = rule.condition.get("max_order_quantity", float("inf"))

        order_value = trade.notional_value

        if order_value > max_order_value:
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                description=f"订单金额超限: {order_value:,.2f} > {max_order_value:,.2f}",
                details={
                    "order_value": order_value,
                    "max_order_value": max_order_value,
                },
                trade_id=trade.trade_id,
                account_id=trade.account_id,
                symbol=trade.symbol,
                trader_id=trade.trader_id,
            )

            return ComplianceCheckResult(
                passed=False,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=violation.description,
                details=violation.details,
                violation=violation,
            )

        if trade.quantity > max_order_quantity:
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                severity=rule.severity,
                description=f"订单数量超限: {trade.quantity:,.0f} > {max_order_quantity:,.0f}",
                details={
                    "order_quantity": trade.quantity,
                    "max_order_quantity": max_order_quantity,
                },
                trade_id=trade.trade_id,
                account_id=trade.account_id,
                symbol=trade.symbol,
                trader_id=trade.trader_id,
            )

            return ComplianceCheckResult(
                passed=False,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                rule_type=rule.rule_type,
                message=violation.description,
                details=violation.details,
                violation=violation,
            )

        return ComplianceCheckResult(
            passed=True,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            rule_type=rule.rule_type,
            message="订单大小在限制内",
        )

    def _add_trade_to_history(self, trade: TradeForCompliance):
        """添加交易到历史记录"""
        self.trade_history.append(trade)

        # 限制历史大小
        if len(self.trade_history) > self.trade_history_max_size:
            self.trade_history = self.trade_history[-self.trade_history_max_size:]

    def _trigger_alert(self, violation: ComplianceViolation):
        """触发告警"""
        should_alert = False

        if violation.severity == RuleSeverity.WARNING and self.config.alert_on_warning:
            should_alert = True
        elif violation.severity == RuleSeverity.VIOLATION and self.config.alert_on_violation:
            should_alert = True
        elif violation.severity == RuleSeverity.CRITICAL and self.config.alert_on_critical:
            should_alert = True

        if should_alert and self._on_alert:
            self._on_alert(f"compliance_{violation.severity.value}", {
                "violation_id": violation.violation_id,
                "rule_name": violation.rule_name,
                "description": violation.description,
                "severity": violation.severity.value,
            })

    # ==================== 违规管理 ====================

    def get_violation(self, violation_id: str) -> Optional[ComplianceViolation]:
        """获取违规详情"""
        return self.violations.get(violation_id)

    def get_violations(
        self,
        status: Optional[ViolationStatus] = None,
        severity: Optional[RuleSeverity] = None,
        account_id: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ComplianceViolation]:
        """获取违规列表"""
        violations = list(self.violations.values())

        if status:
            violations = [v for v in violations if v.status == status]

        if severity:
            violations = [v for v in violations if v.severity == severity]

        if account_id:
            violations = [v for v in violations if v.account_id == account_id]

        if symbol:
            violations = [v for v in violations if v.symbol == symbol]

        if start_time:
            violations = [v for v in violations if v.detected_at >= start_time]

        if end_time:
            violations = [v for v in violations if v.detected_at <= end_time]

        # 按时间倒序
        violations.sort(key=lambda v: v.detected_at, reverse=True)

        return violations[:limit]

    def update_violation_status(
        self,
        violation_id: str,
        new_status: ViolationStatus,
        notes: str = "",
        actor: str = "system"
    ) -> bool:
        """更新违规状态"""
        if violation_id not in self.violations:
            return False

        violation = self.violations[violation_id]
        old_status = violation.status

        violation.status = new_status
        violation.resolution_notes = notes

        if new_status == ViolationStatus.RESOLVED:
            violation.resolved_at = datetime.now()
            violation.resolved_by = actor
            self.stats["violations_resolved"] += 1

        self._add_audit_entry(
            action="update_violation_status",
            actor=actor,
            target_type="violation",
            target_id=violation_id,
            old_value={"status": old_status.value},
            new_value={"status": new_status.value, "notes": notes},
        )

        return True

    def assign_violation(
        self,
        violation_id: str,
        assignee: str,
        actor: str = "system"
    ) -> bool:
        """分配违规处理人"""
        if violation_id not in self.violations:
            return False

        violation = self.violations[violation_id]
        old_assignee = violation.assigned_to

        violation.assigned_to = assignee
        if violation.status == ViolationStatus.OPEN:
            violation.status = ViolationStatus.UNDER_REVIEW

        self._add_audit_entry(
            action="assign_violation",
            actor=actor,
            target_type="violation",
            target_id=violation_id,
            old_value={"assigned_to": old_assignee},
            new_value={"assigned_to": assignee},
        )

        return True

    # ==================== 审计日志 ====================

    def _add_audit_entry(
        self,
        action: str,
        actor: str,
        target_type: str,
        target_id: str,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ):
        """添加审计日志"""
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            action=action,
            actor=actor,
            target_type=target_type,
            target_id=target_id,
            old_value=old_value,
            new_value=new_value,
            notes=notes,
        )
        self.audit_log.append(entry)

    def get_audit_log(
        self,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        actor: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """获取审计日志"""
        entries = self.audit_log.copy()

        if target_type:
            entries = [e for e in entries if e.target_type == target_type]

        if target_id:
            entries = [e for e in entries if e.target_id == target_id]

        if actor:
            entries = [e for e in entries if e.actor == actor]

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]

        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        # 按时间倒序
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries[:limit]

    # ==================== 报告生成 ====================

    def generate_report(
        self,
        report_type: str = "daily",
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        generated_by: str = "system"
    ) -> ComplianceReport:
        """生成合规报告"""
        # 设置默认时间范围
        if not period_end:
            period_end = datetime.now()

        if not period_start:
            if report_type == "daily":
                period_start = period_end - timedelta(days=1)
            elif report_type == "weekly":
                period_start = period_end - timedelta(weeks=1)
            elif report_type == "monthly":
                period_start = period_end - timedelta(days=30)
            else:
                period_start = period_end - timedelta(days=1)

        # 获取期间内的违规
        violations = self.get_violations(
            start_time=period_start,
            end_time=period_end,
            limit=10000
        )

        # 统计
        violations_by_type: Dict[str, int] = {}
        violations_by_severity: Dict[str, int] = {}
        violations_by_account: Dict[str, int] = {}

        violations_resolved = 0
        violations_open = 0

        for v in violations:
            # 按类型统计
            type_key = v.rule_type.value
            violations_by_type[type_key] = violations_by_type.get(type_key, 0) + 1

            # 按严重性统计
            severity_key = v.severity.value
            violations_by_severity[severity_key] = violations_by_severity.get(severity_key, 0) + 1

            # 按账户统计
            if v.account_id:
                violations_by_account[v.account_id] = violations_by_account.get(v.account_id, 0) + 1

            # 状态统计
            if v.status == ViolationStatus.RESOLVED:
                violations_resolved += 1
            elif v.status in [ViolationStatus.OPEN, ViolationStatus.UNDER_REVIEW]:
                violations_open += 1

        # 计算合规率
        total_checks = self.stats["total_checks"]
        failed_checks = self.stats["failed_checks"]
        compliance_rate = ((total_checks - failed_checks) / total_checks * 100) if total_checks > 0 else 100.0

        # 统计期间交易数
        trades_in_period = len([
            t for t in self.trade_history
            if period_start <= t.timestamp <= period_end
        ])

        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            total_trades_checked=trades_in_period,
            violations_detected=len(violations),
            violations_resolved=violations_resolved,
            violations_open=violations_open,
            violations_by_type=violations_by_type,
            violations_by_severity=violations_by_severity,
            violations_by_account=violations_by_account,
            compliance_rate=compliance_rate,
            violations=violations[:100],  # 只包含最近100条
            generated_by=generated_by,
        )

        self._add_audit_entry(
            action="generate_report",
            actor=generated_by,
            target_type="report",
            target_id=report.report_id,
            new_value={"report_type": report_type},
        )

        return report

    # ==================== 监控控制 ====================

    def on_violation(self, callback: Callable[[ComplianceViolation], None]):
        """注册违规回调"""
        self._on_violation = callback

    def on_alert(self, callback: Callable[[str, Dict[str, Any]], None]):
        """注册告警回调"""
        self._on_alert = callback

    async def start_monitoring(self, check_interval: float = 60.0):
        """启动监控"""
        if self.state == MonitorState.RUNNING:
            return

        self.state = MonitorState.RUNNING
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(check_interval)
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

    async def _monitoring_loop(self, interval: float):
        """监控循环"""
        while self.state == MonitorState.RUNNING:
            try:
                # 清理过期的交易计数
                self._cleanup_trade_counts()

                # 自动生成每日报告
                if self.config.auto_generate_daily_report:
                    now = datetime.now()
                    if now.hour == 0 and now.minute < interval / 60:
                        self.generate_report("daily")

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # 记录错误但继续运行
                logger.exception("Compliance monitoring loop iteration failed; continuing")
                # Back off before retrying so a failure raised before the sleep
                # above cannot spin this loop hot.
                await asyncio.sleep(interval)

    def _cleanup_trade_counts(self):
        """清理过期的交易计数"""
        now = datetime.now()
        current_minute = now.strftime("%Y%m%d%H%M")
        current_hour = now.strftime("%Y%m%d%H")
        current_day = now.strftime("%Y%m%d")

        for account_id in list(self.trade_counts.keys()):
            counts = self.trade_counts[account_id]
            # 保留当前时间窗口的计数
            self.trade_counts[account_id] = {
                k: v for k, v in counts.items()
                if k >= current_day  # 只保留当天及以后的
            }

    # ==================== 统计和状态 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self.state.value,
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "restricted_securities": len(self.restricted_securities),
            "total_violations": len(self.violations),
            "open_violations": len([v for v in self.violations.values() if v.status == ViolationStatus.OPEN]),
            "trade_history_size": len(self.trade_history),
            "audit_log_size": len(self.audit_log),
            **self.stats,
        }

    def reset_statistics(self):
        """重置统计"""
        self.stats = {
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "violations_created": 0,
            "violations_resolved": 0,
        }


# ==================== 工厂函数 ====================

def create_compliance_monitor(
    config: Optional[ComplianceConfig] = None,
    position_provider: Optional[PositionProvider] = None,
    market_data_provider: Optional[MarketDataProvider] = None,
) -> ComplianceMonitor:
    """创建合规监控引擎"""
    return ComplianceMonitor(
        config=config,
        position_provider=position_provider,
        market_data_provider=market_data_provider,
    )


def create_compliance_rule(
    rule_type: RuleType,
    name: str,
    description: str,
    condition: Dict[str, Any],
    severity: RuleSeverity = RuleSeverity.WARNING,
    rule_id: Optional[str] = None,
) -> ComplianceRule:
    """创建合规规则"""
    return ComplianceRule(
        rule_id=rule_id or str(uuid.uuid4()),
        rule_type=rule_type,
        name=name,
        description=description,
        condition=condition,
        severity=severity,
    )


def create_trade_for_compliance(
    account_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    trade_id: Optional[str] = None,
    **kwargs
) -> TradeForCompliance:
    """创建待检查交易"""
    return TradeForCompliance(
        trade_id=trade_id or str(uuid.uuid4()),
        account_id=account_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        **kwargs
    )


def create_restricted_security(
    symbol: str,
    restriction_type: RestrictionType,
    reason: str,
    **kwargs
) -> RestrictedSecurity:
    """创建受限证券"""
    return RestrictedSecurity(
        symbol=symbol,
        restriction_type=restriction_type,
        reason=reason,
        **kwargs
    )


def quick_compliance_check(
    account_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    monitor: Optional[ComplianceMonitor] = None
) -> List[ComplianceCheckResult]:
    """快速合规检查"""
    monitor = monitor or create_compliance_monitor()
    trade = create_trade_for_compliance(
        account_id=account_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
    )
    return monitor.check_trade(trade)


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "RuleType",
    "RuleSeverity",
    "ViolationStatus",
    "RestrictionType",
    "MonitorState",
    # 数据类
    "ComplianceRule",
    "TradeForCompliance",
    "ComplianceViolation",
    "RestrictedSecurity",
    "AuditEntry",
    "ComplianceCheckResult",
    "ComplianceReport",
    "ComplianceConfig",
    # 接口
    "PositionProvider",
    "MarketDataProvider",
    # 模拟实现
    "SimulatedPositionProvider",
    "SimulatedMarketDataProvider",
    # 核心引擎
    "ComplianceMonitor",
    # 工厂函数
    "create_compliance_monitor",
    "create_compliance_rule",
    "create_trade_for_compliance",
    "create_restricted_security",
    "quick_compliance_check",
]
