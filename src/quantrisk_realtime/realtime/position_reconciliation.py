"""
持仓对账引擎模块
Position Reconciliation Engine

提供系统持仓与经纪商持仓的三路对账、差异检测和修复功能。

功能特性:
- 三路对账: 系统持仓 vs 订单累计 vs 经纪商持仓
- 差异检测与分类
- 自动/手动对账修复
- 对账审计日志
- 定时对账任务
- 多账户支持

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid
from ..core.simulated import simulated_default

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class ReconciliationStatus(Enum):
    """对账状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    MATCHED = "matched"
    DISCREPANCY = "discrepancy"
    RESOLVED = "resolved"
    FAILED = "failed"


class DiscrepancyType(Enum):
    """差异类型"""
    QUANTITY_MISMATCH = "quantity_mismatch"  # 数量不匹配
    PRICE_MISMATCH = "price_mismatch"  # 成本价不匹配
    MISSING_IN_SYSTEM = "missing_in_system"  # 系统缺失
    MISSING_AT_BROKER = "missing_at_broker"  # 经纪商缺失
    SIDE_MISMATCH = "side_mismatch"  # 方向不匹配
    ACCOUNT_MISMATCH = "account_mismatch"  # 账户不匹配


class ResolutionAction(Enum):
    """修复动作"""
    NO_ACTION = "no_action"
    ADJUST_SYSTEM = "adjust_system"  # 调整系统持仓
    ADJUST_BROKER = "adjust_broker"  # 调整经纪商持仓（通常需要人工）
    CREATE_POSITION = "create_position"  # 创建持仓
    CLOSE_POSITION = "close_position"  # 关闭持仓
    MANUAL_REVIEW = "manual_review"  # 人工审核


class ReconciliationSource(Enum):
    """对账数据源"""
    SYSTEM = "system"  # 系统内部持仓
    ORDER_DERIVED = "order_derived"  # 订单推导持仓
    BROKER = "broker"  # 经纪商持仓
    EXCHANGE = "exchange"  # 交易所持仓


class ReconciliationFrequency(Enum):
    """对账频率"""
    REALTIME = "realtime"  # 实时
    MINUTE = "minute"  # 每分钟
    HOURLY = "hourly"  # 每小时
    DAILY = "daily"  # 每日
    ON_DEMAND = "on_demand"  # 按需


class ReconciliationState(Enum):
    """引擎状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ==================== 数据结构 ====================

@dataclass
class PositionSnapshot:
    """持仓快照"""
    symbol: str
    account_id: str
    quantity: Decimal
    avg_cost: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    side: str  # "long" or "short"
    currency: str = "USD"
    source: ReconciliationSource = ReconciliationSource.SYSTEM
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "account_id": self.account_id,
            "quantity": float(self.quantity),
            "avg_cost": float(self.avg_cost),
            "market_value": float(self.market_value),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "side": self.side,
            "currency": self.currency,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Discrepancy:
    """差异记录"""
    discrepancy_id: str
    symbol: str
    account_id: str
    discrepancy_type: DiscrepancyType
    system_value: Optional[Any]
    broker_value: Optional[Any]
    order_derived_value: Optional[Any]
    difference: Decimal
    severity: str  # "low", "medium", "high", "critical"
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_action: Optional[ResolutionAction] = None
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "discrepancy_id": self.discrepancy_id,
            "symbol": self.symbol,
            "account_id": self.account_id,
            "discrepancy_type": self.discrepancy_type.value,
            "system_value": self._convert_value(self.system_value),
            "broker_value": self._convert_value(self.broker_value),
            "order_derived_value": self._convert_value(self.order_derived_value),
            "difference": float(self.difference),
            "severity": self.severity,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_action": self.resolution_action.value if self.resolution_action else None,
            "resolution_notes": self.resolution_notes,
        }

    def _convert_value(self, value: Any) -> Any:
        """转换值"""
        if isinstance(value, Decimal):
            return float(value)
        return value


@dataclass
class ReconciliationResult:
    """对账结果"""
    reconciliation_id: str
    account_id: str
    status: ReconciliationStatus
    start_time: datetime
    end_time: Optional[datetime]
    total_positions: int
    matched_positions: int
    discrepancies: List[Discrepancy]
    system_positions: List[PositionSnapshot]
    broker_positions: List[PositionSnapshot]
    order_derived_positions: List[PositionSnapshot]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "reconciliation_id": self.reconciliation_id,
            "account_id": self.account_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_positions": self.total_positions,
            "matched_positions": self.matched_positions,
            "discrepancy_count": len(self.discrepancies),
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "summary": self.summary,
        }


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    entry_id: str
    timestamp: datetime
    action: str
    account_id: str
    symbol: Optional[str]
    details: Dict[str, Any]
    user_id: Optional[str] = None
    ip_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "details": self.details,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
        }


@dataclass
class ReconciliationConfig:
    """对账配置"""
    frequency: ReconciliationFrequency = ReconciliationFrequency.HOURLY
    auto_resolve_threshold: Decimal = Decimal("0.01")  # 自动修复阈值（差异比例）
    quantity_tolerance: Decimal = Decimal("0.0001")  # 数量容差
    price_tolerance: Decimal = Decimal("0.01")  # 价格容差
    enable_auto_resolve: bool = False  # 是否启用自动修复
    notify_on_discrepancy: bool = True  # 发现差异时通知
    critical_threshold: Decimal = Decimal("1000")  # 严重阈值（金额）
    max_history_days: int = 30  # 历史记录保留天数
    excluded_symbols: Set[str] = field(default_factory=set)  # 排除的标的


# ==================== 抽象接口 ====================

class BrokerPositionProvider(ABC):
    """经纪商持仓数据提供者接口"""

    @abstractmethod
    async def get_positions(self, account_id: str) -> List[PositionSnapshot]:
        """获取经纪商持仓"""
        pass

    @abstractmethod
    async def get_position(self, account_id: str, symbol: str) -> Optional[PositionSnapshot]:
        """获取单个持仓"""
        pass


class SystemPositionProvider(ABC):
    """系统持仓数据提供者接口"""

    @abstractmethod
    async def get_positions(self, account_id: str) -> List[PositionSnapshot]:
        """获取系统持仓"""
        pass

    @abstractmethod
    async def update_position(self, position: PositionSnapshot) -> bool:
        """更新系统持仓"""
        pass

    @abstractmethod
    async def create_position(self, position: PositionSnapshot) -> bool:
        """创建系统持仓"""
        pass


class OrderHistoryProvider(ABC):
    """订单历史数据提供者接口"""

    @abstractmethod
    async def get_filled_orders(
        self,
        account_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """获取已成交订单"""
        pass


# ==================== 模拟实现 ====================

class SimulatedBrokerProvider(BrokerPositionProvider):
    """模拟经纪商持仓提供者"""

    def __init__(self):
        self.positions: Dict[str, Dict[str, PositionSnapshot]] = {}

    async def get_positions(self, account_id: str) -> List[PositionSnapshot]:
        """获取经纪商持仓"""
        return list(self.positions.get(account_id, {}).values())

    async def get_position(self, account_id: str, symbol: str) -> Optional[PositionSnapshot]:
        """获取单个持仓"""
        return self.positions.get(account_id, {}).get(symbol)

    def set_position(self, position: PositionSnapshot):
        """设置持仓（用于测试）"""
        if position.account_id not in self.positions:
            self.positions[position.account_id] = {}
        self.positions[position.account_id][position.symbol] = position


class SimulatedSystemProvider(SystemPositionProvider):
    """模拟系统持仓提供者"""

    def __init__(self):
        self.positions: Dict[str, Dict[str, PositionSnapshot]] = {}

    async def get_positions(self, account_id: str) -> List[PositionSnapshot]:
        """获取系统持仓"""
        return list(self.positions.get(account_id, {}).values())

    async def update_position(self, position: PositionSnapshot) -> bool:
        """更新系统持仓"""
        if position.account_id not in self.positions:
            self.positions[position.account_id] = {}
        self.positions[position.account_id][position.symbol] = position
        return True

    async def create_position(self, position: PositionSnapshot) -> bool:
        """创建系统持仓"""
        return await self.update_position(position)

    def set_position(self, position: PositionSnapshot):
        """设置持仓（用于测试）"""
        if position.account_id not in self.positions:
            self.positions[position.account_id] = {}
        self.positions[position.account_id][position.symbol] = position


class SimulatedOrderHistoryProvider(OrderHistoryProvider):
    """模拟订单历史提供者"""

    def __init__(self):
        self.orders: List[Dict[str, Any]] = []

    async def get_filled_orders(
        self,
        account_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """获取已成交订单"""
        return [
            order for order in self.orders
            if order.get("account_id") == account_id
            and start_time <= order.get("fill_time", datetime.now()) <= end_time
            and order.get("status") == "filled"
        ]

    def add_order(self, order: Dict[str, Any]):
        """添加订单（用于测试）"""
        self.orders.append(order)


# ==================== 核心引擎 ====================

class PositionReconciliationEngine:
    """持仓对账引擎"""

    def __init__(
        self,
        broker_provider: BrokerPositionProvider,
        system_provider: SystemPositionProvider,
        order_provider: Optional[OrderHistoryProvider] = None,
        config: Optional[ReconciliationConfig] = None,
    ):
        self.broker_provider = broker_provider
        self.system_provider = system_provider
        self.order_provider = order_provider
        self.config = config or ReconciliationConfig()

        # 状态
        self.state = ReconciliationState.IDLE
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # 历史记录
        self.reconciliation_history: List[ReconciliationResult] = []
        self.audit_log: List[AuditLogEntry] = []

        # 回调
        self._on_discrepancy: Optional[Callable[[Discrepancy], None]] = None
        self._on_resolution: Optional[Callable[[Discrepancy], None]] = None
        self._on_reconciliation_complete: Optional[Callable[[ReconciliationResult], None]] = None

        # 统计
        self._stats = {
            "total_reconciliations": 0,
            "total_discrepancies": 0,
            "auto_resolved": 0,
            "manual_resolved": 0,
            "last_reconciliation": None,
        }

    # ==================== 回调注册 ====================

    def on_discrepancy(self, callback: Callable[[Discrepancy], None]):
        """注册差异发现回调"""
        self._on_discrepancy = callback

    def on_resolution(self, callback: Callable[[Discrepancy], None]):
        """注册修复完成回调"""
        self._on_resolution = callback

    def on_reconciliation_complete(self, callback: Callable[[ReconciliationResult], None]):
        """注册对账完成回调"""
        self._on_reconciliation_complete = callback

    # ==================== 核心对账逻辑 ====================

    async def reconcile(self, account_id: str) -> ReconciliationResult:
        """执行对账"""
        reconciliation_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        self._log_audit(
            action="reconciliation_started",
            account_id=account_id,
            details={"reconciliation_id": reconciliation_id},
        )

        try:
            # 获取三方持仓数据
            system_positions = await self.system_provider.get_positions(account_id)
            broker_positions = await self.broker_provider.get_positions(account_id)
            order_derived_positions = await self._derive_positions_from_orders(account_id)

            # 执行三路对账
            discrepancies = self._compare_positions(
                system_positions,
                broker_positions,
                order_derived_positions,
                account_id,
            )

            # 计算匹配数
            all_symbols = set()
            for pos in system_positions + broker_positions:
                all_symbols.add(pos.symbol)

            matched_count = len(all_symbols) - len(discrepancies)

            # 确定状态
            status = ReconciliationStatus.MATCHED if not discrepancies else ReconciliationStatus.DISCREPANCY

            # 创建结果
            result = ReconciliationResult(
                reconciliation_id=reconciliation_id,
                account_id=account_id,
                status=status,
                start_time=start_time,
                end_time=datetime.now(),
                total_positions=len(all_symbols),
                matched_positions=max(0, matched_count),
                discrepancies=discrepancies,
                system_positions=system_positions,
                broker_positions=broker_positions,
                order_derived_positions=order_derived_positions,
                summary=self._generate_summary(discrepancies),
            )

            # 保存历史
            self.reconciliation_history.append(result)
            self._stats["total_reconciliations"] += 1
            self._stats["total_discrepancies"] += len(discrepancies)
            self._stats["last_reconciliation"] = datetime.now().isoformat()

            # 触发回调
            for disc in discrepancies:
                if self._on_discrepancy:
                    self._on_discrepancy(disc)

            if self._on_reconciliation_complete:
                self._on_reconciliation_complete(result)

            # 自动修复
            if self.config.enable_auto_resolve:
                await self._auto_resolve(discrepancies)

            self._log_audit(
                action="reconciliation_completed",
                account_id=account_id,
                details={
                    "reconciliation_id": reconciliation_id,
                    "status": status.value,
                    "discrepancy_count": len(discrepancies),
                },
            )

            return result

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            self._log_audit(
                action="reconciliation_failed",
                account_id=account_id,
                details={"error": str(e)},
            )
            raise

    def _compare_positions(
        self,
        system_positions: List[PositionSnapshot],
        broker_positions: List[PositionSnapshot],
        order_derived_positions: List[PositionSnapshot],
        account_id: str,
    ) -> List[Discrepancy]:
        """比较持仓并生成差异"""
        discrepancies = []

        # 转换为字典便于查找
        system_dict = {pos.symbol: pos for pos in system_positions}
        broker_dict = {pos.symbol: pos for pos in broker_positions}
        order_dict = {pos.symbol: pos for pos in order_derived_positions}

        # 所有标的
        all_symbols = set(system_dict.keys()) | set(broker_dict.keys())

        for symbol in all_symbols:
            # 跳过排除的标的
            if symbol in self.config.excluded_symbols:
                continue

            system_pos = system_dict.get(symbol)
            broker_pos = broker_dict.get(symbol)
            order_pos = order_dict.get(symbol)

            # 检查缺失
            if system_pos and not broker_pos:
                discrepancies.append(self._create_discrepancy(
                    symbol=symbol,
                    account_id=account_id,
                    disc_type=DiscrepancyType.MISSING_AT_BROKER,
                    system_value=float(system_pos.quantity),
                    broker_value=None,
                    order_value=float(order_pos.quantity) if order_pos else None,
                    difference=system_pos.quantity,
                    description=f"Position {symbol} exists in system but not at broker",
                ))
            elif broker_pos and not system_pos:
                discrepancies.append(self._create_discrepancy(
                    symbol=symbol,
                    account_id=account_id,
                    disc_type=DiscrepancyType.MISSING_IN_SYSTEM,
                    system_value=None,
                    broker_value=float(broker_pos.quantity),
                    order_value=float(order_pos.quantity) if order_pos else None,
                    difference=broker_pos.quantity,
                    description=f"Position {symbol} exists at broker but not in system",
                ))
            elif system_pos and broker_pos:
                # 检查数量差异
                qty_diff = abs(system_pos.quantity - broker_pos.quantity)
                if qty_diff > self.config.quantity_tolerance:
                    discrepancies.append(self._create_discrepancy(
                        symbol=symbol,
                        account_id=account_id,
                        disc_type=DiscrepancyType.QUANTITY_MISMATCH,
                        system_value=float(system_pos.quantity),
                        broker_value=float(broker_pos.quantity),
                        order_value=float(order_pos.quantity) if order_pos else None,
                        difference=qty_diff,
                        description=f"Quantity mismatch for {symbol}: system={system_pos.quantity}, broker={broker_pos.quantity}",
                    ))

                # 检查方向差异
                if system_pos.side != broker_pos.side:
                    discrepancies.append(self._create_discrepancy(
                        symbol=symbol,
                        account_id=account_id,
                        disc_type=DiscrepancyType.SIDE_MISMATCH,
                        system_value=system_pos.side,
                        broker_value=broker_pos.side,
                        order_value=order_pos.side if order_pos else None,
                        difference=Decimal(0),
                        description=f"Side mismatch for {symbol}: system={system_pos.side}, broker={broker_pos.side}",
                    ))

                # 检查成本价差异
                price_diff = abs(system_pos.avg_cost - broker_pos.avg_cost)
                if price_diff > self.config.price_tolerance:
                    discrepancies.append(self._create_discrepancy(
                        symbol=symbol,
                        account_id=account_id,
                        disc_type=DiscrepancyType.PRICE_MISMATCH,
                        system_value=float(system_pos.avg_cost),
                        broker_value=float(broker_pos.avg_cost),
                        order_value=float(order_pos.avg_cost) if order_pos else None,
                        difference=price_diff,
                        description=f"Average cost mismatch for {symbol}: system={system_pos.avg_cost}, broker={broker_pos.avg_cost}",
                    ))

        return discrepancies

    def _create_discrepancy(
        self,
        symbol: str,
        account_id: str,
        disc_type: DiscrepancyType,
        system_value: Any,
        broker_value: Any,
        order_value: Any,
        difference: Decimal,
        description: str,
    ) -> Discrepancy:
        """创建差异记录"""
        # 计算严重程度
        severity = self._calculate_severity(difference, disc_type)

        return Discrepancy(
            discrepancy_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            account_id=account_id,
            discrepancy_type=disc_type,
            system_value=system_value,
            broker_value=broker_value,
            order_derived_value=order_value,
            difference=difference if isinstance(difference, Decimal) else Decimal(str(difference)),
            severity=severity,
            description=description,
        )

    def _calculate_severity(self, difference: Decimal, disc_type: DiscrepancyType) -> str:
        """计算差异严重程度"""
        if disc_type in [DiscrepancyType.MISSING_IN_SYSTEM, DiscrepancyType.MISSING_AT_BROKER]:
            return "critical"

        if disc_type == DiscrepancyType.SIDE_MISMATCH:
            return "critical"

        abs_diff = abs(float(difference))
        if abs_diff >= float(self.config.critical_threshold):
            return "critical"
        elif abs_diff >= float(self.config.critical_threshold) / 10:
            return "high"
        elif abs_diff >= float(self.config.critical_threshold) / 100:
            return "medium"
        else:
            return "low"

    async def _derive_positions_from_orders(self, account_id: str) -> List[PositionSnapshot]:
        """从订单历史推导持仓"""
        if not self.order_provider:
            return []

        try:
            # 获取最近的订单历史
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.config.max_history_days)

            orders = await self.order_provider.get_filled_orders(account_id, start_time, end_time)

            # 聚合持仓
            positions: Dict[str, Dict[str, Any]] = {}

            for order in orders:
                symbol = order.get("symbol")
                if not symbol:
                    continue

                if symbol not in positions:
                    positions[symbol] = {
                        "quantity": Decimal(0),
                        "total_cost": Decimal(0),
                        "realized_pnl": Decimal(0),
                    }

                qty = Decimal(str(order.get("filled_quantity", 0)))
                price = Decimal(str(order.get("avg_price", 0)))
                side = order.get("side", "buy")

                if side == "buy":
                    positions[symbol]["quantity"] += qty
                    positions[symbol]["total_cost"] += qty * price
                else:
                    positions[symbol]["quantity"] -= qty
                    positions[symbol]["total_cost"] -= qty * price

            # 转换为 PositionSnapshot
            result = []
            for symbol, data in positions.items():
                if data["quantity"] != 0:
                    avg_cost = data["total_cost"] / data["quantity"] if data["quantity"] else Decimal(0)
                    result.append(PositionSnapshot(
                        symbol=symbol,
                        account_id=account_id,
                        quantity=abs(data["quantity"]),
                        avg_cost=abs(avg_cost),
                        market_value=Decimal(0),  # 需要市场价格
                        unrealized_pnl=Decimal(0),
                        realized_pnl=data["realized_pnl"],
                        side="long" if data["quantity"] > 0 else "short",
                        source=ReconciliationSource.ORDER_DERIVED,
                    ))

            return result

        except Exception as e:
            logger.error(f"Failed to derive positions from orders: {e}")
            return []

    def _generate_summary(self, discrepancies: List[Discrepancy]) -> Dict[str, Any]:
        """生成对账摘要"""
        summary = {
            "total_discrepancies": len(discrepancies),
            "by_type": {},
            "by_severity": {},
            "critical_items": [],
        }

        for disc in discrepancies:
            # 按类型统计
            disc_type = disc.discrepancy_type.value
            summary["by_type"][disc_type] = summary["by_type"].get(disc_type, 0) + 1

            # 按严重程度统计
            summary["by_severity"][disc.severity] = summary["by_severity"].get(disc.severity, 0) + 1

            # 收集严重项
            if disc.severity == "critical":
                summary["critical_items"].append({
                    "symbol": disc.symbol,
                    "type": disc_type,
                    "description": disc.description,
                })

        return summary

    # ==================== 自动修复 ====================

    async def _auto_resolve(self, discrepancies: List[Discrepancy]):
        """自动修复差异"""
        for disc in discrepancies:
            if disc.resolved:
                continue

            # 只自动修复低严重程度的差异
            if disc.severity in ["low", "medium"]:
                # 检查差异是否在自动修复阈值内
                if disc.difference <= self.config.auto_resolve_threshold:
                    await self.resolve_discrepancy(
                        disc,
                        ResolutionAction.ADJUST_SYSTEM,
                        "Auto-resolved within tolerance",
                    )
                    self._stats["auto_resolved"] += 1

    async def resolve_discrepancy(
        self,
        discrepancy: Discrepancy,
        action: ResolutionAction,
        notes: str = "",
    ) -> bool:
        """修复差异"""
        try:
            if action == ResolutionAction.ADJUST_SYSTEM:
                # 调整系统持仓以匹配经纪商
                if discrepancy.broker_value is not None:
                    # 创建调整后的持仓
                    adjusted_position = PositionSnapshot(
                        symbol=discrepancy.symbol,
                        account_id=discrepancy.account_id,
                        quantity=Decimal(str(discrepancy.broker_value)),
                        avg_cost=Decimal(0),  # 需要从经纪商获取
                        market_value=Decimal(0),
                        unrealized_pnl=Decimal(0),
                        realized_pnl=Decimal(0),
                        side="long",
                        source=ReconciliationSource.SYSTEM,
                    )
                    await self.system_provider.update_position(adjusted_position)

            elif action == ResolutionAction.CREATE_POSITION:
                # 创建缺失的持仓
                if discrepancy.discrepancy_type == DiscrepancyType.MISSING_IN_SYSTEM:
                    new_position = PositionSnapshot(
                        symbol=discrepancy.symbol,
                        account_id=discrepancy.account_id,
                        quantity=Decimal(str(discrepancy.broker_value or 0)),
                        avg_cost=Decimal(0),
                        market_value=Decimal(0),
                        unrealized_pnl=Decimal(0),
                        realized_pnl=Decimal(0),
                        side="long",
                        source=ReconciliationSource.SYSTEM,
                    )
                    await self.system_provider.create_position(new_position)

            # 标记为已修复
            discrepancy.resolved = True
            discrepancy.resolved_at = datetime.now()
            discrepancy.resolution_action = action
            discrepancy.resolution_notes = notes

            self._stats["manual_resolved"] += 1

            self._log_audit(
                action="discrepancy_resolved",
                account_id=discrepancy.account_id,
                symbol=discrepancy.symbol,
                details={
                    "discrepancy_id": discrepancy.discrepancy_id,
                    "resolution_action": action.value,
                    "notes": notes,
                },
            )

            if self._on_resolution:
                self._on_resolution(discrepancy)

            return True

        except Exception as e:
            logger.error(f"Failed to resolve discrepancy: {e}")
            return False

    # ==================== 定时任务 ====================

    async def start(self, account_ids: List[str]):
        """启动定时对账"""
        if self._running:
            return

        self._running = True
        self.state = ReconciliationState.RUNNING
        self._task = asyncio.create_task(self._reconciliation_loop(account_ids))
        logger.info("Position reconciliation engine started")

    async def stop(self):
        """停止定时对账"""
        self._running = False
        self.state = ReconciliationState.IDLE
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Position reconciliation engine stopped")

    async def _reconciliation_loop(self, account_ids: List[str]):
        """对账循环"""
        interval = self._get_interval()

        while self._running:
            try:
                for account_id in account_ids:
                    await self.reconcile(account_id)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconciliation loop error: {e}")
                self.state = ReconciliationState.ERROR
                await asyncio.sleep(60)  # 错误后等待1分钟

    def _get_interval(self) -> float:
        """获取对账间隔（秒）"""
        intervals = {
            ReconciliationFrequency.REALTIME: 10,
            ReconciliationFrequency.MINUTE: 60,
            ReconciliationFrequency.HOURLY: 3600,
            ReconciliationFrequency.DAILY: 86400,
            ReconciliationFrequency.ON_DEMAND: 0,
        }
        return intervals.get(self.config.frequency, 3600)

    # ==================== 审计日志 ====================

    def _log_audit(
        self,
        action: str,
        account_id: str,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ):
        """记录审计日志"""
        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            action=action,
            account_id=account_id,
            symbol=symbol,
            details=details or {},
            user_id=user_id,
        )
        self.audit_log.append(entry)

        # 保持日志大小可控
        max_entries = 10000
        if len(self.audit_log) > max_entries:
            self.audit_log = self.audit_log[-max_entries:]

    # ==================== 查询接口 ====================

    def get_history(
        self,
        account_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ReconciliationResult]:
        """获取对账历史"""
        results = self.reconciliation_history
        if account_id:
            results = [r for r in results if r.account_id == account_id]
        return results[-limit:]

    def get_unresolved_discrepancies(
        self,
        account_id: Optional[str] = None,
    ) -> List[Discrepancy]:
        """获取未解决的差异"""
        discrepancies = []
        for result in self.reconciliation_history:
            if account_id and result.account_id != account_id:
                continue
            for disc in result.discrepancies:
                if not disc.resolved:
                    discrepancies.append(disc)
        return discrepancies

    def get_audit_log(
        self,
        account_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """获取审计日志"""
        logs = self.audit_log
        if account_id:
            logs = [log for log in logs if log.account_id == account_id]
        return logs[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "state": self.state.value,
            "config": {
                "frequency": self.config.frequency.value,
                "auto_resolve_enabled": self.config.enable_auto_resolve,
                "quantity_tolerance": float(self.config.quantity_tolerance),
                "price_tolerance": float(self.config.price_tolerance),
            },
            "history_count": len(self.reconciliation_history),
            "unresolved_count": len(self.get_unresolved_discrepancies()),
        }


# ==================== 工厂函数 ====================

def create_reconciliation_engine(
    broker_provider: Optional[BrokerPositionProvider] = None,
    system_provider: Optional[SystemPositionProvider] = None,
    order_provider: Optional[OrderHistoryProvider] = None,
    frequency: str = "hourly",
    auto_resolve: bool = False,
) -> PositionReconciliationEngine:
    """创建对账引擎"""
    freq_map = {
        "realtime": ReconciliationFrequency.REALTIME,
        "minute": ReconciliationFrequency.MINUTE,
        "hourly": ReconciliationFrequency.HOURLY,
        "daily": ReconciliationFrequency.DAILY,
        "on_demand": ReconciliationFrequency.ON_DEMAND,
    }

    config = ReconciliationConfig(
        frequency=freq_map.get(frequency, ReconciliationFrequency.HOURLY),
        enable_auto_resolve=auto_resolve,
    )

    return PositionReconciliationEngine(
        broker_provider=broker_provider or simulated_default(SimulatedBrokerProvider, "data from create_reconciliation_engine's simulated broker_provider (SimulatedBrokerProvider)"),
        system_provider=system_provider or simulated_default(SimulatedSystemProvider, "data from create_reconciliation_engine's simulated system_provider (SimulatedSystemProvider)"),
        order_provider=order_provider,
        config=config,
    )


def create_position_snapshot(
    symbol: str,
    account_id: str,
    quantity: float,
    avg_cost: float,
    side: str = "long",
    market_value: float = 0,
    unrealized_pnl: float = 0,
    realized_pnl: float = 0,
    source: str = "system",
) -> PositionSnapshot:
    """创建持仓快照"""
    source_map = {
        "system": ReconciliationSource.SYSTEM,
        "broker": ReconciliationSource.BROKER,
        "order_derived": ReconciliationSource.ORDER_DERIVED,
        "exchange": ReconciliationSource.EXCHANGE,
    }

    return PositionSnapshot(
        symbol=symbol,
        account_id=account_id,
        quantity=Decimal(str(quantity)),
        avg_cost=Decimal(str(avg_cost)),
        market_value=Decimal(str(market_value)),
        unrealized_pnl=Decimal(str(unrealized_pnl)),
        realized_pnl=Decimal(str(realized_pnl)),
        side=side,
        source=source_map.get(source, ReconciliationSource.SYSTEM),
    )


async def quick_reconcile(
    system_positions: List[Dict[str, Any]],
    broker_positions: List[Dict[str, Any]],
    account_id: str = "default",
) -> Dict[str, Any]:
    """快速对账（便捷函数）"""
    # 创建模拟提供者
    system_provider = SimulatedSystemProvider()
    broker_provider = SimulatedBrokerProvider()

    # 设置持仓
    for pos in system_positions:
        snapshot = create_position_snapshot(
            symbol=pos["symbol"],
            account_id=account_id,
            quantity=pos.get("quantity", 0),
            avg_cost=pos.get("avg_cost", 0),
            side=pos.get("side", "long"),
            source="system",
        )
        system_provider.set_position(snapshot)

    for pos in broker_positions:
        snapshot = create_position_snapshot(
            symbol=pos["symbol"],
            account_id=account_id,
            quantity=pos.get("quantity", 0),
            avg_cost=pos.get("avg_cost", 0),
            side=pos.get("side", "long"),
            source="broker",
        )
        broker_provider.set_position(snapshot)

    # 创建引擎并执行对账
    engine = PositionReconciliationEngine(
        broker_provider=broker_provider,
        system_provider=system_provider,
    )

    result = await engine.reconcile(account_id)

    return {
        "status": result.status.value,
        "matched": result.matched_positions,
        "total": result.total_positions,
        "discrepancies": [d.to_dict() for d in result.discrepancies],
        "summary": result.summary,
    }


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "ReconciliationStatus",
    "DiscrepancyType",
    "ResolutionAction",
    "ReconciliationSource",
    "ReconciliationFrequency",
    "ReconciliationState",
    # 数据类
    "PositionSnapshot",
    "Discrepancy",
    "ReconciliationResult",
    "AuditLogEntry",
    "ReconciliationConfig",
    # 接口
    "BrokerPositionProvider",
    "SystemPositionProvider",
    "OrderHistoryProvider",
    # 模拟实现
    "SimulatedBrokerProvider",
    "SimulatedSystemProvider",
    "SimulatedOrderHistoryProvider",
    # 核心引擎
    "PositionReconciliationEngine",
    # 工厂函数
    "create_reconciliation_engine",
    "create_position_snapshot",
    "quick_reconcile",
]
