"""
实时保证金监控器

提供期货/杠杆交易的实时保证金监控、预警和强平机制。
这是P0级紧急风控模块，防止保证金爆仓。

核心功能:
1. 每秒检查保证金状态
2. 分级预警 (注意/警告/危险/紧急)
3. 自动强平机制
4. 完整审计日志

使用方式:
    monitor = RealtimeMarginMonitor(portfolio_service, alert_service)
    await monitor.start()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from decimal import Decimal
import uuid

from ..core.mock_guard import MockDataDisabledError, require_real_data

logger = logging.getLogger(__name__)


# ============ 枚举定义 ============

class AlertLevel(str, Enum):
    """告警等级"""
    NORMAL = "normal"       # 正常 (保证金使用率 < 60%)
    NOTICE = "notice"       # 注意 (60% - 70%)
    WARNING = "warning"     # 警告 (70% - 85%)
    DANGER = "danger"       # 危险 (85% - 95%)
    EMERGENCY = "emergency" # 紧急 (>= 95%) - 触发强平


class LiquidationReason(str, Enum):
    """强平原因"""
    MARGIN_CALL = "margin_call"           # 保证金不足
    DAILY_LOSS_LIMIT = "daily_loss_limit" # 单日亏损超限
    DRAWDOWN_LIMIT = "drawdown_limit"     # 回撤超限
    MANUAL = "manual"                      # 手动触发
    SYSTEM = "system"                      # 系统维护


class MonitorState(str, Enum):
    """监控器状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ============ 数据类定义 ============

@dataclass
class MarginThresholds:
    """保证金阈值配置"""
    notice_ratio: float = 0.60      # 注意阈值 (60%)
    warning_ratio: float = 0.70     # 警告阈值 (70%)
    danger_ratio: float = 0.85      # 危险阈值 (85%)
    emergency_ratio: float = 0.95   # 紧急阈值 (95%) - 触发强平

    # 回撤阈值
    max_daily_loss: float = 0.05    # 单日最大亏损 (5%)
    max_drawdown: float = 0.10      # 最大回撤 (10%)

    # 监控间隔
    check_interval_seconds: float = 1.0  # 检查间隔 (秒)

    def get_alert_level(self, margin_ratio: float) -> AlertLevel:
        """根据保证金使用率获取告警等级"""
        if margin_ratio >= self.emergency_ratio:
            return AlertLevel.EMERGENCY
        elif margin_ratio >= self.danger_ratio:
            return AlertLevel.DANGER
        elif margin_ratio >= self.warning_ratio:
            return AlertLevel.WARNING
        elif margin_ratio >= self.notice_ratio:
            return AlertLevel.NOTICE
        else:
            return AlertLevel.NORMAL


@dataclass
class MarginStatus:
    """保证金状态快照"""
    # 基础信息
    timestamp: datetime = field(default_factory=datetime.now)
    account_id: str = ""

    # 资金状态
    total_equity: float = 0.0           # 总权益
    available_margin: float = 0.0       # 可用保证金
    used_margin: float = 0.0            # 已用保证金
    maintenance_margin: float = 0.0     # 维持保证金

    # 盈亏状态
    unrealized_pnl: float = 0.0         # 未实现盈亏
    realized_pnl_today: float = 0.0     # 今日已实现盈亏

    # 风险指标
    margin_ratio: float = 0.0           # 保证金使用率 (used/available)
    margin_safety_ratio: float = 0.0    # 保证金安全比率 (equity/maintenance)
    daily_pnl_ratio: float = 0.0        # 今日盈亏比率
    current_drawdown: float = 0.0       # 当前回撤

    # 告警状态
    alert_level: AlertLevel = AlertLevel.NORMAL
    alert_message: str = ""

    # 持仓信息
    position_count: int = 0
    largest_position_ratio: float = 0.0  # 最大持仓占比

    def is_safe(self) -> bool:
        """是否安全"""
        return self.alert_level in [AlertLevel.NORMAL, AlertLevel.NOTICE]

    def needs_action(self) -> bool:
        """是否需要采取行动"""
        return self.alert_level in [AlertLevel.DANGER, AlertLevel.EMERGENCY]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'account_id': self.account_id,
            'total_equity': self.total_equity,
            'available_margin': self.available_margin,
            'used_margin': self.used_margin,
            'margin_ratio': self.margin_ratio,
            'margin_safety_ratio': self.margin_safety_ratio,
            'unrealized_pnl': self.unrealized_pnl,
            'daily_pnl_ratio': self.daily_pnl_ratio,
            'current_drawdown': self.current_drawdown,
            'alert_level': self.alert_level.value,
            'alert_message': self.alert_message,
            'position_count': self.position_count,
        }


@dataclass
class MarginAlert:
    """保证金告警"""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    account_id: str = ""
    level: AlertLevel = AlertLevel.NORMAL
    message: str = ""
    margin_status: Optional[MarginStatus] = None

    # 通知状态
    notified: bool = False
    notified_channels: List[str] = field(default_factory=list)

    # 处理状态
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'account_id': self.account_id,
            'level': self.level.value,
            'message': self.message,
            'margin_status': self.margin_status.to_dict() if self.margin_status else None,
            'notified': self.notified,
            'acknowledged': self.acknowledged,
        }


@dataclass
class LiquidationOrder:
    """强平订单"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    account_id: str = ""

    # 强平原因
    reason: LiquidationReason = LiquidationReason.MARGIN_CALL
    trigger_margin_ratio: float = 0.0

    # 强平内容
    positions_to_close: List[Dict[str, Any]] = field(default_factory=list)
    target_margin_ratio: float = 0.50  # 目标保证金使用率

    # 执行状态
    status: str = "pending"  # pending, executing, completed, failed, cancelled
    executed_at: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None

    # 审计信息
    approved_by: str = "system"  # system 或 用户ID
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def add_audit_log(self, action: str, details: Dict[str, Any]):
        """添加审计日志"""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
        })

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'timestamp': self.timestamp.isoformat(),
            'account_id': self.account_id,
            'reason': self.reason.value,
            'trigger_margin_ratio': self.trigger_margin_ratio,
            'positions_to_close': self.positions_to_close,
            'target_margin_ratio': self.target_margin_ratio,
            'status': self.status,
            'approved_by': self.approved_by,
            'audit_log': self.audit_log,
        }


# ============ 主类: 实时保证金监控器 ============

class RealtimeMarginMonitor:
    """
    实时保证金监控器

    核心职责:
    1. 持续监控保证金状态 (每秒)
    2. 分级预警通知
    3. 触发强平机制
    4. 记录完整审计日志

    使用示例:
        monitor = RealtimeMarginMonitor(
            portfolio_service=portfolio_service,
            alert_callback=send_alert,
            liquidation_callback=execute_liquidation
        )

        # 启动监控
        await monitor.start()

        # 停止监控
        await monitor.stop()
    """

    def __init__(
        self,
        portfolio_service: Any = None,
        thresholds: Optional[MarginThresholds] = None,
        alert_callback: Optional[Callable] = None,
        liquidation_callback: Optional[Callable] = None,
        account_id: str = "default",
    ):
        """
        初始化监控器

        Args:
            portfolio_service: 组合服务，用于获取持仓和资金信息
            thresholds: 保证金阈值配置
            alert_callback: 告警回调函数 async def callback(alert: MarginAlert)
            liquidation_callback: 强平回调函数 async def callback(order: LiquidationOrder)
            account_id: 账户ID
        """
        self.portfolio_service = portfolio_service
        self.thresholds = thresholds or MarginThresholds()
        self.alert_callback = alert_callback
        self.liquidation_callback = liquidation_callback
        self.account_id = account_id

        # 状态管理
        self._state = MonitorState.STOPPED
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_status: Optional[MarginStatus] = None
        self._last_alert_level = AlertLevel.NORMAL

        # 历史记录 (内存缓存最近1000条)
        self._status_history: List[MarginStatus] = []
        self._alert_history: List[MarginAlert] = []
        self._liquidation_history: List[LiquidationOrder] = []
        self._max_history_size = 1000

        # 统计信息
        self._check_count = 0
        self._alert_count = 0
        self._liquidation_count = 0
        self._start_time: Optional[datetime] = None

        # 去重: 避免重复告警
        self._alert_cooldown: Dict[AlertLevel, datetime] = {}
        self._alert_cooldown_seconds = {
            AlertLevel.NOTICE: 300,    # 5分钟
            AlertLevel.WARNING: 60,    # 1分钟
            AlertLevel.DANGER: 30,     # 30秒
            AlertLevel.EMERGENCY: 0,   # 立即
        }

    @property
    def state(self) -> MonitorState:
        """获取监控器状态"""
        return self._state

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._state == MonitorState.RUNNING

    @property
    def last_status(self) -> Optional[MarginStatus]:
        """获取最近的保证金状态"""
        return self._last_status

    # ============ 生命周期管理 ============

    async def start(self):
        """启动监控"""
        if self._state == MonitorState.RUNNING:
            logger.warning("Margin monitor is already running")
            return

        logger.info(f"Starting margin monitor for account {self.account_id}")
        self._state = MonitorState.STARTING
        self._start_time = datetime.now()

        try:
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            self._state = MonitorState.RUNNING
            logger.info("Margin monitor started successfully")
        except Exception as e:
            self._state = MonitorState.ERROR
            logger.error(f"Failed to start margin monitor: {e}")
            raise

    async def stop(self):
        """停止监控"""
        if self._state == MonitorState.STOPPED:
            logger.warning("Margin monitor is already stopped")
            return

        logger.info("Stopping margin monitor...")
        self._state = MonitorState.STOPPED

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Margin monitor stopped")

    async def pause(self):
        """暂停监控"""
        if self._state == MonitorState.RUNNING:
            self._state = MonitorState.PAUSED
            logger.info("Margin monitor paused")

    async def resume(self):
        """恢复监控"""
        if self._state == MonitorState.PAUSED:
            self._state = MonitorState.RUNNING
            logger.info("Margin monitor resumed")

    # ============ 核心监控循环 ============

    async def _monitoring_loop(self):
        """监控主循环 - 每秒检查保证金状态"""
        logger.info("Entering monitoring loop")

        while self._state in [MonitorState.RUNNING, MonitorState.PAUSED]:
            try:
                if self._state == MonitorState.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # 1. 获取保证金状态
                status = await self._get_margin_status()
                self._last_status = status
                self._check_count += 1

                # 2. 添加到历史记录
                self._add_to_history(status)

                # 3. 检查是否需要告警
                if status.alert_level != AlertLevel.NORMAL:
                    await self._process_alert(status)

                # 4. 检查是否需要强平
                if status.alert_level == AlertLevel.EMERGENCY:
                    await self._trigger_liquidation(status)

                # 5. 等待下一次检查
                await asyncio.sleep(self.thresholds.check_interval_seconds)

            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                self._state = MonitorState.ERROR
                await asyncio.sleep(5)  # 错误后等待5秒重试
                self._state = MonitorState.RUNNING

        logger.info("Exiting monitoring loop")

    async def _get_margin_status(self) -> MarginStatus:
        """获取当前保证金状态"""
        status = MarginStatus(
            timestamp=datetime.now(),
            account_id=self.account_id,
        )

        try:
            if self.portfolio_service:
                # 从组合服务获取真实数据
                portfolio_data = await self._fetch_portfolio_data()
                status = self._parse_portfolio_data(portfolio_data, status)
            else:
                # 无服务时使用模拟数据 (仅用于测试) - only on explicit opt-in, like PnLStreamer
                require_real_data(
                    f"margin status for account {self.account_id}",
                    "Construct the monitor with a portfolio_service.",
                )
                status = await self._get_mock_status(status)

            # 计算告警等级
            status.alert_level = self.thresholds.get_alert_level(status.margin_ratio)
            status.alert_message = self._generate_alert_message(status)

        except MockDataDisabledError:
            raise
        except Exception as e:
            logger.error(f"Failed to get margin status: {e}")
            status.alert_level = AlertLevel.DANGER
            status.alert_message = f"无法获取保证金状态: {str(e)}"

        return status

    async def _fetch_portfolio_data(self) -> Dict[str, Any]:
        """从组合服务获取数据"""
        # TODO: 实现真实的数据获取逻辑
        # 这里需要调用 portfolio_service 的相关方法
        if hasattr(self.portfolio_service, 'get_account_summary'):
            return await self.portfolio_service.get_account_summary(self.account_id)
        return {}

    def _parse_portfolio_data(self, data: Dict[str, Any], status: MarginStatus) -> MarginStatus:
        """解析组合数据"""
        status.total_equity = float(data.get('total_equity', 0))
        status.available_margin = float(data.get('available_margin', 0))
        status.used_margin = float(data.get('used_margin', 0))
        status.maintenance_margin = float(data.get('maintenance_margin', 0))
        status.unrealized_pnl = float(data.get('unrealized_pnl', 0))
        status.realized_pnl_today = float(data.get('realized_pnl_today', 0))
        status.position_count = int(data.get('position_count', 0))

        # 计算比率
        if status.available_margin > 0:
            status.margin_ratio = status.used_margin / status.available_margin

        if status.maintenance_margin > 0:
            status.margin_safety_ratio = status.total_equity / status.maintenance_margin

        if status.total_equity > 0:
            status.daily_pnl_ratio = status.realized_pnl_today / status.total_equity

        return status

    async def _get_mock_status(self, status: MarginStatus) -> MarginStatus:
        """生成模拟状态 (仅用于测试)"""
        import random

        status.total_equity = 1000000.0
        status.available_margin = 800000.0
        status.used_margin = random.uniform(400000, 700000)  # 模拟波动
        status.maintenance_margin = status.used_margin * 0.8
        status.unrealized_pnl = random.uniform(-50000, 50000)
        status.position_count = random.randint(5, 15)

        status.margin_ratio = status.used_margin / status.available_margin
        status.margin_safety_ratio = status.total_equity / status.maintenance_margin

        return status

    def _generate_alert_message(self, status: MarginStatus) -> str:
        """生成告警消息"""
        ratio_pct = status.margin_ratio * 100

        if status.alert_level == AlertLevel.EMERGENCY:
            return f"紧急! 保证金使用率 {ratio_pct:.1f}% 已超过强平线 {self.thresholds.emergency_ratio*100:.0f}%，正在执行强平"
        elif status.alert_level == AlertLevel.DANGER:
            return f"危险! 保证金使用率 {ratio_pct:.1f}% 接近强平线，请立即减仓"
        elif status.alert_level == AlertLevel.WARNING:
            return f"警告! 保证金使用率 {ratio_pct:.1f}%，建议降低仓位"
        elif status.alert_level == AlertLevel.NOTICE:
            return f"注意: 保证金使用率 {ratio_pct:.1f}%，请关注风险"
        else:
            return f"正常: 保证金使用率 {ratio_pct:.1f}%"

    # ============ 告警处理 ============

    async def _process_alert(self, status: MarginStatus):
        """处理告警"""
        level = status.alert_level

        # 检查是否在冷却期内
        if self._is_in_cooldown(level):
            return

        # 检查是否是级别升级
        is_escalation = self._is_level_escalation(level)

        # 创建告警
        alert = MarginAlert(
            account_id=self.account_id,
            level=level,
            message=status.alert_message,
            margin_status=status,
        )

        # 添加到历史
        self._alert_history.append(alert)
        self._alert_count += 1

        # 更新冷却时间
        self._alert_cooldown[level] = datetime.now()
        self._last_alert_level = level

        # 记录日志
        log_method = {
            AlertLevel.NOTICE: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.DANGER: logger.error,
            AlertLevel.EMERGENCY: logger.critical,
        }.get(level, logger.info)

        log_method(f"Margin alert [{level.value}]: {status.alert_message}")

        # 触发回调
        if self.alert_callback:
            try:
                await self.alert_callback(alert)
                alert.notified = True
            except Exception as e:
                logger.error(f"Failed to send alert callback: {e}")

    def _is_in_cooldown(self, level: AlertLevel) -> bool:
        """检查是否在冷却期内"""
        if level not in self._alert_cooldown:
            return False

        cooldown_seconds = self._alert_cooldown_seconds.get(level, 60)
        if cooldown_seconds == 0:
            return False

        last_alert_time = self._alert_cooldown[level]
        elapsed = (datetime.now() - last_alert_time).total_seconds()

        return elapsed < cooldown_seconds

    def _is_level_escalation(self, current_level: AlertLevel) -> bool:
        """检查是否是级别升级"""
        level_order = [AlertLevel.NORMAL, AlertLevel.NOTICE, AlertLevel.WARNING,
                       AlertLevel.DANGER, AlertLevel.EMERGENCY]

        current_idx = level_order.index(current_level)
        last_idx = level_order.index(self._last_alert_level)

        return current_idx > last_idx

    # ============ 强平处理 ============

    async def _trigger_liquidation(self, status: MarginStatus):
        """触发强平"""
        logger.critical(f"Triggering liquidation for account {self.account_id}")

        # 创建强平订单
        order = LiquidationOrder(
            account_id=self.account_id,
            reason=LiquidationReason.MARGIN_CALL,
            trigger_margin_ratio=status.margin_ratio,
            target_margin_ratio=0.50,  # 目标降到50%
        )

        # 添加审计日志
        order.add_audit_log("created", {
            "trigger_margin_ratio": status.margin_ratio,
            "margin_status": status.to_dict(),
        })

        try:
            # 获取需要平仓的持仓列表
            positions_to_close = await self._calculate_positions_to_close(status, order)
            order.positions_to_close = positions_to_close

            order.add_audit_log("positions_calculated", {
                "positions_count": len(positions_to_close),
                "positions": positions_to_close,
            })

            # 执行强平
            if self.liquidation_callback:
                order.status = "executing"
                result = await self.liquidation_callback(order)
                order.execution_result = result
                order.status = "completed" if result.get('success') else "failed"
                order.executed_at = datetime.now()

                order.add_audit_log("executed", {
                    "status": order.status,
                    "result": result,
                })
            else:
                logger.warning("No liquidation callback configured, logging only")
                order.status = "logged_only"

        except Exception as e:
            logger.error(f"Liquidation failed: {e}", exc_info=True)
            order.status = "failed"
            order.add_audit_log("failed", {"error": str(e)})

        # 添加到历史
        self._liquidation_history.append(order)
        self._liquidation_count += 1

        logger.info(f"Liquidation order {order.order_id} status: {order.status}")

    async def _calculate_positions_to_close(
        self,
        status: MarginStatus,
        order: LiquidationOrder
    ) -> List[Dict[str, Any]]:
        """
        计算需要平仓的持仓列表

        策略: 按流动性排序，优先平仓流动性好的持仓
        目标: 将保证金使用率降到目标水平
        """
        positions_to_close = []

        if not self.portfolio_service:
            return positions_to_close

        try:
            # 获取所有持仓
            if hasattr(self.portfolio_service, 'get_positions'):
                positions = await self.portfolio_service.get_positions(self.account_id)
            else:
                positions = []

            # 按流动性排序 (假设有volume字段)
            sorted_positions = sorted(
                positions,
                key=lambda p: p.get('avg_volume', 0),
                reverse=True  # 流动性好的优先
            )

            # 计算需要释放的保证金
            current_margin = status.used_margin
            target_margin = status.available_margin * order.target_margin_ratio
            margin_to_release = current_margin - target_margin

            if margin_to_release <= 0:
                return positions_to_close

            # 逐个添加持仓直到满足目标
            released_margin = 0
            for pos in sorted_positions:
                if released_margin >= margin_to_release:
                    break

                pos_margin = pos.get('margin_required', 0)
                positions_to_close.append({
                    'symbol': pos.get('symbol'),
                    'quantity': pos.get('quantity'),
                    'side': 'sell' if pos.get('side') == 'long' else 'buy',
                    'margin_to_release': pos_margin,
                    'current_price': pos.get('current_price'),
                    'unrealized_pnl': pos.get('unrealized_pnl'),
                })
                released_margin += pos_margin

        except Exception as e:
            logger.error(f"Failed to calculate positions to close: {e}")

        return positions_to_close

    # ============ 历史记录管理 ============

    def _add_to_history(self, status: MarginStatus):
        """添加到历史记录"""
        self._status_history.append(status)

        # 限制历史记录大小
        if len(self._status_history) > self._max_history_size:
            self._status_history = self._status_history[-self._max_history_size:]

    def get_status_history(self, limit: int = 100) -> List[MarginStatus]:
        """获取状态历史"""
        return self._status_history[-limit:]

    def get_alert_history(self, limit: int = 100) -> List[MarginAlert]:
        """获取告警历史"""
        return self._alert_history[-limit:]

    def get_liquidation_history(self, limit: int = 100) -> List[LiquidationOrder]:
        """获取强平历史"""
        return self._liquidation_history[-limit:]

    # ============ 统计信息 ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        uptime = None
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()

        return {
            'state': self._state.value,
            'account_id': self.account_id,
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'uptime_seconds': uptime,
            'check_count': self._check_count,
            'alert_count': self._alert_count,
            'liquidation_count': self._liquidation_count,
            'last_status': self._last_status.to_dict() if self._last_status else None,
            'thresholds': {
                'notice': self.thresholds.notice_ratio,
                'warning': self.thresholds.warning_ratio,
                'danger': self.thresholds.danger_ratio,
                'emergency': self.thresholds.emergency_ratio,
            },
        }

    # ============ 手动操作 ============

    async def force_check(self) -> MarginStatus:
        """强制检查一次保证金状态"""
        status = await self._get_margin_status()
        self._last_status = status
        self._add_to_history(status)
        return status

    async def manual_liquidation(
        self,
        reason: str = "manual",
        positions: Optional[List[Dict]] = None
    ) -> LiquidationOrder:
        """手动触发强平"""
        logger.warning(f"Manual liquidation triggered for account {self.account_id}")

        status = await self._get_margin_status()

        order = LiquidationOrder(
            account_id=self.account_id,
            reason=LiquidationReason.MANUAL,
            trigger_margin_ratio=status.margin_ratio,
            approved_by="manual",
        )

        if positions:
            order.positions_to_close = positions
        else:
            order.positions_to_close = await self._calculate_positions_to_close(
                status, order
            )

        order.add_audit_log("manual_trigger", {"reason": reason})

        if self.liquidation_callback:
            result = await self.liquidation_callback(order)
            order.execution_result = result
            order.status = "completed" if result.get('success') else "failed"

        self._liquidation_history.append(order)
        return order

    def update_thresholds(self, **kwargs):
        """更新阈值配置"""
        for key, value in kwargs.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
                logger.info(f"Updated threshold {key} = {value}")
