"""
风控告警服务
Risk Alert Service

提供多渠道风险告警通知功能:
- 邮件告警
- WebSocket实时推送
- 短信告警(预留接口)
- 告警历史记录

作者: TraderX Team
创建: 2025-12-23
"""
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque

logger = logging.getLogger(__name__)


class AlertChannel(str, Enum):
    """告警渠道"""
    EMAIL = "email"
    WEBSOCKET = "websocket"
    SMS = "sms"
    WEBHOOK = "webhook"
    CONSOLE = "console"  # 用于调试


class AlertPriority(str, Enum):
    """告警优先级"""
    INFO = "info"           # 信息
    LOW = "low"             # 低
    MEDIUM = "medium"       # 中
    HIGH = "high"           # 高
    CRITICAL = "critical"   # 紧急


class AlertCategory(str, Enum):
    """告警类别"""
    MARGIN = "margin"                   # 保证金告警
    POSITION = "position"               # 持仓告警
    LOSS = "loss"                       # 损失告警
    VOLATILITY = "volatility"           # 波动率告警
    LIQUIDITY = "liquidity"             # 流动性告警
    SYSTEM = "system"                   # 系统告警
    COMPLIANCE = "compliance"           # 合规告警
    CONCENTRATION = "concentration"     # 集中度告警


@dataclass
class AlertConfig:
    """告警配置"""
    # 渠道配置
    enabled_channels: Set[AlertChannel] = field(
        default_factory=lambda: {AlertChannel.WEBSOCKET, AlertChannel.CONSOLE}
    )

    # 告警阈值
    cooldown_seconds: Dict[AlertPriority, int] = field(default_factory=lambda: {
        AlertPriority.INFO: 300,        # 5分钟
        AlertPriority.LOW: 180,         # 3分钟
        AlertPriority.MEDIUM: 60,       # 1分钟
        AlertPriority.HIGH: 30,         # 30秒
        AlertPriority.CRITICAL: 0,      # 无冷却
    })

    # 邮件配置
    email_recipients: List[str] = field(default_factory=list)
    email_from: str = "alerts@traderx.com"

    # 短信配置
    sms_recipients: List[str] = field(default_factory=list)

    # Webhook配置
    webhook_urls: List[str] = field(default_factory=list)

    # 历史记录
    max_history_size: int = 1000
    retain_days: int = 30

    # 聚合配置
    aggregation_window_seconds: int = 60


@dataclass
class Alert:
    """告警实体"""
    id: str
    timestamp: datetime
    priority: AlertPriority
    category: AlertCategory
    title: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    # 源信息
    source: str = "risk_engine"
    user_id: Optional[str] = None
    portfolio_id: Optional[str] = None

    # 状态
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    # 关联数据
    related_symbols: List[str] = field(default_factory=list)
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'category': self.category.value,
            'title': self.title,
            'message': self.message,
            'details': self.details,
            'source': self.source,
            'user_id': self.user_id,
            'portfolio_id': self.portfolio_id,
            'acknowledged': self.acknowledged,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'related_symbols': self.related_symbols,
            'threshold_value': self.threshold_value,
            'actual_value': self.actual_value,
        }


class AlertSender(ABC):
    """告警发送器抽象基类"""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """发送告警"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查发送器是否可用"""
        pass


class ConsoleSender(AlertSender):
    """控制台告警发送器（用于调试）"""

    async def send(self, alert: Alert) -> bool:
        priority_emoji = {
            AlertPriority.INFO: "ℹ️",
            AlertPriority.LOW: "📢",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🚨",
            AlertPriority.CRITICAL: "🔴",
        }

        emoji = priority_emoji.get(alert.priority, "📌")
        print(f"\n{emoji} [{alert.priority.value.upper()}] {alert.title}")
        print(f"   时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   类别: {alert.category.value}")
        print(f"   消息: {alert.message}")
        if alert.actual_value is not None and alert.threshold_value is not None:
            print(f"   当前值: {alert.actual_value:.2f} | 阈值: {alert.threshold_value:.2f}")
        if alert.related_symbols:
            print(f"   相关标的: {', '.join(alert.related_symbols)}")
        print()

        return True

    def is_available(self) -> bool:
        return True


class WebSocketSender(AlertSender):
    """WebSocket告警发送器"""

    def __init__(self):
        self._subscribers: Set[Callable[[Alert], asyncio.Future]] = set()

    def subscribe(self, callback: Callable[[Alert], asyncio.Future]):
        """订阅告警"""
        self._subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[Alert], asyncio.Future]):
        """取消订阅"""
        self._subscribers.discard(callback)

    async def send(self, alert: Alert) -> bool:
        """发送告警到所有订阅者"""
        if not self._subscribers:
            logger.debug("No WebSocket subscribers for alert")
            return True

        tasks = []
        for callback in self._subscribers:
            try:
                tasks.append(callback(alert))
            except Exception as e:
                logger.error(f"WebSocket subscriber error: {e}")

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return True

    def is_available(self) -> bool:
        return True


class EmailSender(AlertSender):
    """邮件告警发送器"""

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self._configured = bool(smtp_host and smtp_port)

    async def send(self, alert: Alert, recipients: List[str] = None) -> bool:
        """
        发送邮件告警

        Args:
            alert: 告警实体
            recipients: 收件人列表（可选，如未提供则跳过发送）

        Returns:
            是否发送成功
        """
        if not self._configured:
            logger.warning("Email sender not configured")
            return False

        if not recipients:
            logger.debug("No email recipients specified, skipping send")
            return True

        try:
            subject = f"[{alert.priority.value.upper()}] {alert.title}"
            body = self._format_email_body(alert)

            # 🔧 2025-12-25: 实现实际邮件发送逻辑
            await self._send_email(recipients, subject, body)
            logger.info(f"Email alert sent successfully: {subject}")

            return True

        except ImportError:
            logger.warning("aiosmtplib not installed, email sending disabled")
            return False
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    async def _send_email(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        from_addr: str = "alerts@traderx.com"
    ) -> None:
        """
        实际发送邮件

        Args:
            recipients: 收件人列表
            subject: 邮件主题
            body: 邮件正文
            from_addr: 发件人地址
        """
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
        except ImportError:
            logger.warning("aiosmtplib not available, falling back to log output")
            logger.info(f"[EMAIL] To: {recipients}, Subject: {subject}")
            logger.info(f"[EMAIL] Body:\n{body}")
            return

        # 构建邮件
        message = MIMEMultipart()
        message["From"] = from_addr
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain", "utf-8"))

        # 发送邮件
        try:
            if self.use_tls:
                await aiosmtplib.send(
                    message,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    username=self.username,
                    password=self.password,
                    start_tls=True,
                )
            else:
                await aiosmtplib.send(
                    message,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    username=self.username,
                    password=self.password,
                )
        except aiosmtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            raise

    def _format_email_body(self, alert: Alert) -> str:
        """格式化邮件正文"""
        lines = [
            f"风险告警通知",
            f"=" * 50,
            f"",
            f"优先级: {alert.priority.value.upper()}",
            f"类别: {alert.category.value}",
            f"时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"标题: {alert.title}",
            f"详情: {alert.message}",
            f"",
        ]

        if alert.actual_value is not None:
            lines.append(f"当前值: {alert.actual_value:.4f}")
        if alert.threshold_value is not None:
            lines.append(f"阈值: {alert.threshold_value:.4f}")
        if alert.related_symbols:
            lines.append(f"相关标的: {', '.join(alert.related_symbols)}")

        lines.extend([
            f"",
            f"=" * 50,
            f"此邮件由TraderX风控系统自动发送",
        ])

        return "\n".join(lines)

    def is_available(self) -> bool:
        return self._configured


class WebhookSender(AlertSender):
    """Webhook告警发送器"""

    def __init__(self, urls: List[str] = None):
        self.urls = urls or []

    async def send(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        if not self.urls:
            return True

        try:
            import aiohttp

            payload = alert.to_dict()

            async with aiohttp.ClientSession() as session:
                tasks = []
                for url in self.urls:
                    tasks.append(
                        session.post(
                            url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=10)
                        )
                    )

                results = await asyncio.gather(*tasks, return_exceptions=True)

                success = all(
                    not isinstance(r, Exception) and r.status < 400
                    for r in results
                )
                return success

        except ImportError:
            logger.warning("aiohttp not installed, webhook alerts disabled")
            return False
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
            return False

    def is_available(self) -> bool:
        return bool(self.urls)


class AlertService:
    """
    风控告警服务

    管理告警的生成、发送、聚合和历史记录。
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        初始化告警服务

        Args:
            config: 告警配置
        """
        self.config = config or AlertConfig()

        # 告警发送器
        self._senders: Dict[AlertChannel, AlertSender] = {}
        self._setup_senders()

        # 告警历史
        self._history: deque = deque(maxlen=self.config.max_history_size)

        # 冷却追踪 {(category, user_id): last_alert_time}
        self._cooldowns: Dict[tuple, datetime] = {}

        # 聚合缓冲
        self._aggregation_buffer: Dict[str, List[Alert]] = {}

        # 统计
        self._stats = {
            'total_sent': 0,
            'by_priority': {p.value: 0 for p in AlertPriority},
            'by_category': {c.value: 0 for c in AlertCategory},
            'by_channel': {c.value: 0 for c in AlertChannel},
            'suppressed': 0,
        }

        # 运行状态
        self._running = False
        self._aggregation_task: Optional[asyncio.Task] = None

        logger.info("AlertService initialized")

    def _setup_senders(self):
        """设置告警发送器"""
        self._senders[AlertChannel.CONSOLE] = ConsoleSender()
        self._senders[AlertChannel.WEBSOCKET] = WebSocketSender()
        self._senders[AlertChannel.EMAIL] = EmailSender()
        self._senders[AlertChannel.WEBHOOK] = WebhookSender(
            self.config.webhook_urls
        )

    async def start(self):
        """启动告警服务"""
        if self._running:
            return

        self._running = True

        # 启动聚合任务
        self._aggregation_task = asyncio.create_task(
            self._aggregation_loop()
        )

        logger.info("AlertService started")

    async def stop(self):
        """停止告警服务"""
        self._running = False

        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass

        # 发送缓冲区中的告警
        await self._flush_aggregation_buffer()

        logger.info("AlertService stopped")

    async def send_alert(
        self,
        priority: AlertPriority,
        category: AlertCategory,
        title: str,
        message: str,
        user_id: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        related_symbols: Optional[List[str]] = None,
        threshold_value: Optional[float] = None,
        actual_value: Optional[float] = None,
        channels: Optional[Set[AlertChannel]] = None,
        bypass_cooldown: bool = False
    ) -> Optional[Alert]:
        """
        发送告警

        Args:
            priority: 优先级
            category: 类别
            title: 标题
            message: 消息内容
            user_id: 用户ID
            portfolio_id: 组合ID
            details: 详细信息
            related_symbols: 相关标的
            threshold_value: 阈值
            actual_value: 实际值
            channels: 发送渠道（默认使用配置）
            bypass_cooldown: 是否跳过冷却检查

        Returns:
            发送的Alert对象，如果被抑制则返回None
        """
        # 检查冷却
        if not bypass_cooldown and not self._check_cooldown(
            priority, category, user_id
        ):
            self._stats['suppressed'] += 1
            logger.debug(f"Alert suppressed by cooldown: {title}")
            return None

        # 创建告警
        alert = Alert(
            id=self._generate_alert_id(),
            timestamp=datetime.now(),
            priority=priority,
            category=category,
            title=title,
            message=message,
            details=details or {},
            user_id=user_id,
            portfolio_id=portfolio_id,
            related_symbols=related_symbols or [],
            threshold_value=threshold_value,
            actual_value=actual_value,
        )

        # 添加到历史
        self._history.append(alert)

        # 更新冷却
        self._update_cooldown(priority, category, user_id)

        # 确定发送渠道
        send_channels = channels or self.config.enabled_channels

        # 发送到各渠道
        for channel in send_channels:
            if channel in self._senders:
                sender = self._senders[channel]
                if sender.is_available():
                    try:
                        success = await sender.send(alert)
                        if success:
                            self._stats['by_channel'][channel.value] += 1
                    except Exception as e:
                        logger.error(f"Failed to send alert via {channel}: {e}")

        # 更新统计
        self._stats['total_sent'] += 1
        self._stats['by_priority'][priority.value] += 1
        self._stats['by_category'][category.value] += 1

        return alert

    def _check_cooldown(
        self,
        priority: AlertPriority,
        category: AlertCategory,
        user_id: Optional[str]
    ) -> bool:
        """检查是否在冷却期内"""
        key = (category, user_id)
        last_time = self._cooldowns.get(key)

        if last_time is None:
            return True

        cooldown_seconds = self.config.cooldown_seconds.get(priority, 60)
        if cooldown_seconds == 0:
            return True

        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed >= cooldown_seconds

    def _update_cooldown(
        self,
        priority: AlertPriority,
        category: AlertCategory,
        user_id: Optional[str]
    ):
        """更新冷却时间"""
        key = (category, user_id)
        self._cooldowns[key] = datetime.now()

    def _generate_alert_id(self) -> str:
        """生成告警ID"""
        import uuid
        return f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

    async def _aggregation_loop(self):
        """告警聚合循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.aggregation_window_seconds)
                await self._flush_aggregation_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation loop error: {e}")

    async def _flush_aggregation_buffer(self):
        """刷新聚合缓冲区"""
        # 目前简单实现，后续可扩展为真正的聚合逻辑
        pass

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """
        确认告警

        Args:
            alert_id: 告警ID
            acknowledged_by: 确认人

        Returns:
            是否成功
        """
        for alert in self._history:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def get_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        priority: Optional[AlertPriority] = None,
        category: Optional[AlertCategory] = None,
        user_id: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        获取告警历史

        Args:
            start_time: 开始时间
            end_time: 结束时间
            priority: 优先级过滤
            category: 类别过滤
            user_id: 用户ID过滤
            acknowledged: 确认状态过滤
            limit: 返回数量限制

        Returns:
            告警列表
        """
        result = []

        for alert in reversed(self._history):
            # 时间过滤
            if start_time and alert.timestamp < start_time:
                continue
            if end_time and alert.timestamp > end_time:
                continue

            # 属性过滤
            if priority and alert.priority != priority:
                continue
            if category and alert.category != category:
                continue
            if user_id and alert.user_id != user_id:
                continue
            if acknowledged is not None and alert.acknowledged != acknowledged:
                continue

            result.append(alert)

            if len(result) >= limit:
                break

        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        return {
            **self._stats,
            'history_size': len(self._history),
            'cooldowns_active': len(self._cooldowns),
            'running': self._running,
        }

    def subscribe_websocket(
        self,
        callback: Callable[[Alert], asyncio.Future]
    ):
        """订阅WebSocket告警"""
        ws_sender = self._senders.get(AlertChannel.WEBSOCKET)
        if isinstance(ws_sender, WebSocketSender):
            ws_sender.subscribe(callback)

    def unsubscribe_websocket(
        self,
        callback: Callable[[Alert], asyncio.Future]
    ):
        """取消订阅WebSocket告警"""
        ws_sender = self._senders.get(AlertChannel.WEBSOCKET)
        if isinstance(ws_sender, WebSocketSender):
            ws_sender.unsubscribe(callback)

    # 便捷方法
    async def margin_warning(
        self,
        user_id: str,
        current_ratio: float,
        threshold: float,
        portfolio_id: Optional[str] = None
    ) -> Optional[Alert]:
        """发送保证金预警"""
        return await self.send_alert(
            priority=AlertPriority.HIGH,
            category=AlertCategory.MARGIN,
            title="保证金使用率预警",
            message=f"保证金使用率已达 {current_ratio*100:.1f}%，超过预警阈值 {threshold*100:.1f}%",
            user_id=user_id,
            portfolio_id=portfolio_id,
            threshold_value=threshold,
            actual_value=current_ratio,
        )

    async def margin_critical(
        self,
        user_id: str,
        current_ratio: float,
        portfolio_id: Optional[str] = None
    ) -> Optional[Alert]:
        """发送保证金紧急告警"""
        return await self.send_alert(
            priority=AlertPriority.CRITICAL,
            category=AlertCategory.MARGIN,
            title="保证金紧急告警 - 可能触发强平",
            message=f"保证金使用率已达 {current_ratio*100:.1f}%，即将触发强制平仓！",
            user_id=user_id,
            portfolio_id=portfolio_id,
            actual_value=current_ratio,
            bypass_cooldown=True,  # 紧急告警不受冷却限制
        )

    async def loss_alert(
        self,
        user_id: str,
        loss_amount: float,
        loss_pct: float,
        period: str = "daily",
        portfolio_id: Optional[str] = None
    ) -> Optional[Alert]:
        """发送损失告警"""
        return await self.send_alert(
            priority=AlertPriority.HIGH,
            category=AlertCategory.LOSS,
            title=f"{period}损失告警",
            message=f"{period}损失达 ${abs(loss_amount):,.2f} ({abs(loss_pct)*100:.1f}%)",
            user_id=user_id,
            portfolio_id=portfolio_id,
            actual_value=loss_pct,
            details={'loss_amount': loss_amount, 'period': period},
        )

    async def volatility_alert(
        self,
        symbol: str,
        current_vol: float,
        threshold: float,
        user_id: Optional[str] = None
    ) -> Optional[Alert]:
        """发送波动率告警"""
        return await self.send_alert(
            priority=AlertPriority.MEDIUM,
            category=AlertCategory.VOLATILITY,
            title=f"{symbol} 波动率异常",
            message=f"{symbol} 当前波动率 {current_vol*100:.1f}% 超过阈值 {threshold*100:.1f}%",
            user_id=user_id,
            related_symbols=[symbol],
            threshold_value=threshold,
            actual_value=current_vol,
        )

    async def concentration_alert(
        self,
        symbol: str,
        concentration_pct: float,
        limit_pct: float,
        user_id: Optional[str] = None,
        portfolio_id: Optional[str] = None
    ) -> Optional[Alert]:
        """发送集中度告警"""
        return await self.send_alert(
            priority=AlertPriority.MEDIUM,
            category=AlertCategory.CONCENTRATION,
            title="持仓集中度过高",
            message=f"{symbol} 持仓占比 {concentration_pct*100:.1f}% 超过限制 {limit_pct*100:.1f}%",
            user_id=user_id,
            portfolio_id=portfolio_id,
            related_symbols=[symbol],
            threshold_value=limit_pct,
            actual_value=concentration_pct,
        )

    async def liquidation_notice(
        self,
        user_id: str,
        symbols: List[str],
        liquidation_value: float,
        reason: str,
        portfolio_id: Optional[str] = None
    ) -> Optional[Alert]:
        """发送强平通知"""
        return await self.send_alert(
            priority=AlertPriority.CRITICAL,
            category=AlertCategory.MARGIN,
            title="强制平仓执行通知",
            message=f"已执行强制平仓，涉及标的: {', '.join(symbols)}，平仓金额: ${liquidation_value:,.2f}，原因: {reason}",
            user_id=user_id,
            portfolio_id=portfolio_id,
            related_symbols=symbols,
            details={'liquidation_value': liquidation_value, 'reason': reason},
            bypass_cooldown=True,
        )


# 全局单例
_alert_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """获取告警服务单例"""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service


async def send_risk_alert(
    priority: AlertPriority,
    category: AlertCategory,
    title: str,
    message: str,
    **kwargs
) -> Optional[Alert]:
    """便捷函数：发送风险告警"""
    service = get_alert_service()
    return await service.send_alert(
        priority=priority,
        category=category,
        title=title,
        message=message,
        **kwargs
    )


if __name__ == "__main__":
    # 测试示例
    async def test_alerts():
        service = AlertService()
        await service.start()

        # 测试各种告警
        await service.margin_warning(
            user_id="user123",
            current_ratio=0.75,
            threshold=0.70
        )

        await asyncio.sleep(1)

        await service.loss_alert(
            user_id="user123",
            loss_amount=5000,
            loss_pct=0.05
        )

        await asyncio.sleep(1)

        await service.margin_critical(
            user_id="user123",
            current_ratio=0.96
        )

        await asyncio.sleep(1)

        # 获取统计
        stats = service.get_stats()
        print(f"\n告警统计: {json.dumps(stats, indent=2)}")

        # 获取历史
        history = service.get_history(limit=10)
        print(f"\n告警历史: {len(history)} 条记录")
        for alert in history:
            print(f"  - [{alert.priority.value}] {alert.title}")

        await service.stop()

    asyncio.run(test_alerts())
