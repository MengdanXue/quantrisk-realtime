"""
交易事件流系统 (Trading Event Stream)

提供实时交易事件的记录、分发、查询和回放功能。
支持事件溯源模式，用于审计追踪和合规监控。

主要功能:
- 事件生成与标准化
- 实时事件分发
- 持久化存储
- 事件查询与过滤
- 事件回放
- 审计追踪
- 合规检查点

事件类型:
- OrderEvent: 订单相关事件
- TradeEvent: 成交事件
- PositionEvent: 持仓变动事件
- RiskEvent: 风控事件
- SystemEvent: 系统事件
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Optional, Any, Callable, Set,
    TypeVar, Generic, Awaitable, Union
)
from abc import ABC, abstractmethod
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class EventType(str, Enum):
    """事件类型"""
    # 订单事件
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_MODIFIED = "order_modified"
    ORDER_EXPIRED = "order_expired"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILLED = "order_partial_filled"

    # 成交事件
    TRADE_EXECUTED = "trade_executed"
    TRADE_CORRECTED = "trade_corrected"
    TRADE_CANCELLED = "trade_cancelled"

    # 持仓事件
    POSITION_OPENED = "position_opened"
    POSITION_INCREASED = "position_increased"
    POSITION_DECREASED = "position_decreased"
    POSITION_CLOSED = "position_closed"
    POSITION_LIQUIDATED = "position_liquidated"

    # 风控事件
    RISK_ALERT = "risk_alert"
    RISK_BREACH = "risk_breach"
    MARGIN_CALL = "margin_call"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"

    # 系统事件
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"
    HEARTBEAT = "heartbeat"

    # 合规事件
    COMPLIANCE_CHECK = "compliance_check"
    COMPLIANCE_VIOLATION = "compliance_violation"


class EventPriority(str, Enum):
    """事件优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(str, Enum):
    """事件状态"""
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    REPLAYED = "replayed"


class StreamState(str, Enum):
    """流状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


# ==================== 数据类定义 ====================

@dataclass
class TradingEvent:
    """交易事件"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING

    # 事件数据
    data: Dict[str, Any] = field(default_factory=dict)

    # 关联信息
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 处理信息
    processed_at: Optional[datetime] = None
    processor: Optional[str] = None
    error_message: Optional[str] = None

    # 审计信息
    user_id: Optional[str] = None
    account_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "priority": self.priority.value,
            "status": self.status.value,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "metadata": self.metadata,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "processor": self.processor,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "account_id": self.account_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradingEvent":
        """从字典创建"""
        return cls(
            event_id=d["event_id"],
            event_type=EventType(d["event_type"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            source=d["source"],
            priority=EventPriority(d.get("priority", "normal")),
            status=EventStatus(d.get("status", "pending")),
            data=d.get("data", {}),
            correlation_id=d.get("correlation_id"),
            causation_id=d.get("causation_id"),
            metadata=d.get("metadata", {}),
            processed_at=datetime.fromisoformat(d["processed_at"]) if d.get("processed_at") else None,
            processor=d.get("processor"),
            error_message=d.get("error_message"),
            user_id=d.get("user_id"),
            account_id=d.get("account_id"),
            session_id=d.get("session_id"),
        )

    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "TradingEvent":
        """从JSON创建"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class EventFilter:
    """事件过滤器"""
    event_types: Optional[Set[EventType]] = None
    sources: Optional[Set[str]] = None
    priorities: Optional[Set[EventPriority]] = None
    statuses: Optional[Set[EventStatus]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    account_id: Optional[str] = None
    symbols: Optional[Set[str]] = None

    def matches(self, event: TradingEvent) -> bool:
        """检查事件是否匹配过滤器"""
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.sources and event.source not in self.sources:
            return False
        if self.priorities and event.priority not in self.priorities:
            return False
        if self.statuses and event.status not in self.statuses:
            return False
        if self.start_time and event.timestamp < self.start_time:
            return False
        if self.end_time and event.timestamp > self.end_time:
            return False
        if self.correlation_id and event.correlation_id != self.correlation_id:
            return False
        if self.user_id and event.user_id != self.user_id:
            return False
        if self.account_id and event.account_id != self.account_id:
            return False
        if self.symbols:
            event_symbol = event.data.get("symbol")
            if event_symbol and event_symbol not in self.symbols:
                return False
        return True


@dataclass
class Subscription:
    """事件订阅"""
    subscription_id: str
    filter: EventFilter
    callback: Callable[[TradingEvent], Union[None, Awaitable[None]]]
    is_async: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    event_count: int = 0
    last_event_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "subscription_id": self.subscription_id,
            "is_async": self.is_async,
            "created_at": self.created_at.isoformat(),
            "event_count": self.event_count,
            "last_event_at": self.last_event_at.isoformat() if self.last_event_at else None,
        }


@dataclass
class StreamConfig:
    """流配置"""
    max_buffer_size: int = 10000
    max_history_size: int = 100000
    flush_interval: float = 1.0
    batch_size: int = 100
    enable_persistence: bool = True
    persistence_path: Optional[str] = None
    enable_compression: bool = False
    heartbeat_interval: float = 30.0
    retry_failed_events: bool = True
    max_retries: int = 3


@dataclass
class ReplayConfig:
    """回放配置"""
    speed_multiplier: float = 1.0
    preserve_timing: bool = True
    filter: Optional[EventFilter] = None
    start_from: Optional[str] = None  # event_id
    stop_at: Optional[str] = None  # event_id
    max_events: Optional[int] = None


@dataclass
class AuditRecord:
    """审计记录"""
    audit_id: str
    event_id: str
    action: str
    timestamp: datetime
    user_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "audit_id": self.audit_id,
            "event_id": self.event_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "details": self.details,
        }


@dataclass
class ComplianceCheckpoint:
    """合规检查点"""
    checkpoint_id: str
    timestamp: datetime
    event_count: int
    last_event_id: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp.isoformat(),
            "event_count": self.event_count,
            "last_event_id": self.last_event_id,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }


# ==================== 事件存储接口 ====================

class EventStore(ABC):
    """事件存储抽象基类"""

    @abstractmethod
    async def append(self, event: TradingEvent) -> bool:
        """追加事件"""
        pass

    @abstractmethod
    async def get(self, event_id: str) -> Optional[TradingEvent]:
        """获取事件"""
        pass

    @abstractmethod
    async def query(self, filter: EventFilter, limit: int = 100, offset: int = 0) -> List[TradingEvent]:
        """查询事件"""
        pass

    @abstractmethod
    async def count(self, filter: Optional[EventFilter] = None) -> int:
        """计数"""
        pass

    @abstractmethod
    async def get_latest(self, n: int = 1) -> List[TradingEvent]:
        """获取最新事件"""
        pass


class InMemoryEventStore(EventStore):
    """内存事件存储"""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.events: deque = deque(maxlen=max_size)
        self.index: Dict[str, TradingEvent] = {}
        self._lock = threading.Lock()

    async def append(self, event: TradingEvent) -> bool:
        """追加事件"""
        with self._lock:
            # 如果超过最大容量,移除最旧的
            if len(self.events) >= self.max_size:
                old_event = self.events[0]
                self.index.pop(old_event.event_id, None)

            self.events.append(event)
            self.index[event.event_id] = event
        return True

    async def get(self, event_id: str) -> Optional[TradingEvent]:
        """获取事件"""
        return self.index.get(event_id)

    async def query(self, filter: EventFilter, limit: int = 100, offset: int = 0) -> List[TradingEvent]:
        """查询事件"""
        with self._lock:
            matched = [e for e in self.events if filter.matches(e)]
            return matched[offset:offset + limit]

    async def count(self, filter: Optional[EventFilter] = None) -> int:
        """计数"""
        with self._lock:
            if filter is None:
                return len(self.events)
            return sum(1 for e in self.events if filter.matches(e))

    async def get_latest(self, n: int = 1) -> List[TradingEvent]:
        """获取最新事件"""
        with self._lock:
            return list(self.events)[-n:]

    def clear(self):
        """清空存储"""
        with self._lock:
            self.events.clear()
            self.index.clear()


# ==================== 事件处理器 ====================

class EventProcessor(ABC):
    """事件处理器抽象基类"""

    @abstractmethod
    async def process(self, event: TradingEvent) -> bool:
        """处理事件"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """处理器名称"""
        pass


class LoggingProcessor(EventProcessor):
    """日志处理器"""

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self._logger = logging.getLogger(__name__ + ".events")

    async def process(self, event: TradingEvent) -> bool:
        """处理事件"""
        self._logger.log(
            self.log_level,
            f"Event: {event.event_type.value} | "
            f"ID: {event.event_id[:8]} | "
            f"Source: {event.source} | "
            f"Data: {event.data}"
        )
        return True

    @property
    def name(self) -> str:
        return "logging_processor"


class AuditProcessor(EventProcessor):
    """审计处理器"""

    def __init__(self):
        self.audit_records: List[AuditRecord] = []
        self._lock = threading.Lock()

    async def process(self, event: TradingEvent) -> bool:
        """处理事件"""
        record = AuditRecord(
            audit_id=str(uuid.uuid4()),
            event_id=event.event_id,
            action="processed",
            timestamp=datetime.now(),
            user_id=event.user_id,
            details={
                "event_type": event.event_type.value,
                "source": event.source,
            }
        )
        with self._lock:
            self.audit_records.append(record)
        return True

    @property
    def name(self) -> str:
        return "audit_processor"

    def get_audit_trail(self, event_id: Optional[str] = None) -> List[AuditRecord]:
        """获取审计追踪"""
        with self._lock:
            if event_id:
                return [r for r in self.audit_records if r.event_id == event_id]
            return list(self.audit_records)


class ComplianceProcessor(EventProcessor):
    """合规处理器"""

    def __init__(self):
        self.violations: List[Dict[str, Any]] = []
        self.rules: List[Callable[[TradingEvent], Optional[str]]] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: Callable[[TradingEvent], Optional[str]]):
        """添加合规规则"""
        self.rules.append(rule)

    async def process(self, event: TradingEvent) -> bool:
        """处理事件"""
        for rule in self.rules:
            violation = rule(event)
            if violation:
                with self._lock:
                    self.violations.append({
                        "event_id": event.event_id,
                        "timestamp": datetime.now().isoformat(),
                        "violation": violation,
                    })
        return True

    @property
    def name(self) -> str:
        return "compliance_processor"

    def get_violations(self) -> List[Dict[str, Any]]:
        """获取违规记录"""
        with self._lock:
            return list(self.violations)


# ==================== 事件流核心 ====================

class TradingEventStream:
    """交易事件流"""

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.state = StreamState.STOPPED

        # 存储
        self.store = InMemoryEventStore(self.config.max_history_size)

        # 订阅
        self.subscriptions: Dict[str, Subscription] = {}

        # 处理器
        self.processors: List[EventProcessor] = []

        # 缓冲区
        self.buffer: deque = deque(maxlen=self.config.max_buffer_size)

        # 统计
        self.stats = {
            "events_received": 0,
            "events_processed": 0,
            "events_failed": 0,
            "events_replayed": 0,
            "subscriptions_notified": 0,
        }

        # 检查点
        self.checkpoints: List[ComplianceCheckpoint] = []

        # 异步任务
        self._process_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # 回调
        self._on_event_callbacks: List[Callable] = []

    async def start(self):
        """启动事件流"""
        if self.state == StreamState.RUNNING:
            return

        self.state = StreamState.STARTING

        # 启动处理任务
        self._process_task = asyncio.create_task(self._process_loop())

        # 启动心跳
        if self.config.heartbeat_interval > 0:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self.state = StreamState.RUNNING

        # 发送系统启动事件
        await self.emit(
            EventType.SYSTEM_START,
            source="event_stream",
            data={"config": asdict(self.config)},
            priority=EventPriority.HIGH,
        )

    async def stop(self):
        """停止事件流"""
        if self.state == StreamState.STOPPED:
            return

        self.state = StreamState.STOPPING

        # 发送系统停止事件
        await self.emit(
            EventType.SYSTEM_STOP,
            source="event_stream",
            data={"stats": self.stats},
            priority=EventPriority.HIGH,
        )

        # 等待处理完成
        await asyncio.sleep(0.1)

        # 取消任务
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        self.state = StreamState.STOPPED

    async def emit(
        self,
        event_type: EventType,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        account_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TradingEvent:
        """发送事件"""
        event = TradingEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            priority=priority,
            data=data or {},
            correlation_id=correlation_id,
            causation_id=causation_id,
            user_id=user_id,
            account_id=account_id,
            metadata=metadata or {},
        )

        # 添加到缓冲区
        with self._lock:
            self.buffer.append(event)
            self.stats["events_received"] += 1

        # 立即通知高优先级事件
        if priority == EventPriority.CRITICAL:
            await self._process_event(event)

        return event

    def emit_sync(
        self,
        event_type: EventType,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        account_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TradingEvent:
        """同步发送事件"""
        event = TradingEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            priority=priority,
            data=data or {},
            correlation_id=correlation_id,
            causation_id=causation_id,
            user_id=user_id,
            account_id=account_id,
            metadata=metadata or {},
        )

        with self._lock:
            self.buffer.append(event)
            self.stats["events_received"] += 1

        return event

    async def _process_loop(self):
        """处理循环"""
        while self.state in (StreamState.STARTING, StreamState.RUNNING):
            try:
                # 批量处理
                events_to_process = []
                with self._lock:
                    while self.buffer and len(events_to_process) < self.config.batch_size:
                        events_to_process.append(self.buffer.popleft())

                for event in events_to_process:
                    await self._process_event(event)

                await asyncio.sleep(self.config.flush_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process loop error: {e}")

    async def _process_event(self, event: TradingEvent):
        """处理单个事件"""
        try:
            # 存储事件
            await self.store.append(event)

            # 运行处理器
            for processor in self.processors:
                try:
                    await processor.process(event)
                except Exception as e:
                    logger.error(f"Processor {processor.name} error: {e}")

            # 通知订阅者
            await self._notify_subscribers(event)

            # 触发回调
            for callback in self._on_event_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            # 更新状态
            event.status = EventStatus.PROCESSED
            event.processed_at = datetime.now()

            with self._lock:
                self.stats["events_processed"] += 1

        except Exception as e:
            event.status = EventStatus.FAILED
            event.error_message = str(e)
            with self._lock:
                self.stats["events_failed"] += 1
            logger.error(f"Event processing failed: {e}")

    async def _notify_subscribers(self, event: TradingEvent):
        """通知订阅者"""
        for sub_id, subscription in list(self.subscriptions.items()):
            if subscription.filter.matches(event):
                try:
                    if subscription.is_async:
                        await subscription.callback(event)
                    else:
                        subscription.callback(event)

                    subscription.event_count += 1
                    subscription.last_event_at = datetime.now()

                    with self._lock:
                        self.stats["subscriptions_notified"] += 1
                except Exception as e:
                    logger.error(f"Subscription {sub_id} callback error: {e}")

    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.state in (StreamState.STARTING, StreamState.RUNNING):
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                await self.emit(
                    EventType.HEARTBEAT,
                    source="event_stream",
                    data={
                        "state": self.state.value,
                        "stats": self.stats.copy(),
                        "buffer_size": len(self.buffer),
                    },
                    priority=EventPriority.LOW,
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    def subscribe(
        self,
        callback: Callable[[TradingEvent], Union[None, Awaitable[None]]],
        filter: Optional[EventFilter] = None,
        is_async: bool = False,
    ) -> str:
        """订阅事件"""
        subscription_id = str(uuid.uuid4())
        subscription = Subscription(
            subscription_id=subscription_id,
            filter=filter or EventFilter(),
            callback=callback,
            is_async=is_async,
        )
        self.subscriptions[subscription_id] = subscription
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        return self.subscriptions.pop(subscription_id, None) is not None

    def add_processor(self, processor: EventProcessor):
        """添加处理器"""
        self.processors.append(processor)

    def remove_processor(self, name: str) -> bool:
        """移除处理器"""
        for i, processor in enumerate(self.processors):
            if processor.name == name:
                self.processors.pop(i)
                return True
        return False

    def on_event(self, callback: Callable):
        """注册事件回调"""
        self._on_event_callbacks.append(callback)

    async def get_event(self, event_id: str) -> Optional[TradingEvent]:
        """获取事件"""
        return await self.store.get(event_id)

    async def query_events(
        self,
        filter: Optional[EventFilter] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TradingEvent]:
        """查询事件"""
        return await self.store.query(filter or EventFilter(), limit, offset)

    async def get_latest_events(self, n: int = 10) -> List[TradingEvent]:
        """获取最新事件"""
        return await self.store.get_latest(n)

    async def count_events(self, filter: Optional[EventFilter] = None) -> int:
        """统计事件数量"""
        return await self.store.count(filter)

    async def replay(
        self,
        config: Optional[ReplayConfig] = None,
        callback: Optional[Callable[[TradingEvent], Union[None, Awaitable[None]]]] = None,
    ) -> int:
        """回放事件"""
        config = config or ReplayConfig()

        # 获取要回放的事件
        filter = config.filter or EventFilter()
        events = await self.store.query(filter, limit=config.max_events or 10000)

        # 找到起始位置
        if config.start_from:
            start_idx = next(
                (i for i, e in enumerate(events) if e.event_id == config.start_from),
                0
            )
            events = events[start_idx:]

        # 找到结束位置
        if config.stop_at:
            stop_idx = next(
                (i for i, e in enumerate(events) if e.event_id == config.stop_at),
                len(events)
            )
            events = events[:stop_idx + 1]

        # 限制数量
        if config.max_events:
            events = events[:config.max_events]

        replayed_count = 0
        prev_time = None

        for event in events:
            # 保持时间间隔
            if config.preserve_timing and prev_time:
                delay = (event.timestamp - prev_time).total_seconds()
                delay = delay / config.speed_multiplier
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))  # 最大延迟1秒

            prev_time = event.timestamp

            # 标记为回放
            event.status = EventStatus.REPLAYED

            # 调用回调
            if callback:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Replay callback error: {e}")

            replayed_count += 1

        with self._lock:
            self.stats["events_replayed"] += replayed_count

        return replayed_count

    def create_checkpoint(self) -> ComplianceCheckpoint:
        """创建合规检查点"""
        import hashlib

        # 计算校验和
        events = list(self.store.events)
        checksum_data = "".join(e.event_id for e in events[-100:])
        checksum = hashlib.sha256(checksum_data.encode()).hexdigest()[:16]

        checkpoint = ComplianceCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_count=len(events),
            last_event_id=events[-1].event_id if events else "",
            checksum=checksum,
            metadata={
                "stats": self.stats.copy(),
            }
        )

        self.checkpoints.append(checkpoint)
        return checkpoint

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "state": self.state.value,
                "stats": self.stats.copy(),
                "buffer_size": len(self.buffer),
                "store_size": len(self.store.events),
                "subscriptions": len(self.subscriptions),
                "processors": len(self.processors),
                "checkpoints": len(self.checkpoints),
            }


# ==================== 便捷事件创建函数 ====================

def create_order_event(
    event_type: EventType,
    order_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """创建订单事件数据"""
    return {
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        **kwargs
    }


def create_trade_event(
    trade_id: str,
    order_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    **kwargs
) -> Dict[str, Any]:
    """创建成交事件数据"""
    return {
        "trade_id": trade_id,
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        **kwargs
    }


def create_position_event(
    symbol: str,
    side: str,
    quantity: float,
    avg_price: float,
    market_value: float,
    unrealized_pnl: float,
    **kwargs
) -> Dict[str, Any]:
    """创建持仓事件数据"""
    return {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "avg_price": avg_price,
        "market_value": market_value,
        "unrealized_pnl": unrealized_pnl,
        **kwargs
    }


def create_risk_event(
    risk_type: str,
    level: str,
    message: str,
    current_value: float,
    threshold: float,
    **kwargs
) -> Dict[str, Any]:
    """创建风险事件数据"""
    return {
        "risk_type": risk_type,
        "level": level,
        "message": message,
        "current_value": current_value,
        "threshold": threshold,
        **kwargs
    }


# ==================== 工厂函数 ====================

def create_event_stream(
    max_buffer_size: int = 10000,
    max_history_size: int = 100000,
    enable_logging: bool = True,
    enable_audit: bool = True,
) -> TradingEventStream:
    """创建事件流"""
    config = StreamConfig(
        max_buffer_size=max_buffer_size,
        max_history_size=max_history_size,
    )

    stream = TradingEventStream(config)

    if enable_logging:
        stream.add_processor(LoggingProcessor())

    if enable_audit:
        stream.add_processor(AuditProcessor())

    return stream


def create_event_filter(
    event_types: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    priorities: Optional[List[str]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    symbols: Optional[List[str]] = None,
) -> EventFilter:
    """创建事件过滤器"""
    return EventFilter(
        event_types=set(EventType(t) for t in event_types) if event_types else None,
        sources=set(sources) if sources else None,
        priorities=set(EventPriority(p) for p in priorities) if priorities else None,
        start_time=datetime.fromisoformat(start_time) if start_time else None,
        end_time=datetime.fromisoformat(end_time) if end_time else None,
        symbols=set(symbols) if symbols else None,
    )


async def quick_emit(
    stream: TradingEventStream,
    event_type: str,
    source: str,
    data: Dict[str, Any],
) -> TradingEvent:
    """快速发送事件"""
    return await stream.emit(
        event_type=EventType(event_type),
        source=source,
        data=data,
    )


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'EventType',
    'EventPriority',
    'EventStatus',
    'StreamState',
    # 数据类
    'TradingEvent',
    'EventFilter',
    'Subscription',
    'StreamConfig',
    'ReplayConfig',
    'AuditRecord',
    'ComplianceCheckpoint',
    # 存储
    'EventStore',
    'InMemoryEventStore',
    # 处理器
    'EventProcessor',
    'LoggingProcessor',
    'AuditProcessor',
    'ComplianceProcessor',
    # 核心
    'TradingEventStream',
    # 便捷函数
    'create_order_event',
    'create_trade_event',
    'create_position_event',
    'create_risk_event',
    # 工厂函数
    'create_event_stream',
    'create_event_filter',
    'quick_emit',
]
