"""
交易事件流系统测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from quantrisk_realtime.realtime.trading_event_stream import (
    # 枚举
    EventType,
    EventPriority,
    EventStatus,
    StreamState,
    # 数据类
    TradingEvent,
    EventFilter,
    Subscription,
    StreamConfig,
    ReplayConfig,
    AuditRecord,
    ComplianceCheckpoint,
    # 存储
    InMemoryEventStore,
    # 处理器
    LoggingProcessor,
    AuditProcessor,
    ComplianceProcessor,
    # 核心
    TradingEventStream,
    # 便捷函数
    create_order_event,
    create_trade_event,
    create_position_event,
    create_risk_event,
    # 工厂函数
    create_event_stream,
    create_event_filter,
    quick_emit,
)


# ==================== 枚举测试 ====================

class TestEventType:
    """事件类型测试"""

    def test_order_events(self):
        """测试订单事件类型"""
        assert EventType.ORDER_CREATED.value == "order_created"
        assert EventType.ORDER_SUBMITTED.value == "order_submitted"
        assert EventType.ORDER_FILLED.value == "order_filled"

    def test_trade_events(self):
        """测试成交事件类型"""
        assert EventType.TRADE_EXECUTED.value == "trade_executed"
        assert EventType.TRADE_CANCELLED.value == "trade_cancelled"

    def test_position_events(self):
        """测试持仓事件类型"""
        assert EventType.POSITION_OPENED.value == "position_opened"
        assert EventType.POSITION_CLOSED.value == "position_closed"

    def test_risk_events(self):
        """测试风控事件类型"""
        assert EventType.RISK_ALERT.value == "risk_alert"
        assert EventType.MARGIN_CALL.value == "margin_call"

    def test_system_events(self):
        """测试系统事件类型"""
        assert EventType.SYSTEM_START.value == "system_start"
        assert EventType.HEARTBEAT.value == "heartbeat"


class TestEventPriority:
    """事件优先级测试"""

    def test_priority_values(self):
        """测试优先级值"""
        assert EventPriority.LOW.value == "low"
        assert EventPriority.NORMAL.value == "normal"
        assert EventPriority.HIGH.value == "high"
        assert EventPriority.CRITICAL.value == "critical"


class TestEventStatus:
    """事件状态测试"""

    def test_status_values(self):
        """测试状态值"""
        assert EventStatus.PENDING.value == "pending"
        assert EventStatus.PROCESSED.value == "processed"
        assert EventStatus.FAILED.value == "failed"
        assert EventStatus.REPLAYED.value == "replayed"


# ==================== 数据类测试 ====================

class TestTradingEvent:
    """交易事件测试"""

    def test_event_creation(self):
        """测试事件创建"""
        event = TradingEvent(
            event_id="evt-001",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
            data={"order_id": "ord-001"},
        )

        assert event.event_id == "evt-001"
        assert event.event_type == EventType.ORDER_CREATED
        assert event.source == "test"
        assert event.data["order_id"] == "ord-001"

    def test_event_to_dict(self):
        """测试转换为字典"""
        event = TradingEvent(
            event_id="evt-001",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
        )

        d = event.to_dict()

        assert d["event_id"] == "evt-001"
        assert d["event_type"] == "order_created"
        assert d["source"] == "test"

    def test_event_from_dict(self):
        """测试从字典创建"""
        d = {
            "event_id": "evt-001",
            "event_type": "order_created",
            "timestamp": datetime.now().isoformat(),
            "source": "test",
            "data": {"symbol": "AAPL"},
        }

        event = TradingEvent.from_dict(d)

        assert event.event_id == "evt-001"
        assert event.event_type == EventType.ORDER_CREATED
        assert event.data["symbol"] == "AAPL"

    def test_event_json_serialization(self):
        """测试JSON序列化"""
        event = TradingEvent(
            event_id="evt-001",
            event_type=EventType.TRADE_EXECUTED,
            timestamp=datetime.now(),
            source="test",
            data={"price": 150.0},
        )

        json_str = event.to_json()
        restored = TradingEvent.from_json(json_str)

        assert restored.event_id == event.event_id
        assert restored.event_type == event.event_type
        assert restored.data["price"] == 150.0

    def test_event_with_correlation(self):
        """测试带关联ID的事件"""
        event = TradingEvent(
            event_id="evt-002",
            event_type=EventType.ORDER_FILLED,
            timestamp=datetime.now(),
            source="test",
            correlation_id="corr-001",
            causation_id="evt-001",
        )

        assert event.correlation_id == "corr-001"
        assert event.causation_id == "evt-001"


class TestEventFilter:
    """事件过滤器测试"""

    def test_filter_by_event_type(self):
        """测试按事件类型过滤"""
        filter = EventFilter(
            event_types={EventType.ORDER_CREATED, EventType.ORDER_FILLED}
        )

        event1 = TradingEvent(
            event_id="1",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
        )
        event2 = TradingEvent(
            event_id="2",
            event_type=EventType.TRADE_EXECUTED,
            timestamp=datetime.now(),
            source="test",
        )

        assert filter.matches(event1) is True
        assert filter.matches(event2) is False

    def test_filter_by_source(self):
        """测试按来源过滤"""
        filter = EventFilter(sources={"oms", "risk"})

        event1 = TradingEvent(
            event_id="1",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="oms",
        )
        event2 = TradingEvent(
            event_id="2",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="other",
        )

        assert filter.matches(event1) is True
        assert filter.matches(event2) is False

    def test_filter_by_time_range(self):
        """测试按时间范围过滤"""
        now = datetime.now()
        filter = EventFilter(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        event1 = TradingEvent(
            event_id="1",
            event_type=EventType.ORDER_CREATED,
            timestamp=now,
            source="test",
        )
        event2 = TradingEvent(
            event_id="2",
            event_type=EventType.ORDER_CREATED,
            timestamp=now - timedelta(hours=2),
            source="test",
        )

        assert filter.matches(event1) is True
        assert filter.matches(event2) is False

    def test_filter_by_priority(self):
        """测试按优先级过滤"""
        filter = EventFilter(priorities={EventPriority.HIGH, EventPriority.CRITICAL})

        event1 = TradingEvent(
            event_id="1",
            event_type=EventType.RISK_ALERT,
            timestamp=datetime.now(),
            source="test",
            priority=EventPriority.HIGH,
        )
        event2 = TradingEvent(
            event_id="2",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
            priority=EventPriority.NORMAL,
        )

        assert filter.matches(event1) is True
        assert filter.matches(event2) is False

    def test_filter_by_symbol(self):
        """测试按标的过滤"""
        filter = EventFilter(symbols={"AAPL", "GOOGL"})

        event1 = TradingEvent(
            event_id="1",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
            data={"symbol": "AAPL"},
        )
        event2 = TradingEvent(
            event_id="2",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
            data={"symbol": "MSFT"},
        )

        assert filter.matches(event1) is True
        assert filter.matches(event2) is False

    def test_filter_combined(self):
        """测试组合过滤"""
        filter = EventFilter(
            event_types={EventType.ORDER_CREATED},
            sources={"oms"},
            priorities={EventPriority.HIGH},
        )

        event1 = TradingEvent(
            event_id="1",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="oms",
            priority=EventPriority.HIGH,
        )
        event2 = TradingEvent(
            event_id="2",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="oms",
            priority=EventPriority.NORMAL,  # 不匹配
        )

        assert filter.matches(event1) is True
        assert filter.matches(event2) is False


class TestSubscription:
    """订阅测试"""

    def test_subscription_creation(self):
        """测试订阅创建"""
        callback = Mock()
        subscription = Subscription(
            subscription_id="sub-001",
            filter=EventFilter(),
            callback=callback,
        )

        assert subscription.subscription_id == "sub-001"
        assert subscription.event_count == 0

    def test_subscription_to_dict(self):
        """测试订阅转换为字典"""
        callback = Mock()
        subscription = Subscription(
            subscription_id="sub-001",
            filter=EventFilter(),
            callback=callback,
        )

        d = subscription.to_dict()

        assert d["subscription_id"] == "sub-001"
        assert d["is_async"] is False


# ==================== 存储测试 ====================

class TestInMemoryEventStore:
    """内存存储测试"""

    @pytest.fixture
    def store(self):
        return InMemoryEventStore(max_size=100)

    @pytest.mark.asyncio
    async def test_append_event(self, store):
        """测试追加事件"""
        event = TradingEvent(
            event_id="evt-001",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
        )

        result = await store.append(event)

        assert result is True
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_get_event(self, store):
        """测试获取事件"""
        event = TradingEvent(
            event_id="evt-001",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
        )
        await store.append(event)

        retrieved = await store.get("evt-001")

        assert retrieved is not None
        assert retrieved.event_id == "evt-001"

    @pytest.mark.asyncio
    async def test_get_nonexistent_event(self, store):
        """测试获取不存在的事件"""
        retrieved = await store.get("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_query_events(self, store):
        """测试查询事件"""
        for i in range(10):
            event = TradingEvent(
                event_id=f"evt-{i:03d}",
                event_type=EventType.ORDER_CREATED if i % 2 == 0 else EventType.TRADE_EXECUTED,
                timestamp=datetime.now(),
                source="test",
            )
            await store.append(event)

        filter = EventFilter(event_types={EventType.ORDER_CREATED})
        results = await store.query(filter)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_with_limit(self, store):
        """测试带限制的查询"""
        for i in range(10):
            event = TradingEvent(
                event_id=f"evt-{i:03d}",
                event_type=EventType.ORDER_CREATED,
                timestamp=datetime.now(),
                source="test",
            )
            await store.append(event)

        results = await store.query(EventFilter(), limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_get_latest(self, store):
        """测试获取最新事件"""
        for i in range(10):
            event = TradingEvent(
                event_id=f"evt-{i:03d}",
                event_type=EventType.ORDER_CREATED,
                timestamp=datetime.now(),
                source="test",
            )
            await store.append(event)

        latest = await store.get_latest(3)

        assert len(latest) == 3
        assert latest[-1].event_id == "evt-009"

    @pytest.mark.asyncio
    async def test_max_size_limit(self, store):
        """测试最大容量限制"""
        for i in range(150):
            event = TradingEvent(
                event_id=f"evt-{i:03d}",
                event_type=EventType.ORDER_CREATED,
                timestamp=datetime.now(),
                source="test",
            )
            await store.append(event)

        assert await store.count() == 100  # max_size=100

    def test_clear_store(self, store):
        """测试清空存储"""
        asyncio.run(store.append(TradingEvent(
            event_id="evt-001",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
        )))

        store.clear()

        assert asyncio.run(store.count()) == 0


# ==================== 处理器测试 ====================

class TestLoggingProcessor:
    """日志处理器测试"""

    @pytest.mark.asyncio
    async def test_process_event(self):
        """测试处理事件"""
        processor = LoggingProcessor()
        event = TradingEvent(
            event_id="evt-001",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
        )

        result = await processor.process(event)

        assert result is True

    def test_processor_name(self):
        """测试处理器名称"""
        processor = LoggingProcessor()
        assert processor.name == "logging_processor"


class TestAuditProcessor:
    """审计处理器测试"""

    @pytest.mark.asyncio
    async def test_process_event(self):
        """测试处理事件"""
        processor = AuditProcessor()
        event = TradingEvent(
            event_id="evt-001",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
            user_id="user-001",
        )

        result = await processor.process(event)

        assert result is True
        assert len(processor.audit_records) == 1

    @pytest.mark.asyncio
    async def test_get_audit_trail(self):
        """测试获取审计追踪"""
        processor = AuditProcessor()

        for i in range(5):
            event = TradingEvent(
                event_id=f"evt-{i:03d}",
                event_type=EventType.ORDER_CREATED,
                timestamp=datetime.now(),
                source="test",
            )
            await processor.process(event)

        trail = processor.get_audit_trail("evt-002")

        assert len(trail) == 1
        assert trail[0].event_id == "evt-002"

    def test_processor_name(self):
        """测试处理器名称"""
        processor = AuditProcessor()
        assert processor.name == "audit_processor"


class TestComplianceProcessor:
    """合规处理器测试"""

    @pytest.mark.asyncio
    async def test_process_with_violation(self):
        """测试处理带违规的事件"""
        processor = ComplianceProcessor()

        # 添加规则：订单金额超过100万违规
        def check_order_size(event: TradingEvent):
            if event.data.get("notional", 0) > 1000000:
                return "Order size exceeds limit"
            return None

        processor.add_rule(check_order_size)

        event = TradingEvent(
            event_id="evt-001",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
            data={"notional": 2000000},
        )

        await processor.process(event)

        violations = processor.get_violations()
        assert len(violations) == 1

    @pytest.mark.asyncio
    async def test_process_without_violation(self):
        """测试处理无违规的事件"""
        processor = ComplianceProcessor()

        def check_order_size(event: TradingEvent):
            if event.data.get("notional", 0) > 1000000:
                return "Order size exceeds limit"
            return None

        processor.add_rule(check_order_size)

        event = TradingEvent(
            event_id="evt-001",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            source="test",
            data={"notional": 500000},
        )

        await processor.process(event)

        violations = processor.get_violations()
        assert len(violations) == 0


# ==================== 事件流测试 ====================

class TestTradingEventStream:
    """交易事件流测试"""

    @pytest.fixture
    def stream(self):
        config = StreamConfig(
            max_buffer_size=1000,
            max_history_size=10000,
            flush_interval=0.1,
            heartbeat_interval=0,  # 禁用心跳
        )
        return TradingEventStream(config)

    @pytest.mark.asyncio
    async def test_start_stop(self, stream):
        """测试启动和停止"""
        await stream.start()
        assert stream.state == StreamState.RUNNING

        await stream.stop()
        assert stream.state == StreamState.STOPPED

    @pytest.mark.asyncio
    async def test_emit_event(self, stream):
        """测试发送事件"""
        await stream.start()

        event = await stream.emit(
            EventType.ORDER_CREATED,
            source="test",
            data={"order_id": "ord-001"},
        )

        assert event.event_type == EventType.ORDER_CREATED
        assert stream.stats["events_received"] == 2  # 包含系统启动事件

        await stream.stop()

    def test_emit_sync(self, stream):
        """测试同步发送事件"""
        event = stream.emit_sync(
            EventType.ORDER_CREATED,
            source="test",
            data={"order_id": "ord-001"},
        )

        assert event.event_type == EventType.ORDER_CREATED
        assert stream.stats["events_received"] == 1

    @pytest.mark.asyncio
    async def test_emit_with_priority(self, stream):
        """测试带优先级的事件"""
        await stream.start()

        event = await stream.emit(
            EventType.RISK_ALERT,
            source="risk",
            priority=EventPriority.CRITICAL,
            data={"message": "High risk detected"},
        )

        assert event.priority == EventPriority.CRITICAL
        # Critical事件立即处理
        assert event.status == EventStatus.PROCESSED

        await stream.stop()

    @pytest.mark.asyncio
    async def test_subscribe(self, stream):
        """测试订阅"""
        events_received = []

        def callback(event):
            events_received.append(event)

        sub_id = stream.subscribe(callback)

        await stream.start()
        await stream.emit(EventType.ORDER_CREATED, source="test")
        await asyncio.sleep(0.2)  # 等待处理

        assert len(events_received) >= 1
        assert sub_id in stream.subscriptions

        await stream.stop()

    @pytest.mark.asyncio
    async def test_subscribe_with_filter(self, stream):
        """测试带过滤器的订阅"""
        order_events = []

        filter = EventFilter(event_types={EventType.ORDER_CREATED})
        stream.subscribe(lambda e: order_events.append(e), filter=filter)

        await stream.start()
        await stream.emit(EventType.ORDER_CREATED, source="test")
        await stream.emit(EventType.TRADE_EXECUTED, source="test")
        await asyncio.sleep(0.2)

        # 只应收到ORDER_CREATED事件
        order_created_count = sum(1 for e in order_events if e.event_type == EventType.ORDER_CREATED)
        assert order_created_count >= 1

        await stream.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe(self, stream):
        """测试取消订阅"""
        callback = Mock()
        sub_id = stream.subscribe(callback)

        result = stream.unsubscribe(sub_id)

        assert result is True
        assert sub_id not in stream.subscriptions

    @pytest.mark.asyncio
    async def test_add_processor(self, stream):
        """测试添加处理器"""
        processor = LoggingProcessor()
        stream.add_processor(processor)

        assert len(stream.processors) == 1

    @pytest.mark.asyncio
    async def test_remove_processor(self, stream):
        """测试移除处理器"""
        processor = LoggingProcessor()
        stream.add_processor(processor)

        result = stream.remove_processor("logging_processor")

        assert result is True
        assert len(stream.processors) == 0

    @pytest.mark.asyncio
    async def test_get_event(self, stream):
        """测试获取事件"""
        await stream.start()

        event = await stream.emit(EventType.ORDER_CREATED, source="test")
        await asyncio.sleep(0.2)

        retrieved = await stream.get_event(event.event_id)

        assert retrieved is not None
        assert retrieved.event_id == event.event_id

        await stream.stop()

    @pytest.mark.asyncio
    async def test_query_events(self, stream):
        """测试查询事件"""
        await stream.start()

        for i in range(5):
            await stream.emit(EventType.ORDER_CREATED, source="test")
            await stream.emit(EventType.TRADE_EXECUTED, source="test")

        await asyncio.sleep(0.3)

        filter = EventFilter(event_types={EventType.ORDER_CREATED})
        results = await stream.query_events(filter)

        assert len(results) >= 5

        await stream.stop()

    @pytest.mark.asyncio
    async def test_get_latest_events(self, stream):
        """测试获取最新事件"""
        await stream.start()

        for i in range(10):
            await stream.emit(EventType.ORDER_CREATED, source="test", data={"index": i})

        await asyncio.sleep(0.3)

        latest = await stream.get_latest_events(5)

        assert len(latest) == 5

        await stream.stop()

    @pytest.mark.asyncio
    async def test_count_events(self, stream):
        """测试统计事件"""
        await stream.start()

        for i in range(10):
            await stream.emit(EventType.ORDER_CREATED, source="test")

        await asyncio.sleep(0.3)

        count = await stream.count_events()

        assert count >= 10

        await stream.stop()

    @pytest.mark.asyncio
    async def test_on_event_callback(self, stream):
        """测试事件回调"""
        events = []
        stream.on_event(lambda e: events.append(e))

        await stream.start()
        await stream.emit(EventType.ORDER_CREATED, source="test")
        await asyncio.sleep(0.2)

        assert len(events) >= 1

        await stream.stop()

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, stream):
        """测试创建检查点"""
        await stream.start()

        for i in range(5):
            await stream.emit(EventType.ORDER_CREATED, source="test")

        await asyncio.sleep(0.2)

        checkpoint = stream.create_checkpoint()

        assert checkpoint.checkpoint_id is not None
        assert checkpoint.event_count >= 5
        assert len(checkpoint.checksum) == 16

        await stream.stop()

    @pytest.mark.asyncio
    async def test_get_statistics(self, stream):
        """测试获取统计"""
        await stream.start()

        for i in range(5):
            await stream.emit(EventType.ORDER_CREATED, source="test")

        await asyncio.sleep(0.2)

        stats = stream.get_statistics()

        assert stats["state"] == "running"
        assert stats["stats"]["events_received"] >= 5
        assert "subscriptions" in stats

        await stream.stop()

    @pytest.mark.asyncio
    async def test_replay(self, stream):
        """测试回放"""
        await stream.start()

        for i in range(5):
            await stream.emit(
                EventType.ORDER_CREATED,
                source="test",
                data={"index": i},
            )

        await asyncio.sleep(0.3)

        replayed_events = []

        config = ReplayConfig(
            speed_multiplier=10.0,
            preserve_timing=False,
        )

        count = await stream.replay(
            config=config,
            callback=lambda e: replayed_events.append(e),
        )

        assert count >= 5
        assert len(replayed_events) >= 5

        await stream.stop()

    @pytest.mark.asyncio
    async def test_replay_with_filter(self, stream):
        """测试带过滤器的回放"""
        await stream.start()

        for i in range(5):
            await stream.emit(EventType.ORDER_CREATED, source="test")
            await stream.emit(EventType.TRADE_EXECUTED, source="test")

        await asyncio.sleep(0.3)

        replayed_events = []
        filter = EventFilter(event_types={EventType.ORDER_CREATED})
        config = ReplayConfig(filter=filter, preserve_timing=False)

        count = await stream.replay(
            config=config,
            callback=lambda e: replayed_events.append(e),
        )

        # 只回放ORDER_CREATED事件
        for e in replayed_events:
            if e.event_type != EventType.SYSTEM_START:
                assert e.event_type == EventType.ORDER_CREATED

        await stream.stop()

    @pytest.mark.asyncio
    async def test_async_subscriber(self, stream):
        """测试异步订阅者"""
        events = []

        async def async_callback(event):
            await asyncio.sleep(0.01)
            events.append(event)

        stream.subscribe(async_callback, is_async=True)

        await stream.start()
        await stream.emit(EventType.ORDER_CREATED, source="test")
        await asyncio.sleep(0.3)

        assert len(events) >= 1

        await stream.stop()


# ==================== 便捷函数测试 ====================

class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_create_order_event(self):
        """测试创建订单事件"""
        data = create_order_event(
            EventType.ORDER_CREATED,
            order_id="ord-001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
        )

        assert data["order_id"] == "ord-001"
        assert data["symbol"] == "AAPL"
        assert data["side"] == "buy"
        assert data["quantity"] == 100
        assert data["price"] == 150.0

    def test_create_trade_event(self):
        """测试创建成交事件"""
        data = create_trade_event(
            trade_id="trd-001",
            order_id="ord-001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
        )

        assert data["trade_id"] == "trd-001"
        assert data["order_id"] == "ord-001"
        assert data["symbol"] == "AAPL"

    def test_create_position_event(self):
        """测试创建持仓事件"""
        data = create_position_event(
            symbol="AAPL",
            side="long",
            quantity=100,
            avg_price=150.0,
            market_value=15000.0,
            unrealized_pnl=500.0,
        )

        assert data["symbol"] == "AAPL"
        assert data["side"] == "long"
        assert data["unrealized_pnl"] == 500.0

    def test_create_risk_event(self):
        """测试创建风险事件"""
        data = create_risk_event(
            risk_type="var",
            level="warning",
            message="VaR limit exceeded",
            current_value=120000,
            threshold=100000,
        )

        assert data["risk_type"] == "var"
        assert data["level"] == "warning"
        assert data["current_value"] == 120000


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_event_stream(self):
        """测试创建事件流"""
        stream = create_event_stream(
            max_buffer_size=5000,
            enable_logging=True,
            enable_audit=True,
        )

        assert stream.config.max_buffer_size == 5000
        assert len(stream.processors) == 2  # logging + audit

    def test_create_event_stream_minimal(self):
        """测试创建最小事件流"""
        stream = create_event_stream(
            enable_logging=False,
            enable_audit=False,
        )

        assert len(stream.processors) == 0

    def test_create_event_filter(self):
        """测试创建事件过滤器"""
        filter = create_event_filter(
            event_types=["order_created", "order_filled"],
            sources=["oms"],
            priorities=["high"],
        )

        assert EventType.ORDER_CREATED in filter.event_types
        assert "oms" in filter.sources
        assert EventPriority.HIGH in filter.priorities

    @pytest.mark.asyncio
    async def test_quick_emit(self):
        """测试快速发送"""
        stream = create_event_stream(enable_logging=False, enable_audit=False)
        await stream.start()

        event = await quick_emit(
            stream,
            event_type="order_created",
            source="test",
            data={"symbol": "AAPL"},
        )

        assert event.event_type == EventType.ORDER_CREATED
        assert event.data["symbol"] == "AAPL"

        await stream.stop()


# ==================== 边缘情况测试 ====================

class TestEdgeCases:
    """边缘情况测试"""

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """测试空事件流"""
        stream = TradingEventStream()

        events = await stream.query_events()
        assert len(events) == 0

        count = await stream.count_events()
        assert count == 0

    @pytest.mark.asyncio
    async def test_double_start(self):
        """测试重复启动"""
        stream = TradingEventStream()
        await stream.start()
        await stream.start()  # 应该无影响

        assert stream.state == StreamState.RUNNING

        await stream.stop()

    @pytest.mark.asyncio
    async def test_double_stop(self):
        """测试重复停止"""
        stream = TradingEventStream()
        await stream.start()
        await stream.stop()
        await stream.stop()  # 应该无影响

        assert stream.state == StreamState.STOPPED

    def test_unsubscribe_nonexistent(self):
        """测试取消不存在的订阅"""
        stream = TradingEventStream()
        result = stream.unsubscribe("nonexistent")
        assert result is False

    def test_remove_nonexistent_processor(self):
        """测试移除不存在的处理器"""
        stream = TradingEventStream()
        result = stream.remove_processor("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_callback_error_handling(self):
        """测试回调错误处理"""
        stream = TradingEventStream(StreamConfig(heartbeat_interval=0))

        def bad_callback(event):
            raise ValueError("Test error")

        stream.subscribe(bad_callback)

        await stream.start()
        # 不应该因为回调错误而崩溃
        await stream.emit(EventType.ORDER_CREATED, source="test")
        await asyncio.sleep(0.2)

        await stream.stop()


# ==================== 序列化测试 ====================

class TestSerialization:
    """序列化测试"""

    def test_audit_record_to_dict(self):
        """测试审计记录序列化"""
        record = AuditRecord(
            audit_id="aud-001",
            event_id="evt-001",
            action="processed",
            timestamp=datetime.now(),
            user_id="user-001",
        )

        d = record.to_dict()

        assert d["audit_id"] == "aud-001"
        assert d["event_id"] == "evt-001"
        assert d["action"] == "processed"

    def test_checkpoint_to_dict(self):
        """测试检查点序列化"""
        checkpoint = ComplianceCheckpoint(
            checkpoint_id="cp-001",
            timestamp=datetime.now(),
            event_count=100,
            last_event_id="evt-100",
            checksum="abc123",
        )

        d = checkpoint.to_dict()

        assert d["checkpoint_id"] == "cp-001"
        assert d["event_count"] == 100
        assert d["checksum"] == "abc123"

    def test_stream_config_to_dict(self):
        """测试流配置序列化"""
        from dataclasses import asdict

        config = StreamConfig(
            max_buffer_size=5000,
            enable_persistence=True,
        )

        d = asdict(config)

        assert d["max_buffer_size"] == 5000
        assert d["enable_persistence"] is True
