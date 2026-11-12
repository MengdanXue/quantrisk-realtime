"""
执行监控引擎测试
Test Execution Monitor
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from quantrisk_realtime.realtime.execution_monitor import (
    # 枚举
    OrderState,
    ExecutionVenue,
    SlippageType,
    AlertSeverity,
    MonitorState,
    # 数据类
    OrderExecution,
    ExecutionFill,
    ExecutionAlert,
    ExecutionMetrics,
    MonitorConfig,
    ExecutionReport,
    # 接口
    MarketDataProvider,
    # 模拟实现
    SimulatedMarketDataProvider,
    # 核心引擎
    ExecutionMonitor,
    # 工厂函数
    create_execution_monitor,
    create_order_execution,
    quick_execution_report,
)


# ==================== 枚举测试 ====================

class TestEnums:
    """枚举测试"""

    def test_order_state_values(self):
        """测试订单状态枚举值"""
        assert OrderState.PENDING.value == "pending"
        assert OrderState.SUBMITTED.value == "submitted"
        assert OrderState.FILLED.value == "filled"
        assert OrderState.CANCELLED.value == "cancelled"
        assert OrderState.REJECTED.value == "rejected"

    def test_execution_venue_values(self):
        """测试执行场所枚举值"""
        assert ExecutionVenue.EXCHANGE.value == "exchange"
        assert ExecutionVenue.DARK_POOL.value == "dark_pool"
        assert ExecutionVenue.OTC.value == "otc"

    def test_slippage_type_values(self):
        """测试滑点类型枚举值"""
        assert SlippageType.POSITIVE.value == "positive"
        assert SlippageType.NEGATIVE.value == "negative"
        assert SlippageType.ZERO.value == "zero"

    def test_alert_severity_values(self):
        """测试告警严重程度枚举值"""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


# ==================== 数据类测试 ====================

class TestOrderExecution:
    """订单执行记录测试"""

    def test_create_order(self):
        """测试创建订单"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
        )

        assert order.order_id == "ORD001"
        assert order.symbol == "AAPL"
        assert order.side == "buy"
        assert order.state == OrderState.PENDING

    def test_fill_rate(self):
        """测试成交率计算"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
            filled_quantity=Decimal("75"),
        )

        assert order.fill_rate == Decimal("0.75")

    def test_fill_rate_zero_quantity(self):
        """测试零数量成交率"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("0"),
            requested_price=None,
        )

        assert order.fill_rate == Decimal("0")

    def test_slippage_buy_positive(self):
        """测试买入正滑点"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
            arrival_price=Decimal("150.00"),
            avg_fill_price=Decimal("149.50"),  # 成交价低于到达价,有利
            filled_quantity=Decimal("100"),
        )

        assert order.slippage == Decimal("0.50")  # 正滑点
        assert order.slippage_type == SlippageType.POSITIVE

    def test_slippage_buy_negative(self):
        """测试买入负滑点"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
            arrival_price=Decimal("150.00"),
            avg_fill_price=Decimal("150.50"),  # 成交价高于到达价,不利
            filled_quantity=Decimal("100"),
        )

        assert order.slippage == Decimal("-0.50")  # 负滑点
        assert order.slippage_type == SlippageType.NEGATIVE

    def test_slippage_sell_positive(self):
        """测试卖出正滑点"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="sell",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
            arrival_price=Decimal("150.00"),
            avg_fill_price=Decimal("150.50"),  # 成交价高于到达价,有利
            filled_quantity=Decimal("100"),
        )

        assert order.slippage == Decimal("0.50")  # 正滑点
        assert order.slippage_type == SlippageType.POSITIVE

    def test_slippage_bps(self):
        """测试滑点基点计算"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
            arrival_price=Decimal("100.00"),
            avg_fill_price=Decimal("100.10"),  # 0.1%滑点
            filled_quantity=Decimal("100"),
        )

        assert order.slippage_bps == Decimal("-10")  # -10bps

    def test_latency_calculations(self):
        """测试延迟计算"""
        now = datetime.now()
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
            created_at=now,
            submitted_at=now + timedelta(milliseconds=50),
            acknowledged_at=now + timedelta(milliseconds=100),
            first_fill_at=now + timedelta(milliseconds=150),
            completed_at=now + timedelta(milliseconds=200),
        )

        assert order.submission_latency_ms == pytest.approx(50, abs=1)
        assert order.ack_latency_ms == pytest.approx(50, abs=1)
        assert order.first_fill_latency_ms == pytest.approx(100, abs=1)
        assert order.total_execution_time_ms == pytest.approx(150, abs=1)

    def test_is_complete(self):
        """测试订单完成状态"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
        )

        order.state = OrderState.PENDING
        assert order.is_complete is False

        order.state = OrderState.SUBMITTED
        assert order.is_complete is False

        order.state = OrderState.FILLED
        assert order.is_complete is True

        order.state = OrderState.CANCELLED
        assert order.is_complete is True

    def test_notional(self):
        """测试名义金额计算"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
            filled_quantity=Decimal("100"),
            avg_fill_price=Decimal("150.00"),
        )

        assert order.notional == Decimal("15000.00")

    def test_to_dict(self):
        """测试字典转换"""
        order = OrderExecution(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            requested_quantity=Decimal("100"),
            requested_price=None,
            arrival_price=Decimal("150.00"),
            filled_quantity=Decimal("100"),
            avg_fill_price=Decimal("150.10"),
        )

        d = order.to_dict()
        assert d["order_id"] == "ORD001"
        assert d["symbol"] == "AAPL"
        assert d["side"] == "buy"
        assert d["filled_quantity"] == 100.0


class TestExecutionFill:
    """成交记录测试"""

    def test_create_fill(self):
        """测试创建成交记录"""
        fill = ExecutionFill(
            fill_id="F001",
            order_id="ORD001",
            quantity=Decimal("50"),
            price=Decimal("150.00"),
            venue=ExecutionVenue.EXCHANGE,
        )

        assert fill.fill_id == "F001"
        assert fill.quantity == Decimal("50")
        assert fill.venue == ExecutionVenue.EXCHANGE

    def test_to_dict(self):
        """测试字典转换"""
        fill = ExecutionFill(
            fill_id="F001",
            order_id="ORD001",
            quantity=Decimal("50"),
            price=Decimal("150.00"),
            venue=ExecutionVenue.EXCHANGE,
            fees=Decimal("1.50"),
        )

        d = fill.to_dict()
        assert d["fill_id"] == "F001"
        assert d["quantity"] == 50.0
        assert d["fees"] == 1.50


class TestExecutionAlert:
    """执行告警测试"""

    def test_create_alert(self):
        """测试创建告警"""
        alert = ExecutionAlert(
            alert_id="AL001",
            order_id="ORD001",
            severity=AlertSeverity.WARNING,
            alert_type="high_slippage",
            message="High slippage detected",
        )

        assert alert.alert_id == "AL001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.acknowledged is False

    def test_to_dict(self):
        """测试字典转换"""
        alert = ExecutionAlert(
            alert_id="AL001",
            order_id="ORD001",
            severity=AlertSeverity.CRITICAL,
            alert_type="slow_execution",
            message="Slow execution",
        )

        d = alert.to_dict()
        assert d["severity"] == "critical"
        assert d["alert_type"] == "slow_execution"


class TestExecutionMetrics:
    """执行指标测试"""

    def test_create_metrics(self):
        """测试创建指标"""
        now = datetime.now()
        metrics = ExecutionMetrics(
            period_start=now - timedelta(hours=1),
            period_end=now,
            total_orders=100,
            filled_orders=95,
        )

        assert metrics.total_orders == 100
        assert metrics.filled_orders == 95

    def test_to_dict(self):
        """测试字典转换"""
        now = datetime.now()
        metrics = ExecutionMetrics(
            period_start=now - timedelta(hours=1),
            period_end=now,
            total_orders=100,
            avg_slippage_bps=Decimal("5.5"),
        )

        d = metrics.to_dict()
        assert d["total_orders"] == 100
        assert d["avg_slippage_bps"] == 5.5


# ==================== 模拟提供者测试 ====================

class TestSimulatedProviders:
    """模拟提供者测试"""

    @pytest.mark.asyncio
    async def test_market_data_provider(self):
        """测试市场数据提供者"""
        provider = SimulatedMarketDataProvider()

        # 初始无数据
        price = await provider.get_current_price("AAPL")
        assert price is None

        # 设置价格
        provider.set_price("AAPL", Decimal("150.00"), Decimal("0.02"))

        price = await provider.get_current_price("AAPL")
        assert price == Decimal("150.00")

        bid, ask = await provider.get_bid_ask("AAPL")
        assert bid == Decimal("149.99")
        assert ask == Decimal("150.01")


# ==================== 引擎核心功能测试 ====================

class TestExecutionMonitor:
    """执行监控引擎测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ExecutionMonitor()

    def test_track_order(self):
        """测试追踪订单"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )

        order_id = self.monitor.track_order(order)

        assert order_id == order.order_id
        assert order_id in self.monitor.orders
        assert order_id in self.monitor.active_orders

    def test_get_order(self):
        """测试获取订单"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )
        self.monitor.track_order(order)

        retrieved = self.monitor.get_order(order.order_id)
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"

    def test_update_order_state(self):
        """测试更新订单状态"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )
        self.monitor.track_order(order)

        result = self.monitor.update_order_state(
            order.order_id,
            OrderState.SUBMITTED,
        )

        assert result is True
        assert order.state == OrderState.SUBMITTED
        assert order.submitted_at is not None

    def test_record_fill(self):
        """测试记录成交"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
            arrival_price=150.00,
        )
        self.monitor.track_order(order)
        self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)

        fill_id = self.monitor.record_fill(
            order.order_id,
            quantity=Decimal("50"),
            price=Decimal("150.05"),
        )

        assert fill_id is not None
        assert order.filled_quantity == Decimal("50")
        assert order.avg_fill_price == Decimal("150.05")
        assert order.state == OrderState.PARTIAL_FILLED

    def test_record_fill_complete(self):
        """测试完全成交"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )
        self.monitor.track_order(order)
        self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)

        self.monitor.record_fill(
            order.order_id,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )

        assert order.state == OrderState.FILLED
        assert order.completed_at is not None
        assert order.order_id not in self.monitor.active_orders

    def test_record_multiple_fills(self):
        """测试多次成交"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )
        self.monitor.track_order(order)
        self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)

        self.monitor.record_fill(
            order.order_id,
            quantity=Decimal("30"),
            price=Decimal("150.00"),
        )
        self.monitor.record_fill(
            order.order_id,
            quantity=Decimal("70"),
            price=Decimal("150.10"),
        )

        assert order.filled_quantity == Decimal("100")
        # 加权平均价: (30*150 + 70*150.10) / 100 = 150.07
        assert order.avg_fill_price == Decimal("150.07")
        assert len(order.fills) == 2

    def test_set_arrival_price(self):
        """测试设置到达价格"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )
        self.monitor.track_order(order)

        result = self.monitor.set_arrival_price(order.order_id, Decimal("150.00"))

        assert result is True
        assert order.arrival_price == Decimal("150.00")


class TestAlertGeneration:
    """告警生成测试"""

    def setup_method(self):
        """测试前准备"""
        config = MonitorConfig(
            slippage_warning_bps=Decimal("10"),
            slippage_critical_bps=Decimal("25"),
            latency_warning_ms=100,
            latency_critical_ms=500,
            alert_cooldown=0,  # 禁用冷却
        )
        self.monitor = ExecutionMonitor(config=config)

    def test_high_slippage_alert(self):
        """测试高滑点告警"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
            arrival_price=100.00,
        )
        self.monitor.track_order(order)
        self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)

        # 记录高滑点成交 (0.2%滑点 = 20bps)
        self.monitor.record_fill(
            order.order_id,
            quantity=Decimal("100"),
            price=Decimal("100.20"),
        )

        alerts = self.monitor.get_alerts(order_id=order.order_id)
        assert len(alerts) >= 1
        assert any(a.alert_type == "high_slippage" for a in alerts)

    def test_critical_slippage_alert(self):
        """测试严重滑点告警"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
            arrival_price=100.00,
        )
        self.monitor.track_order(order)
        self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)

        # 记录严重滑点成交 (0.3%滑点 = 30bps)
        self.monitor.record_fill(
            order.order_id,
            quantity=Decimal("100"),
            price=Decimal("100.30"),
        )

        alerts = self.monitor.get_alerts(severity=AlertSeverity.CRITICAL)
        assert len(alerts) >= 1

    def test_low_fill_rate_alert(self):
        """测试低成交率告警"""
        config = MonitorConfig(
            fill_rate_warning=Decimal("0.90"),
            fill_rate_critical=Decimal("0.50"),
            alert_cooldown=0,
        )
        monitor = ExecutionMonitor(config=config)

        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )
        monitor.track_order(order)
        monitor.update_order_state(order.order_id, OrderState.SUBMITTED)

        # 部分成交后取消
        monitor.record_fill(
            order.order_id,
            quantity=Decimal("30"),
            price=Decimal("150.00"),
        )
        monitor.update_order_state(order.order_id, OrderState.CANCELLED)

        # 应该有低成交率告警
        alerts = monitor.get_alerts(order_id=order.order_id)
        # 30%成交率应该触发critical告警
        low_fill_alerts = [a for a in alerts if a.alert_type == "low_fill_rate"]
        assert len(low_fill_alerts) > 0
        assert any(a.severity == AlertSeverity.CRITICAL for a in low_fill_alerts)

    def test_acknowledge_alert(self):
        """测试确认告警"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
            arrival_price=100.00,
        )
        self.monitor.track_order(order)
        self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)

        self.monitor.record_fill(
            order.order_id,
            quantity=Decimal("100"),
            price=Decimal("100.30"),  # 高滑点
        )

        alerts = self.monitor.get_unacknowledged_alerts()
        if alerts:
            result = self.monitor.acknowledge_alert(alerts[0].alert_id, "MANAGER001")
            assert result is True
            assert alerts[0].acknowledged is True


class TestMetricsCalculation:
    """指标计算测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ExecutionMonitor()

    def test_calculate_metrics_empty(self):
        """测试空数据指标计算"""
        metrics = self.monitor.calculate_metrics()

        assert metrics.total_orders == 0
        assert metrics.avg_fill_rate == Decimal("0")

    def test_calculate_metrics_with_orders(self):
        """测试有订单的指标计算"""
        # 创建多个订单
        for i in range(5):
            order = create_order_execution(
                symbol="AAPL",
                side="buy",
                quantity=100,
                arrival_price=150.00,
            )
            self.monitor.track_order(order)
            self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)
            self.monitor.record_fill(
                order.order_id,
                quantity=Decimal("100"),
                price=Decimal("150.10"),
                venue=ExecutionVenue.EXCHANGE,
            )

        metrics = self.monitor.calculate_metrics()

        assert metrics.total_orders == 5
        assert metrics.filled_orders == 5
        assert metrics.filled_quantity == Decimal("500")
        assert metrics.venue_breakdown.get("exchange", 0) == 5

    def test_calculate_metrics_with_slippage(self):
        """测试包含滑点的指标计算"""
        # 正滑点订单
        order1 = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
            arrival_price=100.00,
        )
        self.monitor.track_order(order1)
        self.monitor.update_order_state(order1.order_id, OrderState.SUBMITTED)
        self.monitor.record_fill(order1.order_id, Decimal("100"), Decimal("99.90"))

        # 负滑点订单
        order2 = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
            arrival_price=100.00,
        )
        self.monitor.track_order(order2)
        self.monitor.update_order_state(order2.order_id, OrderState.SUBMITTED)
        self.monitor.record_fill(order2.order_id, Decimal("100"), Decimal("100.10"))

        metrics = self.monitor.calculate_metrics()

        assert metrics.positive_slippage_count == 1
        assert metrics.negative_slippage_count == 1


class TestReportGeneration:
    """报告生成测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ExecutionMonitor()

    def test_generate_empty_report(self):
        """测试生成空报告"""
        report = self.monitor.generate_report()

        assert report.report_id is not None
        assert report.metrics is not None
        assert report.metrics.total_orders == 0

    def test_generate_daily_report(self):
        """测试生成日报"""
        # 创建一些订单
        for i in range(3):
            order = create_order_execution(
                symbol="AAPL",
                side="buy",
                quantity=100,
                arrival_price=150.00,
            )
            self.monitor.track_order(order)
            self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)
            self.monitor.record_fill(
                order.order_id,
                quantity=Decimal("100"),
                price=Decimal("150.10"),
                venue=ExecutionVenue.EXCHANGE,
            )

        report = self.monitor.generate_report(report_type="daily")

        assert report.report_type == "daily"
        assert report.metrics.total_orders == 3
        assert len(report.recommendations) > 0

    def test_report_to_dict(self):
        """测试报告字典转换"""
        report = self.monitor.generate_report()
        d = report.to_dict()

        assert "report_id" in d
        assert "metrics" in d
        assert "recommendations" in d


class TestRealTimeMonitoring:
    """实时监控测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ExecutionMonitor()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启动停止"""
        await self.monitor.start()
        assert self.monitor.state == MonitorState.RUNNING

        await self.monitor.stop()
        assert self.monitor.state == MonitorState.IDLE


class TestCallbacks:
    """回调功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ExecutionMonitor()

    def test_on_fill_callback(self):
        """测试成交回调"""
        fills_received = []

        def on_fill(order, fill):
            fills_received.append((order, fill))

        self.monitor.on_fill(on_fill)

        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )
        self.monitor.track_order(order)
        self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)
        self.monitor.record_fill(order.order_id, Decimal("100"), Decimal("150.00"))

        assert len(fills_received) == 1

    def test_on_order_complete_callback(self):
        """测试订单完成回调"""
        completed_orders = []

        def on_complete(order):
            completed_orders.append(order)

        self.monitor.on_order_complete(on_complete)

        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )
        self.monitor.track_order(order)
        self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)
        self.monitor.record_fill(order.order_id, Decimal("100"), Decimal("150.00"))

        assert len(completed_orders) == 1

    def test_on_alert_callback(self):
        """测试告警回调"""
        config = MonitorConfig(
            slippage_warning_bps=Decimal("5"),
            alert_cooldown=0,
        )
        monitor = ExecutionMonitor(config=config)

        alerts_received = []

        def on_alert(alert):
            alerts_received.append(alert)

        monitor.on_alert(on_alert)

        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
            arrival_price=100.00,
        )
        monitor.track_order(order)
        monitor.update_order_state(order.order_id, OrderState.SUBMITTED)
        monitor.record_fill(order.order_id, Decimal("100"), Decimal("100.10"))  # 10bps滑点

        assert len(alerts_received) >= 1


class TestStatistics:
    """统计功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ExecutionMonitor()

    def test_get_statistics(self):
        """测试获取统计"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )
        self.monitor.track_order(order)

        stats = self.monitor.get_statistics()

        assert stats["total_orders_tracked"] == 1
        assert stats["active_orders"] == 1
        assert stats["state"] == "idle"


class TestQueryInterfaces:
    """查询接口测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ExecutionMonitor()

    def test_get_active_orders(self):
        """测试获取活跃订单"""
        order1 = create_order_execution(symbol="AAPL", side="buy", quantity=100)
        order2 = create_order_execution(symbol="GOOGL", side="sell", quantity=50)

        self.monitor.track_order(order1)
        self.monitor.track_order(order2)

        active = self.monitor.get_active_orders()
        assert len(active) == 2

    def test_get_completed_orders(self):
        """测试获取已完成订单"""
        order = create_order_execution(symbol="AAPL", side="buy", quantity=100)
        self.monitor.track_order(order)
        self.monitor.update_order_state(order.order_id, OrderState.SUBMITTED)
        self.monitor.record_fill(order.order_id, Decimal("100"), Decimal("150.00"))

        completed = self.monitor.get_completed_orders()
        assert len(completed) == 1

    def test_get_orders_by_symbol(self):
        """测试按标的获取订单"""
        order1 = create_order_execution(symbol="AAPL", side="buy", quantity=100)
        order2 = create_order_execution(symbol="GOOGL", side="sell", quantity=50)
        order3 = create_order_execution(symbol="AAPL", side="sell", quantity=75)

        self.monitor.track_order(order1)
        self.monitor.track_order(order2)
        self.monitor.track_order(order3)

        aapl_orders = self.monitor.get_orders_by_symbol("AAPL")
        assert len(aapl_orders) == 2

    def test_get_orders_by_strategy(self):
        """测试按策略获取订单"""
        order1 = create_order_execution(symbol="AAPL", side="buy", quantity=100, strategy_id="STR001")
        order2 = create_order_execution(symbol="GOOGL", side="sell", quantity=50, strategy_id="STR002")

        self.monitor.track_order(order1)
        self.monitor.track_order(order2)

        str001_orders = self.monitor.get_orders_by_strategy("STR001")
        assert len(str001_orders) == 1


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_execution_monitor(self):
        """测试创建监控引擎"""
        monitor = create_execution_monitor(
            slippage_warning_bps=15,
            latency_warning_ms=200,
        )

        assert monitor is not None
        assert monitor.config.slippage_warning_bps == Decimal("15")
        assert monitor.config.latency_warning_ms == 200

    def test_create_order_execution(self):
        """测试创建订单执行记录"""
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="limit",
            price=150.00,
            arrival_price=149.90,
            strategy_id="STR001",
        )

        assert order.symbol == "AAPL"
        assert order.requested_quantity == Decimal("100")
        assert order.requested_price == Decimal("150.00")
        assert order.arrival_price == Decimal("149.90")
        assert order.strategy_id == "STR001"

    @pytest.mark.asyncio
    async def test_quick_execution_report(self):
        """测试快速执行报告"""
        monitor = create_execution_monitor()

        # 添加一些订单
        order = create_order_execution(symbol="AAPL", side="buy", quantity=100)
        monitor.track_order(order)
        monitor.update_order_state(order.order_id, OrderState.SUBMITTED)
        monitor.record_fill(order.order_id, Decimal("100"), Decimal("150.00"))

        report = await quick_execution_report(monitor, days=1)

        assert "report_id" in report
        assert "metrics" in report


# ==================== 边界情况测试 ====================

class TestEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ExecutionMonitor()

    def test_get_nonexistent_order(self):
        """测试获取不存在的订单"""
        result = self.monitor.get_order("nonexistent")
        assert result is None

    def test_update_nonexistent_order(self):
        """测试更新不存在的订单"""
        result = self.monitor.update_order_state("nonexistent", OrderState.SUBMITTED)
        assert result is False

    def test_fill_nonexistent_order(self):
        """测试成交不存在的订单"""
        result = self.monitor.record_fill(
            "nonexistent",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
        )
        assert result is None

    def test_set_arrival_price_nonexistent(self):
        """测试设置不存在订单的到达价格"""
        result = self.monitor.set_arrival_price("nonexistent", Decimal("150.00"))
        assert result is False

    def test_acknowledge_nonexistent_alert(self):
        """测试确认不存在的告警"""
        result = self.monitor.acknowledge_alert("nonexistent", "MANAGER001")
        assert result is False


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_order_lifecycle(self):
        """测试完整订单生命周期"""
        monitor = create_execution_monitor()

        # 注册回调
        events = []

        def on_fill(order, fill):
            events.append(("fill", order.order_id, fill.quantity))

        def on_complete(order):
            events.append(("complete", order.order_id, order.state.value))

        monitor.on_fill(on_fill)
        monitor.on_order_complete(on_complete)

        # 1. 创建订单
        order = create_order_execution(
            symbol="AAPL",
            side="buy",
            quantity=100,
            arrival_price=150.00,
            strategy_id="MOM_001",
            trader_id="TR001",
        )

        # 2. 开始追踪
        monitor.track_order(order)

        # 3. 提交订单
        monitor.update_order_state(order.order_id, OrderState.SUBMITTED)
        assert order.submitted_at is not None

        # 4. 收到确认
        monitor.update_order_state(order.order_id, OrderState.ACKNOWLEDGED)
        assert order.acknowledged_at is not None

        # 5. 部分成交
        monitor.record_fill(
            order.order_id,
            quantity=Decimal("50"),
            price=Decimal("150.05"),
            venue=ExecutionVenue.EXCHANGE,
        )
        assert order.state == OrderState.PARTIAL_FILLED

        # 6. 完全成交
        monitor.record_fill(
            order.order_id,
            quantity=Decimal("50"),
            price=Decimal("150.10"),
            venue=ExecutionVenue.EXCHANGE,
        )
        assert order.state == OrderState.FILLED

        # 7. 验证结果
        assert order.filled_quantity == Decimal("100")
        assert order.fill_rate == Decimal("1")
        assert order.is_complete is True
        assert len(order.fills) == 2
        assert order.order_id not in monitor.active_orders

        # 8. 验证回调
        assert len(events) == 3  # 2 fills + 1 complete

        # 9. 生成报告
        report = monitor.generate_report()
        assert report.metrics.filled_orders == 1

    @pytest.mark.asyncio
    async def test_multiple_orders_with_different_outcomes(self):
        """测试多个不同结果的订单"""
        monitor = create_execution_monitor()

        # 完全成交订单
        order1 = create_order_execution(symbol="AAPL", side="buy", quantity=100)
        monitor.track_order(order1)
        monitor.update_order_state(order1.order_id, OrderState.SUBMITTED)
        monitor.record_fill(order1.order_id, Decimal("100"), Decimal("150.00"))

        # 取消订单
        order2 = create_order_execution(symbol="GOOGL", side="sell", quantity=50)
        monitor.track_order(order2)
        monitor.update_order_state(order2.order_id, OrderState.SUBMITTED)
        monitor.update_order_state(order2.order_id, OrderState.CANCELLED)

        # 拒绝订单
        order3 = create_order_execution(symbol="MSFT", side="buy", quantity=75)
        monitor.track_order(order3)
        monitor.update_order_state(order3.order_id, OrderState.REJECTED)

        # 计算指标
        metrics = monitor.calculate_metrics()

        assert metrics.total_orders == 3
        assert metrics.filled_orders == 1
        assert metrics.cancelled_orders == 1
        assert metrics.rejected_orders == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
