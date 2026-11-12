"""
Liquidity Risk Monitor 测试

测试流动性风险监控器的所有功能。
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import sys
sys.path.insert(0, "src")

from quantrisk_realtime.realtime.liquidity_risk_monitor import (
    # Enums
    LiquidityLevel,
    AlertSeverity,
    MonitorState,
    ExitStrategy,
    SpreadType,
    # Data Classes
    OrderBookLevel,
    OrderBook,
    LiquidityMetrics,
    LiquidityAlert,
    LiquidationCostEstimate,
    ExitPlan,
    LiquidityCoverageRatio,
    LiquidityStressScenario,
    LiquidityStressResult,
    LiquidityMonitorConfig,
    # Providers
    SimulatedOrderBookProvider,
    SimulatedVolumeDataProvider,
    SimulatedPositionDataProvider,
    # Main Class
    LiquidityRiskMonitor,
    # Factory Functions
    create_liquidity_monitor,
    create_order_book,
    quick_liquidity_check,
)


# ============ OrderBookLevel Tests ============

class TestOrderBookLevel:
    """测试订单簿层级"""

    def test_create_level(self):
        level = OrderBookLevel(price=100.0, quantity=1000, num_orders=5)
        assert level.price == 100.0
        assert level.quantity == 1000
        assert level.num_orders == 5

    def test_to_dict(self):
        level = OrderBookLevel(price=100.0, quantity=1000)
        d = level.to_dict()
        assert d["price"] == 100.0
        assert d["quantity"] == 1000
        assert d["num_orders"] == 1


# ============ OrderBook Tests ============

class TestOrderBook:
    """测试订单簿"""

    def test_create_order_book(self):
        bids = [OrderBookLevel(99.9, 100), OrderBookLevel(99.8, 200)]
        asks = [OrderBookLevel(100.1, 100), OrderBookLevel(100.2, 200)]
        book = OrderBook(symbol="AAPL", timestamp=datetime.now(), bids=bids, asks=asks)

        assert book.symbol == "AAPL"
        assert len(book.bids) == 2
        assert len(book.asks) == 2

    def test_best_bid_ask(self):
        bids = [OrderBookLevel(99.9, 100), OrderBookLevel(99.8, 200)]
        asks = [OrderBookLevel(100.1, 100), OrderBookLevel(100.2, 200)]
        book = OrderBook(symbol="AAPL", timestamp=datetime.now(), bids=bids, asks=asks)

        assert book.best_bid == 99.9
        assert book.best_ask == 100.1

    def test_mid_price(self):
        bids = [OrderBookLevel(99.9, 100)]
        asks = [OrderBookLevel(100.1, 100)]
        book = OrderBook(symbol="AAPL", timestamp=datetime.now(), bids=bids, asks=asks)

        assert book.mid_price == 100.0

    def test_spread(self):
        bids = [OrderBookLevel(99.9, 100)]
        asks = [OrderBookLevel(100.1, 100)]
        book = OrderBook(symbol="AAPL", timestamp=datetime.now(), bids=bids, asks=asks)

        assert book.spread == pytest.approx(0.2, rel=0.001)

    def test_spread_bps(self):
        bids = [OrderBookLevel(99.9, 100)]
        asks = [OrderBookLevel(100.1, 100)]
        book = OrderBook(symbol="AAPL", timestamp=datetime.now(), bids=bids, asks=asks)

        # spread = 0.2, mid = 100, spread_bps = 0.2/100 * 10000 = 20
        assert book.spread_bps == pytest.approx(20.0, rel=0.01)

    def test_get_depth(self):
        bids = [OrderBookLevel(99.9, 100), OrderBookLevel(99.8, 200), OrderBookLevel(99.7, 300)]
        asks = [OrderBookLevel(100.1, 150), OrderBookLevel(100.2, 250)]
        book = OrderBook(symbol="AAPL", timestamp=datetime.now(), bids=bids, asks=asks)

        assert book.get_depth("bid", 2) == 300
        assert book.get_depth("ask", 2) == 400
        assert book.get_depth("bid", 5) == 600

    def test_empty_order_book(self):
        book = OrderBook(symbol="AAPL", timestamp=datetime.now())
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None

    def test_to_dict(self):
        bids = [OrderBookLevel(99.9, 100)]
        asks = [OrderBookLevel(100.1, 100)]
        book = OrderBook(symbol="AAPL", timestamp=datetime.now(), bids=bids, asks=asks)

        d = book.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["best_bid"] == 99.9
        assert d["best_ask"] == 100.1


# ============ LiquidityMetrics Tests ============

class TestLiquidityMetrics:
    """测试流动性指标"""

    def test_create_metrics(self):
        metrics = LiquidityMetrics(
            symbol="AAPL",
            timestamp=datetime.now(),
            spread_bps=15.0,
            liquidity_score=75.0,
            liquidity_level=LiquidityLevel.GOOD,
        )
        assert metrics.symbol == "AAPL"
        assert metrics.spread_bps == 15.0
        assert metrics.liquidity_score == 75.0

    def test_to_dict(self):
        metrics = LiquidityMetrics(
            symbol="AAPL",
            timestamp=datetime.now(),
            liquidity_level=LiquidityLevel.MODERATE,
        )
        d = metrics.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["liquidity_level"] == "moderate"


# ============ LiquidityAlert Tests ============

class TestLiquidityAlert:
    """测试流动性告警"""

    def test_create_alert(self):
        alert = LiquidityAlert(
            alert_id="LIQ-000001",
            symbol="AAPL",
            timestamp=datetime.now(),
            severity=AlertSeverity.WARNING,
            alert_type="spread_warning",
            message="Spread widening",
            current_value=60.0,
            threshold=50.0,
        )
        assert alert.alert_id == "LIQ-000001"
        assert alert.severity == AlertSeverity.WARNING
        assert not alert.acknowledged

    def test_to_dict(self):
        alert = LiquidityAlert(
            alert_id="LIQ-000001",
            symbol="AAPL",
            timestamp=datetime.now(),
            severity=AlertSeverity.CRITICAL,
            alert_type="liquidity_critical",
            message="Critical liquidity",
            current_value=15.0,
            threshold=20.0,
        )
        d = alert.to_dict()
        assert d["alert_id"] == "LIQ-000001"
        assert d["severity"] == "critical"


# ============ LiquidationCostEstimate Tests ============

class TestLiquidationCostEstimate:
    """测试清算成本估算"""

    def test_create_estimate(self):
        estimate = LiquidationCostEstimate(
            symbol="AAPL",
            timestamp=datetime.now(),
            position_size=1000,
            position_value=150000,
            spread_cost=75,
            market_impact_cost=150,
            total_cost=225,
            total_cost_bps=15,
            recommended_strategy=ExitStrategy.STAGED,
        )
        assert estimate.total_cost == 225
        assert estimate.recommended_strategy == ExitStrategy.STAGED

    def test_to_dict(self):
        estimate = LiquidationCostEstimate(
            symbol="AAPL",
            timestamp=datetime.now(),
            position_size=1000,
            position_value=150000,
        )
        d = estimate.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["position_size"] == 1000


# ============ ExitPlan Tests ============

class TestExitPlan:
    """测试退出计划"""

    def test_create_plan(self):
        plan = ExitPlan(
            plan_id="EXIT-000001",
            symbol="AAPL",
            created_at=datetime.now(),
            position_size=1000,
            strategy=ExitStrategy.TWAP,
            total_slices=10,
        )
        assert plan.plan_id == "EXIT-000001"
        assert plan.status == "pending"
        assert plan.total_slices == 10

    def test_to_dict(self):
        plan = ExitPlan(
            plan_id="EXIT-000001",
            symbol="AAPL",
            created_at=datetime.now(),
            position_size=1000,
            strategy=ExitStrategy.VWAP,
        )
        d = plan.to_dict()
        assert d["strategy"] == "vwap"


# ============ LiquidityCoverageRatio Tests ============

class TestLiquidityCoverageRatio:
    """测试流动性覆盖率"""

    def test_create_lcr(self):
        lcr = LiquidityCoverageRatio(
            timestamp=datetime.now(),
            hqla_level1=1000000,
            total_hqla=1200000,
            net_cash_outflows=1000000,
            lcr=120.0,
            is_compliant=True,
        )
        assert lcr.lcr == 120.0
        assert lcr.is_compliant

    def test_non_compliant(self):
        lcr = LiquidityCoverageRatio(
            timestamp=datetime.now(),
            lcr=85.0,
            is_compliant=False,
        )
        assert not lcr.is_compliant


# ============ LiquidityStressScenario Tests ============

class TestLiquidityStressScenario:
    """测试流动性压力情景"""

    def test_create_scenario(self):
        scenario = LiquidityStressScenario(
            scenario_id="custom_stress",
            name="Custom Stress",
            description="Custom stress scenario",
            spread_multiplier=3.0,
            depth_reduction=0.5,
        )
        assert scenario.spread_multiplier == 3.0
        assert scenario.depth_reduction == 0.5


# ============ Simulated Providers Tests ============

class TestSimulatedOrderBookProvider:
    """测试模拟订单簿提供者"""

    @pytest.mark.asyncio
    async def test_get_order_book(self):
        provider = SimulatedOrderBookProvider()
        provider.set_base_price("AAPL", 150.0)

        book = await provider.get_order_book("AAPL")
        assert book is not None
        assert book.symbol == "AAPL"
        assert book.best_bid is not None
        assert book.best_ask is not None

    @pytest.mark.asyncio
    async def test_set_custom_order_book(self):
        provider = SimulatedOrderBookProvider()

        custom_book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.9, 500)],
            asks=[OrderBookLevel(150.1, 500)],
        )
        provider.set_order_book("AAPL", custom_book)

        book = await provider.get_order_book("AAPL")
        assert book.best_bid == 149.9
        assert book.best_ask == 150.1


class TestSimulatedVolumeDataProvider:
    """测试模拟成交量数据提供者"""

    @pytest.mark.asyncio
    async def test_get_adv(self):
        provider = SimulatedVolumeDataProvider()
        provider.set_adv("AAPL", 5000000)

        adv = await provider.get_adv("AAPL")
        assert adv == 5000000

    @pytest.mark.asyncio
    async def test_get_current_volume(self):
        provider = SimulatedVolumeDataProvider()
        provider.set_current_volume("AAPL", 2500000)

        volume = await provider.get_current_volume("AAPL")
        assert volume == 2500000

    @pytest.mark.asyncio
    async def test_get_historical_volumes(self):
        provider = SimulatedVolumeDataProvider()
        provider.set_adv("AAPL", 5000000)

        volumes = await provider.get_historical_volumes("AAPL", 20)
        assert len(volumes) == 20


class TestSimulatedPositionDataProvider:
    """测试模拟持仓数据提供者"""

    @pytest.mark.asyncio
    async def test_get_position(self):
        provider = SimulatedPositionDataProvider()
        provider.set_position("AAPL", 1000, 150.0)

        position = await provider.get_position("AAPL")
        assert position["quantity"] == 1000
        assert position["avg_price"] == 150.0

    @pytest.mark.asyncio
    async def test_get_positions(self):
        provider = SimulatedPositionDataProvider()
        provider.set_position("AAPL", 1000, 150.0)
        provider.set_position("GOOGL", 500, 120.0)

        positions = await provider.get_positions()
        assert len(positions) == 2


# ============ LiquidityMonitorConfig Tests ============

class TestLiquidityMonitorConfig:
    """测试流动性监控配置"""

    def test_default_config(self):
        config = LiquidityMonitorConfig()
        assert config.update_interval_sec == 5.0
        assert config.spread_warning_bps == 50.0

    def test_custom_config(self):
        config = LiquidityMonitorConfig(
            update_interval_sec=10.0,
            spread_warning_bps=30.0,
        )
        assert config.update_interval_sec == 10.0
        assert config.spread_warning_bps == 30.0

    def test_to_dict(self):
        config = LiquidityMonitorConfig()
        d = config.to_dict()
        assert "update_interval_sec" in d
        assert "spread_warning_bps" in d


# ============ LiquidityRiskMonitor Tests ============

class TestLiquidityRiskMonitor:
    """测试流动性风险监控器"""

    def test_create_monitor(self):
        monitor = LiquidityRiskMonitor()
        assert monitor.state == MonitorState.STOPPED

    @pytest.mark.asyncio
    async def test_start_stop(self):
        monitor = LiquidityRiskMonitor()
        await monitor.start()
        assert monitor.state == MonitorState.RUNNING

        await monitor.stop()
        assert monitor.state == MonitorState.STOPPED

    @pytest.mark.asyncio
    async def test_pause_resume(self):
        monitor = LiquidityRiskMonitor()
        await monitor.start()

        await monitor.pause()
        assert monitor.state == MonitorState.PAUSED

        await monitor.resume()
        assert monitor.state == MonitorState.RUNNING

        await monitor.stop()


class TestSymbolManagement:
    """测试标的管理"""

    def test_add_symbol(self):
        monitor = LiquidityRiskMonitor()
        monitor.add_symbol("AAPL")
        monitor.add_symbol("GOOGL")

        symbols = monitor.get_watched_symbols()
        assert "AAPL" in symbols
        assert "GOOGL" in symbols

    def test_remove_symbol(self):
        monitor = LiquidityRiskMonitor()
        monitor.add_symbol("AAPL")
        monitor.add_symbol("GOOGL")
        monitor.remove_symbol("AAPL")

        symbols = monitor.get_watched_symbols()
        assert "AAPL" not in symbols
        assert "GOOGL" in symbols


class TestMetricsCalculation:
    """测试指标计算"""

    @pytest.mark.asyncio
    async def test_calculate_metrics(self):
        order_book_provider = SimulatedOrderBookProvider()
        order_book_provider.set_base_price("AAPL", 150.0)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)
        volume_provider.set_current_volume("AAPL", 2500000)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        metrics = await monitor.calculate_metrics("AAPL")
        assert metrics is not None
        assert metrics.symbol == "AAPL"
        assert metrics.adv == 5000000
        assert metrics.liquidity_score >= 0
        assert metrics.liquidity_score <= 100

    @pytest.mark.asyncio
    async def test_liquidity_score_calculation(self):
        order_book_provider = SimulatedOrderBookProvider()

        # 创建高流动性订单簿
        high_liq_book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.99, 10000), OrderBookLevel(149.98, 20000)],
            asks=[OrderBookLevel(150.01, 10000), OrderBookLevel(150.02, 20000)],
        )
        order_book_provider.set_order_book("AAPL", high_liq_book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 10000000)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        metrics = await monitor.calculate_metrics("AAPL")
        # 高流动性应该有较高分数
        assert metrics.liquidity_score >= 50

    @pytest.mark.asyncio
    async def test_liquidity_level_determination(self):
        order_book_provider = SimulatedOrderBookProvider()
        volume_provider = SimulatedVolumeDataProvider()

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        # 测试不同分数对应的等级
        assert monitor._determine_liquidity_level(85) == LiquidityLevel.EXCELLENT
        assert monitor._determine_liquidity_level(65) == LiquidityLevel.GOOD
        assert monitor._determine_liquidity_level(45) == LiquidityLevel.MODERATE
        assert monitor._determine_liquidity_level(25) == LiquidityLevel.LOW
        assert monitor._determine_liquidity_level(10) == LiquidityLevel.CRITICAL
        assert monitor._determine_liquidity_level(2) == LiquidityLevel.FROZEN

    @pytest.mark.asyncio
    async def test_metrics_history(self):
        order_book_provider = SimulatedOrderBookProvider()
        order_book_provider.set_base_price("AAPL", 150.0)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )
        monitor.add_symbol("AAPL")

        # 计算多次指标
        for _ in range(5):
            metrics = await monitor.calculate_metrics("AAPL")
            monitor._store_metrics("AAPL", metrics)

        history = monitor.get_metrics_history("AAPL")
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_get_latest_metrics(self):
        order_book_provider = SimulatedOrderBookProvider()
        order_book_provider.set_base_price("AAPL", 150.0)

        volume_provider = SimulatedVolumeDataProvider()

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )
        monitor.add_symbol("AAPL")

        metrics = await monitor.calculate_metrics("AAPL")
        monitor._store_metrics("AAPL", metrics)

        latest = monitor.get_latest_metrics("AAPL")
        assert latest is not None
        assert latest.symbol == "AAPL"


class TestAlertManagement:
    """测试告警管理"""

    @pytest.mark.asyncio
    async def test_alert_generation(self):
        order_book_provider = SimulatedOrderBookProvider()

        # 创建宽价差订单簿触发告警
        wide_spread_book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.0, 100)],
            asks=[OrderBookLevel(151.0, 100)],  # 200bps价差
        )
        order_book_provider.set_order_book("AAPL", wide_spread_book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 1000000)

        alerts_received = []

        def on_alert(alert):
            alerts_received.append(alert)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
            on_alert=on_alert,
        )

        metrics = await monitor.calculate_metrics("AAPL")
        await monitor._check_alerts(metrics)

        # 应该触发价差告警
        assert len(alerts_received) >= 1

    def test_acknowledge_alert(self):
        monitor = LiquidityRiskMonitor()

        # 手动添加告警
        alert = LiquidityAlert(
            alert_id="LIQ-000001",
            symbol="AAPL",
            timestamp=datetime.now(),
            severity=AlertSeverity.WARNING,
            alert_type="spread_warning",
            message="Test",
            current_value=60.0,
            threshold=50.0,
        )
        monitor._alerts.append(alert)

        # 确认告警
        result = monitor.acknowledge_alert("LIQ-000001", "admin")
        assert result is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "admin"

    def test_get_alerts_filtered(self):
        monitor = LiquidityRiskMonitor()

        # 添加多个告警
        for i in range(5):
            alert = LiquidityAlert(
                alert_id=f"LIQ-{i:06d}",
                symbol="AAPL" if i < 3 else "GOOGL",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING if i % 2 == 0 else AlertSeverity.CRITICAL,
                alert_type="test",
                message="Test",
                current_value=0,
                threshold=0,
            )
            monitor._alerts.append(alert)

        # 按标的过滤
        aapl_alerts = monitor.get_alerts(symbol="AAPL")
        assert len(aapl_alerts) == 3

        # 按严重程度过滤
        critical_alerts = monitor.get_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 2


class TestLiquidationCostEstimation:
    """测试清算成本估算"""

    @pytest.mark.asyncio
    async def test_estimate_normal_urgency(self):
        order_book_provider = SimulatedOrderBookProvider()
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.95, 1000)],
            asks=[OrderBookLevel(150.05, 1000)],
        )
        order_book_provider.set_order_book("AAPL", book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        estimate = await monitor.estimate_liquidation_cost("AAPL", 10000, urgency="normal")
        assert estimate.position_size == 10000
        assert estimate.total_cost > 0
        assert estimate.recommended_slices >= 1

    @pytest.mark.asyncio
    async def test_estimate_emergency_urgency(self):
        order_book_provider = SimulatedOrderBookProvider()
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.95, 1000)],
            asks=[OrderBookLevel(150.05, 1000)],
        )
        order_book_provider.set_order_book("AAPL", book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        estimate = await monitor.estimate_liquidation_cost("AAPL", 10000, urgency="emergency")
        assert estimate.recommended_strategy == ExitStrategy.IMMEDIATE

    @pytest.mark.asyncio
    async def test_estimate_large_position(self):
        order_book_provider = SimulatedOrderBookProvider()
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.95, 1000)],
            asks=[OrderBookLevel(150.05, 1000)],
        )
        order_book_provider.set_order_book("AAPL", book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 1000000)  # 较小ADV

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        # 大仓位（50% ADV）
        estimate = await monitor.estimate_liquidation_cost("AAPL", 500000, urgency="normal")
        # 50% ADV (0.5) 属于 0.1 < x <= 0.5 范围，应该是 STAGED
        assert estimate.recommended_strategy in [ExitStrategy.STAGED, ExitStrategy.TWAP]
        assert estimate.recommended_slices >= 3


class TestExitPlan:
    """测试退出计划"""

    @pytest.mark.asyncio
    async def test_create_exit_plan(self):
        order_book_provider = SimulatedOrderBookProvider()
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.95, 1000)],
            asks=[OrderBookLevel(150.05, 1000)],
        )
        order_book_provider.set_order_book("AAPL", book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        plan = await monitor.create_exit_plan("AAPL", 10000)
        assert plan.plan_id.startswith("EXIT-")
        assert plan.position_size == 10000
        assert plan.status == "pending"
        assert len(plan.slices) > 0

    @pytest.mark.asyncio
    async def test_get_exit_plan(self):
        order_book_provider = SimulatedOrderBookProvider()
        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)

        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.95, 1000)],
            asks=[OrderBookLevel(150.05, 1000)],
        )
        order_book_provider.set_order_book("AAPL", book)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        plan = await monitor.create_exit_plan("AAPL", 10000)
        retrieved = monitor.get_exit_plan(plan.plan_id)
        assert retrieved is not None
        assert retrieved.plan_id == plan.plan_id

    @pytest.mark.asyncio
    async def test_cancel_exit_plan(self):
        order_book_provider = SimulatedOrderBookProvider()
        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)

        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.95, 1000)],
            asks=[OrderBookLevel(150.05, 1000)],
        )
        order_book_provider.set_order_book("AAPL", book)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        plan = await monitor.create_exit_plan("AAPL", 10000)
        result = monitor.cancel_exit_plan(plan.plan_id)
        assert result is True
        assert plan.status == "cancelled"

    @pytest.mark.asyncio
    async def test_get_all_exit_plans(self):
        order_book_provider = SimulatedOrderBookProvider()
        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)
        volume_provider.set_adv("GOOGL", 3000000)

        for symbol in ["AAPL", "GOOGL"]:
            book = OrderBook(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=[OrderBookLevel(149.95, 1000)],
                asks=[OrderBookLevel(150.05, 1000)],
            )
            order_book_provider.set_order_book(symbol, book)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        await monitor.create_exit_plan("AAPL", 10000)
        await monitor.create_exit_plan("GOOGL", 5000)

        plans = monitor.get_all_exit_plans()
        assert len(plans) == 2


class TestStressTesting:
    """测试压力测试"""

    @pytest.mark.asyncio
    async def test_run_stress_test(self):
        order_book_provider = SimulatedOrderBookProvider()
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.95, 1000)],
            asks=[OrderBookLevel(150.05, 1000)],
        )
        order_book_provider.set_order_book("AAPL", book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        result = await monitor.run_stress_test("AAPL", "normal_stress")
        assert result.scenario.scenario_id == "normal_stress"
        assert result.stressed_spread_bps > 0
        assert result.risk_level in ["low", "moderate", "high", "severe"]

    @pytest.mark.asyncio
    async def test_run_all_stress_tests(self):
        order_book_provider = SimulatedOrderBookProvider()
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.95, 1000)],
            asks=[OrderBookLevel(150.05, 1000)],
        )
        order_book_provider.set_order_book("AAPL", book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        results = await monitor.run_all_stress_tests("AAPL")
        assert len(results) == 4  # 4个预定义情景

    def test_get_stress_scenarios(self):
        monitor = LiquidityRiskMonitor()
        scenarios = monitor.get_stress_scenarios()

        assert "normal_stress" in scenarios
        assert "severe_stress" in scenarios
        assert "extreme_stress" in scenarios
        assert "flash_crash" in scenarios

    def test_add_custom_stress_scenario(self):
        monitor = LiquidityRiskMonitor()

        custom = LiquidityStressScenario(
            scenario_id="custom",
            name="Custom",
            description="Custom scenario",
            spread_multiplier=4.0,
        )
        monitor.add_stress_scenario(custom)

        scenarios = monitor.get_stress_scenarios()
        assert "custom" in scenarios


class TestLCRCalculation:
    """测试LCR计算"""

    @pytest.mark.asyncio
    async def test_calculate_lcr_compliant(self):
        monitor = LiquidityRiskMonitor()

        lcr = await monitor.calculate_lcr(
            hqla_level1=1000000,
            hqla_level2a=200000,
            hqla_level2b=100000,
            cash_outflows=1000000,
            cash_inflows=200000,
        )

        assert lcr.total_hqla > 0
        assert lcr.lcr > 100  # 应该合规
        assert lcr.is_compliant is True

    @pytest.mark.asyncio
    async def test_calculate_lcr_non_compliant(self):
        monitor = LiquidityRiskMonitor()

        lcr = await monitor.calculate_lcr(
            hqla_level1=500000,
            hqla_level2a=100000,
            hqla_level2b=50000,
            cash_outflows=1000000,
            cash_inflows=100000,
        )

        assert lcr.is_compliant is False

    @pytest.mark.asyncio
    async def test_calculate_lcr_inflow_cap(self):
        """测试现金流入上限（75%流出）"""
        monitor = LiquidityRiskMonitor()

        lcr = await monitor.calculate_lcr(
            hqla_level1=1000000,
            hqla_level2a=0,
            hqla_level2b=0,
            cash_outflows=1000000,
            cash_inflows=900000,  # 超过75%
        )

        # 流入应被限制为750000
        assert lcr.net_cash_outflows == 250000


class TestStatistics:
    """测试统计信息"""

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        order_book_provider = SimulatedOrderBookProvider()
        volume_provider = SimulatedVolumeDataProvider()

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        monitor.add_symbol("AAPL")
        monitor.add_symbol("GOOGL")

        stats = monitor.get_statistics()
        assert stats["state"] == "stopped"
        assert stats["watched_symbols"] == 2


# ============ Factory Function Tests ============

class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_liquidity_monitor(self):
        monitor = create_liquidity_monitor()
        assert monitor is not None
        assert monitor.state == MonitorState.STOPPED

    def test_create_liquidity_monitor_with_config(self):
        config = LiquidityMonitorConfig(update_interval_sec=10.0)
        monitor = create_liquidity_monitor(config=config)
        assert monitor.config.update_interval_sec == 10.0

    def test_create_order_book(self):
        book = create_order_book(
            symbol="AAPL",
            bids=[(149.9, 100), (149.8, 200)],
            asks=[(150.1, 100), (150.2, 200)],
        )
        assert book.symbol == "AAPL"
        assert book.best_bid == 149.9
        assert book.best_ask == 150.1

    @pytest.mark.asyncio
    async def test_quick_liquidity_check(self):
        order_book_provider = SimulatedOrderBookProvider()
        order_book_provider.set_base_price("AAPL", 150.0)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 5000000)

        metrics = await quick_liquidity_check(
            "AAPL",
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        assert metrics is not None
        assert metrics.symbol == "AAPL"


# ============ Edge Cases Tests ============

class TestEdgeCases:
    """测试边界情况"""

    @pytest.mark.asyncio
    async def test_empty_order_book(self):
        order_book_provider = SimulatedOrderBookProvider()
        book = OrderBook(symbol="AAPL", timestamp=datetime.now())
        order_book_provider.set_order_book("AAPL", book)

        volume_provider = SimulatedVolumeDataProvider()

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        # 空订单簿应该返回None或处理得当
        metrics = await monitor.calculate_metrics("AAPL")
        # 由于mid_price为None，指标计算会有默认值
        assert metrics is not None

    @pytest.mark.asyncio
    async def test_zero_adv(self):
        order_book_provider = SimulatedOrderBookProvider()
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.95, 1000)],
            asks=[OrderBookLevel(150.05, 1000)],
        )
        order_book_provider.set_order_book("AAPL", book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 0)

        monitor = LiquidityRiskMonitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
        )

        metrics = await monitor.calculate_metrics("AAPL")
        assert metrics is not None

    def test_acknowledge_nonexistent_alert(self):
        monitor = LiquidityRiskMonitor()
        result = monitor.acknowledge_alert("NONEXISTENT", "admin")
        assert result is False

    @pytest.mark.asyncio
    async def test_stress_test_unknown_scenario(self):
        monitor = LiquidityRiskMonitor()

        with pytest.raises(ValueError):
            await monitor.run_stress_test("AAPL", "unknown_scenario")


# ============ Serialization Tests ============

class TestSerialization:
    """测试序列化"""

    def test_liquidity_metrics_serialization(self):
        metrics = LiquidityMetrics(
            symbol="AAPL",
            timestamp=datetime.now(),
            spread_bps=15.0,
            liquidity_score=75.0,
            liquidity_level=LiquidityLevel.GOOD,
        )
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["symbol"] == "AAPL"
        assert isinstance(d["timestamp"], str)

    def test_stress_result_serialization(self):
        scenario = LiquidityStressScenario(
            scenario_id="test",
            name="Test",
            description="Test scenario",
        )
        result = LiquidityStressResult(
            scenario=scenario,
            timestamp=datetime.now(),
            risk_level="moderate",
        )
        d = result.to_dict()

        assert "scenario" in d
        assert "risk_level" in d


# ============ Integration Tests ============

class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_monitoring_workflow(self):
        """测试完整监控流程"""
        # 设置提供者
        order_book_provider = SimulatedOrderBookProvider()
        volume_provider = SimulatedVolumeDataProvider()
        position_provider = SimulatedPositionDataProvider()

        # 设置数据
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(149.9 - i * 0.1, 1000 * (5 - i)) for i in range(5)],
            asks=[OrderBookLevel(150.1 + i * 0.1, 1000 * (5 - i)) for i in range(5)],
        )
        order_book_provider.set_order_book("AAPL", book)
        volume_provider.set_adv("AAPL", 5000000)
        volume_provider.set_current_volume("AAPL", 2500000)
        position_provider.set_position("AAPL", 10000, 150.0)

        # 创建监控器
        alerts_received = []
        metrics_received = []

        monitor = create_liquidity_monitor(
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
            position_provider=position_provider,
            on_alert=lambda a: alerts_received.append(a),
            on_metrics_update=lambda s, m: metrics_received.append((s, m)),
        )

        # 添加标的
        monitor.add_symbol("AAPL")

        # 计算指标
        metrics = await monitor.calculate_metrics("AAPL")
        assert metrics is not None
        assert metrics.liquidity_level in LiquidityLevel

        # 估算清算成本
        cost = await monitor.estimate_liquidation_cost("AAPL", 10000)
        assert cost.total_cost > 0

        # 创建退出计划
        plan = await monitor.create_exit_plan("AAPL", 10000)
        assert plan.status == "pending"

        # 运行压力测试
        stress_results = await monitor.run_all_stress_tests("AAPL")
        assert len(stress_results) == 4

        # 计算LCR
        lcr = await monitor.calculate_lcr(
            hqla_level1=1500000,
            hqla_level2a=300000,
            hqla_level2b=100000,
            cash_outflows=1000000,
            cash_inflows=200000,
        )
        assert lcr.is_compliant

        # 检查统计
        stats = monitor.get_statistics()
        assert stats["watched_symbols"] == 1

    @pytest.mark.asyncio
    async def test_alert_cooldown(self):
        """测试告警冷却机制"""
        config = LiquidityMonitorConfig(alert_cooldown_sec=2.0)

        order_book_provider = SimulatedOrderBookProvider()
        # 创建会触发告警的宽价差
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[OrderBookLevel(148.0, 100)],
            asks=[OrderBookLevel(152.0, 100)],  # 很宽的价差
        )
        order_book_provider.set_order_book("AAPL", book)

        volume_provider = SimulatedVolumeDataProvider()
        volume_provider.set_adv("AAPL", 1000000)

        alerts_received = []

        monitor = LiquidityRiskMonitor(
            config=config,
            order_book_provider=order_book_provider,
            volume_provider=volume_provider,
            on_alert=lambda a: alerts_received.append(a),
        )

        # 第一次检查
        metrics1 = await monitor.calculate_metrics("AAPL")
        await monitor._check_alerts(metrics1)
        first_count = len(alerts_received)

        # 立即第二次检查（应该被冷却阻止）
        metrics2 = await monitor.calculate_metrics("AAPL")
        await monitor._check_alerts(metrics2)
        second_count = len(alerts_received)

        # 由于冷却，不应该有新告警
        assert second_count == first_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
