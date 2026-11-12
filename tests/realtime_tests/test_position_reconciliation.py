"""
持仓对账引擎测试
Position Reconciliation Engine Tests
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from quantrisk_realtime.realtime.position_reconciliation import (
    # 枚举
    ReconciliationStatus,
    DiscrepancyType,
    ResolutionAction,
    ReconciliationSource,
    ReconciliationFrequency,
    ReconciliationState,
    # 数据类
    PositionSnapshot,
    Discrepancy,
    ReconciliationResult,
    AuditLogEntry,
    ReconciliationConfig,
    # 接口实现
    SimulatedBrokerProvider,
    SimulatedSystemProvider,
    SimulatedOrderHistoryProvider,
    # 核心引擎
    PositionReconciliationEngine,
    # 工厂函数
    create_reconciliation_engine,
    create_position_snapshot,
    quick_reconcile,
)


# ==================== 枚举测试 ====================

class TestReconciliationStatus:
    """对账状态枚举测试"""

    def test_status_values(self):
        """测试状态值"""
        assert ReconciliationStatus.PENDING.value == "pending"
        assert ReconciliationStatus.IN_PROGRESS.value == "in_progress"
        assert ReconciliationStatus.MATCHED.value == "matched"
        assert ReconciliationStatus.DISCREPANCY.value == "discrepancy"
        assert ReconciliationStatus.RESOLVED.value == "resolved"
        assert ReconciliationStatus.FAILED.value == "failed"


class TestDiscrepancyType:
    """差异类型枚举测试"""

    def test_type_values(self):
        """测试类型值"""
        assert DiscrepancyType.QUANTITY_MISMATCH.value == "quantity_mismatch"
        assert DiscrepancyType.PRICE_MISMATCH.value == "price_mismatch"
        assert DiscrepancyType.MISSING_IN_SYSTEM.value == "missing_in_system"
        assert DiscrepancyType.MISSING_AT_BROKER.value == "missing_at_broker"
        assert DiscrepancyType.SIDE_MISMATCH.value == "side_mismatch"


class TestResolutionAction:
    """修复动作枚举测试"""

    def test_action_values(self):
        """测试动作值"""
        assert ResolutionAction.NO_ACTION.value == "no_action"
        assert ResolutionAction.ADJUST_SYSTEM.value == "adjust_system"
        assert ResolutionAction.CREATE_POSITION.value == "create_position"
        assert ResolutionAction.MANUAL_REVIEW.value == "manual_review"


class TestReconciliationFrequency:
    """对账频率枚举测试"""

    def test_frequency_values(self):
        """测试频率值"""
        assert ReconciliationFrequency.REALTIME.value == "realtime"
        assert ReconciliationFrequency.MINUTE.value == "minute"
        assert ReconciliationFrequency.HOURLY.value == "hourly"
        assert ReconciliationFrequency.DAILY.value == "daily"
        assert ReconciliationFrequency.ON_DEMAND.value == "on_demand"


# ==================== 数据类测试 ====================

class TestPositionSnapshot:
    """持仓快照测试"""

    def test_create_snapshot(self):
        """测试创建快照"""
        snapshot = PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150.50"),
            market_value=Decimal("15500"),
            unrealized_pnl=Decimal("450"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        assert snapshot.symbol == "AAPL"
        assert snapshot.account_id == "ACC001"
        assert snapshot.quantity == Decimal("100")
        assert snapshot.avg_cost == Decimal("150.50")
        assert snapshot.side == "long"

    def test_snapshot_to_dict(self):
        """测试快照转字典"""
        snapshot = PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150.50"),
            market_value=Decimal("15500"),
            unrealized_pnl=Decimal("450"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        data = snapshot.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["quantity"] == 100.0
        assert data["avg_cost"] == 150.50
        assert "timestamp" in data


class TestDiscrepancy:
    """差异记录测试"""

    def test_create_discrepancy(self):
        """测试创建差异"""
        disc = Discrepancy(
            discrepancy_id="D001",
            symbol="AAPL",
            account_id="ACC001",
            discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
            system_value=100,
            broker_value=105,
            order_derived_value=100,
            difference=Decimal("5"),
            severity="medium",
            description="Quantity mismatch",
        )
        assert disc.symbol == "AAPL"
        assert disc.discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH
        assert disc.difference == Decimal("5")
        assert disc.resolved is False

    def test_discrepancy_to_dict(self):
        """测试差异转字典"""
        disc = Discrepancy(
            discrepancy_id="D001",
            symbol="AAPL",
            account_id="ACC001",
            discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
            system_value=100,
            broker_value=105,
            order_derived_value=100,
            difference=Decimal("5"),
            severity="medium",
            description="Quantity mismatch",
        )
        data = disc.to_dict()
        assert data["discrepancy_type"] == "quantity_mismatch"
        assert data["difference"] == 5.0
        assert data["resolved"] is False


class TestReconciliationConfig:
    """对账配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = ReconciliationConfig()
        assert config.frequency == ReconciliationFrequency.HOURLY
        assert config.enable_auto_resolve is False
        assert config.notify_on_discrepancy is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = ReconciliationConfig(
            frequency=ReconciliationFrequency.REALTIME,
            enable_auto_resolve=True,
            quantity_tolerance=Decimal("0.001"),
        )
        assert config.frequency == ReconciliationFrequency.REALTIME
        assert config.enable_auto_resolve is True
        assert config.quantity_tolerance == Decimal("0.001")


class TestAuditLogEntry:
    """审计日志测试"""

    def test_create_entry(self):
        """测试创建条目"""
        entry = AuditLogEntry(
            entry_id="E001",
            timestamp=datetime.now(),
            action="reconciliation_started",
            account_id="ACC001",
            symbol=None,
            details={"reconciliation_id": "R001"},
        )
        assert entry.action == "reconciliation_started"
        assert entry.account_id == "ACC001"

    def test_entry_to_dict(self):
        """测试条目转字典"""
        entry = AuditLogEntry(
            entry_id="E001",
            timestamp=datetime.now(),
            action="reconciliation_started",
            account_id="ACC001",
            symbol=None,
            details={"reconciliation_id": "R001"},
        )
        data = entry.to_dict()
        assert data["action"] == "reconciliation_started"
        assert "timestamp" in data


# ==================== 提供者测试 ====================

class TestSimulatedBrokerProvider:
    """模拟经纪商提供者测试"""

    @pytest.mark.asyncio
    async def test_get_positions_empty(self):
        """测试获取空持仓"""
        provider = SimulatedBrokerProvider()
        positions = await provider.get_positions("ACC001")
        assert positions == []

    @pytest.mark.asyncio
    async def test_set_and_get_position(self):
        """测试设置和获取持仓"""
        provider = SimulatedBrokerProvider()
        snapshot = PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            source=ReconciliationSource.BROKER,
        )
        provider.set_position(snapshot)

        positions = await provider.get_positions("ACC001")
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_single_position(self):
        """测试获取单个持仓"""
        provider = SimulatedBrokerProvider()
        snapshot = PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        provider.set_position(snapshot)

        pos = await provider.get_position("ACC001", "AAPL")
        assert pos is not None
        assert pos.quantity == Decimal("100")

        pos2 = await provider.get_position("ACC001", "MSFT")
        assert pos2 is None


class TestSimulatedSystemProvider:
    """模拟系统提供者测试"""

    @pytest.mark.asyncio
    async def test_get_positions_empty(self):
        """测试获取空持仓"""
        provider = SimulatedSystemProvider()
        positions = await provider.get_positions("ACC001")
        assert positions == []

    @pytest.mark.asyncio
    async def test_update_position(self):
        """测试更新持仓"""
        provider = SimulatedSystemProvider()
        snapshot = PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        result = await provider.update_position(snapshot)
        assert result is True

        positions = await provider.get_positions("ACC001")
        assert len(positions) == 1

    @pytest.mark.asyncio
    async def test_create_position(self):
        """测试创建持仓"""
        provider = SimulatedSystemProvider()
        snapshot = PositionSnapshot(
            symbol="MSFT",
            account_id="ACC001",
            quantity=Decimal("50"),
            avg_cost=Decimal("300"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        result = await provider.create_position(snapshot)
        assert result is True


class TestSimulatedOrderHistoryProvider:
    """模拟订单历史提供者测试"""

    @pytest.mark.asyncio
    async def test_get_filled_orders_empty(self):
        """测试获取空订单"""
        provider = SimulatedOrderHistoryProvider()
        orders = await provider.get_filled_orders(
            "ACC001",
            datetime.now() - timedelta(days=1),
            datetime.now(),
        )
        assert orders == []

    @pytest.mark.asyncio
    async def test_add_and_get_orders(self):
        """测试添加和获取订单"""
        provider = SimulatedOrderHistoryProvider()
        order = {
            "order_id": "O001",
            "account_id": "ACC001",
            "symbol": "AAPL",
            "side": "buy",
            "filled_quantity": 100,
            "avg_price": 150.0,
            "status": "filled",
            "fill_time": datetime.now(),
        }
        provider.add_order(order)

        orders = await provider.get_filled_orders(
            "ACC001",
            datetime.now() - timedelta(days=1),
            datetime.now() + timedelta(hours=1),
        )
        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"


# ==================== 核心引擎测试 ====================

class TestPositionReconciliationEngine:
    """对账引擎测试"""

    def setup_method(self):
        """测试前准备"""
        self.broker = SimulatedBrokerProvider()
        self.system = SimulatedSystemProvider()
        self.engine = PositionReconciliationEngine(
            broker_provider=self.broker,
            system_provider=self.system,
        )

    @pytest.mark.asyncio
    async def test_reconcile_empty(self):
        """测试空持仓对账"""
        result = await self.engine.reconcile("ACC001")
        assert result.status == ReconciliationStatus.MATCHED
        assert result.total_positions == 0
        assert len(result.discrepancies) == 0

    @pytest.mark.asyncio
    async def test_reconcile_matched(self):
        """测试匹配的持仓对账"""
        # 设置相同的持仓
        system_pos = PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        broker_pos = PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        self.system.set_position(system_pos)
        self.broker.set_position(broker_pos)

        result = await self.engine.reconcile("ACC001")
        assert result.status == ReconciliationStatus.MATCHED
        assert result.total_positions == 1
        assert result.matched_positions == 1
        assert len(result.discrepancies) == 0

    @pytest.mark.asyncio
    async def test_reconcile_quantity_mismatch(self):
        """测试数量不匹配"""
        system_pos = PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        broker_pos = PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("105"),  # 差异
            avg_cost=Decimal("150"),
            market_value=Decimal("15750"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        self.system.set_position(system_pos)
        self.broker.set_position(broker_pos)

        result = await self.engine.reconcile("ACC001")
        assert result.status == ReconciliationStatus.DISCREPANCY
        assert len(result.discrepancies) == 1
        assert result.discrepancies[0].discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH

    @pytest.mark.asyncio
    async def test_reconcile_missing_in_system(self):
        """测试系统缺失持仓"""
        broker_pos = PositionSnapshot(
            symbol="MSFT",
            account_id="ACC001",
            quantity=Decimal("50"),
            avg_cost=Decimal("300"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        self.broker.set_position(broker_pos)

        result = await self.engine.reconcile("ACC001")
        assert result.status == ReconciliationStatus.DISCREPANCY
        assert len(result.discrepancies) == 1
        assert result.discrepancies[0].discrepancy_type == DiscrepancyType.MISSING_IN_SYSTEM

    @pytest.mark.asyncio
    async def test_reconcile_missing_at_broker(self):
        """测试经纪商缺失持仓"""
        system_pos = PositionSnapshot(
            symbol="GOOGL",
            account_id="ACC001",
            quantity=Decimal("25"),
            avg_cost=Decimal("2800"),
            market_value=Decimal("70000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        self.system.set_position(system_pos)

        result = await self.engine.reconcile("ACC001")
        assert result.status == ReconciliationStatus.DISCREPANCY
        assert len(result.discrepancies) == 1
        assert result.discrepancies[0].discrepancy_type == DiscrepancyType.MISSING_AT_BROKER

    @pytest.mark.asyncio
    async def test_reconcile_side_mismatch(self):
        """测试方向不匹配"""
        system_pos = PositionSnapshot(
            symbol="TSLA",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("200"),
            market_value=Decimal("20000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        broker_pos = PositionSnapshot(
            symbol="TSLA",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("200"),
            market_value=Decimal("20000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="short",  # 方向不同
        )

        self.system.set_position(system_pos)
        self.broker.set_position(broker_pos)

        result = await self.engine.reconcile("ACC001")
        assert result.status == ReconciliationStatus.DISCREPANCY
        assert any(d.discrepancy_type == DiscrepancyType.SIDE_MISMATCH for d in result.discrepancies)

    @pytest.mark.asyncio
    async def test_reconcile_multiple_discrepancies(self):
        """测试多个差异"""
        # 系统有 AAPL 和 MSFT
        self.system.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))
        self.system.set_position(PositionSnapshot(
            symbol="MSFT",
            account_id="ACC001",
            quantity=Decimal("50"),
            avg_cost=Decimal("300"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))

        # 经纪商有 AAPL（数量不同）和 GOOGL（系统没有）
        self.broker.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("110"),
            avg_cost=Decimal("150"),
            market_value=Decimal("16500"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))
        self.broker.set_position(PositionSnapshot(
            symbol="GOOGL",
            account_id="ACC001",
            quantity=Decimal("25"),
            avg_cost=Decimal("2800"),
            market_value=Decimal("70000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))

        result = await self.engine.reconcile("ACC001")
        assert result.status == ReconciliationStatus.DISCREPANCY
        assert len(result.discrepancies) >= 2

    @pytest.mark.asyncio
    async def test_resolve_discrepancy(self):
        """测试修复差异"""
        disc = Discrepancy(
            discrepancy_id="D001",
            symbol="AAPL",
            account_id="ACC001",
            discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
            system_value=100,
            broker_value=105,
            order_derived_value=None,
            difference=Decimal("5"),
            severity="medium",
            description="Quantity mismatch",
        )

        result = await self.engine.resolve_discrepancy(
            disc,
            ResolutionAction.ADJUST_SYSTEM,
            "Manual adjustment",
        )
        assert result is True
        assert disc.resolved is True
        assert disc.resolution_action == ResolutionAction.ADJUST_SYSTEM

    @pytest.mark.asyncio
    async def test_callbacks(self):
        """测试回调"""
        discrepancies_received = []
        results_received = []

        def on_disc(d):
            discrepancies_received.append(d)

        def on_complete(r):
            results_received.append(r)

        self.engine.on_discrepancy(on_disc)
        self.engine.on_reconciliation_complete(on_complete)

        # 设置不匹配的持仓
        self.system.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))
        self.broker.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("105"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15750"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))

        await self.engine.reconcile("ACC001")

        assert len(discrepancies_received) == 1
        assert len(results_received) == 1

    def test_get_statistics(self):
        """测试获取统计"""
        stats = self.engine.get_statistics()
        assert "total_reconciliations" in stats
        assert "state" in stats
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_get_history(self):
        """测试获取历史"""
        await self.engine.reconcile("ACC001")
        await self.engine.reconcile("ACC001")

        history = self.engine.get_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_unresolved_discrepancies(self):
        """测试获取未解决差异"""
        self.broker.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))

        await self.engine.reconcile("ACC001")
        unresolved = self.engine.get_unresolved_discrepancies()
        assert len(unresolved) == 1

    @pytest.mark.asyncio
    async def test_audit_log(self):
        """测试审计日志"""
        await self.engine.reconcile("ACC001")
        logs = self.engine.get_audit_log()
        assert len(logs) >= 2  # started + completed


class TestEngineStartStop:
    """引擎启停测试"""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启停"""
        engine = create_reconciliation_engine(frequency="on_demand")
        assert engine.state == ReconciliationState.IDLE

        await engine.start(["ACC001"])
        assert engine.state == ReconciliationState.RUNNING

        await engine.stop()
        assert engine.state == ReconciliationState.IDLE


class TestAutoResolve:
    """自动修复测试"""

    @pytest.mark.asyncio
    async def test_auto_resolve_small_difference(self):
        """测试小差异自动修复"""
        broker = SimulatedBrokerProvider()
        system = SimulatedSystemProvider()

        config = ReconciliationConfig(
            enable_auto_resolve=True,
            auto_resolve_threshold=Decimal("10"),
            quantity_tolerance=Decimal("0.001"),
        )

        engine = PositionReconciliationEngine(
            broker_provider=broker,
            system_provider=system,
            config=config,
        )

        # 设置小差异
        system.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))
        broker.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100.005"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000.75"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))

        result = await engine.reconcile("ACC001")
        # 差异应该被自动修复
        assert engine._stats["auto_resolved"] >= 0


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_reconciliation_engine(self):
        """测试创建对账引擎"""
        engine = create_reconciliation_engine()
        assert engine is not None
        assert engine.state == ReconciliationState.IDLE

    def test_create_engine_with_frequency(self):
        """测试创建带频率的引擎"""
        engine = create_reconciliation_engine(frequency="realtime")
        assert engine.config.frequency == ReconciliationFrequency.REALTIME

    def test_create_engine_with_auto_resolve(self):
        """测试创建启用自动修复的引擎"""
        engine = create_reconciliation_engine(auto_resolve=True)
        assert engine.config.enable_auto_resolve is True

    def test_create_position_snapshot(self):
        """测试创建持仓快照"""
        snapshot = create_position_snapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=100,
            avg_cost=150.0,
            side="long",
        )
        assert snapshot.symbol == "AAPL"
        assert snapshot.quantity == Decimal("100")
        assert snapshot.avg_cost == Decimal("150.0")

    def test_create_position_snapshot_with_source(self):
        """测试创建带来源的持仓快照"""
        snapshot = create_position_snapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=100,
            avg_cost=150.0,
            source="broker",
        )
        assert snapshot.source == ReconciliationSource.BROKER


class TestQuickReconcile:
    """快速对账测试"""

    @pytest.mark.asyncio
    async def test_quick_reconcile_matched(self):
        """测试快速对账匹配"""
        system_positions = [
            {"symbol": "AAPL", "quantity": 100, "avg_cost": 150.0, "side": "long"},
        ]
        broker_positions = [
            {"symbol": "AAPL", "quantity": 100, "avg_cost": 150.0, "side": "long"},
        ]

        result = await quick_reconcile(system_positions, broker_positions)
        assert result["status"] == "matched"
        assert len(result["discrepancies"]) == 0

    @pytest.mark.asyncio
    async def test_quick_reconcile_discrepancy(self):
        """测试快速对账差异"""
        system_positions = [
            {"symbol": "AAPL", "quantity": 100, "avg_cost": 150.0, "side": "long"},
        ]
        broker_positions = [
            {"symbol": "AAPL", "quantity": 110, "avg_cost": 150.0, "side": "long"},
        ]

        result = await quick_reconcile(system_positions, broker_positions)
        assert result["status"] == "discrepancy"
        assert len(result["discrepancies"]) > 0


# ==================== 边界情况测试 ====================

class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_excluded_symbols(self):
        """测试排除的标的"""
        broker = SimulatedBrokerProvider()
        system = SimulatedSystemProvider()

        config = ReconciliationConfig(
            excluded_symbols={"AAPL"},
        )

        engine = PositionReconciliationEngine(
            broker_provider=broker,
            system_provider=system,
            config=config,
        )

        # 设置不匹配的 AAPL（应被排除）
        system.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))
        broker.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("200"),  # 大差异
            avg_cost=Decimal("150"),
            market_value=Decimal("30000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))

        result = await engine.reconcile("ACC001")
        # AAPL 被排除，不应有差异
        assert len(result.discrepancies) == 0

    @pytest.mark.asyncio
    async def test_price_tolerance(self):
        """测试价格容差"""
        broker = SimulatedBrokerProvider()
        system = SimulatedSystemProvider()

        config = ReconciliationConfig(
            price_tolerance=Decimal("1.0"),  # 1元容差
        )

        engine = PositionReconciliationEngine(
            broker_provider=broker,
            system_provider=system,
            config=config,
        )

        # 设置价格差异在容差内
        system.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150.00"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))
        broker.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150.50"),  # 差异0.5，在容差内
            market_value=Decimal("15050"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))

        result = await engine.reconcile("ACC001")
        # 价格差异在容差内，不应有差异
        assert not any(d.discrepancy_type == DiscrepancyType.PRICE_MISMATCH for d in result.discrepancies)


# ==================== 序列化测试 ====================

class TestSerialization:
    """序列化测试"""

    def test_position_snapshot_serialization(self):
        """测试持仓快照序列化"""
        snapshot = create_position_snapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=100,
            avg_cost=150.0,
        )
        data = snapshot.to_dict()
        assert isinstance(data, dict)
        assert data["symbol"] == "AAPL"

    def test_discrepancy_serialization(self):
        """测试差异序列化"""
        disc = Discrepancy(
            discrepancy_id="D001",
            symbol="AAPL",
            account_id="ACC001",
            discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
            system_value=100,
            broker_value=105,
            order_derived_value=None,
            difference=Decimal("5"),
            severity="medium",
            description="Test",
        )
        data = disc.to_dict()
        assert isinstance(data, dict)
        assert data["difference"] == 5.0

    @pytest.mark.asyncio
    async def test_result_serialization(self):
        """测试结果序列化"""
        engine = create_reconciliation_engine()
        result = await engine.reconcile("ACC001")
        data = result.to_dict()
        assert isinstance(data, dict)
        assert "reconciliation_id" in data
        assert "status" in data


# ==================== 严重程度测试 ====================

class TestSeverityCalculation:
    """严重程度计算测试"""

    @pytest.mark.asyncio
    async def test_critical_severity_missing(self):
        """测试缺失为严重"""
        broker = SimulatedBrokerProvider()
        system = SimulatedSystemProvider()

        engine = PositionReconciliationEngine(
            broker_provider=broker,
            system_provider=system,
        )

        broker.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))

        result = await engine.reconcile("ACC001")
        assert len(result.discrepancies) == 1
        assert result.discrepancies[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_critical_severity_side_mismatch(self):
        """测试方向不匹配为严重"""
        broker = SimulatedBrokerProvider()
        system = SimulatedSystemProvider()

        engine = PositionReconciliationEngine(
            broker_provider=broker,
            system_provider=system,
        )

        system.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        ))
        broker.set_position(PositionSnapshot(
            symbol="AAPL",
            account_id="ACC001",
            quantity=Decimal("100"),
            avg_cost=Decimal("150"),
            market_value=Decimal("15000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="short",
        ))

        result = await engine.reconcile("ACC001")
        side_mismatches = [d for d in result.discrepancies if d.discrepancy_type == DiscrepancyType.SIDE_MISMATCH]
        assert len(side_mismatches) == 1
        assert side_mismatches[0].severity == "critical"
