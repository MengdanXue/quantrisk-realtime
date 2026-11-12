"""
Hedging Engine Tests - 对冲引擎测试

测试对冲引擎的所有功能模块。

作者: TraderX Team
创建: 2025-12-24
"""

import asyncio
import pytest
import sys
from datetime import datetime, timedelta
from typing import List
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from quantrisk_realtime.realtime.hedging_engine import (
    # Enums
    HedgeType,
    HedgeStrategy,
    HedgeInstrument,
    OrderType,
    OrderSide,
    HedgeStatus,
    AlertSeverity,
    EngineState,
    # Data Classes
    PositionData,
    PortfolioGreeks,
    HedgeOrder,
    HedgeResult,
    DeltaHedgeConfig,
    GammaHedgeConfig,
    VegaHedgeConfig,
    CorrelationHedgeConfig,
    TailRiskHedgeConfig,
    HedgeAlert,
    RebalanceSchedule,
    EngineConfig,
    # Interfaces
    PositionProvider,
    MarketDataProvider,
    OrderExecutor,
    # Simulated Providers
    SimulatedPositionProvider,
    SimulatedMarketDataProvider,
    SimulatedOrderExecutor,
    # Main Class
    HedgingEngine,
    # Factory Functions
    create_hedging_engine,
    create_position_data,
)


# ============ Fixtures ============


@pytest.fixture
def simulated_position_provider():
    """模拟持仓提供者"""
    provider = SimulatedPositionProvider()

    # 添加一些测试持仓
    provider.add_position(PositionData(
        position_id="POS001",
        symbol="AAPL",
        quantity=100,
        price=150.0,
        current_price=155.0,
        market_value=15500.0,
        delta=100.0,
        gamma=0.5,
        vega=50.0,
    ))

    provider.add_position(PositionData(
        position_id="POS002",
        symbol="GOOGL",
        quantity=-50,
        price=2800.0,
        current_price=2850.0,
        market_value=-142500.0,
        delta=-50.0,
        gamma=0.3,
        vega=30.0,
    ))

    return provider


@pytest.fixture
def simulated_market_data_provider():
    """模拟市场数据提供者"""
    provider = SimulatedMarketDataProvider()
    provider.set_price("AAPL", 155.0)
    provider.set_price("GOOGL", 2850.0)
    provider.set_volatility("AAPL", 0.25)
    provider.set_volatility("GOOGL", 0.30)
    return provider


@pytest.fixture
def simulated_order_executor():
    """模拟订单执行器"""
    return SimulatedOrderExecutor()


@pytest.fixture
def hedging_engine(
    simulated_position_provider,
    simulated_market_data_provider,
    simulated_order_executor
):
    """对冲引擎"""
    return create_hedging_engine(
        delta_hedge_enabled=True,
        position_provider=simulated_position_provider,
        market_data_provider=simulated_market_data_provider,
        order_executor=simulated_order_executor,
    )


# ============ Enum Tests ============


def test_hedge_type_enum():
    """测试对冲类型枚举"""
    assert HedgeType.DELTA.value == "delta"
    assert HedgeType.GAMMA.value == "gamma"
    assert HedgeType.VEGA.value == "vega"
    assert HedgeType.CORRELATION.value == "correlation"
    assert HedgeType.TAIL_RISK.value == "tail_risk"


def test_hedge_strategy_enum():
    """测试对冲策略枚举"""
    assert HedgeStrategy.STATIC.value == "static"
    assert HedgeStrategy.DYNAMIC.value == "dynamic"
    assert HedgeStrategy.BANDWIDTH.value == "bandwidth"
    assert HedgeStrategy.GAMMA_SCALPING.value == "gamma_scalping"


def test_hedge_instrument_enum():
    """测试对冲工具枚举"""
    assert HedgeInstrument.UNDERLYING.value == "underlying"
    assert HedgeInstrument.FUTURES.value == "futures"
    assert HedgeInstrument.OPTIONS.value == "options"


def test_order_side_enum():
    """测试订单方向枚举"""
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"


def test_hedge_status_enum():
    """测试对冲状态枚举"""
    assert HedgeStatus.PENDING.value == "pending"
    assert HedgeStatus.EXECUTING.value == "executing"
    assert HedgeStatus.COMPLETED.value == "completed"
    assert HedgeStatus.FAILED.value == "failed"


# ============ Data Class Tests ============


def test_position_data_creation():
    """测试持仓数据创建"""
    position = PositionData(
        position_id="TEST001",
        symbol="AAPL",
        quantity=100,
        price=150.0,
        current_price=155.0,
        market_value=15500.0,
        delta=100.0,
        gamma=0.5,
        vega=50.0,
    )

    assert position.position_id == "TEST001"
    assert position.symbol == "AAPL"
    assert position.quantity == 100
    assert position.delta == 100.0
    assert position.gamma == 0.5
    assert position.vega == 50.0


def test_position_data_to_dict():
    """测试持仓数据转字典"""
    position = create_position_data(
        position_id="TEST001",
        symbol="AAPL",
        quantity=100,
        price=150.0,
        current_price=155.0,
        delta=100.0,
    )

    data = position.to_dict()
    assert data["position_id"] == "TEST001"
    assert data["symbol"] == "AAPL"
    assert data["quantity"] == 100
    assert data["delta"] == 100.0


def test_portfolio_greeks_creation():
    """测试组合希腊字母创建"""
    greeks = PortfolioGreeks(
        total_delta=50.0,
        total_gamma=0.8,
        total_vega=80.0,
    )

    assert greeks.total_delta == 50.0
    assert greeks.total_gamma == 0.8
    assert greeks.total_vega == 80.0


def test_portfolio_greeks_to_dict():
    """测试组合希腊字母转字典"""
    greeks = PortfolioGreeks(
        total_delta=50.0,
        total_gamma=0.8,
        total_vega=80.0,
        by_underlying={"AAPL": {"delta": 50.0}},
    )

    data = greeks.to_dict()
    assert data["total_delta"] == 50.0
    assert "AAPL" in data["by_underlying"]


def test_hedge_order_creation():
    """测试对冲订单创建"""
    order = HedgeOrder(
        order_id="ORDER001",
        hedge_type=HedgeType.DELTA,
        symbol="AAPL",
        instrument=HedgeInstrument.UNDERLYING,
        side=OrderSide.SELL,
        quantity=100,
        current_delta=100.0,
        target_delta=0.0,
    )

    assert order.order_id == "ORDER001"
    assert order.hedge_type == HedgeType.DELTA
    assert order.side == OrderSide.SELL
    assert order.status == HedgeStatus.PENDING


def test_hedge_order_to_dict():
    """测试对冲订单转字典"""
    order = HedgeOrder(
        order_id="ORDER001",
        hedge_type=HedgeType.DELTA,
        symbol="AAPL",
        instrument=HedgeInstrument.UNDERLYING,
        side=OrderSide.SELL,
        quantity=100,
    )

    data = order.to_dict()
    assert data["order_id"] == "ORDER001"
    assert data["hedge_type"] == "delta"
    assert data["side"] == "sell"


def test_hedge_result_creation():
    """测试对冲结果创建"""
    result = HedgeResult(
        hedge_id="HEDGE001",
        hedge_type=HedgeType.DELTA,
        strategy=HedgeStrategy.BANDWIDTH,
        before_delta=100.0,
        after_delta=5.0,
        delta_reduction=95.0,
    )

    assert result.hedge_id == "HEDGE001"
    assert result.hedge_type == HedgeType.DELTA
    assert result.delta_reduction == 95.0


def test_hedge_result_to_dict():
    """测试对冲结果转字典"""
    result = HedgeResult(
        hedge_id="HEDGE001",
        hedge_type=HedgeType.DELTA,
        strategy=HedgeStrategy.BANDWIDTH,
        before_delta=100.0,
        after_delta=5.0,
        delta_reduction=95.0,
    )

    data = result.to_dict()
    assert data["hedge_id"] == "HEDGE001"
    assert data["delta_reduction"] == 95.0


# ============ Config Tests ============


def test_delta_hedge_config():
    """测试Delta对冲配置"""
    config = DeltaHedgeConfig(
        enabled=True,
        strategy=HedgeStrategy.BANDWIDTH,
        target_delta=0.0,
        bandwidth=0.05,
    )

    assert config.enabled is True
    assert config.strategy == HedgeStrategy.BANDWIDTH
    assert config.target_delta == 0.0
    assert config.bandwidth == 0.05


def test_gamma_hedge_config():
    """测试Gamma对冲配置"""
    config = GammaHedgeConfig(
        enabled=True,
        strategy=HedgeStrategy.GAMMA_SCALPING,
        scalping_enabled=True,
    )

    assert config.enabled is True
    assert config.scalping_enabled is True


def test_vega_hedge_config():
    """测试Vega对冲配置"""
    config = VegaHedgeConfig(
        enabled=True,
        strategy=HedgeStrategy.VEGA_NEUTRAL,
        use_options=True,
    )

    assert config.enabled is True
    assert config.use_options is True


def test_engine_config():
    """测试引擎配置"""
    config = EngineConfig()

    assert config.delta_hedge.enabled is True
    assert config.transaction_cost_bps == 5
    assert config.max_hedge_orders_per_hour == 20


# ============ Provider Tests ============


@pytest.mark.asyncio
async def test_simulated_position_provider(simulated_position_provider):
    """测试模拟持仓提供者"""
    positions = await simulated_position_provider.get_all_positions()
    assert len(positions) == 2

    position = await simulated_position_provider.get_position("POS001")
    assert position is not None
    assert position.symbol == "AAPL"


@pytest.mark.asyncio
async def test_simulated_market_data_provider(simulated_market_data_provider):
    """测试模拟市场数据提供者"""
    price = await simulated_market_data_provider.get_current_price("AAPL")
    assert price == 155.0

    vol = await simulated_market_data_provider.get_volatility("AAPL")
    assert vol == 0.25


@pytest.mark.asyncio
async def test_simulated_order_executor(simulated_order_executor):
    """测试模拟订单执行器"""
    order = HedgeOrder(
        order_id="ORDER001",
        hedge_type=HedgeType.DELTA,
        symbol="AAPL",
        instrument=HedgeInstrument.UNDERLYING,
        side=OrderSide.SELL,
        quantity=100,
    )

    success = await simulated_order_executor.execute_order(order)
    assert success is True
    assert order.status == HedgeStatus.COMPLETED
    assert order.executed_at is not None


# ============ Engine Lifecycle Tests ============


@pytest.mark.asyncio
async def test_engine_creation(hedging_engine):
    """测试引擎创建"""
    assert hedging_engine is not None
    assert hedging_engine.get_state() == EngineState.IDLE
    assert not hedging_engine.is_running()


@pytest.mark.asyncio
async def test_engine_start_stop(hedging_engine):
    """测试引擎启动停止"""
    await hedging_engine.start()
    assert hedging_engine.is_running()
    assert hedging_engine.get_state() == EngineState.RUNNING

    # 等待一下让引擎运行
    await asyncio.sleep(0.5)

    await hedging_engine.stop()
    assert not hedging_engine.is_running()
    assert hedging_engine.get_state() == EngineState.IDLE


@pytest.mark.asyncio
async def test_engine_double_start(hedging_engine):
    """测试引擎重复启动"""
    await hedging_engine.start()
    await hedging_engine.start()  # 应该不会报错
    assert hedging_engine.is_running()
    await hedging_engine.stop()


# ============ Greeks Calculation Tests ============


@pytest.mark.asyncio
async def test_update_greeks(hedging_engine):
    """测试更新希腊字母"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)  # 等待更新

    greeks = hedging_engine.get_current_greeks()
    assert greeks is not None
    # Delta = 100 * 100 + (-50) * (-50) = 10000 + 2500 = 12500
    assert greeks.total_delta == 12500.0
    # Gamma = 100 * 0.5 + (-50) * 0.3 = 50 - 15 = 35
    assert greeks.total_gamma == 35.0

    await hedging_engine.stop()


@pytest.mark.asyncio
async def test_greeks_by_underlying(hedging_engine):
    """测试按标的分组的希腊字母"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    greeks = hedging_engine.get_current_greeks()
    assert greeks is not None
    assert "AAPL" in greeks.by_underlying
    assert "GOOGL" in greeks.by_underlying

    await hedging_engine.stop()


# ============ Delta Hedging Tests ============


@pytest.mark.asyncio
async def test_execute_delta_hedge(hedging_engine):
    """测试执行Delta对冲"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    result = await hedging_engine.execute_delta_hedge(target_delta=0.0)

    assert result is not None
    assert result.hedge_type == HedgeType.DELTA
    assert result.before_delta == 12500.0  # 100*100 + (-50)*(-50)
    assert len(result.orders) > 0

    await hedging_engine.stop()


@pytest.mark.asyncio
async def test_delta_hedge_effectiveness(hedging_engine):
    """测试Delta对冲效果"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    result = await hedging_engine.execute_delta_hedge()

    assert result.effectiveness >= 0.0
    assert result.effectiveness <= 1.0
    assert result.delta_reduction >= 0.0

    await hedging_engine.stop()


@pytest.mark.asyncio
async def test_delta_hedge_cost_calculation(hedging_engine):
    """测试Delta对冲成本计算"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    result = await hedging_engine.execute_delta_hedge()

    assert result.total_cost >= 0.0
    assert result.transaction_costs >= 0.0

    await hedging_engine.stop()


# ============ Gamma Hedging Tests ============


@pytest.mark.asyncio
async def test_execute_gamma_hedge():
    """测试执行Gamma对冲"""
    engine = create_hedging_engine(
        delta_hedge_enabled=False,
        gamma_hedge_enabled=True,
    )

    # 添加持仓
    provider = SimulatedPositionProvider()
    provider.add_position(PositionData(
        position_id="POS001",
        symbol="AAPL",
        quantity=100,
        price=150.0,
        current_price=155.0,
        market_value=15500.0,
        gamma=0.5,
    ))

    engine._position_provider = provider

    await engine.start()
    await asyncio.sleep(0.5)

    result = await engine.execute_gamma_hedge()

    assert result is not None
    assert result.hedge_type == HedgeType.GAMMA

    await engine.stop()


# ============ Vega Hedging Tests ============


@pytest.mark.asyncio
async def test_execute_vega_hedge():
    """测试执行Vega对冲"""
    engine = create_hedging_engine(
        delta_hedge_enabled=False,
        vega_hedge_enabled=True,
    )

    # 添加持仓
    provider = SimulatedPositionProvider()
    provider.add_position(PositionData(
        position_id="POS001",
        symbol="AAPL",
        quantity=100,
        price=150.0,
        current_price=155.0,
        market_value=15500.0,
        vega=50.0,
    ))

    engine._position_provider = provider

    await engine.start()
    await asyncio.sleep(0.5)

    result = await engine.execute_vega_hedge()

    assert result is not None
    assert result.hedge_type == HedgeType.VEGA

    await engine.stop()


# ============ Cost Calculation Tests ============


def test_calculate_hedge_cost(hedging_engine):
    """测试计算对冲成本"""
    cost_info = hedging_engine.calculate_hedge_cost(
        symbol="AAPL",
        quantity=100,
        price=155.0,
    )

    assert "notional" in cost_info
    assert "transaction_cost" in cost_info
    assert "slippage" in cost_info
    assert "total_cost" in cost_info
    assert cost_info["notional"] == 15500.0


def test_hedge_cost_components(hedging_engine):
    """测试对冲成本组成"""
    cost_info = hedging_engine.calculate_hedge_cost(
        symbol="AAPL",
        quantity=100,
        price=100.0,
    )

    assert cost_info["transaction_cost"] > 0
    assert cost_info["slippage"] > 0
    assert cost_info["total_cost"] == cost_info["transaction_cost"] + cost_info["slippage"]


# ============ Alert Tests ============


def test_create_alert(hedging_engine):
    """测试创建警报"""
    alert = hedging_engine.create_alert(
        hedge_type=HedgeType.DELTA,
        severity=AlertSeverity.HIGH,
        message="Delta exceeds threshold",
        current_value=100.0,
        threshold=50.0,
    )

    assert alert is not None
    assert alert.hedge_type == HedgeType.DELTA
    assert alert.severity == AlertSeverity.HIGH


def test_get_alerts(hedging_engine):
    """测试获取警报"""
    hedging_engine.create_alert(
        hedge_type=HedgeType.DELTA,
        severity=AlertSeverity.HIGH,
        message="Test alert",
        current_value=100.0,
        threshold=50.0,
    )

    alerts = hedging_engine.get_alerts()
    assert len(alerts) > 0


def test_alert_callback(hedging_engine):
    """测试警报回调"""
    received_alerts = []

    def callback(alert: HedgeAlert):
        received_alerts.append(alert)

    hedging_engine.on_alert(callback)

    hedging_engine.create_alert(
        hedge_type=HedgeType.DELTA,
        severity=AlertSeverity.MEDIUM,
        message="Test",
        current_value=50.0,
        threshold=40.0,
    )

    assert len(received_alerts) == 1


# ============ Rebalance Schedule Tests ============


@pytest.mark.asyncio
async def test_rebalance_schedules(hedging_engine):
    """测试再平衡计划"""
    await hedging_engine.start()
    await asyncio.sleep(0.2)

    schedules = hedging_engine.get_rebalance_schedules()
    assert len(schedules) > 0

    await hedging_engine.stop()


@pytest.mark.asyncio
async def test_scheduled_rebalance_execution(hedging_engine):
    """测试计划的再平衡执行"""
    # 设置短间隔
    hedging_engine._config.delta_hedge.rehedge_interval_seconds = 1

    await hedging_engine.start()

    # 重新初始化schedules
    hedging_engine._initialize_schedules()

    await asyncio.sleep(1.5)

    results = hedging_engine.get_hedge_results()
    # 应该至少执行了一次对冲

    await hedging_engine.stop()


# ============ Callback Tests ============


@pytest.mark.asyncio
async def test_hedge_result_callback(hedging_engine):
    """测试对冲结果回调"""
    received_results = []

    def callback(result: HedgeResult):
        received_results.append(result)

    hedging_engine.on_hedge_result(callback)

    await hedging_engine.start()
    await asyncio.sleep(0.5)

    await hedging_engine.execute_delta_hedge()

    assert len(received_results) > 0

    await hedging_engine.stop()


# ============ Statistics Tests ============


@pytest.mark.asyncio
async def test_get_statistics(hedging_engine):
    """测试获取统计信息"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    stats = hedging_engine.get_statistics()

    assert "state" in stats
    assert "running" in stats
    assert "total_hedges" in stats
    assert "current_greeks" in stats

    await hedging_engine.stop()


@pytest.mark.asyncio
async def test_hedge_count_by_type(hedging_engine):
    """测试按类型统计对冲次数"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    await hedging_engine.execute_delta_hedge()

    stats = hedging_engine.get_statistics()
    assert "hedge_count_by_type" in stats

    await hedging_engine.stop()


# ============ Order Management Tests ============


@pytest.mark.asyncio
async def test_get_hedge_orders(hedging_engine):
    """测试获取对冲订单"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    result = await hedging_engine.execute_delta_hedge()

    orders = hedging_engine.get_hedge_orders()
    assert len(orders) > 0

    await hedging_engine.stop()


@pytest.mark.asyncio
async def test_get_hedge_result_by_id(hedging_engine):
    """测试通过ID获取对冲结果"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    result = await hedging_engine.execute_delta_hedge()
    hedge_id = result.hedge_id

    retrieved_result = hedging_engine.get_hedge_result(hedge_id)
    assert retrieved_result is not None
    assert retrieved_result.hedge_id == hedge_id

    await hedging_engine.stop()


# ============ Factory Function Tests ============


def test_create_hedging_engine():
    """测试创建对冲引擎工厂函数"""
    engine = create_hedging_engine(
        delta_hedge_enabled=True,
        gamma_hedge_enabled=False,
    )

    assert engine is not None
    assert engine._config.delta_hedge.enabled is True
    assert engine._config.gamma_hedge.enabled is False


def test_create_position_data_factory():
    """测试创建持仓数据工厂函数"""
    position = create_position_data(
        position_id="TEST001",
        symbol="AAPL",
        quantity=100,
        price=150.0,
        current_price=155.0,
        delta=100.0,
    )

    assert position.position_id == "TEST001"
    assert position.delta == 100.0


# ============ Integration Tests ============


@pytest.mark.asyncio
async def test_full_hedging_workflow(hedging_engine):
    """测试完整对冲流程"""
    # 启动引擎
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    # 检查初始希腊字母
    greeks_before = hedging_engine.get_current_greeks()
    assert greeks_before is not None

    # 执行对冲
    result = await hedging_engine.execute_delta_hedge()

    # 验证结果
    assert result.hedge_type == HedgeType.DELTA
    assert result.total_orders > 0

    # 检查订单
    orders = result.orders
    for order in orders:
        assert order.status == HedgeStatus.COMPLETED

    # 检查统计
    stats = hedging_engine.get_statistics()
    assert stats["total_hedges"] >= 1

    # 停止引擎
    await hedging_engine.stop()


@pytest.mark.asyncio
async def test_multiple_hedge_executions(hedging_engine):
    """测试多次对冲执行"""
    await hedging_engine.start()
    await asyncio.sleep(0.5)

    # 执行多次对冲
    for _ in range(3):
        await hedging_engine.execute_delta_hedge()
        await asyncio.sleep(0.2)

    # 检查历史
    results = hedging_engine.get_hedge_results()
    assert len(results) >= 3

    await hedging_engine.stop()


# ============ Edge Case Tests ============


@pytest.mark.asyncio
async def test_hedge_with_no_positions():
    """测试没有持仓时的对冲"""
    engine = create_hedging_engine(
        delta_hedge_enabled=True,
        position_provider=SimulatedPositionProvider(),
    )

    await engine.start()
    await asyncio.sleep(0.5)

    greeks = engine.get_current_greeks()
    # 应该返回None或空的greeks

    await engine.stop()


@pytest.mark.asyncio
async def test_hedge_with_zero_delta():
    """测试Delta为0时的对冲"""
    provider = SimulatedPositionProvider()
    provider.add_position(PositionData(
        position_id="POS001",
        symbol="AAPL",
        quantity=100,
        price=150.0,
        current_price=155.0,
        market_value=15500.0,
        delta=0.0,  # Delta为0
    ))

    engine = create_hedging_engine(
        delta_hedge_enabled=True,
        position_provider=provider,
    )

    await engine.start()
    await asyncio.sleep(0.5)

    result = await engine.execute_delta_hedge()
    # 应该没有创建订单或订单很少

    await engine.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
