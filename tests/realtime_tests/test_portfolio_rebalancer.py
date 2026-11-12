"""
组合再平衡器测试
Testing Portfolio Rebalancer

作者: TraderX 量化团队
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

# 确保src目录在路径中
import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from quantrisk_realtime.realtime.portfolio_rebalancer import (
    # 枚举
    RebalanceStrategy,
    RebalanceFrequency,
    DriftType,
    TaxLotMethod,
    OrderType,
    OrderSide,
    OrderStatus,
    RebalanceState,
    AlertType,
    TaxEventType,
    # 数据类
    Position,
    TaxLot,
    PositionDrift,
    RebalanceOrder,
    RebalancePlan,
    WhatIfScenario,
    TaxHarvestingOpportunity,
    RebalanceAlert,
    RebalancerConfig,
    # 接口和模拟实现
    PortfolioProvider,
    PriceProvider,
    TargetProvider,
    OrderExecutor,
    SimulatedPortfolioProvider,
    SimulatedPriceProvider,
    SimulatedTargetProvider,
    SimulatedOrderExecutor,
    # 核心引擎
    PortfolioRebalancer,
    # 工厂函数
    create_portfolio_rebalancer,
    create_position,
    create_tax_lot,
    quick_rebalance_analysis,
)


# ==================== 测试夹具 ====================

@pytest.fixture
def rebalancer_config():
    """测试配置"""
    return RebalancerConfig(
        absolute_drift_threshold=0.02,
        relative_drift_threshold=0.10,
        min_trade_value=100.0,
        max_turnover_rate=0.20,
        commission_rate=0.001,
        slippage_bps=5.0,
        short_term_tax_rate=0.37,
        long_term_tax_rate=0.20,
        calendar_frequency=RebalanceFrequency.QUARTERLY,
    )


@pytest.fixture
def portfolio_provider():
    """模拟组合提供者"""
    return SimulatedPortfolioProvider()


@pytest.fixture
def price_provider():
    """模拟价格提供者"""
    provider = SimulatedPriceProvider()
    # 设置默认价格
    provider.set_price("AAPL", 150.0)
    provider.set_price("GOOGL", 2800.0)
    provider.set_price("MSFT", 300.0)
    provider.set_price("AMZN", 3200.0)
    return provider


@pytest.fixture
def target_provider():
    """模拟目标提供者"""
    provider = SimulatedTargetProvider()
    provider.set_targets("portfolio1", {
        "AAPL": 0.25,
        "GOOGL": 0.25,
        "MSFT": 0.25,
        "AMZN": 0.25,
    })
    return provider


@pytest.fixture
def order_executor():
    """模拟订单执行器"""
    return SimulatedOrderExecutor()


@pytest.fixture
def rebalancer(rebalancer_config, portfolio_provider, price_provider, target_provider, order_executor):
    """再平衡器"""
    return PortfolioRebalancer(
        config=rebalancer_config,
        portfolio_provider=portfolio_provider,
        price_provider=price_provider,
        target_provider=target_provider,
        order_executor=order_executor,
    )


@pytest.fixture
def sample_positions():
    """示例持仓"""
    return [
        create_position("AAPL", 200, 30000, 0.30, 0.25),
        create_position("GOOGL", 10, 28000, 0.28, 0.25),
        create_position("MSFT", 80, 24000, 0.24, 0.25),
        create_position("AMZN", 5, 16000, 0.16, 0.25),
    ]


# ==================== 枚举测试 (10个) ====================

def test_rebalance_strategy_enum():
    """测试再平衡策略枚举"""
    assert RebalanceStrategy.CALENDAR.value == "calendar"
    assert RebalanceStrategy.THRESHOLD.value == "threshold"
    assert RebalanceStrategy.TACTICAL.value == "tactical"
    assert RebalanceStrategy.TAX_AWARE.value == "tax_aware"
    assert len(list(RebalanceStrategy)) >= 4


def test_rebalance_frequency_enum():
    """测试再平衡频率枚举"""
    assert RebalanceFrequency.DAILY.value == "daily"
    assert RebalanceFrequency.WEEKLY.value == "weekly"
    assert RebalanceFrequency.MONTHLY.value == "monthly"
    assert RebalanceFrequency.QUARTERLY.value == "quarterly"
    assert len(list(RebalanceFrequency)) >= 4


def test_drift_type_enum():
    """测试漂移类型枚举"""
    assert DriftType.ABSOLUTE.value == "absolute"
    assert DriftType.RELATIVE.value == "relative"
    assert DriftType.TRACKING_ERROR.value == "tracking_error"
    assert DriftType.RISK_BASED.value == "risk_based"


def test_tax_lot_method_enum():
    """测试税务批次方法枚举"""
    assert TaxLotMethod.FIFO.value == "fifo"
    assert TaxLotMethod.LIFO.value == "lifo"
    assert TaxLotMethod.HIFO.value == "hifo"
    assert TaxLotMethod.MIN_TAX.value == "minimize_tax"


def test_order_type_enum():
    """测试订单类型枚举"""
    assert OrderType.MARKET.value == "market"
    assert OrderType.LIMIT.value == "limit"
    assert OrderType.TWAP.value == "twap"
    assert OrderType.VWAP.value == "vwap"


def test_order_side_enum():
    """测试订单方向枚举"""
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"


def test_order_status_enum():
    """测试订单状态枚举"""
    assert OrderStatus.PENDING.value == "pending"
    assert OrderStatus.SUBMITTED.value == "submitted"
    assert OrderStatus.FILLED.value == "filled"
    assert OrderStatus.CANCELLED.value == "cancelled"


def test_rebalance_state_enum():
    """测试再平衡状态枚举"""
    assert RebalanceState.IDLE.value == "idle"
    assert RebalanceState.ANALYZING.value == "analyzing"
    assert RebalanceState.GENERATING.value == "generating"
    assert RebalanceState.EXECUTING.value == "executing"


def test_alert_type_enum():
    """测试告警类型枚举"""
    assert AlertType.DRIFT_THRESHOLD.value == "drift_threshold"
    assert AlertType.CALENDAR_DUE.value == "calendar_due"
    assert AlertType.TAX_OPPORTUNITY.value == "tax_opportunity"
    assert AlertType.COST_ALERT.value == "cost_alert"


def test_tax_event_type_enum():
    """测试税务事件类型枚举"""
    assert TaxEventType.SHORT_TERM_GAIN.value == "short_term_gain"
    assert TaxEventType.LONG_TERM_GAIN.value == "long_term_gain"
    assert TaxEventType.SHORT_TERM_LOSS.value == "short_term_loss"
    assert TaxEventType.LONG_TERM_LOSS.value == "long_term_loss"


# ==================== 数据类测试 (10个) ====================

def test_position_creation():
    """测试持仓创建"""
    position = create_position("AAPL", 100, 15000, 0.30, 0.25)
    assert position.symbol == "AAPL"
    assert position.quantity == 100
    assert position.weight == 0.30
    assert position.target_weight == 0.25


def test_position_to_dict():
    """测试持仓转字典"""
    position = create_position("AAPL", 100, 15000, 0.30, 0.25, cost_basis=14000)
    pos_dict = position.to_dict()
    assert pos_dict["symbol"] == "AAPL"
    assert pos_dict["quantity"] == 100
    assert pos_dict["cost_basis"] == 14000


def test_tax_lot_creation():
    """测试税务批次创建"""
    acquisition_date = datetime.now() - timedelta(days=400)
    tax_lot = create_tax_lot(
        "AAPL", 50, 7000, acquisition_date,
        holding_period_days=400,
        is_long_term=True
    )
    assert tax_lot.symbol == "AAPL"
    assert tax_lot.quantity == 50
    assert tax_lot.is_long_term is True


def test_tax_lot_to_dict():
    """测试税务批次转字典"""
    acquisition_date = datetime.now() - timedelta(days=400)
    tax_lot = create_tax_lot("AAPL", 50, 7000, acquisition_date)
    lot_dict = tax_lot.to_dict()
    assert lot_dict["symbol"] == "AAPL"
    assert "acquisition_date" in lot_dict


def test_position_drift_to_dict():
    """测试持仓漂移转字典"""
    drift = PositionDrift(
        symbol="AAPL",
        current_weight=0.30,
        target_weight=0.25,
        drift=0.05,
        drift_pct=20.0,
        drift_type=DriftType.ABSOLUTE,
        urgency_score=0.8,
    )
    drift_dict = drift.to_dict()
    assert drift_dict["symbol"] == "AAPL"
    assert drift_dict["drift"] == 0.05
    assert drift_dict["urgency_score"] == 0.8


def test_rebalance_order_creation():
    """测试再平衡订单创建"""
    order = RebalanceOrder(
        order_id="order1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        estimated_price=150.0,
    )
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.status == OrderStatus.PENDING


def test_rebalance_order_to_dict():
    """测试再平衡订单转字典"""
    order = RebalanceOrder(
        order_id="order1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
    )
    order_dict = order.to_dict()
    assert order_dict["order_id"] == "order1"
    assert order_dict["side"] == "buy"


def test_rebalance_plan_creation():
    """测试再平衡计划创建"""
    plan = RebalancePlan(
        plan_id="plan1",
        portfolio_id="portfolio1",
        strategy=RebalanceStrategy.THRESHOLD,
    )
    assert plan.plan_id == "plan1"
    assert plan.strategy == RebalanceStrategy.THRESHOLD
    assert plan.state == RebalanceState.IDLE


def test_rebalance_plan_to_dict():
    """测试再平衡计划转字典"""
    plan = RebalancePlan(
        plan_id="plan1",
        portfolio_id="portfolio1",
        strategy=RebalanceStrategy.THRESHOLD,
        total_turnover=5000.0,
    )
    plan_dict = plan.to_dict()
    assert plan_dict["plan_id"] == "plan1"
    assert plan_dict["total_turnover"] == 5000.0


def test_rebalancer_config_to_dict(rebalancer_config):
    """测试配置转字典"""
    config_dict = rebalancer_config.to_dict()
    assert config_dict["absolute_drift_threshold"] == 0.02
    assert config_dict["commission_rate"] == 0.001


# ==================== 提供者测试 (5个) ====================

def test_simulated_portfolio_provider():
    """测试模拟组合提供者"""
    provider = SimulatedPortfolioProvider()
    positions = [create_position("AAPL", 100, 15000, 0.30, 0.25)]
    provider.set_positions("portfolio1", positions)

    retrieved = provider.get_positions("portfolio1")
    assert len(retrieved) == 1
    assert retrieved[0].symbol == "AAPL"


def test_simulated_price_provider():
    """测试模拟价格提供者"""
    provider = SimulatedPriceProvider()
    provider.set_price("AAPL", 150.0)

    price = provider.get_price("AAPL")
    assert price == 150.0


def test_simulated_price_provider_batch():
    """测试批量获取价格"""
    provider = SimulatedPriceProvider()
    provider.set_price("AAPL", 150.0)
    provider.set_price("GOOGL", 2800.0)

    prices = provider.get_prices(["AAPL", "GOOGL"])
    assert len(prices) == 2
    assert prices["AAPL"] == 150.0


def test_simulated_target_provider():
    """测试模拟目标提供者"""
    provider = SimulatedTargetProvider()
    targets = {"AAPL": 0.25, "GOOGL": 0.25}
    provider.set_targets("portfolio1", targets)

    retrieved = provider.get_target_weights("portfolio1")
    assert retrieved["AAPL"] == 0.25


def test_simulated_order_executor():
    """测试模拟订单执行器"""
    executor = SimulatedOrderExecutor()
    order = RebalanceOrder(
        order_id="order1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        estimated_price=150.0,
    )

    success = executor.execute_order(order)
    assert success
    assert order.status == OrderStatus.FILLED


# ==================== 漂移监控测试 (8个) ====================

def test_calculate_drift_absolute(rebalancer, portfolio_provider, sample_positions):
    """测试计算绝对漂移"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    drifts = rebalancer.calculate_drift("portfolio1", DriftType.ABSOLUTE)
    assert len(drifts) == 4

    # AAPL应该有最大漂移 (0.30 - 0.25 = 0.05)
    aapl_drift = next(d for d in drifts if d.symbol == "AAPL")
    assert abs(aapl_drift.drift - 0.05) < 0.01


def test_calculate_drift_relative(rebalancer, portfolio_provider, sample_positions):
    """测试计算相对漂移"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    drifts = rebalancer.calculate_drift("portfolio1", DriftType.RELATIVE)
    assert len(drifts) == 4


def test_drift_urgency_score(rebalancer, portfolio_provider, sample_positions):
    """测试漂移紧急程度评分"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    drifts = rebalancer.calculate_drift("portfolio1")

    # 检查紧急程度评分
    for drift in drifts:
        assert 0 <= drift.urgency_score <= 1.0


def test_check_drift_threshold_exceeded(rebalancer, portfolio_provider):
    """测试漂移超过阈值"""
    # 创建超过阈值的持仓
    positions = [
        create_position("AAPL", 300, 45000, 0.45, 0.25),  # 漂移 0.20
        create_position("GOOGL", 10, 28000, 0.28, 0.25),
        create_position("MSFT", 50, 15000, 0.15, 0.25),
        create_position("AMZN", 3, 9600, 0.096, 0.25),
    ]
    portfolio_provider.set_positions("portfolio1", positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    exceeded = rebalancer.check_drift_threshold("portfolio1")
    assert exceeded


def test_check_drift_threshold_not_exceeded(rebalancer, portfolio_provider):
    """测试漂移未超过阈值"""
    # 创建轻微漂移的持仓
    positions = [
        create_position("AAPL", 200, 30000, 0.26, 0.25),  # 漂移 0.01
        create_position("GOOGL", 10, 28000, 0.26, 0.25),
        create_position("MSFT", 80, 24000, 0.24, 0.25),
        create_position("AMZN", 8, 25600, 0.24, 0.25),
    ]
    portfolio_provider.set_positions("portfolio1", positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    exceeded = rebalancer.check_drift_threshold("portfolio1")
    assert not exceeded


def test_drift_calculation_empty_portfolio(rebalancer):
    """测试空组合漂移计算"""
    drifts = rebalancer.calculate_drift("empty_portfolio")
    assert len(drifts) == 0


def test_drift_recommended_trade(rebalancer, portfolio_provider, sample_positions):
    """测试漂移推荐交易"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    drifts = rebalancer.calculate_drift("portfolio1")

    for drift in drifts:
        if abs(drift.drift) > 0.01:
            # 应该有交易建议
            assert drift.recommended_trade_value != 0
            assert drift.recommended_trade_shares != 0


def test_drift_to_dict(rebalancer, portfolio_provider, sample_positions):
    """测试漂移转字典"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    drifts = rebalancer.calculate_drift("portfolio1")
    drift_dict = drifts[0].to_dict()

    assert "symbol" in drift_dict
    assert "drift" in drift_dict
    assert "urgency_score" in drift_dict


# ==================== 日历再平衡测试 (5个) ====================

def test_schedule_calendar_rebalance_quarterly(rebalancer):
    """测试调度季度再平衡"""
    rebalancer.schedule_calendar_rebalance()
    assert rebalancer.config.next_calendar_rebalance is not None

    days_until = (rebalancer.config.next_calendar_rebalance - datetime.now()).days
    assert 85 <= days_until <= 95  # 大约90天


def test_schedule_calendar_rebalance_monthly(rebalancer):
    """测试调度月度再平衡"""
    rebalancer.config.calendar_frequency = RebalanceFrequency.MONTHLY
    rebalancer.schedule_calendar_rebalance()

    days_until = (rebalancer.config.next_calendar_rebalance - datetime.now()).days
    assert 25 <= days_until <= 35  # 大约30天


def test_is_calendar_rebalance_due_not_scheduled(rebalancer):
    """测试未调度的日历再平衡"""
    assert not rebalancer.is_calendar_rebalance_due()


def test_is_calendar_rebalance_due_past(rebalancer):
    """测试已到期的日历再平衡"""
    rebalancer.config.next_calendar_rebalance = datetime.now() - timedelta(days=1)
    assert rebalancer.is_calendar_rebalance_due()


def test_is_calendar_rebalance_due_future(rebalancer):
    """测试未到期的日历再平衡"""
    rebalancer.config.next_calendar_rebalance = datetime.now() + timedelta(days=30)
    assert not rebalancer.is_calendar_rebalance_due()


# ==================== 税务感知测试 (8个) ====================

def test_find_tax_harvesting_opportunities(rebalancer, portfolio_provider):
    """测试寻找税收收割机会"""
    # 创建有损失的持仓
    loss_lot = create_tax_lot(
        "AAPL", 100, 16000,
        datetime.now() - timedelta(days=100),
        holding_period_days=100,
        unrealized_gain=-1000,  # 损失
    )
    position = create_position(
        "AAPL", 100, 15000, 0.30, 0.25,
        unrealized_pnl=-1000,
        tax_lots=[loss_lot]
    )

    portfolio_provider.set_positions("portfolio1", [position])

    opportunities = rebalancer.find_tax_harvesting_opportunities("portfolio1")
    assert len(opportunities) > 0
    assert opportunities[0].unrealized_loss > 0


def test_tax_harvesting_no_opportunities(rebalancer, portfolio_provider):
    """测试无税收收割机会"""
    # 创建有收益的持仓
    gain_lot = create_tax_lot(
        "AAPL", 100, 14000,
        datetime.now() - timedelta(days=400),
        holding_period_days=400,
        unrealized_gain=1000,  # 收益
    )
    position = create_position(
        "AAPL", 100, 15000, 0.30, 0.25,
        unrealized_pnl=1000,
        tax_lots=[gain_lot]
    )

    portfolio_provider.set_positions("portfolio1", [position])

    opportunities = rebalancer.find_tax_harvesting_opportunities("portfolio1")
    assert len(opportunities) == 0


def test_tax_lot_selection_fifo(rebalancer):
    """测试FIFO税务批次选择"""
    lots = [
        create_tax_lot("AAPL", 50, 7000, datetime.now() - timedelta(days=300)),
        create_tax_lot("AAPL", 50, 7500, datetime.now() - timedelta(days=200)),
        create_tax_lot("AAPL", 50, 8000, datetime.now() - timedelta(days=100)),
    ]
    position = create_position("AAPL", 150, 22500, 0.30, 0.25, tax_lots=lots)

    selected = rebalancer.select_tax_lots(position, 75, TaxLotMethod.FIFO)
    assert len(selected) == 2
    assert selected[0].cost_basis == 7000


def test_tax_lot_selection_lifo(rebalancer):
    """测试LIFO税务批次选择"""
    lots = [
        create_tax_lot("AAPL", 50, 7000, datetime.now() - timedelta(days=300)),
        create_tax_lot("AAPL", 50, 7500, datetime.now() - timedelta(days=200)),
        create_tax_lot("AAPL", 50, 8000, datetime.now() - timedelta(days=100)),
    ]
    position = create_position("AAPL", 150, 22500, 0.30, 0.25, tax_lots=lots)

    selected = rebalancer.select_tax_lots(position, 75, TaxLotMethod.LIFO)
    assert len(selected) == 2
    assert selected[0].cost_basis == 8000


def test_tax_lot_selection_hifo(rebalancer):
    """测试HIFO税务批次选择"""
    lots = [
        create_tax_lot("AAPL", 50, 7000, datetime.now() - timedelta(days=300)),
        create_tax_lot("AAPL", 50, 8000, datetime.now() - timedelta(days=200)),
        create_tax_lot("AAPL", 50, 7500, datetime.now() - timedelta(days=100)),
    ]
    position = create_position("AAPL", 150, 22500, 0.30, 0.25, tax_lots=lots)

    selected = rebalancer.select_tax_lots(position, 75, TaxLotMethod.HIFO)
    assert len(selected) == 2
    assert selected[0].cost_basis == 8000


def test_calculate_tax_impact_short_term(rebalancer):
    """测试计算短期税务影响"""
    lots = [
        create_tax_lot(
            "AAPL", 50, 7000,
            datetime.now() - timedelta(days=100),
            holding_period_days=100,
            is_long_term=False,
            unrealized_gain=500,
        ),
    ]
    position = create_position("AAPL", 50, 7500, 0.30, 0.25, tax_lots=lots)

    tax_impact = rebalancer.calculate_tax_impact(position, 50)
    # 短期税率 0.37
    expected_tax = 500 * 0.37
    assert abs(tax_impact - expected_tax) < 1.0


def test_calculate_tax_impact_long_term(rebalancer):
    """测试计算长期税务影响"""
    lots = [
        create_tax_lot(
            "AAPL", 50, 7000,
            datetime.now() - timedelta(days=400),
            holding_period_days=400,
            is_long_term=True,
            unrealized_gain=500,
        ),
    ]
    position = create_position("AAPL", 50, 7500, 0.30, 0.25, tax_lots=lots)

    tax_impact = rebalancer.calculate_tax_impact(position, 50)
    # 长期税率 0.20
    expected_tax = 500 * 0.20
    assert abs(tax_impact - expected_tax) < 1.0


def test_wash_sale_detection(rebalancer, portfolio_provider):
    """测试洗售检测"""
    # 创建最近购买的损失批次
    loss_lot = create_tax_lot(
        "AAPL", 100, 16000,
        datetime.now() - timedelta(days=15),
        holding_period_days=15,
        unrealized_gain=-1000,
    )
    position = create_position(
        "AAPL", 100, 15000, 0.30, 0.25,
        unrealized_pnl=-1000,
        tax_lots=[loss_lot]
    )

    portfolio_provider.set_positions("portfolio1", [position])

    opportunities = rebalancer.find_tax_harvesting_opportunities("portfolio1")
    if opportunities:
        assert opportunities[0].wash_sale_risk is True


# ==================== 交易成本优化测试 (5个) ====================

def test_estimate_transaction_cost(rebalancer):
    """测试估算交易成本"""
    cost = rebalancer.estimate_transaction_cost(
        "AAPL", 100, OrderSide.BUY, 100000
    )
    assert cost > 0


def test_estimate_transaction_cost_commission(rebalancer):
    """测试佣金计算"""
    quantity = 100
    price = 150.0
    trade_value = quantity * price

    cost = rebalancer.estimate_transaction_cost(
        "AAPL", quantity, OrderSide.BUY, 1000000
    )

    # 佣金应该至少是 trade_value * commission_rate
    min_cost = trade_value * rebalancer.config.commission_rate
    assert cost >= min_cost


def test_optimize_order_size(rebalancer):
    """测试优化订单大小"""
    orders = [
        RebalanceOrder(
            order_id="o1", symbol="AAPL", side=OrderSide.BUY,
            quantity=100, estimated_price=150.0  # 15000
        ),
        RebalanceOrder(
            order_id="o2", symbol="GOOGL", side=OrderSide.BUY,
            quantity=0.02, estimated_price=2800.0  # 56 - 小于min_trade_value (100)
        ),
    ]

    optimized = rebalancer.optimize_order_size(orders, 100000)
    # 第二个订单应该被过滤掉
    assert len(optimized) == 1
    assert optimized[0].symbol == "AAPL"


def test_transaction_cost_market_impact(rebalancer):
    """测试市场冲击成本"""
    # 大额交易应该有更高的市场冲击
    cost_small = rebalancer.estimate_transaction_cost(
        "AAPL", 10, OrderSide.BUY, 1000000
    )
    cost_large = rebalancer.estimate_transaction_cost(
        "AAPL", 1000, OrderSide.BUY, 1000000
    )

    assert cost_large > cost_small


def test_transaction_cost_zero_quantity(rebalancer):
    """测试零数量交易成本"""
    cost = rebalancer.estimate_transaction_cost(
        "AAPL", 0, OrderSide.BUY, 100000
    )
    assert cost == 0


# ==================== 再平衡计划生成测试 (8个) ====================

def test_generate_rebalance_plan(rebalancer, portfolio_provider, sample_positions):
    """测试生成再平衡计划"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    plan = rebalancer.generate_rebalance_plan("portfolio1")
    assert plan.portfolio_id == "portfolio1"
    assert len(plan.orders) > 0


def test_rebalance_plan_respects_turnover_limit(rebalancer, portfolio_provider):
    """测试再平衡计划遵守换手率限制"""
    # 创建极度偏离的持仓
    positions = [
        create_position("AAPL", 500, 75000, 0.75, 0.25),
        create_position("GOOGL", 3, 8400, 0.084, 0.25),
        create_position("MSFT", 30, 9000, 0.09, 0.25),
        create_position("AMZN", 2, 6400, 0.064, 0.25),
    ]
    portfolio_provider.set_positions("portfolio1", positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    plan = rebalancer.generate_rebalance_plan("portfolio1")
    # 换手率应该不超过限制
    assert plan.turnover_rate <= rebalancer.config.max_turnover_rate + 0.01


def test_rebalance_plan_cost_estimation(rebalancer, portfolio_provider, sample_positions):
    """测试再平衡计划成本估算"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    plan = rebalancer.generate_rebalance_plan("portfolio1")
    assert plan.estimated_total_cost > 0
    assert plan.estimated_commission > 0


def test_rebalance_plan_tax_awareness(rebalancer, portfolio_provider):
    """测试税务感知再平衡"""
    # 创建有税务批次的持仓
    lots = [
        create_tax_lot(
            "AAPL", 200, 28000,
            datetime.now() - timedelta(days=400),
            holding_period_days=400,
            is_long_term=True,
            unrealized_gain=2000,
        ),
    ]
    positions = [
        create_position("AAPL", 200, 30000, 0.30, 0.25, tax_lots=lots),
        create_position("GOOGL", 10, 28000, 0.28, 0.25),
        create_position("MSFT", 80, 24000, 0.24, 0.25),
        create_position("AMZN", 5, 16000, 0.16, 0.25),
    ]
    portfolio_provider.set_positions("portfolio1", positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    rebalancer.config.optimize_for_tax = True
    plan = rebalancer.generate_rebalance_plan("portfolio1")

    # 应该考虑税务影响
    sell_orders = [o for o in plan.orders if o.side == OrderSide.SELL]
    for order in sell_orders:
        if order.symbol == "AAPL":
            assert len(order.tax_lots_used) > 0


def test_rebalance_plan_drift_reduction(rebalancer, portfolio_provider, sample_positions):
    """测试再平衡计划漂移减少"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    plan = rebalancer.generate_rebalance_plan("portfolio1")
    assert plan.drift_reduction >= 0
    assert plan.post_drift_max <= plan.pre_drift_max


def test_execute_rebalance_plan(rebalancer, portfolio_provider, sample_positions):
    """测试执行再平衡计划"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    plan = rebalancer.generate_rebalance_plan("portfolio1")

    # 确保计划状态可以被执行
    if plan.state == RebalanceState.IDLE:
        # 手动修改状态以便测试 - 在实际实现中会通过其他方式处理
        pass

    success = rebalancer.execute_rebalance_plan(plan)

    assert success
    assert plan.state == RebalanceState.COMPLETED


def test_execute_rebalance_plan_updates_stats(rebalancer, portfolio_provider, sample_positions):
    """测试执行再平衡更新统计"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    initial_rebalances = rebalancer.stats["total_rebalances"]

    plan = rebalancer.generate_rebalance_plan("portfolio1")
    rebalancer.execute_rebalance_plan(plan)

    assert rebalancer.stats["total_rebalances"] == initial_rebalances + 1


def test_rebalance_plan_empty_portfolio(rebalancer):
    """测试空组合再平衡计划"""
    plan = rebalancer.generate_rebalance_plan("empty_portfolio")
    assert len(plan.orders) == 0


# ==================== 假设分析测试 (5个) ====================

def test_create_what_if_scenario(rebalancer):
    """测试创建假设场景"""
    scenario = rebalancer.create_what_if_scenario(
        "Test Scenario",
        "portfolio1",
        weight_changes={"AAPL": 0.30, "GOOGL": 0.20},
    )
    assert scenario.scenario_name == "Test Scenario"
    assert "AAPL" in scenario.weight_changes


def test_analyze_what_if_scenario(rebalancer, portfolio_provider, sample_positions):
    """测试分析假设场景"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    scenario = rebalancer.create_what_if_scenario(
        "Test",
        "portfolio1",
        weight_changes={"AAPL": 0.30},
    )

    analyzed = rebalancer.analyze_what_if_scenario(scenario, "portfolio1")
    assert analyzed.rebalance_plan is not None


def test_compare_scenarios(rebalancer):
    """测试比较多个场景"""
    scenario1 = rebalancer.create_what_if_scenario(
        "Scenario 1", "portfolio1",
    )
    scenario1.predicted_cost = 100.0
    scenario1.predicted_tax = 50.0

    scenario2 = rebalancer.create_what_if_scenario(
        "Scenario 2", "portfolio1",
    )
    scenario2.predicted_cost = 150.0
    scenario2.predicted_tax = 30.0

    comparison = rebalancer.compare_scenarios([scenario1.scenario_id, scenario2.scenario_id])
    assert "best_for_cost" in comparison
    assert comparison["best_for_cost"] == scenario1.scenario_id


def test_what_if_scenario_to_dict(rebalancer):
    """测试假设场景转字典"""
    scenario = rebalancer.create_what_if_scenario(
        "Test", "portfolio1",
        weight_changes={"AAPL": 0.30},
    )
    scenario_dict = scenario.to_dict()
    assert scenario_dict["scenario_name"] == "Test"
    assert "weight_changes" in scenario_dict


def test_multiple_scenarios(rebalancer):
    """测试多个场景"""
    for i in range(5):
        rebalancer.create_what_if_scenario(f"Scenario {i}", "portfolio1")

    assert len(rebalancer.scenarios) == 5


# ==================== 告警测试 (5个) ====================

def test_create_drift_alert(rebalancer, portfolio_provider):
    """测试创建漂移告警"""
    positions = [
        create_position("AAPL", 400, 60000, 0.60, 0.25),  # 大漂移
        create_position("GOOGL", 5, 14000, 0.14, 0.25),
    ]
    portfolio_provider.set_positions("portfolio1", positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    rebalancer.check_drift_threshold("portfolio1")

    alerts = rebalancer.get_alerts(alert_type=AlertType.DRIFT_THRESHOLD)
    assert len(alerts) > 0


def test_get_alerts_filtered(rebalancer, portfolio_provider):
    """测试获取过滤的告警"""
    # 创建告警
    rebalancer._create_alert(
        AlertType.DRIFT_THRESHOLD,
        "portfolio1",
        "Test alert 1"
    )
    rebalancer._create_alert(
        AlertType.TAX_OPPORTUNITY,
        "portfolio2",
        "Test alert 2"
    )

    # 按组合过滤
    alerts = rebalancer.get_alerts(portfolio_id="portfolio1")
    assert len(alerts) == 1
    assert alerts[0].portfolio_id == "portfolio1"


def test_acknowledge_alert(rebalancer):
    """测试确认告警"""
    alert = rebalancer._create_alert(
        AlertType.DRIFT_THRESHOLD,
        "portfolio1",
        "Test alert"
    )

    success = rebalancer.acknowledge_alert(alert.alert_id, "test_user")
    assert success

    acknowledged = rebalancer.get_alerts(acknowledged=True)
    assert len(acknowledged) > 0


def test_alert_callback(rebalancer):
    """测试告警回调"""
    alerts_received = []

    def on_alert(alert):
        alerts_received.append(alert)

    rebalancer.on_alert(on_alert)

    rebalancer._create_alert(
        AlertType.DRIFT_THRESHOLD,
        "portfolio1",
        "Test alert"
    )

    assert len(alerts_received) == 1


def test_alert_to_dict(rebalancer):
    """测试告警转字典"""
    alert = rebalancer._create_alert(
        AlertType.DRIFT_THRESHOLD,
        "portfolio1",
        "Test alert",
        severity="warning",
    )

    alert_dict = alert.to_dict()
    assert alert_dict["message"] == "Test alert"
    assert alert_dict["severity"] == "warning"


# ==================== 异步监控测试 (3个) ====================

@pytest.mark.asyncio
async def test_start_stop_monitoring(rebalancer, portfolio_provider, sample_positions):
    """测试启动停止监控"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    await rebalancer.start_monitoring(["portfolio1"])
    assert rebalancer.state == RebalanceState.MONITORING

    await asyncio.sleep(0.1)

    await rebalancer.stop_monitoring()
    assert rebalancer.state == RebalanceState.IDLE


@pytest.mark.asyncio
async def test_monitoring_drift_detection(rebalancer, portfolio_provider):
    """测试监控漂移检测"""
    # 创建超过阈值的持仓
    positions = [
        create_position("AAPL", 400, 60000, 0.60, 0.25),
        create_position("GOOGL", 5, 14000, 0.14, 0.25),
    ]
    portfolio_provider.set_positions("portfolio1", positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    rebalancer.config.monitor_interval_seconds = 0.1

    await rebalancer.start_monitoring(["portfolio1"])
    await asyncio.sleep(0.3)
    await rebalancer.stop_monitoring()

    # 应该生成了再平衡计划
    assert len(rebalancer.rebalance_plans) > 0


@pytest.mark.asyncio
async def test_monitoring_tax_harvesting(rebalancer, portfolio_provider):
    """测试监控税收收割"""
    # 创建有损失的持仓
    loss_lot = create_tax_lot(
        "AAPL", 100, 16000,
        datetime.now() - timedelta(days=100),
        holding_period_days=100,
        unrealized_gain=-1000,
    )
    position = create_position(
        "AAPL", 100, 15000, 0.25, 0.25,
        unrealized_pnl=-1000,
        tax_lots=[loss_lot]
    )

    portfolio_provider.set_positions("portfolio1", [position])
    portfolio_provider.set_total_value("portfolio1", 100000)

    rebalancer.config.monitor_interval_seconds = 0.1
    rebalancer.config.enable_tax_harvesting = True

    await rebalancer.start_monitoring(["portfolio1"])
    await asyncio.sleep(0.3)
    await rebalancer.stop_monitoring()

    # 应该创建了税收机会告警
    alerts = rebalancer.get_alerts(alert_type=AlertType.TAX_OPPORTUNITY)
    assert len(alerts) > 0


# ==================== 统计和查询测试 (5个) ====================

def test_get_statistics(rebalancer, portfolio_provider, sample_positions):
    """测试获取统计信息"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    rebalancer.generate_rebalance_plan("portfolio1")

    stats = rebalancer.get_statistics()
    assert "state" in stats
    assert "total_plans" in stats
    assert stats["total_plans"] >= 1


def test_get_rebalance_plan(rebalancer, portfolio_provider, sample_positions):
    """测试获取再平衡计划"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    plan = rebalancer.generate_rebalance_plan("portfolio1")
    retrieved = rebalancer.get_rebalance_plan(plan.plan_id)

    assert retrieved is not None
    assert retrieved.plan_id == plan.plan_id


def test_get_rebalance_history(rebalancer, portfolio_provider, sample_positions):
    """测试获取再平衡历史"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    # 生成多个计划
    for _ in range(3):
        rebalancer.generate_rebalance_plan("portfolio1")

    history = rebalancer.get_rebalance_history("portfolio1")
    assert len(history) == 3


def test_get_rebalance_history_filtered(rebalancer, portfolio_provider, sample_positions):
    """测试获取过滤的再平衡历史"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)
    portfolio_provider.set_positions("portfolio2", sample_positions)

    rebalancer.generate_rebalance_plan("portfolio1")
    rebalancer.generate_rebalance_plan("portfolio2")

    history = rebalancer.get_rebalance_history("portfolio1")
    assert len(history) == 1
    assert history[0].portfolio_id == "portfolio1"


def test_statistics_accumulation(rebalancer, portfolio_provider, sample_positions):
    """测试统计累积"""
    portfolio_provider.set_positions("portfolio1", sample_positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    initial_stats = rebalancer.get_statistics()

    plan = rebalancer.generate_rebalance_plan("portfolio1")
    rebalancer.execute_rebalance_plan(plan)

    final_stats = rebalancer.get_statistics()
    assert final_stats["total_rebalances"] > initial_stats["total_rebalances"]


# ==================== 工厂函数测试 (3个) ====================

def test_create_portfolio_rebalancer():
    """测试创建再平衡器"""
    rebalancer = create_portfolio_rebalancer()
    assert rebalancer is not None
    assert isinstance(rebalancer, PortfolioRebalancer)


def test_create_position_factory():
    """测试创建持仓"""
    position = create_position("AAPL", 100, 15000, 0.30, 0.25)
    assert position.symbol == "AAPL"
    assert position.quantity == 100


def test_quick_rebalance_analysis():
    """测试快速再平衡分析"""
    positions = [
        {"symbol": "AAPL", "quantity": 200, "market_value": 30000, "weight": 0.30},
        {"symbol": "GOOGL", "quantity": 10, "market_value": 28000, "weight": 0.28},
    ]
    target_weights = {
        "AAPL": 0.25,
        "GOOGL": 0.25,
    }

    result = quick_rebalance_analysis(positions, target_weights, 100000)

    assert "drifts" in result
    assert "plan" in result
    assert "statistics" in result


# ==================== 边界条件测试 (3个) ====================

def test_rebalance_zero_value_portfolio(rebalancer):
    """测试零值组合再平衡"""
    plan = rebalancer.generate_rebalance_plan("zero_portfolio")
    assert len(plan.orders) == 0


def test_drift_calculation_zero_target_weight(rebalancer, portfolio_provider):
    """测试零目标权重漂移计算"""
    positions = [
        create_position("AAPL", 100, 15000, 0.30, 0.0),  # 目标为0
    ]
    portfolio_provider.set_positions("portfolio1", positions)
    portfolio_provider.set_total_value("portfolio1", 100000)

    drifts = rebalancer.calculate_drift("portfolio1")
    assert len(drifts) >= 1


def test_execute_empty_plan(rebalancer):
    """测试执行空计划"""
    plan = RebalancePlan(
        plan_id="empty",
        portfolio_id="portfolio1",
        strategy=RebalanceStrategy.THRESHOLD,
    )

    success = rebalancer.execute_rebalance_plan(plan)
    assert success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
