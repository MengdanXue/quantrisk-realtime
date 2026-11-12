"""
订单管理系统测试
Order Management System Tests

测试:
- 订单创建和验证
- 订单提交和成交
- 订单取消和修改
- 仓位管理
- 风险检查
- OMS集成

作者: TraderX Team
创建: 2025-12-24
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

import sys
sys.path.insert(0, 'src')

from quantrisk_realtime.realtime.order_manager import (
    # Enums
    OrderType,
    OrderSide,
    OrderStatus,
    PositionSide,
    RiskCheckResult,
    TimeInForce,
    # Data Classes
    Order,
    Fill,
    Position,
    RiskLimits,
    OMSConfig,
    # Classes
    RiskChecker,
    PositionManager,
    OrderManager,
    OMS,
    # Factory Functions
    create_oms,
    create_order,
)


# ============ Test Fixtures ============

@pytest.fixture
def order_manager():
    """订单管理器fixture"""
    return OrderManager()


@pytest.fixture
def position_manager():
    """仓位管理器fixture"""
    return PositionManager()


@pytest.fixture
def risk_checker():
    """风险检查器fixture"""
    return RiskChecker()


@pytest.fixture
def oms():
    """OMS fixture"""
    return create_oms(
        enable_risk_checks=True,
        max_position_pct=0.20,
        max_daily_trades=100,
    )


# ============ Order Tests ============

class TestOrder:
    """订单测试"""

    def test_create_order(self):
        """测试创建订单"""
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        assert order.order_id == "ORD-001"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING

    def test_order_remaining_quantity(self):
        """测试剩余数量"""
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order.filled_quantity = 30

        assert order.remaining_quantity == 70

    def test_order_is_active(self):
        """测试活跃状态"""
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert order.is_active is True

        order.status = OrderStatus.FILLED
        assert order.is_active is False

    def test_order_is_complete(self):
        """测试完成状态"""
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert order.is_complete is False

        order.status = OrderStatus.FILLED
        assert order.is_complete is True

    def test_order_to_dict(self):
        """测试序列化"""
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            limit_price=150.0,
        )

        d = order.to_dict()

        assert d['order_id'] == "ORD-001"
        assert d['symbol'] == "AAPL"
        assert d['side'] == "buy"
        assert d['quantity'] == 100
        assert d['limit_price'] == 150.0


# ============ Position Tests ============

class TestPosition:
    """持仓测试"""

    def test_create_position(self):
        """测试创建持仓"""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
            average_price=150.0,
            market_price=155.0,
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.average_price == 150.0

    def test_position_market_value(self):
        """测试市值计算"""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
            average_price=150.0,
            market_price=155.0,
        )

        assert position.market_value == 15500.0
        assert position.cost_basis == 15000.0

    def test_position_update_market_price(self):
        """测试更新市价"""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
            average_price=150.0,
        )

        position.update_market_price(160.0)

        assert position.market_price == 160.0
        assert position.unrealized_pnl == 1000.0  # (160-150)*100

    def test_position_short_pnl(self):
        """测试空仓盈亏"""
        position = Position(
            symbol="AAPL",
            side=PositionSide.SHORT,
            quantity=100,
            average_price=150.0,
        )

        position.update_market_price(140.0)

        # 空仓盈利
        assert position.unrealized_pnl == 1000.0  # (150-140)*100


# ============ OrderManager Tests ============

class TestOrderManager:
    """订单管理器测试"""

    def test_create_order(self, order_manager):
        """测试创建订单"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        assert order.order_id is not None
        assert order.symbol == "AAPL"
        assert order.status == OrderStatus.PENDING

    def test_create_limit_order(self, order_manager):
        """测试创建限价单"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.0

    def test_create_limit_order_without_price_fails(self, order_manager):
        """测试创建限价单缺少价格"""
        with pytest.raises(ValueError, match="Limit order requires limit_price"):
            order_manager.create_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
            )

    def test_submit_order(self, order_manager):
        """测试提交订单"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        submitted = order_manager.submit_order(order.order_id)

        assert submitted.status == OrderStatus.SUBMITTED
        assert submitted.submitted_at is not None

    def test_fill_order_full(self, order_manager):
        """测试完全成交"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order_manager.submit_order(order.order_id)
        order_manager.accept_order(order.order_id)

        filled_order, fill = order_manager.fill_order(
            order.order_id,
            fill_quantity=100,
            fill_price=150.0,
            commission=1.5,
        )

        assert filled_order.status == OrderStatus.FILLED
        assert filled_order.filled_quantity == 100
        assert filled_order.average_price == 150.0
        assert fill.quantity == 100
        assert fill.price == 150.0

    def test_fill_order_partial(self, order_manager):
        """测试部分成交"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order_manager.submit_order(order.order_id)
        order_manager.accept_order(order.order_id)

        filled_order, _ = order_manager.fill_order(
            order.order_id,
            fill_quantity=50,
            fill_price=150.0,
        )

        assert filled_order.status == OrderStatus.PARTIAL_FILLED
        assert filled_order.filled_quantity == 50
        assert filled_order.remaining_quantity == 50

    def test_cancel_order(self, order_manager):
        """测试取消订单"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order_manager.submit_order(order.order_id)

        cancelled = order_manager.cancel_order(order.order_id, "Test cancel")

        assert cancelled.status == OrderStatus.CANCELLED
        assert cancelled.cancelled_at is not None
        assert cancelled.rejection_reason == "Test cancel"

    def test_modify_order(self, order_manager):
        """测试修改订单"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )

        modified = order_manager.modify_order(
            order.order_id,
            quantity=150,
            limit_price=148.0,
        )

        assert modified.quantity == 150
        assert modified.limit_price == 148.0

    def test_get_active_orders(self, order_manager):
        """测试获取活跃订单"""
        order1 = order_manager.create_order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order2 = order_manager.create_order(symbol="MSFT", side=OrderSide.BUY, quantity=50)
        order_manager.submit_order(order1.order_id)

        active = order_manager.get_active_orders()

        assert len(active) == 2

    def test_cancel_all_orders(self, order_manager):
        """测试取消所有订单"""
        order_manager.create_order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order_manager.create_order(symbol="MSFT", side=OrderSide.BUY, quantity=50)

        cancelled = order_manager.cancel_all_orders()

        assert cancelled == 2
        assert len(order_manager.get_active_orders()) == 0

    def test_get_statistics(self, order_manager):
        """测试获取统计"""
        order = order_manager.create_order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order_manager.submit_order(order.order_id)
        order_manager.accept_order(order.order_id)
        order_manager.fill_order(order.order_id, 100, 150.0, 1.5)

        stats = order_manager.get_statistics()

        assert stats['filled_orders'] == 1
        assert stats['total_fills'] == 1
        assert stats['total_commission'] == 1.5


# ============ PositionManager Tests ============

class TestPositionManager:
    """仓位管理器测试"""

    def test_update_position_new_long(self, position_manager):
        """测试新建多仓"""
        fill = Fill(
            fill_id="FILL-001",
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=1.5,
        )

        position = position_manager.update_position(fill)

        assert position.symbol == "AAPL"
        assert position.side == PositionSide.LONG
        assert position.quantity == 100
        assert position.average_price == 150.0

    def test_update_position_add_long(self, position_manager):
        """测试加多仓"""
        # 首次买入
        fill1 = Fill(
            fill_id="FILL-001",
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=1.5,
        )
        position_manager.update_position(fill1)

        # 再次买入
        fill2 = Fill(
            fill_id="FILL-002",
            order_id="ORD-002",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=160.0,
            commission=1.6,
        )
        position = position_manager.update_position(fill2)

        assert position.quantity == 200
        assert position.average_price == 155.0  # (100*150 + 100*160) / 200

    def test_update_position_close_long(self, position_manager):
        """测试平多仓"""
        # 开多仓
        fill1 = Fill(
            fill_id="FILL-001",
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=1.5,
        )
        position_manager.update_position(fill1)

        # 平仓
        fill2 = Fill(
            fill_id="FILL-002",
            order_id="ORD-002",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            price=160.0,
            commission=1.6,
        )
        position = position_manager.update_position(fill2)

        assert position.quantity == 0
        assert position.side == PositionSide.FLAT
        assert position.realized_pnl == 1000.0  # (160-150)*100

    def test_update_position_new_short(self, position_manager):
        """测试新建空仓"""
        fill = Fill(
            fill_id="FILL-001",
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            price=150.0,
            commission=1.5,
        )

        position = position_manager.update_position(fill)

        assert position.side == PositionSide.SHORT
        assert position.quantity == 100

    def test_get_total_pnl(self, position_manager):
        """测试获取总盈亏"""
        # 开仓
        fill1 = Fill(
            fill_id="FILL-001",
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=1.5,
        )
        position_manager.update_position(fill1)

        # 更新市价
        position_manager.update_market_prices({'AAPL': 160.0})

        unrealized, realized = position_manager.get_total_pnl()

        assert unrealized == 1000.0
        assert realized == 0.0

    def test_close_position(self, position_manager):
        """测试平仓"""
        fill = Fill(
            fill_id="FILL-001",
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=1.5,
        )
        position_manager.update_position(fill)

        realized = position_manager.close_position("AAPL", 160.0)

        assert realized == 1000.0

        position = position_manager.get_position("AAPL")
        assert position.quantity == 0


# ============ RiskChecker Tests ============

class TestRiskChecker:
    """风险检查器测试"""

    def test_check_order_pass(self, risk_checker):
        """测试订单通过风险检查"""
        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            limit_price=150.0,
        )

        result, reason = risk_checker.check_order(order, {}, 1_000_000)

        assert result == RiskCheckResult.PASS

    def test_check_order_reject_size(self):
        """测试订单大小超限"""
        limits = RiskLimits(max_order_size=10_000)
        checker = RiskChecker(limits)

        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            limit_price=150.0,  # 15000 > 10000
        )

        result, reason = checker.check_order(order, {}, 1_000_000)

        assert result == RiskCheckResult.REJECT_ORDER_SIZE

    def test_check_order_reject_concentration(self):
        """测试持仓集中度超限"""
        limits = RiskLimits(max_position_pct=0.10)
        checker = RiskChecker(limits)

        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            limit_price=150.0,  # 15000 / 100000 = 15% > 10%
        )

        result, reason = checker.check_order(order, {}, 100_000)

        assert result == RiskCheckResult.REJECT_CONCENTRATION

    def test_check_order_reject_daily_limit(self):
        """测试日交易次数超限"""
        limits = RiskLimits(max_daily_trades=2)
        checker = RiskChecker(limits)

        # 模拟已交易2次
        checker._daily_trades = 2

        order = Order(
            order_id="ORD-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            limit_price=150.0,
        )

        result, reason = checker.check_order(order, {}, 1_000_000)

        assert result == RiskCheckResult.REJECT_DAILY_LIMIT


# ============ OMS Integration Tests ============

class TestOMS:
    """OMS集成测试"""

    @pytest.mark.asyncio
    async def test_oms_start_stop(self, oms):
        """测试OMS启动停止"""
        await oms.start()
        assert oms._running is True

        await oms.stop()
        assert oms._running is False

    @pytest.mark.asyncio
    async def test_submit_order(self, oms):
        """测试提交订单"""
        await oms.start()

        order = await oms.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
        )

        assert order.status == OrderStatus.SUBMITTED
        assert order.symbol == "AAPL"

        await oms.stop()

    @pytest.mark.asyncio
    async def test_submit_and_fill_order(self, oms):
        """测试提交并成交订单"""
        await oms.start()

        order = await oms.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )

        # 接受订单
        oms.order_manager.accept_order(order.order_id)

        # 成交
        filled_order, fill, position = await oms.execute_fill(
            order.order_id,
            fill_quantity=100,
            fill_price=150.0,
        )

        assert filled_order.status == OrderStatus.FILLED
        assert position.symbol == "AAPL"
        assert position.quantity == 100

        await oms.stop()

    @pytest.mark.asyncio
    async def test_cancel_order(self, oms):
        """测试取消订单"""
        await oms.start()

        order = await oms.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
        )

        cancelled = await oms.cancel_order(order.order_id, "Test")

        assert cancelled.status == OrderStatus.CANCELLED

        await oms.stop()

    @pytest.mark.asyncio
    async def test_risk_check_reject(self):
        """测试风险检查拒绝"""
        oms = create_oms(
            enable_risk_checks=True,
            max_position_pct=0.05,  # 5%限制
        )
        oms.set_portfolio_value(100_000)

        await oms.start()

        # 尝试买入超过限制的仓位
        order = await oms.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            limit_price=100.0,  # 10000 / 100000 = 10% > 5%
        )

        assert order.status == OrderStatus.REJECTED

        await oms.stop()

    @pytest.mark.asyncio
    async def test_portfolio_summary(self, oms):
        """测试组合摘要"""
        await oms.start()

        # 提交并成交订单
        order = await oms.submit_order(symbol="AAPL", side="buy", quantity=100)
        oms.order_manager.accept_order(order.order_id)
        await oms.execute_fill(order.order_id, 100, 150.0)

        # 更新市价
        oms.update_market_prices({'AAPL': 160.0})

        summary = oms.get_portfolio_summary()

        assert summary['position_count'] == 1
        assert summary['total_exposure'] == 16000.0  # 100 * 160
        assert summary['unrealized_pnl'] == 1000.0  # (160-150) * 100

        await oms.stop()

    @pytest.mark.asyncio
    async def test_callback_on_fill(self, oms):
        """测试成交回调"""
        filled_orders = []

        def on_fill(order, fill):
            filled_orders.append((order, fill))

        oms.register_callback('on_order_filled', on_fill)

        await oms.start()

        order = await oms.submit_order(symbol="AAPL", side="buy", quantity=100)
        oms.order_manager.accept_order(order.order_id)
        await oms.execute_fill(order.order_id, 100, 150.0)

        assert len(filled_orders) == 1
        assert filled_orders[0][0].order_id == order.order_id

        await oms.stop()


# ============ Factory Functions Tests ============

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_oms(self):
        """测试创建OMS"""
        oms = create_oms(
            enable_risk_checks=True,
            max_position_pct=0.15,
            commission_rate=0.002,
        )

        assert oms.config.enable_risk_checks is True
        assert oms.config.default_commission_rate == 0.002
        assert oms.risk_checker.limits.max_position_pct == 0.15

    def test_create_order(self):
        """测试创建订单对象"""
        order = create_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="limit",
            limit_price=150.0,
        )

        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.0


# ============ Edge Cases Tests ============

class TestEdgeCases:
    """边界情况测试"""

    def test_fill_more_than_remaining(self, order_manager):
        """测试成交超过剩余数量应抛出异常"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order_manager.submit_order(order.order_id)
        order_manager.accept_order(order.order_id)

        # 尝试成交150，应该抛出异常（事务保护）
        with pytest.raises(ValueError, match="exceeds remaining quantity"):
            order_manager.fill_order(
                order.order_id,
                fill_quantity=150,
                fill_price=150.0,
            )

    def test_cancel_filled_order_fails(self, order_manager):
        """测试取消已成交订单"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order_manager.submit_order(order.order_id)
        order_manager.accept_order(order.order_id)
        order_manager.fill_order(order.order_id, 100, 150.0)

        with pytest.raises(ValueError, match="already complete"):
            order_manager.cancel_order(order.order_id)

    def test_modify_filled_order_fails(self, order_manager):
        """测试修改已成交订单"""
        order = order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order_manager.submit_order(order.order_id)
        order_manager.accept_order(order.order_id)
        order_manager.fill_order(order.order_id, 100, 150.0)

        with pytest.raises(ValueError, match="cannot be modified"):
            order_manager.modify_order(order.order_id, quantity=200)

    def test_reverse_position(self, position_manager):
        """测试反向开仓"""
        # 开多仓 100
        fill1 = Fill(
            fill_id="F1", order_id="O1", symbol="AAPL",
            side=OrderSide.BUY, quantity=100, price=150.0, commission=0
        )
        position_manager.update_position(fill1)

        # 卖出 150（平100多 + 开50空）
        fill2 = Fill(
            fill_id="F2", order_id="O2", symbol="AAPL",
            side=OrderSide.SELL, quantity=150, price=160.0, commission=0
        )
        position = position_manager.update_position(fill2)

        assert position.side == PositionSide.SHORT
        assert position.quantity == 50
        assert position.average_price == 160.0


# ============ Run Tests ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
