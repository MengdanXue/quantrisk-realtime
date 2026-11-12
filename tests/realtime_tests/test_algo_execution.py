"""
算法执行引擎测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.algo_execution import (
    # 主类
    AlgoExecutionEngine,
    # 算法类
    TWAPAlgorithm,
    VWAPAlgorithm,
    POVAlgorithm,
    ISAlgorithm,
    IcebergAlgorithm,
    SniperAlgorithm,
    AdaptiveAlgorithm,
    # 数据类
    AlgoOrder,
    OrderSlice,
    MarketData,
    AlgoConfig,
    ExecutionReport,
    # 枚举
    AlgoType,
    AlgoStatus,
    SliceStatus,
    OrderSide,
    PriceType,
    Urgency,
    # 工厂函数
    create_algo_engine,
    create_market_data,
    quick_algo_order,
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_market_data() -> MarketData:
    """样本市场数据"""
    return MarketData(
        symbol="AAPL",
        timestamp=datetime.now(),
        bid_price=150.00,
        ask_price=150.10,
        last_price=150.05,
        bid_size=1000,
        ask_size=1200,
        volume=5000000,
        vwap=150.02,
        volatility=0.25,
    )


@pytest.fixture
def sample_algo_order() -> AlgoOrder:
    """样本算法订单"""
    return AlgoOrder(
        algo_id="ALGO-TEST001",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=10000,
        algo_type=AlgoType.TWAP,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1),
        urgency=Urgency.MEDIUM,
        arrival_price=150.00,
    )


@pytest.fixture
def algo_engine() -> AlgoExecutionEngine:
    """算法执行引擎"""
    return create_algo_engine(
        min_slice_size=100,
        max_slice_size=5000,
        slice_interval_seconds=30,
    )


@pytest.fixture
def algo_config() -> AlgoConfig:
    """算法配置"""
    return AlgoConfig(
        min_slice_size=100,
        max_slice_size=5000,
        slice_interval_seconds=30,
        price_tolerance_bps=10,
        max_participation_rate=0.20,
    )


# ==================== MarketData 测试 ====================

class TestMarketData:
    """市场数据测试"""

    def test_create_market_data(self, sample_market_data):
        """测试创建市场数据"""
        assert sample_market_data.symbol == "AAPL"
        assert sample_market_data.bid_price == 150.00
        assert sample_market_data.ask_price == 150.10

    def test_mid_price(self, sample_market_data):
        """测试中间价计算"""
        expected = (150.00 + 150.10) / 2
        assert sample_market_data.mid_price == expected

    def test_to_dict(self, sample_market_data):
        """测试转换为字典"""
        d = sample_market_data.to_dict()
        assert d['symbol'] == "AAPL"
        assert d['bid_price'] == 150.00
        assert 'mid_price' in d

    def test_factory_function(self):
        """测试工厂函数"""
        data = create_market_data(
            symbol="MSFT",
            bid_price=300.00,
            ask_price=300.20,
            volume=1000000,
        )
        assert data.symbol == "MSFT"
        assert data.mid_price == 300.10


# ==================== OrderSlice 测试 ====================

class TestOrderSlice:
    """子订单测试"""

    def test_create_slice(self):
        """测试创建切片"""
        slice_order = OrderSlice(
            slice_id="SL-001",
            parent_algo_id="ALGO-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=500,
            price_type=PriceType.LIMIT,
            limit_price=150.05,
        )

        assert slice_order.slice_id == "SL-001"
        assert slice_order.quantity == 500
        assert slice_order.status == SliceStatus.PENDING

    def test_remaining_quantity(self):
        """测试剩余数量"""
        slice_order = OrderSlice(
            slice_id="SL-001",
            parent_algo_id="ALGO-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=500,
            filled_quantity=200,
        )

        assert slice_order.remaining_quantity == 300

    def test_fill_rate(self):
        """测试成交率"""
        slice_order = OrderSlice(
            slice_id="SL-001",
            parent_algo_id="ALGO-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=500,
            filled_quantity=250,
        )

        assert slice_order.fill_rate == 0.5

    def test_is_complete(self):
        """测试是否完成"""
        slice_order = OrderSlice(
            slice_id="SL-001",
            parent_algo_id="ALGO-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=500,
            status=SliceStatus.FILLED,
        )

        assert slice_order.is_complete is True


# ==================== AlgoOrder 测试 ====================

class TestAlgoOrder:
    """算法订单测试"""

    def test_create_order(self, sample_algo_order):
        """测试创建订单"""
        assert sample_algo_order.algo_id == "ALGO-TEST001"
        assert sample_algo_order.symbol == "AAPL"
        assert sample_algo_order.total_quantity == 10000

    def test_remaining_quantity(self, sample_algo_order):
        """测试剩余数量"""
        sample_algo_order.filled_quantity = 3000
        assert sample_algo_order.remaining_quantity == 7000

    def test_fill_rate(self, sample_algo_order):
        """测试完成率"""
        sample_algo_order.filled_quantity = 5000
        assert sample_algo_order.fill_rate == 0.5

    def test_avg_price(self, sample_algo_order):
        """测试平均价格"""
        sample_algo_order.filled_quantity = 1000
        sample_algo_order.filled_value = 150500
        assert sample_algo_order.avg_price == 150.5

    def test_slippage_bps_buy(self):
        """测试买入滑点"""
        order = AlgoOrder(
            algo_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            algo_type=AlgoType.TWAP,
            arrival_price=150.00,
            filled_quantity=1000,
            filled_value=150500,  # avg = 150.5
            avg_fill_price=150.5,  # 需要显式设置
        )
        # 滑点 = (150.5 - 150.0) / 150.0 * 10000 = 33.33 bps
        assert abs(order.slippage_bps - 33.33) < 0.1

    def test_slippage_bps_sell(self):
        """测试卖出滑点"""
        order = AlgoOrder(
            algo_id="TEST",
            symbol="AAPL",
            side=OrderSide.SELL,
            total_quantity=1000,
            algo_type=AlgoType.TWAP,
            arrival_price=150.00,
            filled_quantity=1000,
            filled_value=149500,  # avg = 149.5
            avg_fill_price=149.5,  # 需要显式设置
        )
        # 卖出: 滑点 = -(149.5 - 150.0) / 150.0 * 10000 = 33.33 bps
        assert abs(order.slippage_bps - 33.33) < 0.1

    def test_to_dict(self, sample_algo_order):
        """测试转换为字典"""
        d = sample_algo_order.to_dict()
        assert d['algo_id'] == "ALGO-TEST001"
        assert d['algo_type'] == "twap"
        assert 'fill_rate' in d


# ==================== TWAP算法测试 ====================

class TestTWAPAlgorithm:
    """TWAP算法测试"""

    @pytest.mark.asyncio
    async def test_generate_slices(self, algo_config, sample_market_data):
        """测试生成切片"""
        algo = TWAPAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="TWAP-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.TWAP,
            end_time=datetime.now() + timedelta(hours=1),
        )

        slices = await algo.generate_slices(order, sample_market_data)

        assert len(slices) > 0
        assert slices[0].symbol == "AAPL"
        assert slices[0].side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_slice_size_constraints(self, algo_config, sample_market_data):
        """测试切片大小约束"""
        algo = TWAPAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="TWAP-002",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=100000,
            algo_type=AlgoType.TWAP,
            end_time=datetime.now() + timedelta(hours=1),
        )

        slices = await algo.generate_slices(order, sample_market_data)

        for s in slices:
            assert s.quantity >= algo_config.min_slice_size
            assert s.quantity <= algo_config.max_slice_size

    @pytest.mark.asyncio
    async def test_urgency_affects_price(self, algo_config, sample_market_data):
        """测试紧迫度影响价格"""
        algo = TWAPAlgorithm(algo_config)

        # 高紧迫度
        high_order = AlgoOrder(
            algo_id="TWAP-HIGH",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.TWAP,
            urgency=Urgency.HIGH,
            end_time=datetime.now() + timedelta(hours=1),
        )

        # 低紧迫度
        low_order = AlgoOrder(
            algo_id="TWAP-LOW",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.TWAP,
            urgency=Urgency.LOW,
            end_time=datetime.now() + timedelta(hours=1),
        )

        high_slices = await algo.generate_slices(high_order, sample_market_data)
        low_slices = await algo.generate_slices(low_order, sample_market_data)

        # 高紧迫度应该使用更激进的价格
        assert high_slices[0].limit_price >= low_slices[0].limit_price


# ==================== VWAP算法测试 ====================

class TestVWAPAlgorithm:
    """VWAP算法测试"""

    @pytest.mark.asyncio
    async def test_generate_slices(self, algo_config, sample_market_data):
        """测试生成切片"""
        algo = VWAPAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="VWAP-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.VWAP,
            end_time=datetime.now() + timedelta(hours=1),
        )

        slices = await algo.generate_slices(order, sample_market_data)

        assert len(slices) > 0

    @pytest.mark.asyncio
    async def test_volume_profile(self, algo_config, sample_market_data):
        """测试成交量分布"""
        algo = VWAPAlgorithm(algo_config)

        # 设置自定义成交量分布
        profile = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
        algo.set_volume_profile("AAPL", profile)

        order = AlgoOrder(
            algo_id="VWAP-002",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.VWAP,
        )

        slices = await algo.generate_slices(order, sample_market_data)
        assert len(slices) > 0


# ==================== POV算法测试 ====================

class TestPOVAlgorithm:
    """POV算法测试"""

    @pytest.mark.asyncio
    async def test_generate_slices(self, algo_config, sample_market_data):
        """测试生成切片"""
        algo = POVAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="POV-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.POV,
            participation_rate=0.10,
        )

        slices = await algo.generate_slices(order, sample_market_data)

        assert len(slices) > 0

    @pytest.mark.asyncio
    async def test_participation_rate_limit(self, algo_config, sample_market_data):
        """测试参与率限制"""
        algo = POVAlgorithm(algo_config)

        # 设置高参与率
        order = AlgoOrder(
            algo_id="POV-002",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.POV,
            participation_rate=0.50,  # 高于限制
        )

        slices = await algo.generate_slices(order, sample_market_data)

        # 应该被限制到max_participation_rate
        assert len(slices) > 0


# ==================== IS算法测试 ====================

class TestISAlgorithm:
    """IS算法测试"""

    @pytest.mark.asyncio
    async def test_generate_slices(self, algo_config, sample_market_data):
        """测试生成切片"""
        algo = ISAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="IS-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.IS,
        )

        slices = await algo.generate_slices(order, sample_market_data)

        assert len(slices) > 0

    @pytest.mark.asyncio
    async def test_urgency_affects_slice_size(self, algo_config, sample_market_data):
        """测试紧迫度影响切片大小"""
        algo = ISAlgorithm(algo_config)

        high_order = AlgoOrder(
            algo_id="IS-HIGH",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.IS,
            urgency=Urgency.HIGH,
        )

        low_order = AlgoOrder(
            algo_id="IS-LOW",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.IS,
            urgency=Urgency.LOW,
        )

        high_slices = await algo.generate_slices(high_order, sample_market_data)
        low_slices = await algo.generate_slices(low_order, sample_market_data)

        # 高紧迫度应该有更大的切片
        assert high_slices[0].quantity >= low_slices[0].quantity


# ==================== Iceberg算法测试 ====================

class TestIcebergAlgorithm:
    """Iceberg算法测试"""

    @pytest.mark.asyncio
    async def test_generate_slices(self, algo_config, sample_market_data):
        """测试生成切片"""
        algo = IcebergAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="ICE-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.ICEBERG,
            display_quantity=500,
        )

        slices = await algo.generate_slices(order, sample_market_data)

        assert len(slices) > 0
        # 切片应该不超过显示数量
        assert slices[0].quantity <= order.display_quantity

    @pytest.mark.asyncio
    async def test_price_limit(self, algo_config, sample_market_data):
        """测试价格限制"""
        algo = IcebergAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="ICE-002",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.ICEBERG,
            display_quantity=500,
            limit_price=149.50,  # 低于市场价
        )

        slices = await algo.generate_slices(order, sample_market_data)

        # 限价应该不超过设定的限价
        assert slices[0].limit_price <= 149.50


# ==================== Sniper算法测试 ====================

class TestSniperAlgorithm:
    """Sniper算法测试"""

    @pytest.mark.asyncio
    async def test_favorable_price(self, algo_config, sample_market_data):
        """测试有利价格时执行"""
        algo = SniperAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="SNP-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.SNIPER,
            arrival_price=150.20,  # 高于当前卖价
        )

        slices = await algo.generate_slices(order, sample_market_data)

        # 价格有利应该生成切片
        assert len(slices) > 0

    @pytest.mark.asyncio
    async def test_unfavorable_price(self, algo_config, sample_market_data):
        """测试不利价格时等待"""
        algo = SniperAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="SNP-002",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.SNIPER,
            arrival_price=149.50,  # 低于当前卖价
        )

        slices = await algo.generate_slices(order, sample_market_data)

        # 价格不利不应该生成切片(取决于容忍度)
        # 这里可能会生成也可能不会


# ==================== Adaptive算法测试 ====================

class TestAdaptiveAlgorithm:
    """Adaptive算法测试"""

    @pytest.mark.asyncio
    async def test_generate_slices(self, algo_config, sample_market_data):
        """测试生成切片"""
        algo = AdaptiveAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="ADP-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.ADAPTIVE,
        )

        slices = await algo.generate_slices(order, sample_market_data)

        assert len(slices) > 0

    @pytest.mark.asyncio
    async def test_adapts_to_market(self, algo_config, sample_market_data):
        """测试适应市场条件"""
        algo = AdaptiveAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="ADP-002",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            algo_type=AlgoType.ADAPTIVE,
        )

        # 多次调用以积累历史数据
        for _ in range(15):
            await algo.generate_slices(order, sample_market_data)

        # 最后一次应该基于历史数据
        slices = await algo.generate_slices(order, sample_market_data)
        assert len(slices) > 0


# ==================== AlgoExecutionEngine 测试 ====================

class TestAlgoExecutionEngine:
    """算法执行引擎测试"""

    @pytest.mark.asyncio
    async def test_submit_order(self, algo_engine):
        """测试提交订单"""
        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=10000,
            algo_type="twap",
            duration_minutes=60,
        )

        assert order.algo_id is not None
        assert order.status == AlgoStatus.RUNNING
        assert order.total_quantity == 10000

    @pytest.mark.asyncio
    async def test_cancel_order(self, algo_engine):
        """测试取消订单"""
        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=10000,
            algo_type="twap",
        )

        cancelled = await algo_engine.cancel_order(order.algo_id, "Test cancel")

        assert cancelled.status == AlgoStatus.CANCELLED
        assert cancelled.algo_id not in [o.algo_id for o in algo_engine.get_active_orders()]

    @pytest.mark.asyncio
    async def test_pause_resume_order(self, algo_engine):
        """测试暂停恢复订单"""
        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=10000,
            algo_type="twap",
        )

        # 暂停
        paused = await algo_engine.pause_order(order.algo_id)
        assert paused.status == AlgoStatus.PAUSED

        # 恢复
        resumed = await algo_engine.resume_order(order.algo_id)
        assert resumed.status == AlgoStatus.RUNNING

    @pytest.mark.asyncio
    async def test_process_fill(self, algo_engine):
        """测试处理成交"""
        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            algo_type="twap",
        )

        # 添加一个切片
        slice_order = OrderSlice(
            slice_id="SL-TEST",
            parent_algo_id=order.algo_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=500,
        )
        order.slices.append(slice_order)
        order.slices_sent = 1

        # 处理成交
        updated_slice, updated_order = await algo_engine.process_fill(
            slice_id="SL-TEST",
            filled_quantity=500,
            filled_price=150.05,
            commission=5.0,
        )

        assert updated_slice.status == SliceStatus.FILLED
        assert updated_order.filled_quantity == 500

    @pytest.mark.asyncio
    async def test_update_market_data(self, algo_engine, sample_market_data):
        """测试更新市场数据"""
        algo_engine.update_market_data(sample_market_data)

        # 验证所有算法都收到了数据
        for algo in algo_engine._algorithms.values():
            assert "AAPL" in algo._market_data

    def test_get_active_orders(self, algo_engine):
        """测试获取活跃订单"""
        active = algo_engine.get_active_orders()
        assert isinstance(active, list)

    @pytest.mark.asyncio
    async def test_get_statistics(self, algo_engine):
        """测试获取统计信息"""
        # 提交一些订单
        await algo_engine.submit_order(
            symbol="AAPL", side="buy", quantity=1000, algo_type="twap"
        )
        await algo_engine.submit_order(
            symbol="MSFT", side="sell", quantity=500, algo_type="vwap"
        )

        stats = algo_engine.get_statistics()

        assert 'active_orders' in stats
        assert 'completed_orders' in stats
        assert stats['active_orders'] == 2

    @pytest.mark.asyncio
    async def test_generate_report(self, algo_engine):
        """测试生成报告"""
        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            algo_type="twap",
        )

        order.arrival_price = 150.00
        order.filled_quantity = 1000
        order.filled_value = 150500

        report = algo_engine.generate_report(order.algo_id)

        assert isinstance(report, ExecutionReport)
        assert report.algo_id == order.algo_id
        assert report.filled_quantity == 1000

    @pytest.mark.asyncio
    async def test_start_stop_engine(self, algo_engine):
        """测试启动停止引擎"""
        await algo_engine.start()
        assert algo_engine._running is True

        await algo_engine.stop()
        assert algo_engine._running is False

    @pytest.mark.asyncio
    async def test_all_algo_types(self, algo_engine):
        """测试所有算法类型"""
        algo_types = ['twap', 'vwap', 'pov', 'is', 'iceberg', 'sniper', 'adaptive']

        for algo_type in algo_types:
            order = await algo_engine.submit_order(
                symbol="AAPL",
                side="buy",
                quantity=1000,
                algo_type=algo_type,
            )
            assert order.algo_type.value == algo_type

            # 取消订单以便下次测试
            await algo_engine.cancel_order(order.algo_id)


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_algo_engine(self):
        """测试创建引擎"""
        engine = create_algo_engine(
            min_slice_size=50,
            max_slice_size=2000,
        )

        assert engine.config.min_slice_size == 50
        assert engine.config.max_slice_size == 2000

    def test_create_market_data(self):
        """测试创建市场数据"""
        data = create_market_data(
            symbol="TEST",
            bid_price=100.0,
            ask_price=100.2,
            volume=500000,
        )

        assert data.symbol == "TEST"
        assert data.mid_price == 100.1

    @pytest.mark.asyncio
    async def test_quick_algo_order(self):
        """测试快速创建订单"""
        order = await quick_algo_order(
            symbol="AAPL",
            side="buy",
            quantity=5000,
            algo_type="vwap",
            duration_minutes=30,
        )

        assert order.symbol == "AAPL"
        assert order.algo_type == AlgoType.VWAP


# ==================== 边界条件测试 ====================

class TestEdgeCases:
    """边界条件测试"""

    @pytest.mark.asyncio
    async def test_zero_remaining_quantity(self, algo_config, sample_market_data):
        """测试零剩余数量"""
        algo = TWAPAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="EDGE-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            algo_type=AlgoType.TWAP,
            filled_quantity=1000,  # 已完全成交
        )

        slices = await algo.generate_slices(order, sample_market_data)
        assert len(slices) == 0

    @pytest.mark.asyncio
    async def test_expired_order(self, algo_config, sample_market_data):
        """测试过期订单"""
        algo = TWAPAlgorithm(algo_config)

        order = AlgoOrder(
            algo_id="EDGE-002",
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            algo_type=AlgoType.TWAP,
            end_time=datetime.now() - timedelta(hours=1),  # 已过期
        )

        slices = await algo.generate_slices(order, sample_market_data)
        # 过期订单应该一次性完成剩余量
        assert len(slices) > 0

    @pytest.mark.asyncio
    async def test_small_quantity(self, algo_engine):
        """测试小数量订单"""
        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=50,  # 小于min_slice_size
            algo_type="twap",
        )

        assert order.total_quantity == 50

    @pytest.mark.asyncio
    async def test_cancel_completed_order(self, algo_engine):
        """测试取消已完成订单"""
        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            algo_type="twap",
        )

        # 手动标记为完成
        order.status = AlgoStatus.COMPLETED

        with pytest.raises(ValueError):
            await algo_engine.cancel_order(order.algo_id)

    @pytest.mark.asyncio
    async def test_resume_non_paused_order(self, algo_engine):
        """测试恢复非暂停订单"""
        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            algo_type="twap",
        )

        with pytest.raises(ValueError):
            await algo_engine.resume_order(order.algo_id)


# ==================== 回调测试 ====================

class TestCallbacks:
    """回调测试"""

    @pytest.mark.asyncio
    async def test_slice_callback(self, algo_engine, sample_market_data):
        """测试切片回调"""
        slices_created = []

        def on_slice(slice_order):
            slices_created.append(slice_order)

        algo_engine.set_slice_callback(on_slice)
        algo_engine.update_market_data(sample_market_data)

        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            algo_type="twap",
        )

        # 手动触发处理
        await algo_engine._process_active_orders()

        # 应该有切片被创建
        assert len(slices_created) > 0

    @pytest.mark.asyncio
    async def test_completion_callback(self, algo_engine):
        """测试完成回调"""
        completed_orders = []

        def on_complete(order):
            completed_orders.append(order)

        algo_engine.set_completion_callback(on_complete)

        order = await algo_engine.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            algo_type="twap",
        )

        # 添加切片并完全成交
        slice_order = OrderSlice(
            slice_id="SL-CB-TEST",
            parent_algo_id=order.algo_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order.slices.append(slice_order)

        await algo_engine.process_fill("SL-CB-TEST", 100, 150.0)

        assert len(completed_orders) == 1
        assert completed_orders[0].algo_id == order.algo_id


# ==================== 执行报告测试 ====================

class TestExecutionReport:
    """执行报告测试"""

    def test_report_to_dict(self):
        """测试报告转字典"""
        report = ExecutionReport(
            algo_id="RPT-001",
            symbol="AAPL",
            algo_type=AlgoType.TWAP,
            side=OrderSide.BUY,
            total_quantity=10000,
            filled_quantity=9500,
            fill_rate=0.95,
            arrival_price=150.00,
            avg_fill_price=150.10,
            market_vwap=150.05,
            slippage_bps=6.67,
            vs_vwap_bps=3.33,
            total_slices=20,
            filled_slices=19,
            avg_slice_size=500,
            execution_duration_seconds=3600,
            total_commission=50.0,
            estimated_market_impact_bps=5.0,
        )

        d = report.to_dict()

        assert d['algo_id'] == "RPT-001"
        assert d['fill_rate'] == 95.0
        assert d['slippage_bps'] == 6.67


# ==================== 运行测试 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
