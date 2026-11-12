"""
滑点追踪器测试
Test Slippage Tracker
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from quantrisk_realtime.realtime.slippage_tracker import (
    # 枚举
    SlippageType,
    AttributionCategory,
    OrderSide,
    OrderType,
    MarketCondition,
    AlertLevel,
    TrackerState,
    # 数据类
    OrderRecord,
    SlippageResult,
    SlippageStatistics,
    SlippagePredict,
    SlippageAttribution,
    OptimizationRecommendation,
    SlippageAlert,
    TrackerConfig,
    # 数据提供者
    OrderBookProvider,
    MarketDataProvider,
    HistoricalDataProvider,
    SimulatedOrderBookProvider,
    SimulatedMarketDataProvider,
    SimulatedHistoricalDataProvider,
    # 主类
    SlippageTracker,
    # 工厂函数
    create_slippage_tracker,
    create_order_record,
    quick_slippage_analysis,
)


# ==================== 枚举测试 ====================

class TestEnums:
    """枚举测试"""

    def test_slippage_type_values(self):
        """测试滑点类型枚举值"""
        assert SlippageType.PRICE.value == "price"
        assert SlippageType.TIMING.value == "timing"
        assert SlippageType.MARKET_IMPACT.value == "market_impact"
        assert SlippageType.TOTAL.value == "total"

    def test_attribution_category_values(self):
        """测试归因类别枚举值"""
        assert AttributionCategory.EXECUTION_DELAY.value == "execution_delay"
        assert AttributionCategory.MARKET_CONDITION.value == "market_condition"
        assert AttributionCategory.ORDER_TYPE.value == "order_type"
        assert AttributionCategory.LIQUIDITY.value == "liquidity"

    def test_order_side_values(self):
        """测试订单方向枚举值"""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_type_values(self):
        """测试订单类型枚举值"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"

    def test_market_condition_values(self):
        """测试市场状态枚举值"""
        assert MarketCondition.NORMAL.value == "normal"
        assert MarketCondition.VOLATILE.value == "volatile"
        assert MarketCondition.ILLIQUID.value == "illiquid"

    def test_alert_level_values(self):
        """测试警报级别枚举值"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"

    def test_tracker_state_values(self):
        """测试追踪器状态枚举值"""
        assert TrackerState.IDLE.value == "idle"
        assert TrackerState.RUNNING.value == "running"


# ==================== 数据类测试 ====================

class TestOrderRecord:
    """订单记录测试"""

    def test_create_order_record(self):
        """测试创建订单记录"""
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        assert order.order_id == "ORD001"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100.0

    def test_spread_bps_calculation(self):
        """测试价差计算"""
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            mid_price=100.0,
            spread=0.10,
        )

        assert order.spread_bps == 10.0  # 0.10/100 * 10000

    def test_order_size_pct_calculation(self):
        """测试订单大小占比计算"""
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000.0,
            volume=100000.0,
        )

        assert order.order_size_pct == 0.01  # 1000/100000

    def test_execution_delay_calculation(self):
        """测试执行延迟计算"""
        decision_time = datetime.now()
        execution_time = decision_time + timedelta(seconds=5)

        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            decision_time=decision_time,
            execution_time=execution_time,
        )

        assert order.execution_delay_seconds == 5.0

    def test_to_dict(self):
        """测试转换为字典"""
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        data = order.to_dict()
        assert data['order_id'] == "ORD001"
        assert data['symbol'] == "AAPL"
        assert data['side'] == "buy"


class TestSlippageResult:
    """滑点结果测试"""

    def test_create_slippage_result(self):
        """测试创建滑点结果"""
        result = SlippageResult(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            price_slippage=5.0,
            timing_slippage=2.0,
            market_impact_slippage=3.0,
            total_slippage=10.0,
            slippage_cost=100.0,
            execution_quality_score=85.0,
        )

        assert result.total_slippage == 10.0
        assert result.slippage_cost == 100.0
        assert result.execution_quality_score == 85.0

    def test_to_dict(self):
        """测试转换为字典"""
        result = SlippageResult(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            price_slippage=5.0,
            timing_slippage=2.0,
            market_impact_slippage=3.0,
            total_slippage=10.0,
            slippage_cost=100.0,
            execution_quality_score=85.0,
        )

        data = result.to_dict()
        assert data['order_id'] == "ORD001"
        assert data['total_slippage'] == 10.0


class TestSlippageStatistics:
    """滑点统计测试"""

    def test_create_statistics(self):
        """测试创建统计"""
        stats = SlippageStatistics(
            dimension="symbol",
            dimension_value="AAPL",
            total_orders=100,
            total_notional=1000000.0,
            avg_price_slippage=5.0,
            avg_timing_slippage=2.0,
            avg_market_impact_slippage=3.0,
            avg_total_slippage=10.0,
            median_slippage=9.0,
            std_slippage=2.5,
            max_slippage=20.0,
            min_slippage=2.0,
            buy_avg_slippage=11.0,
            sell_avg_slippage=9.0,
            total_slippage_cost=10000.0,
        )

        assert stats.dimension == "symbol"
        assert stats.total_orders == 100
        assert stats.avg_total_slippage == 10.0

    def test_to_dict(self):
        """测试转换为字典"""
        stats = SlippageStatistics(
            dimension="symbol",
            dimension_value="AAPL",
            total_orders=100,
            total_notional=1000000.0,
            avg_price_slippage=5.0,
            avg_timing_slippage=2.0,
            avg_market_impact_slippage=3.0,
            avg_total_slippage=10.0,
            median_slippage=9.0,
            std_slippage=2.5,
            max_slippage=20.0,
            min_slippage=2.0,
            buy_avg_slippage=11.0,
            sell_avg_slippage=9.0,
            total_slippage_cost=10000.0,
        )

        data = stats.to_dict()
        assert data['dimension'] == "symbol"
        assert data['total_orders'] == 100


class TestSlippagePredict:
    """滑点预测测试"""

    def test_create_prediction(self):
        """测试创建预测"""
        prediction = SlippagePredict(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000.0,
            predicted_slippage=8.5,
            confidence=0.75,
            slippage_low=6.0,
            slippage_high=11.0,
            prediction_method="weighted_ensemble",
            historical_avg=8.0,
            orderbook_estimate=9.0,
            volatility_estimate=8.5,
            risk_level="low",
        )

        assert prediction.predicted_slippage == 8.5
        assert prediction.confidence == 0.75
        assert prediction.risk_level == "low"

    def test_to_dict(self):
        """测试转换为字典"""
        prediction = SlippagePredict(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000.0,
            predicted_slippage=8.5,
            confidence=0.75,
            slippage_low=6.0,
            slippage_high=11.0,
            prediction_method="weighted_ensemble",
            historical_avg=8.0,
            orderbook_estimate=9.0,
            volatility_estimate=8.5,
            risk_level="low",
        )

        data = prediction.to_dict()
        assert data['predicted_slippage'] == 8.5


class TestSlippageAttribution:
    """滑点归因测试"""

    def test_create_attribution(self):
        """测试创建归因"""
        attribution = SlippageAttribution(
            order_id="ORD001",
            symbol="AAPL",
            total_slippage=10.0,
            execution_delay_contribution=0.25,
            market_condition_contribution=0.20,
            order_type_contribution=0.15,
            liquidity_contribution=0.15,
            order_size_contribution=0.10,
            time_of_day_contribution=0.10,
            volatility_contribution=0.05,
        )

        assert attribution.total_slippage == 10.0
        total_contrib = (
            attribution.execution_delay_contribution +
            attribution.market_condition_contribution +
            attribution.order_type_contribution +
            attribution.liquidity_contribution +
            attribution.order_size_contribution +
            attribution.time_of_day_contribution +
            attribution.volatility_contribution
        )
        assert abs(total_contrib - 1.0) < 0.01

    def test_to_dict(self):
        """测试转换为字典"""
        attribution = SlippageAttribution(
            order_id="ORD001",
            symbol="AAPL",
            total_slippage=10.0,
            execution_delay_contribution=0.25,
            market_condition_contribution=0.20,
            order_type_contribution=0.15,
            liquidity_contribution=0.15,
            order_size_contribution=0.10,
            time_of_day_contribution=0.10,
            volatility_contribution=0.05,
        )

        data = attribution.to_dict()
        assert data['order_id'] == "ORD001"


class TestOptimizationRecommendation:
    """优化建议测试"""

    def test_create_recommendation(self):
        """测试创建建议"""
        rec = OptimizationRecommendation(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10000.0,
            recommendation_type="comprehensive",
            best_execution_time="10:30-11:30",
            time_reasoning="Avoid opening volatility",
            suggested_split=5,
            split_interval_seconds=300.0,
            split_reasoning="Large order, split to reduce impact",
            suggested_order_type=OrderType.LIMIT,
            limit_price=149.50,
            order_type_reasoning="High slippage expected",
            expected_slippage_reduction=5.0,
            expected_cost_saving=500.0,
        )

        assert rec.suggested_split == 5
        assert rec.expected_slippage_reduction == 5.0

    def test_to_dict(self):
        """测试转换为字典"""
        rec = OptimizationRecommendation(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10000.0,
            recommendation_type="comprehensive",
            best_execution_time="10:30-11:30",
            time_reasoning="Avoid opening volatility",
            suggested_split=5,
            split_interval_seconds=300.0,
            split_reasoning="Large order, split to reduce impact",
            suggested_order_type=OrderType.LIMIT,
            limit_price=149.50,
            order_type_reasoning="High slippage expected",
            expected_slippage_reduction=5.0,
            expected_cost_saving=500.0,
        )

        data = rec.to_dict()
        assert data['suggested_split'] == 5


class TestSlippageAlert:
    """滑点警报测试"""

    def test_create_alert(self):
        """测试创建警报"""
        alert = SlippageAlert(
            alert_id="ALERT001",
            order_id="ORD001",
            symbol="AAPL",
            level=AlertLevel.WARNING,
            message="High slippage detected",
            actual_slippage=15.0,
            expected_slippage=5.0,
            threshold=10.0,
            recommendation="Consider splitting order",
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.actual_slippage == 15.0

    def test_to_dict(self):
        """测试转换为字典"""
        alert = SlippageAlert(
            alert_id="ALERT001",
            order_id="ORD001",
            symbol="AAPL",
            level=AlertLevel.WARNING,
            message="High slippage detected",
            actual_slippage=15.0,
            expected_slippage=5.0,
            threshold=10.0,
            recommendation="Consider splitting order",
        )

        data = alert.to_dict()
        assert data['alert_id'] == "ALERT001"


class TestTrackerConfig:
    """追踪器配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = TrackerConfig()

        assert config.warning_slippage_bps == 10.0
        assert config.critical_slippage_bps == 30.0
        assert config.history_window_days == 30

    def test_custom_config(self):
        """测试自定义配置"""
        config = TrackerConfig(
            warning_slippage_bps=15.0,
            critical_slippage_bps=40.0,
            history_window_days=60,
        )

        assert config.warning_slippage_bps == 15.0
        assert config.critical_slippage_bps == 40.0
        assert config.history_window_days == 60


# ==================== 数据提供者测试 ====================

class TestSimulatedOrderBookProvider:
    """模拟订单簿提供者测试"""

    @pytest.mark.asyncio
    async def test_default_orderbook(self):
        """测试默认订单簿"""
        provider = SimulatedOrderBookProvider()
        orderbook = await provider.get_orderbook("AAPL", depth=5)

        assert 'bids' in orderbook
        assert 'asks' in orderbook
        assert len(orderbook['bids']) == 5
        assert len(orderbook['asks']) == 5

    @pytest.mark.asyncio
    async def test_set_orderbook(self):
        """测试设置订单簿"""
        provider = SimulatedOrderBookProvider()
        bids = [(100.0, 100), (99.99, 200)]
        asks = [(100.01, 100), (100.02, 200)]
        provider.set_orderbook("AAPL", bids, asks)

        orderbook = await provider.get_orderbook("AAPL", depth=5)
        assert orderbook['bids'] == bids
        assert orderbook['asks'] == asks


class TestSimulatedMarketDataProvider:
    """模拟市场数据提供者测试"""

    @pytest.mark.asyncio
    async def test_default_market_data(self):
        """测试默认市场数据"""
        provider = SimulatedMarketDataProvider()
        data = await provider.get_market_data("AAPL")

        assert 'bid' in data
        assert 'ask' in data
        assert 'last' in data
        assert 'volume' in data

    @pytest.mark.asyncio
    async def test_set_market_data(self):
        """测试设置市场数据"""
        provider = SimulatedMarketDataProvider()
        custom_data = {
            'bid': 150.0,
            'ask': 150.1,
            'last': 150.05,
            'volume': 2000000,
            'volatility': 0.25,
        }
        provider.set_market_data("AAPL", custom_data)

        data = await provider.get_market_data("AAPL")
        assert data['bid'] == 150.0
        assert data['volume'] == 2000000


class TestSimulatedHistoricalDataProvider:
    """模拟历史数据提供者测试"""

    @pytest.mark.asyncio
    async def test_default_historical_data(self):
        """测试默认历史数据"""
        provider = SimulatedHistoricalDataProvider()
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        data = await provider.get_historical_slippage("AAPL", start_date, end_date)
        assert len(data) == 30  # 30 days

    @pytest.mark.asyncio
    async def test_add_historical_slippage(self):
        """测试添加历史滑点数据"""
        provider = SimulatedHistoricalDataProvider()
        provider.add_historical_slippage("AAPL", 5.0)
        provider.add_historical_slippage("AAPL", 7.0)

        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        data = await provider.get_historical_slippage("AAPL", start_date, end_date)

        assert data == [5.0, 7.0]


# ==================== 滑点追踪器测试 ====================

class TestSlippageTracker:
    """滑点追踪器测试"""

    def test_create_tracker(self):
        """测试创建追踪器"""
        tracker = SlippageTracker()

        assert tracker.state == TrackerState.IDLE
        assert tracker.config.warning_slippage_bps == 10.0

    def test_create_tracker_with_config(self):
        """测试使用配置创建追踪器"""
        config = TrackerConfig(warning_slippage_bps=20.0)
        tracker = SlippageTracker(config=config)

        assert tracker.config.warning_slippage_bps == 20.0

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启动和停止"""
        tracker = SlippageTracker()

        await tracker.start()
        assert tracker.state == TrackerState.RUNNING

        await tracker.stop()
        assert tracker.state == TrackerState.STOPPED

    @pytest.mark.asyncio
    async def test_add_order(self):
        """测试添加订单"""
        tracker = SlippageTracker()
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        await tracker.add_order(order)
        recent_orders = tracker.get_recent_orders(1)
        assert len(recent_orders) == 1
        assert recent_orders[0].order_id == "ORD001"

    @pytest.mark.asyncio
    async def test_calculate_price_slippage_buy(self):
        """测试计算买入价格滑点"""
        tracker = SlippageTracker()
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            arrival_price=100.0,
            execution_price=100.5,  # 执行价高于到达价，负滑点
        )

        result = await tracker.calculate_slippage(order)
        assert result.price_slippage > 0  # 买入时价格上涨是负滑点

    @pytest.mark.asyncio
    async def test_calculate_price_slippage_sell(self):
        """测试计算卖出价格滑点"""
        tracker = SlippageTracker()
        order = OrderRecord(
            order_id="ORD002",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=100.0,
            arrival_price=100.0,
            execution_price=99.5,  # 执行价低于到达价，负滑点
        )

        result = await tracker.calculate_slippage(order)
        assert result.price_slippage > 0  # 卖出时价格下跌是负滑点

    @pytest.mark.asyncio
    async def test_calculate_timing_slippage(self):
        """测试计算时间滑点"""
        tracker = SlippageTracker()
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            decision_price=100.0,
            arrival_price=100.2,
            execution_price=100.3,
        )

        result = await tracker.calculate_slippage(order)
        assert result.timing_slippage > 0

    @pytest.mark.asyncio
    async def test_calculate_market_impact_slippage(self):
        """测试计算市场影响滑点"""
        tracker = SlippageTracker()
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            volume=1000000.0,
            volatility=0.25,
            arrival_price=100.0,
            execution_price=100.0,
        )

        result = await tracker.calculate_slippage(order)
        assert result.market_impact_slippage > 0

    @pytest.mark.asyncio
    async def test_calculate_total_slippage(self):
        """测试计算总滑点"""
        tracker = SlippageTracker()
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000.0,
            decision_price=100.0,
            arrival_price=100.2,
            execution_price=100.5,
            volume=1000000.0,
            volatility=0.20,
        )

        result = await tracker.calculate_slippage(order)
        assert result.total_slippage > 0
        # 总滑点应该是各部分之和
        expected_total = (
            result.price_slippage +
            result.timing_slippage +
            result.market_impact_slippage
        )
        assert abs(result.total_slippage - expected_total) < 0.01

    @pytest.mark.asyncio
    async def test_calculate_slippage_cost(self):
        """测试计算滑点成本"""
        tracker = SlippageTracker()
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000.0,
            arrival_price=100.0,
            execution_price=100.1,
            volume=1000000.0,
            volatility=0.20,
        )

        result = await tracker.calculate_slippage(order)
        # 名义价值 = 1000 * 100.1 = 100100
        # 滑点成本 = 名义价值 * 滑点bps / 10000
        assert result.slippage_cost > 0

    @pytest.mark.asyncio
    async def test_execution_quality_score(self):
        """测试执行质量得分"""
        tracker = SlippageTracker()

        # 好的执行
        good_order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            decision_price=100.0,
            arrival_price=100.0,
            execution_price=100.0,
            decision_time=datetime.now(),
            execution_time=datetime.now() + timedelta(seconds=1),
            market_condition=MarketCondition.NORMAL,
        )

        result = await tracker.calculate_slippage(good_order)
        assert result.execution_quality_score > 90

    @pytest.mark.asyncio
    async def test_get_statistics_by_symbol(self):
        """测试按品种统计"""
        tracker = SlippageTracker()

        # 添加多个订单
        for i in range(10):
            order = OrderRecord(
                order_id=f"ORD{i:03d}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                arrival_price=100.0,
                execution_price=100.0 + i * 0.01,
                execution_time=datetime.now(),
            )
            await tracker.add_order(order)
            await tracker.calculate_slippage(order)

        stats = await tracker.get_statistics("symbol")
        assert "AAPL" in stats
        assert stats["AAPL"].total_orders == 10

    @pytest.mark.asyncio
    async def test_get_statistics_by_time(self):
        """测试按时间段统计"""
        tracker = SlippageTracker()

        # 添加不同时段的订单
        now = datetime.now()
        for hour in [10, 13, 15]:
            exec_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            order = OrderRecord(
                order_id=f"ORD_{hour}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                arrival_price=100.0,
                execution_price=100.1,
                execution_time=exec_time,
            )
            await tracker.add_order(order)
            await tracker.calculate_slippage(order)

        # Use date range that includes today
        start_date = now - timedelta(days=1)
        end_date = now + timedelta(days=1)
        stats = await tracker.get_statistics("time", start_date, end_date)
        # At least one time bucket should have orders
        assert len(stats) >= 1

    @pytest.mark.asyncio
    async def test_get_statistics_by_size(self):
        """测试按订单大小统计"""
        tracker = SlippageTracker()

        # 添加不同大小的订单
        sizes = [100, 5000, 50000]  # small, medium, large
        for i, size in enumerate(sizes):
            order = OrderRecord(
                order_id=f"ORD{i:03d}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=size,
                volume=1000000.0,
                arrival_price=100.0,
                execution_price=100.1,
                execution_time=datetime.now(),
            )
            await tracker.add_order(order)
            await tracker.calculate_slippage(order)

        stats = await tracker.get_statistics("size")
        assert "small" in stats or "medium" in stats or "large" in stats

    @pytest.mark.asyncio
    async def test_get_statistics_by_condition(self):
        """测试按市场状态统计"""
        tracker = SlippageTracker()

        # 添加不同市场状态的订单
        conditions = [MarketCondition.NORMAL, MarketCondition.VOLATILE, MarketCondition.ILLIQUID]
        for i, condition in enumerate(conditions):
            order = OrderRecord(
                order_id=f"ORD{i:03d}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                arrival_price=100.0,
                execution_price=100.1,
                execution_time=datetime.now(),
                market_condition=condition,
            )
            await tracker.add_order(order)
            await tracker.calculate_slippage(order)

        stats = await tracker.get_statistics("condition")
        assert MarketCondition.NORMAL.value in stats

    @pytest.mark.asyncio
    async def test_predict_slippage_from_history(self):
        """测试基于历史预测滑点"""
        tracker = SlippageTracker()

        # 添加历史订单
        for i in range(20):
            order = OrderRecord(
                order_id=f"ORD{i:03d}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                arrival_price=100.0,
                execution_price=100.05,
            )
            await tracker.add_order(order)
            await tracker.calculate_slippage(order)

        prediction = await tracker.predict_slippage("AAPL", OrderSide.BUY, 100.0)
        assert prediction.predicted_slippage > 0
        assert 0 <= prediction.confidence <= 1

    @pytest.mark.asyncio
    async def test_predict_slippage_orderbook(self):
        """测试基于订单簿预测滑点"""
        # 设置订单簿
        orderbook_provider = SimulatedOrderBookProvider()
        bids = [(100.0, 1000), (99.99, 2000)]
        asks = [(100.01, 1000), (100.02, 2000)]
        orderbook_provider.set_orderbook("AAPL", bids, asks)

        tracker = SlippageTracker(orderbook_provider=orderbook_provider)

        prediction = await tracker.predict_slippage("AAPL", OrderSide.BUY, 500.0)
        assert prediction.orderbook_estimate > 0

    @pytest.mark.asyncio
    async def test_predict_slippage_volatility(self):
        """测试基于波动率预测滑点"""
        # 设置市场数据
        market_data_provider = SimulatedMarketDataProvider()
        market_data_provider.set_market_data("AAPL", {
            'bid': 100.0,
            'ask': 100.02,
            'last': 100.01,
            'volume': 1000000,
            'volatility': 0.30,  # 高波动率
        })

        tracker = SlippageTracker(market_data_provider=market_data_provider)

        prediction = await tracker.predict_slippage("AAPL", OrderSide.BUY, 10000.0)
        assert prediction.volatility_estimate > 0

    @pytest.mark.asyncio
    async def test_predict_slippage_risk_level(self):
        """测试滑点风险评估"""
        tracker = SlippageTracker()

        # 低滑点
        prediction = await tracker.predict_slippage("AAPL", OrderSide.BUY, 100.0)
        # 风险级别应该是 low/medium/high 之一
        assert prediction.risk_level in ['low', 'medium', 'high']

    @pytest.mark.asyncio
    async def test_attribute_slippage(self):
        """测试滑点归因"""
        tracker = SlippageTracker()

        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            decision_price=100.0,
            arrival_price=100.2,
            execution_price=100.5,
            decision_time=datetime.now() - timedelta(seconds=10),
            execution_time=datetime.now(),
            volume=1000000.0,
            volatility=0.25,
            spread=0.10,
            mid_price=100.0,
            market_condition=MarketCondition.VOLATILE,
        )

        await tracker.add_order(order)
        await tracker.calculate_slippage(order)

        attribution = await tracker.attribute_slippage("ORD001")

        # 验证归因
        assert attribution.order_id == "ORD001"
        assert attribution.total_slippage > 0

        # 验证归因总和接近1
        total_contrib = (
            attribution.execution_delay_contribution +
            attribution.market_condition_contribution +
            attribution.order_type_contribution +
            attribution.liquidity_contribution +
            attribution.order_size_contribution +
            attribution.time_of_day_contribution +
            attribution.volatility_contribution
        )
        assert abs(total_contrib - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_generate_optimization_recommendation(self):
        """测试生成优化建议"""
        tracker = SlippageTracker()

        recommendation = await tracker.generate_optimization_recommendation(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50000.0,
            current_price=100.0,
        )

        assert recommendation.symbol == "AAPL"
        assert recommendation.suggested_split > 0
        assert recommendation.suggested_order_type in [OrderType.MARKET, OrderType.LIMIT]
        assert recommendation.expected_slippage_reduction >= 0

    @pytest.mark.asyncio
    async def test_recommendation_split_large_order(self):
        """测试大订单拆分建议"""
        market_data_provider = SimulatedMarketDataProvider()
        market_data_provider.set_market_data("AAPL", {
            'bid': 100.0,
            'ask': 100.02,
            'last': 100.01,
            'volume': 1000000,
            'volatility': 0.20,
        })

        tracker = SlippageTracker(market_data_provider=market_data_provider)

        # 大订单: 10% of daily volume
        recommendation = await tracker.generate_optimization_recommendation(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100000.0,
            current_price=100.0,
        )

        assert recommendation.suggested_split > 1

    @pytest.mark.asyncio
    async def test_recommendation_order_type(self):
        """测试订单类型建议"""
        tracker = SlippageTracker()

        # 高滑点情况
        recommendation = await tracker.generate_optimization_recommendation(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100000.0,
            current_price=100.0,
        )

        # 应该建议限价单或其他优化
        assert recommendation.order_type_reasoning != ""

    @pytest.mark.asyncio
    async def test_alert_warning_threshold(self):
        """测试警告阈值触发"""
        tracker = SlippageTracker()
        alerts = []

        def alert_callback(alert: SlippageAlert):
            alerts.append(alert)

        tracker.add_alert_callback(alert_callback)

        # 创建高滑点订单
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            arrival_price=100.0,
            execution_price=101.5,  # 1.5% 滑点 = 150 bps
        )

        await tracker.calculate_slippage(order)

        # 应该触发警报
        assert len(alerts) > 0
        assert alerts[0].level in [AlertLevel.WARNING, AlertLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_alert_critical_threshold(self):
        """测试严重阈值触发"""
        tracker = SlippageTracker()
        alerts = []

        async def async_alert_callback(alert: SlippageAlert):
            alerts.append(alert)

        tracker.add_alert_callback(async_alert_callback)

        # 创建非常高滑点订单
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            arrival_price=100.0,
            execution_price=103.5,  # 3.5% 滑点 = 350 bps
        )

        await tracker.calculate_slippage(order)

        # 应该触发严重警报
        assert len(alerts) > 0
        assert alerts[0].level == AlertLevel.CRITICAL

    def test_get_recent_orders(self):
        """测试获取最近订单"""
        tracker = SlippageTracker()

        # 添加订单
        for i in range(50):
            order = OrderRecord(
                order_id=f"ORD{i:03d}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
            )
            asyncio.run(tracker.add_order(order))

        recent = tracker.get_recent_orders(10)
        assert len(recent) == 10
        assert recent[0].order_id == "ORD040"  # 最旧的10个中的第一个

    def test_get_slippage_result(self):
        """测试获取滑点结果"""
        tracker = SlippageTracker()

        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            arrival_price=100.0,
            execution_price=100.1,
        )

        asyncio.run(tracker.calculate_slippage(order))

        result = tracker.get_slippage_result("ORD001")
        assert result is not None
        assert result.order_id == "ORD001"

    def test_clear_history(self):
        """测试清除历史"""
        tracker = SlippageTracker()

        # 添加数据
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )
        asyncio.run(tracker.add_order(order))

        tracker.clear_history()

        recent = tracker.get_recent_orders(10)
        assert len(recent) == 0


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_slippage_tracker(self):
        """测试创建滑点追踪器"""
        tracker = create_slippage_tracker(
            warning_slippage_bps=15.0,
            critical_slippage_bps=40.0,
        )

        assert tracker.config.warning_slippage_bps == 15.0
        assert tracker.config.critical_slippage_bps == 40.0

    def test_create_order_record(self):
        """测试创建订单记录"""
        order = create_order_record(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=100.0,
            arrival_price=100.0,
        )

        assert order.order_id == "ORD001"
        assert order.side == OrderSide.BUY
        assert order.arrival_price == 100.0

    @pytest.mark.asyncio
    async def test_quick_slippage_analysis(self):
        """测试快速滑点分析"""
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            arrival_price=100.0,
            execution_price=100.1,
        )

        result = await quick_slippage_analysis(order)
        assert result.order_id == "ORD001"
        assert result.total_slippage > 0


# ==================== 综合场景测试 ====================

class TestIntegrationScenarios:
    """综合场景测试"""

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """测试完整工作流"""
        # 1. 创建追踪器
        tracker = create_slippage_tracker()
        await tracker.start()

        # 2. 添加订单并计算滑点
        order = OrderRecord(
            order_id="ORD001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000.0,
            decision_price=100.0,
            arrival_price=100.2,
            execution_price=100.5,
            decision_time=datetime.now() - timedelta(seconds=5),
            execution_time=datetime.now(),
            volume=1000000.0,
            volatility=0.20,
            spread=0.10,
            mid_price=100.0,
            market_condition=MarketCondition.NORMAL,
        )

        await tracker.add_order(order)
        result = await tracker.calculate_slippage(order)

        # 3. 获取统计
        stats = await tracker.get_statistics("symbol")

        # 4. 预测滑点
        prediction = await tracker.predict_slippage("AAPL", OrderSide.BUY, 5000.0)

        # 5. 归因分析
        attribution = await tracker.attribute_slippage("ORD001")

        # 6. 优化建议
        recommendation = await tracker.generate_optimization_recommendation(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10000.0,
            current_price=100.5,
        )

        # 验证
        assert result.total_slippage > 0
        assert "AAPL" in stats
        assert prediction.predicted_slippage > 0
        assert attribution.total_slippage > 0
        assert recommendation.suggested_split > 0

        await tracker.stop()

    @pytest.mark.asyncio
    async def test_batch_analysis(self):
        """测试批量分析"""
        tracker = create_slippage_tracker()

        # 创建多个订单
        orders = []
        for i in range(50):
            order = OrderRecord(
                order_id=f"ORD{i:03d}",
                symbol="AAPL" if i % 2 == 0 else "GOOGL",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=100.0 * (i + 1),
                arrival_price=100.0,
                execution_price=100.0 + i * 0.01,
                execution_time=datetime.now() - timedelta(days=i),
                volume=1000000.0,
            )
            orders.append(order)
            await tracker.add_order(order)
            await tracker.calculate_slippage(order)

        # 分析
        stats_symbol = await tracker.get_statistics("symbol")
        stats_time = await tracker.get_statistics("time")

        assert len(stats_symbol) >= 2  # AAPL and GOOGL
        assert len(stats_time) > 0

    @pytest.mark.asyncio
    async def test_realtime_monitoring(self):
        """测试实时监控"""
        tracker = create_slippage_tracker()
        alerts_received = []

        def alert_callback(alert: SlippageAlert):
            alerts_received.append(alert)

        tracker.add_alert_callback(alert_callback)
        await tracker.start()

        # 模拟订单流
        for i in range(10):
            order = OrderRecord(
                order_id=f"ORD{i:03d}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                arrival_price=100.0,
                execution_price=100.0 + i * 0.5,  # 逐渐增加滑点
            )
            await tracker.add_order(order)
            await tracker.calculate_slippage(order)

        # 应该有一些警报
        assert len(alerts_received) > 0

        await tracker.stop()
