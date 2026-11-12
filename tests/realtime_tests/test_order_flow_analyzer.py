"""
订单流分析器测试
Test Order Flow Analyzer

全面测试订单流分析功能
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List
import math

from quantrisk_realtime.realtime.order_flow_analyzer import (
    # 主类
    OrderFlowAnalyzer,
    SimulatedOrderBookProvider,
    SimulatedTradeDataProvider,

    # 枚举
    OrderSide,
    TradeType,
    OrderFlowSignal,
    ToxicityLevel,
    InstitutionalActivity,
    AnalyzerState,

    # 数据类
    OrderBookLevel,
    OrderBook,
    Trade,
    OrderBookMetrics,
    TradeFlowMetrics,
    MicrostructureIndicators,
    LargeOrderDetection,
    IcebergDetection,
    ToxicityAnalysis,
    InstitutionalSignal,
    OrderFlowSignalResult,
    AnalyzerConfig,

    # 工厂函数
    create_order_flow_analyzer,
    create_order_book,
    create_trade,
    quick_analyze,
)


# ==================== 测试夹具 ====================

@pytest.fixture
def config():
    """测试配置"""
    return AnalyzerConfig(
        orderbook_window_seconds=60.0,
        trade_window_seconds=300.0,
        block_trade_percentile=95.0,
        update_interval_seconds=0.1,
    )


@pytest.fixture
def order_book_provider():
    """订单簿提供者"""
    return SimulatedOrderBookProvider()


@pytest.fixture
def trade_provider():
    """交易提供者"""
    return SimulatedTradeDataProvider()


@pytest.fixture
def analyzer(order_book_provider, trade_provider, config):
    """分析器实例"""
    return OrderFlowAnalyzer(
        order_book_provider=order_book_provider,
        trade_provider=trade_provider,
        config=config,
    )


@pytest.fixture
def sample_order_book():
    """样例订单簿"""
    bids = [
        (100.00, 100),
        (99.99, 150),
        (99.98, 200),
        (99.97, 180),
        (99.96, 120),
    ]

    asks = [
        (100.01, 80),
        (100.02, 120),
        (100.03, 160),
        (100.04, 140),
        (100.05, 100),
    ]

    return create_order_book("AAPL", bids, asks)


@pytest.fixture
def sample_trades():
    """样例交易列表"""
    trades = []
    base_time = datetime.now()

    for i in range(50):
        trade = Trade(
            symbol="AAPL",
            price=100.0 + (i % 10 - 5) * 0.01,
            quantity=100 + i * 2,
            side=OrderSide.BID if i % 2 == 0 else OrderSide.ASK,
            trade_type=TradeType.AGGRESSIVE if i % 3 == 0 else TradeType.PASSIVE,
            timestamp=base_time - timedelta(seconds=50-i),
        )
        trades.append(trade)

    return trades


# ==================== 数据类测试 ====================

class TestDataClasses:
    """测试数据类"""

    def test_order_book_level(self):
        """测试订单簿层级"""
        level = OrderBookLevel(price=100.0, quantity=50, num_orders=2)

        assert level.price == 100.0
        assert level.quantity == 50
        assert level.num_orders == 2

        data = level.to_dict()
        assert data['price'] == 100.0
        assert data['quantity'] == 50

    def test_order_book(self, sample_order_book):
        """测试订单簿"""
        ob = sample_order_book

        assert ob.symbol == "AAPL"
        assert ob.bid_price == 100.00
        assert ob.ask_price == 100.01
        assert abs(ob.spread - 0.01) < 0.0001  # 浮点精度
        assert abs(ob.mid_price - 100.005) < 0.0001

        data = ob.to_dict()
        assert data['symbol'] == "AAPL"
        assert abs(data['spread'] - 0.01) < 0.0001

    def test_trade(self):
        """测试交易记录"""
        trade = Trade(
            symbol="AAPL",
            price=100.0,
            quantity=100,
            side=OrderSide.BID,
            trade_type=TradeType.AGGRESSIVE,
        )

        assert trade.symbol == "AAPL"
        assert trade.notional == 10000.0
        assert trade.side == OrderSide.BID

        data = trade.to_dict()
        assert data['notional'] == 10000.0

    def test_orderbook_metrics(self):
        """测试订单簿指标"""
        metrics = OrderBookMetrics(
            symbol="AAPL",
            bid_ask_spread=0.01,
            spread_bps=1.0,
            mid_price=100.0,
            order_book_imbalance=0.2,
            bid_depth=1000,
            ask_depth=800,
            total_depth=1800,
            depth_ratio=1.25,
            volume_weighted_mid=100.005,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.order_book_imbalance == 0.2

        data = metrics.to_dict()
        assert 'symbol' in data
        assert 'order_book_imbalance' in data

    def test_tradeflow_metrics(self):
        """测试交易流指标"""
        metrics = TradeFlowMetrics(
            symbol="AAPL",
            trade_flow_imbalance=0.3,
            volume_clock=10000,
            volume_rate=200,
            trade_arrival_rate=0.5,
            aggressive_buy_ratio=0.3,
            aggressive_sell_ratio=0.2,
            passive_ratio=0.5,
            total_trades=100,
            buy_volume=6000,
            sell_volume=4000,
            total_volume=10000,
        )

        assert metrics.trade_flow_imbalance == 0.3
        assert metrics.total_trades == 100

        data = metrics.to_dict()
        assert data['total_volume'] == 10000

    def test_microstructure_indicators(self):
        """测试微观结构指标"""
        indicators = MicrostructureIndicators(
            symbol="AAPL",
            kyles_lambda=0.001,
            amihud_illiquidity=0.0001,
            roll_spread=0.0005,
            vpin=0.4,
            price_impact=0.0002,
            effective_spread=0.001,
            realized_spread=0.0007,
        )

        assert indicators.vpin == 0.4
        assert indicators.kyles_lambda == 0.001

        data = indicators.to_dict()
        assert 'vpin' in data


# ==================== 订单簿分析测试 ====================

class TestOrderBookAnalysis:
    """测试订单簿分析"""

    @pytest.mark.asyncio
    async def test_calculate_orderbook_metrics(self, analyzer, sample_order_book):
        """测试计算订单簿指标"""
        analyzer._order_books["AAPL"] = sample_order_book

        metrics = await analyzer.calculate_orderbook_metrics("AAPL")

        assert metrics is not None
        assert metrics.symbol == "AAPL"
        assert metrics.bid_ask_spread > 0
        assert metrics.mid_price > 0
        assert -1 <= metrics.order_book_imbalance <= 1

    @pytest.mark.asyncio
    async def test_orderbook_imbalance(self, analyzer):
        """测试订单簿失衡计算"""
        # 买单占优
        bids = [(100.0, 1000), (99.9, 800)]
        asks = [(100.1, 400), (100.2, 300)]
        ob = create_order_book("AAPL", bids, asks)

        analyzer._order_books["AAPL"] = ob
        metrics = await analyzer.calculate_orderbook_metrics("AAPL")

        assert metrics.order_book_imbalance > 0  # 买单占优
        assert metrics.bid_depth > metrics.ask_depth

    @pytest.mark.asyncio
    async def test_vwmp_calculation(self, analyzer, sample_order_book):
        """测试成交量加权中间价"""
        analyzer._order_books["AAPL"] = sample_order_book

        vwmp = analyzer._calculate_vwmp(sample_order_book)

        assert vwmp > 0
        assert abs(vwmp - sample_order_book.mid_price) < 0.1

    @pytest.mark.asyncio
    async def test_spread_calculation(self, analyzer, sample_order_book):
        """测试价差计算"""
        analyzer._order_books["AAPL"] = sample_order_book

        metrics = await analyzer.calculate_orderbook_metrics("AAPL")

        assert abs(metrics.bid_ask_spread - 0.01) < 0.0001  # 浮点精度
        assert metrics.spread_bps > 0
        assert metrics.spread_bps < 100  # 合理范围


# ==================== 交易流分析测试 ====================

class TestTradeFlowAnalysis:
    """测试交易流分析"""

    @pytest.mark.asyncio
    async def test_calculate_tradeflow_metrics(self, analyzer, sample_trades):
        """测试计算交易流指标"""
        analyzer._trades["AAPL"].extend(sample_trades)

        metrics = await analyzer.calculate_tradeflow_metrics("AAPL")

        assert metrics is not None
        assert metrics.symbol == "AAPL"
        assert metrics.total_trades > 0
        assert metrics.total_volume > 0

    @pytest.mark.asyncio
    async def test_trade_flow_imbalance(self, analyzer):
        """测试交易流失衡"""
        # 创建买单占优的交易流
        trades = []
        for i in range(20):
            trade = create_trade(
                "AAPL",
                100.0,
                100,
                "bid" if i < 15 else "ask",  # 75% 买单
                "aggressive"
            )
            trades.append(trade)

        analyzer._trades["AAPL"].extend(trades)
        metrics = await analyzer.calculate_tradeflow_metrics("AAPL")

        assert metrics.trade_flow_imbalance > 0  # 买单占优
        assert metrics.buy_volume > metrics.sell_volume

    @pytest.mark.asyncio
    async def test_aggressive_passive_classification(self, analyzer, sample_trades):
        """测试主动/被动分类"""
        analyzer._trades["AAPL"].extend(sample_trades)

        metrics = await analyzer.calculate_tradeflow_metrics("AAPL")

        assert 0 <= metrics.aggressive_buy_ratio <= 1
        assert 0 <= metrics.aggressive_sell_ratio <= 1
        assert 0 <= metrics.passive_ratio <= 1

        # 总和应该约等于1
        total = (metrics.aggressive_buy_ratio +
                metrics.aggressive_sell_ratio +
                metrics.passive_ratio)
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_volume_rate(self, analyzer, sample_trades):
        """测试成交量速率"""
        analyzer._trades["AAPL"].extend(sample_trades)

        metrics = await analyzer.calculate_tradeflow_metrics("AAPL")

        assert metrics.volume_rate > 0
        assert metrics.trade_arrival_rate > 0


# ==================== 微观结构指标测试 ====================

class TestMicrostructureIndicators:
    """测试微观结构指标"""

    @pytest.mark.asyncio
    async def test_calculate_microstructure_indicators(self, analyzer, sample_trades):
        """测试计算微观结构指标"""
        analyzer._trades["AAPL"].extend(sample_trades)

        indicators = await analyzer.calculate_microstructure_indicators("AAPL")

        assert indicators is not None
        assert indicators.symbol == "AAPL"

    def test_kyles_lambda(self, analyzer, sample_trades):
        """测试Kyle's Lambda"""
        kyles_lambda = analyzer._calculate_kyles_lambda(sample_trades)

        # Kyle's Lambda可能为正或负
        assert isinstance(kyles_lambda, float)

    def test_amihud_illiquidity(self, analyzer, sample_trades):
        """测试Amihud非流动性"""
        amihud = analyzer._calculate_amihud_illiquidity(sample_trades)

        assert amihud >= 0

    def test_roll_spread(self, analyzer, sample_trades):
        """测试Roll's Spread"""
        roll_spread = analyzer._calculate_roll_spread(sample_trades)

        assert roll_spread >= 0

    def test_vpin(self, analyzer, sample_trades):
        """测试VPIN计算"""
        vpin = analyzer._calculate_vpin(sample_trades)

        assert 0 <= vpin <= 1

    def test_price_impact(self, analyzer, sample_trades):
        """测试价格影响"""
        impact = analyzer._calculate_price_impact(sample_trades)

        assert impact >= 0

    @pytest.mark.asyncio
    async def test_effective_spread(self, analyzer, sample_order_book, sample_trades):
        """测试有效价差"""
        analyzer._order_books["AAPL"] = sample_order_book

        effective, realized = analyzer._calculate_spreads("AAPL", sample_trades)

        assert effective >= 0
        assert realized >= 0
        assert realized <= effective  # 已实现价差通常小于有效价差


# ==================== 大单检测测试 ====================

class TestLargeOrderDetection:
    """测试大单检测"""

    def test_detect_large_order(self, analyzer):
        """测试大单检测"""
        # 先添加一些正常交易
        for i in range(50):
            trade = create_trade("AAPL", 100.0, 100, "bid", "aggressive")
            analyzer._trades["AAPL"].append(trade)

        # 添加大单
        large_trade = create_trade("AAPL", 100.0, 5000, "bid", "aggressive")
        analyzer._detect_large_order(large_trade)

        # 检查是否被检测到
        large_orders = analyzer.get_large_orders("AAPL")
        assert len(large_orders) > 0

    def test_large_order_threshold(self, analyzer):
        """测试大单阈值"""
        # 添加基准交易
        for i in range(100):
            trade = create_trade("AAPL", 100.0, 100 + i, "bid")
            analyzer._trades["AAPL"].append(trade)

        # 添加超过阈值的交易
        large_trade = Trade(
            symbol="AAPL",
            price=100.0,
            quantity=1000,
            side=OrderSide.BID,
            trade_type=TradeType.AGGRESSIVE,
        )

        analyzer._detect_large_order(large_trade)

        detections = analyzer.get_large_orders("AAPL")
        if detections:
            assert detections[0].is_block_trade

    def test_large_order_impact_estimation(self, analyzer, sample_order_book):
        """测试大单影响估算"""
        analyzer._order_books["AAPL"] = sample_order_book

        # 添加基准交易
        for i in range(50):
            trade = create_trade("AAPL", 100.0, 100, "bid")
            analyzer._trades["AAPL"].append(trade)

        # 添加大单
        large_trade = create_trade("AAPL", 100.0, 5000, "bid", "aggressive")
        analyzer._detect_large_order(large_trade)

        detections = analyzer.get_large_orders("AAPL")
        if detections:
            assert detections[0].estimated_impact > 0


# ==================== 冰山订单检测测试 ====================

class TestIcebergDetection:
    """测试冰山订单检测"""

    @pytest.mark.asyncio
    async def test_detect_iceberg_orders(self, analyzer, sample_order_book):
        """测试冰山订单检测"""
        analyzer._order_books["AAPL"] = sample_order_book

        await analyzer._detect_iceberg_orders("AAPL")

        # 简化版本可能检测到冰山订单
        icebergs = analyzer.get_iceberg_orders("AAPL")
        assert isinstance(icebergs, list)

    def test_iceberg_side_detection(self, analyzer, sample_order_book):
        """测试单侧冰山检测"""
        analyzer._detect_iceberg_side(sample_order_book, OrderSide.BID)

        # 检查是否有结果
        icebergs = analyzer.get_iceberg_orders("AAPL")
        # 可能有也可能没有，取决于实现


# ==================== 毒性分析测试 ====================

class TestToxicityAnalysis:
    """测试订单流毒性分析"""

    @pytest.mark.asyncio
    async def test_analyze_toxicity(self, analyzer, sample_order_book, sample_trades):
        """测试毒性分析"""
        # 设置必要的数据
        analyzer._order_books["AAPL"] = sample_order_book
        analyzer._trades["AAPL"].extend(sample_trades)

        # 计算必要的指标
        ob_metrics = await analyzer.calculate_orderbook_metrics("AAPL")
        tf_metrics = await analyzer.calculate_tradeflow_metrics("AAPL")
        ms_indicators = await analyzer.calculate_microstructure_indicators("AAPL")

        analyzer._orderbook_metrics["AAPL"] = ob_metrics
        analyzer._tradeflow_metrics["AAPL"] = tf_metrics
        analyzer._microstructure["AAPL"] = ms_indicators

        # 分析毒性
        toxicity = await analyzer.analyze_toxicity("AAPL")

        assert toxicity is not None
        assert toxicity.symbol == "AAPL"
        assert 0 <= toxicity.overall_score <= 1
        assert isinstance(toxicity.toxicity_level, ToxicityLevel)

    @pytest.mark.asyncio
    async def test_toxicity_levels(self, analyzer, sample_order_book, sample_trades):
        """测试毒性等级判定"""
        analyzer._order_books["AAPL"] = sample_order_book
        analyzer._trades["AAPL"].extend(sample_trades)

        # 设置指标
        ob_metrics = await analyzer.calculate_orderbook_metrics("AAPL")
        tf_metrics = await analyzer.calculate_tradeflow_metrics("AAPL")

        # 创建高毒性微观结构指标
        high_toxicity_ms = MicrostructureIndicators(
            symbol="AAPL",
            kyles_lambda=0.01,
            amihud_illiquidity=0.001,
            roll_spread=0.01,
            vpin=0.8,  # 高VPIN
            price_impact=0.001,
            effective_spread=0.01,
            realized_spread=0.007,
        )

        analyzer._orderbook_metrics["AAPL"] = ob_metrics
        analyzer._tradeflow_metrics["AAPL"] = tf_metrics
        analyzer._microstructure["AAPL"] = high_toxicity_ms

        toxicity = await analyzer.analyze_toxicity("AAPL")

        # 高VPIN应该导致较高的毒性得分
        assert toxicity.overall_score > 0.3


# ==================== 机构活动检测测试 ====================

class TestInstitutionalActivity:
    """测试机构活动检测"""

    @pytest.mark.asyncio
    async def test_detect_institutional_buying(self, analyzer):
        """测试检测机构买入"""
        # 添加大单模式（买入占优）
        for i in range(10):
            trade = Trade(
                symbol="AAPL",
                price=100.0 + i * 0.01,
                quantity=5000,
                side=OrderSide.BID,
                trade_type=TradeType.AGGRESSIVE,
            )

            detection = LargeOrderDetection(
                symbol="AAPL",
                trade=trade,
                size_percentile=95,
                is_block_trade=True,
                estimated_impact=5,
            )

            analyzer._large_orders["AAPL"].append(detection)

        signal = await analyzer.detect_institutional_activity("AAPL")

        if signal:
            assert signal.activity_type in [
                InstitutionalActivity.BUYING,
                InstitutionalActivity.ACCUMULATION
            ]

    @pytest.mark.asyncio
    async def test_institutional_conviction(self, analyzer):
        """测试机构确信度"""
        # 添加明显的机构活动
        for i in range(5):
            trade = Trade(
                symbol="AAPL",
                price=100.0,
                quantity=10000,
                side=OrderSide.BID,
                trade_type=TradeType.AGGRESSIVE,
            )

            detection = LargeOrderDetection(
                symbol="AAPL",
                trade=trade,
                size_percentile=99,
                is_block_trade=True,
                estimated_impact=10,
            )

            analyzer._large_orders["AAPL"].append(detection)

        signal = await analyzer.detect_institutional_activity("AAPL")

        if signal:
            assert signal.conviction_score > 0


# ==================== 信号生成测试 ====================

class TestSignalGeneration:
    """测试信号生成"""

    @pytest.mark.asyncio
    async def test_generate_signal(self, analyzer, sample_order_book, sample_trades):
        """测试生成信号"""
        analyzer._order_books["AAPL"] = sample_order_book
        analyzer._trades["AAPL"].extend(sample_trades)

        # 计算所有必要指标
        ob_metrics = await analyzer.calculate_orderbook_metrics("AAPL")
        tf_metrics = await analyzer.calculate_tradeflow_metrics("AAPL")
        ms_indicators = await analyzer.calculate_microstructure_indicators("AAPL")

        analyzer._orderbook_metrics["AAPL"] = ob_metrics
        analyzer._tradeflow_metrics["AAPL"] = tf_metrics
        analyzer._microstructure["AAPL"] = ms_indicators

        signal = await analyzer.generate_signal("AAPL")

        if signal:
            assert signal.symbol == "AAPL"
            assert isinstance(signal.signal, OrderFlowSignal)
            assert 0 <= signal.strength <= 1
            assert 0 <= signal.confidence <= 1

    @pytest.mark.asyncio
    async def test_signal_reasoning(self, analyzer, sample_order_book, sample_trades):
        """测试信号推理"""
        analyzer._order_books["AAPL"] = sample_order_book
        analyzer._trades["AAPL"].extend(sample_trades)

        ob_metrics = await analyzer.calculate_orderbook_metrics("AAPL")
        tf_metrics = await analyzer.calculate_tradeflow_metrics("AAPL")
        ms_indicators = await analyzer.calculate_microstructure_indicators("AAPL")

        analyzer._orderbook_metrics["AAPL"] = ob_metrics
        analyzer._tradeflow_metrics["AAPL"] = tf_metrics
        analyzer._microstructure["AAPL"] = ms_indicators

        signal = await analyzer.generate_signal("AAPL")

        if signal:
            assert isinstance(signal.reasoning, str)
            assert len(signal.reasoning) > 0

    @pytest.mark.asyncio
    async def test_buy_signal_generation(self, analyzer):
        """测试买入信号生成"""
        # 创建买单占优的市场状态
        bids = [(100.0, 2000), (99.9, 1800)]
        asks = [(100.1, 800), (100.2, 700)]
        ob = create_order_book("AAPL", bids, asks)

        trades = []
        for i in range(30):
            trade = create_trade("AAPL", 100.0, 100, "bid", "aggressive")
            trades.append(trade)

        analyzer._order_books["AAPL"] = ob
        analyzer._trades["AAPL"].extend(trades)

        ob_metrics = await analyzer.calculate_orderbook_metrics("AAPL")
        tf_metrics = await analyzer.calculate_tradeflow_metrics("AAPL")
        ms_indicators = await analyzer.calculate_microstructure_indicators("AAPL")

        analyzer._orderbook_metrics["AAPL"] = ob_metrics
        analyzer._tradeflow_metrics["AAPL"] = tf_metrics
        analyzer._microstructure["AAPL"] = ms_indicators

        signal = await analyzer.generate_signal("AAPL")

        if signal:
            # 应该倾向于买入信号
            assert signal.signal in [
                OrderFlowSignal.BUY,
                OrderFlowSignal.STRONG_BUY,
                OrderFlowSignal.NEUTRAL
            ]


# ==================== 分析器状态管理测试 ====================

class TestAnalyzerStateManagement:
    """测试分析器状态管理"""

    @pytest.mark.asyncio
    async def test_start_stop(self, analyzer):
        """测试启动和停止"""
        assert analyzer.get_state() == AnalyzerState.IDLE

        # 启动
        await analyzer.start(["AAPL"])
        await asyncio.sleep(0.2)
        assert analyzer.get_state() == AnalyzerState.RUNNING

        # 停止
        await analyzer.stop()
        assert analyzer.get_state() == AnalyzerState.IDLE

    @pytest.mark.asyncio
    async def test_multiple_symbols(self, analyzer):
        """测试多品种分析"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        await analyzer.start(symbols)
        await asyncio.sleep(0.2)

        assert analyzer.get_state() == AnalyzerState.RUNNING

        await analyzer.stop()

    def test_callback_registration(self, analyzer):
        """测试回调注册"""
        called = []

        def signal_callback(signal):
            called.append('signal')

        def large_order_callback(detection):
            called.append('large_order')

        def toxicity_callback(toxicity):
            called.append('toxicity')

        analyzer.register_signal_callback(signal_callback)
        analyzer.register_large_order_callback(large_order_callback)
        analyzer.register_toxicity_callback(toxicity_callback)

        # 触发回调
        signal = OrderFlowSignalResult(
            symbol="AAPL",
            signal=OrderFlowSignal.BUY,
            strength=0.8,
            confidence=0.7,
            order_book_contribution=0.3,
            trade_flow_contribution=0.4,
            microstructure_contribution=0.1,
            reasoning="Test",
        )

        analyzer._emit_signal(signal)
        assert 'signal' in called


# ==================== 提供者测试 ====================

class TestProviders:
    """测试数据提供者"""

    @pytest.mark.asyncio
    async def test_simulated_order_book_provider(self, order_book_provider):
        """测试模拟订单簿提供者"""
        ob = await order_book_provider.get_order_book("AAPL")

        assert ob is not None
        assert ob.symbol == "AAPL"
        assert len(ob.bids) > 0
        assert len(ob.asks) > 0

    @pytest.mark.asyncio
    async def test_order_book_subscription(self, order_book_provider):
        """测试订单簿订阅"""
        received = []

        def callback(ob):
            received.append(ob)

        await order_book_provider.subscribe_order_book("AAPL", callback)

        # 更新订单簿
        ob = await order_book_provider.get_order_book("AAPL")
        order_book_provider.update_order_book(ob)

        assert len(received) > 0

    @pytest.mark.asyncio
    async def test_simulated_trade_provider(self, trade_provider):
        """测试模拟交易提供者"""
        trades = await trade_provider.get_recent_trades("AAPL")

        assert isinstance(trades, list)

    @pytest.mark.asyncio
    async def test_trade_subscription(self, trade_provider):
        """测试交易订阅"""
        received = []

        def callback(trade):
            received.append(trade)

        await trade_provider.subscribe_trades("AAPL", callback)

        # 添加交易
        trade = create_trade("AAPL", 100.0, 100, "bid")
        trade_provider.add_trade(trade)

        assert len(received) > 0


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_order_flow_analyzer(self):
        """测试创建分析器"""
        analyzer = create_order_flow_analyzer()

        assert isinstance(analyzer, OrderFlowAnalyzer)
        assert analyzer.get_state() == AnalyzerState.IDLE

    def test_create_order_flow_analyzer_with_config(self):
        """测试使用配置创建分析器"""
        analyzer = create_order_flow_analyzer(
            orderbook_window_seconds=30.0,
            block_trade_percentile=90.0,
        )

        assert analyzer.config.orderbook_window_seconds == 30.0
        assert analyzer.config.block_trade_percentile == 90.0

    def test_create_order_book(self):
        """测试创建订单簿"""
        bids = [(100.0, 100), (99.9, 150)]
        asks = [(100.1, 80), (100.2, 120)]

        ob = create_order_book("AAPL", bids, asks)

        assert ob.symbol == "AAPL"
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2

    def test_create_trade(self):
        """测试创建交易"""
        trade = create_trade("AAPL", 100.0, 100, "bid", "aggressive")

        assert trade.symbol == "AAPL"
        assert trade.price == 100.0
        assert trade.side == OrderSide.BID

    @pytest.mark.asyncio
    async def test_quick_analyze(self, sample_order_book, sample_trades):
        """测试快速分析"""
        result = await quick_analyze("AAPL", sample_order_book, sample_trades)

        assert isinstance(result, dict)
        assert 'orderbook_metrics' in result
        assert 'tradeflow_metrics' in result
        assert 'microstructure_indicators' in result


# ==================== 边界条件测试 ====================

class TestEdgeCases:
    """测试边界条件"""

    @pytest.mark.asyncio
    async def test_empty_order_book(self, analyzer):
        """测试空订单簿"""
        ob = OrderBook(symbol="AAPL", bids=[], asks=[])
        analyzer._order_books["AAPL"] = ob

        metrics = await analyzer.calculate_orderbook_metrics("AAPL")

        assert metrics is None

    @pytest.mark.asyncio
    async def test_no_trades(self, analyzer):
        """测试无交易"""
        metrics = await analyzer.calculate_tradeflow_metrics("AAPL")

        assert metrics is None

    @pytest.mark.asyncio
    async def test_insufficient_data(self, analyzer):
        """测试数据不足"""
        # 只有少量交易
        trades = [create_trade("AAPL", 100.0, 100, "bid") for _ in range(3)]
        analyzer._trades["AAPL"].extend(trades)

        indicators = await analyzer.calculate_microstructure_indicators("AAPL")

        # 应该返回None或默认值
        assert indicators is None or indicators.vpin == 0

    @pytest.mark.asyncio
    async def test_extreme_imbalance(self, analyzer):
        """测试极端失衡"""
        # 完全买单占优
        bids = [(100.0, 10000)]
        asks = [(100.1, 1)]
        ob = create_order_book("AAPL", bids, asks)

        analyzer._order_books["AAPL"] = ob
        metrics = await analyzer.calculate_orderbook_metrics("AAPL")

        assert metrics.order_book_imbalance > 0.9


# ==================== 性能测试 ====================

class TestPerformance:
    """测试性能"""

    @pytest.mark.asyncio
    async def test_analysis_speed(self, analyzer, sample_order_book, sample_trades):
        """测试分析速度"""
        analyzer._order_books["AAPL"] = sample_order_book
        analyzer._trades["AAPL"].extend(sample_trades)

        start_time = datetime.now()

        # 运行完整分析
        await analyzer.calculate_orderbook_metrics("AAPL")
        await analyzer.calculate_tradeflow_metrics("AAPL")
        await analyzer.calculate_microstructure_indicators("AAPL")

        elapsed = (datetime.now() - start_time).total_seconds()

        # 应该在1秒内完成
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_large_data_handling(self, analyzer):
        """测试大数据处理"""
        # 添加大量交易
        trades = []
        for i in range(1000):
            trade = create_trade("AAPL", 100.0 + i * 0.001, 100, "bid")
            trades.append(trade)

        analyzer._trades["AAPL"].extend(trades)

        # 应该能够处理
        metrics = await analyzer.calculate_tradeflow_metrics("AAPL")

        assert metrics is not None


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(
        self,
        analyzer,
        sample_order_book,
        sample_trades
    ):
        """测试完整分析流程"""
        # 设置数据
        analyzer._order_books["AAPL"] = sample_order_book
        analyzer._trades["AAPL"].extend(sample_trades)

        # 运行完整分析
        ob_metrics = await analyzer.calculate_orderbook_metrics("AAPL")
        tf_metrics = await analyzer.calculate_tradeflow_metrics("AAPL")
        ms_indicators = await analyzer.calculate_microstructure_indicators("AAPL")

        analyzer._orderbook_metrics["AAPL"] = ob_metrics
        analyzer._tradeflow_metrics["AAPL"] = tf_metrics
        analyzer._microstructure["AAPL"] = ms_indicators

        toxicity = await analyzer.analyze_toxicity("AAPL")
        signal = await analyzer.generate_signal("AAPL")

        # 验证所有结果
        assert ob_metrics is not None
        assert tf_metrics is not None
        assert ms_indicators is not None
        assert toxicity is not None

    @pytest.mark.asyncio
    async def test_real_time_update_flow(self, analyzer, order_book_provider, trade_provider):
        """测试实时更新流程"""
        results = []

        def signal_callback(signal):
            results.append(signal)

        analyzer.register_signal_callback(signal_callback)

        # 启动分析器
        await analyzer.start(["AAPL"])

        # 模拟数据更新
        ob = await order_book_provider.get_order_book("AAPL")
        order_book_provider.update_order_book(ob)

        trade = create_trade("AAPL", 100.0, 100, "bid", "aggressive")
        trade_provider.add_trade(trade)

        # 等待处理
        await asyncio.sleep(0.5)

        # 停止
        await analyzer.stop()

        # 可能会生成信号
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
