"""
市场数据聚合器测试
Market Data Aggregator Tests
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from quantrisk_realtime.realtime.market_data_aggregator import (
    # Enums
    DataSource,
    MarketState,
    BarInterval,
    SubscriptionType,
    DataQuality,
    # Data Classes
    Tick,
    Quote,
    Bar,
    MarketDepth,
    DataSourceConfig,
    Subscription,
    QualityMetrics,
    AggregatorConfig,
    # Bar Builder
    BarBuilder,
    # Data Source
    DataSourceHandler,
    SimulatedDataSource,
    # Aggregator
    MarketDataAggregator,
    # Factory Functions
    create_aggregator,
    create_source_config,
    create_tick,
    create_quote,
    create_market_depth,
    quick_aggregate,
)


# ============== Tick Tests ==============

class TestTick:
    """Tick数据类测试"""

    def test_tick_creation(self):
        """测试Tick创建"""
        tick = Tick(
            symbol="AAPL",
            price=150.0,
            size=100,
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        assert tick.symbol == "AAPL"
        assert tick.price == 150.0
        assert tick.size == 100
        assert tick.source == DataSource.EXCHANGE

    def test_tick_auto_trade_id(self):
        """测试Tick自动生成trade_id"""
        tick = Tick(
            symbol="AAPL",
            price=150.0,
            size=100,
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        assert tick.trade_id != ""
        assert "AAPL_" in tick.trade_id

    def test_tick_with_side(self):
        """测试带方向的Tick"""
        tick = Tick(
            symbol="AAPL",
            price=150.0,
            size=100,
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
            side="buy",
        )

        assert tick.side == "buy"


# ============== Quote Tests ==============

class TestQuote:
    """Quote数据类测试"""

    def test_quote_creation(self):
        """测试Quote创建"""
        quote = Quote(
            symbol="AAPL",
            bid_price=149.95,
            bid_size=500,
            ask_price=150.05,
            ask_size=300,
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        assert quote.symbol == "AAPL"
        assert quote.bid_price == 149.95
        assert quote.ask_price == 150.05

    def test_quote_mid_price_auto_calc(self):
        """测试Quote自动计算中间价"""
        quote = Quote(
            symbol="AAPL",
            bid_price=149.00,
            bid_size=500,
            ask_price=151.00,
            ask_size=300,
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        assert quote.mid_price == 150.0

    def test_quote_spread_auto_calc(self):
        """测试Quote自动计算价差"""
        quote = Quote(
            symbol="AAPL",
            bid_price=149.00,
            bid_size=500,
            ask_price=151.00,
            ask_size=300,
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        assert quote.spread == 2.0

    def test_quote_spread_bps_auto_calc(self):
        """测试Quote自动计算价差基点"""
        quote = Quote(
            symbol="AAPL",
            bid_price=149.00,
            bid_size=500,
            ask_price=151.00,
            ask_size=300,
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        # spread_bps = (2 / 150) * 10000 = 133.33
        assert abs(quote.spread_bps - 133.33) < 0.1


# ============== Bar Tests ==============

class TestBar:
    """Bar数据类测试"""

    def test_bar_creation(self):
        """测试Bar创建"""
        now = datetime.now()
        bar = Bar(
            symbol="AAPL",
            interval=BarInterval.MINUTE_1,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=10000,
            vwap=150.5,
            timestamp=now,
            close_timestamp=now + timedelta(minutes=1),
            source=DataSource.EXCHANGE,
        )

        assert bar.symbol == "AAPL"
        assert bar.interval == BarInterval.MINUTE_1
        assert bar.open == 150.0
        assert bar.close == 151.0

    def test_bar_typical_price(self):
        """测试Bar典型价格"""
        now = datetime.now()
        bar = Bar(
            symbol="AAPL",
            interval=BarInterval.MINUTE_1,
            open=150.0,
            high=153.0,
            low=148.0,
            close=151.0,
            volume=10000,
            vwap=150.5,
            timestamp=now,
            close_timestamp=now + timedelta(minutes=1),
            source=DataSource.EXCHANGE,
        )

        # typical = (153 + 148 + 151) / 3 = 150.67
        assert abs(bar.typical_price - 150.67) < 0.01

    def test_bar_range(self):
        """测试Bar价格区间"""
        now = datetime.now()
        bar = Bar(
            symbol="AAPL",
            interval=BarInterval.MINUTE_1,
            open=150.0,
            high=153.0,
            low=148.0,
            close=151.0,
            volume=10000,
            vwap=150.5,
            timestamp=now,
            close_timestamp=now + timedelta(minutes=1),
            source=DataSource.EXCHANGE,
        )

        assert bar.range == 5.0

    def test_bar_body(self):
        """测试Bar实体大小"""
        now = datetime.now()
        bar = Bar(
            symbol="AAPL",
            interval=BarInterval.MINUTE_1,
            open=150.0,
            high=153.0,
            low=148.0,
            close=151.0,
            volume=10000,
            vwap=150.5,
            timestamp=now,
            close_timestamp=now + timedelta(minutes=1),
            source=DataSource.EXCHANGE,
        )

        assert bar.body == 1.0

    def test_bar_is_bullish(self):
        """测试阳线判断"""
        now = datetime.now()

        bullish_bar = Bar(
            symbol="AAPL",
            interval=BarInterval.MINUTE_1,
            open=150.0,
            high=153.0,
            low=148.0,
            close=152.0,
            volume=10000,
            vwap=150.5,
            timestamp=now,
            close_timestamp=now + timedelta(minutes=1),
            source=DataSource.EXCHANGE,
        )
        assert bullish_bar.is_bullish is True

        bearish_bar = Bar(
            symbol="AAPL",
            interval=BarInterval.MINUTE_1,
            open=152.0,
            high=153.0,
            low=148.0,
            close=150.0,
            volume=10000,
            vwap=150.5,
            timestamp=now,
            close_timestamp=now + timedelta(minutes=1),
            source=DataSource.EXCHANGE,
        )
        assert bearish_bar.is_bullish is False


# ============== MarketDepth Tests ==============

class TestMarketDepth:
    """市场深度测试"""

    def test_depth_creation(self):
        """测试深度数据创建"""
        depth = MarketDepth(
            symbol="AAPL",
            bids=[(150.0, 100), (149.9, 200), (149.8, 300)],
            asks=[(150.1, 150), (150.2, 250)],
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        assert depth.symbol == "AAPL"
        assert len(depth.bids) == 3
        assert len(depth.asks) == 2

    def test_depth_bid_depth(self):
        """测试买盘深度"""
        depth = MarketDepth(
            symbol="AAPL",
            bids=[(150.0, 100), (149.9, 200), (149.8, 300)],
            asks=[(150.1, 150), (150.2, 250)],
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        assert depth.bid_depth == 600

    def test_depth_ask_depth(self):
        """测试卖盘深度"""
        depth = MarketDepth(
            symbol="AAPL",
            bids=[(150.0, 100), (149.9, 200), (149.8, 300)],
            asks=[(150.1, 150), (150.2, 250)],
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        assert depth.ask_depth == 400

    def test_depth_imbalance(self):
        """测试买卖盘不平衡度"""
        depth = MarketDepth(
            symbol="AAPL",
            bids=[(150.0, 600)],
            asks=[(150.1, 400)],
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        # imbalance = (600 - 400) / (600 + 400) = 0.2
        assert depth.imbalance == 0.2

    def test_depth_imbalance_zero_total(self):
        """测试零深度情况"""
        depth = MarketDepth(
            symbol="AAPL",
            bids=[],
            asks=[],
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        assert depth.imbalance == 0.0


# ============== BarBuilder Tests ==============

class TestBarBuilder:
    """K线构建器测试"""

    def test_bar_builder_creation(self):
        """测试BarBuilder创建"""
        builder = BarBuilder("AAPL", BarInterval.MINUTE_1, DataSource.EXCHANGE)

        assert builder.symbol == "AAPL"
        assert builder.interval == BarInterval.MINUTE_1

    def test_bar_builder_single_tick(self):
        """测试单个Tick不生成Bar"""
        builder = BarBuilder("AAPL", BarInterval.MINUTE_1, DataSource.EXCHANGE)

        tick = Tick(
            symbol="AAPL",
            price=150.0,
            size=100,
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        bar = builder.add_tick(tick)
        assert bar is None

    def test_bar_builder_force_complete(self):
        """测试强制完成Bar"""
        builder = BarBuilder("AAPL", BarInterval.MINUTE_1, DataSource.EXCHANGE)

        tick = Tick(
            symbol="AAPL",
            price=150.0,
            size=100,
            timestamp=datetime.now(),
            source=DataSource.EXCHANGE,
        )

        builder.add_tick(tick)
        bar = builder.force_complete()

        assert bar is not None
        assert bar.symbol == "AAPL"
        assert bar.open == 150.0
        assert bar.close == 150.0
        assert bar.high == 150.0
        assert bar.low == 150.0
        assert bar.volume == 100

    def test_bar_builder_multiple_ticks(self):
        """测试多个Tick"""
        builder = BarBuilder("AAPL", BarInterval.MINUTE_1, DataSource.EXCHANGE)
        base_time = datetime(2025, 1, 1, 10, 0, 0)

        # 第一个Tick
        tick1 = Tick(
            symbol="AAPL",
            price=150.0,
            size=100,
            timestamp=base_time,
            source=DataSource.EXCHANGE,
        )
        builder.add_tick(tick1)

        # 第二个Tick(更高价)
        tick2 = Tick(
            symbol="AAPL",
            price=152.0,
            size=200,
            timestamp=base_time + timedelta(seconds=10),
            source=DataSource.EXCHANGE,
        )
        builder.add_tick(tick2)

        # 第三个Tick(更低价)
        tick3 = Tick(
            symbol="AAPL",
            price=149.0,
            size=150,
            timestamp=base_time + timedelta(seconds=20),
            source=DataSource.EXCHANGE,
        )
        builder.add_tick(tick3)

        bar = builder.force_complete()

        assert bar.open == 150.0
        assert bar.high == 152.0
        assert bar.low == 149.0
        assert bar.close == 149.0
        assert bar.volume == 450
        assert bar.trade_count == 3

    def test_bar_builder_bar_completion(self):
        """测试Bar自动完成(跨周期)"""
        builder = BarBuilder("AAPL", BarInterval.MINUTE_1, DataSource.EXCHANGE)

        # 第一分钟的Tick
        tick1 = Tick(
            symbol="AAPL",
            price=150.0,
            size=100,
            timestamp=datetime(2025, 1, 1, 10, 0, 30),
            source=DataSource.EXCHANGE,
        )
        bar = builder.add_tick(tick1)
        assert bar is None

        # 第二分钟的Tick - 应该完成第一分钟的Bar
        tick2 = Tick(
            symbol="AAPL",
            price=151.0,
            size=100,
            timestamp=datetime(2025, 1, 1, 10, 1, 30),
            source=DataSource.EXCHANGE,
        )
        bar = builder.add_tick(tick2)

        assert bar is not None
        assert bar.close == 150.0  # 第一分钟的收盘价

    def test_bar_builder_vwap(self):
        """测试VWAP计算"""
        builder = BarBuilder("AAPL", BarInterval.MINUTE_1, DataSource.EXCHANGE)
        base_time = datetime(2025, 1, 1, 10, 0, 0)

        # Tick 1: price=100, size=100, value=10000
        tick1 = Tick(
            symbol="AAPL",
            price=100.0,
            size=100,
            timestamp=base_time,
            source=DataSource.EXCHANGE,
        )
        builder.add_tick(tick1)

        # Tick 2: price=200, size=100, value=20000
        tick2 = Tick(
            symbol="AAPL",
            price=200.0,
            size=100,
            timestamp=base_time + timedelta(seconds=10),
            source=DataSource.EXCHANGE,
        )
        builder.add_tick(tick2)

        bar = builder.force_complete()

        # VWAP = (10000 + 20000) / 200 = 150
        assert bar.vwap == 150.0


# ============== SimulatedDataSource Tests ==============

class TestSimulatedDataSource:
    """模拟数据源测试"""

    def test_simulated_source_creation(self):
        """测试模拟数据源创建"""
        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )

        source = SimulatedDataSource(config)
        assert source.config.source_id == "sim_1"
        assert not source.is_connected

    @pytest.mark.asyncio
    async def test_simulated_source_connect(self):
        """测试连接"""
        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )

        source = SimulatedDataSource(config)
        result = await source.connect()

        assert result is True
        assert source.is_connected

        await source.disconnect()

    @pytest.mark.asyncio
    async def test_simulated_source_disconnect(self):
        """测试断开连接"""
        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )

        source = SimulatedDataSource(config)
        await source.connect()
        await source.disconnect()

        assert not source.is_connected

    @pytest.mark.asyncio
    async def test_simulated_source_subscribe(self):
        """测试订阅"""
        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )

        source = SimulatedDataSource(config)
        await source.connect()

        result = await source.subscribe(["AAPL", "GOOGL"], SubscriptionType.QUOTE)
        assert result is True

        await source.disconnect()

    @pytest.mark.asyncio
    async def test_simulated_source_data_generation(self):
        """测试数据生成"""
        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )

        source = SimulatedDataSource(config)

        received_ticks = []
        received_quotes = []

        source.add_callback('tick', lambda t: received_ticks.append(t))
        source.add_callback('quote', lambda q: received_quotes.append(q))

        await source.connect()
        await source.subscribe(["AAPL"], SubscriptionType.ALL)

        # 等待数据生成
        await asyncio.sleep(0.5)

        await source.disconnect()

        # 应该收到一些数据
        assert len(received_ticks) > 0
        assert len(received_quotes) > 0


# ============== MarketDataAggregator Tests ==============

class TestMarketDataAggregator:
    """市场数据聚合器测试"""

    def test_aggregator_creation(self):
        """测试聚合器创建"""
        aggregator = MarketDataAggregator()

        assert aggregator is not None
        assert aggregator.config.enable_bar_aggregation is True

    def test_aggregator_with_custom_config(self):
        """测试自定义配置创建"""
        config = AggregatorConfig(
            enable_bar_aggregation=False,
            enable_quality_monitoring=False,
        )
        aggregator = MarketDataAggregator(config)

        assert aggregator.config.enable_bar_aggregation is False

    def test_register_source(self):
        """测试注册数据源"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )

        result = aggregator.register_source(config)

        assert result is True
        assert "sim_1" in aggregator.get_sources()

    def test_register_duplicate_source(self):
        """测试注册重复数据源"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )

        aggregator.register_source(config)
        result = aggregator.register_source(config)

        assert result is False

    def test_unregister_source(self):
        """测试注销数据源"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )

        aggregator.register_source(config)
        result = aggregator.unregister_source("sim_1")

        assert result is True
        assert "sim_1" not in aggregator.get_sources()

    def test_subscribe(self):
        """测试订阅"""
        aggregator = MarketDataAggregator()

        sub_id = aggregator.subscribe("AAPL", SubscriptionType.QUOTE)

        assert sub_id is not None
        assert sub_id.startswith("sub_")
        assert len(aggregator.get_subscriptions()) == 1

    def test_subscribe_multiple(self):
        """测试多个订阅"""
        aggregator = MarketDataAggregator()

        sub1 = aggregator.subscribe("AAPL", SubscriptionType.QUOTE)
        sub2 = aggregator.subscribe("AAPL", SubscriptionType.TICK)
        sub3 = aggregator.subscribe("GOOGL", SubscriptionType.QUOTE)

        assert len(aggregator.get_subscriptions()) == 3
        assert len(aggregator.get_symbol_subscriptions("AAPL")) == 2
        assert len(aggregator.get_symbol_subscriptions("GOOGL")) == 1

    def test_unsubscribe(self):
        """测试取消订阅"""
        aggregator = MarketDataAggregator()

        sub_id = aggregator.subscribe("AAPL", SubscriptionType.QUOTE)
        result = aggregator.unsubscribe(sub_id)

        assert result is True
        assert len(aggregator.get_subscriptions()) == 0

    def test_unsubscribe_invalid(self):
        """测试取消无效订阅"""
        aggregator = MarketDataAggregator()

        result = aggregator.unsubscribe("invalid_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_all(self):
        """测试连接所有数据源"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )
        aggregator.register_source(config)

        results = await aggregator.connect_all()

        assert results["sim_1"] is True

        await aggregator.disconnect_all()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启动和停止"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )
        aggregator.register_source(config)

        result = await aggregator.start()
        assert result is True

        await aggregator.stop()

    @pytest.mark.asyncio
    async def test_data_flow(self):
        """测试数据流"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )
        aggregator.register_source(config)

        received_quotes = []
        aggregator.add_callback('quote', lambda q: received_quotes.append(q))

        await aggregator.start()
        aggregator.subscribe("AAPL", SubscriptionType.QUOTE)

        await asyncio.sleep(0.5)

        await aggregator.stop()

        # 应该收到报价数据
        assert len(received_quotes) > 0
        assert received_quotes[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_latest_quote(self):
        """测试获取最新报价"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )
        aggregator.register_source(config)

        await aggregator.start()
        aggregator.subscribe("AAPL", SubscriptionType.ALL)

        await asyncio.sleep(0.5)

        quote = aggregator.get_latest_quote("AAPL")

        await aggregator.stop()

        assert quote is not None
        assert quote.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_latest_tick(self):
        """测试获取最新Tick"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )
        aggregator.register_source(config)

        await aggregator.start()
        aggregator.subscribe("AAPL", SubscriptionType.ALL)

        await asyncio.sleep(0.5)

        tick = aggregator.get_latest_tick("AAPL")

        await aggregator.stop()

        assert tick is not None
        assert tick.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_mid_price(self):
        """测试获取中间价"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )
        aggregator.register_source(config)

        await aggregator.start()
        aggregator.subscribe("AAPL", SubscriptionType.ALL)

        await asyncio.sleep(0.5)

        mid_price = aggregator.get_mid_price("AAPL")

        await aggregator.stop()

        assert mid_price is not None
        assert mid_price > 0

    def test_get_statistics(self):
        """测试获取统计信息"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )
        aggregator.register_source(config)
        aggregator.subscribe("AAPL", SubscriptionType.QUOTE)

        stats = aggregator.get_statistics()

        assert 'sources' in stats
        assert 'subscriptions' in stats
        assert 'data' in stats
        assert stats['subscriptions']['total'] == 1

    @pytest.mark.asyncio
    async def test_subscription_callback(self):
        """测试订阅回调"""
        aggregator = MarketDataAggregator()

        config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )
        aggregator.register_source(config)

        callback_data = []

        def my_callback(quote):
            callback_data.append(quote)

        await aggregator.start()
        aggregator.subscribe("AAPL", SubscriptionType.QUOTE, callback=my_callback)

        await asyncio.sleep(0.5)

        await aggregator.stop()

        # 回调应该被调用
        assert len(callback_data) > 0

    @pytest.mark.asyncio
    async def test_bar_aggregation(self):
        """测试Bar聚合"""
        config = AggregatorConfig(
            enable_bar_aggregation=True,
            bar_intervals=[BarInterval.SECOND_1],
        )
        aggregator = MarketDataAggregator(config)

        source_config = DataSourceConfig(
            source_id="sim_1",
            source_type=DataSource.SIMULATION,
            name="Test Simulator",
        )
        aggregator.register_source(source_config)

        received_bars = []
        aggregator.add_callback('bar', lambda b: received_bars.append(b))

        await aggregator.start()
        aggregator.subscribe("AAPL", SubscriptionType.BAR, interval=BarInterval.SECOND_1)

        # 等待足够时间生成Bar
        await asyncio.sleep(2.5)

        await aggregator.stop()

        # 应该收到至少一个Bar
        assert len(received_bars) >= 1


# ============== QualityMetrics Tests ==============

class TestQualityMetrics:
    """数据质量指标测试"""

    def test_quality_metrics_creation(self):
        """测试质量指标创建"""
        metrics = QualityMetrics(
            symbol="AAPL",
            source=DataSource.EXCHANGE,
            timestamp=datetime.now(),
            latency_ms=50.0,
            quality=DataQuality.GOOD,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.latency_ms == 50.0
        assert metrics.quality == DataQuality.GOOD

    def test_quality_metrics_to_dict(self):
        """测试质量指标转字典"""
        now = datetime.now()
        metrics = QualityMetrics(
            symbol="AAPL",
            source=DataSource.EXCHANGE,
            timestamp=now,
            latency_ms=50.0,
            quality=DataQuality.GOOD,
        )

        data = metrics.to_dict()

        assert data['symbol'] == "AAPL"
        assert data['latency_ms'] == 50.0
        assert data['quality'] == 'good'


# ============== Factory Function Tests ==============

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_aggregator(self):
        """测试创建聚合器"""
        aggregator = create_aggregator(
            enable_bar_aggregation=True,
            bar_intervals=['1m', '5m'],
        )

        assert aggregator is not None
        assert aggregator.config.enable_bar_aggregation is True

    def test_create_aggregator_with_invalid_interval(self):
        """测试创建聚合器(无效周期)"""
        aggregator = create_aggregator(
            bar_intervals=['1m', 'invalid'],
        )

        assert aggregator is not None

    def test_create_source_config(self):
        """测试创建数据源配置"""
        config = create_source_config(
            source_id="sim_1",
            name="Test Simulator",
            source_type='simulation',
            priority=2,
        )

        assert config.source_id == "sim_1"
        assert config.source_type == DataSource.SIMULATION
        assert config.priority == 2

    def test_create_source_config_invalid_type(self):
        """测试创建数据源配置(无效类型)"""
        config = create_source_config(
            source_id="sim_1",
            name="Test Simulator",
            source_type='invalid',
        )

        assert config.source_type == DataSource.SIMULATION

    def test_create_tick(self):
        """测试创建Tick"""
        tick = create_tick(
            symbol="AAPL",
            price=150.0,
            size=100,
            source='exchange',
        )

        assert tick.symbol == "AAPL"
        assert tick.price == 150.0
        assert tick.source == DataSource.EXCHANGE

    def test_create_quote(self):
        """测试创建Quote"""
        quote = create_quote(
            symbol="AAPL",
            bid_price=149.95,
            bid_size=500,
            ask_price=150.05,
            ask_size=300,
            source='exchange',
        )

        assert quote.symbol == "AAPL"
        assert quote.source == DataSource.EXCHANGE

    def test_create_market_depth(self):
        """测试创建市场深度"""
        depth = create_market_depth(
            symbol="AAPL",
            bids=[(150.0, 100), (149.9, 200)],
            asks=[(150.1, 150)],
            source='exchange',
        )

        assert depth.symbol == "AAPL"
        assert len(depth.bids) == 2


# ============== Integration Tests ==============

class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_multi_symbol_subscription(self):
        """测试多标的订阅"""
        aggregator = create_aggregator()

        config = create_source_config(
            source_id="sim_1",
            name="Simulator",
        )
        aggregator.register_source(config)

        await aggregator.start()

        # 订阅多个标的
        aggregator.subscribe("AAPL", SubscriptionType.ALL)
        aggregator.subscribe("GOOGL", SubscriptionType.ALL)
        aggregator.subscribe("MSFT", SubscriptionType.ALL)

        await asyncio.sleep(0.5)

        # 检查所有标的都有数据
        assert aggregator.get_latest_quote("AAPL") is not None
        assert aggregator.get_latest_quote("GOOGL") is not None
        assert aggregator.get_latest_quote("MSFT") is not None

        await aggregator.stop()

    @pytest.mark.asyncio
    async def test_quick_aggregate(self):
        """测试快速聚合"""
        result = await quick_aggregate(
            symbols=["AAPL", "GOOGL"],
            duration_sec=1.0,
        )

        assert result['ticks_collected'] > 0
        assert result['quotes_collected'] > 0
        assert 'statistics' in result

    @pytest.mark.asyncio
    async def test_callback_cleanup(self):
        """测试回调清理"""
        aggregator = create_aggregator()

        callback_count = 0

        def my_callback(quote):
            nonlocal callback_count
            callback_count += 1

        aggregator.add_callback('quote', my_callback)
        aggregator.remove_callback('quote', my_callback)

        config = create_source_config("sim_1", "Simulator")
        aggregator.register_source(config)

        await aggregator.start()
        aggregator.subscribe("AAPL", SubscriptionType.QUOTE)

        await asyncio.sleep(0.3)

        await aggregator.stop()

        # 回调应该没有被调用(因为已移除)
        assert callback_count == 0


# ============== Edge Case Tests ==============

class TestEdgeCases:
    """边界情况测试"""

    def test_get_nonexistent_quote(self):
        """测试获取不存在的报价"""
        aggregator = MarketDataAggregator()

        quote = aggregator.get_latest_quote("NONEXISTENT")
        assert quote is None

    def test_get_nonexistent_tick(self):
        """测试获取不存在的Tick"""
        aggregator = MarketDataAggregator()

        tick = aggregator.get_latest_tick("NONEXISTENT")
        assert tick is None

    def test_get_nonexistent_mid_price(self):
        """测试获取不存在的中间价"""
        aggregator = MarketDataAggregator()

        mid_price = aggregator.get_mid_price("NONEXISTENT")
        assert mid_price is None

    def test_unsubscribe_nonexistent(self):
        """测试取消不存在的订阅"""
        aggregator = MarketDataAggregator()

        result = aggregator.unsubscribe("nonexistent_sub")
        assert result is False

    def test_bar_builder_force_complete_empty(self):
        """测试空BarBuilder强制完成"""
        builder = BarBuilder("AAPL", BarInterval.MINUTE_1, DataSource.EXCHANGE)

        bar = builder.force_complete()
        assert bar is None
