"""
数据源管理器测试
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from quantrisk_realtime.datasources.manager import (
    DataSourceManager,
    ManagerConfig,
    SelectionStrategy,
    AggregationMethod,
    DataSourceEntry,
    HealthStatus,
    AggregatedQuote,
    create_manager,
)
from quantrisk_realtime.datasources.base import (
    DataSource,
    DataSourceConfig,
    DataSourceState,
    DataSourceError,
    MarketType,
    DataFrequency,
    OHLCV,
    Quote,
    DataSourceStats,
)


class MockDataSource(DataSource):
    """模拟数据源"""

    def __init__(self, name: str = "mock", markets: list = None):
        config = DataSourceConfig(name=name)
        super().__init__(config)
        self._name = name
        self._markets = markets or [MarketType.CRYPTO]
        self._mock_ohlcv = []
        self._mock_quote = None
        self._should_fail = False
        self._delay = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_markets(self):
        return self._markets

    async def connect(self) -> bool:
        self.state = DataSourceState.RUNNING
        return True

    async def disconnect(self) -> None:
        self.state = DataSourceState.IDLE

    async def fetch_ohlcv(self, symbol, frequency=None, start_date=None, end_date=None, limit=100):
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_fail:
            raise DataSourceError(f"{self.name} failed")
        return self._mock_ohlcv

    async def fetch_quote(self, symbol):
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_fail:
            raise DataSourceError(f"{self.name} failed")
        return self._mock_quote


class TestManagerConfig:
    """ManagerConfig测试"""

    def test_defaults(self):
        """测试默认配置"""
        config = ManagerConfig()

        assert config.selection_strategy == SelectionStrategy.PRIORITY
        assert config.aggregation_method == AggregationMethod.FIRST
        assert config.enable_fallback is True
        assert config.fallback_timeout == 5.0
        assert config.health_check_interval == 60.0

    def test_custom_config(self):
        """测试自定义配置"""
        config = ManagerConfig(
            selection_strategy=SelectionStrategy.ROUND_ROBIN,
            enable_fallback=False,
            fallback_timeout=10.0,
        )

        assert config.selection_strategy == SelectionStrategy.ROUND_ROBIN
        assert config.enable_fallback is False
        assert config.fallback_timeout == 10.0

    def test_to_dict(self):
        """测试转字典"""
        config = ManagerConfig()
        d = config.to_dict()

        assert d['selection_strategy'] == 'priority'
        assert d['enable_fallback'] is True


class TestDataSourceEntry:
    """DataSourceEntry测试"""

    def test_creation(self):
        """测试创建"""
        source = MockDataSource("test")
        entry = DataSourceEntry(
            source=source,
            priority=10,
            weight=1.5,
            markets=[MarketType.CRYPTO],
        )

        assert entry.source == source
        assert entry.priority == 10
        assert entry.weight == 1.5
        assert entry.enabled is True

    def test_to_dict(self):
        """测试转字典"""
        source = MockDataSource("test")
        entry = DataSourceEntry(
            source=source,
            priority=5,
            markets=[MarketType.CRYPTO],
        )

        d = entry.to_dict()
        assert d['name'] == 'test'
        assert d['priority'] == 5
        assert d['enabled'] is True


class TestHealthStatus:
    """HealthStatus测试"""

    def test_creation(self):
        """测试创建"""
        status = HealthStatus(
            source_name="test",
            is_healthy=True,
            state=DataSourceState.RUNNING,
            latency_ms=50.0,
            error_rate=0.01,
            last_check=datetime.now(),
        )

        assert status.source_name == "test"
        assert status.is_healthy is True
        assert status.latency_ms == 50.0

    def test_to_dict(self):
        """测试转字典"""
        status = HealthStatus(
            source_name="test",
            is_healthy=True,
            state=DataSourceState.RUNNING,
            latency_ms=50.0,
            error_rate=0.01,
            last_check=datetime.now(),
        )

        d = status.to_dict()
        assert d['source_name'] == 'test'
        assert d['is_healthy'] is True
        assert d['state'] == 'running'


class TestAggregatedQuote:
    """AggregatedQuote测试"""

    def test_creation(self):
        """测试创建"""
        quotes = {
            'source1': Quote(symbol='BTC', timestamp=datetime.now(), bid=100, ask=101),
            'source2': Quote(symbol='BTC', timestamp=datetime.now(), bid=99, ask=102),
        }

        agg = AggregatedQuote(
            symbol='BTC',
            timestamp=datetime.now(),
            quotes=quotes,
            best_bid=100,
            best_ask=101,
            avg_price=100.5,
            spread=1.0,
            sources_count=2,
        )

        assert agg.symbol == 'BTC'
        assert agg.best_bid == 100
        assert agg.best_ask == 101
        assert agg.sources_count == 2

    def test_to_dict(self):
        """测试转字典"""
        agg = AggregatedQuote(
            symbol='BTC',
            timestamp=datetime.now(),
            quotes={},
            best_bid=100,
            best_ask=101,
            avg_price=100.5,
            spread=1.0,
            sources_count=0,
        )

        d = agg.to_dict()
        assert d['symbol'] == 'BTC'
        assert d['best_bid'] == 100


class TestDataSourceManager:
    """DataSourceManager测试"""

    def test_creation(self):
        """测试创建"""
        manager = DataSourceManager()

        assert manager.config.selection_strategy == SelectionStrategy.PRIORITY
        assert len(manager._sources) == 0

    def test_register(self):
        """测试注册数据源"""
        manager = DataSourceManager()
        source = MockDataSource("test")

        manager.register(source, priority=10)

        assert "test" in manager._sources
        assert manager._sources["test"].priority == 10

    def test_unregister(self):
        """测试注销数据源"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        manager.register(source)

        result = manager.unregister("test")

        assert result is True
        assert "test" not in manager._sources

    def test_unregister_nonexistent(self):
        """测试注销不存在的数据源"""
        manager = DataSourceManager()

        result = manager.unregister("nonexistent")

        assert result is False

    def test_get_source(self):
        """测试获取数据源"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        manager.register(source)

        result = manager.get_source("test")

        assert result == source

    def test_get_source_nonexistent(self):
        """测试获取不存在的数据源"""
        manager = DataSourceManager()

        result = manager.get_source("nonexistent")

        assert result is None

    def test_get_sources_for_market(self):
        """测试获取市场数据源"""
        manager = DataSourceManager()
        crypto_source = MockDataSource("crypto", [MarketType.CRYPTO])
        stock_source = MockDataSource("stock", [MarketType.STOCK_US])

        manager.register(crypto_source)
        manager.register(stock_source)

        crypto_sources = manager.get_sources_for_market(MarketType.CRYPTO)
        stock_sources = manager.get_sources_for_market(MarketType.STOCK_US)

        assert len(crypto_sources) == 1
        assert crypto_sources[0].name == "crypto"
        assert len(stock_sources) == 1
        assert stock_sources[0].name == "stock"

    def test_enable_disable_source(self):
        """测试启用/禁用数据源"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        manager.register(source)

        # 禁用
        manager.disable_source("test")
        assert manager._sources["test"].enabled is False

        # 启用
        manager.enable_source("test")
        assert manager._sources["test"].enabled is True

    def test_get_source_names(self):
        """测试获取数据源名称列表"""
        manager = DataSourceManager()
        manager.register(MockDataSource("source1"))
        manager.register(MockDataSource("source2"))

        names = manager.get_source_names()

        assert "source1" in names
        assert "source2" in names


class TestDataSourceManagerAsync:
    """DataSourceManager异步测试"""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启动和停止"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        manager.register(source)

        await manager.start()
        assert manager._running is True
        assert source.state == DataSourceState.RUNNING

        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self):
        """测试获取OHLCV"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        source._mock_ohlcv = [
            OHLCV(
                timestamp=datetime.now(),
                open=100, high=105, low=95, close=102,
                volume=1000, symbol='BTC', source='test'
            )
        ]
        manager.register(source)
        await manager.start()

        result = await manager.fetch_ohlcv("BTC")

        assert len(result) == 1
        assert result[0].close == 102

        await manager.stop()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_with_fallback(self):
        """测试带故障转移的OHLCV获取"""
        manager = DataSourceManager()

        # 第一个数据源会失败
        source1 = MockDataSource("source1")
        source1._should_fail = True
        manager.register(source1, priority=10)

        # 第二个数据源正常
        source2 = MockDataSource("source2")
        source2._mock_ohlcv = [
            OHLCV(
                timestamp=datetime.now(),
                open=100, high=105, low=95, close=102,
                volume=1000, symbol='BTC', source='source2'
            )
        ]
        manager.register(source2, priority=5)

        await manager.start()

        result = await manager.fetch_ohlcv("BTC")

        assert len(result) == 1
        assert result[0].source == 'source2'

        await manager.stop()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_all_fail(self):
        """测试所有数据源失败"""
        manager = DataSourceManager()

        source1 = MockDataSource("source1")
        source1._should_fail = True
        manager.register(source1)

        await manager.start()

        with pytest.raises(DataSourceError):
            await manager.fetch_ohlcv("BTC")

        await manager.stop()

    @pytest.mark.asyncio
    async def test_fetch_quote(self):
        """测试获取报价"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        source._mock_quote = Quote(
            symbol='BTC',
            timestamp=datetime.now(),
            bid=100,
            ask=101,
            last_price=100.5,
        )
        manager.register(source)
        await manager.start()

        result = await manager.fetch_quote("BTC")

        assert result is not None
        assert result.bid == 100
        assert result.ask == 101

        await manager.stop()

    @pytest.mark.asyncio
    async def test_fetch_aggregated_quote(self):
        """测试获取聚合报价"""
        manager = DataSourceManager()

        source1 = MockDataSource("source1")
        source1._mock_quote = Quote(
            symbol='BTC',
            timestamp=datetime.now(),
            bid=100,
            ask=102,
            last_price=101,
        )
        manager.register(source1)

        source2 = MockDataSource("source2")
        source2._mock_quote = Quote(
            symbol='BTC',
            timestamp=datetime.now(),
            bid=99,
            ask=101,
            last_price=100,
        )
        manager.register(source2)

        await manager.start()

        result = await manager.fetch_aggregated_quote("BTC")

        assert result is not None
        assert result.best_bid == 100  # max of bids
        assert result.best_ask == 101  # min of asks
        assert result.sources_count == 2

        await manager.stop()

    @pytest.mark.asyncio
    async def test_fetch_quotes_batch(self):
        """测试批量获取报价"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        source._mock_quote = Quote(
            symbol='BTC',
            timestamp=datetime.now(),
            bid=100,
            ask=101,
        )
        manager.register(source)
        await manager.start()

        result = await manager.fetch_quotes(["BTC", "ETH"])

        # 由于mock只返回一个quote，两个symbol都会得到同样的结果
        assert "BTC" in result or "ETH" in result

        await manager.stop()

    @pytest.mark.asyncio
    async def test_timeout_fallback(self):
        """测试超时后故障转移"""
        config = ManagerConfig(fallback_timeout=0.1)
        manager = DataSourceManager(config)

        # 第一个数据源很慢
        source1 = MockDataSource("slow")
        source1._delay = 1.0  # 1秒延迟
        source1._mock_ohlcv = [
            OHLCV(
                timestamp=datetime.now(),
                open=100, high=105, low=95, close=102,
                volume=1000, symbol='BTC', source='slow'
            )
        ]
        manager.register(source1, priority=10)

        # 第二个数据源快速
        source2 = MockDataSource("fast")
        source2._mock_ohlcv = [
            OHLCV(
                timestamp=datetime.now(),
                open=100, high=105, low=95, close=103,
                volume=1000, symbol='BTC', source='fast'
            )
        ]
        manager.register(source2, priority=5)

        await manager.start()

        result = await manager.fetch_ohlcv("BTC")

        assert result[0].source == 'fast'

        await manager.stop()


class TestSelectionStrategies:
    """选择策略测试"""

    @pytest.mark.asyncio
    async def test_priority_selection(self):
        """测试优先级选择"""
        config = ManagerConfig(selection_strategy=SelectionStrategy.PRIORITY)
        manager = DataSourceManager(config)

        source_low = MockDataSource("low")
        source_high = MockDataSource("high")

        manager.register(source_low, priority=1)
        manager.register(source_high, priority=10)

        await manager.start()

        sources = manager._select_sources()

        assert sources[0].name == "high"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_round_robin_selection(self):
        """测试轮询选择"""
        config = ManagerConfig(selection_strategy=SelectionStrategy.ROUND_ROBIN)
        manager = DataSourceManager(config)

        source1 = MockDataSource("source1")
        source2 = MockDataSource("source2")

        manager.register(source1)
        manager.register(source2)

        await manager.start()

        # 第一次选择
        sources1 = manager._select_sources()
        # 第二次选择
        sources2 = manager._select_sources()

        # 轮询应该返回不同顺序
        assert sources1[0].name != sources2[0].name or len(sources1) == 1

        await manager.stop()


class TestHealthCheck:
    """健康检查测试"""

    @pytest.mark.asyncio
    async def test_get_health_status(self):
        """测试获取健康状态"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        manager.register(source)

        await manager.start()
        await manager._check_all_health()

        status = manager.get_health_status()

        assert "test" in status
        assert status["test"].source_name == "test"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_check_health_single(self):
        """测试检查单个数据源健康"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        manager.register(source)

        await manager.start()

        status = await manager.check_health("test")

        assert status is not None
        assert status.source_name == "test"

        await manager.stop()


class TestStats:
    """统计信息测试"""

    def test_get_stats(self):
        """测试获取统计"""
        manager = DataSourceManager()
        source = MockDataSource("test")
        manager.register(source)

        stats = manager.get_stats()

        assert stats['sources_count'] == 1
        assert stats['enabled_count'] == 1
        assert 'test' in stats['sources']


class TestFactoryFunction:
    """工厂函数测试"""

    def test_create_manager(self):
        """测试创建管理器"""
        manager = create_manager()

        assert isinstance(manager, DataSourceManager)
        assert manager.config.selection_strategy == SelectionStrategy.PRIORITY

    def test_create_manager_custom(self):
        """测试创建自定义管理器"""
        manager = create_manager(
            selection_strategy=SelectionStrategy.FASTEST,
            enable_fallback=False,
        )

        assert manager.config.selection_strategy == SelectionStrategy.FASTEST
        assert manager.config.enable_fallback is False


class TestNoFallback:
    """禁用故障转移测试"""

    @pytest.mark.asyncio
    async def test_no_fallback_on_failure(self):
        """测试禁用故障转移时失败"""
        config = ManagerConfig(enable_fallback=False)
        manager = DataSourceManager(config)

        source1 = MockDataSource("source1")
        source1._should_fail = True
        manager.register(source1, priority=10)

        source2 = MockDataSource("source2")
        source2._mock_ohlcv = [
            OHLCV(
                timestamp=datetime.now(),
                open=100, high=105, low=95, close=102,
                volume=1000, symbol='BTC', source='source2'
            )
        ]
        manager.register(source2, priority=5)

        await manager.start()

        # 第一个失败后不应该尝试第二个
        with pytest.raises(DataSourceError):
            await manager.fetch_ohlcv("BTC")

        await manager.stop()
