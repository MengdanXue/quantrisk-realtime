"""
基准追踪器测试 (Benchmark Tracker Tests)

测试覆盖:
- 基准管理
- 成分股管理
- 组合-基准映射
- 偏离分析
- 跟踪指标计算
- 再平衡生成
- 告警管理
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio

from quantrisk_realtime.realtime.benchmark_tracker import (
    # 枚举
    BenchmarkType,
    WeightingMethod,
    RebalanceFrequency,
    DeviationType,
    TrackerState,
    AlertType,
    # 数据类
    BenchmarkConstituent,
    Benchmark,
    PortfolioHolding,
    WeightDeviation,
    TrackingMetrics,
    RebalanceOrder,
    RebalanceResult,
    TrackerAlert,
    TrackerConfig,
    # 模拟实现
    SimulatedPriceProvider,
    SimulatedPortfolioProvider,
    # 核心引擎
    BenchmarkTracker,
    # 工厂函数
    create_benchmark_tracker,
    create_benchmark,
    create_constituent,
    create_portfolio_holding,
    quick_deviation_analysis,
)


# ==================== 测试夹具 ====================

@pytest.fixture
def config():
    """创建测试配置"""
    return TrackerConfig(
        weight_tolerance=0.01,
        tracking_error_threshold=0.02,
        min_trade_value=100.0,
        max_turnover=0.3,
    )


@pytest.fixture
def price_provider():
    """创建模拟价格提供者"""
    provider = SimulatedPriceProvider()
    provider.set_price("AAPL", 150.0)
    provider.set_price("GOOGL", 100.0)
    provider.set_price("MSFT", 300.0)
    provider.set_price("AMZN", 120.0)
    provider.set_price("META", 200.0)
    return provider


@pytest.fixture
def portfolio_provider():
    """创建模拟组合提供者"""
    provider = SimulatedPortfolioProvider()

    # 设置测试组合
    holdings = [
        PortfolioHolding(symbol="AAPL", quantity=100, market_value=15000, weight=0.15),
        PortfolioHolding(symbol="GOOGL", quantity=200, market_value=20000, weight=0.20),
        PortfolioHolding(symbol="MSFT", quantity=50, market_value=15000, weight=0.15),
        PortfolioHolding(symbol="AMZN", quantity=250, market_value=30000, weight=0.30),
    ]
    provider.set_holdings("TEST_PORTFOLIO", holdings)
    provider.set_total_value("TEST_PORTFOLIO", 100000)
    provider.set_cash("TEST_PORTFOLIO", 20000)

    return provider


@pytest.fixture
def tracker(config, price_provider, portfolio_provider):
    """创建基准追踪器"""
    return BenchmarkTracker(
        config=config,
        price_provider=price_provider,
        portfolio_provider=portfolio_provider,
    )


@pytest.fixture
def sample_benchmark():
    """创建示例基准"""
    benchmark = Benchmark(
        benchmark_id="SP500_MINI",
        name="S&P 500 Mini",
        benchmark_type=BenchmarkType.INDEX,
        weighting_method=WeightingMethod.MARKET_CAP,
        rebalance_frequency=RebalanceFrequency.QUARTERLY,
    )
    benchmark.constituents = [
        BenchmarkConstituent(symbol="AAPL", name="Apple Inc.", weight=0.20, price=150.0),
        BenchmarkConstituent(symbol="GOOGL", name="Alphabet Inc.", weight=0.15, price=100.0),
        BenchmarkConstituent(symbol="MSFT", name="Microsoft Corp.", weight=0.25, price=300.0),
        BenchmarkConstituent(symbol="AMZN", name="Amazon.com Inc.", weight=0.20, price=120.0),
        BenchmarkConstituent(symbol="META", name="Meta Platforms", weight=0.20, price=200.0),
    ]
    return benchmark


# ==================== 枚举测试 ====================

class TestEnums:
    """枚举测试"""

    def test_benchmark_type_values(self):
        """测试基准类型枚举值"""
        assert BenchmarkType.INDEX.value == "index"
        assert BenchmarkType.ETF.value == "etf"
        assert BenchmarkType.CUSTOM.value == "custom"

    def test_weighting_method_values(self):
        """测试权重方法枚举值"""
        assert WeightingMethod.MARKET_CAP.value == "market_cap"
        assert WeightingMethod.EQUAL_WEIGHT.value == "equal_weight"

    def test_rebalance_frequency_values(self):
        """测试再平衡频率枚举值"""
        assert RebalanceFrequency.DAILY.value == "daily"
        assert RebalanceFrequency.QUARTERLY.value == "quarterly"

    def test_deviation_type_values(self):
        """测试偏离类型枚举值"""
        assert DeviationType.UNDERWEIGHT.value == "underweight"
        assert DeviationType.OVERWEIGHT.value == "overweight"
        assert DeviationType.MISSING.value == "missing"


# ==================== 数据类测试 ====================

class TestBenchmarkConstituent:
    """成分股测试"""

    def test_create_constituent(self):
        """测试创建成分股"""
        constituent = BenchmarkConstituent(
            symbol="AAPL",
            name="Apple Inc.",
            weight=0.20,
            price=150.0,
        )
        assert constituent.symbol == "AAPL"
        assert constituent.weight == 0.20

    def test_constituent_to_dict(self):
        """测试成分股序列化"""
        constituent = BenchmarkConstituent(
            symbol="AAPL",
            name="Apple Inc.",
            weight=0.20,
        )
        data = constituent.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["weight"] == 0.20


class TestBenchmark:
    """基准测试"""

    def test_create_benchmark(self, sample_benchmark):
        """测试创建基准"""
        assert sample_benchmark.benchmark_id == "SP500_MINI"
        assert sample_benchmark.constituent_count == 5

    def test_total_weight(self, sample_benchmark):
        """测试总权重计算"""
        assert sample_benchmark.total_weight == 1.0

    def test_get_constituent(self, sample_benchmark):
        """测试获取成分股"""
        constituent = sample_benchmark.get_constituent("AAPL")
        assert constituent is not None
        assert constituent.weight == 0.20

    def test_get_nonexistent_constituent(self, sample_benchmark):
        """测试获取不存在的成分股"""
        constituent = sample_benchmark.get_constituent("XYZ")
        assert constituent is None

    def test_benchmark_to_dict(self, sample_benchmark):
        """测试基准序列化"""
        data = sample_benchmark.to_dict()
        assert data["benchmark_id"] == "SP500_MINI"
        assert data["constituent_count"] == 5


class TestPortfolioHolding:
    """组合持仓测试"""

    def test_create_holding(self):
        """测试创建持仓"""
        holding = PortfolioHolding(
            symbol="AAPL",
            quantity=100,
            market_value=15000,
            weight=0.15,
        )
        assert holding.symbol == "AAPL"
        assert holding.quantity == 100

    def test_holding_to_dict(self):
        """测试持仓序列化"""
        holding = PortfolioHolding(
            symbol="AAPL",
            quantity=100,
            market_value=15000,
            weight=0.15,
        )
        data = holding.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["market_value"] == 15000


class TestWeightDeviation:
    """权重偏离测试"""

    def test_create_deviation(self):
        """测试创建偏离"""
        deviation = WeightDeviation(
            symbol="AAPL",
            benchmark_weight=0.20,
            portfolio_weight=0.15,
            deviation=-0.05,
            deviation_pct=-25.0,
            deviation_type=DeviationType.UNDERWEIGHT,
        )
        assert deviation.symbol == "AAPL"
        assert deviation.deviation == -0.05

    def test_deviation_to_dict(self):
        """测试偏离序列化"""
        deviation = WeightDeviation(
            symbol="AAPL",
            benchmark_weight=0.20,
            portfolio_weight=0.15,
            deviation=-0.05,
            deviation_pct=-25.0,
            deviation_type=DeviationType.UNDERWEIGHT,
        )
        data = deviation.to_dict()
        assert data["deviation_type"] == "underweight"


class TestTrackingMetrics:
    """跟踪指标测试"""

    def test_create_metrics(self):
        """测试创建跟踪指标"""
        metrics = TrackingMetrics(
            tracking_error=0.02,
            correlation=0.98,
            beta=1.05,
        )
        assert metrics.tracking_error == 0.02
        assert metrics.correlation == 0.98

    def test_metrics_to_dict(self):
        """测试跟踪指标序列化"""
        metrics = TrackingMetrics(
            tracking_error=0.02,
            total_active_share=0.15,
        )
        data = metrics.to_dict()
        assert data["tracking_error"] == 0.02


# ==================== 基准管理测试 ====================

class TestBenchmarkManagement:
    """基准管理测试"""

    def test_add_benchmark(self, tracker, sample_benchmark):
        """测试添加基准"""
        result = tracker.add_benchmark(sample_benchmark)
        assert result is True
        assert sample_benchmark.benchmark_id in tracker.benchmarks

    def test_add_duplicate_benchmark(self, tracker, sample_benchmark):
        """测试添加重复基准"""
        tracker.add_benchmark(sample_benchmark)
        result = tracker.add_benchmark(sample_benchmark)
        assert result is False

    def test_update_benchmark(self, tracker, sample_benchmark):
        """测试更新基准"""
        tracker.add_benchmark(sample_benchmark)
        result = tracker.update_benchmark(sample_benchmark.benchmark_id, {"name": "Updated Name"})
        assert result is True

        benchmark = tracker.get_benchmark(sample_benchmark.benchmark_id)
        assert benchmark.name == "Updated Name"

    def test_remove_benchmark(self, tracker, sample_benchmark):
        """测试移除基准"""
        tracker.add_benchmark(sample_benchmark)
        result = tracker.remove_benchmark(sample_benchmark.benchmark_id)
        assert result is True
        assert sample_benchmark.benchmark_id not in tracker.benchmarks

    def test_get_all_benchmarks(self, tracker, sample_benchmark):
        """测试获取所有基准"""
        tracker.add_benchmark(sample_benchmark)

        benchmark2 = create_benchmark("Test Benchmark 2")
        tracker.add_benchmark(benchmark2)

        benchmarks = tracker.get_all_benchmarks()
        assert len(benchmarks) == 2


# ==================== 成分股管理测试 ====================

class TestConstituentManagement:
    """成分股管理测试"""

    def test_add_constituent(self, tracker, sample_benchmark):
        """测试添加成分股"""
        tracker.add_benchmark(sample_benchmark)

        new_constituent = create_constituent("NVDA", "NVIDIA Corp.", 0.10)
        result = tracker.add_constituent(sample_benchmark.benchmark_id, new_constituent)
        assert result is True

        benchmark = tracker.get_benchmark(sample_benchmark.benchmark_id)
        assert benchmark.constituent_count == 6

    def test_add_duplicate_constituent(self, tracker, sample_benchmark):
        """测试添加重复成分股"""
        tracker.add_benchmark(sample_benchmark)

        duplicate = create_constituent("AAPL", "Apple Inc.", 0.10)
        result = tracker.add_constituent(sample_benchmark.benchmark_id, duplicate)
        assert result is False

    def test_update_constituent(self, tracker, sample_benchmark):
        """测试更新成分股"""
        tracker.add_benchmark(sample_benchmark)
        result = tracker.update_constituent(
            sample_benchmark.benchmark_id,
            "AAPL",
            {"weight": 0.25}
        )
        assert result is True

        benchmark = tracker.get_benchmark(sample_benchmark.benchmark_id)
        constituent = benchmark.get_constituent("AAPL")
        assert constituent.weight == 0.25

    def test_remove_constituent(self, tracker, sample_benchmark):
        """测试移除成分股"""
        tracker.add_benchmark(sample_benchmark)
        result = tracker.remove_constituent(sample_benchmark.benchmark_id, "AAPL")
        assert result is True

        benchmark = tracker.get_benchmark(sample_benchmark.benchmark_id)
        assert benchmark.constituent_count == 4

    def test_update_constituent_prices(self, tracker, sample_benchmark):
        """测试更新成分股价格"""
        tracker.add_benchmark(sample_benchmark)
        result = tracker.update_constituent_prices(sample_benchmark.benchmark_id)
        assert result is True


# ==================== 组合-基准映射测试 ====================

class TestPortfolioBenchmarkMapping:
    """组合-基准映射测试"""

    def test_assign_benchmark(self, tracker, sample_benchmark):
        """测试分配基准"""
        tracker.add_benchmark(sample_benchmark)
        result = tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)
        assert result is True

    def test_assign_nonexistent_benchmark(self, tracker):
        """测试分配不存在的基准"""
        result = tracker.assign_benchmark("TEST_PORTFOLIO", "NONEXISTENT")
        assert result is False

    def test_unassign_benchmark(self, tracker, sample_benchmark):
        """测试取消基准分配"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        result = tracker.unassign_benchmark("TEST_PORTFOLIO")
        assert result is True
        assert tracker.get_portfolio_benchmark("TEST_PORTFOLIO") is None

    def test_get_portfolio_benchmark(self, tracker, sample_benchmark):
        """测试获取组合基准"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        benchmark_id = tracker.get_portfolio_benchmark("TEST_PORTFOLIO")
        assert benchmark_id == sample_benchmark.benchmark_id


# ==================== 偏离分析测试 ====================

class TestDeviationAnalysis:
    """偏离分析测试"""

    def test_analyze_deviations(self, tracker, sample_benchmark):
        """测试偏离分析"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        deviations = tracker.analyze_deviations("TEST_PORTFOLIO")
        assert len(deviations) > 0

    def test_deviation_types(self, tracker, sample_benchmark):
        """测试偏离类型"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        deviations = tracker.analyze_deviations("TEST_PORTFOLIO")

        # 检查是否有各种偏离类型
        deviation_types = {d.deviation_type for d in deviations}
        assert len(deviation_types) > 0

    def test_missing_constituent_deviation(self, tracker, sample_benchmark):
        """测试缺失成分股偏离"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        deviations = tracker.analyze_deviations("TEST_PORTFOLIO")

        # META 在基准中但不在组合中
        meta_deviation = next((d for d in deviations if d.symbol == "META"), None)
        assert meta_deviation is not None
        assert meta_deviation.deviation_type == DeviationType.MISSING

    def test_no_benchmark_returns_empty(self, tracker):
        """测试无基准返回空"""
        deviations = tracker.analyze_deviations("NO_BENCHMARK_PORTFOLIO")
        assert deviations == []

    def test_deviation_callback(self, tracker, sample_benchmark):
        """测试偏离回调"""
        callback_data = []

        def on_deviation(portfolio_id, deviations):
            callback_data.append((portfolio_id, deviations))

        tracker.on_deviation(on_deviation)
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        tracker.analyze_deviations("TEST_PORTFOLIO")
        assert len(callback_data) == 1


# ==================== 跟踪指标测试 ====================

class TestTrackingMetricsCalculation:
    """跟踪指标计算测试"""

    def test_calculate_tracking_metrics(self, tracker, sample_benchmark):
        """测试计算跟踪指标"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        metrics = tracker.calculate_tracking_metrics("TEST_PORTFOLIO")
        assert metrics is not None
        assert metrics.total_active_share >= 0

    def test_tracking_error_calculation(self, tracker, sample_benchmark):
        """测试跟踪误差计算"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        metrics = tracker.calculate_tracking_metrics("TEST_PORTFOLIO")
        assert metrics.tracking_error >= 0

    def test_active_share_calculation(self, tracker, sample_benchmark):
        """测试主动份额计算"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        metrics = tracker.calculate_tracking_metrics("TEST_PORTFOLIO")
        # 主动份额应在 0-1 之间
        assert 0 <= metrics.total_active_share <= 1


# ==================== 再平衡测试 ====================

class TestRebalancing:
    """再平衡测试"""

    def test_generate_rebalance(self, tracker, sample_benchmark):
        """测试生成再平衡"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        result = tracker.generate_rebalance("TEST_PORTFOLIO")
        assert result is not None
        assert result.benchmark_id == sample_benchmark.benchmark_id

    def test_rebalance_orders_generated(self, tracker, sample_benchmark):
        """测试再平衡订单生成"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        result = tracker.generate_rebalance("TEST_PORTFOLIO")
        assert len(result.orders) > 0

    def test_rebalance_turnover_limit(self, tracker, sample_benchmark):
        """测试再平衡换手率限制"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        result = tracker.generate_rebalance("TEST_PORTFOLIO", max_turnover=0.1)
        assert result.total_turnover <= 0.1

    def test_rebalance_callback(self, tracker, sample_benchmark):
        """测试再平衡回调"""
        callback_data = []

        def on_rebalance(result):
            callback_data.append(result)

        tracker.on_rebalance(on_rebalance)
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        tracker.generate_rebalance("TEST_PORTFOLIO")
        assert len(callback_data) == 1

    def test_rebalance_history(self, tracker, sample_benchmark):
        """测试再平衡历史"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        tracker.generate_rebalance("TEST_PORTFOLIO")
        tracker.generate_rebalance("TEST_PORTFOLIO")

        history = tracker.get_rebalance_history()
        assert len(history) == 2

    def test_rebalance_no_benchmark_raises(self, tracker):
        """测试无基准再平衡抛异常"""
        with pytest.raises(ValueError):
            tracker.generate_rebalance("NO_BENCHMARK_PORTFOLIO")


# ==================== 告警测试 ====================

class TestAlerts:
    """告警测试"""

    def test_get_alerts(self, tracker, sample_benchmark):
        """测试获取告警"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        # 触发告警检查
        tracker.check_and_alert("TEST_PORTFOLIO")

        alerts = tracker.get_alerts()
        # 可能有告警也可能没有，取决于偏离程度
        assert isinstance(alerts, list)

    def test_acknowledge_alert(self, tracker, sample_benchmark):
        """测试确认告警"""
        # 手动创建告警
        alert = tracker._create_alert(
            AlertType.WEIGHT_DEVIATION,
            sample_benchmark.benchmark_id,
            "Test alert",
            severity="warning",
        )

        result = tracker.acknowledge_alert(alert.alert_id, "test_user")
        assert result is True

        alerts = tracker.get_alerts(acknowledged=True)
        assert any(a.alert_id == alert.alert_id for a in alerts)

    def test_alert_callback(self, tracker, sample_benchmark):
        """测试告警回调"""
        callback_data = []

        def on_alert(alert):
            callback_data.append(alert)

        tracker.on_alert(on_alert)
        tracker.add_benchmark(sample_benchmark)

        tracker._create_alert(
            AlertType.TRACKING_ERROR,
            sample_benchmark.benchmark_id,
            "Test alert",
        )

        assert len(callback_data) == 1

    def test_filter_alerts_by_type(self, tracker, sample_benchmark):
        """测试按类型过滤告警"""
        tracker.add_benchmark(sample_benchmark)

        tracker._create_alert(AlertType.TRACKING_ERROR, sample_benchmark.benchmark_id, "TE alert")
        tracker._create_alert(AlertType.WEIGHT_DEVIATION, sample_benchmark.benchmark_id, "WD alert")

        te_alerts = tracker.get_alerts(alert_type=AlertType.TRACKING_ERROR)
        assert all(a.alert_type == AlertType.TRACKING_ERROR for a in te_alerts)


# ==================== 监控控制测试 ====================

class TestMonitoringControl:
    """监控控制测试"""

    @pytest.mark.asyncio
    async def test_start_stop_tracking(self, tracker):
        """测试启动停止追踪"""
        await tracker.start_tracking()
        assert tracker.state == TrackerState.TRACKING

        await asyncio.sleep(0.1)

        await tracker.stop_tracking()
        assert tracker.state == TrackerState.STOPPED

    def test_get_statistics(self, tracker, sample_benchmark):
        """测试获取统计"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)
        tracker.analyze_deviations("TEST_PORTFOLIO")

        stats = tracker.get_statistics()
        assert "benchmarks_count" in stats
        assert stats["benchmarks_count"] == 1


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_benchmark_tracker(self):
        """测试创建基准追踪器"""
        tracker = create_benchmark_tracker()
        assert tracker is not None

    def test_create_benchmark(self):
        """测试创建基准"""
        benchmark = create_benchmark(
            name="Test Benchmark",
            benchmark_type=BenchmarkType.INDEX,
        )
        assert benchmark.name == "Test Benchmark"
        assert benchmark.benchmark_id is not None

    def test_create_constituent(self):
        """测试创建成分股"""
        constituent = create_constituent(
            symbol="AAPL",
            name="Apple Inc.",
            weight=0.20,
        )
        assert constituent.symbol == "AAPL"
        assert constituent.weight == 0.20

    def test_create_portfolio_holding(self):
        """测试创建组合持仓"""
        holding = create_portfolio_holding(
            symbol="AAPL",
            quantity=100,
            market_value=15000,
            total_portfolio_value=100000,
        )
        assert holding.symbol == "AAPL"
        assert holding.weight == 0.15

    def test_quick_deviation_analysis(self):
        """测试快速偏离分析"""
        portfolio_holdings = [
            {"symbol": "AAPL", "quantity": 100, "market_value": 15000},
            {"symbol": "GOOGL", "quantity": 200, "market_value": 20000},
        ]
        benchmark_weights = {
            "AAPL": 0.20,
            "GOOGL": 0.15,
            "MSFT": 0.25,
        }

        deviations = quick_deviation_analysis(
            portfolio_holdings,
            benchmark_weights,
            total_portfolio_value=100000,
        )
        assert len(deviations) > 0


# ==================== 边界情况测试 ====================

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_portfolio(self, tracker, sample_benchmark):
        """测试空组合"""
        tracker.add_benchmark(sample_benchmark)
        tracker.assign_benchmark("EMPTY_PORTFOLIO", sample_benchmark.benchmark_id)

        # 设置空组合
        tracker.portfolio_provider.set_holdings("EMPTY_PORTFOLIO", [])
        tracker.portfolio_provider.set_total_value("EMPTY_PORTFOLIO", 0)

        deviations = tracker.analyze_deviations("EMPTY_PORTFOLIO")
        assert len(deviations) == 0  # 组合值为0，返回空

    def test_empty_benchmark(self, tracker):
        """测试空基准"""
        empty_benchmark = create_benchmark("Empty")
        tracker.add_benchmark(empty_benchmark)
        tracker.assign_benchmark("TEST_PORTFOLIO", empty_benchmark.benchmark_id)

        deviations = tracker.analyze_deviations("TEST_PORTFOLIO")
        # 应该有EXTRA类型的偏离（组合中有但基准中没有）
        assert all(d.deviation_type == DeviationType.EXTRA for d in deviations if d.portfolio_weight > 0)

    def test_update_nonexistent_benchmark(self, tracker):
        """测试更新不存在的基准"""
        result = tracker.update_benchmark("NONEXISTENT", {"name": "New Name"})
        assert result is False

    def test_remove_nonexistent_benchmark(self, tracker):
        """测试移除不存在的基准"""
        result = tracker.remove_benchmark("NONEXISTENT")
        assert result is False

    def test_zero_weight_constituent(self, tracker):
        """测试零权重成分股"""
        benchmark = create_benchmark("ZeroWeight")
        benchmark.constituents.append(create_constituent("ZERO", "Zero Weight", 0.0))
        tracker.add_benchmark(benchmark)

        assert benchmark.total_weight == 0.0


# ==================== 序列化测试 ====================

class TestSerialization:
    """序列化测试"""

    def test_config_to_dict(self, config):
        """测试配置序列化"""
        data = config.to_dict()
        assert "weight_tolerance" in data
        assert "tracking_error_threshold" in data

    def test_rebalance_order_to_dict(self):
        """测试再平衡订单序列化"""
        order = RebalanceOrder(
            order_id="ORD001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            target_weight=0.20,
            current_weight=0.15,
        )
        data = order.to_dict()
        assert data["order_id"] == "ORD001"
        assert data["side"] == "buy"

    def test_rebalance_result_to_dict(self):
        """测试再平衡结果序列化"""
        result = RebalanceResult(
            rebalance_id="REB001",
            benchmark_id="SP500",
            total_buy_value=50000,
            total_sell_value=30000,
        )
        data = result.to_dict()
        assert data["rebalance_id"] == "REB001"

    def test_tracker_alert_to_dict(self):
        """测试告警序列化"""
        alert = TrackerAlert(
            alert_id="ALT001",
            alert_type=AlertType.TRACKING_ERROR,
            benchmark_id="SP500",
            message="Tracking error exceeded",
        )
        data = alert.to_dict()
        assert data["alert_type"] == "tracking_error"


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""

    def test_full_tracking_workflow(self, tracker, sample_benchmark):
        """测试完整追踪工作流"""
        # 1. 添加基准
        tracker.add_benchmark(sample_benchmark)

        # 2. 分配给组合
        tracker.assign_benchmark("TEST_PORTFOLIO", sample_benchmark.benchmark_id)

        # 3. 更新成分股价格
        tracker.update_constituent_prices(sample_benchmark.benchmark_id)

        # 4. 分析偏离
        deviations = tracker.analyze_deviations("TEST_PORTFOLIO")
        assert len(deviations) > 0

        # 5. 计算跟踪指标
        metrics = tracker.calculate_tracking_metrics("TEST_PORTFOLIO")
        assert metrics.total_active_share > 0

        # 6. 生成再平衡
        rebalance = tracker.generate_rebalance("TEST_PORTFOLIO")
        assert len(rebalance.orders) > 0

        # 7. 检查告警
        tracker.check_and_alert("TEST_PORTFOLIO")

        # 8. 验证统计
        stats = tracker.get_statistics()
        assert stats["rebalances_generated"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
