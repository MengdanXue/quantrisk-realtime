"""
执行质量监控器测试
Execution Quality Monitor Tests
"""

import pytest
import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from quantrisk_realtime.realtime.execution_quality_monitor import (
    # 枚举
    BenchmarkType,
    ExecutionSide,
    ExecutionStatus,
    AlertType,
    AlertSeverity,
    MonitorState,
    # 数据类
    ExecutionRecord,
    CostBreakdown,
    QualityMetrics,
    ExecutionAlert,
    ExecutionStatistics,
    MonitorConfig,
    # 计算器
    QualityCalculator,
    # 核心监控器
    ExecutionQualityMonitor,
    # 工厂函数
    create_execution_monitor,
    create_execution_record,
    quick_quality_analysis,
)


# ==================== 测试数据 ====================

@pytest.fixture
def basic_config():
    """基础配置"""
    return MonitorConfig(
        high_slippage_threshold=10.0,
        high_impact_threshold=15.0,
        enable_alerts=True,
    )


@pytest.fixture
def basic_execution_record():
    """基础执行记录"""
    return ExecutionRecord(
        execution_id="exec-001",
        symbol="AAPL",
        side=ExecutionSide.BUY,
        order_id="order-001",
        order_quantity=1000.0,
        order_time=datetime.now(),
        executed_quantity=1000.0,
        executed_price=150.50,
        execution_time=datetime.now(),
        status=ExecutionStatus.FILLED,
        arrival_price=150.00,
        decision_time=datetime.now() - timedelta(seconds=5),
        vwap_price=150.30,
        closing_price=150.80,
        market_volume=100000.0,
        broker="Broker-A",
        algo_name="TWAP",
    )


@pytest.fixture
def quality_monitor(basic_config):
    """质量监控器"""
    return ExecutionQualityMonitor(basic_config)


# ==================== ExecutionRecord 测试 ====================

class TestExecutionRecord:
    """执行记录测试"""

    def test_execution_record_creation(self):
        """测试创建执行记录"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            arrival_price=150.00,
        )

        assert record.execution_id == "exec-001"
        assert record.symbol == "AAPL"
        assert record.side == ExecutionSide.BUY
        assert record.order_quantity == 1000.0

    def test_execution_record_with_optional_fields(self):
        """测试包含可选字段的创建"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.SELL,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            arrival_price=150.00,
            executed_quantity=800.0,
            executed_price=149.50,
            status=ExecutionStatus.PARTIAL,
            broker="Broker-A",
            algo_name="VWAP",
        )

        assert record.executed_quantity == 800.0
        assert record.executed_price == 149.50
        assert record.status == ExecutionStatus.PARTIAL
        assert record.broker == "Broker-A"
        assert record.algo_name == "VWAP"

    def test_execution_record_to_dict(self):
        """测试转换为字典"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            arrival_price=150.00,
        )

        data = record.to_dict()
        assert isinstance(data, dict)
        assert data['execution_id'] == "exec-001"
        assert data['symbol'] == "AAPL"
        assert data['side'] == "buy"


# ==================== MonitorConfig 测试 ====================

class TestMonitorConfig:
    """监控配置测试"""

    def test_config_default_values(self):
        """测试默认值"""
        config = MonitorConfig()

        assert config.high_slippage_threshold == 10.0
        assert config.high_impact_threshold == 15.0
        assert config.enable_alerts is True

    def test_config_custom_values(self):
        """测试自定义值"""
        config = MonitorConfig(
            high_slippage_threshold=20.0,
            high_impact_threshold=25.0,
            slow_execution_threshold=600.0,
            enable_alerts=False,
        )

        assert config.high_slippage_threshold == 20.0
        assert config.high_impact_threshold == 25.0
        assert config.slow_execution_threshold == 600.0
        assert config.enable_alerts is False

    def test_config_to_dict(self):
        """测试配置序列化"""
        config = MonitorConfig()
        data = config.to_dict()

        assert isinstance(data, dict)
        assert 'high_slippage_threshold' in data
        assert 'enable_alerts' in data


# ==================== QualityCalculator 测试 ====================

class TestQualityCalculator:
    """质量计算器测试"""

    def test_implementation_shortfall_buy(self):
        """测试买单实施缺口计算"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=150.50,
            arrival_price=150.00,
        )

        calculator = QualityCalculator()
        is_value, components = calculator.calculate_implementation_shortfall(
            record, 150.00
        )

        # 买入时，执行价格高于基准价格，IS为正
        assert is_value > 0
        assert 'execution_cost' in components

    def test_implementation_shortfall_sell(self):
        """测试卖单实施缺口计算"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.SELL,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=149.50,
            arrival_price=150.00,
        )

        calculator = QualityCalculator()
        is_value, components = calculator.calculate_implementation_shortfall(
            record, 150.00
        )

        # 卖出时，执行价格低于基准价格，IS为正
        assert is_value > 0

    def test_vwap_slippage_calculation(self):
        """测试VWAP滑点计算"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=150.50,
            vwap_price=150.30,
            arrival_price=150.00,
        )

        calculator = QualityCalculator()
        slippage = calculator.calculate_vwap_slippage(record)

        # 买入时，执行价格高于VWAP，滑点为正
        assert slippage > 0

    def test_arrival_price_slippage_calculation(self):
        """测试到达价格滑点计算"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=150.50,
            arrival_price=150.00,
        )

        calculator = QualityCalculator()
        slippage = calculator.calculate_arrival_price_slippage(record)

        # 买入时，执行价格高于到达价格，滑点为正
        assert slippage > 0
        # 计算验证: (150.50 - 150.00) / 150.00 * 10000 ≈ 33.33 bps
        expected_slippage = (0.50 / 150.00) * 10000
        assert abs(slippage - expected_slippage) < 0.1

    def test_market_impact_calculation(self):
        """测试市场冲击成本计算"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=150.50,
            arrival_price=150.00,
        )

        calculator = QualityCalculator()
        impact = calculator.calculate_market_impact(
            record,
            initial_price=150.00,
            final_price=150.80,
        )

        # 买入后价格上涨，市场冲击为正
        assert impact > 0

    def test_timing_cost_calculation(self):
        """测试时机成本计算"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=150.50,
            arrival_price=150.00,
        )

        calculator = QualityCalculator()
        timing_cost = calculator.calculate_timing_cost(
            record,
            decision_price=149.80,
            order_price=150.00,
        )

        # 决策到下单期间价格上涨，时机成本为正
        assert timing_cost > 0

    def test_opportunity_cost_calculation(self):
        """测试机会成本计算"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=800.0,  # 部分成交
            executed_price=150.50,
            arrival_price=150.00,
        )

        calculator = QualityCalculator()
        opp_cost = calculator.calculate_opportunity_cost(
            record,
            target_quantity=1000.0,
            closing_price=151.00,
        )

        # 有未成交部分，价格上涨，机会成本为正
        assert opp_cost > 0

    def test_opportunity_cost_full_fill(self):
        """测试完全成交时机会成本为0"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,  # 完全成交
            executed_price=150.50,
            arrival_price=150.00,
        )

        calculator = QualityCalculator()
        opp_cost = calculator.calculate_opportunity_cost(
            record,
            target_quantity=1000.0,
            closing_price=151.00,
        )

        # 完全成交，机会成本应为0
        assert opp_cost == 0.0

    def test_quality_score_calculation(self):
        """测试质量得分计算"""
        calculator = QualityCalculator()

        # 优秀执行
        score = calculator.calculate_quality_score(
            slippage_bps=2.0,
            impact_bps=3.0,
            fill_rate=1.0,
            execution_duration=180.0,
            target_duration=300.0,
        )

        assert 70 <= score <= 100

    def test_quality_score_poor_execution(self):
        """测试低质量执行得分"""
        calculator = QualityCalculator()

        # 差劲执行
        score = calculator.calculate_quality_score(
            slippage_bps=50.0,  # 高滑点
            impact_bps=40.0,     # 高冲击
            fill_rate=0.5,       # 低成交率
            execution_duration=900.0,  # 慢执行
            target_duration=300.0,
        )

        assert 0 <= score < 50

    def test_quality_score_bounds(self):
        """测试质量得分边界"""
        calculator = QualityCalculator()

        # 极端情况
        score = calculator.calculate_quality_score(
            slippage_bps=1000.0,
            impact_bps=1000.0,
            fill_rate=0.0,
            execution_duration=10000.0,
            target_duration=300.0,
        )

        # 得分应在0-100之间
        assert 0 <= score <= 100


# ==================== ExecutionQualityMonitor 测试 ====================

class TestExecutionQualityMonitor:
    """执行质量监控器测试"""

    def test_monitor_creation(self, basic_config):
        """测试创建监控器"""
        monitor = ExecutionQualityMonitor(basic_config)
        assert monitor is not None
        assert monitor.state == MonitorState.IDLE

    def test_add_execution(self, quality_monitor, basic_execution_record):
        """测试添加执行记录"""
        execution_id = quality_monitor.add_execution(basic_execution_record)

        assert execution_id == basic_execution_record.execution_id
        assert quality_monitor.stats['total_executions'] == 1

        # 验证可以获取执行记录
        retrieved = quality_monitor.get_execution(execution_id)
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"

    def test_update_execution(self, quality_monitor, basic_execution_record):
        """测试更新执行记录"""
        execution_id = quality_monitor.add_execution(basic_execution_record)

        # 更新执行数量和价格
        quality_monitor.update_execution(
            execution_id,
            executed_quantity=1200.0,
            executed_price=150.75,
        )

        record = quality_monitor.get_execution(execution_id)
        assert record.executed_quantity == 1200.0
        assert record.executed_price == 150.75

    def test_get_executions_by_symbol(self, quality_monitor):
        """测试按品种获取执行记录"""
        # 添加多个执行记录
        for i in range(3):
            record = ExecutionRecord(
                execution_id=f"exec-{i}",
                symbol="AAPL",
                side=ExecutionSide.BUY,
                order_id=f"order-{i}",
                order_quantity=1000.0,
                order_time=datetime.now(),
                arrival_price=150.00,
            )
            quality_monitor.add_execution(record)

        # 添加不同品种
        other_record = ExecutionRecord(
            execution_id="exec-other",
            symbol="GOOGL",
            side=ExecutionSide.BUY,
            order_id="order-other",
            order_quantity=500.0,
            order_time=datetime.now(),
            arrival_price=2800.00,
        )
        quality_monitor.add_execution(other_record)

        # 获取AAPL的执行记录
        aapl_executions = quality_monitor.get_executions_by_symbol("AAPL")
        assert len(aapl_executions) == 3
        assert all(e.symbol == "AAPL" for e in aapl_executions)

    def test_calculate_quality_metrics(self, quality_monitor, basic_execution_record):
        """测试计算质量指标"""
        execution_id = quality_monitor.add_execution(basic_execution_record)

        metrics = quality_monitor.calculate_quality_metrics(execution_id)

        assert metrics is not None
        assert metrics.execution_id == execution_id
        assert metrics.symbol == "AAPL"
        assert metrics.fill_rate > 0
        assert 0 <= metrics.quality_score <= 100

    def test_calculate_quality_metrics_with_benchmark(
        self, quality_monitor, basic_execution_record
    ):
        """测试使用指定基准计算质量指标"""
        execution_id = quality_monitor.add_execution(basic_execution_record)

        # 使用VWAP基准
        metrics = quality_monitor.calculate_quality_metrics(
            execution_id,
            benchmark_type=BenchmarkType.VWAP,
        )

        assert metrics.benchmark_type == BenchmarkType.VWAP
        assert metrics.vwap_slippage != 0

    def test_calculate_cost_breakdown(self, quality_monitor, basic_execution_record):
        """测试计算成本分解"""
        execution_id = quality_monitor.add_execution(basic_execution_record)

        breakdown = quality_monitor.calculate_cost_breakdown(execution_id)

        assert breakdown is not None
        assert breakdown.execution_id == execution_id
        assert breakdown.total_cost != 0
        assert breakdown.slippage_bps != 0

    def test_check_alerts_high_slippage(self, quality_monitor):
        """测试高滑点告警"""
        # 创建高滑点执行记录
        record = ExecutionRecord(
            execution_id="exec-high-slip",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=152.00,  # 高于到达价格2美元
            execution_time=datetime.now(),
            status=ExecutionStatus.FILLED,
            arrival_price=150.00,
            vwap_price=150.30,
        )

        execution_id = quality_monitor.add_execution(record)
        alerts = quality_monitor.check_alerts(execution_id)

        # 应该触发高滑点告警
        assert len(alerts) > 0
        assert any(a.alert_type == AlertType.HIGH_SLIPPAGE for a in alerts)

    def test_check_alerts_slow_execution(self, quality_monitor):
        """测试慢执行告警"""
        # 创建慢执行记录
        order_time = datetime.now() - timedelta(seconds=600)  # 10分钟前
        record = ExecutionRecord(
            execution_id="exec-slow",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=order_time,
            executed_quantity=1000.0,
            executed_price=150.50,
            execution_time=datetime.now(),
            status=ExecutionStatus.FILLED,
            arrival_price=150.00,
            vwap_price=150.30,
        )

        execution_id = quality_monitor.add_execution(record)
        alerts = quality_monitor.check_alerts(execution_id)

        # 应该触发慢执行告警
        assert any(a.alert_type == AlertType.SLOW_EXECUTION for a in alerts)

    def test_check_alerts_disabled(self):
        """测试禁用告警"""
        config = MonitorConfig(enable_alerts=False)
        monitor = ExecutionQualityMonitor(config)

        # 创建高滑点执行记录
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=155.00,  # 很高的滑点
            execution_time=datetime.now(),
            status=ExecutionStatus.FILLED,
            arrival_price=150.00,
            vwap_price=150.30,
        )

        execution_id = monitor.add_execution(record)
        alerts = monitor.check_alerts(execution_id)

        # 告警被禁用，不应触发
        assert len(alerts) == 0

    def test_calculate_statistics(self, quality_monitor):
        """测试计算统计"""
        # 添加多个执行记录
        for i in range(5):
            record = ExecutionRecord(
                execution_id=f"exec-{i}",
                symbol="AAPL",
                side=ExecutionSide.BUY,
                order_id=f"order-{i}",
                order_quantity=1000.0,
                order_time=datetime.now(),
                executed_quantity=1000.0,
                executed_price=150.00 + i * 0.1,
                execution_time=datetime.now(),
                status=ExecutionStatus.FILLED,
                arrival_price=150.00,
                vwap_price=150.30,
                broker="Broker-A",
                algo_name="TWAP",
            )
            quality_monitor.add_execution(record)

        stats = quality_monitor.calculate_statistics()

        assert stats.total_executions == 5
        assert stats.total_volume > 0
        assert stats.avg_slippage_bps >= 0

    def test_calculate_statistics_with_time_range(self, quality_monitor):
        """测试指定时间范围的统计"""
        now = datetime.now()

        # 添加不同时间的记录
        old_record = ExecutionRecord(
            execution_id="exec-old",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-old",
            order_quantity=1000.0,
            order_time=now - timedelta(days=2),
            executed_quantity=1000.0,
            executed_price=150.50,
            execution_time=now - timedelta(days=2),
            status=ExecutionStatus.FILLED,
            arrival_price=150.00,
        )

        recent_record = ExecutionRecord(
            execution_id="exec-recent",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-recent",
            order_quantity=1000.0,
            order_time=now,
            executed_quantity=1000.0,
            executed_price=150.50,
            execution_time=now,
            status=ExecutionStatus.FILLED,
            arrival_price=150.00,
        )

        quality_monitor.add_execution(old_record)
        quality_monitor.add_execution(recent_record)

        # 只统计最近1天
        stats = quality_monitor.calculate_statistics(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(hours=1),
        )

        assert stats.total_executions == 1

    def test_calculate_statistics_by_symbol(self, quality_monitor):
        """测试按品种统计"""
        # 添加不同品种的记录
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            record = ExecutionRecord(
                execution_id=f"exec-{symbol}",
                symbol=symbol,
                side=ExecutionSide.BUY,
                order_id=f"order-{symbol}",
                order_quantity=1000.0,
                order_time=datetime.now(),
                executed_quantity=1000.0,
                executed_price=150.50,
                execution_time=datetime.now(),
                status=ExecutionStatus.FILLED,
                arrival_price=150.00,
            )
            quality_monitor.add_execution(record)

        # 只统计AAPL
        stats = quality_monitor.calculate_statistics(symbols=["AAPL"])

        assert stats.total_executions == 1

    def test_broker_stats(self, quality_monitor):
        """测试经纪商统计"""
        # 添加不同经纪商的记录
        for i, broker in enumerate(["Broker-A", "Broker-B", "Broker-A"]):
            record = ExecutionRecord(
                execution_id=f"exec-{i}",
                symbol="AAPL",
                side=ExecutionSide.BUY,
                order_id=f"order-{i}",
                order_quantity=1000.0,
                order_time=datetime.now(),
                executed_quantity=1000.0,
                executed_price=150.50 + i * 0.1,
                execution_time=datetime.now(),
                status=ExecutionStatus.FILLED,
                arrival_price=150.00,
                broker=broker,
            )
            quality_monitor.add_execution(record)

        stats = quality_monitor.calculate_statistics()

        assert "Broker-A" in stats.broker_stats
        assert "Broker-B" in stats.broker_stats
        assert stats.broker_stats["Broker-A"]["count"] == 2
        assert stats.broker_stats["Broker-B"]["count"] == 1

    def test_algo_stats(self, quality_monitor):
        """测试算法统计"""
        # 添加不同算法的记录
        for i, algo in enumerate(["TWAP", "VWAP", "TWAP"]):
            record = ExecutionRecord(
                execution_id=f"exec-{i}",
                symbol="AAPL",
                side=ExecutionSide.BUY,
                order_id=f"order-{i}",
                order_quantity=1000.0,
                order_time=datetime.now(),
                executed_quantity=1000.0,
                executed_price=150.50 + i * 0.1,
                execution_time=datetime.now(),
                status=ExecutionStatus.FILLED,
                arrival_price=150.00,
                algo_name=algo,
            )
            quality_monitor.add_execution(record)

        stats = quality_monitor.calculate_statistics()

        assert "TWAP" in stats.algo_stats
        assert "VWAP" in stats.algo_stats
        assert stats.algo_stats["TWAP"]["count"] == 2

    def test_hourly_stats(self, quality_monitor):
        """测试时段统计"""
        now = datetime.now()

        # 添加不同时段的记录
        for i in range(3):
            time = now.replace(hour=(now.hour + i) % 24)
            record = ExecutionRecord(
                execution_id=f"exec-{i}",
                symbol="AAPL",
                side=ExecutionSide.BUY,
                order_id=f"order-{i}",
                order_quantity=1000.0,
                order_time=time,
                executed_quantity=1000.0,
                executed_price=150.50,
                execution_time=time,
                status=ExecutionStatus.FILLED,
                arrival_price=150.00,
            )
            quality_monitor.add_execution(record)

        stats = quality_monitor.calculate_statistics()

        # 应该有统计数据
        assert len(stats.hourly_stats) > 0

    def test_get_alerts(self, quality_monitor):
        """测试获取告警列表"""
        # 创建触发告警的执行记录
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=155.00,  # 高滑点
            execution_time=datetime.now(),
            status=ExecutionStatus.FILLED,
            arrival_price=150.00,
            vwap_price=150.30,
        )

        execution_id = quality_monitor.add_execution(record)
        quality_monitor.check_alerts(execution_id)

        alerts = quality_monitor.get_alerts()
        assert len(alerts) > 0

    def test_get_statistics_summary(self, quality_monitor, basic_execution_record):
        """测试获取统计摘要"""
        quality_monitor.add_execution(basic_execution_record)

        summary = quality_monitor.get_statistics_summary()

        assert 'state' in summary
        assert 'total_executions' in summary
        assert summary['total_executions'] == 1

    def test_alert_callback(self, quality_monitor):
        """测试告警回调"""
        callback_triggered = []

        def alert_callback(alert):
            callback_triggered.append(alert)

        quality_monitor.on_alert(alert_callback)

        # 创建触发告警的执行记录
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=155.00,
            execution_time=datetime.now(),
            status=ExecutionStatus.FILLED,
            arrival_price=150.00,
            vwap_price=150.30,
        )

        execution_id = quality_monitor.add_execution(record)
        quality_monitor.check_alerts(execution_id)

        assert len(callback_triggered) > 0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, quality_monitor):
        """测试启动和停止监控"""
        await quality_monitor.start_monitoring()
        assert quality_monitor.state == MonitorState.ACTIVE

        await quality_monitor.stop_monitoring()
        assert quality_monitor.state == MonitorState.IDLE

    @pytest.mark.asyncio
    async def test_monitoring_loop(self, quality_monitor):
        """测试监控循环"""
        execution_complete_triggered = []

        def on_complete(record, metrics):
            execution_complete_triggered.append((record, metrics))

        quality_monitor.on_execution_complete(on_complete)

        # 添加已完成的执行记录
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=150.50,
            execution_time=datetime.now(),
            status=ExecutionStatus.FILLED,
            arrival_price=150.00,
            vwap_price=150.30,
        )

        quality_monitor.add_execution(record)

        # 启动监控
        await quality_monitor.start_monitoring()

        # 等待处理
        await asyncio.sleep(2)

        # 停止监控
        await quality_monitor.stop_monitoring()

        # 回调应该被触发
        assert len(execution_complete_triggered) > 0


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_execution_monitor(self):
        """测试创建监控器"""
        monitor = create_execution_monitor(
            high_slippage_threshold=15.0,
            high_impact_threshold=20.0,
            enable_alerts=True,
        )

        assert monitor is not None
        assert monitor.config.high_slippage_threshold == 15.0
        assert monitor.config.high_impact_threshold == 20.0

    def test_create_execution_record(self):
        """测试创建执行记录"""
        record = create_execution_record(
            symbol="AAPL",
            side="buy",
            order_quantity=1000.0,
            arrival_price=150.00,
        )

        assert record is not None
        assert record.symbol == "AAPL"
        assert record.side == ExecutionSide.BUY
        assert record.order_quantity == 1000.0
        assert record.arrival_price == 150.00

    def test_create_execution_record_with_optional_fields(self):
        """测试创建带可选字段的执行记录"""
        record = create_execution_record(
            symbol="AAPL",
            side="sell",
            order_quantity=1000.0,
            arrival_price=150.00,
            executed_quantity=800.0,
            executed_price=149.50,
            broker="Broker-A",
            algo_name="VWAP",
        )

        assert record.side == ExecutionSide.SELL
        assert record.executed_quantity == 800.0
        assert record.broker == "Broker-A"

    def test_quick_quality_analysis(self):
        """测试快速质量分析"""
        result = quick_quality_analysis(
            symbol="AAPL",
            side="buy",
            order_quantity=1000.0,
            executed_quantity=1000.0,
            executed_price=150.50,
            arrival_price=150.00,
            vwap_price=150.30,
        )

        assert result is not None
        assert result['symbol'] == "AAPL"
        assert 'quality_metrics' in result
        assert 'cost_breakdown' in result
        assert 'alerts' in result
        assert 'statistics' in result

    def test_quick_quality_analysis_with_high_slippage(self):
        """测试高滑点的快速质量分析"""
        result = quick_quality_analysis(
            symbol="AAPL",
            side="buy",
            order_quantity=1000.0,
            executed_quantity=1000.0,
            executed_price=155.00,  # 高滑点
            arrival_price=150.00,
            vwap_price=150.30,
        )

        # 应该触发告警
        assert len(result['alerts']) > 0
        assert any(
            a['alert_type'] == AlertType.HIGH_SLIPPAGE.value
            for a in result['alerts']
        )


# ==================== 边界情况测试 ====================

class TestEdgeCases:
    """边界情况测试"""

    def test_zero_executed_quantity(self, quality_monitor):
        """测试零执行数量"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=0.0,  # 未执行
            executed_price=0.0,
            arrival_price=150.00,
        )

        execution_id = quality_monitor.add_execution(record)
        metrics = quality_monitor.calculate_quality_metrics(execution_id)

        # 应该能正常处理
        assert metrics is not None
        assert metrics.fill_rate == 0.0

    def test_zero_arrival_price(self, quality_monitor):
        """测试零到达价格"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=1000.0,
            executed_price=150.50,
            arrival_price=0.0,  # 无效的到达价格
        )

        execution_id = quality_monitor.add_execution(record)
        breakdown = quality_monitor.calculate_cost_breakdown(execution_id)

        # 应该返回零成本
        assert breakdown.total_cost == 0.0

    def test_partial_fill(self, quality_monitor):
        """测试部分成交"""
        record = ExecutionRecord(
            execution_id="exec-001",
            symbol="AAPL",
            side=ExecutionSide.BUY,
            order_id="order-001",
            order_quantity=1000.0,
            order_time=datetime.now(),
            executed_quantity=600.0,  # 60%成交
            executed_price=150.50,
            execution_time=datetime.now(),
            status=ExecutionStatus.PARTIAL,
            arrival_price=150.00,
            closing_price=151.00,
        )

        execution_id = quality_monitor.add_execution(record)
        metrics = quality_monitor.calculate_quality_metrics(execution_id)

        assert metrics.fill_rate == 0.6

    def test_empty_statistics(self):
        """测试空统计"""
        monitor = create_execution_monitor()

        stats = monitor.calculate_statistics()

        assert stats.total_executions == 0
        assert stats.total_volume == 0.0


# ==================== 性能测试 ====================

class TestPerformance:
    """性能测试"""

    def test_large_number_of_executions(self, quality_monitor):
        """测试大量执行记录"""
        num_executions = 1000

        for i in range(num_executions):
            record = ExecutionRecord(
                execution_id=f"exec-{i}",
                symbol=f"SYM{i % 10}",
                side=ExecutionSide.BUY if i % 2 == 0 else ExecutionSide.SELL,
                order_id=f"order-{i}",
                order_quantity=1000.0,
                order_time=datetime.now(),
                executed_quantity=1000.0,
                executed_price=150.00 + (i % 100) * 0.01,
                execution_time=datetime.now(),
                status=ExecutionStatus.FILLED,
                arrival_price=150.00,
                vwap_price=150.30,
            )
            quality_monitor.add_execution(record)

        assert quality_monitor.stats['total_executions'] == num_executions

        # 测试统计计算性能
        stats = quality_monitor.calculate_statistics()
        assert stats.total_executions == num_executions

    def test_concurrent_metric_calculation(self, quality_monitor):
        """测试并发质量指标计算"""
        # 添加多个执行记录
        execution_ids = []
        for i in range(100):
            record = ExecutionRecord(
                execution_id=f"exec-{i}",
                symbol="AAPL",
                side=ExecutionSide.BUY,
                order_id=f"order-{i}",
                order_quantity=1000.0,
                order_time=datetime.now(),
                executed_quantity=1000.0,
                executed_price=150.50,
                execution_time=datetime.now(),
                status=ExecutionStatus.FILLED,
                arrival_price=150.00,
                vwap_price=150.30,
            )
            execution_ids.append(quality_monitor.add_execution(record))

        # 并发计算质量指标
        for execution_id in execution_ids:
            metrics = quality_monitor.calculate_quality_metrics(execution_id)
            assert metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
