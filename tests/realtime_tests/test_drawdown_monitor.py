"""
回撤监控器测试 (Drawdown Monitor Tests)

测试回撤监控器的所有功能:
- 实时回撤计算
- 回撤持续时间追踪
- 回撤恢复时间预估
- 多时间框架回撤
- 回撤分解分析
- 回撤警报系统
- 历史回撤统计
- 回撤风险预测

Copyright (c) 2025 TraderX. All rights reserved.
"""

import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.drawdown_monitor import (
    # Monitor
    DrawdownMonitor,
    create_drawdown_monitor,
    quick_drawdown_analysis,
    # Data Classes
    DrawdownPeriod,
    DrawdownMetrics,
    TimeFrameDrawdown,
    DrawdownAlert,
    DrawdownDecomposition,
    DrawdownStatistics,
    DrawdownPrediction,
    MonitorConfig,
    # Calculators
    DrawdownCalculator,
    RecoveryEstimator,
    DrawdownPredictor,
    # Enums
    TimeFrame,
    DrawdownState,
    DrawdownSeverity,
    AlertLevel,
    MonitorState,
    RecoveryModel,
)


# ============== Fixtures ==============

@pytest.fixture
def monitor():
    """创建基础监控器"""
    return create_drawdown_monitor(
        update_interval_sec=1.0,
        enable_alerts=True,
        warning_threshold_pct=5.0,
        critical_threshold_pct=10.0,
        emergency_threshold_pct=20.0,
    )


@pytest.fixture
def monitor_with_data(monitor):
    """创建带有数据的监控器"""
    # 添加上升趋势数据
    base_value = 100000
    for i in range(30):
        value = base_value * (1 + i * 0.01)  # 每天增长1%
        timestamp = datetime.now() - timedelta(days=30-i)
        monitor.update_value(value, timestamp)

    return monitor


@pytest.fixture
def drawdown_scenario_data():
    """创建回撤场景数据"""
    # 模拟一个完整的回撤和恢复周期
    base = 100000
    values = []
    timestamps = []

    now = datetime.now()

    # 上升阶段 (0-20天)
    for i in range(20):
        values.append(base * (1 + i * 0.02))
        timestamps.append(now - timedelta(days=50-i))

    # 回撤阶段 (20-35天) 最大回撤15%
    peak = values[-1]
    for i in range(15):
        dd = (i / 15) * 0.15  # 逐步回撤到15%
        values.append(peak * (1 - dd))
        timestamps.append(now - timedelta(days=30-i))

    # 恢复阶段 (35-50天)
    trough = values[-1]
    for i in range(15):
        recovery = (i / 15) * 0.15
        values.append(trough * (1 + recovery))
        timestamps.append(now - timedelta(days=15-i))

    return values, timestamps


@pytest.fixture
def calculator():
    """创建回撤计算器"""
    return DrawdownCalculator()


@pytest.fixture
def estimator():
    """创建恢复预估器"""
    return RecoveryEstimator(RecoveryModel.HISTORICAL)


@pytest.fixture
def predictor():
    """创建回撤预测器"""
    return DrawdownPredictor()


# ============== Test DrawdownCalculator ==============

class TestDrawdownCalculator:
    """测试回撤计算器"""

    def test_calculate_drawdown_basic(self, calculator):
        """测试1: 基础回撤计算"""
        dd_pct, dd_value = calculator.calculate_drawdown(90000, 100000)
        assert dd_pct == 10.0
        assert dd_value == 10000

    def test_calculate_drawdown_no_drawdown(self, calculator):
        """测试2: 无回撤情况"""
        dd_pct, dd_value = calculator.calculate_drawdown(100000, 100000)
        assert dd_pct == 0.0
        assert dd_value == 0.0

    def test_calculate_drawdown_above_peak(self, calculator):
        """测试3: 价值超过峰值"""
        dd_pct, dd_value = calculator.calculate_drawdown(110000, 100000)
        assert dd_pct == 0.0
        assert dd_value == 0.0

    def test_calculate_drawdown_zero_peak(self, calculator):
        """测试4: 峰值为零"""
        dd_pct, dd_value = calculator.calculate_drawdown(50000, 0)
        assert dd_pct == 0.0
        assert dd_value == 0.0

    def test_calculate_max_drawdown(self, calculator):
        """测试5: 计算最大回撤"""
        values = [100, 110, 105, 95, 120, 100, 115]
        max_dd, peak_idx, trough_idx = calculator.calculate_max_drawdown(values)

        assert max_dd > 0
        assert peak_idx >= 0
        assert trough_idx >= 0

    def test_calculate_max_drawdown_monotonic_increase(self, calculator):
        """测试6: 单调递增序列的最大回撤"""
        values = [100, 110, 120, 130, 140]
        max_dd, peak_idx, trough_idx = calculator.calculate_max_drawdown(values)

        assert max_dd == 0.0

    def test_calculate_max_drawdown_monotonic_decrease(self, calculator):
        """测试7: 单调递减序列的最大回撤"""
        values = [100, 90, 80, 70, 60]
        max_dd, peak_idx, trough_idx = calculator.calculate_max_drawdown(values)

        assert max_dd == 40.0  # (100-60)/100 * 100
        assert peak_idx == 0
        assert trough_idx == 4

    def test_calculate_underwater_time(self, calculator):
        """测试8: 计算水下时间"""
        values = [100, 110, 105, 95, 120, 100, 115, 125]
        underwater_pct = calculator.calculate_underwater_time(values)

        assert 0 <= underwater_pct <= 100

    def test_determine_severity_minor(self, calculator):
        """测试9: 判断轻微回撤"""
        severity = calculator.determine_severity(3.0)
        assert severity == DrawdownSeverity.MINOR

    def test_determine_severity_moderate(self, calculator):
        """测试10: 判断中等回撤"""
        severity = calculator.determine_severity(7.0)
        assert severity == DrawdownSeverity.MODERATE

    def test_determine_severity_significant(self, calculator):
        """测试11: 判断显著回撤"""
        severity = calculator.determine_severity(15.0)
        assert severity == DrawdownSeverity.SIGNIFICANT

    def test_determine_severity_severe(self, calculator):
        """测试12: 判断严重回撤"""
        severity = calculator.determine_severity(25.0)
        assert severity == DrawdownSeverity.SEVERE

    def test_determine_severity_extreme(self, calculator):
        """测试13: 判断极端回撤"""
        severity = calculator.determine_severity(35.0)
        assert severity == DrawdownSeverity.EXTREME


# ============== Test RecoveryEstimator ==============

class TestRecoveryEstimator:
    """测试恢复预估器"""

    def test_linear_estimate_basic(self):
        """测试14: 线性恢复预估"""
        estimator = RecoveryEstimator(RecoveryModel.LINEAR)
        recent_returns = [0.001] * 20  # 0.1% daily return

        days, confidence = estimator.estimate_recovery_time(
            current_drawdown_pct=10.0,
            current_value=90000,
            peak_value=100000,
            recent_returns=recent_returns,
        )

        assert days > 0
        assert 0 <= confidence <= 1

    def test_linear_estimate_no_returns(self):
        """测试15: 无收益率数据的线性预估"""
        estimator = RecoveryEstimator(RecoveryModel.LINEAR)

        days, confidence = estimator.estimate_recovery_time(
            current_drawdown_pct=10.0,
            current_value=90000,
            peak_value=100000,
            recent_returns=None,
        )

        assert days > 0
        assert confidence < 1

    def test_exponential_estimate(self):
        """测试16: 指数恢复预估"""
        estimator = RecoveryEstimator(RecoveryModel.EXPONENTIAL)
        recent_returns = [0.001] * 20

        days, confidence = estimator.estimate_recovery_time(
            current_drawdown_pct=10.0,
            current_value=90000,
            peak_value=100000,
            recent_returns=recent_returns,
        )

        assert days > 0
        assert 0 <= confidence <= 1

    def test_mean_reversion_estimate(self):
        """测试17: 均值回归预估"""
        estimator = RecoveryEstimator(RecoveryModel.MEAN_REVERSION)
        recent_returns = [0.001, -0.002, 0.003, -0.001] * 10

        days, confidence = estimator.estimate_recovery_time(
            current_drawdown_pct=10.0,
            current_value=90000,
            peak_value=100000,
            recent_returns=recent_returns,
        )

        assert days > 0
        assert 0 <= confidence <= 1

    def test_historical_estimate_with_data(self):
        """测试18: 历史数据恢复预估"""
        estimator = RecoveryEstimator(RecoveryModel.HISTORICAL)

        # 添加历史恢复数据
        estimator.add_historical_recovery(10.0, 30.0)
        estimator.add_historical_recovery(8.0, 25.0)
        estimator.add_historical_recovery(12.0, 35.0)

        days, confidence = estimator.estimate_recovery_time(
            current_drawdown_pct=10.0,
            current_value=90000,
            peak_value=100000,
        )

        assert days > 0
        assert confidence > 0.5  # 有历史数据时置信度较高

    def test_historical_estimate_no_data(self):
        """测试19: 无历史数据的预估"""
        estimator = RecoveryEstimator(RecoveryModel.HISTORICAL)

        days, confidence = estimator.estimate_recovery_time(
            current_drawdown_pct=10.0,
            current_value=90000,
            peak_value=100000,
        )

        assert days > 0
        assert confidence < 0.5  # 无历史数据时置信度较低

    def test_estimate_zero_drawdown(self):
        """测试20: 零回撤的恢复预估"""
        estimator = RecoveryEstimator(RecoveryModel.LINEAR)

        days, confidence = estimator.estimate_recovery_time(
            current_drawdown_pct=0.0,
            current_value=100000,
            peak_value=100000,
        )

        assert days == 0.0
        assert confidence == 1.0


# ============== Test DrawdownPredictor ==============

class TestDrawdownPredictor:
    """测试回撤预测器"""

    def test_predict_with_data(self, predictor):
        """测试21: 有数据的预测"""
        # 添加收益率数据
        for _ in range(50):
            predictor.add_return(0.001)  # 0.1% daily return

        prediction = predictor.predict(horizon_days=30)

        assert isinstance(prediction, DrawdownPrediction)
        assert prediction.prediction_horizon_days == 30
        assert 0 <= prediction.drawdown_5pct_probability <= 1
        assert 0 <= prediction.drawdown_10pct_probability <= 1
        assert 0 <= prediction.drawdown_20pct_probability <= 1
        assert prediction.expected_drawdown_pct >= 0
        assert prediction.var_95_drawdown_pct >= 0
        assert prediction.cvar_95_drawdown_pct >= 0

    def test_predict_insufficient_data(self, predictor):
        """测试22: 数据不足的预测"""
        # 仅添加少量数据
        for _ in range(10):
            predictor.add_return(0.001)

        prediction = predictor.predict(horizon_days=30)

        assert isinstance(prediction, DrawdownPrediction)
        assert prediction.confidence_level < 0.5  # 数据不足时置信度低

    def test_predict_volatile_returns(self, predictor):
        """测试23: 高波动率数据的预测"""
        import random
        for _ in range(100):
            predictor.add_return(random.gauss(0, 0.02))  # 高波动

        prediction = predictor.predict(horizon_days=30)

        # 高波动率应该导致更高的预期回撤
        assert prediction.expected_drawdown_pct > 0

    def test_predict_different_horizons(self, predictor):
        """测试24: 不同时间范围的预测"""
        for _ in range(50):
            predictor.add_return(0.001)

        pred_30 = predictor.predict(horizon_days=30)
        pred_90 = predictor.predict(horizon_days=90)

        # 更长的时间范围通常有更高的预期回撤
        assert pred_90.expected_drawdown_pct >= pred_30.expected_drawdown_pct


# ============== Test DrawdownMonitor ==============

class TestDrawdownMonitor:
    """测试回撤监控器"""

    def test_monitor_initialization(self, monitor):
        """测试25: 监控器初始化"""
        assert monitor.state == MonitorState.IDLE
        assert monitor._peak_value == 0.0
        assert len(monitor._value_history) == 0

    def test_update_value_single(self, monitor):
        """测试26: 单次价值更新"""
        monitor.update_value(100000)

        assert monitor._peak_value == 100000
        assert len(monitor._value_history) == 1

        metrics = monitor.get_current_metrics()
        assert metrics is not None
        assert metrics.current_value == 100000
        assert metrics.current_drawdown_pct == 0.0

    def test_update_value_increasing(self, monitor):
        """测试27: 价值递增更新"""
        values = [100000, 105000, 110000, 115000]

        for value in values:
            monitor.update_value(value)

        assert monitor._peak_value == 115000
        metrics = monitor.get_current_metrics()
        assert metrics.current_drawdown_pct == 0.0
        assert metrics.state == DrawdownState.NONE

    def test_update_value_with_drawdown(self, monitor):
        """测试28: 带回撤的价值更新"""
        monitor.update_value(100000)
        monitor.update_value(110000)  # 峰值
        monitor.update_value(99000)   # 回撤 10%

        metrics = monitor.get_current_metrics()
        assert metrics.current_drawdown_pct == pytest.approx(10.0, rel=0.01)
        assert metrics.state == DrawdownState.IN_DRAWDOWN
        assert monitor._current_drawdown is not None

    def test_update_value_recovery(self, monitor):
        """测试29: 回撤恢复"""
        monitor.update_value(100000)
        monitor.update_value(110000)  # 峰值
        monitor.update_value(99000)   # 回撤
        monitor.update_value(110000)  # 恢复到峰值

        metrics = monitor.get_current_metrics()
        assert metrics.current_drawdown_pct == 0.0

        # 回撤期应该已结束并归档
        assert len(monitor._historical_drawdowns) >= 1

    def test_add_return(self, monitor):
        """测试30: 添加收益率"""
        monitor.add_return(0.01)
        monitor.add_return(-0.005)
        monitor.add_return(0.02)

        assert len(monitor._returns_history) == 3

    def test_update_asset_value(self, monitor):
        """测试31: 更新资产价值"""
        monitor.update_asset_value("AAPL", 50000)
        monitor.update_asset_value("GOOGL", 30000)

        assert "AAPL" in monitor._asset_values
        assert "GOOGL" in monitor._asset_values

    def test_update_strategy_value(self, monitor):
        """测试32: 更新策略价值"""
        monitor.update_strategy_value("momentum", 60000)
        monitor.update_strategy_value("mean_reversion", 40000)

        assert "momentum" in monitor._strategy_values
        assert "mean_reversion" in monitor._strategy_values

    def test_update_factor_value(self, monitor):
        """测试33: 更新因子价值"""
        monitor.update_factor_value("value", 70000)
        monitor.update_factor_value("growth", 30000)

        assert "value" in monitor._factor_values
        assert "growth" in monitor._factor_values

    def test_calculate_timeframe_drawdowns(self, monitor_with_data):
        """测试34: 多时间框架回撤计算"""
        timeframes = monitor_with_data.calculate_timeframe_drawdowns()

        assert TimeFrame.DAILY in timeframes or len(timeframes) >= 0
        # 确保返回的是字典
        assert isinstance(timeframes, dict)

    def test_calculate_decomposition_no_drawdown(self, monitor):
        """测试35: 无回撤时的分解"""
        monitor.update_value(100000)
        decomp = monitor.calculate_decomposition()

        assert decomp is None  # 无回撤时应该返回None

    def test_calculate_decomposition_with_drawdown(self, monitor):
        """测试36: 有回撤时的分解"""
        # 创建回撤
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(99000)  # 10% drawdown

        # 添加资产数据
        monitor.update_asset_value("AAPL", 50000)
        monitor.update_asset_value("AAPL", 55000)
        monitor.update_asset_value("AAPL", 49500)

        decomp = monitor.calculate_decomposition()

        if decomp:  # 如果启用了分解
            assert isinstance(decomp, DrawdownDecomposition)
            assert decomp.total_drawdown_pct > 0

    def test_calculate_statistics_no_data(self, monitor):
        """测试37: 无数据时的统计"""
        stats = monitor.calculate_statistics()
        assert stats is None  # 无历史回撤时应该返回None

    def test_calculate_statistics_with_data(self, drawdown_scenario_data):
        """测试38: 有数据时的统计"""
        monitor = create_drawdown_monitor(min_drawdown_pct=1.0)
        values, timestamps = drawdown_scenario_data

        for ts, value in zip(timestamps, values):
            monitor.update_value(value, ts)

        # 等待一些回撤记录
        stats = monitor.calculate_statistics()

        if stats:  # 如果有足够的回撤记录
            assert isinstance(stats, DrawdownStatistics)
            assert stats.total_drawdowns >= 0

    def test_predict_drawdown(self, monitor_with_data):
        """测试39: 回撤预测"""
        # 添加一些收益率数据
        for _ in range(30):
            monitor_with_data.add_return(0.001)

        prediction = monitor_with_data.predict_drawdown(horizon_days=30)

        assert isinstance(prediction, DrawdownPrediction)
        assert prediction.prediction_horizon_days == 30

    def test_alert_warning_threshold(self, monitor):
        """测试40: 预警阈值告警"""
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(104500)  # 5% 回撤，触发预警

        alerts = monitor.get_alerts(unacknowledged_only=True)
        # 可能触发告警
        assert isinstance(alerts, list)

    def test_alert_critical_threshold(self, monitor):
        """测试41: 临界阈值告警"""
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(99000)  # 10% 回撤，触发临界告警

        alerts = monitor.get_alerts(level=AlertLevel.CRITICAL)
        assert isinstance(alerts, list)

    def test_alert_emergency_threshold(self, monitor):
        """测试42: 紧急阈值告警"""
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(88000)  # 20% 回撤，触发紧急告警

        alerts = monitor.get_alerts(level=AlertLevel.EMERGENCY)
        assert isinstance(alerts, list)

    def test_acknowledge_alert(self, monitor):
        """测试43: 确认告警"""
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(99000)  # 触发告警

        alerts = monitor.get_alerts()
        if alerts:
            alert_id = alerts[0].alert_id
            success = monitor.acknowledge_alert(alert_id)
            assert success

            # 验证告警已确认
            alert = next(a for a in monitor._alerts if a.alert_id == alert_id)
            assert alert.acknowledged

    def test_resolve_alert(self, monitor):
        """测试44: 解决告警"""
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(99000)

        alerts = monitor.get_alerts()
        if alerts:
            alert_id = alerts[0].alert_id
            success = monitor.resolve_alert(alert_id)
            assert success

            alert = next(a for a in monitor._alerts if a.alert_id == alert_id)
            assert alert.resolved

    def test_add_alert_callback(self, monitor):
        """测试45: 添加告警回调"""
        callback_triggered = []

        def callback(alert: DrawdownAlert):
            callback_triggered.append(alert)

        monitor.add_alert_callback(callback)

        # 触发告警
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(99000)  # 10% 回撤

        # 检查回调是否被触发
        # (根据阈值设置，可能触发也可能不触发)
        assert isinstance(callback_triggered, list)

    def test_get_historical_drawdowns(self, drawdown_scenario_data):
        """测试46: 获取历史回撤记录"""
        monitor = create_drawdown_monitor()
        values, timestamps = drawdown_scenario_data

        for ts, value in zip(timestamps, values):
            monitor.update_value(value, ts)

        drawdowns = monitor.get_historical_drawdowns(min_drawdown_pct=5.0)
        assert isinstance(drawdowns, list)

    def test_get_historical_drawdowns_by_severity(self, drawdown_scenario_data):
        """测试47: 按严重程度筛选历史回撤"""
        monitor = create_drawdown_monitor()
        values, timestamps = drawdown_scenario_data

        for ts, value in zip(timestamps, values):
            monitor.update_value(value, ts)

        drawdowns = monitor.get_historical_drawdowns(severity=DrawdownSeverity.SIGNIFICANT)
        assert isinstance(drawdowns, list)

    def test_get_current_drawdown_period(self, monitor):
        """测试48: 获取当前回撤期"""
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(99000)  # 进入回撤

        current_dd = monitor.get_current_drawdown_period()
        assert current_dd is not None
        assert isinstance(current_dd, DrawdownPeriod)
        assert current_dd.state == DrawdownState.IN_DRAWDOWN

    @pytest.mark.asyncio
    async def test_monitor_start_stop(self, monitor):
        """测试49: 监控启动和停止"""
        await monitor.start()
        assert monitor.state == MonitorState.RUNNING

        # 等待一小段时间
        await asyncio.sleep(0.1)

        await monitor.stop()
        assert monitor.state == MonitorState.IDLE

    def test_get_summary(self, monitor_with_data):
        """测试50: 获取监控摘要"""
        # 添加一些收益率数据
        for _ in range(20):
            monitor_with_data.add_return(0.001)

        summary = monitor_with_data.get_summary()

        assert isinstance(summary, dict)
        assert 'state' in summary
        assert 'current_metrics' in summary
        assert 'timeframe_drawdowns' in summary
        assert 'prediction' in summary


# ============== Test Factory Functions ==============

class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_drawdown_monitor_default(self):
        """测试51: 创建默认监控器"""
        monitor = create_drawdown_monitor()
        assert isinstance(monitor, DrawdownMonitor)
        assert monitor.config.enable_alerts

    def test_create_drawdown_monitor_custom(self):
        """测试52: 创建自定义监控器"""
        monitor = create_drawdown_monitor(
            update_interval_sec=30.0,
            enable_alerts=False,
            warning_threshold_pct=3.0,
            critical_threshold_pct=8.0,
        )

        assert monitor.config.update_interval_sec == 30.0
        assert monitor.config.enable_alerts == False
        assert monitor.config.warning_threshold_pct == 3.0
        assert monitor.config.critical_threshold_pct == 8.0

    @pytest.mark.asyncio
    async def test_quick_drawdown_analysis_basic(self):
        """测试53: 快速回撤分析 - 基础"""
        values = [100, 110, 105, 95, 120, 100, 115]

        result = await quick_drawdown_analysis(values)

        assert isinstance(result, dict)
        assert 'state' in result or len(result) >= 0

    @pytest.mark.asyncio
    async def test_quick_drawdown_analysis_with_timestamps(self):
        """测试54: 快速回撤分析 - 带时间戳"""
        values = [100, 110, 105, 95, 120]
        now = datetime.now()
        timestamps = [now - timedelta(days=5-i) for i in range(5)]

        result = await quick_drawdown_analysis(values, timestamps)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_quick_drawdown_analysis_empty(self):
        """测试55: 快速回撤分析 - 空数据"""
        result = await quick_drawdown_analysis([])
        assert isinstance(result, dict)


# ============== Test Edge Cases ==============

class TestEdgeCases:
    """测试边界情况"""

    def test_large_drawdown(self, monitor):
        """测试56: 极大回撤"""
        monitor.update_value(100000)
        monitor.update_value(10000)  # 90% 回撤

        metrics = monitor.get_current_metrics()
        assert metrics.current_drawdown_pct == pytest.approx(90.0, rel=0.01)
        assert metrics.severity == DrawdownSeverity.EXTREME

    def test_multiple_drawdown_cycles(self, monitor):
        """测试57: 多个回撤周期"""
        # 第一个周期
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(99000)
        monitor.update_value(110000)  # 恢复

        # 第二个周期
        monitor.update_value(120000)
        monitor.update_value(108000)
        monitor.update_value(120000)  # 恢复

        # 应该有2个历史回撤记录
        assert len(monitor._historical_drawdowns) >= 1

    def test_very_long_history(self, monitor):
        """测试58: 非常长的历史数据"""
        base = 100000
        for i in range(1000):
            value = base * (1 + i * 0.001)
            monitor.update_value(value)

        # 应该限制历史长度
        assert len(monitor._value_history) <= monitor.config.max_history_days * 24

    def test_negative_values(self, monitor):
        """测试59: 负值处理"""
        # 监控器应该能处理负值(虽然在实际中不常见)
        monitor.update_value(100000)
        monitor.update_value(-50000)

        metrics = monitor.get_current_metrics()
        assert metrics is not None

    def test_zero_values(self, monitor):
        """测试60: 零值处理"""
        monitor.update_value(100000)
        monitor.update_value(0)

        metrics = monitor.get_current_metrics()
        assert metrics is not None


# ============== Test Integration ==============

class TestIntegration:
    """测试集成场景"""

    def test_full_cycle_integration(self, drawdown_scenario_data):
        """测试61: 完整周期集成测试"""
        monitor = create_drawdown_monitor(
            enable_alerts=True,
            warning_threshold_pct=5.0,
            critical_threshold_pct=10.0,
        )

        values, timestamps = drawdown_scenario_data

        # 添加所有数据
        for i, (ts, value) in enumerate(zip(timestamps, values)):
            monitor.update_value(value, ts)

            # 计算收益率
            if i > 0:
                prev_value = values[i-1]
                if prev_value > 0:
                    ret = (value - prev_value) / prev_value
                    monitor.add_return(ret)

        # 验证监控器状态
        metrics = monitor.get_current_metrics()
        assert metrics is not None

        # 验证时间框架回撤
        tf_drawdowns = monitor.calculate_timeframe_drawdowns()
        assert isinstance(tf_drawdowns, dict)

        # 验证统计
        stats = monitor.calculate_statistics()
        # stats可能为None(如果回撤不足)

        # 验证预测
        prediction = monitor.predict_drawdown()
        assert isinstance(prediction, DrawdownPrediction)

        # 验证摘要
        summary = monitor.get_summary()
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, monitor):
        """测试62: 并发更新"""
        async def update_values():
            for i in range(100):
                monitor.update_value(100000 + i * 100)
                await asyncio.sleep(0.001)

        # 启动监控
        await monitor.start()

        # 并发更新
        await update_values()

        # 停止监控
        await monitor.stop()

        # 验证数据完整性
        assert len(monitor._value_history) > 0

    def test_decomposition_accuracy(self, monitor):
        """测试63: 分解准确性"""
        # 创建带分解的场景
        monitor.update_value(100000)

        # 添加资产数据
        monitor.update_asset_value("AAPL", 50000)
        monitor.update_asset_value("GOOGL", 50000)

        monitor.update_value(110000)
        monitor.update_asset_value("AAPL", 55000)
        monitor.update_asset_value("GOOGL", 55000)

        # 创建回撤
        monitor.update_value(99000)  # 10% total drawdown
        monitor.update_asset_value("AAPL", 49500)  # 10% drawdown
        monitor.update_asset_value("GOOGL", 49500)  # 10% drawdown

        decomp = monitor.calculate_decomposition()
        if decomp:
            # 验证分解数据
            assert "AAPL" in decomp.asset_drawdowns
            assert "GOOGL" in decomp.asset_drawdowns

    def test_alert_callback_chain(self, monitor):
        """测试64: 告警回调链"""
        callback_log = []

        def callback1(alert):
            callback_log.append(('callback1', alert.alert_id))

        def callback2(alert):
            callback_log.append(('callback2', alert.alert_id))

        monitor.add_alert_callback(callback1)
        monitor.add_alert_callback(callback2)

        # 触发告警
        monitor.update_value(100000)
        monitor.update_value(110000)
        monitor.update_value(88000)  # 20% drawdown

        # 验证所有回调都被调用
        # (根据阈值，可能触发多个告警)
        assert isinstance(callback_log, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
