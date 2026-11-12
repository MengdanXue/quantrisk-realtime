"""
组合风险监控器测试
Portfolio Risk Monitor Tests
"""

import pytest
import asyncio
import math
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from quantrisk_realtime.realtime.portfolio_risk_monitor import (
    # Enums
    VaRMethod,
    StressScenario,
    RiskLimitType,
    AlertSeverity,
    MonitorState,
    # Data Classes
    PositionRisk,
    VaRResult,
    CVaRResult,
    StressTestResult,
    RiskLimit,
    RiskAlert,
    RiskMetrics,
    MonitorConfig,
    # Calculators
    VaRCalculator,
    HistoricalVaR,
    ParametricVaR,
    MonteCarloVaR,
    CornishFisherVaR,
    CVaRCalculator,
    # Stress Test
    StressTestEngine,
    # Monitor
    PortfolioRiskMonitor,
    # Factory Functions
    create_risk_monitor,
    create_position_risk,
    create_risk_limit,
    quick_risk_analysis,
)


# ============== PositionRisk Tests ==============

class TestPositionRisk:
    """持仓风险测试"""

    def test_position_risk_creation(self):
        """测试持仓风险创建"""
        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=15000,
            weight=0.3,
            volatility=0.25,
        )

        assert pos.symbol == "AAPL"
        assert pos.quantity == 100
        assert pos.market_value == 15000
        assert pos.weight == 0.3
        assert pos.volatility == 0.25

    def test_position_risk_to_dict(self):
        """测试转换为字典"""
        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=15000,
            weight=0.3,
            volatility=0.25,
            beta=1.2,
        )

        data = pos.to_dict()
        assert data['symbol'] == "AAPL"
        assert data['beta'] == 1.2


# ============== VaR Result Tests ==============

class TestVaRResult:
    """VaR结果测试"""

    def test_var_result_creation(self):
        """测试VaR结果创建"""
        result = VaRResult(
            var_value=5000,
            confidence_level=0.95,
            holding_period=1,
            method=VaRMethod.HISTORICAL,
            timestamp=datetime.now(),
            portfolio_value=100000,
        )

        assert result.var_value == 5000
        assert result.confidence_level == 0.95

    def test_var_result_to_dict(self):
        """测试转换为字典"""
        result = VaRResult(
            var_value=5000,
            confidence_level=0.95,
            holding_period=1,
            method=VaRMethod.HISTORICAL,
            timestamp=datetime.now(),
            portfolio_value=100000,
            var_pct=5.0,
        )

        data = result.to_dict()
        assert data['var_value'] == 5000
        assert data['method'] == 'historical'


# ============== RiskLimit Tests ==============

class TestRiskLimit:
    """风险限额测试"""

    def test_risk_limit_creation(self):
        """测试风险限额创建"""
        limit = RiskLimit(
            limit_type=RiskLimitType.VAR_LIMIT,
            limit_value=10000,
        )

        assert limit.limit_type == RiskLimitType.VAR_LIMIT
        assert limit.limit_value == 10000
        assert not limit.is_breached

    def test_risk_limit_update(self):
        """测试风险限额更新"""
        limit = RiskLimit(
            limit_type=RiskLimitType.VAR_LIMIT,
            limit_value=10000,
            warning_threshold=0.8,
        )

        limit.update(5000)
        assert limit.current_value == 5000
        assert limit.utilization == 0.5
        assert not limit.is_warning
        assert not limit.is_breached

    def test_risk_limit_warning(self):
        """测试预警触发"""
        limit = RiskLimit(
            limit_type=RiskLimitType.VAR_LIMIT,
            limit_value=10000,
            warning_threshold=0.8,
        )

        limit.update(8500)
        assert limit.is_warning
        assert not limit.is_breached

    def test_risk_limit_breach(self):
        """测试限额突破"""
        limit = RiskLimit(
            limit_type=RiskLimitType.VAR_LIMIT,
            limit_value=10000,
        )

        limit.update(12000)
        assert limit.is_breached
        assert limit.utilization == 1.2


# ============== Historical VaR Tests ==============

class TestHistoricalVaR:
    """历史模拟法VaR测试"""

    def test_historical_var_basic(self):
        """测试基本计算"""
        calculator = HistoricalVaR(confidence_level=0.95, holding_period=1)

        # 生成100个日收益率
        returns = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05] * 10

        result = calculator.calculate(returns, 100000)

        assert result.var_value > 0
        assert result.method == VaRMethod.HISTORICAL
        assert result.confidence_level == 0.95

    def test_historical_var_empty_returns(self):
        """测试空收益率"""
        calculator = HistoricalVaR()

        result = calculator.calculate([], 100000)

        assert result.var_value == 0

    def test_historical_var_holding_period(self):
        """测试多日持有期"""
        calculator = HistoricalVaR(confidence_level=0.95, holding_period=1)
        returns = [-0.02] * 50 + [0.02] * 50

        result_1d = calculator.calculate(returns, 100000)

        calculator.holding_period = 10
        result_10d = calculator.calculate(returns, 100000)

        # 10日VaR应该约等于1日VaR * sqrt(10)
        assert result_10d.var_value > result_1d.var_value


# ============== Parametric VaR Tests ==============

class TestParametricVaR:
    """参数法VaR测试"""

    def test_parametric_var_basic(self):
        """测试基本计算"""
        calculator = ParametricVaR(confidence_level=0.95, holding_period=1)

        returns = [-0.02, -0.01, 0.0, 0.01, 0.02] * 20

        result = calculator.calculate(returns, 100000)

        assert result.var_value > 0
        assert result.method == VaRMethod.PARAMETRIC

    def test_parametric_var_empty_returns(self):
        """测试空收益率"""
        calculator = ParametricVaR()

        result = calculator.calculate([], 100000)
        assert result.var_value == 0

    def test_parametric_var_single_return(self):
        """测试单个收益率"""
        calculator = ParametricVaR()

        result = calculator.calculate([0.01], 100000)
        assert result.var_value == 0


# ============== Monte Carlo VaR Tests ==============

class TestMonteCarloVaR:
    """蒙特卡洛VaR测试"""

    def test_monte_carlo_var_basic(self):
        """测试基本计算"""
        calculator = MonteCarloVaR(
            confidence_level=0.95,
            holding_period=1,
            simulations=1000,
        )

        returns = [-0.02, -0.01, 0.0, 0.01, 0.02] * 20

        result = calculator.calculate(returns, 100000)

        assert result.var_value > 0
        assert result.method == VaRMethod.MONTE_CARLO

    def test_monte_carlo_var_simulations(self):
        """测试模拟次数影响"""
        returns = [-0.02, -0.01, 0.0, 0.01, 0.02] * 20

        calculator_1k = MonteCarloVaR(simulations=1000)
        calculator_10k = MonteCarloVaR(simulations=10000)

        result_1k = calculator_1k.calculate(returns, 100000)
        result_10k = calculator_10k.calculate(returns, 100000)

        # 两者应该接近但不完全相同
        assert abs(result_1k.var_value - result_10k.var_value) < result_10k.var_value * 0.3


# ============== Cornish-Fisher VaR Tests ==============

class TestCornishFisherVaR:
    """Cornish-Fisher VaR测试"""

    def test_cornish_fisher_var_basic(self):
        """测试基本计算"""
        calculator = CornishFisherVaR(confidence_level=0.95)

        # 需要足够多的数据点计算偏度和峰度
        returns = [-0.02, -0.01, 0.0, 0.01, 0.02] * 20

        result = calculator.calculate(returns, 100000)

        assert result.var_value > 0
        assert result.method == VaRMethod.CORNISH_FISHER

    def test_cornish_fisher_vs_parametric(self):
        """测试与参数法对比"""
        # 正态分布收益率
        import random
        random.seed(42)
        returns = [random.gauss(0, 0.02) for _ in range(100)]

        cf_calc = CornishFisherVaR(confidence_level=0.95)
        param_calc = ParametricVaR(confidence_level=0.95)

        cf_result = cf_calc.calculate(returns, 100000)
        param_result = param_calc.calculate(returns, 100000)

        # 正态分布下两者应该接近
        ratio = cf_result.var_value / param_result.var_value if param_result.var_value > 0 else 1
        assert 0.7 < ratio < 1.5


# ============== CVaR Tests ==============

class TestCVaRCalculator:
    """CVaR计算器测试"""

    def test_cvar_basic(self):
        """测试基本计算"""
        calculator = CVaRCalculator(confidence_level=0.95)

        returns = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05] * 10

        result = calculator.calculate(returns, 100000)

        assert result.cvar_value > 0
        assert result.var_value > 0
        assert result.cvar_value >= result.var_value  # CVaR >= VaR

    def test_cvar_empty_returns(self):
        """测试空收益率"""
        calculator = CVaRCalculator()

        result = calculator.calculate([], 100000)
        assert result.cvar_value == 0

    def test_cvar_tail_count(self):
        """测试尾部样本数"""
        calculator = CVaRCalculator(confidence_level=0.95)

        returns = list(range(-50, 50))  # 100个样本
        result = calculator.calculate([r/100 for r in returns], 100000)

        # 5%的样本应该在尾部
        assert result.tail_loss_count == 6  # floor(0.05 * 100) + 1


# ============== Stress Test Tests ==============

class TestStressTestEngine:
    """压力测试引擎测试"""

    def test_stress_engine_creation(self):
        """测试引擎创建"""
        engine = StressTestEngine()
        assert engine is not None

    def test_run_market_crash(self):
        """测试市场崩盘场景"""
        engine = StressTestEngine()

        positions = [
            PositionRisk(
                symbol="AAPL",
                quantity=100,
                market_value=15000,
                weight=0.5,
                volatility=0.25,
            ),
            PositionRisk(
                symbol="GOOGL",
                quantity=50,
                market_value=15000,
                weight=0.5,
                volatility=0.30,
            ),
        ]

        result = engine.run_stress_test(positions, 30000, StressScenario.MARKET_CRASH)

        assert result.scenario == StressScenario.MARKET_CRASH
        assert result.portfolio_loss > 0  # 损失用正数表示

    def test_run_all_scenarios(self):
        """测试运行所有场景"""
        engine = StressTestEngine()

        positions = [
            PositionRisk(
                symbol="AAPL",
                quantity=100,
                market_value=10000,
                weight=1.0,
                volatility=0.25,
            ),
        ]

        results = engine.run_all_scenarios(positions, 10000)

        # 应该有多个场景结果(不包括CUSTOM)
        assert len(results) >= 6

    def test_custom_scenario(self):
        """测试自定义场景"""
        engine = StressTestEngine()

        positions = [
            PositionRisk(
                symbol="AAPL",
                quantity=100,
                market_value=10000,
                weight=1.0,
                volatility=0.25,
            ),
        ]

        custom_params = {
            'equity_shock': -0.30,
            'volatility_multiplier': 2.0,
        }

        result = engine.run_stress_test(
            positions, 10000, StressScenario.CUSTOM, custom_params
        )

        assert result.scenario == StressScenario.CUSTOM


# ============== Portfolio Risk Monitor Tests ==============

class TestPortfolioRiskMonitor:
    """组合风险监控器测试"""

    def test_monitor_creation(self):
        """测试监控器创建"""
        monitor = PortfolioRiskMonitor()

        assert monitor is not None
        assert monitor.state == MonitorState.IDLE

    def test_monitor_with_config(self):
        """测试带配置的创建"""
        config = MonitorConfig(
            update_interval_sec=30,
            var_method=VaRMethod.PARAMETRIC,
        )
        monitor = PortfolioRiskMonitor(config)

        assert monitor.config.update_interval_sec == 30
        assert monitor.config.var_method == VaRMethod.PARAMETRIC

    def test_update_position(self):
        """测试更新持仓"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=15000,
            weight=1.0,
            volatility=0.25,
        )

        monitor.update_position(pos)

        positions = monitor.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"

    def test_remove_position(self):
        """测试移除持仓"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=15000,
            weight=1.0,
            volatility=0.25,
        )

        monitor.update_position(pos)
        monitor.remove_position("AAPL")

        assert len(monitor.get_positions()) == 0

    def test_add_return(self):
        """测试添加收益率"""
        monitor = PortfolioRiskMonitor()

        for i in range(10):
            monitor.add_return(0.01 if i % 2 == 0 else -0.01)

        assert len(monitor._returns_history) == 10

    def test_calculate_var(self):
        """测试计算VaR"""
        monitor = PortfolioRiskMonitor()

        # 添加持仓
        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        # 添加收益率
        for _ in range(50):
            monitor.add_return(-0.02)
        for _ in range(50):
            monitor.add_return(0.02)

        result = monitor.calculate_var(0.95)

        assert result.var_value > 0
        assert result.portfolio_value == 100000

    def test_calculate_cvar(self):
        """测试计算CVaR"""
        monitor = PortfolioRiskMonitor()

        # 添加持仓
        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        # 添加收益率
        for _ in range(50):
            monitor.add_return(-0.02)
        for _ in range(50):
            monitor.add_return(0.02)

        result = monitor.calculate_cvar(0.95)

        assert result.cvar_value > 0
        assert result.cvar_value >= result.var_value

    def test_calculate_all_var(self):
        """测试计算所有VaR"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        for i in range(100):
            monitor.add_return(0.01 if i % 2 == 0 else -0.01)

        results = monitor.calculate_all_var()

        assert 'var' in results
        assert 'cvar' in results
        assert '95%' in results['var']
        assert '99%' in results['var']

    def test_set_limit(self):
        """测试设置限额"""
        monitor = PortfolioRiskMonitor()

        monitor.set_limit(RiskLimitType.VAR_LIMIT, 10000)

        limits = monitor.get_limits()
        assert RiskLimitType.VAR_LIMIT in limits
        assert limits[RiskLimitType.VAR_LIMIT].limit_value == 10000

    def test_remove_limit(self):
        """测试移除限额"""
        monitor = PortfolioRiskMonitor()

        monitor.set_limit(RiskLimitType.VAR_LIMIT, 10000)
        monitor.remove_limit(RiskLimitType.VAR_LIMIT)

        assert RiskLimitType.VAR_LIMIT not in monitor.get_limits()

    def test_check_limits(self):
        """测试检查限额"""
        monitor = PortfolioRiskMonitor()

        # 设置限额
        monitor.set_limit(RiskLimitType.CONCENTRATION_LIMIT, 0.5)

        # 添加高集中度持仓
        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        # 添加收益率（用于VaR计算）
        for _ in range(50):
            monitor.add_return(0.01)

        breached = monitor.check_limits()

        # 集中度100%应该突破50%限额
        assert len(breached) > 0

    def test_run_stress_test(self):
        """测试运行压力测试"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        result = monitor.run_stress_test(StressScenario.MARKET_CRASH)

        assert result.scenario == StressScenario.MARKET_CRASH
        assert result.portfolio_loss > 0  # 损失用正数表示

    def test_run_all_stress_tests(self):
        """测试运行所有压力测试"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        results = monitor.run_all_stress_tests()

        assert len(results) >= 6

    def test_calculate_metrics(self):
        """测试计算综合指标"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
            correlation=0.5,
        )
        monitor.update_position(pos)

        for i in range(100):
            monitor.add_return(0.01 if i % 2 == 0 else -0.01)

        metrics = monitor.calculate_metrics()

        assert metrics.portfolio_value == 100000
        assert metrics.var_95 > 0
        assert metrics.var_99 > 0

    def test_get_statistics(self):
        """测试获取统计信息"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        stats = monitor.get_statistics()

        assert 'state' in stats
        assert 'portfolio_value' in stats
        assert stats['position_count'] == 1

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """测试启动和停止"""
        config = MonitorConfig(update_interval_sec=0.1)
        monitor = PortfolioRiskMonitor(config)

        await monitor.start()
        assert monitor.state == MonitorState.RUNNING

        await asyncio.sleep(0.2)

        await monitor.stop()
        assert monitor.state == MonitorState.IDLE

    def test_alert_callback(self):
        """测试告警回调"""
        monitor = PortfolioRiskMonitor()

        alerts_received = []

        def alert_callback(alert):
            alerts_received.append(alert)

        monitor.add_alert_callback(alert_callback)

        # 设置极低的限额以触发告警
        monitor.set_limit(RiskLimitType.CONCENTRATION_LIMIT, 0.1)

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        # 添加收益率
        for _ in range(50):
            monitor.add_return(0.01)

        monitor.check_limits()

        # 应该触发告警
        assert len(alerts_received) > 0

    def test_acknowledge_alert(self):
        """测试确认告警"""
        monitor = PortfolioRiskMonitor()

        # 设置极低的限额
        monitor.set_limit(RiskLimitType.CONCENTRATION_LIMIT, 0.1)

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        # 添加收益率
        for _ in range(50):
            monitor.add_return(0.01)

        monitor.check_limits()

        alerts = monitor.get_alerts()
        if alerts:
            result = monitor.acknowledge_alert(alerts[0].alert_id)
            assert result is True

            acknowledged = monitor.get_alerts(unacknowledged_only=True)
            assert len(acknowledged) < len(alerts)


# ============== Factory Function Tests ==============

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_risk_monitor(self):
        """测试创建监控器"""
        monitor = create_risk_monitor(
            update_interval_sec=30,
            var_method='parametric',
            lookback_days=252,
        )

        assert monitor is not None
        assert monitor.config.update_interval_sec == 30
        assert monitor.config.var_method == VaRMethod.PARAMETRIC

    def test_create_risk_monitor_invalid_method(self):
        """测试无效方法"""
        monitor = create_risk_monitor(var_method='invalid')

        assert monitor.config.var_method == VaRMethod.HISTORICAL

    def test_create_position_risk(self):
        """测试创建持仓风险"""
        pos = create_position_risk(
            symbol="AAPL",
            quantity=100,
            market_value=15000,
            weight=0.3,
            volatility=0.25,
            beta=1.2,
        )

        assert pos.symbol == "AAPL"
        assert pos.beta == 1.2

    def test_create_risk_limit(self):
        """测试创建风险限额"""
        limit = create_risk_limit(
            limit_type='var_limit',
            limit_value=10000,
            warning_threshold=0.8,
        )

        assert limit.limit_type == RiskLimitType.VAR_LIMIT
        assert limit.limit_value == 10000

    def test_create_risk_limit_invalid_type(self):
        """测试无效限额类型"""
        limit = create_risk_limit(
            limit_type='invalid',
            limit_value=10000,
        )

        assert limit.limit_type == RiskLimitType.VAR_LIMIT

    @pytest.mark.asyncio
    async def test_quick_risk_analysis(self):
        """测试快速风险分析"""
        positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'market_value': 15000, 'volatility': 0.25},
            {'symbol': 'GOOGL', 'quantity': 50, 'market_value': 10000, 'volatility': 0.30},
        ]

        returns = [0.01, -0.01, 0.02, -0.02] * 25

        result = await quick_risk_analysis(positions, returns)

        assert 'portfolio_value' in result
        assert 'var' in result
        assert 'cvar' in result
        assert 'stress_tests' in result
        assert 'metrics' in result


# ============== Edge Case Tests ==============

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_positions(self):
        """测试空持仓"""
        monitor = PortfolioRiskMonitor()

        metrics = monitor.calculate_metrics()
        assert metrics.portfolio_value == 0

    def test_empty_returns(self):
        """测试空收益率"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        result = monitor.calculate_var(0.95)
        assert result.var_value == 0

    def test_single_position(self):
        """测试单个持仓"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        for _ in range(50):
            monitor.add_return(0.01)

        metrics = monitor.calculate_metrics()
        assert metrics.concentration == 1.0  # 100%集中度

    def test_zero_portfolio_value(self):
        """测试零组合价值"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=0,
            market_value=0,
            weight=0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        result = monitor.calculate_var(0.95)
        assert result.var_pct == 0

    def test_all_negative_returns(self):
        """测试全负收益"""
        monitor = PortfolioRiskMonitor()

        pos = PositionRisk(
            symbol="AAPL",
            quantity=100,
            market_value=100000,
            weight=1.0,
            volatility=0.25,
        )
        monitor.update_position(pos)

        for _ in range(100):
            monitor.add_return(-0.01)

        result = monitor.calculate_var(0.95)
        assert result.var_value > 0

    def test_max_drawdown_calculation(self):
        """测试最大回撤计算"""
        monitor = PortfolioRiskMonitor()

        # 模拟价值变化: 100 -> 120 -> 90 -> 110
        pos = PositionRisk(symbol="AAPL", quantity=100, market_value=100, weight=1.0, volatility=0.25)
        monitor.update_position(pos)

        pos.market_value = 120
        monitor.update_position(pos)

        pos.market_value = 90
        monitor.update_position(pos)

        pos.market_value = 110
        monitor.update_position(pos)

        max_dd = monitor._calculate_max_drawdown()
        # 最大回撤从120到90: (120-90)/120 = 25%
        assert abs(max_dd - 0.25) < 0.01


# ============== Serialization Tests ==============

class TestSerialization:
    """序列化测试"""

    def test_var_result_serialization(self):
        """测试VaR结果序列化"""
        result = VaRResult(
            var_value=5000,
            confidence_level=0.95,
            holding_period=1,
            method=VaRMethod.HISTORICAL,
            timestamp=datetime.now(),
            portfolio_value=100000,
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert 'timestamp' in data

    def test_cvar_result_serialization(self):
        """测试CVaR结果序列化"""
        result = CVaRResult(
            cvar_value=6000,
            var_value=5000,
            confidence_level=0.95,
            holding_period=1,
            method=VaRMethod.HISTORICAL,
            timestamp=datetime.now(),
            portfolio_value=100000,
        )

        data = result.to_dict()
        assert isinstance(data, dict)

    def test_stress_test_result_serialization(self):
        """测试压力测试结果序列化"""
        result = StressTestResult(
            scenario=StressScenario.MARKET_CRASH,
            scenario_name="Market Crash",
            portfolio_loss=-10000,
            portfolio_loss_pct=-10.0,
            position_losses={'AAPL': -5000},
            timestamp=datetime.now(),
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert data['scenario'] == 'market_crash'

    def test_risk_metrics_serialization(self):
        """测试风险指标序列化"""
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=100000,
            var_95=5000,
            var_99=7000,
        )

        data = metrics.to_dict()
        assert isinstance(data, dict)
        assert 'var_95' in data
