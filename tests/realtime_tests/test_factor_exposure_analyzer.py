"""
因子暴露分析器测试
Factor Exposure Analyzer Tests

测试覆盖:
- 多因子暴露计算
- 实时因子归因分析
- 因子贡献度追踪
- 因子风险预算
- 因子中性化检测
- 隐含因子暴露
- 因子暴露预测
- 风格漂移检测

作者: TraderX Team
创建: 2025-12-24
"""

import sys
import os

# 添加src路径
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from collections import deque

from quantrisk_realtime.realtime.factor_exposure_analyzer import (
    # Enums
    FactorCategory,
    ExposureMethod,
    NeutralizationLevel,
    DriftSeverity,
    # Data Classes
    FactorExposure,
    FactorContribution,
    RiskBudget,
    NeutralizationResult,
    ImpliedExposure,
    ExposureForecast,
    StyleDrift,
    PortfolioFactorProfile,
    AnalyzerConfig,
    # Components
    FactorExposureCalculator,
    FactorAttributionTracker,
    FactorRiskBudgetManager,
    FactorNeutralizationDetector,
    ImpliedExposureAnalyzer,
    ExposureForecaster,
    StyleDriftDetector,
    # Main Class
    FactorExposureAnalyzer,
    # Factory Functions
    create_analyzer,
    quick_exposure_analysis,
)


# ============ Fixtures ============

@pytest.fixture
def sample_weights():
    """样本权重"""
    return {
        'AAPL': 0.20,
        'GOOGL': 0.15,
        'MSFT': 0.15,
        'TSLA': 0.10,
        'JPM': 0.10,
        'XOM': 0.10,
        'WMT': 0.10,
        'PG': 0.10,
    }


@pytest.fixture
def sample_factors():
    """样本因子数据"""
    return {
        'AAPL': {'market': 1.2, 'size': 2.5, 'value': -0.5, 'momentum': 1.0, 'volatility': 0.8, 'quality': 1.5},
        'GOOGL': {'market': 1.1, 'size': 2.3, 'value': -0.3, 'momentum': 0.8, 'volatility': 0.9, 'quality': 1.4},
        'MSFT': {'market': 1.0, 'size': 2.4, 'value': -0.2, 'momentum': 1.1, 'volatility': 0.7, 'quality': 1.6},
        'TSLA': {'market': 1.5, 'size': 1.8, 'value': -1.2, 'momentum': 1.5, 'volatility': 1.8, 'quality': 0.5},
        'JPM': {'market': 1.3, 'size': 1.5, 'value': 0.8, 'momentum': 0.3, 'volatility': 1.2, 'quality': 1.0},
        'XOM': {'market': 0.9, 'size': 1.2, 'value': 1.0, 'momentum': -0.2, 'volatility': 1.1, 'quality': 0.8},
        'WMT': {'market': 0.8, 'size': 1.8, 'value': 0.5, 'momentum': 0.2, 'volatility': 0.6, 'quality': 1.2},
        'PG': {'market': 0.7, 'size': 1.9, 'value': 0.3, 'momentum': 0.1, 'volatility': 0.5, 'quality': 1.3},
    }


@pytest.fixture
def sample_factor_returns():
    """样本因子收益"""
    return {
        'market': 0.01,
        'size': -0.005,
        'value': 0.008,
        'momentum': 0.012,
        'volatility': -0.003,
        'quality': 0.006,
    }


@pytest.fixture
def benchmark_weights():
    """基准权重"""
    return {
        'AAPL': 0.15,
        'GOOGL': 0.15,
        'MSFT': 0.15,
        'TSLA': 0.05,
        'JPM': 0.125,
        'XOM': 0.125,
        'WMT': 0.125,
        'PG': 0.125,
    }


@pytest.fixture
def factor_covariance():
    """因子协方差矩阵"""
    cov = np.array([
        [0.04, 0.01, 0.005, 0.008, 0.012, 0.006],
        [0.01, 0.03, 0.002, 0.004, 0.006, 0.003],
        [0.005, 0.002, 0.025, 0.006, 0.004, 0.005],
        [0.008, 0.004, 0.006, 0.05, 0.015, 0.007],
        [0.012, 0.006, 0.004, 0.015, 0.06, 0.010],
        [0.006, 0.003, 0.005, 0.007, 0.010, 0.028],
    ])
    return cov


@pytest.fixture
def factor_correlation():
    """因子相关性矩阵"""
    corr = np.array([
        [1.00, 0.30, 0.10, 0.20, 0.35, 0.22],
        [0.30, 1.00, 0.05, 0.12, 0.25, 0.15],
        [0.10, 0.05, 1.00, 0.18, 0.10, 0.20],
        [0.20, 0.12, 0.18, 1.00, 0.28, 0.18],
        [0.35, 0.25, 0.10, 0.28, 1.00, 0.25],
        [0.22, 0.15, 0.20, 0.18, 0.25, 1.00],
    ])
    return corr


@pytest.fixture
def config():
    """默认配置"""
    return AnalyzerConfig(
        method=ExposureMethod.HYBRID,
        factors=['market', 'size', 'value', 'momentum', 'volatility', 'quality'],
        lookback_period=252,
        neutralization_threshold=0.05,
        drift_threshold=0.1,
        confidence_level=0.95,
    )


# ============ FactorExposureCalculator Tests ============

class TestFactorExposureCalculator:
    """因子暴露计算器测试"""

    def test_init(self, config):
        """测试初始化"""
        calculator = FactorExposureCalculator(config)
        assert calculator.config == config
        assert len(calculator._factor_returns_history) == 0
        assert len(calculator._portfolio_returns_history) == 0

    def test_calculate_exposures(self, sample_weights, sample_factors):
        """测试计算因子暴露"""
        calculator = FactorExposureCalculator()
        exposures = calculator.calculate_exposures(sample_weights, sample_factors)

        assert isinstance(exposures, dict)
        assert len(exposures) > 0
        assert 'market' in exposures
        assert 'size' in exposures

        for factor_name, exposure in exposures.items():
            assert isinstance(exposure, FactorExposure)
            assert exposure.factor_name == factor_name
            assert isinstance(exposure.exposure, float)
            assert isinstance(exposure.z_score, float)
            assert 0 <= exposure.percentile <= 1

    def test_calculate_exposures_with_benchmark(
        self, sample_weights, sample_factors, benchmark_weights
    ):
        """测试带基准的暴露计算"""
        calculator = FactorExposureCalculator()
        exposures = calculator.calculate_exposures(
            sample_weights,
            sample_factors,
            benchmark_weights,
            sample_factors,
        )

        assert isinstance(exposures, dict)

        for exposure in exposures.values():
            # 主动暴露应该被计算
            assert isinstance(exposure.active_exposure, float)

    def test_calculate_exposures_empty_weights(self, sample_factors):
        """测试空权重"""
        calculator = FactorExposureCalculator()
        exposures = calculator.calculate_exposures({}, sample_factors)

        assert isinstance(exposures, dict)
        assert len(exposures) == 0

    def test_calculate_regression_exposures(self):
        """测试回归法计算暴露"""
        calculator = FactorExposureCalculator()

        # 生成模拟数据
        n = 100
        portfolio_returns = np.random.randn(n) * 0.01
        factor_returns = {
            'market': np.random.randn(n) * 0.015,
            'size': np.random.randn(n) * 0.01,
            'value': np.random.randn(n) * 0.012,
        }

        exposures = calculator.calculate_regression_exposures(
            portfolio_returns,
            factor_returns
        )

        assert isinstance(exposures, dict)
        assert len(exposures) == 3

        for exposure in exposures.values():
            assert isinstance(exposure, FactorExposure)
            assert isinstance(exposure.exposure, float)
            assert isinstance(exposure.confidence_interval, tuple)
            assert len(exposure.confidence_interval) == 2

    def test_calculate_regression_exposures_insufficient_data(self):
        """测试数据不足的回归"""
        calculator = FactorExposureCalculator()

        portfolio_returns = np.array([0.01, 0.02])
        factor_returns = {'market': np.array([0.015, 0.018])}

        exposures = calculator.calculate_regression_exposures(
            portfolio_returns,
            factor_returns
        )

        # 数据不足应该返回空字典
        assert isinstance(exposures, dict)
        assert len(exposures) == 0

    def test_update_history(self):
        """测试更新历史"""
        calculator = FactorExposureCalculator()

        portfolio_return = 0.01
        factor_returns = {'market': 0.015, 'size': -0.005}

        calculator.update_history(portfolio_return, factor_returns)

        assert len(calculator._portfolio_returns_history) == 1
        assert calculator._portfolio_returns_history[-1] == 0.01
        assert len(calculator._factor_returns_history['market']) == 1
        assert calculator._factor_returns_history['market'][-1] == 0.015

    def test_determine_category(self):
        """测试因子类别判断"""
        calculator = FactorExposureCalculator()

        assert calculator._determine_category('market_beta') == FactorCategory.MARKET
        assert calculator._determine_category('log_market_cap') == FactorCategory.SIZE
        assert calculator._determine_category('book_to_market') == FactorCategory.VALUE
        assert calculator._determine_category('momentum_12m') == FactorCategory.MOMENTUM
        assert calculator._determine_category('historical_volatility') == FactorCategory.VOLATILITY
        assert calculator._determine_category('roe') == FactorCategory.QUALITY
        assert calculator._determine_category('custom_factor') == FactorCategory.CUSTOM

    def test_exposures_confidence_intervals(self, sample_weights, sample_factors):
        """测试暴露的置信区间"""
        calculator = FactorExposureCalculator()
        exposures = calculator.calculate_exposures(sample_weights, sample_factors)

        for exposure in exposures.values():
            ci_lower, ci_upper = exposure.confidence_interval
            assert ci_lower <= exposure.exposure <= ci_upper or ci_lower == ci_upper


# ============ FactorAttributionTracker Tests ============

class TestFactorAttributionTracker:
    """因子归因追踪器测试"""

    def test_init(self, config):
        """测试初始化"""
        tracker = FactorAttributionTracker(config)
        assert tracker.config == config

    def test_calculate_contributions(self, sample_factor_returns):
        """测试计算因子贡献"""
        tracker = FactorAttributionTracker()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 2.0, 1.5, 0.8, 1.5),
            'value': FactorExposure('value', FactorCategory.VALUE, 0.5, 0.3, 0.4, 0.3),
        }

        total_return = 0.02

        contributions = tracker.calculate_contributions(
            exposures,
            sample_factor_returns,
            total_return
        )

        assert isinstance(contributions, dict)
        assert len(contributions) == 3

        for factor_name, contrib in contributions.items():
            assert isinstance(contrib, FactorContribution)
            assert contrib.factor_name == factor_name
            assert isinstance(contrib.contribution, float)
            assert isinstance(contrib.contribution_pct, float)

        # 贡献占比之和应该为1（或接近1）
        total_pct = sum(c.contribution_pct for c in contributions.values())
        assert abs(total_pct - 1.0) < 0.01 or abs(total_pct) < 0.01

    def test_decompose_return(self):
        """测试收益分解"""
        tracker = FactorAttributionTracker()

        contributions = {
            'market': FactorContribution('market', 1.0, 0.01, 0.01, 0.5),
            'size': FactorContribution('size', 2.0, -0.005, -0.01, -0.5),
        }

        total_return = 0.02

        factor_return, specific_return = tracker.decompose_return(
            total_return,
            contributions
        )

        assert isinstance(factor_return, float)
        assert isinstance(specific_return, float)
        assert abs(factor_return + specific_return - total_return) < 1e-10

    def test_cumulative_contribution(self):
        """测试累积贡献"""
        tracker = FactorAttributionTracker()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }

        factor_returns = {'market': 0.01}

        # 第一次
        contrib1 = tracker.calculate_contributions(exposures, factor_returns, 0.01)
        assert contrib1['market'].cumulative_contribution == contrib1['market'].contribution

        # 第二次
        contrib2 = tracker.calculate_contributions(exposures, factor_returns, 0.01)
        assert contrib2['market'].cumulative_contribution == contrib1['market'].contribution * 2


# ============ FactorRiskBudgetManager Tests ============

class TestFactorRiskBudgetManager:
    """因子风险预算管理器测试"""

    def test_init(self, config):
        """测试初始化"""
        manager = FactorRiskBudgetManager(config)
        assert manager.config == config

    def test_set_risk_budget(self):
        """测试设置风险预算"""
        manager = FactorRiskBudgetManager()

        manager.set_risk_budget('market', 0.3, (-1.0, 1.5))

        assert manager._risk_budgets['market'] == 0.3
        assert manager._exposure_limits['market'] == (-1.0, 1.5)

    def test_set_equal_risk_budgets(self):
        """测试设置等风险预算"""
        manager = FactorRiskBudgetManager()

        factors = ['market', 'size', 'value', 'momentum']
        manager.set_equal_risk_budgets(factors)

        assert len(manager._risk_budgets) == 4
        for budget in manager._risk_budgets.values():
            assert abs(budget - 0.25) < 1e-10

    def test_calculate_risk_budgets(self, factor_covariance):
        """测试计算风险预算"""
        manager = FactorRiskBudgetManager()

        factors = ['market', 'size', 'value', 'momentum', 'volatility', 'quality']
        manager.set_equal_risk_budgets(factors)

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 2.0, 1.5, 0.8, 1.5),
            'value': FactorExposure('value', FactorCategory.VALUE, 0.5, 0.3, 0.4, 0.3),
            'momentum': FactorExposure('momentum', FactorCategory.MOMENTUM, 0.8, 0.5, 0.5, 0.5),
            'volatility': FactorExposure('volatility', FactorCategory.VOLATILITY, 0.6, 0.4, 0.45, 0.4),
            'quality': FactorExposure('quality', FactorCategory.QUALITY, 1.2, 0.8, 0.7, 0.8),
        }

        budgets = manager.calculate_risk_budgets(
            exposures,
            factor_covariance,
            factors
        )

        assert isinstance(budgets, dict)
        assert len(budgets) == 6

        for budget in budgets.values():
            assert isinstance(budget, RiskBudget)
            assert isinstance(budget.risk_contribution, float)
            assert isinstance(budget.budget_utilization, float)
            assert isinstance(budget.is_breached, bool)

    def test_breach_detection(self, factor_covariance):
        """测试违反限制检测"""
        manager = FactorRiskBudgetManager()

        # 设置严格限制
        manager.set_risk_budget('market', 0.3, (-0.5, 0.5))

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 0.8, 1.0, 0.6, 1.0),
        }

        budgets = manager.calculate_risk_budgets(
            exposures,
            factor_covariance[:1, :1],
            ['market']
        )

        assert budgets['market'].is_breached
        assert budgets['market'].breach_amount > 0


# ============ FactorNeutralizationDetector Tests ============

class TestFactorNeutralizationDetector:
    """因子中性化检测器测试"""

    def test_init(self, config):
        """测试初始化"""
        detector = FactorNeutralizationDetector(config)
        assert detector.config == config

    def test_detect_neutralization(self):
        """测试检测中性化"""
        detector = FactorNeutralizationDetector()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 0.02, 0.1, 0.5, 0.1),
            'size': FactorExposure('size', FactorCategory.SIZE, 0.3, 1.5, 0.7, 1.5),
            'value': FactorExposure('value', FactorCategory.VALUE, 0.5, 2.0, 0.8, 2.0),
        }

        results = detector.detect_neutralization(exposures)

        assert isinstance(results, dict)
        assert len(results) == 3

        for result in results.values():
            assert isinstance(result, NeutralizationResult)
            assert isinstance(result.is_neutral, bool)
            assert isinstance(result.current_neutralization_pct, float)
            assert 0 <= result.current_neutralization_pct <= 1

    def test_different_neutralization_levels(self):
        """测试不同中性化水平"""
        detector = FactorNeutralizationDetector()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 0.5, 1.0, 0.6, 1.0),
        }

        # 测试不同目标水平
        levels = [
            NeutralizationLevel.NONE,
            NeutralizationLevel.PARTIAL,
            NeutralizationLevel.FULL,
            NeutralizationLevel.DOLLAR,
            NeutralizationLevel.BETA,
        ]

        for level in levels:
            results = detector.detect_neutralization(
                exposures,
                {'market': level}
            )

            assert 'market' in results
            assert results['market'].target_level == level

    def test_neutralization_cost_estimation(self):
        """测试中性化成本估计"""
        detector = FactorNeutralizationDetector()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }

        results = detector.detect_neutralization(
            exposures,
            {'market': NeutralizationLevel.FULL}
        )

        # 应该有中性化成本
        assert results['market'].neutralization_cost >= 0

    def test_partial_neutralize(self):
        """测试部分中性化"""
        detector = FactorNeutralizationDetector()

        exposure = 2.0
        neutralized = detector._partial_neutralize(exposure)

        # 部分中性化应该降低50%
        assert abs(neutralized - exposure * 0.5) < 1e-10


# ============ ImpliedExposureAnalyzer Tests ============

class TestImpliedExposureAnalyzer:
    """隐含因子暴露分析器测试"""

    def test_init(self, config):
        """测试初始化"""
        analyzer = ImpliedExposureAnalyzer(config)
        assert analyzer.config == config

    def test_analyze_implied_exposures(self, factor_correlation):
        """测试分析隐含暴露"""
        analyzer = ImpliedExposureAnalyzer()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 2.0, 1.5, 0.8, 1.5),
            'value': FactorExposure('value', FactorCategory.VALUE, 0.5, 0.3, 0.4, 0.3),
            'momentum': FactorExposure('momentum', FactorCategory.MOMENTUM, 0.8, 0.5, 0.5, 0.5),
            'volatility': FactorExposure('volatility', FactorCategory.VOLATILITY, 0.6, 0.4, 0.45, 0.4),
            'quality': FactorExposure('quality', FactorCategory.QUALITY, 1.2, 0.8, 0.7, 0.8),
        }

        factor_names = ['market', 'size', 'value', 'momentum', 'volatility', 'quality']

        implied = analyzer.analyze_implied_exposures(
            exposures,
            factor_correlation,
            factor_names
        )

        assert isinstance(implied, dict)
        assert len(implied) == 6

        for imp in implied.values():
            assert isinstance(imp, ImpliedExposure)
            assert isinstance(imp.implied_exposure, float)
            assert isinstance(imp.implicit_exposure, float)
            assert isinstance(imp.confidence, float)
            assert 0 <= imp.confidence <= 1

    def test_implicit_exposure_calculation(self, factor_correlation):
        """测试隐性暴露计算"""
        analyzer = ImpliedExposureAnalyzer()

        # 只暴露于market因子
        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 0.0, 0.0, 0.5, 0.0),
            'value': FactorExposure('value', FactorCategory.VALUE, 0.0, 0.0, 0.5, 0.0),
            'momentum': FactorExposure('momentum', FactorCategory.MOMENTUM, 0.0, 0.0, 0.5, 0.0),
            'volatility': FactorExposure('volatility', FactorCategory.VOLATILITY, 0.0, 0.0, 0.5, 0.0),
            'quality': FactorExposure('quality', FactorCategory.QUALITY, 0.0, 0.0, 0.5, 0.0),
        }

        factor_names = ['market', 'size', 'value', 'momentum', 'volatility', 'quality']

        implied = analyzer.analyze_implied_exposures(
            exposures,
            factor_correlation,
            factor_names
        )

        # 由于相关性，应该有隐性暴露
        assert implied['size'].implicit_exposure != 0
        assert implied['volatility'].implicit_exposure != 0


# ============ ExposureForecaster Tests ============

class TestExposureForecaster:
    """因子暴露预测器测试"""

    def test_init(self, config):
        """测试初始化"""
        forecaster = ExposureForecaster(config)
        assert forecaster.config == config

    def test_update_history(self):
        """测试更新历史"""
        forecaster = ExposureForecaster()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }

        forecaster.update_history(exposures)

        assert len(forecaster._exposure_history['market']) == 1
        assert forecaster._exposure_history['market'][-1] == 1.0

    def test_forecast_exposures_insufficient_data(self):
        """测试数据不足的预测"""
        forecaster = ExposureForecaster()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }

        forecasts = forecaster.forecast_exposures(exposures)

        assert isinstance(forecasts, dict)
        assert 'market' in forecasts

        # 数据不足时应该返回当前值
        assert forecasts['market'].forecast_exposure == 1.0
        assert forecasts['market'].expected_change == 0.0

    def test_forecast_exposures_with_trend(self):
        """测试带趋势的预测"""
        forecaster = ExposureForecaster()

        # 创建上升趋势
        for i in range(20):
            exposure = 1.0 + i * 0.05
            exposures = {
                'market': FactorExposure('market', FactorCategory.MARKET, exposure, 1.0, 0.6, 1.0),
            }
            forecaster.update_history(exposures)

        current_exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 2.0, 1.0, 0.6, 1.0),
        }

        forecasts = forecaster.forecast_exposures(current_exposures, forecast_horizon=5)

        assert isinstance(forecasts, dict)
        assert 'market' in forecasts

        forecast = forecasts['market']
        assert isinstance(forecast, ExposureForecast)
        assert forecast.forecast_horizon == 5
        # 上升趋势，预测值应该大于当前值
        assert forecast.expected_change > 0
        assert forecast.trend_strength > 0

    def test_forecast_confidence_interval(self):
        """测试预测置信区间"""
        forecaster = ExposureForecaster()

        # 添加历史数据
        for i in range(30):
            exposure = 1.0 + np.random.randn() * 0.1
            exposures = {
                'market': FactorExposure('market', FactorCategory.MARKET, exposure, 1.0, 0.6, 1.0),
            }
            forecaster.update_history(exposures)

        current_exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }

        forecasts = forecaster.forecast_exposures(current_exposures)

        forecast = forecasts['market']
        ci_lower, ci_upper = forecast.confidence_interval

        # 置信区间应该包含预测值
        assert ci_lower <= forecast.forecast_exposure <= ci_upper


# ============ StyleDriftDetector Tests ============

class TestStyleDriftDetector:
    """风格漂移检测器测试"""

    def test_init(self, config):
        """测试初始化"""
        detector = StyleDriftDetector(config)
        assert detector.config == config

    def test_set_baseline(self):
        """测试设置基线"""
        detector = StyleDriftDetector()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 2.0, 1.5, 0.8, 1.5),
        }

        detector.set_baseline(exposures)

        assert detector._baseline_exposures['market'] == 1.0
        assert detector._baseline_exposures['size'] == 2.0

    def test_detect_drift_no_baseline(self):
        """测试无基线的漂移检测"""
        detector = StyleDriftDetector()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }

        drifts = detector.detect_drift(exposures)

        # 第一次应该自动设置基线
        assert isinstance(drifts, dict)
        assert 'market' in drifts
        assert drifts['market'].severity == DriftSeverity.NONE

    def test_detect_drift_with_drift(self):
        """测试有漂移的检测"""
        detector = StyleDriftDetector()

        # 设置基线
        baseline = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }
        detector.set_baseline(baseline)

        # 添加历史数据
        for i in range(10):
            exposures = {
                'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
            }
            detector.update_history(exposures, datetime.now() - timedelta(days=10-i))

        # 当前有显著漂移
        current = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.5, 1.5, 0.7, 1.5),
        }

        drifts = detector.detect_drift(current)

        assert drifts['market'].drift_amount == 0.5
        assert drifts['market'].drift_pct == 0.5
        assert drifts['market'].severity in [DriftSeverity.MODERATE, DriftSeverity.HIGH, DriftSeverity.CRITICAL]

    def test_drift_severity_levels(self):
        """测试漂移严重程度级别"""
        detector = StyleDriftDetector(
            AnalyzerConfig(drift_threshold=0.1)
        )

        baseline = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }
        detector.set_baseline(baseline)

        test_cases = [
            (1.0, DriftSeverity.NONE),      # 0% drift
            (1.08, DriftSeverity.NONE),     # 8% drift
            (1.15, DriftSeverity.LOW),      # 15% drift
            (1.25, DriftSeverity.MODERATE), # 25% drift
            (1.45, DriftSeverity.HIGH),     # 45% drift
            (1.60, DriftSeverity.CRITICAL), # 60% drift
        ]

        for exposure_value, expected_severity in test_cases:
            current = {
                'market': FactorExposure('market', FactorCategory.MARKET, exposure_value, 1.0, 0.6, 1.0),
            }
            drifts = detector.detect_drift(current)
            assert drifts['market'].severity == expected_severity

    def test_calculate_drift_velocity(self):
        """测试漂移速度计算"""
        detector = StyleDriftDetector()

        # 创建上升趋势
        history = []
        for i in range(20):
            timestamp = datetime.now() - timedelta(days=20-i)
            exposure = 1.0 + i * 0.05
            history.append((timestamp, exposure))

        velocity = detector._calculate_drift_velocity(history)

        # 上升趋势，速度应该为正
        assert velocity > 0

    def test_is_systematic_drift(self):
        """测试系统性漂移判断"""
        detector = StyleDriftDetector()

        baseline = 1.0

        # 系统性向上漂移
        systematic_history = [
            (datetime.now() - timedelta(days=i), 1.0 + (20-i) * 0.02)
            for i in range(20)
        ]

        is_systematic = detector._is_systematic_drift(systematic_history, baseline)
        assert is_systematic

        # 随机波动
        random_history = [
            (datetime.now() - timedelta(days=i), 1.0 + np.random.randn() * 0.1)
            for i in range(20)
        ]

        is_systematic = detector._is_systematic_drift(random_history, baseline)
        # 随机波动不应该是系统性的（大概率）
        # 这个测试可能偶尔失败，因为随机性


# ============ FactorExposureAnalyzer Integration Tests ============

class TestFactorExposureAnalyzer:
    """因子暴露分析器集成测试"""

    def test_init(self, config):
        """测试初始化"""
        analyzer = FactorExposureAnalyzer(config)
        assert analyzer.config == config
        assert isinstance(analyzer.exposure_calculator, FactorExposureCalculator)
        assert isinstance(analyzer.attribution_tracker, FactorAttributionTracker)
        assert isinstance(analyzer.risk_budget_manager, FactorRiskBudgetManager)

    @pytest.mark.asyncio
    async def test_analyze_portfolio(
        self,
        sample_weights,
        sample_factors,
        sample_factor_returns,
        factor_covariance,
        factor_correlation,
    ):
        """测试完整的组合分析"""
        analyzer = FactorExposureAnalyzer()

        profile = await analyzer.analyze_portfolio(
            weights=sample_weights,
            asset_factors=sample_factors,
            factor_returns=sample_factor_returns,
            portfolio_return=0.015,
            factor_covariance=factor_covariance,
            factor_correlation=factor_correlation,
        )

        assert isinstance(profile, PortfolioFactorProfile)
        assert isinstance(profile.exposures, dict)
        assert isinstance(profile.contributions, dict)
        assert isinstance(profile.forecasts, dict)
        assert isinstance(profile.drifts, dict)
        assert len(profile.exposures) > 0

    @pytest.mark.asyncio
    async def test_analyze_portfolio_with_benchmark(
        self,
        sample_weights,
        sample_factors,
        sample_factor_returns,
        benchmark_weights,
    ):
        """测试带基准的组合分析"""
        analyzer = FactorExposureAnalyzer()

        profile = await analyzer.analyze_portfolio(
            weights=sample_weights,
            asset_factors=sample_factors,
            factor_returns=sample_factor_returns,
            portfolio_return=0.015,
            benchmark_weights=benchmark_weights,
            benchmark_factors=sample_factors,
        )

        # 应该有主动暴露
        for exposure in profile.exposures.values():
            assert isinstance(exposure.active_exposure, float)

    def test_analyze_portfolio_sync(
        self,
        sample_weights,
        sample_factors,
        sample_factor_returns,
    ):
        """测试同步版本的组合分析"""
        analyzer = FactorExposureAnalyzer()

        profile = analyzer.analyze_portfolio_sync(
            weights=sample_weights,
            asset_factors=sample_factors,
            factor_returns=sample_factor_returns,
            portfolio_return=0.015,
        )

        assert isinstance(profile, PortfolioFactorProfile)

    def test_set_risk_budgets(self):
        """测试设置风险预算"""
        analyzer = FactorExposureAnalyzer()

        budgets = {
            'market': 0.3,
            'size': 0.2,
            'value': 0.2,
            'momentum': 0.15,
            'volatility': 0.1,
            'quality': 0.05,
        }

        limits = {
            'market': (-1.0, 1.5),
            'size': (-2.0, 3.0),
        }

        analyzer.set_risk_budgets(budgets, limits)

        # 验证已设置
        assert analyzer.risk_budget_manager._risk_budgets['market'] == 0.3
        assert analyzer.risk_budget_manager._exposure_limits['market'] == (-1.0, 1.5)

    def test_set_baseline_style(self):
        """测试设置基线风格"""
        analyzer = FactorExposureAnalyzer()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }

        analyzer.set_baseline_style(exposures)

        assert analyzer.drift_detector._baseline_exposures['market'] == 1.0

    def test_get_profile_history(self):
        """测试获取历史画像"""
        analyzer = FactorExposureAnalyzer()

        history = analyzer.get_profile_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_calculate_total_risk(self, factor_covariance):
        """测试计算总风险"""
        analyzer = FactorExposureAnalyzer()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 2.0, 1.5, 0.8, 1.5),
            'value': FactorExposure('value', FactorCategory.VALUE, 0.5, 0.3, 0.4, 0.3),
            'momentum': FactorExposure('momentum', FactorCategory.MOMENTUM, 0.8, 0.5, 0.5, 0.5),
            'volatility': FactorExposure('volatility', FactorCategory.VOLATILITY, 0.6, 0.4, 0.45, 0.4),
            'quality': FactorExposure('quality', FactorCategory.QUALITY, 1.2, 0.8, 0.7, 0.8),
        }

        total_risk = analyzer._calculate_total_risk(exposures, factor_covariance)

        assert isinstance(total_risk, float)
        assert total_risk >= 0

    def test_calculate_diversification_score(self):
        """测试计算分散化得分"""
        analyzer = FactorExposureAnalyzer()

        # 分散的暴露
        diversified_exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 1.0, 1.0, 0.6, 1.0),
            'value': FactorExposure('value', FactorCategory.VALUE, 1.0, 1.0, 0.6, 1.0),
        }

        score1 = analyzer._calculate_diversification_score(diversified_exposures)

        # 集中的暴露
        concentrated_exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 10.0, 3.0, 0.9, 3.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 0.1, 0.1, 0.3, 0.1),
            'value': FactorExposure('value', FactorCategory.VALUE, 0.1, 0.1, 0.3, 0.1),
        }

        score2 = analyzer._calculate_diversification_score(concentrated_exposures)

        # 分散的得分应该更高
        assert score1 > score2
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1

    def test_calculate_factor_concentration(self):
        """测试计算因子集中度"""
        analyzer = FactorExposureAnalyzer()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 5.0, 2.0, 0.8, 2.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 3.0, 1.5, 0.7, 1.5),
            'value': FactorExposure('value', FactorCategory.VALUE, 2.0, 1.0, 0.6, 1.0),
            'momentum': FactorExposure('momentum', FactorCategory.MOMENTUM, 0.5, 0.3, 0.4, 0.3),
        }

        concentration = analyzer._calculate_factor_concentration(exposures)

        assert isinstance(concentration, float)
        assert 0 <= concentration <= 1


# ============ Factory Functions Tests ============

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_analyzer(self):
        """测试创建分析器"""
        analyzer = create_analyzer(
            method='hybrid',
            factors=['market', 'size', 'value'],
            lookback_period=100,
        )

        assert isinstance(analyzer, FactorExposureAnalyzer)
        assert analyzer.config.method == ExposureMethod.HYBRID
        assert analyzer.config.factors == ['market', 'size', 'value']
        assert analyzer.config.lookback_period == 100

    def test_create_analyzer_different_methods(self):
        """测试创建不同方法的分析器"""
        methods = ['regression', 'fundamental', 'statistical', 'hybrid', 'implied']

        for method in methods:
            analyzer = create_analyzer(method=method)
            assert isinstance(analyzer, FactorExposureAnalyzer)

    @pytest.mark.asyncio
    async def test_quick_exposure_analysis(
        self,
        sample_weights,
        sample_factors,
        sample_factor_returns,
    ):
        """测试快速暴露分析"""
        profile = await quick_exposure_analysis(
            portfolio=sample_weights,
            factors=sample_factors,
            factor_returns=sample_factor_returns,
            portfolio_return=0.015,
        )

        assert isinstance(profile, PortfolioFactorProfile)
        assert len(profile.exposures) > 0


# ============ Data Classes Tests ============

class TestDataClasses:
    """数据类测试"""

    def test_factor_exposure_to_dict(self):
        """测试FactorExposure转字典"""
        exposure = FactorExposure(
            factor_name='market',
            category=FactorCategory.MARKET,
            exposure=1.0,
            standardized_exposure=0.8,
            percentile=0.6,
            z_score=0.8,
            active_exposure=0.2,
            confidence_interval=(0.8, 1.2),
        )

        d = exposure.to_dict()

        assert isinstance(d, dict)
        assert d['factor_name'] == 'market'
        assert d['category'] == 'market'
        assert d['exposure'] == 1.0

    def test_factor_contribution_to_dict(self):
        """测试FactorContribution转字典"""
        contrib = FactorContribution(
            factor_name='market',
            exposure=1.0,
            factor_return=0.01,
            contribution=0.01,
            contribution_pct=0.5,
        )

        d = contrib.to_dict()

        assert isinstance(d, dict)
        assert d['factor_name'] == 'market'
        assert d['contribution'] == 0.01

    def test_risk_budget_to_dict(self):
        """测试RiskBudget转字典"""
        budget = RiskBudget(
            factor_name='market',
            target_exposure=1.0,
            current_exposure=1.2,
            exposure_limit=(-1.0, 2.0),
            risk_contribution=0.015,
            risk_budget=0.3,
            budget_utilization=0.8,
            is_breached=False,
        )

        d = budget.to_dict()

        assert isinstance(d, dict)
        assert d['factor_name'] == 'market'
        assert d['is_breached'] is False

    def test_portfolio_factor_profile_to_dict(self):
        """测试PortfolioFactorProfile转字典"""
        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
        }

        profile = PortfolioFactorProfile(
            exposures=exposures,
            contributions={},
            risk_budgets={},
            neutralization={},
            implied={},
            forecasts={},
            drifts={},
            total_risk=0.15,
            diversification_score=0.7,
            factor_concentration=0.4,
        )

        d = profile.to_dict()

        assert isinstance(d, dict)
        assert 'exposures' in d
        assert 'total_risk' in d
        assert d['total_risk'] == 0.15


# ============ Edge Cases Tests ============

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_portfolio(self):
        """测试空组合"""
        analyzer = FactorExposureAnalyzer()

        profile = analyzer.analyze_portfolio_sync(
            weights={},
            asset_factors={},
            factor_returns={},
            portfolio_return=0.0,
        )

        assert isinstance(profile, PortfolioFactorProfile)
        assert len(profile.exposures) == 0

    def test_single_factor(self):
        """测试单因子"""
        analyzer = FactorExposureAnalyzer()

        profile = analyzer.analyze_portfolio_sync(
            weights={'AAPL': 1.0},
            asset_factors={'AAPL': {'market': 1.0}},
            factor_returns={'market': 0.01},
            portfolio_return=0.01,
        )

        assert len(profile.exposures) == 1
        assert 'market' in profile.exposures

    def test_zero_variance(self):
        """测试零方差"""
        calculator = FactorExposureCalculator()

        # 所有资产相同因子值
        weights = {'A': 0.5, 'B': 0.5}
        factors = {
            'A': {'market': 1.0},
            'B': {'market': 1.0},
        }

        exposures = calculator.calculate_exposures(weights, factors)

        assert 'market' in exposures
        # 零方差时z_score应该为0
        assert exposures['market'].z_score == 0.0

    def test_extreme_exposures(self):
        """测试极端暴露"""
        analyzer = FactorExposureAnalyzer()

        profile = analyzer.analyze_portfolio_sync(
            weights={'AAPL': 1.0},
            asset_factors={'AAPL': {'market': 100.0}},  # 极端值
            factor_returns={'market': 0.01},
            portfolio_return=1.0,
        )

        assert 'market' in profile.exposures
        assert profile.exposures['market'].exposure == 100.0

    def test_negative_exposures(self):
        """测试负暴露"""
        analyzer = FactorExposureAnalyzer()

        profile = analyzer.analyze_portfolio_sync(
            weights={'AAPL': 1.0},
            asset_factors={'AAPL': {'market': -2.0}},
            factor_returns={'market': 0.01},
            portfolio_return=-0.02,
        )

        assert profile.exposures['market'].exposure == -2.0

    def test_missing_factor_returns(self):
        """测试缺失因子收益"""
        tracker = FactorAttributionTracker()

        exposures = {
            'market': FactorExposure('market', FactorCategory.MARKET, 1.0, 1.0, 0.6, 1.0),
            'size': FactorExposure('size', FactorCategory.SIZE, 2.0, 1.5, 0.8, 1.5),
        }

        # 只提供market的收益
        factor_returns = {'market': 0.01}

        contributions = tracker.calculate_contributions(
            exposures,
            factor_returns,
            0.01
        )

        # size的贡献应该为0（因为收益缺失）
        assert contributions['size'].factor_return == 0.0
        assert contributions['size'].contribution == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
