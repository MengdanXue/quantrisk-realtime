"""
绩效归因模块测试
Performance Attribution Tests

测试:
- Brinson归因分析
- 因子归因分析
- 风险归因分析
- 交易归因分析
- 多期间链接

作者: TraderX Team
创建: 2025-12-24
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np

import sys
sys.path.insert(0, 'src')

from quantrisk_realtime.realtime.performance_attribution import (
    # Enums
    AttributionMethod,
    AttributionPeriod,
    FactorType,
    # Data Classes
    PositionData,
    BenchmarkData,
    BrinsonResult,
    FactorResult,
    RiskResult,
    TradingResult,
    AttributionConfig,
    AttributionSummary,
    # Attributors
    BrinsonAttributor,
    FactorAttributor,
    RiskAttributor,
    TradingAttributor,
    PerformanceAttribution,
    # Factory Functions
    create_attribution,
    create_position,
    create_benchmark,
    quick_attribution,
)


# ============ Test Fixtures ============

@pytest.fixture
def simple_portfolio():
    """简单组合"""
    return [
        PositionData('AAPL', 0.3, 0.10, 'Technology'),
        PositionData('MSFT', 0.2, 0.08, 'Technology'),
        PositionData('JPM', 0.25, 0.05, 'Financials'),
        PositionData('XOM', 0.25, -0.02, 'Energy'),
    ]


@pytest.fixture
def simple_benchmark():
    """简单基准"""
    return [
        BenchmarkData('TECH_IDX', 0.4, 0.07, 'Technology'),
        BenchmarkData('FIN_IDX', 0.3, 0.04, 'Financials'),
        BenchmarkData('ENERGY_IDX', 0.3, 0.01, 'Energy'),
    ]


@pytest.fixture
def factor_portfolio():
    """带因子暴露的组合"""
    return [
        PositionData('AAPL', 0.4, 0.10, 'Technology',
                    factor_exposures={'market': 1.2, 'size': -0.3, 'value': -0.5, 'momentum': 0.8}),
        PositionData('JPM', 0.3, 0.05, 'Financials',
                    factor_exposures={'market': 1.1, 'size': 0.2, 'value': 0.6, 'momentum': -0.2}),
        PositionData('XOM', 0.3, -0.02, 'Energy',
                    factor_exposures={'market': 0.9, 'size': 0.4, 'value': 0.3, 'momentum': -0.5}),
    ]


@pytest.fixture
def sample_trades():
    """样本交易记录"""
    return [
        {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'price': 150.50,
            'benchmark_price': 150.00,
            'commission': 5.0,
            'realized_return': 0.05,
        },
        {
            'symbol': 'MSFT',
            'side': 'buy',
            'quantity': 50,
            'price': 300.20,
            'benchmark_price': 300.00,
            'commission': 5.0,
            'realized_return': 0.03,
        },
        {
            'symbol': 'JPM',
            'side': 'sell',
            'quantity': 80,
            'price': 145.00,
            'benchmark_price': 145.50,
            'commission': 5.0,
            'realized_return': -0.02,
        },
    ]


@pytest.fixture
def returns_matrix():
    """收益率矩阵"""
    np.random.seed(42)
    # 60天，3个资产
    return np.random.randn(60, 3) * 0.02


# ============ PositionData Tests ============

class TestPositionData:
    """持仓数据测试"""

    def test_create_position_data(self):
        """测试创建持仓数据"""
        pos = PositionData(
            symbol='AAPL',
            weight=0.3,
            return_value=0.10,
            sector='Technology',
        )

        assert pos.symbol == 'AAPL'
        assert pos.weight == 0.3
        assert pos.return_value == 0.10
        assert pos.sector == 'Technology'

    def test_position_with_factor_exposures(self):
        """测试带因子暴露的持仓"""
        pos = PositionData(
            symbol='AAPL',
            weight=0.3,
            return_value=0.10,
            factor_exposures={'market': 1.2, 'size': -0.3},
        )

        assert pos.factor_exposures['market'] == 1.2
        assert pos.factor_exposures['size'] == -0.3


# ============ BrinsonAttributor Tests ============

class TestBrinsonAttributor:
    """Brinson归因测试"""

    def test_brinson_basic(self, simple_portfolio, simple_benchmark):
        """测试基本Brinson归因"""
        attributor = BrinsonAttributor()
        result = attributor.calculate(simple_portfolio, simple_benchmark)

        assert isinstance(result, BrinsonResult)
        assert result.allocation_effect is not None
        assert result.selection_effect is not None
        assert result.interaction_effect is not None

    def test_brinson_total_equals_sum(self, simple_portfolio, simple_benchmark):
        """测试总超额收益等于各效应之和"""
        attributor = BrinsonAttributor()
        result = attributor.calculate(simple_portfolio, simple_benchmark)

        total_from_components = (
            result.allocation_effect +
            result.selection_effect +
            result.interaction_effect
        )

        assert abs(result.total_active_return - total_from_components) < 1e-10

    def test_brinson_sector_breakdown(self, simple_portfolio, simple_benchmark):
        """测试行业分解"""
        attributor = BrinsonAttributor()
        result = attributor.calculate(simple_portfolio, simple_benchmark)

        # 应该有各行业的分解
        assert 'Technology' in result.sector_allocation
        assert 'Financials' in result.sector_allocation
        assert 'Energy' in result.sector_allocation

    def test_brinson_identical_portfolio(self):
        """测试组合与基准相同时"""
        portfolio = [PositionData('A', 0.5, 0.1, 'S1'), PositionData('B', 0.5, 0.05, 'S2')]
        benchmark = [BenchmarkData('A', 0.5, 0.1, 'S1'), BenchmarkData('B', 0.5, 0.05, 'S2')]

        attributor = BrinsonAttributor()
        result = attributor.calculate(portfolio, benchmark)

        # 完全相同时，所有效应应为0
        assert abs(result.allocation_effect) < 1e-10
        assert abs(result.selection_effect) < 1e-10
        assert abs(result.interaction_effect) < 1e-10

    def test_brinson_to_dict(self, simple_portfolio, simple_benchmark):
        """测试序列化"""
        attributor = BrinsonAttributor()
        result = attributor.calculate(simple_portfolio, simple_benchmark)

        d = result.to_dict()

        assert 'allocation_effect' in d
        assert 'selection_effect' in d
        assert 'interaction_effect' in d
        assert 'total_active_return' in d


# ============ FactorAttributor Tests ============

class TestFactorAttributor:
    """因子归因测试"""

    def test_factor_basic(self, factor_portfolio):
        """测试基本因子归因"""
        attributor = FactorAttributor()
        attributor.set_factor_returns({
            'market': 0.05,
            'size': 0.02,
            'value': 0.01,
            'momentum': 0.03,
        })

        portfolio_return = sum(p.weight * p.return_value for p in factor_portfolio)
        result = attributor.calculate(factor_portfolio, portfolio_return)

        assert isinstance(result, FactorResult)
        assert 'market' in result.factor_returns
        assert result.total_factor_return is not None

    def test_factor_specific_return(self, factor_portfolio):
        """测试特异性收益"""
        attributor = FactorAttributor()
        attributor.set_factor_returns({
            'market': 0.05,
            'size': 0.02,
            'value': 0.01,
            'momentum': 0.03,
        })

        portfolio_return = sum(p.weight * p.return_value for p in factor_portfolio)
        result = attributor.calculate(factor_portfolio, portfolio_return)

        # 总收益 = 因子收益 + 特异性收益
        total = result.total_factor_return + result.specific_return
        assert abs(total - portfolio_return) < 1e-10

    def test_factor_r_squared(self, factor_portfolio):
        """测试R方"""
        attributor = FactorAttributor()
        attributor.set_factor_returns({
            'market': 0.05,
            'size': 0.02,
            'value': 0.01,
            'momentum': 0.03,
        })

        portfolio_return = sum(p.weight * p.return_value for p in factor_portfolio)
        result = attributor.calculate(factor_portfolio, portfolio_return)

        assert 0 <= result.r_squared <= 1

    def test_factor_exposures_aggregation(self, factor_portfolio):
        """测试因子暴露聚合"""
        attributor = FactorAttributor()
        attributor.set_factor_returns({'market': 0.05})

        portfolio_return = sum(p.weight * p.return_value for p in factor_portfolio)
        result = attributor.calculate(factor_portfolio, portfolio_return)

        # 组合市场暴露应该是加权平均
        expected_market = (0.4 * 1.2 + 0.3 * 1.1 + 0.3 * 0.9)
        assert abs(result.factor_exposures['market'] - expected_market) < 1e-10


# ============ RiskAttributor Tests ============

class TestRiskAttributor:
    """风险归因测试"""

    def test_risk_basic(self, returns_matrix):
        """测试基本风险归因"""
        weights = {'A': 0.4, 'B': 0.35, 'C': 0.25}
        asset_names = ['A', 'B', 'C']

        attributor = RiskAttributor()
        result = attributor.calculate(weights, returns_matrix, asset_names)

        assert isinstance(result, RiskResult)
        assert result.total_var is not None
        assert result.total_volatility is not None

    def test_risk_var_contributions_sum_to_one(self, returns_matrix):
        """测试VaR贡献和为1"""
        weights = {'A': 0.4, 'B': 0.35, 'C': 0.25}
        asset_names = ['A', 'B', 'C']

        attributor = RiskAttributor()
        result = attributor.calculate(weights, returns_matrix, asset_names)

        total_contribution = sum(result.var_contribution.values())
        assert abs(total_contribution - 1.0) < 1e-10

    def test_risk_diversification_ratio(self, returns_matrix):
        """测试分散化比率"""
        weights = {'A': 0.4, 'B': 0.35, 'C': 0.25}
        asset_names = ['A', 'B', 'C']

        attributor = RiskAttributor()
        result = attributor.calculate(weights, returns_matrix, asset_names)

        # 分散化比率应该大于等于1
        assert result.diversification_ratio >= 1.0

    def test_risk_marginal_var(self, returns_matrix):
        """测试边际VaR"""
        weights = {'A': 0.4, 'B': 0.35, 'C': 0.25}
        asset_names = ['A', 'B', 'C']

        attributor = RiskAttributor()
        result = attributor.calculate(weights, returns_matrix, asset_names)

        # 应该有每个资产的边际VaR
        assert 'A' in result.marginal_var
        assert 'B' in result.marginal_var
        assert 'C' in result.marginal_var


# ============ TradingAttributor Tests ============

class TestTradingAttributor:
    """交易归因测试"""

    def test_trading_basic(self, sample_trades):
        """测试基本交易归因"""
        benchmark_returns = {'AAPL': 0.04, 'MSFT': 0.02, 'JPM': 0.01}

        attributor = TradingAttributor()
        result = attributor.calculate(sample_trades, benchmark_returns)

        assert isinstance(result, TradingResult)
        assert result.execution_cost is not None
        assert result.slippage_cost is not None

    def test_trading_alpha_relationship(self, sample_trades):
        """测试Alpha关系"""
        benchmark_returns = {'AAPL': 0.04, 'MSFT': 0.02, 'JPM': 0.01}

        attributor = TradingAttributor()
        result = attributor.calculate(sample_trades, benchmark_returns)

        # 净Alpha = 毛Alpha - 执行成本
        expected_net = result.gross_alpha - result.execution_cost
        assert abs(result.net_alpha - expected_net) < 1e-10

    def test_trading_empty_trades(self):
        """测试空交易列表"""
        attributor = TradingAttributor()
        result = attributor.calculate([], {})

        assert result.timing_effect == 0.0
        assert result.selection_effect == 0.0
        assert result.execution_cost == 0.0

    def test_trading_slippage_calculation(self):
        """测试滑点计算"""
        trades = [{
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'price': 101.0,  # 买入价格高于基准
            'benchmark_price': 100.0,
            'commission': 0,
        }]

        attributor = TradingAttributor()
        result = attributor.calculate(trades, {'AAPL': 0.0})

        # 买入时滑点为正（不利）
        assert result.slippage_cost > 0


# ============ PerformanceAttribution Tests ============

class TestPerformanceAttribution:
    """绩效归因主类测试"""

    @pytest.mark.asyncio
    async def test_analyze_brinson(self, simple_portfolio, simple_benchmark):
        """测试Brinson归因分析"""
        attribution = PerformanceAttribution()

        portfolio_return = sum(p.weight * p.return_value for p in simple_portfolio)
        benchmark_return = sum(b.weight * b.return_value for b in simple_benchmark)

        result = await attribution.analyze(
            portfolio=simple_portfolio,
            benchmark=simple_benchmark,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
        )

        assert isinstance(result, AttributionSummary)
        assert result.brinson is not None
        assert result.active_return == portfolio_return - benchmark_return

    @pytest.mark.asyncio
    async def test_analyze_factor(self, factor_portfolio, simple_benchmark):
        """测试因子归因分析"""
        config = AttributionConfig(method=AttributionMethod.FACTOR)
        attribution = PerformanceAttribution(config)

        portfolio_return = sum(p.weight * p.return_value for p in factor_portfolio)
        benchmark_return = sum(b.weight * b.return_value for b in simple_benchmark)

        factor_returns = {
            'market': 0.05,
            'size': 0.02,
            'value': 0.01,
            'momentum': 0.03,
        }

        result = await attribution.analyze(
            portfolio=factor_portfolio,
            benchmark=simple_benchmark,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            factor_returns=factor_returns,
        )

        assert isinstance(result, AttributionSummary)
        assert result.factor is not None

    @pytest.mark.asyncio
    async def test_analyze_risk(self, simple_portfolio, simple_benchmark, returns_matrix):
        """测试风险归因分析"""
        config = AttributionConfig(method=AttributionMethod.RISK)
        attribution = PerformanceAttribution(config)

        # 调整returns_matrix大小以匹配组合
        returns_matrix_4 = np.random.randn(60, 4) * 0.02

        portfolio_return = sum(p.weight * p.return_value for p in simple_portfolio)
        benchmark_return = sum(b.weight * b.return_value for b in simple_benchmark)

        result = await attribution.analyze(
            portfolio=simple_portfolio,
            benchmark=simple_benchmark,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            returns_matrix=returns_matrix_4,
        )

        assert isinstance(result, AttributionSummary)
        assert result.risk is not None

    @pytest.mark.asyncio
    async def test_analyze_trading(self, simple_portfolio, simple_benchmark, sample_trades):
        """测试交易归因分析"""
        config = AttributionConfig(method=AttributionMethod.TRADING)
        attribution = PerformanceAttribution(config)

        portfolio_return = sum(p.weight * p.return_value for p in simple_portfolio)
        benchmark_return = sum(b.weight * b.return_value for b in simple_benchmark)

        result = await attribution.analyze(
            portfolio=simple_portfolio,
            benchmark=simple_benchmark,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            trades=sample_trades,
        )

        assert isinstance(result, AttributionSummary)
        assert result.trading is not None

    def test_analyze_sync(self, simple_portfolio, simple_benchmark):
        """测试同步分析"""
        attribution = PerformanceAttribution()

        portfolio_return = sum(p.weight * p.return_value for p in simple_portfolio)
        benchmark_return = sum(b.weight * b.return_value for b in simple_benchmark)

        result = attribution.analyze_sync(
            portfolio=simple_portfolio,
            benchmark=simple_benchmark,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
        )

        assert isinstance(result, AttributionSummary)

    def test_history_tracking(self, simple_portfolio, simple_benchmark):
        """测试历史记录"""
        attribution = PerformanceAttribution()

        portfolio_return = sum(p.weight * p.return_value for p in simple_portfolio)
        benchmark_return = sum(b.weight * b.return_value for b in simple_benchmark)

        # 执行两次分析
        attribution.analyze_sync(
            portfolio=simple_portfolio,
            benchmark=simple_benchmark,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
        )
        attribution.analyze_sync(
            portfolio=simple_portfolio,
            benchmark=simple_benchmark,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
        )

        history = attribution.get_history()
        assert len(history) == 2

    def test_clear_history(self, simple_portfolio, simple_benchmark):
        """测试清除历史"""
        attribution = PerformanceAttribution()

        portfolio_return = sum(p.weight * p.return_value for p in simple_portfolio)
        benchmark_return = sum(b.weight * b.return_value for b in simple_benchmark)

        attribution.analyze_sync(
            portfolio=simple_portfolio,
            benchmark=simple_benchmark,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
        )

        attribution.clear_history()
        assert len(attribution.get_history()) == 0


# ============ Multi-Period Tests ============

class TestMultiPeriod:
    """多期间测试"""

    @pytest.mark.asyncio
    async def test_analyze_multi_period(self, simple_portfolio, simple_benchmark):
        """测试多期间分析"""
        attribution = PerformanceAttribution()

        portfolio_return = sum(p.weight * p.return_value for p in simple_portfolio)
        benchmark_return = sum(b.weight * b.return_value for b in simple_benchmark)

        periods = [
            {
                'portfolio': simple_portfolio,
                'benchmark': simple_benchmark,
                'portfolio_return': portfolio_return * 0.8,
                'benchmark_return': benchmark_return * 0.9,
            },
            {
                'portfolio': simple_portfolio,
                'benchmark': simple_benchmark,
                'portfolio_return': portfolio_return * 1.2,
                'benchmark_return': benchmark_return * 1.1,
            },
        ]

        results = await attribution.analyze_multi_period(periods)

        assert len(results) == 2

    def test_link_periods_geometric(self, simple_portfolio, simple_benchmark):
        """测试几何链接"""
        attribution = PerformanceAttribution()

        summaries = [
            AttributionSummary(
                portfolio_return=0.10,
                benchmark_return=0.08,
                active_return=0.02,
                brinson=BrinsonResult(0.01, 0.005, 0.005, 0.02),
            ),
            AttributionSummary(
                portfolio_return=0.05,
                benchmark_return=0.04,
                active_return=0.01,
                brinson=BrinsonResult(0.005, 0.003, 0.002, 0.01),
            ),
        ]

        linked = attribution.link_periods(summaries, method='geometric')

        # 几何链接: (1.10 * 1.05) - 1 = 0.155
        expected_portfolio = (1 + 0.10) * (1 + 0.05) - 1
        assert abs(linked.portfolio_return - expected_portfolio) < 1e-10

    def test_link_periods_arithmetic(self, simple_portfolio, simple_benchmark):
        """测试算术链接"""
        attribution = PerformanceAttribution()

        summaries = [
            AttributionSummary(
                portfolio_return=0.10,
                benchmark_return=0.08,
                active_return=0.02,
            ),
            AttributionSummary(
                portfolio_return=0.05,
                benchmark_return=0.04,
                active_return=0.01,
            ),
        ]

        linked = attribution.link_periods(summaries, method='arithmetic')

        # 算术链接: 0.10 + 0.05 = 0.15
        assert abs(linked.portfolio_return - 0.15) < 1e-10


# ============ Factory Functions Tests ============

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_attribution(self):
        """测试创建归因分析器"""
        attribution = create_attribution(method='brinson', period='daily')

        assert attribution.config.method == AttributionMethod.BRINSON
        assert attribution.config.period == AttributionPeriod.DAILY

    def test_create_attribution_factor(self):
        """测试创建因子归因分析器"""
        attribution = create_attribution(method='factor', period='monthly')

        assert attribution.config.method == AttributionMethod.FACTOR
        assert attribution.config.period == AttributionPeriod.MONTHLY

    def test_create_position(self):
        """测试创建持仓"""
        pos = create_position(
            symbol='AAPL',
            weight=0.3,
            return_value=0.10,
            sector='Technology',
            market=1.2,
            size=-0.3,
        )

        assert pos.symbol == 'AAPL'
        assert pos.weight == 0.3
        assert pos.factor_exposures['market'] == 1.2
        assert pos.factor_exposures['size'] == -0.3

    def test_create_benchmark(self):
        """测试创建基准"""
        bench = create_benchmark(
            symbol='SPY',
            weight=1.0,
            return_value=0.05,
            sector='Index',
        )

        assert bench.symbol == 'SPY'
        assert bench.weight == 1.0

    @pytest.mark.asyncio
    async def test_quick_attribution(self):
        """测试快速归因"""
        portfolio = [
            {'symbol': 'AAPL', 'weight': 0.5, 'return': 0.10, 'sector': 'Tech'},
            {'symbol': 'JPM', 'weight': 0.5, 'return': 0.05, 'sector': 'Fin'},
        ]

        benchmark = [
            {'symbol': 'TECH', 'weight': 0.5, 'return': 0.08, 'sector': 'Tech'},
            {'symbol': 'FIN', 'weight': 0.5, 'return': 0.04, 'sector': 'Fin'},
        ]

        result = await quick_attribution(portfolio, benchmark, method='brinson')

        assert isinstance(result, AttributionSummary)
        assert result.brinson is not None


# ============ Edge Cases Tests ============

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_portfolio(self):
        """测试空组合"""
        attributor = BrinsonAttributor()
        result = attributor.calculate([], [])

        assert result.allocation_effect == 0.0
        assert result.selection_effect == 0.0

    def test_single_position(self):
        """测试单一持仓"""
        portfolio = [PositionData('AAPL', 1.0, 0.10, 'Tech')]
        benchmark = [BenchmarkData('TECH', 1.0, 0.08, 'Tech')]

        attributor = BrinsonAttributor()
        result = attributor.calculate(portfolio, benchmark)

        # 单一持仓相同行业，配置效应应为0
        assert abs(result.allocation_effect) < 1e-10

    def test_missing_sector(self):
        """测试缺失行业"""
        portfolio = [PositionData('AAPL', 0.5, 0.10), PositionData('MSFT', 0.5, 0.08)]
        benchmark = [BenchmarkData('TECH', 1.0, 0.08)]

        attributor = BrinsonAttributor()
        result = attributor.calculate(portfolio, benchmark)

        # 应该归类到'Other'
        assert 'Other' in result.sector_allocation or 'TECH' in result.sector_allocation

    def test_zero_weight(self):
        """测试零权重"""
        portfolio = [
            PositionData('AAPL', 0.0, 0.10, 'Tech'),
            PositionData('MSFT', 1.0, 0.08, 'Tech'),
        ]
        benchmark = [BenchmarkData('TECH', 1.0, 0.08, 'Tech')]

        attributor = BrinsonAttributor()
        result = attributor.calculate(portfolio, benchmark)

        # 零权重不应影响结果
        assert result.allocation_effect is not None

    def test_negative_returns(self):
        """测试负收益"""
        portfolio = [PositionData('AAPL', 0.5, -0.10, 'Tech'), PositionData('JPM', 0.5, -0.05, 'Fin')]
        benchmark = [BenchmarkData('TECH', 0.5, -0.08, 'Tech'), BenchmarkData('FIN', 0.5, -0.04, 'Fin')]

        attributor = BrinsonAttributor()
        result = attributor.calculate(portfolio, benchmark)

        assert result.total_active_return is not None


# ============ Serialization Tests ============

class TestSerialization:
    """序列化测试"""

    def test_attribution_summary_to_dict(self):
        """测试归因摘要序列化"""
        summary = AttributionSummary(
            portfolio_return=0.10,
            benchmark_return=0.08,
            active_return=0.02,
            brinson=BrinsonResult(0.01, 0.005, 0.005, 0.02),
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
        )

        d = summary.to_dict()

        assert d['portfolio_return'] == 0.10
        assert d['benchmark_return'] == 0.08
        assert d['active_return'] == 0.02
        assert d['brinson'] is not None
        assert 'allocation_effect' in d['brinson']

    def test_risk_result_to_dict(self, returns_matrix):
        """测试风险结果序列化"""
        weights = {'A': 0.4, 'B': 0.35, 'C': 0.25}
        asset_names = ['A', 'B', 'C']

        attributor = RiskAttributor()
        result = attributor.calculate(weights, returns_matrix, asset_names)

        d = result.to_dict()

        assert 'var_contribution' in d
        assert 'total_var' in d
        assert 'diversification_ratio' in d


# ============ Run Tests ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
