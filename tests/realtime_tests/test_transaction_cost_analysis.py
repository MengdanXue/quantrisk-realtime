"""
交易成本分析模块测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.transaction_cost_analysis import (
    # 主类
    TransactionCostAnalyzer,
    ExecutionQualityReporter,
    # 数据类
    TradeRecord,
    TCAResult,
    PortfolioTCAResult,
    MarketImpactEstimate,
    TCAConfig,
    # 枚举
    ExecutionBenchmark,
    CostComponent,
    ImpactModel,
    TradeDirection,
    # 模型
    LinearImpactModel,
    SquareRootImpactModel,
    PowerLawImpactModel,
    AlmgrenChrissModel,
    # 工厂函数
    create_tca_analyzer,
    create_trade_record,
    quick_tca,
    estimate_impact,
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_buy_trade() -> TradeRecord:
    """样本买入交易"""
    return TradeRecord(
        trade_id="T001",
        symbol="AAPL",
        direction=TradeDirection.BUY,
        quantity=1000,
        executed_price=150.50,
        executed_time=datetime(2024, 1, 15, 10, 30, 0),
        arrival_price=150.00,
        decision_time=datetime(2024, 1, 15, 10, 25, 0),
        vwap=150.25,
        twap=150.20,
        close_price=151.00,
        open_price=149.50,
        previous_close=149.00,
        bid_price=150.40,
        ask_price=150.60,
        commission=10.0,
        daily_volume=5000000,
        volatility=0.25,
        average_spread=0.02,
        strategy="momentum",
        portfolio_id="P001",
    )


@pytest.fixture
def sample_sell_trade() -> TradeRecord:
    """样本卖出交易"""
    return TradeRecord(
        trade_id="T002",
        symbol="GOOGL",
        direction=TradeDirection.SELL,
        quantity=500,
        executed_price=140.20,
        executed_time=datetime(2024, 1, 15, 14, 30, 0),
        arrival_price=140.50,
        decision_time=datetime(2024, 1, 15, 14, 25, 0),
        vwap=140.40,
        twap=140.35,
        close_price=140.00,
        open_price=141.00,
        previous_close=141.50,
        bid_price=140.15,
        ask_price=140.25,
        commission=8.0,
        daily_volume=3000000,
        volatility=0.30,
        average_spread=0.01,
        strategy="mean_reversion",
        portfolio_id="P001",
    )


@pytest.fixture
def multiple_trades(sample_buy_trade, sample_sell_trade) -> List[TradeRecord]:
    """多笔交易"""
    trades = [sample_buy_trade, sample_sell_trade]

    # 添加更多交易
    for i in range(3, 11):
        trades.append(TradeRecord(
            trade_id=f"T{i:03d}",
            symbol="MSFT" if i % 2 == 0 else "AMZN",
            direction=TradeDirection.BUY if i % 3 == 0 else TradeDirection.SELL,
            quantity=100 * i,
            executed_price=300.0 + i * 5,
            executed_time=datetime(2024, 1, 15, 9 + i, 0, 0),
            arrival_price=300.0 + i * 5 - 0.5,
            vwap=300.0 + i * 5 - 0.3,
            twap=300.0 + i * 5 - 0.2,
            close_price=300.0 + i * 5 + 1,
            open_price=300.0 + i * 5 - 1,
            previous_close=299.0 + i * 5,
            commission=5.0 + i,
            daily_volume=1000000 * i,
            volatility=0.20 + 0.01 * i,
            strategy="momentum" if i % 2 == 0 else "mean_reversion",
            portfolio_id="P001",
        ))

    return trades


@pytest.fixture
def tca_analyzer() -> TransactionCostAnalyzer:
    """TCA分析器"""
    return TransactionCostAnalyzer()


@pytest.fixture
def custom_config() -> TCAConfig:
    """自定义配置"""
    return TCAConfig(
        default_benchmark=ExecutionBenchmark.VWAP,
        impact_model=ImpactModel.SQUARE_ROOT,
        impact_coefficient=0.15,
        default_commission_bps=3.0,
        default_spread_bps=1.5,
    )


# ==================== TradeRecord 测试 ====================

class TestTradeRecord:
    """交易记录测试"""

    def test_create_trade_record(self):
        """测试创建交易记录"""
        trade = TradeRecord(
            trade_id="T001",
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=100,
            executed_price=150.0,
            executed_time=datetime.now(),
        )

        assert trade.trade_id == "T001"
        assert trade.symbol == "AAPL"
        assert trade.direction == TradeDirection.BUY
        assert trade.quantity == 100
        assert trade.executed_price == 150.0

    def test_spread_cost_buy(self, sample_buy_trade):
        """测试买入价差成本"""
        # 买入时，执行价高于中间价为正成本
        spread_cost = sample_buy_trade.spread_cost
        mid_price = (150.40 + 150.60) / 2  # 150.50
        expected = (150.50 - mid_price) / mid_price
        assert abs(spread_cost - expected) < 0.0001

    def test_spread_cost_sell(self, sample_sell_trade):
        """测试卖出价差成本"""
        spread_cost = sample_sell_trade.spread_cost
        mid_price = (140.15 + 140.25) / 2  # 140.20
        expected = (mid_price - 140.20) / mid_price
        assert abs(spread_cost - expected) < 0.0001

    def test_participation_rate(self, sample_buy_trade):
        """测试参与率"""
        rate = sample_buy_trade.participation_rate
        expected = 1000 / 5000000
        assert abs(rate - expected) < 0.0001

    def test_to_dict(self, sample_buy_trade):
        """测试转换为字典"""
        d = sample_buy_trade.to_dict()

        assert d['trade_id'] == "T001"
        assert d['symbol'] == "AAPL"
        assert d['direction'] == "buy"
        assert d['quantity'] == 1000
        assert d['executed_price'] == 150.50

    def test_factory_function(self):
        """测试工厂函数"""
        trade = create_trade_record(
            trade_id="T999",
            symbol="TEST",
            direction="buy",
            quantity=500,
            executed_price=100.0,
            arrival_price=99.5,
            vwap=99.8,
        )

        assert trade.trade_id == "T999"
        assert trade.direction == TradeDirection.BUY
        assert trade.arrival_price == 99.5


# ==================== 市场影响模型测试 ====================

class TestMarketImpactModels:
    """市场影响模型测试"""

    def test_linear_impact(self):
        """测试线性影响模型"""
        model = LinearImpactModel(coefficient=0.1)

        impact = model.estimate_impact(
            quantity=10000,
            daily_volume=1000000,
            volatility=0.25,
            direction=TradeDirection.BUY,
        )

        assert impact > 0
        # 参与率1%, 日波动率约1.58%, 影响约0.158bps * 10000
        assert 0 < impact < 100

    def test_square_root_impact(self):
        """测试平方根影响模型"""
        model = SquareRootImpactModel(coefficient=0.1, exponent=0.5)

        impact = model.estimate_impact(
            quantity=10000,
            daily_volume=1000000,
            volatility=0.25,
            direction=TradeDirection.BUY,
        )

        assert impact > 0

    def test_power_law_impact(self):
        """测试幂律影响模型"""
        model = PowerLawImpactModel(coefficient=0.1)

        impact = model.estimate_impact(
            quantity=10000,
            daily_volume=1000000,
            volatility=0.25,
            direction=TradeDirection.BUY,
        )

        assert impact > 0

    def test_almgren_chriss_impact(self):
        """测试Almgren-Chriss模型"""
        model = AlmgrenChrissModel()

        impact = model.estimate_impact(
            quantity=10000,
            daily_volume=1000000,
            volatility=0.25,
            direction=TradeDirection.BUY,
        )

        assert impact > 0

    def test_impact_scales_with_quantity(self):
        """测试影响随数量增加"""
        model = SquareRootImpactModel()

        impact_small = model.estimate_impact(
            quantity=1000,
            daily_volume=1000000,
            volatility=0.25,
            direction=TradeDirection.BUY,
        )

        impact_large = model.estimate_impact(
            quantity=10000,
            daily_volume=1000000,
            volatility=0.25,
            direction=TradeDirection.BUY,
        )

        assert impact_large > impact_small

    def test_impact_scales_with_volatility(self):
        """测试影响随波动率增加"""
        model = SquareRootImpactModel()

        impact_low_vol = model.estimate_impact(
            quantity=10000,
            daily_volume=1000000,
            volatility=0.15,
            direction=TradeDirection.BUY,
        )

        impact_high_vol = model.estimate_impact(
            quantity=10000,
            daily_volume=1000000,
            volatility=0.35,
            direction=TradeDirection.BUY,
        )

        assert impact_high_vol > impact_low_vol

    def test_zero_volume_returns_zero(self):
        """测试零成交量返回零"""
        model = SquareRootImpactModel()

        impact = model.estimate_impact(
            quantity=10000,
            daily_volume=0,
            volatility=0.25,
            direction=TradeDirection.BUY,
        )

        assert impact == 0


# ==================== TCA分析器测试 ====================

class TestTransactionCostAnalyzer:
    """TCA分析器测试"""

    @pytest.mark.asyncio
    async def test_analyze_buy_trade(self, tca_analyzer, sample_buy_trade):
        """测试分析买入交易"""
        result = await tca_analyzer.analyze_trade(sample_buy_trade)

        assert result.trade_id == "T001"
        assert result.symbol == "AAPL"
        assert result.direction == TradeDirection.BUY
        assert result.quantity == 1000
        assert result.notional_value == 1000 * 150.50

    @pytest.mark.asyncio
    async def test_analyze_sell_trade(self, tca_analyzer, sample_sell_trade):
        """测试分析卖出交易"""
        result = await tca_analyzer.analyze_trade(sample_sell_trade)

        assert result.trade_id == "T002"
        assert result.direction == TradeDirection.SELL
        assert result.execution_score >= 0
        assert result.execution_score <= 100

    @pytest.mark.asyncio
    async def test_implementation_shortfall_buy(self, tca_analyzer, sample_buy_trade):
        """测试买入执行差价"""
        result = await tca_analyzer.analyze_trade(
            sample_buy_trade,
            benchmark=ExecutionBenchmark.ARRIVAL_PRICE
        )

        # 买入: executed (150.50) > arrival (150.00) = 成本
        # IS = (150.50 - 150.00) / 150.00 * 10000 = 33.33 bps
        expected_is = (150.50 - 150.00) / 150.00 * 10000
        assert abs(result.implementation_shortfall - expected_is) < 0.01

    @pytest.mark.asyncio
    async def test_implementation_shortfall_sell(self, tca_analyzer, sample_sell_trade):
        """测试卖出执行差价"""
        result = await tca_analyzer.analyze_trade(
            sample_sell_trade,
            benchmark=ExecutionBenchmark.ARRIVAL_PRICE
        )

        # 卖出: executed (140.20) < arrival (140.50) = 成本
        # IS = -(140.20 - 140.50) / 140.50 * 10000 = 21.35 bps
        expected_is = -(140.20 - 140.50) / 140.50 * 10000
        assert abs(result.implementation_shortfall - expected_is) < 0.01

    @pytest.mark.asyncio
    async def test_benchmark_comparison(self, tca_analyzer, sample_buy_trade):
        """测试基准比较"""
        comparison = tca_analyzer.compare_benchmarks(sample_buy_trade)

        assert 'arrival_price' in comparison
        assert 'vwap' in comparison
        assert 'twap' in comparison
        assert 'close_price' in comparison

    @pytest.mark.asyncio
    async def test_analyze_multiple_trades(self, tca_analyzer, multiple_trades):
        """测试批量分析"""
        results = await tca_analyzer.analyze_trades(multiple_trades)

        assert len(results) == len(multiple_trades)
        for result in results:
            assert isinstance(result, TCAResult)

    @pytest.mark.asyncio
    async def test_analyze_portfolio(self, tca_analyzer, multiple_trades):
        """测试组合分析"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        result = await tca_analyzer.analyze_portfolio(
            portfolio_id="P001",
            start_date=start_date,
            end_date=end_date,
            trades=multiple_trades,
        )

        assert isinstance(result, PortfolioTCAResult)
        assert result.total_trades == len(multiple_trades)
        assert result.total_notional > 0
        assert len(result.by_symbol) > 0
        assert len(result.by_direction) > 0

    @pytest.mark.asyncio
    async def test_portfolio_cost_breakdown(self, tca_analyzer, multiple_trades):
        """测试组合成本分解"""
        result = await tca_analyzer.analyze_portfolio(
            portfolio_id="P001",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            trades=multiple_trades,
        )

        # 验证成本组成
        assert 'AAPL' in result.by_symbol or 'MSFT' in result.by_symbol
        assert 'buy' in result.by_direction or 'sell' in result.by_direction

    def test_estimate_market_impact(self, tca_analyzer):
        """测试市场影响估计"""
        estimate = tca_analyzer.estimate_market_impact(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=50000,
            daily_volume=10000000,
            volatility=0.25,
        )

        assert isinstance(estimate, MarketImpactEstimate)
        assert estimate.estimated_impact_bps > 0
        assert 0 <= estimate.impact_confidence <= 1
        assert estimate.impact_low < estimate.estimated_impact_bps < estimate.impact_high

    def test_add_and_clear_history(self, tca_analyzer, sample_buy_trade):
        """测试添加和清除历史"""
        tca_analyzer.add_trade(sample_buy_trade)
        assert len(tca_analyzer._trade_history) == 1

        tca_analyzer.clear_history()
        assert len(tca_analyzer._trade_history) == 0

    @pytest.mark.asyncio
    async def test_cost_breakdown(self, tca_analyzer, sample_buy_trade):
        """测试成本分解"""
        await tca_analyzer.analyze_trade(sample_buy_trade)
        breakdown = tca_analyzer.get_cost_breakdown(sample_buy_trade.trade_id)

        assert breakdown is not None
        assert 'implementation_shortfall' in breakdown
        assert 'market_impact' in breakdown
        assert 'timing_cost' in breakdown
        assert 'spread_cost' in breakdown
        assert 'commission_cost' in breakdown
        assert 'total_cost' in breakdown


# ==================== 配置测试 ====================

class TestTCAConfig:
    """配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = TCAConfig()

        assert config.default_benchmark == ExecutionBenchmark.ARRIVAL_PRICE
        assert config.impact_model == ImpactModel.SQUARE_ROOT
        assert config.default_commission_bps == 5.0

    def test_custom_config(self, custom_config):
        """测试自定义配置"""
        assert custom_config.default_benchmark == ExecutionBenchmark.VWAP
        assert custom_config.impact_coefficient == 0.15

    @pytest.mark.asyncio
    async def test_analyzer_with_custom_config(self, custom_config, sample_buy_trade):
        """测试使用自定义配置的分析器"""
        analyzer = TransactionCostAnalyzer(custom_config)
        result = await analyzer.analyze_trade(sample_buy_trade)

        # 应该使用VWAP作为基准
        assert result.benchmark == ExecutionBenchmark.VWAP


# ==================== 执行质量报告测试 ====================

class TestExecutionQualityReporter:
    """执行质量报告测试"""

    @pytest.mark.asyncio
    async def test_summary_report(self, tca_analyzer, multiple_trades):
        """测试摘要报告"""
        portfolio_result = await tca_analyzer.analyze_portfolio(
            portfolio_id="P001",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            trades=multiple_trades,
        )

        reporter = ExecutionQualityReporter(tca_analyzer)
        report = reporter.generate_summary_report(portfolio_result)

        assert report['report_type'] == 'execution_quality_summary'
        assert 'overview' in report
        assert 'cost_breakdown' in report
        assert 'execution_quality' in report
        assert 'analysis_by_dimension' in report

    @pytest.mark.asyncio
    async def test_trade_detail_report(self, tca_analyzer, multiple_trades):
        """测试交易明细报告"""
        results = await tca_analyzer.analyze_trades(multiple_trades)

        reporter = ExecutionQualityReporter(tca_analyzer)
        detail_report = reporter.generate_trade_detail_report(results)

        assert len(detail_report) == len(multiple_trades)
        for item in detail_report:
            assert 'trade_id' in item
            assert 'implementation_shortfall' in item
            assert 'execution_score' in item

    def test_benchmark_comparison_report(self, tca_analyzer, multiple_trades):
        """测试基准比较报告"""
        reporter = ExecutionQualityReporter(tca_analyzer)
        comparison = reporter.generate_benchmark_comparison(multiple_trades)

        assert 'by_trade' in comparison
        assert 'averages' in comparison


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_tca_analyzer(self):
        """测试创建TCA分析器"""
        analyzer = create_tca_analyzer(
            default_benchmark='vwap',
            impact_model='linear',
        )

        assert analyzer.config.default_benchmark == ExecutionBenchmark.VWAP
        assert analyzer.config.impact_model == ImpactModel.LINEAR

    @pytest.mark.asyncio
    async def test_quick_tca(self, sample_buy_trade):
        """测试快速TCA"""
        result = await quick_tca(sample_buy_trade, benchmark='vwap')

        assert isinstance(result, TCAResult)
        assert result.benchmark == ExecutionBenchmark.VWAP

    def test_estimate_impact_function(self):
        """测试市场影响估计函数"""
        estimate = estimate_impact(
            symbol="AAPL",
            direction="buy",
            quantity=10000,
            daily_volume=5000000,
            volatility=0.25,
            model='square_root',
        )

        assert isinstance(estimate, MarketImpactEstimate)
        assert estimate.model_used == ImpactModel.SQUARE_ROOT


# ==================== 边界条件测试 ====================

class TestEdgeCases:
    """边界条件测试"""

    @pytest.mark.asyncio
    async def test_zero_arrival_price(self, tca_analyzer):
        """测试零到达价格"""
        trade = TradeRecord(
            trade_id="T_EDGE_1",
            symbol="TEST",
            direction=TradeDirection.BUY,
            quantity=100,
            executed_price=100.0,
            executed_time=datetime.now(),
            arrival_price=0,  # 零到达价格
        )

        result = await tca_analyzer.analyze_trade(trade)
        # 应该回退到执行价格
        assert result.benchmark_price == trade.executed_price

    @pytest.mark.asyncio
    async def test_zero_volume(self, tca_analyzer):
        """测试零成交量"""
        trade = TradeRecord(
            trade_id="T_EDGE_2",
            symbol="TEST",
            direction=TradeDirection.BUY,
            quantity=100,
            executed_price=100.0,
            executed_time=datetime.now(),
            arrival_price=99.5,
            daily_volume=0,
            volatility=0.25,
        )

        result = await tca_analyzer.analyze_trade(trade)
        assert result.market_impact == 0

    @pytest.mark.asyncio
    async def test_empty_portfolio(self, tca_analyzer):
        """测试空组合"""
        result = await tca_analyzer.analyze_portfolio(
            portfolio_id="EMPTY",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            trades=[],
        )

        assert result.total_trades == 0
        assert result.total_notional == 0

    @pytest.mark.asyncio
    async def test_large_quantity(self, tca_analyzer):
        """测试大数量订单"""
        trade = TradeRecord(
            trade_id="T_LARGE",
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=1000000,  # 100万股
            executed_price=150.0,
            executed_time=datetime.now(),
            arrival_price=149.0,
            daily_volume=5000000,  # 20%参与率
            volatility=0.25,
        )

        result = await tca_analyzer.analyze_trade(trade)
        # 高参与率应该导致较大市场影响
        assert result.market_impact > 0

    @pytest.mark.asyncio
    async def test_perfect_execution(self, tca_analyzer):
        """测试完美执行"""
        trade = TradeRecord(
            trade_id="T_PERFECT",
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=100,
            executed_price=100.0,
            executed_time=datetime.now(),
            arrival_price=100.0,  # 等于执行价
            vwap=100.0,
            twap=100.0,
            commission=0,
        )

        result = await tca_analyzer.analyze_trade(trade)
        # 完美执行的IS应该接近0
        assert abs(result.implementation_shortfall) < 0.01


# ==================== 数据类序列化测试 ====================

class TestSerialization:
    """序列化测试"""

    @pytest.mark.asyncio
    async def test_tca_result_to_dict(self, tca_analyzer, sample_buy_trade):
        """测试TCA结果序列化"""
        result = await tca_analyzer.analyze_trade(sample_buy_trade)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d['trade_id'] == "T001"
        assert isinstance(d['implementation_shortfall'], float)
        assert isinstance(d['analyzed_at'], str)

    @pytest.mark.asyncio
    async def test_portfolio_result_to_dict(self, tca_analyzer, multiple_trades):
        """测试组合结果序列化"""
        result = await tca_analyzer.analyze_portfolio(
            portfolio_id="P001",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            trades=multiple_trades,
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d['portfolio_id'] == "P001"
        assert isinstance(d['total_trades'], int)
        assert isinstance(d['by_symbol'], dict)

    def test_market_impact_estimate_to_dict(self, tca_analyzer):
        """测试市场影响估计序列化"""
        estimate = tca_analyzer.estimate_market_impact(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=10000,
            daily_volume=5000000,
            volatility=0.25,
        )

        d = estimate.to_dict()

        assert isinstance(d, dict)
        assert d['symbol'] == "AAPL"
        assert d['direction'] == "buy"
        assert isinstance(d['estimated_impact_bps'], float)


# ==================== 执行得分测试 ====================

class TestExecutionScore:
    """执行得分测试"""

    @pytest.mark.asyncio
    async def test_score_range(self, tca_analyzer, multiple_trades):
        """测试得分范围"""
        results = await tca_analyzer.analyze_trades(multiple_trades)

        for result in results:
            assert 0 <= result.execution_score <= 100

    @pytest.mark.asyncio
    async def test_good_execution_high_score(self, tca_analyzer):
        """测试良好执行应有高分"""
        trade = TradeRecord(
            trade_id="T_GOOD",
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=100,
            executed_price=100.0,
            executed_time=datetime.now(),
            arrival_price=100.05,  # 略高于执行价(好)
            vwap=100.02,          # 略高于执行价(好)
            daily_volume=10000000,
            volatility=0.20,
        )

        result = await tca_analyzer.analyze_trade(trade)
        assert result.execution_score >= 80

    @pytest.mark.asyncio
    async def test_poor_execution_low_score(self, tca_analyzer):
        """测试差执行应有低分"""
        trade = TradeRecord(
            trade_id="T_POOR",
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=500000,  # 高参与率
            executed_price=105.0,
            executed_time=datetime.now(),
            arrival_price=100.0,  # 远低于执行价(差)
            vwap=101.0,           # 低于执行价(差)
            daily_volume=1000000,
            volatility=0.20,
        )

        result = await tca_analyzer.analyze_trade(trade)
        assert result.execution_score < 80


# ==================== 分析维度测试 ====================

class TestAnalysisDimensions:
    """分析维度测试"""

    @pytest.mark.asyncio
    async def test_by_symbol_analysis(self, tca_analyzer, multiple_trades):
        """测试按品种分析"""
        result = await tca_analyzer.analyze_portfolio(
            portfolio_id="P001",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            trades=multiple_trades,
        )

        assert len(result.by_symbol) > 0
        for symbol, stats in result.by_symbol.items():
            assert 'trades' in stats
            assert 'notional' in stats
            assert 'avg_cost_bps' in stats
            assert 'avg_score' in stats

    @pytest.mark.asyncio
    async def test_by_direction_analysis(self, tca_analyzer, multiple_trades):
        """测试按方向分析"""
        result = await tca_analyzer.analyze_portfolio(
            portfolio_id="P001",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            trades=multiple_trades,
        )

        assert 'buy' in result.by_direction or 'sell' in result.by_direction

    @pytest.mark.asyncio
    async def test_by_strategy_analysis(self, tca_analyzer, multiple_trades):
        """测试按策略分析"""
        result = await tca_analyzer.analyze_portfolio(
            portfolio_id="P001",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            trades=multiple_trades,
        )

        assert len(result.by_strategy) > 0
        assert 'momentum' in result.by_strategy or 'mean_reversion' in result.by_strategy

    @pytest.mark.asyncio
    async def test_by_time_analysis(self, tca_analyzer, multiple_trades):
        """测试按时间分析"""
        result = await tca_analyzer.analyze_portfolio(
            portfolio_id="P001",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            trades=multiple_trades,
        )

        # 应该有时间桶分析
        assert isinstance(result.by_time_of_day, dict)
