"""
Market Impact Estimator 测试套件
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from quantrisk_realtime.realtime.market_impact_estimator import (
    # 主类
    MarketImpactEstimator,

    # 数据类
    ImpactParameters,
    ImpactEstimate,
    ImpactDecomposition,
    ExecutionPath,
    ExecutionSlice,
    RealTimeImpact,
    ScenarioAnalysis,
    EstimatorConfig,

    # 枚举
    ImpactModelType,
    ImpactType,
    ExecutionStrategy,
    TradeDirection,
    MarketCondition,

    # 模型
    LinearImpactModel,
    SquareRootImpactModel,
    AlmgrenChrissImpactModel,
    KyleLambdaImpactModel,

    # 工厂函数
    create_impact_estimator,
    create_impact_parameters,
    quick_impact_estimate,
)


# ==================== Fixtures ====================

@pytest.fixture
def estimator():
    """创建默认估算器"""
    return create_impact_estimator()


@pytest.fixture
def parameters():
    """创建测试参数"""
    return create_impact_parameters(
        symbol="AAPL",
        price=150.0,
        daily_volume=50_000_000,
        volatility=0.25,
        spread_bps=5.0,
    )


@pytest.fixture
def high_volume_parameters():
    """高成交量参数"""
    return create_impact_parameters(
        symbol="SPY",
        price=400.0,
        daily_volume=100_000_000,
        volatility=0.15,
        spread_bps=2.0,
    )


@pytest.fixture
def low_liquidity_parameters():
    """低流动性参数"""
    return create_impact_parameters(
        symbol="SMALLCAP",
        price=50.0,
        daily_volume=1_000_000,
        volatility=0.40,
        spread_bps=20.0,
        market_condition=MarketCondition.ILLIQUID,
    )


# ==================== 基础功能测试 ====================

class TestBasicFunctionality:
    """基础功能测试"""

    @pytest.mark.asyncio
    async def test_estimator_initialization(self):
        """测试估算器初始化"""
        estimator = create_impact_estimator()
        assert estimator is not None
        assert estimator.config is not None
        assert estimator.config.default_model == ImpactModelType.SQUARE_ROOT

    @pytest.mark.asyncio
    async def test_custom_config(self):
        """测试自定义配置"""
        config = EstimatorConfig(
            default_model=ImpactModelType.ALMGREN_CHRISS,
            auto_calibrate=False,
            max_participation_rate=0.15,
        )
        estimator = MarketImpactEstimator(config=config)

        assert estimator.config.default_model == ImpactModelType.ALMGREN_CHRISS
        assert estimator.config.auto_calibrate is False
        assert estimator.config.max_participation_rate == 0.15

    def test_impact_parameters_creation(self):
        """测试参数创建"""
        params = create_impact_parameters(
            symbol="TSLA",
            price=200.0,
            daily_volume=80_000_000,
            volatility=0.50,
            spread_bps=8.0,
        )

        assert params.symbol == "TSLA"
        assert params.price == 200.0
        assert params.daily_volume == 80_000_000
        assert params.volatility == 0.50
        assert params.spread_bps == 8.0

    def test_parameters_to_dict(self, parameters):
        """测试参数字典转换"""
        param_dict = parameters.to_dict()

        assert 'symbol' in param_dict
        assert 'price' in param_dict
        assert 'daily_volume' in param_dict
        assert 'volatility' in param_dict
        assert param_dict['symbol'] == "AAPL"


# ==================== 冲击模型测试 ====================

class TestImpactModels:
    """冲击模型测试"""

    def test_linear_model(self, parameters):
        """测试线性模型"""
        model = LinearImpactModel()
        quantity = 100_000

        total, temp, perm = model.estimate_impact(
            quantity, parameters, TradeDirection.BUY
        )

        assert total > 0
        assert temp > 0
        assert perm > 0
        assert abs(total - (temp + perm)) < 0.01  # 总和一致

    def test_square_root_model(self, parameters):
        """测试平方根模型"""
        model = SquareRootImpactModel()
        quantity = 100_000

        total, temp, perm = model.estimate_impact(
            quantity, parameters, TradeDirection.BUY
        )

        assert total > 0
        assert temp > perm  # 临时冲击应该更大

    def test_almgren_chriss_model(self, parameters):
        """测试Almgren-Chriss模型"""
        model = AlmgrenChrissImpactModel()
        quantity = 100_000

        total, temp, perm = model.estimate_impact(
            quantity, parameters, TradeDirection.BUY
        )

        assert total > 0
        assert temp >= 0
        assert perm >= 0

    def test_kyle_lambda_model(self, parameters):
        """测试Kyle Lambda模型"""
        model = KyleLambdaImpactModel()
        quantity = 100_000

        total, temp, perm = model.estimate_impact(
            quantity, parameters, TradeDirection.BUY
        )

        assert total > 0
        assert perm > temp  # Kyle模型主要是永久冲击

    def test_impact_scaling_with_quantity(self, parameters):
        """测试冲击随数量的变化"""
        model = SquareRootImpactModel()

        # 小订单
        total_small, _, _ = model.estimate_impact(
            10_000, parameters, TradeDirection.BUY
        )

        # 大订单
        total_large, _, _ = model.estimate_impact(
            1_000_000, parameters, TradeDirection.BUY
        )

        # 大订单冲击应该更大
        assert total_large > total_small

        # 但不是线性关系(平方根模型)
        assert total_large < total_small * (1_000_000 / 10_000)

    def test_zero_volume_handling(self):
        """测试零成交量处理"""
        model = SquareRootImpactModel()
        params = create_impact_parameters(
            symbol="TEST",
            price=100.0,
            daily_volume=0,  # 零成交量
            volatility=0.20,
        )

        total, temp, perm = model.estimate_impact(
            10_000, params, TradeDirection.BUY
        )

        # 应该返回零冲击
        assert total == 0.0
        assert temp == 0.0
        assert perm == 0.0


# ==================== 冲击估算测试 ====================

class TestImpactEstimation:
    """冲击估算测试"""

    @pytest.mark.asyncio
    async def test_basic_impact_estimate(self, estimator, parameters):
        """测试基本冲击估算"""
        estimate = await estimator.estimate_impact(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=100_000,
            current_price=150.0,
            parameters=parameters,
        )

        assert estimate is not None
        assert estimate.symbol == "AAPL"
        assert estimate.direction == TradeDirection.BUY
        assert estimate.quantity == 100_000
        assert estimate.estimated_impact > 0
        assert estimate.temporary_impact > 0
        assert estimate.permanent_impact > 0
        assert 0 <= estimate.confidence <= 1

    @pytest.mark.asyncio
    async def test_different_models(self, estimator, parameters):
        """测试不同模型"""
        quantity = 100_000

        # Linear模型
        linear_est = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, quantity, 150.0,
            parameters, ImpactModelType.LINEAR
        )

        # Square Root模型
        sqrt_est = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, quantity, 150.0,
            parameters, ImpactModelType.SQUARE_ROOT
        )

        # Almgren-Chriss模型
        ac_est = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, quantity, 150.0,
            parameters, ImpactModelType.ALMGREN_CHRISS
        )

        # 所有估计应该都大于0
        assert linear_est.estimated_impact > 0
        assert sqrt_est.estimated_impact > 0
        assert ac_est.estimated_impact > 0

        # 不同模型结果应该不同
        assert linear_est.estimated_impact != sqrt_est.estimated_impact

    @pytest.mark.asyncio
    async def test_participation_rate_calculation(self, estimator, parameters):
        """测试参与率计算"""
        quantity = 500_000  # 1% of daily volume

        estimate = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, quantity, 150.0, parameters
        )

        expected_rate = 500_000 / 50_000_000
        assert abs(estimate.participation_rate - expected_rate) < 0.0001

    @pytest.mark.asyncio
    async def test_cost_calculation(self, estimator, parameters):
        """测试成本计算"""
        quantity = 100_000
        price = 150.0

        estimate = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, quantity, price, parameters
        )

        # 验证成本计算
        expected_cost = quantity * price * estimate.estimated_cost_bps / 10000
        assert abs(estimate.estimated_cost_currency - expected_cost) < 1.0

    @pytest.mark.asyncio
    async def test_confidence_interval(self, estimator, parameters):
        """测试置信区间"""
        estimate = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, 100_000, 150.0, parameters
        )

        # 置信区间应该合理
        assert estimate.impact_low < estimate.estimated_impact
        assert estimate.estimated_impact < estimate.impact_high
        assert estimate.impact_low >= 0

    @pytest.mark.asyncio
    async def test_impact_decomposition(self, estimator, parameters):
        """测试冲击分解"""
        estimate = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, 100_000, 150.0, parameters
        )

        decomp = estimate.decomposition
        assert decomp is not None
        assert decomp.temporary_impact > 0
        assert decomp.permanent_impact > 0
        assert abs(decomp.total_impact - estimate.estimated_impact) < 0.1

        # 检查组件
        assert decomp.spread_component >= 0
        assert decomp.liquidity_component >= 0
        assert decomp.information_component >= 0
        assert decomp.volatility_component >= 0

    @pytest.mark.asyncio
    async def test_market_condition_adjustment(self, estimator):
        """测试市场状态调整"""
        # 正常市场
        normal_params = create_impact_parameters(
            "TEST", 100.0, 10_000_000, 0.20,
            market_condition=MarketCondition.NORMAL
        )

        # 高波动市场
        volatile_params = create_impact_parameters(
            "TEST", 100.0, 10_000_000, 0.20,
            market_condition=MarketCondition.VOLATILE
        )

        normal_est = await estimator.estimate_impact(
            "TEST", TradeDirection.BUY, 100_000, 100.0, normal_params
        )

        volatile_est = await estimator.estimate_impact(
            "TEST", TradeDirection.BUY, 100_000, 100.0, volatile_params
        )

        # 高波动市场冲击应该更大
        assert volatile_est.estimated_impact > normal_est.estimated_impact


# ==================== 执行路径优化测试 ====================

class TestExecutionPathOptimization:
    """执行路径优化测试"""

    @pytest.mark.asyncio
    async def test_twap_path(self, estimator, parameters):
        """测试TWAP路径"""
        path = await estimator.optimize_execution_path(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=1_000_000,
            current_price=150.0,
            strategy=ExecutionStrategy.TWAP,
            duration_minutes=60,
            parameters=parameters,
        )

        assert path is not None
        assert path.strategy == ExecutionStrategy.TWAP
        assert path.total_quantity == 1_000_000
        assert len(path.slices) > 0
        assert path.total_duration_seconds == 3600

        # 验证切片总量
        total_slice_quantity = sum(s.quantity for s in path.slices)
        assert abs(total_slice_quantity - 1_000_000) < 1.0

    @pytest.mark.asyncio
    async def test_vwap_path(self, estimator, parameters):
        """测试VWAP路径"""
        path = await estimator.optimize_execution_path(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=500_000,
            current_price=150.0,
            strategy=ExecutionStrategy.VWAP,
            parameters=parameters,
        )

        assert path.strategy == ExecutionStrategy.VWAP
        assert len(path.slices) > 0

    @pytest.mark.asyncio
    async def test_pov_path(self, estimator, parameters):
        """测试POV路径"""
        path = await estimator.optimize_execution_path(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=500_000,
            current_price=150.0,
            strategy=ExecutionStrategy.POV,
            parameters=parameters,
        )

        assert path.strategy == ExecutionStrategy.POV
        assert len(path.slices) > 0

    @pytest.mark.asyncio
    async def test_implementation_shortfall_path(self, estimator, parameters):
        """测试Implementation Shortfall路径"""
        path = await estimator.optimize_execution_path(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=1_000_000,
            current_price=150.0,
            strategy=ExecutionStrategy.IMPLEMENTATION_SHORTFALL,
            parameters=parameters,
        )

        assert path.strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL
        assert len(path.slices) > 0

        # IS策略应该前端加载
        first_slice = path.slices[0]
        last_slice = path.slices[-1]
        assert first_slice.quantity > last_slice.quantity

    @pytest.mark.asyncio
    async def test_execution_risk_metrics(self, estimator, parameters):
        """测试执行风险指标"""
        path = await estimator.optimize_execution_path(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=1_000_000,
            current_price=150.0,
            parameters=parameters,
        )

        assert path.execution_risk >= 0
        assert path.timing_risk >= 0
        assert path.total_expected_impact > 0
        assert path.total_expected_cost > 0

    @pytest.mark.asyncio
    async def test_slice_continuity(self, estimator, parameters):
        """测试切片连续性"""
        path = await estimator.optimize_execution_path(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            quantity=500_000,
            current_price=150.0,
            parameters=parameters,
        )

        # 验证切片时间连续性
        for i in range(len(path.slices) - 1):
            current_slice = path.slices[i]
            next_slice = path.slices[i + 1]
            assert current_slice.end_time == next_slice.start_time


# ==================== 实时跟踪测试 ====================

class TestRealtimeTracking:
    """实时跟踪测试"""

    @pytest.mark.asyncio
    async def test_realtime_impact_tracking(self, estimator, parameters):
        """测试实时冲击跟踪"""
        track = await estimator.track_realtime_impact(
            track_id="track_001",
            symbol="AAPL",
            direction=TradeDirection.BUY,
            target_quantity=1_000_000,
            executed_quantity=400_000,
            initial_price=150.0,
            current_price=150.5,
            average_execution_price=150.2,
            elapsed_time_seconds=1200,
            parameters=parameters,
        )

        assert track is not None
        assert track.symbol == "AAPL"
        assert track.target_quantity == 1_000_000
        assert track.executed_quantity == 400_000
        assert track.remaining_quantity == 600_000
        assert track.execution_progress == 0.4

    @pytest.mark.asyncio
    async def test_realized_impact_calculation(self, estimator, parameters):
        """测试已实现冲击计算"""
        # 买入场景
        track_buy = await estimator.track_realtime_impact(
            "track_buy",
            "AAPL",
            TradeDirection.BUY,
            100_000, 50_000,
            150.0, 150.5, 150.3,
            600, parameters
        )

        # 买入冲击应该为正(价格上涨)
        assert track_buy.realized_impact > 0

        # 卖出场景
        track_sell = await estimator.track_realtime_impact(
            "track_sell",
            "AAPL",
            TradeDirection.SELL,
            100_000, 50_000,
            150.0, 149.5, 149.7,
            600, parameters
        )

        # 卖出冲击应该为正(价格下跌)
        assert track_sell.realized_impact > 0

    @pytest.mark.asyncio
    async def test_estimated_remaining_impact(self, estimator, parameters):
        """测试剩余冲击估计"""
        track = await estimator.track_realtime_impact(
            "track_002",
            "AAPL",
            TradeDirection.BUY,
            1_000_000, 200_000,
            150.0, 150.2, 150.1,
            300, parameters
        )

        assert track.estimated_remaining_impact >= 0
        assert track.total_projected_impact > 0

    @pytest.mark.asyncio
    async def test_cost_tracking(self, estimator, parameters):
        """测试成本跟踪"""
        track = await estimator.track_realtime_impact(
            "track_003",
            "AAPL",
            TradeDirection.BUY,
            1_000_000, 500_000,
            150.0, 150.5, 150.3,
            1800, parameters
        )

        assert track.realized_cost > 0
        assert track.estimated_total_cost > track.realized_cost

    @pytest.mark.asyncio
    async def test_complete_execution(self, estimator, parameters):
        """测试完整执行"""
        track = await estimator.track_realtime_impact(
            "track_complete",
            "AAPL",
            TradeDirection.BUY,
            1_000_000, 1_000_000,  # 完全执行
            150.0, 150.5, 150.2,
            3600, parameters
        )

        assert track.execution_progress == 1.0
        assert track.remaining_quantity == 0
        assert track.estimated_remaining_impact == 0

    @pytest.mark.asyncio
    async def test_track_retrieval(self, estimator, parameters):
        """测试跟踪检索"""
        # 创建跟踪
        await estimator.track_realtime_impact(
            "track_retrieve",
            "AAPL",
            TradeDirection.BUY,
            100_000, 50_000,
            150.0, 150.2, 150.1,
            600, parameters
        )

        # 检索跟踪
        track = estimator.get_realtime_track("track_retrieve")
        assert track is not None
        assert track.symbol == "AAPL"


# ==================== 参数校准测试 ====================

class TestParameterCalibration:
    """参数校准测试"""

    @pytest.mark.asyncio
    async def test_basic_calibration(self, estimator):
        """测试基本校准"""
        historical_trades = [
            {
                'realized_impact': 5.0,
                'participation_rate': 0.01,
                'volatility': 0.25,
            },
            {
                'realized_impact': 8.0,
                'participation_rate': 0.02,
                'volatility': 0.25,
            },
            {
                'realized_impact': 12.0,
                'participation_rate': 0.03,
                'volatility': 0.25,
            },
        ]

        # 数据不足，应该返回默认参数
        params = await estimator.calibrate_parameters(
            "AAPL", historical_trades
        )

        assert params is not None
        assert params.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_calibration_with_sufficient_data(self, estimator):
        """测试充足数据校准"""
        # 生成足够的历史数据
        historical_trades = []
        for i in range(30):
            historical_trades.append({
                'realized_impact': 5.0 + i * 0.2,
                'participation_rate': 0.01 + i * 0.001,
                'volatility': 0.25,
            })

        params = await estimator.calibrate_parameters(
            "AAPL", historical_trades, ImpactModelType.LINEAR
        )

        assert params.calibrated_at is not None
        assert params.linear_coefficient > 0

    @pytest.mark.asyncio
    async def test_parameters_storage(self, estimator, parameters):
        """测试参数存储和检索"""
        # 设置参数
        estimator.set_parameters("AAPL", parameters)

        # 检索参数
        retrieved = estimator.get_parameters("AAPL")
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.price == parameters.price


# ==================== 情景分析测试 ====================

class TestScenarioAnalysis:
    """情景分析测试"""

    @pytest.mark.asyncio
    async def test_scenario_analysis(self, estimator, parameters):
        """测试情景分析"""
        analysis = await estimator.analyze_scenarios(
            symbol="AAPL",
            direction=TradeDirection.BUY,
            base_quantity=1_000_000,
            current_price=150.0,
            parameters=parameters,
        )

        assert analysis is not None
        assert len(analysis.scenarios) > 0
        assert len(analysis.scenario_results) > 0
        assert analysis.recommended_scenario is not None
        assert analysis.recommendation_reason is not None

    @pytest.mark.asyncio
    async def test_scenario_comparison(self, estimator, parameters):
        """测试情景对比"""
        analysis = await estimator.analyze_scenarios(
            "AAPL", TradeDirection.BUY, 500_000, 150.0, parameters
        )

        # 验证不同情景有不同结果
        results = analysis.scenario_results
        impacts = [r['estimated_impact'] for r in results]

        # 应该有变化
        assert max(impacts) > min(impacts)

    @pytest.mark.asyncio
    async def test_recommendation_logic(self, estimator, high_volume_parameters):
        """测试推荐逻辑"""
        analysis = await estimator.analyze_scenarios(
            "SPY", TradeDirection.BUY, 100_000, 400.0,
            high_volume_parameters
        )

        # 对于高流动性标的，应该有明确推荐
        assert analysis.recommended_scenario in [
            r['name'] for r in analysis.scenario_results
        ]


# ==================== 工厂函数测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_estimator(self):
        """测试创建估算器"""
        estimator = create_impact_estimator(
            default_model="linear",
            auto_calibrate=False,
        )

        assert estimator is not None
        assert estimator.config.default_model == ImpactModelType.LINEAR
        assert estimator.config.auto_calibrate is False

    def test_create_parameters(self):
        """测试创建参数"""
        params = create_impact_parameters(
            symbol="MSFT",
            price=300.0,
            daily_volume=20_000_000,
            volatility=0.22,
            spread_bps=4.0,
            linear_coefficient=0.15,
        )

        assert params.symbol == "MSFT"
        assert params.price == 300.0
        assert params.linear_coefficient == 0.15

    @pytest.mark.asyncio
    async def test_quick_estimate(self):
        """测试快速估算"""
        estimate = await quick_impact_estimate(
            symbol="GOOGL",
            direction="buy",
            quantity=50_000,
            price=2500.0,
            daily_volume=25_000_000,
            volatility=0.28,
            model="square_root",
        )

        assert estimate is not None
        assert estimate.symbol == "GOOGL"
        assert estimate.direction == TradeDirection.BUY
        assert estimate.estimated_impact > 0


# ==================== 边界条件测试 ====================

class TestEdgeCases:
    """边界条件测试"""

    @pytest.mark.asyncio
    async def test_very_small_order(self, estimator, parameters):
        """测试极小订单"""
        estimate = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, 100, 150.0, parameters
        )

        assert estimate.estimated_impact >= 0
        assert estimate.participation_rate < 0.0001

    @pytest.mark.asyncio
    async def test_very_large_order(self, estimator, parameters):
        """测试极大订单"""
        estimate = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, 10_000_000, 150.0, parameters
        )

        assert estimate.estimated_impact > 0
        assert estimate.participation_rate > 0.1
        # 置信度应该降低
        assert estimate.confidence < 1.0

    @pytest.mark.asyncio
    async def test_zero_quantity(self, estimator, parameters):
        """测试零数量"""
        estimate = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, 0, 150.0, parameters
        )

        assert estimate.estimated_impact == 0
        assert estimate.estimated_cost_currency == 0

    @pytest.mark.asyncio
    async def test_high_volatility(self, estimator):
        """测试高波动率"""
        high_vol_params = create_impact_parameters(
            "VOLATILE", 100.0, 5_000_000, 0.80, 10.0
        )

        estimate = await estimator.estimate_impact(
            "VOLATILE", TradeDirection.BUY, 100_000, 100.0, high_vol_params
        )

        # 高波动率应该导致更高冲击
        assert estimate.estimated_impact > 0

    @pytest.mark.asyncio
    async def test_illiquid_market(self, estimator, low_liquidity_parameters):
        """测试低流动性市场"""
        estimate = await estimator.estimate_impact(
            "SMALLCAP", TradeDirection.BUY, 50_000, 50.0,
            low_liquidity_parameters
        )

        # 低流动性应该导致更高冲击
        assert estimate.estimated_impact > 0
        assert estimate.participation_rate > 0.01


# ==================== 数据类测试 ====================

class TestDataClasses:
    """数据类测试"""

    def test_impact_estimate_to_dict(self, estimator, parameters):
        """测试ImpactEstimate序列化"""
        estimate = ImpactEstimate(
            symbol="TEST",
            direction=TradeDirection.BUY,
            quantity=100_000,
            current_price=100.0,
            estimated_impact=10.0,
            temporary_impact=7.0,
            permanent_impact=3.0,
            confidence=0.85,
            impact_low=8.0,
            impact_high=12.0,
            model_type=ImpactModelType.SQUARE_ROOT,
            participation_rate=0.01,
            estimated_cost_bps=10.0,
            estimated_cost_currency=1000.0,
        )

        result = estimate.to_dict()
        assert 'symbol' in result
        assert 'estimated_impact' in result
        assert result['symbol'] == "TEST"
        assert result['estimated_impact'] == 10.0

    def test_execution_path_to_dict(self):
        """测试ExecutionPath序列化"""
        path = ExecutionPath(
            symbol="TEST",
            direction=TradeDirection.BUY,
            total_quantity=100_000,
            strategy=ExecutionStrategy.TWAP,
            slices=[],
            total_duration_seconds=3600,
            total_expected_impact=10.0,
            total_expected_cost=1000.0,
            execution_risk=5.0,
            timing_risk=3.0,
        )

        result = path.to_dict()
        assert 'symbol' in result
        assert 'strategy' in result
        assert result['strategy'] == 'twap'

    def test_realtime_impact_to_dict(self):
        """测试RealTimeImpact序列化"""
        track = RealTimeImpact(
            symbol="TEST",
            direction=TradeDirection.BUY,
            target_quantity=100_000,
            executed_quantity=50_000,
            remaining_quantity=50_000,
            initial_price=100.0,
            current_price=100.5,
            average_execution_price=100.2,
            realized_impact=2.0,
            estimated_remaining_impact=3.0,
            total_projected_impact=2.5,
            execution_progress=0.5,
            elapsed_time_seconds=1800,
            realized_cost=1000.0,
            estimated_total_cost=2500.0,
        )

        result = track.to_dict()
        assert 'symbol' in result
        assert 'execution_progress' in result
        assert result['execution_progress'] == 0.5


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, estimator, parameters):
        """测试完整工作流"""
        # 1. 估计冲击
        estimate = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, 500_000, 150.0, parameters
        )
        assert estimate.estimated_impact > 0

        # 2. 优化执行路径
        path = await estimator.optimize_execution_path(
            "AAPL", TradeDirection.BUY, 500_000, 150.0,
            ExecutionStrategy.TWAP, 60, parameters
        )
        assert len(path.slices) > 0

        # 3. 实时跟踪
        track = await estimator.track_realtime_impact(
            "workflow_track",
            "AAPL", TradeDirection.BUY,
            500_000, 200_000,
            150.0, 150.3, 150.2,
            1200, parameters
        )
        assert track.execution_progress > 0

    @pytest.mark.asyncio
    async def test_multi_symbol_estimation(self, estimator):
        """测试多品种估算"""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        for symbol in symbols:
            params = create_impact_parameters(
                symbol, 100.0, 10_000_000, 0.25
            )

            estimate = await estimator.estimate_impact(
                symbol, TradeDirection.BUY, 100_000, 100.0, params
            )

            assert estimate.symbol == symbol
            assert estimate.estimated_impact > 0

    @pytest.mark.asyncio
    async def test_history_management(self, estimator, parameters):
        """测试历史管理"""
        # 生成一些估算
        for i in range(5):
            await estimator.estimate_impact(
                "AAPL", TradeDirection.BUY,
                100_000 * (i + 1), 150.0, parameters
            )

        # 清除历史
        estimator.clear_history("AAPL")

        # 验证清除成功(通过生成新估算)
        estimate = await estimator.estimate_impact(
            "AAPL", TradeDirection.BUY, 100_000, 150.0, parameters
        )
        assert estimate is not None


# ==================== 性能测试 ====================

class TestPerformance:
    """性能测试"""

    @pytest.mark.asyncio
    async def test_estimation_performance(self, estimator, parameters):
        """测试估算性能"""
        import time

        start_time = time.time()

        # 执行100次估算
        for i in range(100):
            await estimator.estimate_impact(
                "AAPL", TradeDirection.BUY,
                100_000, 150.0, parameters
            )

        elapsed_time = time.time() - start_time

        # 平均每次估算应该在合理时间内
        avg_time = elapsed_time / 100
        assert avg_time < 0.1  # 每次不超过100ms

    @pytest.mark.asyncio
    async def test_path_optimization_performance(self, estimator, parameters):
        """测试路径优化性能"""
        import time

        start_time = time.time()

        # 执行10次路径优化
        for i in range(10):
            await estimator.optimize_execution_path(
                "AAPL", TradeDirection.BUY,
                1_000_000, 150.0,
                ExecutionStrategy.TWAP, 60, parameters
            )

        elapsed_time = time.time() - start_time

        # 平均每次优化应该在合理时间内
        avg_time = elapsed_time / 10
        assert avg_time < 0.5  # 每次不超过500ms


# ==================== 运行所有测试 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
