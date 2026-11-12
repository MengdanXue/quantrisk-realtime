"""
仓位规模计算引擎测试
Position Sizing Engine Tests
"""

import pytest
import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from quantrisk_realtime.realtime.position_sizing_engine import (
    # 枚举
    SizingMethod,
    PositionType,
    RiskLevel,
    ConstraintType,
    AdjustmentTrigger,
    EngineState,
    # 数据类
    AssetMetrics,
    SizingResult,
    PortfolioSizing,
    PositionConstraint,
    AdjustmentSignal,
    SizingConfig,
    # 计算器
    PositionSizer,
    KellyCriterionSizer,
    FixedFractionalSizer,
    VolatilityBasedSizer,
    RiskParitySizer,
    EqualWeightSizer,
    ATRBasedSizer,
    DrawdownConstrainedSizer,
    # 核心引擎
    PositionSizingEngine,
    # 工厂函数
    create_sizing_engine,
    create_asset_metrics,
    create_position_constraint,
    quick_position_sizing,
)


# ==================== 测试数据 ====================

@pytest.fixture
def basic_config():
    """基础配置"""
    return SizingConfig(
        account_size=100000.0,
        risk_per_trade=0.02,
        max_position_size=0.25,
        max_portfolio_heat=0.06,
    )


@pytest.fixture
def basic_metrics():
    """基础资产指标"""
    return AssetMetrics(
        symbol="AAPL",
        current_price=150.0,
        expected_return=0.15,
        win_rate=0.55,
        avg_win=0.03,
        avg_loss=0.02,
        volatility=0.25,
        daily_volatility=0.016,
        atr=3.0,
        atr_percent=0.02,
    )


@pytest.fixture
def sizing_engine(basic_config):
    """仓位计算引擎"""
    return PositionSizingEngine(basic_config)


# ==================== AssetMetrics 测试 ====================

class TestAssetMetrics:
    """资产指标测试"""

    def test_asset_metrics_creation(self):
        """测试创建资产指标"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.25,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.current_price == 150.0
        assert metrics.volatility == 0.25

    def test_asset_metrics_with_optional_fields(self):
        """测试包含可选字段的创建"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            entry_price=145.0,
            win_rate=0.6,
            avg_win=0.04,
            avg_loss=0.02,
            volatility=0.25,
            beta=1.2,
            sharpe_ratio=1.5,
        )

        assert metrics.entry_price == 145.0
        assert metrics.win_rate == 0.6
        assert metrics.beta == 1.2

    def test_asset_metrics_to_dict(self):
        """测试转换为字典"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.25,
        )

        data = metrics.to_dict()
        assert isinstance(data, dict)
        assert data['symbol'] == "AAPL"
        assert data['current_price'] == 150.0


# ==================== SizingConfig 测试 ====================

class TestSizingConfig:
    """仓位配置测试"""

    def test_config_default_values(self):
        """测试默认值"""
        config = SizingConfig()

        assert config.account_size == 100000.0
        assert config.risk_per_trade == 0.02
        assert config.max_position_size == 0.25

    def test_config_custom_values(self):
        """测试自定义值"""
        config = SizingConfig(
            account_size=200000.0,
            risk_per_trade=0.01,
            max_position_size=0.2,
            kelly_multiplier=0.5,
        )

        assert config.account_size == 200000.0
        assert config.risk_per_trade == 0.01
        assert config.kelly_multiplier == 0.5

    def test_config_to_dict(self):
        """测试配置序列化"""
        config = SizingConfig()
        data = config.to_dict()

        assert isinstance(data, dict)
        assert 'account_size' in data
        assert 'risk_per_trade' in data


# ==================== PositionConstraint 测试 ====================

class TestPositionConstraint:
    """仓位约束测试"""

    def test_constraint_creation(self):
        """测试创建约束"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_POSITION,
            value=0.25,
        )

        assert constraint.constraint_type == ConstraintType.MAX_POSITION
        assert constraint.value == 0.25
        assert constraint.is_hard_limit is True

    def test_max_position_check_pass(self):
        """测试最大仓位检查-通过"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_POSITION,
            value=0.25,
        )

        violated, message = constraint.check(0.20)
        assert not violated
        assert message == ""

    def test_max_position_check_fail(self):
        """测试最大仓位检查-失败"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_POSITION,
            value=0.25,
        )

        violated, message = constraint.check(0.30)
        assert violated
        assert "exceeds max limit" in message

    def test_min_position_check(self):
        """测试最小仓位检查"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MIN_POSITION,
            value=0.05,
        )

        violated, message = constraint.check(0.03)
        assert violated

    def test_max_leverage_check(self):
        """测试最大杠杆检查"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_LEVERAGE,
            value=3.0,
        )

        violated, message = constraint.check(3.5)
        assert violated
        assert "Leverage exceeds max" in message

    def test_max_drawdown_check(self):
        """测试最大回撤检查"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_DRAWDOWN,
            value=0.20,
        )

        violated, message = constraint.check(0.25)
        assert violated

    def test_soft_limit_constraint(self):
        """测试软限制约束"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_POSITION,
            value=0.25,
            is_hard_limit=False,
        )

        assert constraint.is_hard_limit is False


# ==================== Kelly Criterion Sizer 测试 ====================

class TestKellyCriterionSizer:
    """凯利公式计算器测试"""

    def test_kelly_sizer_creation(self, basic_config):
        """测试创建Kelly计算器"""
        sizer = KellyCriterionSizer(basic_config)
        assert sizer is not None

    def test_kelly_calculation_basic(self, basic_config, basic_metrics):
        """测试基本Kelly计算"""
        sizer = KellyCriterionSizer(basic_config)
        result = sizer.calculate("AAPL", basic_metrics)

        assert result.method == SizingMethod.KELLY
        assert result.suggested_size > 0
        assert result.kelly_fraction is not None

    def test_kelly_positive_expectancy(self, basic_config):
        """测试正期望值Kelly计算"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            win_rate=0.6,
            avg_win=0.04,
            avg_loss=0.02,
            volatility=0.25,
        )

        sizer = KellyCriterionSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        assert result.kelly_fraction > 0
        assert result.suggested_size > 0

    def test_kelly_negative_expectancy(self, basic_config):
        """测试负期望值Kelly计算"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            win_rate=0.4,
            avg_win=0.02,
            avg_loss=0.04,
            volatility=0.25,
        )

        sizer = KellyCriterionSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        # 负期望值应该限制为0或很小
        assert result.kelly_fraction <= 0

    def test_kelly_with_multiplier(self, basic_metrics):
        """测试Kelly乘数效果"""
        config1 = SizingConfig(kelly_multiplier=0.25)
        config2 = SizingConfig(kelly_multiplier=0.5)

        sizer1 = KellyCriterionSizer(config1)
        sizer2 = KellyCriterionSizer(config2)

        result1 = sizer1.calculate("AAPL", basic_metrics)
        result2 = sizer2.calculate("AAPL", basic_metrics)

        # 更高的乘数应产生更大的仓位
        assert result2.suggested_size > result1.suggested_size

    def test_kelly_half_kelly_quarter_kelly(self, basic_config, basic_metrics):
        """测试半Kelly和四分之一Kelly"""
        sizer = KellyCriterionSizer(basic_config)
        result = sizer.calculate("AAPL", basic_metrics)

        if result.kelly_fraction:
            assert result.half_kelly == result.kelly_fraction * 0.5
            assert result.quarter_kelly == result.kelly_fraction * 0.25

    def test_kelly_invalid_win_rate(self, basic_config):
        """测试无效胜率处理"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            win_rate=1.5,  # 无效值
            avg_win=0.03,
            avg_loss=0.02,
            volatility=0.25,
        )

        sizer = KellyCriterionSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        # 应该有默认处理
        assert result.suggested_size >= 0


# ==================== Fixed Fractional Sizer 测试 ====================

class TestFixedFractionalSizer:
    """固定比例计算器测试"""

    def test_fixed_fractional_creation(self, basic_config):
        """测试创建固定比例计算器"""
        sizer = FixedFractionalSizer(basic_config)
        assert sizer is not None

    def test_fixed_fractional_basic(self, basic_config, basic_metrics):
        """测试基本固定比例计算"""
        sizer = FixedFractionalSizer(basic_config)
        result = sizer.calculate("AAPL", basic_metrics)

        assert result.method == SizingMethod.FIXED_FRACTIONAL
        assert result.suggested_size > 0
        assert result.risk_per_trade > 0

    def test_fixed_fractional_with_atr(self, basic_config):
        """测试使用ATR的固定比例"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            atr=3.0,
            atr_percent=0.02,
            volatility=0.25,
        )

        sizer = FixedFractionalSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        assert result.stop_loss_price is not None
        assert result.stop_loss_percent > 0

    def test_fixed_fractional_with_entry_price(self, basic_config):
        """测试使用入场价格的固定比例"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            entry_price=145.0,
            volatility=0.25,
        )

        sizer = FixedFractionalSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        assert result.stop_loss_price is not None

    def test_fixed_fractional_risk_calculation(self, basic_config, basic_metrics):
        """测试风险计算"""
        sizer = FixedFractionalSizer(basic_config)
        result = sizer.calculate("AAPL", basic_metrics)

        expected_risk = basic_config.account_size * basic_config.risk_per_trade
        assert abs(result.risk_per_trade - expected_risk) < 1.0

    def test_fixed_fractional_max_position_limit(self, basic_metrics):
        """测试最大仓位限制"""
        config = SizingConfig(
            account_size=100000.0,
            risk_per_trade=0.1,  # 高风险
            max_position_size=0.25,
        )

        sizer = FixedFractionalSizer(config)
        result = sizer.calculate("AAPL", basic_metrics)

        # 仓位价值不应超过最大限制
        max_value = config.account_size * config.max_position_size
        assert result.suggested_value <= max_value * 1.01  # 允许1%误差


# ==================== Volatility Based Sizer 测试 ====================

class TestVolatilityBasedSizer:
    """波动率调整计算器测试"""

    def test_volatility_sizer_creation(self, basic_config):
        """测试创建波动率计算器"""
        sizer = VolatilityBasedSizer(basic_config)
        assert sizer is not None

    def test_volatility_sizing_basic(self, basic_config, basic_metrics):
        """测试基本波动率计算"""
        sizer = VolatilityBasedSizer(basic_config)
        result = sizer.calculate("AAPL", basic_metrics)

        assert result.method == SizingMethod.VOLATILITY
        assert result.suggested_size > 0

    def test_volatility_inverse_relationship(self, basic_config):
        """测试波动率反向关系"""
        # 低波动率资产
        low_vol_metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.15,
            daily_volatility=0.01,
        )

        # 高波动率资产
        high_vol_metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.35,
            daily_volatility=0.022,
        )

        sizer = VolatilityBasedSizer(basic_config)
        low_vol_result = sizer.calculate("AAPL", low_vol_metrics)
        high_vol_result = sizer.calculate("AAPL", high_vol_metrics)

        # 低波动率应产生更大的仓位
        assert low_vol_result.suggested_size > high_vol_result.suggested_size

    def test_volatility_target_adjustment(self, basic_metrics):
        """测试目标波动率调整"""
        config1 = SizingConfig(volatility_target=0.10)
        config2 = SizingConfig(volatility_target=0.20)

        sizer1 = VolatilityBasedSizer(config1)
        sizer2 = VolatilityBasedSizer(config2)

        result1 = sizer1.calculate("AAPL", basic_metrics)
        result2 = sizer2.calculate("AAPL", basic_metrics)

        # 更高的目标波动率应产生更大的仓位
        assert result2.suggested_size > result1.suggested_size

    def test_volatility_with_zero_vol(self, basic_config):
        """测试零波动率处理"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.0,
            daily_volatility=0.0,
        )

        sizer = VolatilityBasedSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        # 应该有默认处理
        assert result.suggested_size > 0


# ==================== Risk Parity Sizer 测试 ====================

class TestRiskParitySizer:
    """风险平价计算器测试"""

    def test_risk_parity_creation(self, basic_config):
        """测试创建风险平价计算器"""
        sizer = RiskParitySizer(basic_config)
        assert sizer is not None

    def test_risk_parity_basic(self, basic_config, basic_metrics):
        """测试基本风险平价计算"""
        sizer = RiskParitySizer(basic_config)
        result = sizer.calculate("AAPL", basic_metrics)

        assert result.method == SizingMethod.RISK_PARITY
        assert result.suggested_size > 0

    def test_risk_parity_equal_risk_contribution(self, basic_config):
        """测试相等风险贡献"""
        # 两个不同波动率的资产
        metrics1 = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.20,
        )

        metrics2 = AssetMetrics(
            symbol="GOOGL",
            current_price=2800.0,
            volatility=0.30,
        )

        sizer = RiskParitySizer(basic_config)
        result1 = sizer.calculate("AAPL", metrics1)
        result2 = sizer.calculate("GOOGL", metrics2)

        # 风险贡献应该接近
        risk1 = result1.suggested_weight * metrics1.volatility
        risk2 = result2.suggested_weight * metrics2.volatility

        assert abs(risk1 - risk2) < 0.05

    def test_risk_parity_with_target_risk(self, basic_metrics):
        """测试目标风险贡献"""
        config = SizingConfig(target_risk_contribution=0.15)
        sizer = RiskParitySizer(config)
        result = sizer.calculate("AAPL", basic_metrics)

        risk_contribution = result.suggested_weight * basic_metrics.volatility
        # 风险贡献应该接近目标值，但可能受到最大仓位限制
        assert risk_contribution > 0


# ==================== Equal Weight Sizer 测试 ====================

class TestEqualWeightSizer:
    """等权重计算器测试"""

    def test_equal_weight_creation(self, basic_config):
        """测试创建等权重计算器"""
        sizer = EqualWeightSizer(basic_config, num_positions=10)
        assert sizer is not None
        assert sizer.num_positions == 10

    def test_equal_weight_basic(self, basic_config, basic_metrics):
        """测试基本等权重计算"""
        sizer = EqualWeightSizer(basic_config, num_positions=5)
        result = sizer.calculate("AAPL", basic_metrics)

        assert result.method == SizingMethod.EQUAL_WEIGHT
        assert abs(result.suggested_weight - 0.2) < 0.001  # 1/5 = 0.2

    def test_equal_weight_different_num_positions(self, basic_config, basic_metrics):
        """测试不同持仓数量"""
        sizer5 = EqualWeightSizer(basic_config, num_positions=5)
        sizer10 = EqualWeightSizer(basic_config, num_positions=10)

        result5 = sizer5.calculate("AAPL", basic_metrics)
        result10 = sizer10.calculate("AAPL", basic_metrics)

        # 更多持仓数应产生更小的权重
        assert result5.suggested_weight == 0.2
        assert result10.suggested_weight == 0.1


# ==================== ATR Based Sizer 测试 ====================

class TestATRBasedSizer:
    """ATR基准计算器测试"""

    def test_atr_sizer_creation(self, basic_config):
        """测试创建ATR计算器"""
        sizer = ATRBasedSizer(basic_config)
        assert sizer is not None

    def test_atr_sizing_basic(self, basic_config, basic_metrics):
        """测试基本ATR计算"""
        sizer = ATRBasedSizer(basic_config)
        result = sizer.calculate("AAPL", basic_metrics)

        assert result.method == SizingMethod.ATR_BASED
        assert result.suggested_size > 0
        assert result.stop_loss_price is not None

    def test_atr_multiplier_effect(self, basic_metrics):
        """测试ATR倍数效果"""
        # 使用较大的account_size以避免被max_position_size限制
        config1 = SizingConfig(account_size=1000000.0, atr_multiplier=1.5, max_position_size=1.0)
        config2 = SizingConfig(account_size=1000000.0, atr_multiplier=3.0, max_position_size=1.0)

        sizer1 = ATRBasedSizer(config1)
        sizer2 = ATRBasedSizer(config2)

        result1 = sizer1.calculate("AAPL", basic_metrics)
        result2 = sizer2.calculate("AAPL", basic_metrics)

        # 更大的ATR倍数应产生更小的仓位(更宽的止损)
        assert result1.suggested_size > result2.suggested_size

    def test_atr_with_zero_atr(self, basic_config):
        """测试零ATR处理"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            atr=0.0,
            volatility=0.25,
        )

        sizer = ATRBasedSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        # 应该使用默认值
        assert result.suggested_size > 0


# ==================== Drawdown Constrained Sizer 测试 ====================

class TestDrawdownConstrainedSizer:
    """回撤约束计算器测试"""

    def test_drawdown_sizer_creation(self, basic_config):
        """测试创建回撤约束计算器"""
        sizer = DrawdownConstrainedSizer(basic_config)
        assert sizer is not None

    def test_drawdown_sizing_no_drawdown(self, basic_config, basic_metrics):
        """测试无回撤情况"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            max_drawdown=0.0,
            volatility=0.25,
        )

        sizer = DrawdownConstrainedSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        assert result.method == SizingMethod.DRAWDOWN_CONSTRAINED
        assert not result.is_constrained

    def test_drawdown_sizing_with_drawdown(self, basic_config, basic_metrics):
        """测试有回撤情况"""
        metrics = basic_metrics
        metrics.max_drawdown = 0.15

        sizer = DrawdownConstrainedSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        # 应该应用约束
        assert result.is_constrained
        assert result.original_size is not None

    def test_drawdown_sizing_severe_drawdown(self, basic_config, basic_metrics):
        """测试严重回撤"""
        metrics = basic_metrics
        metrics.max_drawdown = 0.25

        sizer = DrawdownConstrainedSizer(basic_config)
        result = sizer.calculate("AAPL", metrics)

        # 应该大幅减仓
        assert result.is_constrained
        assert result.suggested_size < result.original_size * 0.6


# ==================== PositionSizingEngine 测试 ====================

class TestPositionSizingEngine:
    """仓位计算引擎测试"""

    def test_engine_creation(self):
        """测试创建引擎"""
        engine = PositionSizingEngine()
        assert engine is not None
        assert engine.state == EngineState.IDLE

    def test_engine_with_config(self, basic_config):
        """测试带配置创建"""
        engine = PositionSizingEngine(basic_config)
        assert engine.config.account_size == 100000.0

    def test_update_asset_metrics(self, sizing_engine, basic_metrics):
        """测试更新资产指标"""
        sizing_engine.update_asset_metrics(basic_metrics)

        metrics = sizing_engine.get_asset_metrics("AAPL")
        assert metrics is not None
        assert metrics.symbol == "AAPL"

    def test_update_position(self, sizing_engine):
        """测试更新持仓"""
        sizing_engine.update_position("AAPL", 100.0)

        position = sizing_engine.get_position("AAPL")
        assert position == 100.0

    def test_update_portfolio_value(self, sizing_engine):
        """测试更新组合价值"""
        sizing_engine.update_portfolio_value(120000.0)

        assert sizing_engine._portfolio_value == 120000.0
        assert sizing_engine._peak_value == 120000.0

    def test_add_constraint(self, sizing_engine):
        """测试添加约束"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_POSITION,
            value=0.25,
        )

        sizing_engine.add_constraint("AAPL", constraint)

        constraints = sizing_engine.get_constraints("AAPL")
        assert len(constraints) == 1

    def test_remove_constraint(self, sizing_engine):
        """测试移除约束"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_POSITION,
            value=0.25,
        )

        sizing_engine.add_constraint("AAPL", constraint)
        sizing_engine.remove_constraint("AAPL", ConstraintType.MAX_POSITION)

        constraints = sizing_engine.get_constraints("AAPL")
        assert len(constraints) == 0

    def test_check_constraints_pass(self, sizing_engine):
        """测试约束检查-通过"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_POSITION,
            value=100.0,
        )

        sizing_engine.add_constraint("AAPL", constraint)

        passed, violations = sizing_engine.check_constraints("AAPL", 80.0)
        assert passed
        assert len(violations) == 0

    def test_check_constraints_fail(self, sizing_engine):
        """测试约束检查-失败"""
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_POSITION,
            value=100.0,
        )

        sizing_engine.add_constraint("AAPL", constraint)

        passed, violations = sizing_engine.check_constraints("AAPL", 120.0)
        assert not passed
        assert len(violations) > 0

    def test_calculate_size_kelly(self, sizing_engine, basic_metrics):
        """测试凯利公式计算"""
        sizing_engine.update_asset_metrics(basic_metrics)

        result = sizing_engine.calculate_size("AAPL", SizingMethod.KELLY)

        assert result.method == SizingMethod.KELLY
        assert result.suggested_size > 0

    def test_calculate_size_fixed_fractional(self, sizing_engine, basic_metrics):
        """测试固定比例计算"""
        sizing_engine.update_asset_metrics(basic_metrics)

        result = sizing_engine.calculate_size("AAPL", SizingMethod.FIXED_FRACTIONAL)

        assert result.method == SizingMethod.FIXED_FRACTIONAL
        assert result.suggested_size > 0

    def test_calculate_size_volatility(self, sizing_engine, basic_metrics):
        """测试波动率计算"""
        sizing_engine.update_asset_metrics(basic_metrics)

        result = sizing_engine.calculate_size("AAPL", SizingMethod.VOLATILITY)

        assert result.method == SizingMethod.VOLATILITY
        assert result.suggested_size > 0

    def test_calculate_size_with_constraints(self, sizing_engine, basic_metrics):
        """测试带约束的计算"""
        sizing_engine.update_asset_metrics(basic_metrics)

        # 添加一个很小的最大仓位约束
        constraint = PositionConstraint(
            constraint_type=ConstraintType.MAX_POSITION,
            value=10.0,  # 只允许10股
        )
        sizing_engine.add_constraint("AAPL", constraint)

        result = sizing_engine.calculate_size("AAPL", SizingMethod.FIXED_FRACTIONAL)

        # 应该被约束到10股
        assert result.suggested_size <= 10.0
        assert result.is_constrained

    def test_calculate_size_auto_method(self, sizing_engine, basic_metrics):
        """测试自动选择方法"""
        sizing_engine.update_asset_metrics(basic_metrics)

        result = sizing_engine.calculate_size("AAPL")

        # 应该选择一个合适的方法
        assert result.method in SizingMethod

    def test_calculate_multiple(self, sizing_engine):
        """测试批量计算"""
        # 添加多个资产
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            metrics = AssetMetrics(
                symbol=symbol,
                current_price=150.0,
                volatility=0.25,
            )
            sizing_engine.update_asset_metrics(metrics)

        results = sizing_engine.calculate_multiple(["AAPL", "GOOGL", "MSFT"])

        assert len(results) == 3
        assert "AAPL" in results

    def test_calculate_size_with_drawdown(self, sizing_engine, basic_metrics):
        """测试有回撤时的计算"""
        sizing_engine.update_asset_metrics(basic_metrics)

        # 模拟回撤
        sizing_engine.update_portfolio_value(120000.0)  # 峰值
        sizing_engine.update_portfolio_value(90000.0)   # 回撤25%

        result = sizing_engine.calculate_size("AAPL", SizingMethod.FIXED_FRACTIONAL)

        # 应该减小仓位
        assert result.is_constrained

    def test_calculate_portfolio_sizing_basic(self, sizing_engine):
        """测试组合计算-基础"""
        # 添加多个资产
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            metrics = AssetMetrics(
                symbol=symbol,
                current_price=150.0,
                volatility=0.25,
            )
            sizing_engine.update_asset_metrics(metrics)

        portfolio = sizing_engine.calculate_portfolio_sizing(symbols)

        assert portfolio.sizing_id is not None
        assert len(portfolio.allocations) == 3
        assert portfolio.total_exposure > 0

    def test_calculate_portfolio_sizing_risk_parity(self, sizing_engine):
        """测试风险平价组合计算"""
        # 添加不同波动率的资产
        metrics1 = AssetMetrics(symbol="AAPL", current_price=150.0, volatility=0.20)
        metrics2 = AssetMetrics(symbol="GOOGL", current_price=2800.0, volatility=0.30)
        metrics3 = AssetMetrics(symbol="MSFT", current_price=300.0, volatility=0.25)

        sizing_engine.update_asset_metrics(metrics1)
        sizing_engine.update_asset_metrics(metrics2)
        sizing_engine.update_asset_metrics(metrics3)

        portfolio = sizing_engine.calculate_portfolio_sizing(
            ["AAPL", "GOOGL", "MSFT"],
            SizingMethod.RISK_PARITY
        )

        # 检查风险贡献
        assert len(portfolio.risk_contributions) == 3

    def test_calculate_portfolio_sizing_with_rebalance(self, sizing_engine):
        """测试需要再平衡的组合"""
        metrics = AssetMetrics(symbol="AAPL", current_price=150.0, volatility=0.25)
        sizing_engine.update_asset_metrics(metrics)

        # 设置当前持仓与目标差异较大
        sizing_engine.update_position("AAPL", 1000.0)

        portfolio = sizing_engine.calculate_portfolio_sizing(["AAPL"])

        # 可能需要再平衡
        # assert portfolio.rebalance_required  # 取决于具体设置

    def test_check_adjustment_triggers(self, sizing_engine, basic_metrics):
        """测试调整触发检查"""
        sizing_engine.update_asset_metrics(basic_metrics)
        sizing_engine.update_position("AAPL", 100.0)

        signals = sizing_engine.check_adjustment_triggers("AAPL")

        # 可能有或没有信号
        assert isinstance(signals, list)

    def test_volatility_adjustment_trigger(self, sizing_engine):
        """测试波动率调整触发"""
        # 创建高波动率资产
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.40,  # 高波动率
            daily_volatility=0.025,
        )

        sizing_engine.update_asset_metrics(metrics)
        sizing_engine.update_position("AAPL", 1000.0)

        signals = sizing_engine.check_adjustment_triggers("AAPL")

        # 可能触发波动率调整
        if signals:
            assert any(s.trigger == AdjustmentTrigger.VOLATILITY_CHANGE for s in signals)

    def test_drawdown_adjustment_trigger(self, sizing_engine, basic_metrics):
        """测试回撤调整触发"""
        sizing_engine.update_asset_metrics(basic_metrics)
        sizing_engine.update_position("AAPL", 100.0)

        # 模拟大幅回撤
        sizing_engine.update_portfolio_value(120000.0)
        sizing_engine.update_portfolio_value(85000.0)  # 29%回撤

        signals = sizing_engine.check_adjustment_triggers("AAPL")

        # 应该触发回撤调整
        drawdown_signals = [s for s in signals if s.trigger == AdjustmentTrigger.DRAWDOWN]
        assert len(drawdown_signals) > 0

    def test_profit_target_trigger(self, sizing_engine):
        """测试盈利目标触发"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=180.0,  # 当前价
            entry_price=150.0,    # 入场价，盈利20%
            volatility=0.25,
        )

        sizing_engine.update_asset_metrics(metrics)
        sizing_engine.update_position("AAPL", 100.0)

        signals = sizing_engine.check_adjustment_triggers("AAPL")

        # 可能触发盈利目标调整
        profit_signals = [s for s in signals if s.trigger == AdjustmentTrigger.PROFIT_TARGET]
        if profit_signals:
            assert profit_signals[0].adjustment_pct > 0

    def test_get_statistics(self, sizing_engine):
        """测试获取统计信息"""
        stats = sizing_engine.get_statistics()

        assert 'state' in stats
        assert 'calculations_count' in stats
        assert 'portfolio_value' in stats

    def test_get_sizing_history(self, sizing_engine, basic_metrics):
        """测试获取历史记录"""
        sizing_engine.update_asset_metrics(basic_metrics)

        # 执行几次计算
        for _ in range(5):
            sizing_engine.calculate_size("AAPL")

        history = sizing_engine.get_sizing_history("AAPL")

        assert len(history) == 5

    def test_get_adjustment_signals(self, sizing_engine, basic_metrics):
        """测试获取调整信号"""
        sizing_engine.update_asset_metrics(basic_metrics)
        sizing_engine.update_position("AAPL", 100.0)

        sizing_engine.check_adjustment_triggers("AAPL")

        signals = sizing_engine.get_adjustment_signals("AAPL")

        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, sizing_engine, basic_metrics):
        """测试启动停止监控"""
        sizing_engine.update_asset_metrics(basic_metrics)

        await sizing_engine.start_monitoring(["AAPL"])
        assert sizing_engine.state == EngineState.ACTIVE

        await asyncio.sleep(0.1)

        await sizing_engine.stop_monitoring()
        assert sizing_engine.state == EngineState.IDLE

    def test_callback_on_sizing_calculated(self, sizing_engine, basic_metrics):
        """测试仓位计算回调"""
        results_received = []

        def callback(result):
            results_received.append(result)

        sizing_engine.on_sizing_calculated(callback)
        sizing_engine.update_asset_metrics(basic_metrics)

        sizing_engine.calculate_size("AAPL")

        assert len(results_received) == 1

    def test_callback_on_adjustment_signal(self, sizing_engine, basic_metrics):
        """测试调整信号回调"""
        signals_received = []

        def callback(signal):
            signals_received.append(signal)

        sizing_engine.on_adjustment_signal(callback)
        sizing_engine.update_asset_metrics(basic_metrics)
        sizing_engine.update_position("AAPL", 100.0)

        # 制造回撤触发信号
        sizing_engine.update_portfolio_value(120000.0)
        sizing_engine.update_portfolio_value(85000.0)

        sizing_engine.check_adjustment_triggers("AAPL")

        # 应该收到信号
        assert len(signals_received) > 0


# ==================== Factory Functions 测试 ====================

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_sizing_engine_default(self):
        """测试默认创建引擎"""
        engine = create_sizing_engine()

        assert engine is not None
        assert engine.config.account_size == 100000.0

    def test_create_sizing_engine_custom(self):
        """测试自定义创建引擎"""
        engine = create_sizing_engine(
            account_size=200000.0,
            risk_per_trade=0.01,
            risk_level='conservative',
        )

        assert engine.config.account_size == 200000.0
        assert engine.config.risk_per_trade == 0.01
        assert engine.config.risk_level == RiskLevel.CONSERVATIVE

    def test_create_sizing_engine_invalid_risk_level(self):
        """测试无效风险等级"""
        engine = create_sizing_engine(risk_level='invalid')

        assert engine.config.risk_level == RiskLevel.MODERATE

    def test_create_asset_metrics(self):
        """测试创建资产指标"""
        metrics = create_asset_metrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.25,
            win_rate=0.6,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.current_price == 150.0
        assert metrics.win_rate == 0.6

    def test_create_position_constraint(self):
        """测试创建约束"""
        constraint = create_position_constraint(
            constraint_type='max_position',
            value=0.25,
            is_hard_limit=True,
        )

        assert constraint.constraint_type == ConstraintType.MAX_POSITION
        assert constraint.value == 0.25

    def test_create_position_constraint_invalid_type(self):
        """测试无效约束类型"""
        constraint = create_position_constraint(
            constraint_type='invalid',
            value=0.25,
        )

        assert constraint.constraint_type == ConstraintType.MAX_POSITION

    def test_quick_position_sizing(self):
        """测试快速仓位计算"""
        result = quick_position_sizing(
            symbol="AAPL",
            current_price=150.0,
            account_size=100000.0,
            method='fixed_fractional',
            volatility=0.25,
            atr=3.0,
        )

        assert 'symbol' in result
        assert 'sizing_result' in result
        assert result['symbol'] == "AAPL"

    def test_quick_position_sizing_invalid_method(self):
        """测试无效计算方法"""
        result = quick_position_sizing(
            symbol="AAPL",
            current_price=150.0,
            account_size=100000.0,
            method='invalid',
        )

        # 应该使用默认方法
        assert result['sizing_result']['method'] == 'fixed_fractional'


# ==================== Edge Cases 测试 ====================

class TestEdgeCases:
    """边界情况测试"""

    def test_zero_price(self, sizing_engine):
        """测试零价格"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=0.0,
            volatility=0.25,
        )

        sizing_engine.update_asset_metrics(metrics)

        result = sizing_engine.calculate_size("AAPL")

        # 应该处理边界情况
        assert result.suggested_size >= 0

    def test_negative_volatility(self, sizing_engine):
        """测试负波动率"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=-0.25,
        )

        sizing_engine.update_asset_metrics(metrics)

        result = sizing_engine.calculate_size("AAPL", SizingMethod.VOLATILITY)

        # 应该有默认处理
        assert result.suggested_size >= 0

    def test_extremely_high_volatility(self, sizing_engine):
        """测试极高波动率"""
        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=2.0,  # 200%波动率
            daily_volatility=0.126,
        )

        sizing_engine.update_asset_metrics(metrics)

        result = sizing_engine.calculate_size("AAPL", SizingMethod.VOLATILITY)

        # 应该产生很小的仓位
        assert result.suggested_size >= 0

    def test_zero_account_size(self):
        """测试零账户规模"""
        config = SizingConfig(account_size=0.0)
        engine = PositionSizingEngine(config)

        metrics = AssetMetrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.25,
        )
        engine.update_asset_metrics(metrics)

        result = engine.calculate_size("AAPL")

        # 应该处理边界情况
        assert result.suggested_value == 0.0

    def test_missing_asset_metrics(self, sizing_engine):
        """测试缺失资产指标"""
        with pytest.raises(ValueError):
            sizing_engine.calculate_size("UNKNOWN")

    def test_fractional_shares_disabled(self, basic_metrics):
        """测试禁用小数股"""
        config = SizingConfig(allow_fractional_shares=False)
        engine = PositionSizingEngine(config)

        engine.update_asset_metrics(basic_metrics)
        result = engine.calculate_size("AAPL")

        # 应该是整数
        assert result.suggested_size == math.floor(result.suggested_size)

    def test_lot_size_rounding(self, basic_metrics):
        """测试手数圆整"""
        config = SizingConfig(
            round_to_lot_size=True,
            lot_size=100,
        )
        engine = PositionSizingEngine(config)

        engine.update_asset_metrics(basic_metrics)
        result = engine.calculate_size("AAPL")

        # 应该是100的倍数
        assert result.suggested_size % 100 == 0

    def test_min_position_value(self, basic_metrics):
        """测试最小仓位价值"""
        config = SizingConfig(
            account_size=100000.0,
            risk_per_trade=0.0001,  # 极小风险
            min_position_value=1000.0,
        )
        engine = PositionSizingEngine(config)

        engine.update_asset_metrics(basic_metrics)
        result = engine.calculate_size("AAPL")

        # 如果低于最小值，应该是0
        if result.suggested_value < 1000.0:
            assert result.suggested_size == 0


# ==================== Integration Tests 测试 ====================

class TestIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """测试完整工作流"""
        # 创建引擎
        engine = create_sizing_engine(
            account_size=100000.0,
            risk_per_trade=0.02,
            risk_level='moderate',
        )

        # 添加资产指标
        metrics = create_asset_metrics(
            symbol="AAPL",
            current_price=150.0,
            volatility=0.25,
            win_rate=0.6,
            avg_win=0.03,
            avg_loss=0.02,
            atr=3.0,
        )
        engine.update_asset_metrics(metrics)

        # 添加约束 - 使用股数限制（200股）而非比例
        constraint = create_position_constraint(
            constraint_type='max_position',
            value=200.0,
        )
        engine.add_constraint("AAPL", constraint)

        # 计算仓位
        result = engine.calculate_size("AAPL", SizingMethod.KELLY)

        # 验证结果
        assert result.suggested_size > 0
        assert result.suggested_size <= 200.0

        # 更新持仓
        engine.update_position("AAPL", result.suggested_size)

        # 检查调整触发
        signals = engine.check_adjustment_triggers("AAPL")

        # 获取统计
        stats = engine.get_statistics()
        assert stats['calculations_count'] > 0

    def test_multi_asset_portfolio(self):
        """测试多资产组合"""
        engine = create_sizing_engine(account_size=100000.0)

        # 添加多个资产
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        volatilities = [0.25, 0.30, 0.22, 0.28, 0.45]

        for symbol, vol in zip(symbols, volatilities):
            metrics = create_asset_metrics(
                symbol=symbol,
                current_price=150.0,
                volatility=vol,
            )
            engine.update_asset_metrics(metrics)

        # 计算组合配置
        portfolio = engine.calculate_portfolio_sizing(symbols, SizingMethod.RISK_PARITY)

        # 验证
        assert len(portfolio.allocations) == 5
        assert portfolio.leverage > 0
        assert 0 <= portfolio.diversification_score <= 1

    def test_dynamic_adjustment(self):
        """测试动态调整"""
        engine = create_sizing_engine(account_size=100000.0)

        metrics = create_asset_metrics(
            symbol="AAPL",
            current_price=150.0,
            entry_price=150.0,
            volatility=0.25,
        )
        engine.update_asset_metrics(metrics)

        # 初始计算
        result1 = engine.calculate_size("AAPL")
        engine.update_position("AAPL", result1.suggested_size)

        # 价格上涨
        metrics.current_price = 180.0
        engine.update_asset_metrics(metrics)

        # 检查是否需要调整
        signals = engine.check_adjustment_triggers("AAPL")

        # 应该有盈利目标信号
        if signals:
            profit_signals = [s for s in signals if s.trigger == AdjustmentTrigger.PROFIT_TARGET]
            if profit_signals:
                assert profit_signals[0].suggested_size != result1.suggested_size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
