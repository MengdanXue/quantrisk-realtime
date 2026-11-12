"""
Greeks Calculator 测试
"""

import pytest
import asyncio
import math
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.greeks_calculator import (
    # Enums
    OptionType,
    OptionStyle,
    PricingModel,
    GreekType,
    AlertSeverity,
    CalculatorState,
    # Data Classes
    OptionPosition,
    MarketData,
    Greeks,
    OptionPricing,
    PositionGreeks,
    PortfolioGreeks,
    ScenarioResult,
    PnLAttribution,
    GreeksAlert,
    GreeksLimit,
    CalculatorConfig,
    # Providers
    MarketDataProvider,
    PositionProvider,
    SimulatedMarketDataProvider,
    SimulatedPositionProvider,
    # Pricing Models
    BlackScholesPricer,
    # Main Class
    GreeksCalculator,
    # Factory Functions
    create_greeks_calculator,
    create_option_position,
    create_market_data,
    quick_greeks,
    # Utilities
    norm_cdf,
    norm_pdf,
)


# ============ Test Utilities ============


class TestNormFunctions:
    """正态分布函数测试"""

    def test_norm_cdf_zero(self):
        """测试N(0) = 0.5"""
        assert abs(norm_cdf(0) - 0.5) < 0.0001

    def test_norm_cdf_positive(self):
        """测试正数CDF"""
        assert norm_cdf(1.0) > 0.5
        assert norm_cdf(2.0) > norm_cdf(1.0)

    def test_norm_cdf_negative(self):
        """测试负数CDF"""
        assert norm_cdf(-1.0) < 0.5
        assert norm_cdf(-1.0) + norm_cdf(1.0) - 1.0 < 0.0001  # 对称性

    def test_norm_pdf_zero(self):
        """测试PDF最大值在0"""
        assert norm_pdf(0) > norm_pdf(1)
        assert norm_pdf(0) > norm_pdf(-1)

    def test_norm_pdf_symmetric(self):
        """测试PDF对称性"""
        assert abs(norm_pdf(1.0) - norm_pdf(-1.0)) < 0.0001


# ============ Enum Tests ============


class TestEnums:
    """枚举测试"""

    def test_option_type(self):
        assert OptionType.CALL.value == "call"
        assert OptionType.PUT.value == "put"

    def test_option_style(self):
        assert OptionStyle.EUROPEAN.value == "european"
        assert OptionStyle.AMERICAN.value == "american"

    def test_pricing_model(self):
        assert PricingModel.BLACK_SCHOLES.value == "black_scholes"
        assert PricingModel.BINOMIAL.value == "binomial"

    def test_greek_type(self):
        assert GreekType.DELTA.value == "delta"
        assert GreekType.VANNA.value == "vanna"

    def test_calculator_state(self):
        assert CalculatorState.IDLE.value == "idle"
        assert CalculatorState.RUNNING.value == "running"


# ============ Data Class Tests ============


class TestOptionPosition:
    """期权持仓测试"""

    def test_create_option_position(self):
        expiry = datetime.now() + timedelta(days=30)
        position = OptionPosition(
            position_id="POS001",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=expiry,
            quantity=10,
            multiplier=100.0,
        )

        assert position.position_id == "POS001"
        assert position.symbol == "AAPL"
        assert position.option_type == OptionType.CALL
        assert position.quantity == 10

    def test_option_position_to_dict(self):
        expiry = datetime.now() + timedelta(days=30)
        position = OptionPosition(
            position_id="POS001",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=expiry,
            quantity=10,
        )

        d = position.to_dict()
        assert d["position_id"] == "POS001"
        assert d["option_type"] == "call"


class TestMarketData:
    """市场数据测试"""

    def test_create_market_data(self):
        data = MarketData(
            symbol="AAPL",
            spot_price=150.0,
            volatility=0.25,
            risk_free_rate=0.05,
            dividend_yield=0.01,
        )

        assert data.symbol == "AAPL"
        assert data.spot_price == 150.0
        assert data.volatility == 0.25

    def test_market_data_to_dict(self):
        data = MarketData(
            symbol="AAPL",
            spot_price=150.0,
            volatility=0.25,
            risk_free_rate=0.05,
        )

        d = data.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["spot_price"] == 150.0


class TestGreeks:
    """希腊字母测试"""

    def test_create_greeks(self):
        greeks = Greeks(
            delta=0.5,
            gamma=0.05,
            theta=-0.02,
            vega=0.15,
            rho=0.10,
        )

        assert greeks.delta == 0.5
        assert greeks.gamma == 0.05
        assert greeks.theta == -0.02

    def test_greeks_scale(self):
        greeks = Greeks(delta=0.5, gamma=0.05)
        scaled = greeks.scale(100)

        assert scaled.delta == 50.0
        assert scaled.gamma == 5.0

    def test_greeks_add(self):
        g1 = Greeks(delta=0.5, gamma=0.05)
        g2 = Greeks(delta=0.3, gamma=0.03)
        result = g1.add(g2)

        assert result.delta == 0.8
        assert result.gamma == 0.08

    def test_greeks_to_dict(self):
        greeks = Greeks(delta=0.5, vega=0.15)
        d = greeks.to_dict()

        assert d["delta"] == 0.5
        assert d["vega"] == 0.15


class TestOptionPricing:
    """期权定价结果测试"""

    def test_create_option_pricing(self):
        greeks = Greeks(delta=0.5)
        pricing = OptionPricing(
            theoretical_price=5.0,
            intrinsic_value=2.0,
            time_value=3.0,
            greeks=greeks,
            model=PricingModel.BLACK_SCHOLES,
        )

        assert pricing.theoretical_price == 5.0
        assert pricing.time_value == 3.0

    def test_option_pricing_to_dict(self):
        greeks = Greeks(delta=0.5)
        pricing = OptionPricing(
            theoretical_price=5.0,
            intrinsic_value=2.0,
            time_value=3.0,
            greeks=greeks,
            model=PricingModel.BLACK_SCHOLES,
        )

        d = pricing.to_dict()
        assert d["theoretical_price"] == 5.0
        assert d["model"] == "black_scholes"


class TestGreeksAlert:
    """希腊字母警报测试"""

    def test_create_greeks_alert(self):
        alert = GreeksAlert(
            alert_id="ALERT001",
            position_id="POS001",
            greek_type=GreekType.DELTA,
            current_value=1500.0,
            threshold=1000.0,
            severity=AlertSeverity.HIGH,
            message="Delta exceeds limit",
        )

        assert alert.alert_id == "ALERT001"
        assert alert.greek_type == GreekType.DELTA
        assert alert.severity == AlertSeverity.HIGH

    def test_greeks_alert_to_dict(self):
        alert = GreeksAlert(
            alert_id="ALERT001",
            position_id=None,
            greek_type=GreekType.GAMMA,
            current_value=150.0,
            threshold=100.0,
            severity=AlertSeverity.MEDIUM,
            message="Gamma warning",
        )

        d = alert.to_dict()
        assert d["greek_type"] == "gamma"
        assert d["severity"] == "medium"


class TestGreeksLimit:
    """希腊字母限额测试"""

    def test_create_greeks_limit(self):
        limit = GreeksLimit(
            greek_type=GreekType.DELTA,
            max_value=1000.0,
            min_value=-1000.0,
            warning_threshold=0.8,
        )

        assert limit.greek_type == GreekType.DELTA
        assert limit.max_value == 1000.0

    def test_greeks_limit_to_dict(self):
        limit = GreeksLimit(
            greek_type=GreekType.VEGA,
            max_value=5000.0,
        )

        d = limit.to_dict()
        assert d["greek_type"] == "vega"
        assert d["max_value"] == 5000.0


class TestCalculatorConfig:
    """配置测试"""

    def test_default_config(self):
        config = CalculatorConfig()

        assert config.pricing_model == PricingModel.BLACK_SCHOLES
        assert config.calculate_higher_order is True

    def test_custom_config(self):
        config = CalculatorConfig(
            pricing_model=PricingModel.BINOMIAL,
            calculate_higher_order=False,
            delta_limit=500.0,
        )

        assert config.pricing_model == PricingModel.BINOMIAL
        assert config.delta_limit == 500.0


# ============ Black-Scholes Pricer Tests ============


class TestBlackScholesPricer:
    """Black-Scholes定价器测试"""

    def test_call_price_atm(self):
        """测试平值看涨期权价格"""
        price = BlackScholesPricer.price(
            OptionType.CALL,
            S=100.0, K=100.0, T=1.0,
            r=0.05, q=0.0, sigma=0.20
        )

        # ATM期权应该有正价值
        assert price > 0
        assert price < 100  # 不会超过现货价格

    def test_put_price_atm(self):
        """测试平值看跌期权价格"""
        price = BlackScholesPricer.price(
            OptionType.PUT,
            S=100.0, K=100.0, T=1.0,
            r=0.05, q=0.0, sigma=0.20
        )

        assert price > 0

    def test_call_put_parity(self):
        """测试Call-Put平价关系"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20

        call_price = BlackScholesPricer.price(OptionType.CALL, S, K, T, r, q, sigma)
        put_price = BlackScholesPricer.price(OptionType.PUT, S, K, T, r, q, sigma)

        # C - P = S*exp(-qT) - K*exp(-rT)
        lhs = call_price - put_price
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

        assert abs(lhs - rhs) < 0.01

    def test_deep_itm_call(self):
        """测试深度实值看涨期权"""
        price = BlackScholesPricer.price(
            OptionType.CALL,
            S=150.0, K=100.0, T=0.5,
            r=0.05, q=0.0, sigma=0.20
        )

        # 应该接近内在价值
        intrinsic = 150.0 - 100.0
        assert price >= intrinsic

    def test_deep_otm_call(self):
        """测试深度虚值看涨期权"""
        price = BlackScholesPricer.price(
            OptionType.CALL,
            S=50.0, K=100.0, T=0.5,
            r=0.05, q=0.0, sigma=0.20
        )

        # 应该很小但为正
        assert price >= 0
        assert price < 1

    def test_expired_option(self):
        """测试到期期权"""
        call_itm = BlackScholesPricer.price(OptionType.CALL, 110, 100, 0, 0.05, 0, 0.2)
        call_otm = BlackScholesPricer.price(OptionType.CALL, 90, 100, 0, 0.05, 0, 0.2)

        assert call_itm == 10.0  # 内在价值
        assert call_otm == 0.0

    def test_greeks_call(self):
        """测试看涨期权希腊字母"""
        greeks = BlackScholesPricer.greeks(
            OptionType.CALL,
            S=100.0, K=100.0, T=1.0,
            r=0.05, q=0.0, sigma=0.20,
        )

        # Delta应该在0到1之间
        assert 0 < greeks.delta < 1
        # ATM Delta应该接近0.5
        assert 0.4 < greeks.delta < 0.7

        # Gamma应该为正
        assert greeks.gamma > 0

        # Theta应该为负（时间衰减）
        assert greeks.theta < 0

        # Vega应该为正
        assert greeks.vega > 0

    def test_greeks_put(self):
        """测试看跌期权希腊字母"""
        greeks = BlackScholesPricer.greeks(
            OptionType.PUT,
            S=100.0, K=100.0, T=1.0,
            r=0.05, q=0.0, sigma=0.20,
        )

        # Put Delta应该在-1到0之间
        assert -1 < greeks.delta < 0

        # Gamma对Call和Put相同
        call_greeks = BlackScholesPricer.greeks(
            OptionType.CALL, 100, 100, 1.0, 0.05, 0, 0.20
        )
        assert abs(greeks.gamma - call_greeks.gamma) < 0.001

    def test_higher_order_greeks(self):
        """测试高阶希腊字母"""
        greeks = BlackScholesPricer.greeks(
            OptionType.CALL,
            S=100.0, K=100.0, T=1.0,
            r=0.05, q=0.0, sigma=0.20,
            calculate_higher_order=True,
        )

        # 高阶希腊字母应该被计算
        assert greeks.vanna != 0
        assert greeks.volga != 0
        assert greeks.charm != 0

    def test_delta_bounds(self):
        """测试Delta边界"""
        # 深度实值Call的Delta接近1
        itm_greeks = BlackScholesPricer.greeks(
            OptionType.CALL, 200, 100, 0.5, 0.05, 0, 0.20
        )
        assert itm_greeks.delta > 0.95

        # 深度虚值Call的Delta接近0
        otm_greeks = BlackScholesPricer.greeks(
            OptionType.CALL, 50, 100, 0.5, 0.05, 0, 0.20
        )
        assert otm_greeks.delta < 0.05


# ============ Provider Tests ============


class TestSimulatedMarketDataProvider:
    """模拟市场数据提供者测试"""

    @pytest.mark.asyncio
    async def test_set_and_get_market_data(self):
        provider = SimulatedMarketDataProvider()

        data = MarketData(
            symbol="AAPL",
            spot_price=150.0,
            volatility=0.25,
            risk_free_rate=0.05,
        )
        provider.set_market_data("AAPL", data)

        result = await provider.get_market_data("AAPL")
        assert result is not None
        assert result.spot_price == 150.0

    @pytest.mark.asyncio
    async def test_get_spot_price(self):
        provider = SimulatedMarketDataProvider()

        data = MarketData(
            symbol="AAPL",
            spot_price=150.0,
            volatility=0.25,
            risk_free_rate=0.05,
        )
        provider.set_market_data("AAPL", data)

        price = await provider.get_spot_price("AAPL")
        assert price == 150.0

    @pytest.mark.asyncio
    async def test_default_spot_price(self):
        provider = SimulatedMarketDataProvider()

        price = await provider.get_spot_price("UNKNOWN")
        assert price == 100.0


class TestSimulatedPositionProvider:
    """模拟持仓提供者测试"""

    @pytest.mark.asyncio
    async def test_add_and_get_positions(self):
        provider = SimulatedPositionProvider()

        position = OptionPosition(
            position_id="POS001",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            quantity=10,
        )
        provider.add_position(position)

        positions = await provider.get_positions()
        assert len(positions) == 1
        assert positions[0].position_id == "POS001"

    @pytest.mark.asyncio
    async def test_get_single_position(self):
        provider = SimulatedPositionProvider()

        position = OptionPosition(
            position_id="POS001",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            quantity=10,
        )
        provider.add_position(position)

        result = await provider.get_position("POS001")
        assert result is not None
        assert result.quantity == 10

    @pytest.mark.asyncio
    async def test_remove_position(self):
        provider = SimulatedPositionProvider()

        position = OptionPosition(
            position_id="POS001",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            quantity=10,
        )
        provider.add_position(position)
        provider.remove_position("POS001")

        positions = await provider.get_positions()
        assert len(positions) == 0


# ============ Calculator Tests ============


@pytest.fixture
def market_provider():
    """市场数据提供者fixture"""
    provider = SimulatedMarketDataProvider()
    provider.set_market_data("AAPL", MarketData(
        symbol="AAPL",
        spot_price=150.0,
        volatility=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.01,
    ))
    return provider


@pytest.fixture
def position_provider():
    """持仓提供者fixture"""
    provider = SimulatedPositionProvider()
    provider.add_position(OptionPosition(
        position_id="POS001",
        symbol="AAPL",
        option_symbol="AAPL_C150",
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
        strike=150.0,
        expiry=datetime.now() + timedelta(days=60),
        quantity=10,
        multiplier=100.0,
    ))
    provider.add_position(OptionPosition(
        position_id="POS002",
        symbol="AAPL",
        option_symbol="AAPL_P145",
        option_type=OptionType.PUT,
        option_style=OptionStyle.EUROPEAN,
        strike=145.0,
        expiry=datetime.now() + timedelta(days=60),
        quantity=-5,  # 卖出
        multiplier=100.0,
    ))
    return provider


@pytest.fixture
def calculator(market_provider, position_provider):
    """计算器fixture"""
    config = CalculatorConfig(
        update_interval_ms=100,
        calculate_higher_order=True,
    )
    return GreeksCalculator(
        config=config,
        market_provider=market_provider,
        position_provider=position_provider,
    )


class TestGreeksCalculator:
    """希腊字母计算器测试"""

    @pytest.mark.asyncio
    async def test_start_stop(self, calculator):
        """测试启动停止"""
        await calculator.start()
        assert calculator._running
        assert calculator.get_state() == CalculatorState.RUNNING

        await calculator.stop()
        assert not calculator._running
        assert calculator.get_state() == CalculatorState.IDLE

    def test_price_option(self, calculator):
        """测试期权定价"""
        pricing = calculator.price_option(
            OptionType.CALL,
            spot=150.0,
            strike=150.0,
            time_to_expiry=60/365,
            volatility=0.25,
            risk_free_rate=0.05,
        )

        assert pricing.theoretical_price > 0
        assert pricing.greeks.delta > 0
        assert pricing.model == PricingModel.BLACK_SCHOLES

    @pytest.mark.asyncio
    async def test_price_position(self, calculator):
        """测试持仓定价"""
        position = OptionPosition(
            position_id="TEST",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=60),
            quantity=10,
            multiplier=100.0,
        )

        pg = await calculator.price_position(position)

        assert pg.position_id == "TEST"
        assert pg.quantity == 10
        assert pg.position_greeks.delta != 0
        assert pg.notional_value > 0

    @pytest.mark.asyncio
    async def test_portfolio_greeks(self, calculator):
        """测试组合希腊字母"""
        await calculator.start()
        await asyncio.sleep(0.3)

        portfolio = calculator.get_portfolio_greeks()
        assert portfolio is not None
        assert len(portfolio.positions) == 2
        assert portfolio.total_delta_equivalent != 0

        await calculator.stop()

    @pytest.mark.asyncio
    async def test_greeks_by_underlying(self, calculator):
        """测试按标的汇总"""
        await calculator.start()
        await asyncio.sleep(0.3)

        portfolio = calculator.get_portfolio_greeks()
        assert "AAPL" in portfolio.by_underlying

        await calculator.stop()

    @pytest.mark.asyncio
    async def test_greeks_by_expiry(self, calculator):
        """测试按到期日汇总"""
        await calculator.start()
        await asyncio.sleep(0.3)

        portfolio = calculator.get_portfolio_greeks()
        assert len(portfolio.by_expiry) > 0

        await calculator.stop()

    @pytest.mark.asyncio
    async def test_get_statistics(self, calculator):
        """测试获取统计"""
        await calculator.start()
        await asyncio.sleep(0.3)

        stats = calculator.get_statistics()
        assert stats["state"] == "running"
        assert stats["positions_tracked"] > 0

        await calculator.stop()


class TestScenarioAnalysis:
    """情景分析测试"""

    @pytest.mark.asyncio
    async def test_analyze_spot_change(self, calculator, market_provider):
        """测试现货变化情景"""
        position = OptionPosition(
            position_id="TEST",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=60),
            quantity=10,
            multiplier=100.0,
        )

        market_data = await market_provider.get_market_data("AAPL")

        result = calculator.analyze_scenario(
            position, market_data,
            spot_change_pct=0.10,  # +10%
        )

        assert result.spot_change == 0.10
        assert result.price_change > 0  # Call价格应该上涨
        assert result.pnl > 0

    @pytest.mark.asyncio
    async def test_analyze_vol_change(self, calculator, market_provider):
        """测试波动率变化情景"""
        position = OptionPosition(
            position_id="TEST",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=60),
            quantity=10,
            multiplier=100.0,
        )

        market_data = await market_provider.get_market_data("AAPL")

        result = calculator.analyze_scenario(
            position, market_data,
            vol_change=0.05,  # +5%波动率
        )

        assert result.vol_change == 0.05
        assert result.price_change > 0  # 波动率上升，期权价格上涨

    @pytest.mark.asyncio
    async def test_analyze_time_decay(self, calculator, market_provider):
        """测试时间衰减情景"""
        position = OptionPosition(
            position_id="TEST",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=60),
            quantity=10,
            multiplier=100.0,
        )

        market_data = await market_provider.get_market_data("AAPL")

        result = calculator.analyze_scenario(
            position, market_data,
            time_change_days=7,  # 7天后
        )

        assert result.time_change == 7
        assert result.price_change < 0  # 时间衰减导致价格下跌

    @pytest.mark.asyncio
    async def test_stress_test(self, calculator, market_provider):
        """测试压力测试"""
        position = OptionPosition(
            position_id="TEST",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=60),
            quantity=10,
            multiplier=100.0,
        )

        market_data = await market_provider.get_market_data("AAPL")

        results = calculator.run_stress_test(position, market_data)

        assert len(results) > 10  # 应该有多个情景
        # 检查上涨情景
        up_scenario = [r for r in results if r.spot_change > 0.1][0]
        assert up_scenario.pnl > 0


class TestPnLAttribution:
    """P&L归因测试"""

    @pytest.mark.asyncio
    async def test_pnl_attribution(self, calculator):
        """测试P&L归因"""
        position = OptionPosition(
            position_id="TEST",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=60),
            quantity=10,
            multiplier=100.0,
        )

        old_market = MarketData(
            symbol="AAPL",
            spot_price=150.0,
            volatility=0.25,
            risk_free_rate=0.05,
        )

        new_market = MarketData(
            symbol="AAPL",
            spot_price=155.0,  # +5
            volatility=0.27,   # +2%
            risk_free_rate=0.05,
        )

        attribution = calculator.attribute_pnl(
            position,
            old_market,
            new_market,
            old_price=8.0,
            new_price=11.0,
            time_passed_days=1,
        )

        assert attribution.total_pnl != 0
        assert attribution.delta_pnl > 0  # 现货上涨，Delta贡献为正
        assert attribution.vega_pnl > 0   # 波动率上涨，Vega贡献为正
        assert attribution.theta_pnl < 0  # 时间衰减

    @pytest.mark.asyncio
    async def test_pnl_attribution_to_dict(self, calculator):
        """测试P&L归因序列化"""
        position = OptionPosition(
            position_id="TEST",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=60),
            quantity=10,
            multiplier=100.0,
        )

        old_market = MarketData("AAPL", 150.0, 0.25, 0.05)
        new_market = MarketData("AAPL", 155.0, 0.25, 0.05)

        attribution = calculator.attribute_pnl(
            position, old_market, new_market, 8.0, 10.0, 1
        )

        d = attribution.to_dict()
        assert "total_pnl" in d
        assert "delta_pnl" in d


class TestDeltaHedging:
    """Delta对冲测试"""

    @pytest.mark.asyncio
    async def test_calculate_delta_hedge(self, calculator):
        """测试计算Delta对冲"""
        await calculator.start()
        await asyncio.sleep(0.3)

        portfolio = calculator.get_portfolio_greeks()
        hedge_orders = calculator.calculate_delta_hedge(portfolio)

        # 应该有AAPL的对冲订单
        assert "AAPL" in hedge_orders

        await calculator.stop()

    @pytest.mark.asyncio
    async def test_gamma_scalping_levels(self, calculator, market_provider):
        """测试Gamma Scalping水平"""
        position = OptionPosition(
            position_id="TEST",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type=OptionType.CALL,
            option_style=OptionStyle.EUROPEAN,
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            quantity=10,
            multiplier=100.0,
        )

        market_data = await market_provider.get_market_data("AAPL")

        levels = calculator.calculate_gamma_scalping_levels(
            position, market_data, gamma_threshold=0.1
        )

        assert "current_price" in levels
        assert "upper_rehedge" in levels
        assert "lower_rehedge" in levels
        assert levels["upper_rehedge"] > levels["current_price"]
        assert levels["lower_rehedge"] < levels["current_price"]


class TestLimitsAndAlerts:
    """限额和警报测试"""

    @pytest.mark.asyncio
    async def test_set_limit(self, calculator):
        """测试设置限额"""
        limit = GreeksLimit(
            greek_type=GreekType.DELTA,
            max_value=500.0,
            min_value=-500.0,
        )
        calculator.set_limit(limit)

        result = calculator.get_limit(GreekType.DELTA)
        assert result is not None
        assert result.max_value == 500.0

    @pytest.mark.asyncio
    async def test_alert_triggered(self, calculator, position_provider):
        """测试触发警报"""
        # 添加大量持仓
        for i in range(20):
            position_provider.add_position(OptionPosition(
                position_id=f"POS_{i}",
                symbol="AAPL",
                option_symbol=f"AAPL_C150_{i}",
                option_type=OptionType.CALL,
                option_style=OptionStyle.EUROPEAN,
                strike=150.0,
                expiry=datetime.now() + timedelta(days=60),
                quantity=100,  # 大量多头
                multiplier=100.0,
            ))

        # 设置低限额
        calculator.set_limit(GreeksLimit(
            greek_type=GreekType.DELTA,
            max_value=100.0,  # 很低的限额
        ))

        received_alerts = []

        def alert_callback(alert):
            received_alerts.append(alert)

        calculator.on_alert(alert_callback)

        await calculator.start()
        await asyncio.sleep(0.3)
        await calculator.stop()

        # 应该触发警报
        assert len(received_alerts) > 0
        assert received_alerts[0].greek_type == GreekType.DELTA


class TestCallbacks:
    """回调测试"""

    @pytest.mark.asyncio
    async def test_greeks_callback(self, calculator):
        """测试希腊字母回调"""
        received = []

        def callback(greeks):
            received.append(greeks)

        calculator.on_greeks_update(callback)

        await calculator.start()
        await asyncio.sleep(0.3)
        await calculator.stop()

        assert len(received) > 0
        assert received[0].total_greeks is not None


class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_greeks_calculator(self):
        """测试创建计算器"""
        calc = create_greeks_calculator(
            pricing_model="black_scholes",
            calculate_higher_order=False,
        )

        assert calc._config.pricing_model == PricingModel.BLACK_SCHOLES
        assert calc._config.calculate_higher_order is False

    def test_create_option_position_factory(self):
        """测试创建持仓"""
        position = create_option_position(
            position_id="POS001",
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type="call",
            strike=150.0,
            expiry=datetime.now() + timedelta(days=30),
            quantity=10,
        )

        assert position.option_type == OptionType.CALL
        assert position.quantity == 10

    def test_create_market_data_factory(self):
        """测试创建市场数据"""
        data = create_market_data(
            symbol="AAPL",
            spot_price=150.0,
            volatility=0.25,
        )

        assert data.symbol == "AAPL"
        assert data.spot_price == 150.0

    def test_quick_greeks(self):
        """测试快速希腊字母计算"""
        greeks = quick_greeks(
            option_type="call",
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            volatility=0.20,
        )

        assert "delta" in greeks
        assert "gamma" in greeks
        assert 0 < greeks["delta"] < 1


class TestEdgeCases:
    """边界情况测试"""

    def test_expired_option_greeks(self, calculator):
        """测试到期期权希腊字母"""
        pricing = calculator.price_option(
            OptionType.CALL,
            spot=110.0,
            strike=100.0,
            time_to_expiry=0,  # 到期
            volatility=0.25,
            risk_free_rate=0.05,
        )

        assert pricing.theoretical_price == 10.0  # 内在价值
        assert pricing.greeks.delta == 1.0  # ITM Call

    def test_zero_volatility(self, calculator):
        """测试零波动率"""
        pricing = calculator.price_option(
            OptionType.CALL,
            spot=110.0,
            strike=100.0,
            time_to_expiry=1.0,
            volatility=0.0,  # 零波动率
            risk_free_rate=0.05,
        )

        # 零波动率应该只有内在价值
        assert pricing.theoretical_price >= 0

    @pytest.mark.asyncio
    async def test_no_positions(self, market_provider):
        """测试无持仓"""
        empty_provider = SimulatedPositionProvider()
        config = CalculatorConfig()
        calc = GreeksCalculator(
            config=config,
            market_provider=market_provider,
            position_provider=empty_provider,
        )

        await calc.start()
        await asyncio.sleep(0.2)

        portfolio = calc.get_portfolio_greeks()
        # 无持仓时应该为None或空
        assert portfolio is None or len(portfolio.positions) == 0

        await calc.stop()

    def test_short_position_greeks(self, calculator):
        """测试空头持仓希腊字母"""
        # 空头持仓的希腊字母应该是负的
        pricing = calculator.price_option(
            OptionType.CALL,
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.5,
            volatility=0.25,
            risk_free_rate=0.05,
        )

        # 空头1手
        short_greeks = pricing.greeks.scale(-100)
        assert short_greeks.delta < 0
        assert short_greeks.gamma < 0


class TestSerialization:
    """序列化测试"""

    def test_portfolio_greeks_to_dict(self):
        """测试组合希腊字母序列化"""
        position_greeks = PositionGreeks(
            position_id="POS001",
            option_symbol="AAPL_C150",
            quantity=10,
            multiplier=100.0,
            unit_greeks=Greeks(delta=0.5),
            position_greeks=Greeks(delta=500),
            notional_value=150000.0,
            market_value=8000.0,
            delta_equivalent=75000.0,
        )

        portfolio = PortfolioGreeks(
            portfolio_id="default",
            timestamp=datetime.now(),
            positions=[position_greeks],
            total_greeks=Greeks(delta=500),
            total_delta_equivalent=75000.0,
            total_notional=150000.0,
            total_market_value=8000.0,
            by_underlying={"AAPL": Greeks(delta=500)},
            by_expiry={"2025-02-23": Greeks(delta=500)},
        )

        d = portfolio.to_dict()

        assert "portfolio_id" in d
        assert "positions" in d
        assert len(d["positions"]) == 1
        assert "by_underlying" in d

    def test_scenario_result_to_dict(self):
        """测试情景结果序列化"""
        result = ScenarioResult(
            scenario_name="Test",
            spot_change=0.1,
            vol_change=0.05,
            time_change=7,
            rate_change=0.01,
            new_price=12.0,
            price_change=2.0,
            pnl=2000.0,
            new_greeks=Greeks(delta=0.6),
            greeks_change=Greeks(delta=0.1),
        )

        d = result.to_dict()
        assert d["scenario_name"] == "Test"
        assert d["pnl"] == 2000.0
