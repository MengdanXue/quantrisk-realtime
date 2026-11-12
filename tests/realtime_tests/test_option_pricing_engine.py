"""
Tests for Option Pricing Engine

测试期权定价引擎的各项功能
"""

import math
import pytest
from datetime import datetime, timedelta

from quantrisk_realtime.realtime.option_pricing_engine import (
    # Enums
    OptionType,
    OptionStyle,
    BarrierType,
    PricingModel,
    VolatilitySurfaceModel,
    CacheStrategy,
    # Data Classes
    OptionContract,
    MarketData,
    GreeksResult,
    PricingResult,
    VolatilityPoint,
    VolatilitySurface,
    PricingConfig,
    # Models
    BlackScholesModel,
    BinomialTreeModel,
    MonteCarloModel,
    LSMModel,
    BarrierModel,
    ImpliedVolatilityCalculator,
    # Main Engine
    OptionPricingEngine,
    # Factory Functions
    create_option_pricing_engine,
    create_option_contract,
    create_market_data,
    # Utilities
    norm_cdf,
    norm_pdf,
    norm_inv,
)


# ============ Fixtures ============

@pytest.fixture
def market_data():
    """标准市场数据"""
    return MarketData(
        symbol="AAPL",
        spot_price=100.0,
        volatility=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.02
    )


@pytest.fixture
def call_option():
    """ATM看涨期权"""
    return OptionContract(
        contract_id="CALL_100",
        underlying_symbol="AAPL",
        option_symbol="AAPL250101C100",
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
        strike=100.0,
        expiry=datetime.now() + timedelta(days=30),
        multiplier=100.0
    )


@pytest.fixture
def put_option():
    """ATM看跌期权"""
    return OptionContract(
        contract_id="PUT_100",
        underlying_symbol="AAPL",
        option_symbol="AAPL250101P100",
        option_type=OptionType.PUT,
        option_style=OptionStyle.EUROPEAN,
        strike=100.0,
        expiry=datetime.now() + timedelta(days=30),
        multiplier=100.0
    )


@pytest.fixture
def american_call():
    """美式看涨期权"""
    return OptionContract(
        contract_id="AMCALL_100",
        underlying_symbol="AAPL",
        option_symbol="AAPL250101C100A",
        option_type=OptionType.CALL,
        option_style=OptionStyle.AMERICAN,
        strike=100.0,
        expiry=datetime.now() + timedelta(days=30),
        multiplier=100.0
    )


@pytest.fixture
def barrier_option():
    """障碍期权"""
    return OptionContract(
        contract_id="BARRIER_100",
        underlying_symbol="AAPL",
        option_symbol="AAPL250101C100B",
        option_type=OptionType.CALL,
        option_style=OptionStyle.EUROPEAN,
        strike=100.0,
        expiry=datetime.now() + timedelta(days=30),
        multiplier=100.0,
        barrier_type=BarrierType.UP_AND_OUT,
        barrier_level=110.0,
        rebate=1.0
    )


@pytest.fixture
def pricing_engine():
    """定价引擎"""
    config = PricingConfig(
        default_model=PricingModel.BLACK_SCHOLES,
        enable_cache=True,
        cache_strategy=CacheStrategy.TIME_BASED,
        mc_simulations=5000,  # 测试时减少模拟次数
        binomial_steps=50
    )
    return OptionPricingEngine(config)


# ============ Utility Tests ============

class TestMathUtilities:
    """测试数学工具函数"""

    def test_norm_cdf(self):
        """测试标准正态分布CDF"""
        assert abs(norm_cdf(0.0) - 0.5) < 1e-10
        assert abs(norm_cdf(1.96) - 0.975) < 1e-3
        assert abs(norm_cdf(-1.96) - 0.025) < 1e-3

    def test_norm_pdf(self):
        """测试标准正态分布PDF"""
        assert abs(norm_pdf(0.0) - 0.3989423) < 1e-6
        assert norm_pdf(-1.0) == pytest.approx(norm_pdf(1.0))

    def test_norm_inv(self):
        """测试标准正态分布逆函数"""
        assert abs(norm_inv(0.5)) < 1e-6
        assert abs(norm_inv(0.975) - 1.96) < 0.1
        assert abs(norm_inv(0.025) + 1.96) < 0.1

        # 测试边界
        with pytest.raises(ValueError):
            norm_inv(0.0)
        with pytest.raises(ValueError):
            norm_inv(1.0)


# ============ Data Class Tests ============

class TestOptionContract:
    """测试期权合约"""

    def test_contract_creation(self, call_option):
        """测试合约创建"""
        assert call_option.contract_id == "CALL_100"
        assert call_option.option_type == OptionType.CALL
        assert call_option.strike == 100.0
        assert call_option.is_call()
        assert not call_option.is_put()
        assert call_option.is_european()
        assert not call_option.is_american()

    def test_time_to_expiry(self, call_option):
        """测试到期时间计算"""
        ttm = call_option.time_to_expiry()
        expected = 30.0 / 365.25
        assert abs(ttm - expected) < 0.01

    def test_intrinsic_value(self, call_option, put_option):
        """测试内在价值"""
        # ATM
        assert call_option.intrinsic_value(100.0) == 0.0
        assert put_option.intrinsic_value(100.0) == 0.0

        # ITM Call
        assert call_option.intrinsic_value(110.0) == 10.0

        # ITM Put
        assert put_option.intrinsic_value(90.0) == 10.0

        # OTM
        assert call_option.intrinsic_value(90.0) == 0.0
        assert put_option.intrinsic_value(110.0) == 0.0

    def test_barrier_option(self, barrier_option):
        """测试障碍期权"""
        assert barrier_option.is_barrier()
        assert barrier_option.barrier_type == BarrierType.UP_AND_OUT
        assert barrier_option.barrier_level == 110.0
        assert barrier_option.rebate == 1.0

    def test_to_dict(self, call_option):
        """测试序列化"""
        data = call_option.to_dict()
        assert data["contract_id"] == "CALL_100"
        assert data["option_type"] == "call"
        assert data["strike"] == 100.0


class TestMarketData:
    """测试市场数据"""

    def test_market_data_creation(self, market_data):
        """测试市场数据创建"""
        assert market_data.symbol == "AAPL"
        assert market_data.spot_price == 100.0
        assert market_data.volatility == 0.25
        assert market_data.risk_free_rate == 0.05

    def test_to_dict(self, market_data):
        """测试序列化"""
        data = market_data.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["spot_price"] == 100.0


class TestGreeksResult:
    """测试Greeks结果"""

    def test_greeks_creation(self):
        """测试Greeks创建"""
        greeks = GreeksResult(
            delta=0.5,
            gamma=0.02,
            theta=-0.01,
            vega=0.1,
            rho=0.05
        )
        assert greeks.delta == 0.5
        assert greeks.gamma == 0.02

    def test_greeks_scale(self):
        """测试Greeks缩放"""
        greeks = GreeksResult(delta=0.5, gamma=0.02)
        scaled = greeks.scale(10)
        assert scaled.delta == 5.0
        assert scaled.gamma == 0.2

    def test_greeks_add(self):
        """测试Greeks相加"""
        g1 = GreeksResult(delta=0.5, gamma=0.02)
        g2 = GreeksResult(delta=0.3, gamma=0.01)
        total = g1.add(g2)
        assert total.delta == pytest.approx(0.8)
        assert total.gamma == pytest.approx(0.03)


# ============ Black-Scholes Model Tests ============

class TestBlackScholesModel:
    """测试Black-Scholes模型"""

    def test_d1_d2_calculation(self):
        """测试d1和d2计算"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, q, sigma)

        # d2 = d1 - sigma * sqrt(T)
        assert d2 == pytest.approx(d1 - sigma * math.sqrt(T))

        # d1 应该接近 (r - q + 0.5*sigma^2)*T / (sigma*sqrt(T))
        expected_d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        assert d1 == pytest.approx(expected_d1)

    def test_call_price_atm(self):
        """测试ATM看涨期权定价"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        price = BlackScholesModel.price(
            OptionType.CALL, S, K, T, r, q, sigma
        )

        # ATM期权价格应该是正数
        assert price > 0

        # 价格应该在合理范围内（0到标的价格之间）
        assert 0 < price < S

    def test_put_price_atm(self):
        """测试ATM看跌期权定价"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        price = BlackScholesModel.price(
            OptionType.PUT, S, K, T, r, q, sigma
        )

        assert price > 0
        assert 0 < price < K

    def test_put_call_parity(self):
        """测试看涨看跌平价关系"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        call_price = BlackScholesModel.price(OptionType.CALL, S, K, T, r, q, sigma)
        put_price = BlackScholesModel.price(OptionType.PUT, S, K, T, r, q, sigma)

        # C - P = S*e^(-qT) - K*e^(-rT)
        parity_diff = call_price - put_price
        expected_diff = S * math.exp(-q * T) - K * math.exp(-r * T)

        assert parity_diff == pytest.approx(expected_diff, rel=1e-6)

    def test_price_at_expiry(self):
        """测试到期时的价格"""
        S, K = 110.0, 100.0

        call_price = BlackScholesModel.price(
            OptionType.CALL, S, K, 0.0, 0.05, 0.0, 0.25
        )
        put_price = BlackScholesModel.price(
            OptionType.PUT, S, K, 0.0, 0.05, 0.0, 0.25
        )

        # 到期时等于内在价值
        assert call_price == pytest.approx(max(S - K, 0))
        assert put_price == pytest.approx(max(K - S, 0))

    def test_greeks_call(self):
        """测试看涨期权Greeks"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        greeks = BlackScholesModel.greeks(
            OptionType.CALL, S, K, T, r, q, sigma
        )

        # Delta应该在0到1之间
        assert 0 < greeks.delta < 1

        # Gamma应该为正
        assert greeks.gamma > 0

        # Theta应该为负（时间衰减）
        assert greeks.theta < 0

        # Vega应该为正
        assert greeks.vega > 0

    def test_greeks_put(self):
        """测试看跌期权Greeks"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        greeks = BlackScholesModel.greeks(
            OptionType.PUT, S, K, T, r, q, sigma
        )

        # Delta应该在-1到0之间
        assert -1 < greeks.delta < 0

        # Gamma应该为正（与Call相同）
        assert greeks.gamma > 0

        # Vega应该为正
        assert greeks.vega > 0


# ============ Binomial Tree Model Tests ============

class TestBinomialTreeModel:
    """测试二叉树模型"""

    def test_european_call_convergence(self):
        """测试欧式看涨期权收敛到Black-Scholes"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        bs_price = BlackScholesModel.price(
            OptionType.CALL, S, K, T, r, q, sigma
        )

        binomial_price = BinomialTreeModel.price(
            OptionType.CALL, S, K, T, r, q, sigma,
            steps=100, american=False
        )

        # 二叉树价格应该接近BS价格
        assert binomial_price == pytest.approx(bs_price, rel=0.05)

    def test_american_call_vs_european(self):
        """测试美式vs欧式看涨期权"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        european_price = BinomialTreeModel.price(
            OptionType.CALL, S, K, T, r, q, sigma,
            steps=100, american=False
        )

        american_price = BinomialTreeModel.price(
            OptionType.CALL, S, K, T, r, q, sigma,
            steps=100, american=True
        )

        # 美式期权价格应该 >= 欧式期权
        assert american_price >= european_price

    def test_american_put_early_exercise(self):
        """测试美式看跌期权提前行权"""
        S, K, T, r, q, sigma = 80.0, 100.0, 1.0, 0.05, 0.0, 0.25

        european_price = BinomialTreeModel.price(
            OptionType.PUT, S, K, T, r, q, sigma,
            steps=100, american=False
        )

        american_price = BinomialTreeModel.price(
            OptionType.PUT, S, K, T, r, q, sigma,
            steps=100, american=True
        )

        # 深度实值美式看跌应该比欧式贵
        assert american_price > european_price


# ============ Monte Carlo Model Tests ============

class TestMonteCarloModel:
    """测试蒙特卡洛模型"""

    def test_call_price_convergence(self):
        """测试看涨期权价格收敛"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        bs_price = BlackScholesModel.price(
            OptionType.CALL, S, K, T, r, q, sigma
        )

        mc_price = MonteCarloModel.price(
            OptionType.CALL, S, K, T, r, q, sigma,
            simulations=10000, time_steps=50, seed=42
        )

        # MC价格应该接近BS价格（允许较大误差）
        assert mc_price == pytest.approx(bs_price, rel=0.1)

    def test_antithetic_variance_reduction(self):
        """测试对偶变量法"""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25

        # 不使用对偶变量
        price_no_antithetic = MonteCarloModel.price(
            OptionType.CALL, S, K, T, r, q, sigma,
            simulations=5000, seed=42, antithetic=False
        )

        # 使用对偶变量
        price_antithetic = MonteCarloModel.price(
            OptionType.CALL, S, K, T, r, q, sigma,
            simulations=5000, seed=42, antithetic=True
        )

        # 两者应该接近（对偶变量法减少方差但不改变期望）
        assert price_antithetic == pytest.approx(price_no_antithetic, rel=0.15)


# ============ LSM Model Tests ============

class TestLSMModel:
    """测试LSM模型"""

    def test_american_put_pricing(self):
        """测试美式看跌期权定价"""
        S, K, T, r, q, sigma = 80.0, 100.0, 1.0, 0.05, 0.0, 0.25

        lsm_price = LSMModel.price(
            OptionType.PUT, S, K, T, r, q, sigma,
            simulations=5000, time_steps=50, seed=42
        )

        # LSM价格应该大于内在价值
        intrinsic = max(K - S, 0)
        assert lsm_price > intrinsic

        # 价格应该在合理范围内
        assert lsm_price < K


# ============ Barrier Model Tests ============

class TestBarrierModel:
    """测试障碍期权模型"""

    def test_up_and_out_call(self):
        """测试上升敲出看涨期权"""
        S, K, H, T, r, q, sigma = 100.0, 100.0, 110.0, 1.0, 0.05, 0.02, 0.25

        barrier_price = BarrierModel.price(
            OptionType.CALL, BarrierType.UP_AND_OUT,
            S, K, H, T, r, q, sigma
        )

        vanilla_price = BlackScholesModel.price(
            OptionType.CALL, S, K, T, r, q, sigma
        )

        # 障碍期权应该比普通期权便宜
        assert barrier_price <= vanilla_price

    def test_barrier_already_breached(self):
        """测试障碍已触及"""
        S, K, H, T, r, q, sigma = 115.0, 100.0, 110.0, 1.0, 0.05, 0.02, 0.25
        rebate = 1.0

        # 上升敲出，标的已在障碍上方
        price = BarrierModel.price(
            OptionType.CALL, BarrierType.UP_AND_OUT,
            S, K, H, T, r, q, sigma, rebate
        )

        # 应该返回补偿
        assert price == rebate


# ============ Implied Volatility Tests ============

class TestImpliedVolatilityCalculator:
    """测试隐含波动率计算器"""

    def test_iv_calculation(self):
        """测试隐含波动率计算"""
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
        true_sigma = 0.25

        # 计算市场价格
        market_price = BlackScholesModel.price(
            OptionType.CALL, S, K, T, r, q, true_sigma
        )

        # 反推隐含波动率
        iv, converged, iterations = ImpliedVolatilityCalculator.calculate(
            OptionType.CALL, market_price, S, K, T, r, q
        )

        assert converged
        assert iv == pytest.approx(true_sigma, rel=1e-3)
        assert iterations < 20

    def test_iv_itm_call(self):
        """测试实值看涨期权隐含波动率"""
        S, K, T, r, q = 110.0, 100.0, 1.0, 0.05, 0.02
        true_sigma = 0.30

        market_price = BlackScholesModel.price(
            OptionType.CALL, S, K, T, r, q, true_sigma
        )

        iv, converged, _ = ImpliedVolatilityCalculator.calculate(
            OptionType.CALL, market_price, S, K, T, r, q
        )

        assert converged
        assert iv == pytest.approx(true_sigma, rel=1e-2)

    def test_iv_put(self):
        """测试看跌期权隐含波动率"""
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
        true_sigma = 0.20

        market_price = BlackScholesModel.price(
            OptionType.PUT, S, K, T, r, q, true_sigma
        )

        iv, converged, _ = ImpliedVolatilityCalculator.calculate(
            OptionType.PUT, market_price, S, K, T, r, q
        )

        assert converged
        assert iv == pytest.approx(true_sigma, rel=1e-3)


# ============ Pricing Engine Tests ============

class TestOptionPricingEngine:
    """测试期权定价引擎"""

    def test_engine_creation(self, pricing_engine):
        """测试引擎创建"""
        assert pricing_engine is not None
        assert pricing_engine.config.default_model == PricingModel.BLACK_SCHOLES

    def test_price_european_call(self, pricing_engine, call_option, market_data):
        """测试欧式看涨期权定价"""
        result = pricing_engine.price_european_option(
            call_option, market_data
        )

        assert result.theoretical_price > 0
        assert result.intrinsic_value >= 0
        assert result.time_value >= 0
        assert result.model == PricingModel.BLACK_SCHOLES

    def test_price_european_put(self, pricing_engine, put_option, market_data):
        """测试欧式看跌期权定价"""
        result = pricing_engine.price_european_option(
            put_option, market_data
        )

        assert result.theoretical_price > 0
        assert result.contract_id == "PUT_100"

    def test_price_american_option(self, pricing_engine, american_call, market_data):
        """测试美式期权定价"""
        result = pricing_engine.price_american_option(
            american_call, market_data, model=PricingModel.BINOMIAL_TREE
        )

        assert result.theoretical_price > 0
        assert result.model == PricingModel.BINOMIAL_TREE

    def test_price_barrier_option(self, pricing_engine, barrier_option, market_data):
        """测试障碍期权定价"""
        result = pricing_engine.price_barrier_option(
            barrier_option, market_data
        )

        assert result.theoretical_price >= 0
        assert result.model == PricingModel.BARRIER

    def test_calculate_greeks(self, pricing_engine, call_option, market_data):
        """测试Greeks计算"""
        greeks = pricing_engine.calculate_greeks(call_option, market_data)

        assert 0 < greeks.delta < 1
        assert greeks.gamma > 0
        assert greeks.theta < 0
        assert greeks.vega > 0

    def test_calculate_implied_volatility(self, pricing_engine, call_option, market_data):
        """测试隐含波动率计算"""
        # 先计算理论价格
        result = pricing_engine.price_european_option(call_option, market_data)
        market_price = result.theoretical_price

        # 反推隐含波动率
        iv, converged = pricing_engine.calculate_implied_volatility(
            call_option, market_price, market_data
        )

        assert converged
        assert iv == pytest.approx(market_data.volatility, rel=1e-2)

    def test_price_portfolio(self, pricing_engine, call_option, put_option, market_data):
        """测试组合定价"""
        contracts = [call_option, put_option]
        market_data_dict = {"AAPL": market_data}
        quantities = {
            call_option.contract_id: 10,
            put_option.contract_id: 5
        }

        portfolio = pricing_engine.price_portfolio(
            contracts, market_data_dict, quantities
        )

        assert len(portfolio.pricing_results) == 2
        assert portfolio.total_value > 0
        assert portfolio.total_greeks is not None

    def test_build_volatility_surface(self, pricing_engine, market_data):
        """测试构建波动率曲面"""
        # 创建不同行权价的期权
        contracts_prices = []
        for strike in [90, 95, 100, 105, 110]:
            contract = create_option_contract(
                f"CALL_{strike}",
                "AAPL",
                f"AAPL250101C{strike}",
                "call",
                "european",
                float(strike),
                datetime.now() + timedelta(days=30)
            )

            # 使用不同的隐含波动率生成市场价格
            iv = 0.20 + (strike - 100) * 0.002  # 波动率微笑
            temp_market = MarketData(
                market_data.symbol,
                market_data.spot_price,
                iv,
                market_data.risk_free_rate,
                market_data.dividend_yield
            )
            result = pricing_engine.price_european_option(contract, temp_market)
            contracts_prices.append((contract, result.theoretical_price))

        # 构建波动率曲面
        surface = pricing_engine.build_volatility_surface(
            "AAPL",
            contracts_prices,
            market_data
        )

        assert len(surface.points) > 0
        assert surface.underlying_symbol == "AAPL"

    def test_caching(self, pricing_engine, call_option, market_data):
        """测试缓存功能"""
        # 第一次定价（缓存未命中）
        result1 = pricing_engine.price_european_option(call_option, market_data)

        # 第二次定价（应该命中缓存）
        result2 = pricing_engine.price_european_option(call_option, market_data)

        assert result1.theoretical_price == result2.theoretical_price

        stats = pricing_engine.get_pricing_statistics()
        assert stats["cache_hits"] > 0

    def test_statistics(self, pricing_engine, call_option, market_data):
        """测试统计信息"""
        # 清空缓存并重置统计确保每次都计数
        pricing_engine.clear_cache(reset_stats=True)

        # 执行几次定价（修改标的价格确保不命中缓存）
        for i in range(5):
            market = MarketData(
                market_data.symbol,
                market_data.spot_price + i * 0.1,  # 每次价格略有不同
                market_data.volatility,
                market_data.risk_free_rate,
                market_data.dividend_yield
            )
            pricing_engine.price_european_option(call_option, market)

        stats = pricing_engine.get_pricing_statistics()

        assert stats["total_pricings"] >= 5
        assert "cache_hit_rate" in stats


# ============ Factory Function Tests ============

class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_engine(self):
        """测试创建引擎"""
        engine = create_option_pricing_engine(
            pricing_model="black_scholes",
            enable_cache=True
        )

        assert engine is not None
        assert engine.config.default_model == PricingModel.BLACK_SCHOLES

    def test_create_contract(self):
        """测试创建合约"""
        contract = create_option_contract(
            "TEST",
            "AAPL",
            "AAPL250101C100",
            "call",
            "european",
            100.0,
            datetime.now() + timedelta(days=30)
        )

        assert contract.contract_id == "TEST"
        assert contract.option_type == OptionType.CALL

    def test_create_market_data(self):
        """测试创建市场数据"""
        market = create_market_data(
            "AAPL",
            100.0,
            0.25,
            0.05,
            0.02
        )

        assert market.symbol == "AAPL"
        assert market.spot_price == 100.0


# ============ Integration Tests ============

class TestIntegration:
    """集成测试"""

    def test_end_to_end_pricing(self):
        """测试端到端定价流程"""
        # 创建引擎
        engine = create_option_pricing_engine()

        # 创建合约
        contract = create_option_contract(
            "TEST_CALL",
            "AAPL",
            "AAPL250101C100",
            "call",
            "european",
            100.0,
            datetime.now() + timedelta(days=30)
        )

        # 创建市场数据
        market = create_market_data("AAPL", 100.0, 0.25)

        # 定价
        result = engine.price_european_option(contract, market)

        # 验证
        assert result.theoretical_price > 0
        assert result.greeks.delta > 0

        # 计算隐含波动率
        iv, converged = engine.calculate_implied_volatility(
            contract, result.theoretical_price, market
        )

        assert converged
        assert iv == pytest.approx(0.25, rel=1e-2)

    def test_multiple_models_comparison(self):
        """测试多模型对比"""
        engine = create_option_pricing_engine()
        contract = create_option_contract(
            "TEST",
            "AAPL",
            "AAPL250101C100",
            "call",
            "european",
            100.0,
            datetime.now() + timedelta(days=90)
        )
        market = create_market_data("AAPL", 100.0, 0.25)

        # Black-Scholes
        bs_result = engine.price_european_option(
            contract, market, PricingModel.BLACK_SCHOLES
        )

        # Monte Carlo
        mc_result = engine.price_european_option(
            contract, market, PricingModel.MONTE_CARLO
        )

        # 两种模型结果应该接近
        assert bs_result.theoretical_price == pytest.approx(
            mc_result.theoretical_price, rel=0.1
        )


# ============ Edge Cases Tests ============

class TestEdgeCases:
    """测试边界情况"""

    def test_zero_time_to_expiry(self, pricing_engine, call_option, market_data):
        """测试零到期时间"""
        # 修改到期时间为过去
        call_option.expiry = datetime.now() - timedelta(days=1)

        result = pricing_engine.price_european_option(call_option, market_data)

        # 应该返回内在价值
        intrinsic = call_option.intrinsic_value(market_data.spot_price)
        assert result.theoretical_price == pytest.approx(intrinsic)

    def test_very_high_volatility(self, pricing_engine, call_option, market_data):
        """测试极高波动率"""
        market_data.volatility = 2.0  # 200%波动率

        result = pricing_engine.price_european_option(call_option, market_data)

        # 应该有合理的价格（不应该崩溃）
        assert result.theoretical_price > 0

    def test_deep_in_the_money(self, pricing_engine, market_data):
        """测试深度实值期权"""
        contract = create_option_contract(
            "DITM",
            "AAPL",
            "AAPL250101C80",
            "call",
            "european",
            80.0,  # 深度实值
            datetime.now() + timedelta(days=30)
        )

        result = pricing_engine.price_european_option(contract, market_data)

        # 价格应该接近内在价值
        intrinsic = contract.intrinsic_value(market_data.spot_price)
        assert result.theoretical_price >= intrinsic

    def test_deep_out_of_the_money(self, pricing_engine, market_data):
        """测试深度虚值期权"""
        contract = create_option_contract(
            "DOTM",
            "AAPL",
            "AAPL250101C150",
            "call",
            "european",
            150.0,  # 深度虚值
            datetime.now() + timedelta(days=30)
        )

        result = pricing_engine.price_european_option(contract, market_data)

        # 价格应该很小但为正
        assert result.theoretical_price > 0
        assert result.theoretical_price < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
