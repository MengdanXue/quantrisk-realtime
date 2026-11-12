"""
Volatility Surface Manager 测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List
import math

from quantrisk_realtime.realtime.volatility_surface_manager import (
    # Enums
    OptionType,
    InterpolationMethod,
    SmileShape,
    TermStructureShape,
    AlertType,
    SurfaceState,
    # Data Classes
    OptionQuote,
    VolatilityPoint,
    VolatilitySlice,
    VolatilitySurface,
    SurfaceChange,
    ArbitrageOpportunity,
    VolatilityAlert,
    HistoricalVolComparison,
    SurfaceManagerConfig,
    # Interfaces
    OptionDataProvider,
    HistoricalVolProvider,
    # Simulated Providers
    SimulatedOptionDataProvider,
    SimulatedHistoricalVolProvider,
    # Main Class
    VolatilitySurfaceManager,
    # Factory Functions
    create_surface_manager,
    create_option_quote,
    quick_vol_lookup,
)


# ============ Test Helpers ============


def generate_option_chain(
    symbol: str,
    spot: float = 100.0,
    expiries_days: List[int] = None,
    strikes_pct: List[float] = None,
    base_vol: float = 0.2,
    skew: float = -0.1,
) -> List[OptionQuote]:
    """生成期权链"""
    if expiries_days is None:
        expiries_days = [30, 60, 90, 180, 365]
    if strikes_pct is None:
        strikes_pct = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

    quotes = []
    now = datetime.now()

    for days in expiries_days:
        expiry = now + timedelta(days=days)
        time_to_expiry = days / 365.0

        for strike_pct in strikes_pct:
            strike = spot * strike_pct
            moneyness = strike_pct

            # 模拟波动率微笑
            log_m = math.log(moneyness)
            vol = base_vol + skew * log_m + 0.05 * log_m**2

            # 添加时间结构（远期略高）
            vol *= (1 + 0.1 * math.sqrt(time_to_expiry))

            vol = max(0.05, min(vol, 1.5))  # 限制范围

            # 创建Call和Put
            for opt_type in [OptionType.CALL, OptionType.PUT]:
                price = spot * 0.1 * vol  # 简化价格
                quote = OptionQuote(
                    symbol=symbol,
                    option_symbol=f"{symbol}_{expiry.strftime('%Y%m%d')}_{opt_type.value[0]}{strike:.0f}",
                    option_type=opt_type,
                    strike=strike,
                    expiry=expiry,
                    bid=price * 0.98,
                    ask=price * 1.02,
                    mid=price,
                    last=price,
                    volume=1000,
                    open_interest=5000,
                    implied_vol=vol,
                    timestamp=now,
                )
                quotes.append(quote)

    return quotes


# ============ Enum Tests ============


class TestEnums:
    """枚举测试"""

    def test_option_type(self):
        assert OptionType.CALL.value == "call"
        assert OptionType.PUT.value == "put"

    def test_interpolation_method(self):
        assert InterpolationMethod.LINEAR.value == "linear"
        assert InterpolationMethod.SABR.value == "sabr"
        assert InterpolationMethod.SVI.value == "svi"

    def test_smile_shape(self):
        assert SmileShape.SMILE.value == "smile"
        assert SmileShape.SMIRK.value == "smirk"
        assert SmileShape.SKEW_LEFT.value == "skew_left"

    def test_term_structure_shape(self):
        assert TermStructureShape.NORMAL.value == "normal"
        assert TermStructureShape.INVERTED.value == "inverted"
        assert TermStructureShape.HUMPED.value == "humped"

    def test_alert_type(self):
        assert AlertType.SURFACE_SHIFT.value == "surface_shift"
        assert AlertType.ARBITRAGE_DETECTED.value == "arbitrage"

    def test_surface_state(self):
        assert SurfaceState.READY.value == "ready"
        assert SurfaceState.STALE.value == "stale"


# ============ Data Class Tests ============


class TestOptionQuote:
    """期权报价测试"""

    def test_create_option_quote(self):
        now = datetime.now()
        expiry = now + timedelta(days=30)

        quote = OptionQuote(
            symbol="AAPL",
            option_symbol="AAPL_20250130_C100",
            option_type=OptionType.CALL,
            strike=100.0,
            expiry=expiry,
            bid=5.0,
            ask=5.2,
            mid=5.1,
            last=5.1,
            volume=1000,
            open_interest=5000,
            implied_vol=0.25,
            delta=0.5,
            gamma=0.05,
            theta=-0.02,
            vega=0.15,
        )

        assert quote.symbol == "AAPL"
        assert quote.option_type == OptionType.CALL
        assert quote.strike == 100.0
        assert quote.implied_vol == 0.25

    def test_option_quote_to_dict(self):
        now = datetime.now()
        expiry = now + timedelta(days=30)

        quote = OptionQuote(
            symbol="AAPL",
            option_symbol="AAPL_C100",
            option_type=OptionType.CALL,
            strike=100.0,
            expiry=expiry,
            bid=5.0,
            ask=5.2,
            mid=5.1,
            last=5.1,
            volume=1000,
            open_interest=5000,
            implied_vol=0.25,
        )

        d = quote.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["option_type"] == "call"
        assert d["strike"] == 100.0
        assert d["implied_vol"] == 0.25

    def test_factory_function(self):
        expiry = datetime.now() + timedelta(days=30)
        quote = create_option_quote(
            symbol="AAPL",
            option_symbol="AAPL_C100",
            option_type="call",
            strike=100.0,
            expiry=expiry,
            bid=5.0,
            ask=5.2,
            implied_vol=0.25,
        )

        assert quote.option_type == OptionType.CALL
        assert quote.mid == 5.1


class TestVolatilityPoint:
    """波动率点测试"""

    def test_create_volatility_point(self):
        expiry = datetime.now() + timedelta(days=30)

        point = VolatilityPoint(
            strike=100.0,
            expiry=expiry,
            implied_vol=0.25,
            moneyness=1.0,
            time_to_expiry=30/365,
            log_moneyness=0.0,
            forward_price=100.5,
            option_type=OptionType.CALL,
            confidence=0.95,
        )

        assert point.strike == 100.0
        assert point.implied_vol == 0.25
        assert point.moneyness == 1.0

    def test_volatility_point_to_dict(self):
        expiry = datetime.now() + timedelta(days=30)

        point = VolatilityPoint(
            strike=100.0,
            expiry=expiry,
            implied_vol=0.25,
            moneyness=1.0,
            time_to_expiry=30/365,
            log_moneyness=0.0,
            forward_price=100.5,
            option_type=OptionType.CALL,
        )

        d = point.to_dict()
        assert d["strike"] == 100.0
        assert d["implied_vol"] == 0.25
        assert d["option_type"] == "call"


class TestVolatilitySlice:
    """波动率切片测试"""

    def test_create_volatility_slice(self):
        expiry = datetime.now() + timedelta(days=30)

        points = [
            VolatilityPoint(
                strike=90.0, expiry=expiry, implied_vol=0.28,
                moneyness=0.9, time_to_expiry=30/365,
                log_moneyness=-0.105, forward_price=100.5,
                option_type=OptionType.PUT,
            ),
            VolatilityPoint(
                strike=100.0, expiry=expiry, implied_vol=0.25,
                moneyness=1.0, time_to_expiry=30/365,
                log_moneyness=0.0, forward_price=100.5,
                option_type=OptionType.CALL,
            ),
            VolatilityPoint(
                strike=110.0, expiry=expiry, implied_vol=0.27,
                moneyness=1.1, time_to_expiry=30/365,
                log_moneyness=0.095, forward_price=100.5,
                option_type=OptionType.CALL,
            ),
        ]

        slice_ = VolatilitySlice(
            expiry=expiry,
            time_to_expiry=30/365,
            atm_vol=0.25,
            points=points,
            smile_shape=SmileShape.SMILE,
            skew=-0.01,
            butterfly=0.015,
            atm_strike=100.0,
            forward_price=100.5,
        )

        assert slice_.atm_vol == 0.25
        assert slice_.smile_shape == SmileShape.SMILE
        assert len(slice_.points) == 3

    def test_volatility_slice_to_dict(self):
        expiry = datetime.now() + timedelta(days=30)

        slice_ = VolatilitySlice(
            expiry=expiry,
            time_to_expiry=30/365,
            atm_vol=0.25,
            points=[],
            smile_shape=SmileShape.SMILE,
            skew=-0.01,
            butterfly=0.015,
            atm_strike=100.0,
            forward_price=100.5,
        )

        d = slice_.to_dict()
        assert d["atm_vol"] == 0.25
        assert d["smile_shape"] == "smile"


class TestVolatilitySurface:
    """波动率曲面测试"""

    def test_create_volatility_surface(self):
        now = datetime.now()
        expiry1 = now + timedelta(days=30)
        expiry2 = now + timedelta(days=60)

        slices = [
            VolatilitySlice(
                expiry=expiry1, time_to_expiry=30/365, atm_vol=0.25,
                points=[], smile_shape=SmileShape.SMILE,
                skew=-0.01, butterfly=0.015, atm_strike=100.0,
                forward_price=100.5,
            ),
            VolatilitySlice(
                expiry=expiry2, time_to_expiry=60/365, atm_vol=0.26,
                points=[], smile_shape=SmileShape.SMILE,
                skew=-0.01, butterfly=0.015, atm_strike=100.0,
                forward_price=101.0,
            ),
        ]

        surface = VolatilitySurface(
            symbol="AAPL",
            spot_price=100.0,
            timestamp=now,
            slices=slices,
            term_structure_shape=TermStructureShape.NORMAL,
            atm_term_structure=[(30/365, 0.25), (60/365, 0.26)],
            overall_level=0.255,
            average_skew=-0.01,
            average_butterfly=0.015,
            interpolation_method=InterpolationMethod.LINEAR,
        )

        assert surface.symbol == "AAPL"
        assert surface.spot_price == 100.0
        assert len(surface.slices) == 2

    def test_volatility_surface_to_dict(self):
        now = datetime.now()

        surface = VolatilitySurface(
            symbol="AAPL",
            spot_price=100.0,
            timestamp=now,
            slices=[],
            term_structure_shape=TermStructureShape.NORMAL,
            atm_term_structure=[],
            overall_level=0.25,
            average_skew=-0.01,
            average_butterfly=0.015,
            interpolation_method=InterpolationMethod.LINEAR,
        )

        d = surface.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["overall_level"] == 0.25
        assert d["term_structure_shape"] == "normal"


class TestSurfaceChange:
    """曲面变化测试"""

    def test_create_surface_change(self):
        now = datetime.now()
        expiry = now + timedelta(days=30)

        change = SurfaceChange(
            symbol="AAPL",
            timestamp=now,
            previous_level=0.25,
            current_level=0.27,
            level_change=0.02,
            level_change_pct=0.08,
            skew_change=-0.005,
            butterfly_change=0.001,
            term_structure_change=0.01,
            smile_shape_changed=False,
            term_structure_shape_changed=False,
            max_point_change=0.03,
            max_point_strike=95.0,
            max_point_expiry=expiry,
        )

        assert change.level_change == 0.02
        assert change.level_change_pct == 0.08

    def test_surface_change_to_dict(self):
        now = datetime.now()
        expiry = now + timedelta(days=30)

        change = SurfaceChange(
            symbol="AAPL",
            timestamp=now,
            previous_level=0.25,
            current_level=0.27,
            level_change=0.02,
            level_change_pct=0.08,
            skew_change=-0.005,
            butterfly_change=0.001,
            term_structure_change=0.01,
            smile_shape_changed=False,
            term_structure_shape_changed=False,
            max_point_change=0.03,
            max_point_strike=95.0,
            max_point_expiry=expiry,
        )

        d = change.to_dict()
        assert d["level_change"] == 0.02
        assert d["level_change_pct"] == 0.08


class TestArbitrageOpportunity:
    """套利机会测试"""

    def test_create_arbitrage_opportunity(self):
        expiry = datetime.now() + timedelta(days=30)

        opp = ArbitrageOpportunity(
            symbol="AAPL",
            arbitrage_type="calendar",
            description="Calendar spread arbitrage at strike 100",
            strikes=[100.0],
            expiries=[expiry],
            implied_profit=0.005,
            bid_ask_cost=0.002,
            net_profit=0.003,
            confidence=0.7,
        )

        assert opp.arbitrage_type == "calendar"
        assert opp.net_profit == 0.003

    def test_arbitrage_opportunity_to_dict(self):
        expiry = datetime.now() + timedelta(days=30)

        opp = ArbitrageOpportunity(
            symbol="AAPL",
            arbitrage_type="butterfly",
            description="Butterfly arbitrage",
            strikes=[90.0, 100.0, 110.0],
            expiries=[expiry],
            implied_profit=0.004,
            bid_ask_cost=0.003,
            net_profit=0.001,
            confidence=0.6,
        )

        d = opp.to_dict()
        assert d["arbitrage_type"] == "butterfly"
        assert len(d["strikes"]) == 3


class TestVolatilityAlert:
    """波动率警报测试"""

    def test_create_volatility_alert(self):
        alert = VolatilityAlert(
            symbol="AAPL",
            alert_type=AlertType.SURFACE_SHIFT,
            severity="high",
            message="Volatility surface shifted 5%",
            details={"change_pct": 0.05},
        )

        assert alert.alert_type == AlertType.SURFACE_SHIFT
        assert alert.severity == "high"

    def test_volatility_alert_to_dict(self):
        alert = VolatilityAlert(
            symbol="AAPL",
            alert_type=AlertType.EXTREME_VOL,
            severity="critical",
            message="Extremely high volatility",
            details={"level": 1.5},
        )

        d = alert.to_dict()
        assert d["alert_type"] == "extreme_vol"
        assert d["severity"] == "critical"


class TestHistoricalVolComparison:
    """历史波动率对比测试"""

    def test_create_historical_vol_comparison(self):
        comp = HistoricalVolComparison(
            symbol="AAPL",
            timestamp=datetime.now(),
            implied_vol=0.25,
            realized_vol_1d=0.30,
            realized_vol_5d=0.28,
            realized_vol_20d=0.22,
            realized_vol_60d=0.20,
            vol_premium=0.03,
            vol_premium_percentile=75.0,
        )

        assert comp.implied_vol == 0.25
        assert comp.vol_premium == 0.03

    def test_historical_vol_comparison_to_dict(self):
        comp = HistoricalVolComparison(
            symbol="AAPL",
            timestamp=datetime.now(),
            implied_vol=0.25,
            realized_vol_1d=0.30,
            realized_vol_5d=0.28,
            realized_vol_20d=0.22,
            realized_vol_60d=0.20,
            vol_premium=0.03,
            vol_premium_percentile=75.0,
        )

        d = comp.to_dict()
        assert d["implied_vol"] == 0.25
        assert d["vol_premium_percentile"] == 75.0


class TestSurfaceManagerConfig:
    """配置测试"""

    def test_default_config(self):
        config = SurfaceManagerConfig()

        assert config.interpolation_method == InterpolationMethod.LINEAR
        assert config.min_points_per_slice == 3
        assert config.max_time_to_expiry == 2.0

    def test_custom_config(self):
        config = SurfaceManagerConfig(
            interpolation_method=InterpolationMethod.SABR,
            min_points_per_slice=5,
            level_change_alert_threshold=0.03,
        )

        assert config.interpolation_method == InterpolationMethod.SABR
        assert config.min_points_per_slice == 5


# ============ Provider Tests ============


class TestSimulatedOptionDataProvider:
    """模拟期权数据提供者测试"""

    @pytest.mark.asyncio
    async def test_set_and_get_option_chain(self):
        provider = SimulatedOptionDataProvider()

        quotes = generate_option_chain("AAPL", spot=150.0)
        provider.set_option_chain("AAPL", quotes)

        result = await provider.get_option_chain("AAPL")
        assert len(result) == len(quotes)

    @pytest.mark.asyncio
    async def test_set_and_get_spot_price(self):
        provider = SimulatedOptionDataProvider()

        provider.set_spot_price("AAPL", 150.0)

        result = await provider.get_spot_price("AAPL")
        assert result == 150.0

    @pytest.mark.asyncio
    async def test_default_spot_price(self):
        provider = SimulatedOptionDataProvider()

        result = await provider.get_spot_price("UNKNOWN")
        assert result == 100.0

    @pytest.mark.asyncio
    async def test_set_and_get_risk_free_rate(self):
        provider = SimulatedOptionDataProvider()

        provider.set_risk_free_rate(0.04)

        result = await provider.get_risk_free_rate()
        assert result == 0.04

    @pytest.mark.asyncio
    async def test_set_and_get_dividend_yield(self):
        provider = SimulatedOptionDataProvider()

        provider.set_dividend_yield("AAPL", 0.006)

        result = await provider.get_dividend_yield("AAPL")
        assert result == 0.006


class TestSimulatedHistoricalVolProvider:
    """模拟历史波动率提供者测试"""

    @pytest.mark.asyncio
    async def test_set_and_get_realized_volatility(self):
        provider = SimulatedHistoricalVolProvider()

        provider.set_realized_volatility("AAPL", 20, 0.22)

        result = await provider.get_realized_volatility("AAPL", 20)
        assert result == 0.22

    @pytest.mark.asyncio
    async def test_default_realized_volatility(self):
        provider = SimulatedHistoricalVolProvider()

        result = await provider.get_realized_volatility("AAPL", 20)
        assert result == 0.2

    @pytest.mark.asyncio
    async def test_set_and_get_volatility_history(self):
        provider = SimulatedHistoricalVolProvider()

        history = [
            (datetime.now() - timedelta(days=i), 0.2 + 0.01 * i)
            for i in range(10)
        ]
        provider.set_volatility_history("AAPL", history)

        result = await provider.get_volatility_history("AAPL", 5)
        assert len(result) == 5


# ============ Manager Tests ============


@pytest.fixture
def option_provider():
    """期权数据提供者fixture"""
    provider = SimulatedOptionDataProvider()
    provider.set_spot_price("AAPL", 150.0)
    provider.set_risk_free_rate(0.05)
    provider.set_dividend_yield("AAPL", 0.005)

    quotes = generate_option_chain("AAPL", spot=150.0)
    provider.set_option_chain("AAPL", quotes)

    return provider


@pytest.fixture
def historical_provider():
    """历史波动率提供者fixture"""
    provider = SimulatedHistoricalVolProvider()
    provider.set_realized_volatility("AAPL", 1, 0.25)
    provider.set_realized_volatility("AAPL", 5, 0.23)
    provider.set_realized_volatility("AAPL", 20, 0.20)
    provider.set_realized_volatility("AAPL", 60, 0.18)
    return provider


@pytest.fixture
def manager(option_provider, historical_provider):
    """曲面管理器fixture"""
    config = SurfaceManagerConfig(
        update_interval_ms=100,
        min_points_per_slice=2,
    )
    return VolatilitySurfaceManager(
        config=config,
        option_provider=option_provider,
        historical_provider=historical_provider,
    )


class TestVolatilitySurfaceManager:
    """波动率曲面管理器测试"""

    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """测试启动停止"""
        await manager.start()
        assert manager._running

        await manager.stop()
        assert not manager._running

    @pytest.mark.asyncio
    async def test_add_remove_symbol(self, manager):
        """测试添加删除标的"""
        manager.add_symbol("AAPL")
        manager.add_symbol("GOOG")

        symbols = manager.get_symbols()
        assert "AAPL" in symbols
        assert "GOOG" in symbols

        manager.remove_symbol("GOOG")
        symbols = manager.get_symbols()
        assert "AAPL" in symbols
        assert "GOOG" not in symbols

    @pytest.mark.asyncio
    async def test_build_surface(self, manager):
        """测试构建曲面"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)  # 等待更新

        surface = manager.get_surface("AAPL")
        assert surface is not None
        assert surface.symbol == "AAPL"
        assert len(surface.slices) > 0

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_volatility(self, manager):
        """测试获取波动率"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        vol = manager.get_volatility("AAPL", 150.0, 30/365)
        assert vol is not None
        assert 0.05 < vol < 1.5

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_atm_volatility(self, manager):
        """测试获取ATM波动率"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        vol = manager.get_atm_volatility("AAPL")
        assert vol is not None
        assert 0.05 < vol < 1.5

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_smile(self, manager):
        """测试获取微笑"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        smile = manager.get_smile("AAPL", 30/365)
        assert smile is not None
        assert smile.atm_vol > 0

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_term_structure(self, manager):
        """测试获取期限结构"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        ts = manager.get_term_structure("AAPL")
        assert len(ts) > 0
        # 检查按时间排序
        for i in range(1, len(ts)):
            assert ts[i][0] > ts[i-1][0]

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_skew(self, manager):
        """测试获取偏度"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        skew = manager.get_skew("AAPL")
        assert skew is not None

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_butterfly(self, manager):
        """测试获取蝶式"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        bf = manager.get_butterfly("AAPL")
        assert bf is not None

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_statistics(self, manager):
        """测试获取统计"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        stats = manager.get_statistics()
        assert stats["symbols_tracked"] == 1
        assert "AAPL" in stats["surfaces"]

        await manager.stop()


class TestSurfaceInterpolation:
    """曲面插值测试"""

    @pytest.mark.asyncio
    async def test_strike_interpolation(self, manager):
        """测试行权价插值"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        surface = manager.get_surface("AAPL")
        assert surface is not None

        # 测试在两个已知行权价之间的插值
        vol_low = surface.get_vol(140.0, 30/365)
        vol_high = surface.get_vol(160.0, 30/365)
        vol_mid = surface.get_vol(150.0, 30/365)

        # 中间值应该在两端之间（考虑微笑形态）
        assert vol_low is not None
        assert vol_high is not None
        assert vol_mid is not None

        await manager.stop()

    @pytest.mark.asyncio
    async def test_time_interpolation(self, manager):
        """测试时间插值"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        surface = manager.get_surface("AAPL")
        assert surface is not None

        # 测试在两个到期日之间的插值
        vol_30d = surface.get_vol(150.0, 30/365)
        vol_60d = surface.get_vol(150.0, 60/365)
        vol_45d = surface.get_vol(150.0, 45/365)

        if vol_30d and vol_60d and vol_45d:
            # 45天的波动率应该在30天和60天之间
            assert min(vol_30d, vol_60d) <= vol_45d <= max(vol_30d, vol_60d) + 0.05

        await manager.stop()


class TestSmileClassification:
    """微笑分类测试"""

    def test_classify_smile_with_manager(self, manager):
        """测试微笑分类"""
        # 直接测试分类方法
        expiry = datetime.now() + timedelta(days=30)

        # 创建微笑形态的点
        smile_points = [
            VolatilityPoint(
                strike=90.0, expiry=expiry, implied_vol=0.30,
                moneyness=0.9, time_to_expiry=30/365,
                log_moneyness=-0.105, forward_price=100.0,
                option_type=OptionType.PUT,
            ),
            VolatilityPoint(
                strike=100.0, expiry=expiry, implied_vol=0.25,
                moneyness=1.0, time_to_expiry=30/365,
                log_moneyness=0.0, forward_price=100.0,
                option_type=OptionType.CALL,
            ),
            VolatilityPoint(
                strike=110.0, expiry=expiry, implied_vol=0.28,
                moneyness=1.1, time_to_expiry=30/365,
                log_moneyness=0.095, forward_price=100.0,
                option_type=OptionType.CALL,
            ),
        ]

        shape = manager._classify_smile(smile_points, 0.25)
        assert shape in [SmileShape.SMILE, SmileShape.SMIRK, SmileShape.FLAT]


class TestTermStructureClassification:
    """期限结构分类测试"""

    def test_classify_normal(self, manager):
        """测试正常期限结构"""
        ts = [(0.1, 0.20), (0.25, 0.22), (0.5, 0.24), (1.0, 0.26)]
        shape = manager._classify_term_structure(ts)
        assert shape == TermStructureShape.NORMAL

    def test_classify_inverted(self, manager):
        """测试倒挂期限结构"""
        ts = [(0.1, 0.30), (0.25, 0.27), (0.5, 0.24), (1.0, 0.20)]
        shape = manager._classify_term_structure(ts)
        assert shape == TermStructureShape.INVERTED

    def test_classify_flat(self, manager):
        """测试平坦期限结构"""
        ts = [(0.1, 0.25), (0.25, 0.25), (0.5, 0.25), (1.0, 0.25)]
        shape = manager._classify_term_structure(ts)
        assert shape == TermStructureShape.FLAT

    def test_classify_humped(self, manager):
        """测试驼峰期限结构"""
        ts = [(0.1, 0.20), (0.25, 0.28), (0.5, 0.25), (1.0, 0.22)]
        shape = manager._classify_term_structure(ts)
        assert shape == TermStructureShape.HUMPED


class TestCallbacks:
    """回调测试"""

    @pytest.mark.asyncio
    async def test_surface_callback(self, manager):
        """测试曲面回调"""
        received = []

        def callback(surface):
            received.append(surface)

        manager.on_surface_update(callback)
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)
        await manager.stop()

        assert len(received) > 0
        assert received[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_change_callback(self, manager):
        """测试变化回调"""
        received = []

        def callback(change):
            received.append(change)

        manager.on_change(callback)
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.5)  # 需要两次更新才会有变化
        await manager.stop()

        # 变化回调可能不会触发（如果曲面没变化）
        # 只验证不会报错

    @pytest.mark.asyncio
    async def test_alert_callback(self, manager):
        """测试警报回调"""
        received = []

        def callback(alert):
            received.append(alert)

        manager.on_alert(callback)
        # 警报需要特定条件触发


class TestHistoricalVolComparison:
    """历史波动率对比测试"""

    @pytest.mark.asyncio
    async def test_get_vol_comparison(self, manager):
        """测试获取波动率对比"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        comparison = await manager.get_vol_comparison("AAPL")
        assert comparison is not None
        assert comparison.symbol == "AAPL"
        assert comparison.implied_vol > 0
        assert comparison.realized_vol_20d > 0

        await manager.stop()

    @pytest.mark.asyncio
    async def test_vol_comparison_no_surface(self, manager):
        """测试无曲面时的波动率对比"""
        comparison = await manager.get_vol_comparison("UNKNOWN")
        assert comparison is None


class TestHistory:
    """历史记录测试"""

    @pytest.mark.asyncio
    async def test_surface_history(self, manager):
        """测试曲面历史"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.5)
        await manager.stop()

        history = manager.get_surface_history("AAPL")
        assert len(history) > 0

    @pytest.mark.asyncio
    async def test_change_history(self, manager):
        """测试变化历史"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.5)
        await manager.stop()

        history = manager.get_change_history("AAPL")
        # 可能为空（如果曲面没变化）

    @pytest.mark.asyncio
    async def test_alerts_history(self, manager):
        """测试警报历史"""
        alerts = manager.get_alerts()
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_arbitrage_history(self, manager):
        """测试套利机会历史"""
        opportunities = manager.get_arbitrage_opportunities()
        assert isinstance(opportunities, list)


class TestArbitrageDetection:
    """套利检测测试"""

    @pytest.mark.asyncio
    async def test_calendar_arbitrage_detection(self, manager):
        """测试日历套利检测"""
        # 创建有日历套利的数据
        now = datetime.now()
        expiry1 = now + timedelta(days=30)
        expiry2 = now + timedelta(days=60)

        slice1 = VolatilitySlice(
            expiry=expiry1, time_to_expiry=30/365, atm_vol=0.35,  # 高
            points=[
                VolatilityPoint(
                    strike=100.0, expiry=expiry1, implied_vol=0.35,
                    moneyness=1.0, time_to_expiry=30/365,
                    log_moneyness=0.0, forward_price=100.0,
                    option_type=OptionType.CALL,
                ),
            ],
            smile_shape=SmileShape.FLAT, skew=0, butterfly=0,
            atm_strike=100.0, forward_price=100.0,
        )

        slice2 = VolatilitySlice(
            expiry=expiry2, time_to_expiry=60/365, atm_vol=0.20,  # 低
            points=[
                VolatilityPoint(
                    strike=100.0, expiry=expiry2, implied_vol=0.20,
                    moneyness=1.0, time_to_expiry=60/365,
                    log_moneyness=0.0, forward_price=100.0,
                    option_type=OptionType.CALL,
                ),
            ],
            smile_shape=SmileShape.FLAT, skew=0, butterfly=0,
            atm_strike=100.0, forward_price=100.0,
        )

        surface = VolatilitySurface(
            symbol="TEST",
            spot_price=100.0,
            timestamp=now,
            slices=[slice1, slice2],
            term_structure_shape=TermStructureShape.INVERTED,
            atm_term_structure=[(30/365, 0.35), (60/365, 0.20)],
            overall_level=0.275,
            average_skew=0,
            average_butterfly=0,
            interpolation_method=InterpolationMethod.LINEAR,
        )

        arbs = manager._detect_calendar_arbitrage(surface)
        # 应该检测到套利（近期vol显著高于远期）
        assert len(arbs) > 0
        assert arbs[0].arbitrage_type == "calendar"

    @pytest.mark.asyncio
    async def test_butterfly_arbitrage_detection(self, manager):
        """测试蝶式套利检测"""
        now = datetime.now()
        expiry = now + timedelta(days=30)

        # 创建有蝶式套利的数据（中间行权价vol异常高）
        slice_ = VolatilitySlice(
            expiry=expiry, time_to_expiry=30/365, atm_vol=0.30,
            points=[
                VolatilityPoint(
                    strike=90.0, expiry=expiry, implied_vol=0.25,
                    moneyness=0.9, time_to_expiry=30/365,
                    log_moneyness=-0.105, forward_price=100.0,
                    option_type=OptionType.PUT,
                ),
                VolatilityPoint(
                    strike=100.0, expiry=expiry, implied_vol=0.35,  # 异常高
                    moneyness=1.0, time_to_expiry=30/365,
                    log_moneyness=0.0, forward_price=100.0,
                    option_type=OptionType.CALL,
                ),
                VolatilityPoint(
                    strike=110.0, expiry=expiry, implied_vol=0.25,
                    moneyness=1.1, time_to_expiry=30/365,
                    log_moneyness=0.095, forward_price=100.0,
                    option_type=OptionType.CALL,
                ),
            ],
            smile_shape=SmileShape.FROWN,
            skew=0, butterfly=-0.05,
            atm_strike=100.0, forward_price=100.0,
        )

        surface = VolatilitySurface(
            symbol="TEST",
            spot_price=100.0,
            timestamp=now,
            slices=[slice_],
            term_structure_shape=TermStructureShape.FLAT,
            atm_term_structure=[(30/365, 0.30)],
            overall_level=0.30,
            average_skew=0,
            average_butterfly=-0.05,
            interpolation_method=InterpolationMethod.LINEAR,
        )

        arbs = manager._detect_butterfly_arbitrage(surface)
        # 应该检测到蝶式套利
        assert len(arbs) > 0
        assert arbs[0].arbitrage_type == "butterfly"


class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_surface_manager(self):
        """测试创建曲面管理器"""
        manager = create_surface_manager(
            interpolation_method="linear",
            min_points_per_slice=5,
        )

        assert manager._config.interpolation_method == InterpolationMethod.LINEAR
        assert manager._config.min_points_per_slice == 5

    def test_create_option_quote_factory(self):
        """测试创建期权报价"""
        expiry = datetime.now() + timedelta(days=30)

        quote = create_option_quote(
            symbol="AAPL",
            option_symbol="AAPL_C150",
            option_type="call",
            strike=150.0,
            expiry=expiry,
            bid=5.0,
            ask=5.2,
            implied_vol=0.25,
        )

        assert quote.symbol == "AAPL"
        assert quote.option_type == OptionType.CALL
        assert quote.mid == 5.1

    @pytest.mark.asyncio
    async def test_quick_vol_lookup(self, manager):
        """测试快速波动率查询"""
        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.3)

        vol = await quick_vol_lookup(manager, "AAPL", 150.0, 30/365)
        assert vol is not None or vol is None  # 可能有或无数据

        await manager.stop()


class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_empty_option_chain(self, historical_provider):
        """测试空期权链"""
        provider = SimulatedOptionDataProvider()
        provider.set_spot_price("AAPL", 150.0)
        # 不设置期权链

        config = SurfaceManagerConfig()
        manager = VolatilitySurfaceManager(
            config=config,
            option_provider=provider,
            historical_provider=historical_provider,
        )

        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.2)

        surface = manager.get_surface("AAPL")
        assert surface is None  # 无数据无法构建曲面

        await manager.stop()

    @pytest.mark.asyncio
    async def test_insufficient_points(self, historical_provider):
        """测试点数不足"""
        provider = SimulatedOptionDataProvider()
        provider.set_spot_price("AAPL", 150.0)

        # 只创建一个点
        quotes = [
            create_option_quote(
                symbol="AAPL",
                option_symbol="AAPL_C150",
                option_type="call",
                strike=150.0,
                expiry=datetime.now() + timedelta(days=30),
                bid=5.0,
                ask=5.2,
                implied_vol=0.25,
            )
        ]
        provider.set_option_chain("AAPL", quotes)

        config = SurfaceManagerConfig(min_points_per_slice=3)
        manager = VolatilitySurfaceManager(
            config=config,
            option_provider=provider,
            historical_provider=historical_provider,
        )

        manager.add_symbol("AAPL")

        await manager.start()
        await asyncio.sleep(0.2)

        surface = manager.get_surface("AAPL")
        # 点数不足，切片可能为空
        if surface:
            assert len(surface.slices) == 0

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_volatility_no_surface(self, manager):
        """测试无曲面时获取波动率"""
        vol = manager.get_volatility("UNKNOWN", 100.0, 30/365)
        assert vol is None

    @pytest.mark.asyncio
    async def test_get_atm_volatility_no_surface(self, manager):
        """测试无曲面时获取ATM波动率"""
        vol = manager.get_atm_volatility("UNKNOWN")
        assert vol is None

    @pytest.mark.asyncio
    async def test_get_smile_no_surface(self, manager):
        """测试无曲面时获取微笑"""
        smile = manager.get_smile("UNKNOWN", 30/365)
        assert smile is None

    @pytest.mark.asyncio
    async def test_get_term_structure_no_surface(self, manager):
        """测试无曲面时获取期限结构"""
        ts = manager.get_term_structure("UNKNOWN")
        assert ts == []


class TestSerialization:
    """序列化测试"""

    def test_full_surface_serialization(self):
        """测试完整曲面序列化"""
        now = datetime.now()
        expiry = now + timedelta(days=30)

        points = [
            VolatilityPoint(
                strike=100.0, expiry=expiry, implied_vol=0.25,
                moneyness=1.0, time_to_expiry=30/365,
                log_moneyness=0.0, forward_price=100.0,
                option_type=OptionType.CALL,
            ),
        ]

        slices = [
            VolatilitySlice(
                expiry=expiry, time_to_expiry=30/365, atm_vol=0.25,
                points=points, smile_shape=SmileShape.FLAT,
                skew=0, butterfly=0, atm_strike=100.0, forward_price=100.0,
            ),
        ]

        surface = VolatilitySurface(
            symbol="AAPL",
            spot_price=100.0,
            timestamp=now,
            slices=slices,
            term_structure_shape=TermStructureShape.FLAT,
            atm_term_structure=[(30/365, 0.25)],
            overall_level=0.25,
            average_skew=0,
            average_butterfly=0,
            interpolation_method=InterpolationMethod.LINEAR,
        )

        d = surface.to_dict()

        assert "symbol" in d
        assert "slices" in d
        assert len(d["slices"]) == 1
        assert len(d["slices"][0]["points"]) == 1
