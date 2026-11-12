"""
Tests for Counterparty Risk Monitor

交易对手风险监控器测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from quantrisk_realtime.realtime.counterparty_risk_monitor import (
    # Enums
    CreditRating,
    CounterpartyType,
    ExposureType,
    CreditEventType,
    AlertSeverity,
    MonitorState,
    # Data Classes
    CounterpartyProfile,
    DefaultProbability,
    Exposure,
    CounterpartyLimit,
    CreditVaR,
    CVAResult,
    CreditEvent,
    ConcentrationMetrics,
    CounterpartyRiskReport,
    CounterpartyRiskConfig,
    # Providers
    SimulatedExposureProvider,
    SimulatedMarketDataProvider,
    # Main Class
    CounterpartyRiskMonitor,
    # Factory Functions
    create_counterparty_monitor,
    create_counterparty_profile,
    create_counterparty_limit,
    quick_counterparty_check,
    # Constants
    RATING_TO_PD,
    RATING_TO_LGD,
)


# ============ Enum Tests ============

class TestCreditRating:
    """信用评级枚举测试"""

    def test_rating_values(self):
        assert CreditRating.AAA.value == "AAA"
        assert CreditRating.BBB.value == "BBB"
        assert CreditRating.D.value == "D"

    def test_investment_grade_ratings(self):
        investment_grade = [
            CreditRating.AAA, CreditRating.AA_PLUS, CreditRating.AA,
            CreditRating.AA_MINUS, CreditRating.A_PLUS, CreditRating.A,
            CreditRating.A_MINUS, CreditRating.BBB_PLUS, CreditRating.BBB,
            CreditRating.BBB_MINUS,
        ]
        assert len(investment_grade) == 10

    def test_speculative_grade_ratings(self):
        speculative_grade = [
            CreditRating.BB_PLUS, CreditRating.BB, CreditRating.BB_MINUS,
            CreditRating.B_PLUS, CreditRating.B, CreditRating.B_MINUS,
            CreditRating.CCC, CreditRating.CC, CreditRating.C, CreditRating.D,
        ]
        assert len(speculative_grade) == 10


class TestCounterpartyType:
    """交易对手类型枚举测试"""

    def test_type_values(self):
        assert CounterpartyType.BANK.value == "bank"
        assert CounterpartyType.BROKER_DEALER.value == "broker_dealer"
        assert CounterpartyType.CLEARING_HOUSE.value == "clearing_house"

    def test_all_types_count(self):
        assert len(CounterpartyType) == 9


class TestExposureType:
    """敞口类型枚举测试"""

    def test_type_values(self):
        assert ExposureType.CURRENT.value == "current"
        assert ExposureType.POTENTIAL.value == "potential"
        assert ExposureType.SETTLEMENT.value == "settlement"


class TestCreditEventType:
    """信用事件类型枚举测试"""

    def test_type_values(self):
        assert CreditEventType.RATING_DOWNGRADE.value == "rating_downgrade"
        assert CreditEventType.DEFAULT.value == "default"
        assert CreditEventType.CDS_SPIKE.value == "cds_spike"

    def test_all_event_types(self):
        assert len(CreditEventType) == 10


# ============ Data Class Tests ============

class TestCounterpartyProfile:
    """交易对手档案测试"""

    def test_create_profile(self):
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Goldman Sachs",
            counterparty_type=CounterpartyType.BROKER_DEALER,
            credit_rating=CreditRating.A,
        )
        assert profile.counterparty_id == "CP001"
        assert profile.name == "Goldman Sachs"
        assert profile.counterparty_type == CounterpartyType.BROKER_DEALER
        assert profile.credit_rating == CreditRating.A
        assert profile.is_active is True

    def test_profile_to_dict(self):
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Goldman Sachs",
            counterparty_type=CounterpartyType.BROKER_DEALER,
            credit_rating=CreditRating.A,
        )
        d = profile.to_dict()
        assert d["counterparty_id"] == "CP001"
        assert d["counterparty_type"] == "broker_dealer"
        assert d["credit_rating"] == "A"

    def test_profile_with_metadata(self):
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Corp",
            counterparty_type=CounterpartyType.CORPORATE,
            credit_rating=CreditRating.BBB,
            metadata={"industry": "Technology"},
        )
        assert profile.metadata["industry"] == "Technology"


class TestDefaultProbability:
    """违约概率测试"""

    def test_create_default_probability(self):
        dp = DefaultProbability(
            counterparty_id="CP001",
            pd_1y=0.001,
            pd_5y=0.005,
            pd_cumulative=0.005,
            lgd=0.45,
            recovery_rate=0.55,
            expected_loss=1000.0,
        )
        assert dp.pd_1y == 0.001
        assert dp.lgd == 0.45
        assert dp.recovery_rate == 0.55

    def test_dp_to_dict(self):
        dp = DefaultProbability(
            counterparty_id="CP001",
            pd_1y=0.001,
            pd_5y=0.005,
            pd_cumulative=0.005,
            lgd=0.45,
            recovery_rate=0.55,
            expected_loss=1000.0,
        )
        d = dp.to_dict()
        assert d["pd_1y"] == 0.001
        assert d["lgd"] == 0.45


class TestExposure:
    """敞口测试"""

    def test_create_exposure(self):
        exposure = Exposure(
            exposure_id="EXP001",
            counterparty_id="CP001",
            exposure_type=ExposureType.CURRENT,
            gross_exposure=1000000.0,
            net_exposure=800000.0,
            collateral=200000.0,
        )
        assert exposure.gross_exposure == 1000000.0
        assert exposure.net_exposure == 800000.0
        assert exposure.collateral == 200000.0

    def test_collateralized_exposure(self):
        exposure = Exposure(
            exposure_id="EXP001",
            counterparty_id="CP001",
            exposure_type=ExposureType.CURRENT,
            gross_exposure=1000000.0,
            net_exposure=800000.0,
            collateral=200000.0,
        )
        assert exposure.collateralized_exposure == 600000.0

    def test_collateralized_exposure_floor(self):
        exposure = Exposure(
            exposure_id="EXP001",
            counterparty_id="CP001",
            exposure_type=ExposureType.CURRENT,
            gross_exposure=1000000.0,
            net_exposure=100000.0,
            collateral=200000.0,
        )
        # Should not go negative
        assert exposure.collateralized_exposure == 0.0

    def test_exposure_to_dict(self):
        exposure = Exposure(
            exposure_id="EXP001",
            counterparty_id="CP001",
            exposure_type=ExposureType.CURRENT,
            gross_exposure=1000000.0,
            net_exposure=800000.0,
        )
        d = exposure.to_dict()
        assert d["exposure_type"] == "current"
        assert d["gross_exposure"] == 1000000.0


class TestCounterpartyLimit:
    """交易对手限额测试"""

    def test_create_limit(self):
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
            current_usage=500000.0,
        )
        assert limit.limit_value == 1000000.0
        assert limit.current_usage == 500000.0

    def test_utilization(self):
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
            current_usage=500000.0,
        )
        assert limit.utilization == 0.5

    def test_available(self):
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
            current_usage=500000.0,
        )
        assert limit.available == 500000.0

    def test_status_normal(self):
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
            current_usage=500000.0,
        )
        assert limit.status == "normal"

    def test_status_warning(self):
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
            current_usage=850000.0,
        )
        assert limit.status == "warning"

    def test_status_critical(self):
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
            current_usage=960000.0,
        )
        assert limit.status == "critical"

    def test_status_breached(self):
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
            current_usage=1100000.0,
        )
        assert limit.status == "breached"

    def test_limit_to_dict(self):
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
            current_usage=500000.0,
        )
        d = limit.to_dict()
        assert d["utilization"] == 0.5
        assert d["status"] == "normal"


class TestCreditVaR:
    """信用VaR测试"""

    def test_create_credit_var(self):
        var = CreditVaR(
            counterparty_id="CP001",
            var_95=50000.0,
            var_99=80000.0,
            expected_shortfall=90000.0,
        )
        assert var.var_95 == 50000.0
        assert var.var_99 == 80000.0

    def test_credit_var_to_dict(self):
        var = CreditVaR(
            counterparty_id="CP001",
            var_95=50000.0,
            var_99=80000.0,
            expected_shortfall=90000.0,
        )
        d = var.to_dict()
        assert "var_95" in d
        assert "methodology" in d


class TestCVAResult:
    """CVA结果测试"""

    def test_create_cva_result(self):
        cva = CVAResult(
            counterparty_id="CP001",
            cva=10000.0,
            dva=2000.0,
            bilateral_cva=8000.0,
            fva=1000.0,
            total_xva=11000.0,
        )
        assert cva.cva == 10000.0
        assert cva.bilateral_cva == 8000.0

    def test_cva_to_dict(self):
        cva = CVAResult(
            counterparty_id="CP001",
            cva=10000.0,
            dva=2000.0,
            bilateral_cva=8000.0,
            fva=1000.0,
            total_xva=11000.0,
        )
        d = cva.to_dict()
        assert d["cva"] == 10000.0


class TestCreditEvent:
    """信用事件测试"""

    def test_create_event(self):
        event = CreditEvent(
            event_id="EVT001",
            counterparty_id="CP001",
            event_type=CreditEventType.RATING_DOWNGRADE,
            severity=AlertSeverity.WARNING,
            description="Rating downgraded from A to BBB",
        )
        assert event.event_type == CreditEventType.RATING_DOWNGRADE
        assert event.severity == AlertSeverity.WARNING
        assert event.acknowledged is False

    def test_event_to_dict(self):
        event = CreditEvent(
            event_id="EVT001",
            counterparty_id="CP001",
            event_type=CreditEventType.DEFAULT,
            severity=AlertSeverity.EMERGENCY,
            description="Counterparty defaulted",
        )
        d = event.to_dict()
        assert d["event_type"] == "default"
        assert d["severity"] == "emergency"


class TestConcentrationMetrics:
    """集中度指标测试"""

    def test_create_concentration(self):
        metrics = ConcentrationMetrics(
            total_exposure=10000000.0,
            top_counterparty_exposure=2500000.0,
            top_5_exposure=7000000.0,
            top_10_exposure=9000000.0,
            herfindahl_index=0.15,
            largest_counterparty_id="CP001",
            largest_counterparty_pct=0.25,
            counterparty_count=20,
        )
        assert metrics.total_exposure == 10000000.0
        assert metrics.herfindahl_index == 0.15

    def test_concentration_to_dict(self):
        metrics = ConcentrationMetrics(
            total_exposure=10000000.0,
            top_counterparty_exposure=2500000.0,
            top_5_exposure=7000000.0,
            top_10_exposure=9000000.0,
            herfindahl_index=0.15,
            largest_counterparty_id="CP001",
            largest_counterparty_pct=0.25,
            counterparty_count=20,
        )
        d = metrics.to_dict()
        assert d["largest_counterparty_pct"] == 0.25


class TestCounterpartyRiskConfig:
    """配置测试"""

    def test_default_config(self):
        config = CounterpartyRiskConfig()
        assert config.update_interval == 60.0
        assert config.var_confidence == 0.99
        assert config.concentration_warning == 0.25

    def test_custom_config(self):
        config = CounterpartyRiskConfig(
            update_interval=30.0,
            var_confidence=0.95,
        )
        assert config.update_interval == 30.0
        assert config.var_confidence == 0.95

    def test_config_to_dict(self):
        config = CounterpartyRiskConfig()
        d = config.to_dict()
        assert "update_interval" in d
        assert "enable_cva" in d


# ============ Constants Tests ============

class TestRatingToPD:
    """评级-违约概率映射测试"""

    def test_aaa_lowest_pd(self):
        assert RATING_TO_PD[CreditRating.AAA] == 0.0001

    def test_d_highest_pd(self):
        assert RATING_TO_PD[CreditRating.D] == 1.0

    def test_pd_monotonically_increasing(self):
        ratings = [
            CreditRating.AAA, CreditRating.AA_PLUS, CreditRating.AA,
            CreditRating.AA_MINUS, CreditRating.A_PLUS, CreditRating.A,
            CreditRating.A_MINUS, CreditRating.BBB_PLUS, CreditRating.BBB,
            CreditRating.BBB_MINUS, CreditRating.BB_PLUS, CreditRating.BB,
            CreditRating.BB_MINUS, CreditRating.B_PLUS, CreditRating.B,
            CreditRating.B_MINUS, CreditRating.CCC, CreditRating.CC,
            CreditRating.C, CreditRating.D,
        ]
        pds = [RATING_TO_PD[r] for r in ratings]
        for i in range(1, len(pds)):
            assert pds[i] >= pds[i-1], f"PD should increase: {ratings[i-1]} -> {ratings[i]}"


class TestRatingToLGD:
    """类型-LGD映射测试"""

    def test_clearing_house_lowest_lgd(self):
        assert RATING_TO_LGD[CounterpartyType.CLEARING_HOUSE] == 0.30

    def test_hedge_fund_highest_lgd(self):
        assert RATING_TO_LGD[CounterpartyType.HEDGE_FUND] == 0.60


# ============ Provider Tests ============

class TestSimulatedExposureProvider:
    """模拟敞口数据提供者测试"""

    @pytest.fixture
    def provider(self):
        return SimulatedExposureProvider()

    async def test_set_and_get_positions(self, provider):
        positions = [
            {"symbol": "AAPL", "quantity": 100, "market_value": 15000},
            {"symbol": "GOOGL", "quantity": 50, "market_value": 10000},
        ]
        provider.set_positions("CP001", positions)
        result = await provider.get_positions("CP001")
        assert len(result) == 2

    async def test_set_and_get_collateral(self, provider):
        provider.set_collateral("CP001", 100000.0)
        result = await provider.get_collateral("CP001")
        assert result == 100000.0

    async def test_get_empty_positions(self, provider):
        result = await provider.get_positions("CP999")
        assert result == []


class TestSimulatedMarketDataProvider:
    """模拟市场数据提供者测试"""

    @pytest.fixture
    def provider(self):
        return SimulatedMarketDataProvider()

    async def test_set_and_get_cds_spread(self, provider):
        provider.set_cds_spread("CP001", 150.0)
        result = await provider.get_cds_spread("CP001")
        assert result == 150.0

    async def test_set_and_get_rating(self, provider):
        provider.set_rating("CP001", CreditRating.A)
        result = await provider.get_credit_rating("CP001")
        assert result == CreditRating.A


# ============ Monitor Tests ============

class TestCounterpartyRiskMonitor:
    """交易对手风险监控器测试"""

    @pytest.fixture
    def monitor(self):
        return CounterpartyRiskMonitor()

    @pytest.fixture
    def sample_profile(self):
        return CounterpartyProfile(
            counterparty_id="CP001",
            name="Goldman Sachs",
            counterparty_type=CounterpartyType.BROKER_DEALER,
            credit_rating=CreditRating.A,
        )

    def test_monitor_creation(self, monitor):
        assert monitor.state == MonitorState.STOPPED

    async def test_start_stop(self, monitor):
        await monitor.start()
        assert monitor.state == MonitorState.RUNNING
        await monitor.stop()
        assert monitor.state == MonitorState.STOPPED

    async def test_pause_resume(self, monitor):
        await monitor.start()
        await monitor.pause()
        assert monitor.state == MonitorState.PAUSED
        await monitor.resume()
        assert monitor.state == MonitorState.RUNNING
        await monitor.stop()

    def test_add_counterparty(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        assert monitor.get_counterparty("CP001") is not None

    def test_get_all_counterparties(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        profile2 = CounterpartyProfile(
            counterparty_id="CP002",
            name="Morgan Stanley",
            counterparty_type=CounterpartyType.BROKER_DEALER,
            credit_rating=CreditRating.A_MINUS,
        )
        monitor.add_counterparty(profile2)
        all_cps = monitor.get_all_counterparties()
        assert len(all_cps) == 2

    def test_update_counterparty(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        sample_profile.credit_rating = CreditRating.BBB
        result = monitor.update_counterparty(sample_profile)
        assert result is True

    def test_update_nonexistent_counterparty(self, monitor, sample_profile):
        result = monitor.update_counterparty(sample_profile)
        assert result is False

    def test_remove_counterparty(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        result = monitor.remove_counterparty("CP001")
        assert result is True
        assert monitor.get_counterparty("CP001") is None

    def test_remove_nonexistent_counterparty(self, monitor):
        result = monitor.remove_counterparty("CP999")
        assert result is False


class TestLimitManagement:
    """限额管理测试"""

    @pytest.fixture
    def monitor(self):
        return CounterpartyRiskMonitor()

    @pytest.fixture
    def sample_limit(self):
        return CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
        )

    def test_set_limit(self, monitor, sample_limit):
        monitor.set_limit(sample_limit)
        limits = monitor.get_limits("CP001")
        assert len(limits) == 1
        assert limits[0].limit_value == 1000000.0

    def test_update_existing_limit(self, monitor, sample_limit):
        monitor.set_limit(sample_limit)
        new_limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=2000000.0,
        )
        monitor.set_limit(new_limit)
        limits = monitor.get_limits("CP001")
        assert len(limits) == 1
        assert limits[0].limit_value == 2000000.0

    def test_multiple_limit_types(self, monitor):
        limit1 = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
        )
        limit2 = CounterpartyLimit(
            limit_id="LIM002",
            counterparty_id="CP001",
            limit_type="net",
            limit_value=500000.0,
        )
        monitor.set_limit(limit1)
        monitor.set_limit(limit2)
        limits = monitor.get_limits("CP001")
        assert len(limits) == 2

    def test_update_limit_usage(self, monitor, sample_limit):
        monitor.set_limit(sample_limit)
        result = monitor.update_limit_usage("CP001", "gross", 500000.0)
        assert result is True
        limits = monitor.get_limits("CP001")
        assert limits[0].current_usage == 500000.0


class TestExposureCalculation:
    """敞口计算测试"""

    @pytest.fixture
    def monitor(self):
        provider = SimulatedExposureProvider()
        provider.set_positions("CP001", [
            {"symbol": "AAPL", "quantity": 100, "market_value": 15000, "netting_set": "NS1"},
            {"symbol": "GOOGL", "quantity": -50, "market_value": -10000, "netting_set": "NS1"},
        ])
        provider.set_collateral("CP001", 2000.0)
        provider.set_netting_sets("CP001", [{"set_id": "NS1"}])
        return CounterpartyRiskMonitor(exposure_provider=provider)

    @pytest.fixture
    def sample_profile(self):
        return CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BROKER_DEALER,
            credit_rating=CreditRating.A,
        )

    async def test_calculate_exposure(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        exposure = await monitor.calculate_exposure("CP001")
        assert exposure is not None
        assert exposure.gross_exposure == 25000.0  # 15000 + 10000
        assert exposure.collateral == 2000.0

    async def test_calculate_exposure_nonexistent(self, monitor):
        exposure = await monitor.calculate_exposure("CP999")
        assert exposure is None

    async def test_get_exposure(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        await monitor.calculate_exposure("CP001")
        exposure = monitor.get_exposure("CP001")
        assert exposure is not None


class TestDefaultProbabilityCalculation:
    """违约概率计算测试"""

    @pytest.fixture
    def monitor(self):
        return CounterpartyRiskMonitor()

    @pytest.fixture
    def sample_profile(self):
        return CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BROKER_DEALER,
            credit_rating=CreditRating.A,
        )

    async def test_calculate_default_probability(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        dp = await monitor.calculate_default_probability("CP001")
        assert dp is not None
        assert dp.pd_1y == RATING_TO_PD[CreditRating.A]
        assert dp.lgd == RATING_TO_LGD[CounterpartyType.BROKER_DEALER]

    async def test_calculate_dp_nonexistent(self, monitor):
        dp = await monitor.calculate_default_probability("CP999")
        assert dp is None

    async def test_recovery_rate(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        dp = await monitor.calculate_default_probability("CP001")
        assert dp.recovery_rate == 1 - dp.lgd


class TestCreditVaRCalculation:
    """信用VaR计算测试"""

    @pytest.fixture
    def monitor(self):
        provider = SimulatedExposureProvider()
        provider.set_positions("CP001", [
            {"symbol": "AAPL", "market_value": 100000},
        ])
        return CounterpartyRiskMonitor(exposure_provider=provider)

    @pytest.fixture
    def sample_profile(self):
        return CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.BBB,
        )

    async def test_calculate_credit_var(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        await monitor.calculate_exposure("CP001")
        await monitor.calculate_default_probability("CP001")
        var = await monitor.calculate_credit_var("CP001")
        assert var is not None
        assert var.var_99 > var.var_95

    async def test_get_credit_var(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        await monitor.calculate_exposure("CP001")
        await monitor.calculate_default_probability("CP001")
        await monitor.calculate_credit_var("CP001")
        var = monitor.get_credit_var("CP001")
        assert var is not None


class TestCVACalculation:
    """CVA计算测试"""

    @pytest.fixture
    def monitor(self):
        provider = SimulatedExposureProvider()
        provider.set_positions("CP001", [
            {"symbol": "SWAP001", "market_value": 500000},
        ])
        return CounterpartyRiskMonitor(exposure_provider=provider)

    @pytest.fixture
    def sample_profile(self):
        return CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.A,
        )

    async def test_calculate_cva(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        await monitor.calculate_exposure("CP001")
        await monitor.calculate_default_probability("CP001")
        cva = await monitor.calculate_cva("CP001")
        assert cva is not None
        assert cva.cva > 0
        assert cva.dva > 0
        assert cva.bilateral_cva == cva.cva - cva.dva

    async def test_get_cva(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        await monitor.calculate_exposure("CP001")
        await monitor.calculate_default_probability("CP001")
        await monitor.calculate_cva("CP001")
        cva = monitor.get_cva("CP001")
        assert cva is not None


class TestConcentrationAnalysis:
    """集中度分析测试"""

    @pytest.fixture
    def monitor(self):
        provider = SimulatedExposureProvider()
        provider.set_positions("CP001", [{"market_value": 500000}])
        provider.set_positions("CP002", [{"market_value": 300000}])
        provider.set_positions("CP003", [{"market_value": 200000}])
        return CounterpartyRiskMonitor(exposure_provider=provider)

    async def test_calculate_concentration(self, monitor):
        # Add counterparties
        for i in range(1, 4):
            profile = CounterpartyProfile(
                counterparty_id=f"CP00{i}",
                name=f"Test {i}",
                counterparty_type=CounterpartyType.BANK,
                credit_rating=CreditRating.A,
            )
            monitor.add_counterparty(profile)
            await monitor.calculate_exposure(f"CP00{i}")

        concentration = monitor.calculate_concentration()
        assert concentration.total_exposure == 1000000.0
        assert concentration.largest_counterparty_id == "CP001"
        assert concentration.largest_counterparty_pct == 0.5

    def test_concentration_empty(self, monitor):
        concentration = monitor.calculate_concentration()
        assert concentration.total_exposure == 0
        assert concentration.counterparty_count == 0


class TestEventManagement:
    """事件管理测试"""

    @pytest.fixture
    def monitor(self):
        return CounterpartyRiskMonitor()

    @pytest.fixture
    def sample_profile(self):
        return CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.A,
        )

    def test_rating_downgrade_event(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        # Create a new profile object with different rating
        updated_profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.BBB,
        )
        monitor.update_counterparty(updated_profile)
        events = monitor.get_events(counterparty_id="CP001")
        assert len(events) == 1
        assert events[0].event_type == CreditEventType.RATING_DOWNGRADE

    def test_rating_upgrade_event(self, monitor):
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.BBB,
        )
        monitor.add_counterparty(profile)
        upgraded_profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.A,
        )
        monitor.update_counterparty(upgraded_profile)
        events = monitor.get_events(counterparty_id="CP001")
        assert len(events) == 1
        assert events[0].event_type == CreditEventType.RATING_UPGRADE

    def test_acknowledge_event(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        updated_profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.BBB,
        )
        monitor.update_counterparty(updated_profile)
        events = monitor.get_events()
        assert len(events) >= 1
        result = monitor.acknowledge_event(events[0].event_id, "admin")
        assert result is True
        assert events[0].acknowledged is True

    def test_filter_events_by_severity(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        # Create major downgrade for critical severity
        updated_profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.CCC,  # Major downgrade from A
        )
        monitor.update_counterparty(updated_profile)
        critical_events = monitor.get_events(severity="critical")
        assert len(critical_events) >= 1


class TestRiskReport:
    """风险报告测试"""

    @pytest.fixture
    def monitor(self):
        provider = SimulatedExposureProvider()
        provider.set_positions("CP001", [{"market_value": 100000}])
        return CounterpartyRiskMonitor(exposure_provider=provider)

    @pytest.fixture
    def sample_profile(self):
        return CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Counterparty",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.A,
        )

    async def test_generate_report(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        report = await monitor.generate_report("CP001")
        assert report is not None
        assert report.counterparty_id == "CP001"
        assert report.risk_level in ["low", "medium", "high", "critical"]

    async def test_report_contains_recommendations(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        report = await monitor.generate_report("CP001")
        assert len(report.recommendations) > 0

    async def test_report_nonexistent(self, monitor):
        report = await monitor.generate_report("CP999")
        assert report is None

    async def test_report_to_dict(self, monitor, sample_profile):
        monitor.add_counterparty(sample_profile)
        report = await monitor.generate_report("CP001")
        d = report.to_dict()
        assert "risk_score" in d
        assert "recommendations" in d


class TestRiskScoring:
    """风险评分测试"""

    @pytest.fixture
    def monitor(self):
        provider = SimulatedExposureProvider()
        provider.set_positions("CP001", [{"market_value": 100000}])
        return CounterpartyRiskMonitor(exposure_provider=provider)

    async def test_low_risk_score(self):
        monitor = CounterpartyRiskMonitor()
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Safe Bank",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.AAA,
        )
        monitor.add_counterparty(profile)
        report = await monitor.generate_report("CP001")
        assert report.risk_level == "low"

    async def test_high_risk_score(self):
        # Need to have exposure and limit usage to get high risk score
        provider = SimulatedExposureProvider()
        provider.set_positions("CP001", [{"market_value": 100000}])
        monitor = CounterpartyRiskMonitor(exposure_provider=provider)
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Risky Fund",
            counterparty_type=CounterpartyType.HEDGE_FUND,
            credit_rating=CreditRating.B,  # Low rating contributes ~42*0.4 = 16.8 points
        )
        monitor.add_counterparty(profile)
        # Add a limit at high utilization to increase score
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=100000.0,
            current_usage=95000.0,  # 95% utilization = 28.5 points
        )
        monitor.set_limit(limit)
        report = await monitor.generate_report("CP001")
        # Risk score = 16.8 (rating) + ~7.5 (exposure/EL ratio) + 28.5 (limit) ≈ 52.8
        assert report.risk_level in ["medium", "high", "critical"]


class TestStatistics:
    """统计测试"""

    @pytest.fixture
    def monitor(self):
        return CounterpartyRiskMonitor()

    def test_get_statistics(self, monitor):
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.A,
        )
        monitor.add_counterparty(profile)
        stats = monitor.get_statistics()
        assert stats["counterparty_count"] == 1
        assert stats["state"] == "stopped"


# ============ Factory Function Tests ============

class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_counterparty_monitor(self):
        monitor = create_counterparty_monitor(var_confidence=0.95)
        assert monitor.config.var_confidence == 0.95

    def test_create_counterparty_profile(self):
        profile = create_counterparty_profile(
            counterparty_id="CP001",
            name="Test Bank",
            counterparty_type="bank",
            credit_rating="A",
        )
        assert profile.counterparty_type == CounterpartyType.BANK
        assert profile.credit_rating == CreditRating.A

    def test_create_counterparty_limit(self):
        limit = create_counterparty_limit(
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
        )
        assert limit.counterparty_id == "CP001"
        assert limit.limit_value == 1000000.0

    async def test_quick_counterparty_check(self):
        monitor = create_counterparty_monitor()
        profile = create_counterparty_profile(
            counterparty_id="CP001",
            name="Test Bank",
            counterparty_type="bank",
            credit_rating="A",
        )
        monitor.add_counterparty(profile)
        result = await quick_counterparty_check(monitor, "CP001")
        assert result["counterparty_id"] == "CP001"
        assert result["rating"] == "A"

    async def test_quick_check_nonexistent(self):
        monitor = create_counterparty_monitor()
        result = await quick_counterparty_check(monitor, "CP999")
        assert "error" in result


# ============ Edge Cases ============

class TestEdgeCases:
    """边界情况测试"""

    def test_zero_limit_value(self):
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=0.0,
            current_usage=100.0,
        )
        assert limit.utilization == 0.0  # Division by zero protection

    def test_negative_exposure(self):
        exposure = Exposure(
            exposure_id="EXP001",
            counterparty_id="CP001",
            exposure_type=ExposureType.CURRENT,
            gross_exposure=0.0,
            net_exposure=-1000.0,  # Negative (net short)
            collateral=500.0,
        )
        assert exposure.collateralized_exposure == 0.0  # Floor at 0

    async def test_empty_counterparty_list(self):
        monitor = CounterpartyRiskMonitor()
        concentration = monitor.calculate_concentration()
        assert concentration.counterparty_count == 0

    def test_very_long_counterparty_name(self):
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="A" * 1000,  # Very long name
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.A,
        )
        assert len(profile.name) == 1000


class TestCallbacks:
    """回调测试"""

    async def test_event_callback(self):
        events_received = []

        def callback(event):
            events_received.append(event)

        monitor = CounterpartyRiskMonitor(event_callback=callback)
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.A,
        )
        monitor.add_counterparty(profile)
        # Create a new profile object with different rating
        updated_profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.BBB,
        )
        monitor.update_counterparty(updated_profile)

        assert len(events_received) == 1
        assert events_received[0].event_type == CreditEventType.RATING_DOWNGRADE

    async def test_limit_callback(self):
        limits_received = []

        def callback(limit):
            limits_received.append(limit)

        provider = SimulatedExposureProvider()
        provider.set_positions("CP001", [{"market_value": 950000}])

        monitor = CounterpartyRiskMonitor(
            exposure_provider=provider,
            limit_callback=callback,
        )

        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.A,
        )
        monitor.add_counterparty(profile)

        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
        )
        monitor.set_limit(limit)

        await monitor.calculate_exposure("CP001")
        await monitor._check_limits("CP001")

        # Should receive callback due to critical utilization
        assert len(limits_received) >= 1


# ============ Integration Tests ============

class TestIntegration:
    """集成测试"""

    async def test_full_workflow(self):
        """完整工作流程测试"""
        # Setup
        exposure_provider = SimulatedExposureProvider()
        exposure_provider.set_positions("CP001", [
            {"symbol": "SWAP001", "market_value": 500000, "netting_set": "NS1"},
            {"symbol": "SWAP002", "market_value": -200000, "netting_set": "NS1"},
        ])
        exposure_provider.set_collateral("CP001", 50000)
        exposure_provider.set_netting_sets("CP001", [{"set_id": "NS1"}])

        market_provider = SimulatedMarketDataProvider()
        market_provider.set_cds_spread("CP001", 100)

        monitor = CounterpartyRiskMonitor(
            exposure_provider=exposure_provider,
            market_data_provider=market_provider,
        )

        # Add counterparty
        profile = CounterpartyProfile(
            counterparty_id="CP001",
            name="Test Bank",
            counterparty_type=CounterpartyType.BANK,
            credit_rating=CreditRating.A,
        )
        monitor.add_counterparty(profile)

        # Set limits
        limit = CounterpartyLimit(
            limit_id="LIM001",
            counterparty_id="CP001",
            limit_type="gross",
            limit_value=1000000.0,
        )
        monitor.set_limit(limit)

        # Calculate all metrics
        exposure = await monitor.calculate_exposure("CP001")
        dp = await monitor.calculate_default_probability("CP001")
        var = await monitor.calculate_credit_var("CP001")
        cva = await monitor.calculate_cva("CP001")

        # Generate report
        report = await monitor.generate_report("CP001")

        # Assertions
        assert exposure is not None
        assert exposure.gross_exposure == 700000.0
        assert dp is not None
        assert dp.pd_1y == RATING_TO_PD[CreditRating.A]
        assert var is not None
        assert cva is not None
        assert report is not None
        assert report.risk_level in ["low", "medium", "high", "critical"]

        # Test statistics
        stats = monitor.get_statistics()
        assert stats["counterparty_count"] == 1
        assert stats["total_exposure"] == exposure.net_exposure
