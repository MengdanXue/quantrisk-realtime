"""
资金费率监控器测试
Testing Funding Rate Monitor

作者: TraderX 量化团队
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

# 确保src目录在路径中
import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from quantrisk_realtime.realtime.funding_rate_monitor import (
    # 枚举
    RateType,
    PaymentFrequency,
    ExchangeType,
    AlertLevel,
    AlertType,
    ArbitrageType,
    TrendDirection,
    MonitorState,
    # 数据类
    FundingRateData,
    HistoricalRate,
    RateTrend,
    RateCorrelation,
    ArbitrageOpportunity,
    PositionCost,
    FundingAlert,
    MonitorConfig,
    # 接口和模拟实现
    FundingRateProvider,
    PriceProvider,
    PositionProvider,
    SimulatedFundingRateProvider,
    SimulatedPriceProvider,
    SimulatedPositionProvider,
    # 核心引擎
    FundingRateMonitor,
    # 工厂函数
    create_funding_rate_monitor,
    create_funding_rate_data,
    quick_funding_check,
)


# ==================== 测试夹具 ====================

@pytest.fixture
def monitor_config():
    """测试配置"""
    return MonitorConfig(
        high_rate_threshold=0.001,
        low_rate_threshold=-0.001,
        extreme_rate_threshold=0.005,
        update_interval_seconds=1.0,
    )


@pytest.fixture
def funding_rate_provider():
    """模拟费率提供者"""
    return SimulatedFundingRateProvider()


@pytest.fixture
def price_provider():
    """模拟价格提供者"""
    return SimulatedPriceProvider()


@pytest.fixture
def position_provider():
    """模拟持仓提供者"""
    return SimulatedPositionProvider()


@pytest.fixture
def monitor(monitor_config, funding_rate_provider, price_provider, position_provider):
    """费率监控器"""
    return FundingRateMonitor(
        config=monitor_config,
        funding_rate_provider=funding_rate_provider,
        price_provider=price_provider,
        position_provider=position_provider,
    )


@pytest.fixture
def sample_rate_data():
    """示例费率数据"""
    return create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
        mark_price=50000.0,
        index_price=50000.0,
        spot_price=50000.0,
    )


# ==================== 枚举测试 ====================

def test_rate_type_enum():
    """测试费率类型枚举"""
    assert RateType.PERPETUAL.value == "perpetual"
    assert RateType.SWAP.value == "swap"
    assert RateType.FUTURES.value == "futures"
    assert RateType.OPTION.value == "option"


def test_payment_frequency_enum():
    """测试支付频率枚举"""
    assert PaymentFrequency.HOURLY.value == "hourly"
    assert PaymentFrequency.FOUR_HOURLY.value == "4hourly"
    assert PaymentFrequency.EIGHT_HOURLY.value == "8hourly"
    assert PaymentFrequency.DAILY.value == "daily"


def test_exchange_type_enum():
    """测试交易所类型枚举"""
    assert ExchangeType.BINANCE.value == "binance"
    assert ExchangeType.BYBIT.value == "bybit"
    assert ExchangeType.OKX.value == "okx"
    assert len(list(ExchangeType)) >= 3


def test_alert_level_enum():
    """测试警报级别枚举"""
    assert AlertLevel.INFO.value == "info"
    assert AlertLevel.WARNING.value == "warning"
    assert AlertLevel.CRITICAL.value == "critical"
    assert AlertLevel.EMERGENCY.value == "emergency"


def test_alert_type_enum():
    """测试警报类型枚举"""
    assert AlertType.HIGH_RATE.value == "high_rate"
    assert AlertType.LOW_RATE.value == "low_rate"
    assert AlertType.RATE_REVERSAL.value == "rate_reversal"
    assert AlertType.ARBITRAGE_OPPORTUNITY.value == "arbitrage_opp"


def test_arbitrage_type_enum():
    """测试套利类型枚举"""
    assert ArbitrageType.SPOT_PERP.value == "spot_perpetual"
    assert ArbitrageType.CROSS_EXCHANGE.value == "cross_exchange"
    assert ArbitrageType.CALENDAR_SPREAD.value == "calendar_spread"


def test_trend_direction_enum():
    """测试趋势方向枚举"""
    assert TrendDirection.BULLISH.value == "bullish"
    assert TrendDirection.BEARISH.value == "bearish"
    assert TrendDirection.NEUTRAL.value == "neutral"
    assert TrendDirection.VOLATILE.value == "volatile"


def test_monitor_state_enum():
    """测试监控器状态枚举"""
    assert MonitorState.IDLE.value == "idle"
    assert MonitorState.MONITORING.value == "monitoring"
    assert MonitorState.PAUSED.value == "paused"


# ==================== 数据类测试 ====================

def test_funding_rate_data_creation(sample_rate_data):
    """测试资金费率数据创建"""
    assert sample_rate_data.symbol == "BTCUSDT"
    assert sample_rate_data.exchange == ExchangeType.BINANCE
    assert sample_rate_data.rate == 0.0001
    assert sample_rate_data.mark_price == 50000.0


def test_funding_rate_data_to_dict(sample_rate_data):
    """测试费率数据转字典"""
    data_dict = sample_rate_data.to_dict()
    assert data_dict["symbol"] == "BTCUSDT"
    assert data_dict["exchange"] == "binance"
    assert data_dict["rate"] == 0.0001
    assert "timestamp" in data_dict


def test_historical_rate_creation():
    """测试历史费率创建"""
    hist = HistoricalRate(
        symbol="ETHUSDT",
        exchange=ExchangeType.BYBIT,
        rate=0.0002,
        mark_price=3000.0,
        index_price=3000.0,
        timestamp=datetime.now(),
        payment_time=datetime.now(),
    )
    assert hist.symbol == "ETHUSDT"
    assert hist.rate == 0.0002


def test_historical_rate_to_dict():
    """测试历史费率转字典"""
    hist = HistoricalRate(
        symbol="ETHUSDT",
        exchange=ExchangeType.BYBIT,
        rate=0.0002,
        mark_price=3000.0,
        index_price=3000.0,
        timestamp=datetime.now(),
        payment_time=datetime.now(),
    )
    hist_dict = hist.to_dict()
    assert hist_dict["symbol"] == "ETHUSDT"
    assert hist_dict["exchange"] == "bybit"


def test_rate_trend_creation():
    """测试费率趋势创建"""
    trend = RateTrend(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        trend_direction=TrendDirection.BULLISH,
        trend_strength=0.8,
        current_rate=0.0001,
        avg_rate=0.00008,
        median_rate=0.00009,
        std_dev=0.00001,
        min_rate=0.00005,
        max_rate=0.00015,
    )
    assert trend.trend_direction == TrendDirection.BULLISH
    assert trend.trend_strength == 0.8


def test_rate_trend_to_dict():
    """测试费率趋势转字典"""
    trend = RateTrend(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        trend_direction=TrendDirection.BULLISH,
        trend_strength=0.8,
        current_rate=0.0001,
        avg_rate=0.00008,
        median_rate=0.00009,
        std_dev=0.00001,
        min_rate=0.00005,
        max_rate=0.00015,
    )
    trend_dict = trend.to_dict()
    assert trend_dict["trend_direction"] == "bullish"
    assert trend_dict["trend_strength"] == 0.8


def test_rate_correlation_creation():
    """测试费率相关性创建"""
    corr = RateCorrelation(
        symbol="BTCUSDT",
        price_correlation=0.6,
        volume_correlation=0.4,
        volatility_correlation=0.7,
    )
    assert corr.price_correlation == 0.6
    assert corr.volume_correlation == 0.4


def test_arbitrage_opportunity_creation():
    """测试套利机会创建"""
    opp = ArbitrageOpportunity(
        opportunity_id="opp123",
        arbitrage_type=ArbitrageType.CROSS_EXCHANGE,
        symbol="BTCUSDT",
        long_exchange=ExchangeType.BINANCE,
        short_exchange=ExchangeType.BYBIT,
        rate_difference=0.0005,
        rate_difference_pct=0.05,
    )
    assert opp.arbitrage_type == ArbitrageType.CROSS_EXCHANGE
    assert opp.rate_difference == 0.0005


def test_arbitrage_opportunity_to_dict():
    """测试套利机会转字典"""
    opp = ArbitrageOpportunity(
        opportunity_id="opp123",
        arbitrage_type=ArbitrageType.CROSS_EXCHANGE,
        symbol="BTCUSDT",
        long_exchange=ExchangeType.BINANCE,
        short_exchange=ExchangeType.BYBIT,
        rate_difference=0.0005,
        rate_difference_pct=0.05,
    )
    opp_dict = opp.to_dict()
    assert opp_dict["arbitrage_type"] == "cross_exchange"
    assert opp_dict["rate_difference"] == 0.0005


def test_position_cost_creation():
    """测试持仓成本创建"""
    cost = PositionCost(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        position_size=1.0,
        entry_price=50000.0,
        current_price=51000.0,
        position_value=51000.0,
        cumulative_funding=50.0,
    )
    assert cost.position_size == 1.0
    assert cost.cumulative_funding == 50.0


def test_funding_alert_creation():
    """测试资金警报创建"""
    alert = FundingAlert(
        alert_id="alert123",
        alert_type=AlertType.HIGH_RATE,
        alert_level=AlertLevel.WARNING,
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        message="High funding rate detected",
        current_rate=0.005,
    )
    assert alert.alert_type == AlertType.HIGH_RATE
    assert alert.alert_level == AlertLevel.WARNING


def test_monitor_config_creation(monitor_config):
    """测试监控器配置创建"""
    assert monitor_config.high_rate_threshold == 0.001
    assert monitor_config.update_interval_seconds == 1.0


def test_monitor_config_to_dict(monitor_config):
    """测试配置转字典"""
    config_dict = monitor_config.to_dict()
    assert config_dict["high_rate_threshold"] == 0.001
    assert "update_interval_seconds" in config_dict


# ==================== 提供者测试 ====================

def test_simulated_funding_rate_provider():
    """测试模拟费率提供者"""
    provider = SimulatedFundingRateProvider()
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)

    retrieved = provider.get_funding_rate("BTCUSDT", ExchangeType.BINANCE)
    assert retrieved is not None
    assert retrieved.rate == 0.0001


def test_simulated_funding_rate_provider_batch():
    """测试批量获取费率"""
    provider = SimulatedFundingRateProvider()

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        rate_data = create_funding_rate_data(
            symbol=symbol,
            exchange=ExchangeType.BINANCE,
            rate=0.0001,
        )
        provider.set_rate(symbol, ExchangeType.BINANCE, rate_data)

    rates = provider.get_funding_rates(["BTCUSDT", "ETHUSDT"], ExchangeType.BINANCE)
    assert len(rates) == 2
    assert "BTCUSDT" in rates


def test_simulated_price_provider():
    """测试模拟价格提供者"""
    provider = SimulatedPriceProvider()
    provider.set_prices("BTCUSDT", ExchangeType.BINANCE, 50000.0, 50010.0, 49990.0)

    spot = provider.get_spot_price("BTCUSDT", ExchangeType.BINANCE)
    mark = provider.get_mark_price("BTCUSDT", ExchangeType.BINANCE)
    index = provider.get_index_price("BTCUSDT", ExchangeType.BINANCE)

    assert spot == 50000.0
    assert mark == 50010.0
    assert index == 49990.0


def test_simulated_position_provider():
    """测试模拟持仓提供者"""
    provider = SimulatedPositionProvider()
    position = {
        "size": 1.0,
        "entry_price": 50000.0,
    }
    provider.set_position("BTCUSDT", ExchangeType.BINANCE, position)

    retrieved = provider.get_position("BTCUSDT", ExchangeType.BINANCE)
    assert retrieved is not None
    assert retrieved["size"] == 1.0


# ==================== 监控器核心功能测试 ====================

def test_monitor_initialization(monitor):
    """测试监控器初始化"""
    assert monitor.state == MonitorState.IDLE
    assert len(monitor.current_rates) == 0
    assert len(monitor.alerts) == 0


def test_update_funding_rate(monitor, funding_rate_provider, sample_rate_data):
    """测试更新费率"""
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, sample_rate_data)

    updated = monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)
    assert updated is not None
    assert updated.rate == 0.0001


def test_update_all_rates(monitor, funding_rate_provider):
    """测试批量更新费率"""
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        rate_data = create_funding_rate_data(
            symbol=symbol,
            exchange=ExchangeType.BINANCE,
            rate=0.0001,
        )
        funding_rate_provider.set_rate(symbol, ExchangeType.BINANCE, rate_data)

    results = monitor.update_all_rates(["BTCUSDT", "ETHUSDT"], [ExchangeType.BINANCE])
    assert len(results) == 2


def test_get_current_rate(monitor, funding_rate_provider, sample_rate_data):
    """测试获取当前费率"""
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, sample_rate_data)
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    current = monitor.get_current_rate("BTCUSDT", ExchangeType.BINANCE)
    assert current is not None
    assert current.rate == 0.0001


def test_historical_rates_storage(monitor, funding_rate_provider, sample_rate_data):
    """测试历史费率存储"""
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, sample_rate_data)

    # 更新多次
    for i in range(5):
        monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    historical = monitor.get_historical_rates("BTCUSDT", ExchangeType.BINANCE)
    assert len(historical) == 5


def test_predict_next_rate_ma(monitor, funding_rate_provider):
    """测试移动平均预测"""
    # 设置历史数据
    rates = [0.0001, 0.00012, 0.00011, 0.00013, 0.00012]
    for rate in rates:
        rate_data = create_funding_rate_data(
            symbol="BTCUSDT",
            exchange=ExchangeType.BINANCE,
            rate=rate,
        )
        funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)
        monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    predicted = monitor.predict_next_rate("BTCUSDT", ExchangeType.BINANCE, method="ma")
    assert predicted is not None
    assert 0.00009 < predicted < 0.00015


def test_predict_next_rate_ema(monitor, funding_rate_provider):
    """测试指数移动平均预测"""
    rates = [0.0001, 0.00012, 0.00011, 0.00013, 0.00012]
    for rate in rates:
        rate_data = create_funding_rate_data(
            symbol="BTCUSDT",
            exchange=ExchangeType.BINANCE,
            rate=rate,
        )
        funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)
        monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    predicted = monitor.predict_next_rate("BTCUSDT", ExchangeType.BINANCE, method="ema")
    assert predicted is not None


def test_analyze_rate_trend(monitor, funding_rate_provider):
    """测试费率趋势分析"""
    # 创建上升趋势
    for i in range(20):
        rate = 0.0001 + i * 0.00001
        rate_data = create_funding_rate_data(
            symbol="BTCUSDT",
            exchange=ExchangeType.BINANCE,
            rate=rate,
        )
        funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)
        monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    trend = monitor.analyze_rate_trend("BTCUSDT", ExchangeType.BINANCE)
    assert trend.trend_direction in [TrendDirection.BULLISH, TrendDirection.NEUTRAL]
    assert trend.current_rate > trend.min_rate


def test_analyze_rate_correlation(monitor):
    """测试费率相关性分析"""
    correlation = monitor.analyze_rate_correlation("BTCUSDT", period_days=30)
    assert correlation.symbol == "BTCUSDT"
    assert -1 <= correlation.price_correlation <= 1


# ==================== 套利测试 ====================

def test_find_cross_exchange_arbitrage(monitor, funding_rate_provider):
    """测试跨交易所套利"""
    # 设置不同交易所的费率 - 费率差需要大于min_arbitrage_profit_rate (0.0005)
    rate1 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    rate2 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BYBIT,
        rate=0.001,  # 差异0.0009 > 0.0005阈值
    )

    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate1)
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BYBIT, rate2)

    monitor.update_all_rates(["BTCUSDT"], [ExchangeType.BINANCE, ExchangeType.BYBIT])

    opportunities = monitor.find_arbitrage_opportunities(["BTCUSDT"])
    assert len(opportunities) > 0
    assert opportunities[0].arbitrage_type == ArbitrageType.CROSS_EXCHANGE


def test_arbitrage_opportunity_filtering(monitor, funding_rate_provider):
    """测试套利机会过滤"""
    # 设置费率差异小于阈值
    rate1 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    rate2 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BYBIT,
        rate=0.00011,  # 差异很小
    )

    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate1)
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BYBIT, rate2)

    monitor.update_all_rates(["BTCUSDT"], [ExchangeType.BINANCE, ExchangeType.BYBIT])

    opportunities = monitor.find_arbitrage_opportunities(["BTCUSDT"])
    # 应该被过滤掉
    assert len(opportunities) == 0


def test_spot_perpetual_arbitrage(monitor, funding_rate_provider, price_provider):
    """测试现货-永续套利"""
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.001,  # 高费率
        mark_price=51000.0,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)
    price_provider.set_prices("BTCUSDT", ExchangeType.BINANCE, 50000.0, 51000.0, 50000.0)

    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    opportunities = monitor.find_arbitrage_opportunities(["BTCUSDT"])
    spot_perp_opps = [o for o in opportunities if o.arbitrage_type == ArbitrageType.SPOT_PERP]
    assert len(spot_perp_opps) > 0


# ==================== 持仓成本测试 ====================

def test_calculate_position_cost(monitor, funding_rate_provider, price_provider):
    """测试计算持仓成本"""
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)
    price_provider.set_prices("BTCUSDT", ExchangeType.BINANCE, 51000.0, 51000.0, 51000.0)

    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    cost = monitor.calculate_position_cost(
        "BTCUSDT",
        ExchangeType.BINANCE,
        position_size=1.0,
        entry_price=50000.0,
    )

    assert cost.position_size == 1.0
    assert cost.entry_price == 50000.0
    assert cost.expected_daily_funding > 0


def test_position_cost_with_provider(monitor, funding_rate_provider, price_provider, position_provider):
    """测试从提供者获取持仓成本"""
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)
    price_provider.set_prices("BTCUSDT", ExchangeType.BINANCE, 51000.0, 51000.0, 51000.0)
    position_provider.set_position("BTCUSDT", ExchangeType.BINANCE, {
        "size": 1.0,
        "entry_price": 50000.0,
    })

    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    cost = monitor.calculate_position_cost("BTCUSDT", ExchangeType.BINANCE)
    assert cost.position_size == 1.0


# ==================== 警报测试 ====================

def test_high_rate_alert(monitor, funding_rate_provider):
    """测试高费率警报"""
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.002,  # 高于阈值
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)

    # 初次更新
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    # 再次更新触发检查
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    alerts = monitor.get_alerts(symbol="BTCUSDT")
    high_rate_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_RATE]
    assert len(high_rate_alerts) > 0


def test_rate_reversal_alert(monitor, funding_rate_provider):
    """测试费率反转警报"""
    # 正费率
    rate1 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.001,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate1)
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    # 负费率
    rate2 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=-0.001,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate2)
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    alerts = monitor.get_alerts(alert_type=AlertType.RATE_REVERSAL)
    assert len(alerts) > 0


def test_rate_spike_alert(monitor, funding_rate_provider):
    """测试费率激增警报"""
    rate1 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate1)
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    # 费率激增
    rate2 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.003,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate2)
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    alerts = monitor.get_alerts(alert_type=AlertType.RATE_SPIKE)
    assert len(alerts) > 0


def test_acknowledge_alert(monitor, funding_rate_provider):
    """测试确认警报"""
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.002,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)

    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    alerts = monitor.get_alerts()
    if alerts:
        alert_id = alerts[0].alert_id
        success = monitor.acknowledge_alert(alert_id, "test_user")
        assert success

        acknowledged = monitor.get_alerts(acknowledged=True)
        assert len(acknowledged) > 0


# ==================== 回调测试 ====================

def test_rate_update_callback(monitor, funding_rate_provider, sample_rate_data):
    """测试费率更新回调"""
    callback_data = []

    def on_update(rate_data):
        callback_data.append(rate_data)

    monitor.on_rate_update(on_update)
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, sample_rate_data)
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    assert len(callback_data) == 1
    assert callback_data[0].symbol == "BTCUSDT"


def test_alert_callback(monitor, funding_rate_provider):
    """测试警报回调"""
    alerts_received = []

    def on_alert(alert):
        alerts_received.append(alert)

    monitor.on_alert(on_alert)

    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.002,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)

    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    assert len(alerts_received) > 0


def test_arbitrage_callback(monitor, funding_rate_provider):
    """测试套利回调"""
    opportunities_received = []

    def on_arbitrage(opp):
        opportunities_received.append(opp)

    monitor.on_arbitrage(on_arbitrage)

    rate1 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    rate2 = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BYBIT,
        rate=0.001,
    )

    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate1)
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BYBIT, rate2)

    monitor.update_all_rates(["BTCUSDT"], [ExchangeType.BINANCE, ExchangeType.BYBIT])
    monitor.find_arbitrage_opportunities(["BTCUSDT"])

    assert len(opportunities_received) > 0


# ==================== 异步监控测试 ====================

@pytest.mark.asyncio
async def test_start_stop_monitoring(monitor, funding_rate_provider):
    """测试启动停止监控"""
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)

    await monitor.start_monitoring(["BTCUSDT"], [ExchangeType.BINANCE])
    assert monitor.state == MonitorState.MONITORING

    await asyncio.sleep(0.1)

    await monitor.stop_monitoring()
    assert monitor.state == MonitorState.STOPPED


# ==================== 统计测试 ====================

def test_get_statistics(monitor, funding_rate_provider):
    """测试获取统计信息"""
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)

    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    stats = monitor.get_statistics()
    assert "state" in stats
    assert "symbols_monitored" in stats
    assert stats["rate_updates"] > 0


# ==================== 工厂函数测试 ====================

def test_create_funding_rate_monitor():
    """测试创建监控器"""
    monitor = create_funding_rate_monitor()
    assert monitor is not None
    assert isinstance(monitor, FundingRateMonitor)


def test_create_funding_rate_data_factory():
    """测试创建费率数据"""
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    assert rate_data.symbol == "BTCUSDT"
    assert rate_data.rate == 0.0001


def test_quick_funding_check():
    """测试快速费率检查"""
    result = quick_funding_check(
        symbol="BTCUSDT",
        exchanges=[ExchangeType.BINANCE, ExchangeType.BYBIT],
        rates={
            ExchangeType.BINANCE: 0.0001,
            ExchangeType.BYBIT: 0.001,
        }
    )

    assert result["symbol"] == "BTCUSDT"
    assert "rates" in result
    assert "arbitrage_opportunities" in result
    assert "statistics" in result


# ==================== 边界条件测试 ====================

def test_empty_historical_rates(monitor):
    """测试空历史费率"""
    historical = monitor.get_historical_rates("NONEXISTENT", ExchangeType.BINANCE)
    assert len(historical) == 0


def test_predict_with_insufficient_data(monitor):
    """测试数据不足时的预测"""
    predicted = monitor.predict_next_rate("NONEXISTENT", ExchangeType.BINANCE)
    assert predicted is None


def test_analyze_trend_with_no_data(monitor):
    """测试无数据时的趋势分析"""
    trend = monitor.analyze_rate_trend("NONEXISTENT", ExchangeType.BINANCE)
    assert trend.trend_direction == TrendDirection.NEUTRAL
    assert trend.trend_strength == 0.0


def test_zero_position_cost(monitor, funding_rate_provider):
    """测试零仓位成本"""
    rate_data = create_funding_rate_data(
        symbol="BTCUSDT",
        exchange=ExchangeType.BINANCE,
        rate=0.0001,
    )
    funding_rate_provider.set_rate("BTCUSDT", ExchangeType.BINANCE, rate_data)
    monitor.update_funding_rate("BTCUSDT", ExchangeType.BINANCE)

    cost = monitor.calculate_position_cost(
        "BTCUSDT",
        ExchangeType.BINANCE,
        position_size=0.0,
        entry_price=50000.0,
    )

    assert cost.position_size == 0.0
    assert cost.expected_daily_funding == 0.0


# ==================== 集成测试 ====================

def test_full_workflow(monitor, funding_rate_provider, price_provider):
    """测试完整工作流程"""
    # 1. 设置多个交易所费率
    for exchange in [ExchangeType.BINANCE, ExchangeType.BYBIT, ExchangeType.OKX]:
        rate = 0.0001 if exchange == ExchangeType.BINANCE else 0.001
        rate_data = create_funding_rate_data(
            symbol="BTCUSDT",
            exchange=exchange,
            rate=rate,
        )
        funding_rate_provider.set_rate("BTCUSDT", exchange, rate_data)
        price_provider.set_prices("BTCUSDT", exchange, 50000.0, 50000.0, 50000.0)

    # 2. 更新费率
    monitor.update_all_rates(
        ["BTCUSDT"],
        [ExchangeType.BINANCE, ExchangeType.BYBIT, ExchangeType.OKX]
    )

    # 3. 分析趋势
    trend = monitor.analyze_rate_trend("BTCUSDT", ExchangeType.BINANCE)
    assert trend is not None

    # 4. 寻找套利机会
    opportunities = monitor.find_arbitrage_opportunities(["BTCUSDT"])
    assert len(opportunities) > 0

    # 5. 计算持仓成本
    cost = monitor.calculate_position_cost(
        "BTCUSDT",
        ExchangeType.BINANCE,
        position_size=1.0,
        entry_price=50000.0,
    )
    assert cost.position_size == 1.0

    # 6. 检查统计
    stats = monitor.get_statistics()
    assert stats["symbols_monitored"] >= 1
    assert stats["arbitrage_opportunities"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
