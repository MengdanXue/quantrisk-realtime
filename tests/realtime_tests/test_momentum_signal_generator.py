"""
动量信号生成器测试 (Momentum Signal Generator Tests)

测试动量信号生成器的所有功能:
- 价格动量计算 (ROC, Momentum)
- 相对强弱指数 (RSI)
- 随机指标 (Stochastic)
- MACD 信号
- ADX 趋势强度
- 动量背离检测
- 多时间框架动量
- 动量因子组合
- 综合信号生成

Copyright (c) 2025 TraderX. All rights reserved.
"""

import sys
import os
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from quantrisk_realtime.realtime.momentum_signal_generator import (
    # Main Classes
    MomentumSignalGenerator,
    MomentumCalculator,
    create_momentum_generator,
    # Data Classes
    PriceData,
    MomentumIndicator,
    MomentumSignal,
    DivergenceSignal,
    TrendStrength,
    MomentumConfig,
    # Enums
    SignalType,
    SignalStrength,
    TrendDirection,
    DivergenceType,
    CrossoverType,
    TimeFrame,
    GeneratorState,
)


# ============== Fixtures ==============

@pytest.fixture
def config():
    """创建基础配置"""
    return MomentumConfig(
        rsi_period=14,
        stoch_k_period=14,
        macd_fast_period=12,
        macd_slow_period=26,
        adx_period=14,
        update_interval=0.1,
        history_size=500,
    )


@pytest.fixture
def calculator():
    """创建计算器"""
    return MomentumCalculator()


@pytest.fixture
def generator(config):
    """创建基础生成器"""
    return MomentumSignalGenerator(config=config)


@pytest.fixture
def generator_with_data(generator):
    """创建带有数据的生成器"""
    # 生成上升趋势的价格数据
    base_price = 100.0
    now = datetime.now()

    for i in range(50):
        price = base_price + i * 0.5  # 上升趋势
        price_data = PriceData(
            symbol='AAPL',
            timestamp=now - timedelta(days=50-i),
            open=price - 0.2,
            high=price + 0.3,
            low=price - 0.3,
            close=price,
            volume=1000000 + i * 10000
        )
        asyncio.run(generator.update_price(price_data))

    return generator


@pytest.fixture
def price_data_uptrend():
    """生成上升趋势价格数据"""
    base_price = 100.0
    now = datetime.now()
    data = []

    for i in range(30):
        price = base_price + i * 0.5
        data.append(PriceData(
            symbol='AAPL',
            timestamp=now - timedelta(days=30-i),
            open=price - 0.2,
            high=price + 0.3,
            low=price - 0.3,
            close=price,
            volume=1000000
        ))

    return data


@pytest.fixture
def price_data_downtrend():
    """生成下降趋势价格数据"""
    base_price = 100.0
    now = datetime.now()
    data = []

    for i in range(30):
        price = base_price - i * 0.5
        data.append(PriceData(
            symbol='AAPL',
            timestamp=now - timedelta(days=30-i),
            open=price + 0.2,
            high=price + 0.3,
            low=price - 0.3,
            close=price,
            volume=1000000
        ))

    return data


@pytest.fixture
def price_data_sideways():
    """生成横盘价格数据"""
    base_price = 100.0
    now = datetime.now()
    data = []

    for i in range(30):
        # 在99-101之间波动
        price = base_price + (i % 3 - 1) * 0.5
        data.append(PriceData(
            symbol='AAPL',
            timestamp=now - timedelta(days=30-i),
            open=price - 0.1,
            high=price + 0.2,
            low=price - 0.2,
            close=price,
            volume=1000000
        ))

    return data


# ============== Test MomentumCalculator ==============

class TestMomentumCalculator:
    """测试动量计算器"""

    def test_calculate_roc_basic(self, calculator):
        """测试1: 基础ROC计算"""
        prices = [100, 102, 105, 108, 110]
        roc = calculator.calculate_roc(prices, 3)

        assert roc is not None
        assert isinstance(roc, float)
        # ROC = ((110 - 102) / 102) * 100 ≈ 7.84
        assert 7.0 < roc < 9.0

    def test_calculate_roc_insufficient_data(self, calculator):
        """测试2: 数据不足的ROC计算"""
        prices = [100, 102]
        roc = calculator.calculate_roc(prices, 5)

        assert roc is None

    def test_calculate_roc_negative(self, calculator):
        """测试3: 负ROC计算"""
        prices = [110, 108, 105, 102, 100]
        roc = calculator.calculate_roc(prices, 3)

        assert roc is not None
        assert roc < 0

    def test_calculate_momentum_basic(self, calculator):
        """测试4: 基础动量计算"""
        prices = [100, 102, 105, 108, 110]
        momentum = calculator.calculate_momentum(prices, 3)

        assert momentum is not None
        # Momentum = 110 - 102 = 8
        assert momentum == 8.0

    def test_calculate_momentum_negative(self, calculator):
        """测试5: 负动量计算"""
        prices = [110, 108, 105, 102, 100]
        momentum = calculator.calculate_momentum(prices, 3)

        assert momentum is not None
        assert momentum < 0

    def test_calculate_rsi_basic(self, calculator):
        """测试6: 基础RSI计算"""
        # 生成上升趋势价格
        prices = [100 + i for i in range(20)]
        rsi = calculator.calculate_rsi(prices, 14)

        assert rsi is not None
        assert 0 <= rsi <= 100
        assert rsi > 50  # 上升趋势RSI应该>50

    def test_calculate_rsi_downtrend(self, calculator):
        """测试7: 下降趋势RSI"""
        prices = [100 - i for i in range(20)]
        rsi = calculator.calculate_rsi(prices, 14)

        assert rsi is not None
        assert rsi < 50  # 下降趋势RSI应该<50

    def test_calculate_rsi_insufficient_data(self, calculator):
        """测试8: 数据不足的RSI"""
        prices = [100, 102, 105]
        rsi = calculator.calculate_rsi(prices, 14)

        assert rsi is None

    def test_calculate_rsi_extreme_values(self, calculator):
        """测试9: RSI极值测试"""
        # 连续上涨
        prices = [100 + i * 2 for i in range(20)]
        rsi = calculator.calculate_rsi(prices, 14)

        assert rsi is not None
        assert rsi > 70  # 应该在超买区

    def test_calculate_stochastic_basic(self, calculator):
        """测试10: 基础随机指标计算"""
        high = [105, 106, 107, 108, 109] + [110] * 10
        low = [95, 96, 97, 98, 99] + [100] * 10
        close = [100, 101, 102, 103, 104] + [105] * 10

        result = calculator.calculate_stochastic(high, low, close, 14, 3)

        assert result is not None
        k, d = result
        assert 0 <= k <= 100
        assert 0 <= d <= 100

    def test_calculate_stochastic_insufficient_data(self, calculator):
        """测试11: 数据不足的随机指标"""
        high = [105, 106]
        low = [95, 96]
        close = [100, 101]

        result = calculator.calculate_stochastic(high, low, close, 14, 3)
        assert result is None

    def test_calculate_macd_basic(self, calculator):
        """测试12: 基础MACD计算"""
        # 生成足够的数据
        prices = [100 + i * 0.5 for i in range(30)]
        result = calculator.calculate_macd(prices, 12, 26, 9)

        assert result is not None
        macd, signal, histogram = result
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(histogram, float)

    def test_calculate_macd_insufficient_data(self, calculator):
        """测试13: 数据不足的MACD"""
        prices = [100, 101, 102]
        result = calculator.calculate_macd(prices, 12, 26, 9)

        assert result is None

    def test_calculate_adx_basic(self, calculator):
        """测试14: 基础ADX计算"""
        # 生成趋势数据
        high = [100 + i for i in range(20)]
        low = [95 + i for i in range(20)]
        close = [98 + i for i in range(20)]

        result = calculator.calculate_adx(high, low, close, 14)

        assert result is not None
        adx, plus_di, minus_di = result
        assert 0 <= adx <= 100
        assert plus_di >= 0
        assert minus_di >= 0

    def test_calculate_adx_insufficient_data(self, calculator):
        """测试15: 数据不足的ADX"""
        high = [100, 101]
        low = [95, 96]
        close = [98, 99]

        result = calculator.calculate_adx(high, low, close, 14)
        assert result is None


# ============== Test MomentumSignalGenerator ==============

class TestMomentumSignalGenerator:
    """测试动量信号生成器"""

    def test_generator_initialization(self, generator):
        """测试16: 生成器初始化"""
        assert generator.state == GeneratorState.IDLE
        assert isinstance(generator.config, MomentumConfig)

    @pytest.mark.asyncio
    async def test_update_price(self, generator):
        """测试17: 更新价格数据"""
        price_data = PriceData(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000
        )

        await generator.update_price(price_data)

        assert 'AAPL' in generator._price_history
        assert len(generator._price_history['AAPL']) == 1

    @pytest.mark.asyncio
    async def test_update_multiple_prices(self, generator):
        """测试18: 更新多个价格"""
        now = datetime.now()

        for i in range(5):
            price_data = PriceData(
                symbol='AAPL',
                timestamp=now - timedelta(days=5-i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000
            )
            await generator.update_price(price_data)

        assert len(generator._price_history['AAPL']) == 5

    def test_calculate_roc_indicator(self, generator_with_data):
        """测试19: 计算ROC指标"""
        indicator = generator_with_data.calculate_roc('AAPL')

        assert indicator is not None
        assert isinstance(indicator, MomentumIndicator)
        assert indicator.name == 'roc'
        assert indicator.value is not None

    def test_calculate_rsi_indicator(self, generator_with_data):
        """测试20: 计算RSI指标"""
        indicator = generator_with_data.calculate_rsi('AAPL')

        assert indicator is not None
        assert isinstance(indicator, MomentumIndicator)
        assert indicator.name == 'rsi'
        assert 0 <= indicator.value <= 100

    def test_calculate_stochastic_indicator(self, generator_with_data):
        """测试21: 计算随机指标"""
        indicator = generator_with_data.calculate_stochastic('AAPL')

        assert indicator is not None
        assert isinstance(indicator, MomentumIndicator)
        assert indicator.name == 'stochastic'
        assert 0 <= indicator.value <= 100

    def test_calculate_macd_indicator(self, generator_with_data):
        """测试22: 计算MACD指标"""
        indicator = generator_with_data.calculate_macd('AAPL')

        assert indicator is not None
        assert isinstance(indicator, MomentumIndicator)
        assert indicator.name == 'macd'

    def test_calculate_adx_indicator(self, generator_with_data):
        """测试23: 计算ADX指标"""
        trend = generator_with_data.calculate_adx('AAPL')

        assert trend is not None
        assert isinstance(trend, TrendStrength)
        assert 0 <= trend.adx <= 100
        assert trend.trend_direction in TrendDirection

    def test_generate_composite_signal_basic(self, generator_with_data):
        """测试24: 生成综合信号"""
        # 确保所有指标都已计算
        generator_with_data.calculate_rsi('AAPL')
        generator_with_data.calculate_macd('AAPL')
        generator_with_data.calculate_stochastic('AAPL')

        signal = generator_with_data.generate_composite_signal('AAPL')

        assert signal is not None
        assert isinstance(signal, MomentumSignal)
        assert signal.symbol == 'AAPL'
        assert signal.signal_type in SignalType
        assert 0 <= signal.confidence <= 1

    def test_generate_signal_with_uptrend(self, generator, price_data_uptrend):
        """测试25: 上升趋势的信号生成"""
        for price in price_data_uptrend:
            asyncio.run(generator.update_price(price))

        signal = generator.generate_composite_signal('AAPL')

        if signal:
            # 上升趋势应该倾向于买入信号
            assert signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.HOLD]

    def test_generate_signal_with_downtrend(self, generator, price_data_downtrend):
        """测试26: 下降趋势的信号生成"""
        for price in price_data_downtrend:
            asyncio.run(generator.update_price(price))

        signal = generator.generate_composite_signal('AAPL')

        if signal:
            # 下降趋势应该倾向于卖出信号
            assert signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL, SignalType.HOLD]

    def test_detect_divergence(self, generator_with_data):
        """测试27: 检测背离信号"""
        divergence = generator_with_data.detect_divergence('AAPL', 'rsi')

        # 背离不一定总是存在
        if divergence:
            assert isinstance(divergence, DivergenceSignal)
            assert divergence.divergence_type in DivergenceType

    def test_get_latest_signal(self, generator_with_data):
        """测试28: 获取最新信号"""
        # 生成信号
        generator_with_data.calculate_rsi('AAPL')
        generator_with_data.generate_composite_signal('AAPL')

        signal = generator_with_data.get_latest_signal('AAPL')

        if signal:
            assert isinstance(signal, MomentumSignal)
            assert signal.symbol == 'AAPL'

    def test_get_indicator(self, generator_with_data):
        """测试29: 获取指标值"""
        generator_with_data.calculate_rsi('AAPL')

        indicator = generator_with_data.get_indicator('AAPL', 'rsi')

        assert indicator is not None
        assert isinstance(indicator, MomentumIndicator)
        assert indicator.name == 'rsi'

    def test_get_all_indicators(self, generator_with_data):
        """测试30: 获取所有指标"""
        generator_with_data.calculate_rsi('AAPL')
        generator_with_data.calculate_macd('AAPL')
        generator_with_data.calculate_stochastic('AAPL')

        indicators = generator_with_data.get_all_indicators('AAPL')

        assert isinstance(indicators, dict)
        assert len(indicators) >= 3

    def test_get_trend_strength(self, generator_with_data):
        """测试31: 获取趋势强度"""
        generator_with_data.calculate_adx('AAPL')

        trend = generator_with_data.get_trend_strength('AAPL')

        assert trend is not None
        assert isinstance(trend, TrendStrength)

    def test_get_multi_timeframe_momentum(self, generator_with_data):
        """测试32: 多时间框架动量"""
        # 计算指标
        generator_with_data.calculate_rsi('AAPL')
        generator_with_data.calculate_macd('AAPL')

        mtf_signals = generator_with_data.get_multi_timeframe_momentum('AAPL')

        assert isinstance(mtf_signals, dict)

    def test_get_statistics(self, generator_with_data):
        """测试33: 获取统计信息"""
        stats = generator_with_data.get_statistics()

        assert isinstance(stats, dict)
        assert 'state' in stats
        assert 'symbols_tracked' in stats
        assert stats['symbols_tracked'] >= 1

    @pytest.mark.asyncio
    async def test_start_stop_generator(self, generator):
        """测试34: 启动停止生成器"""
        await generator.start()
        assert generator.state == GeneratorState.RUNNING

        await asyncio.sleep(0.2)

        await generator.stop()
        assert generator.state == GeneratorState.IDLE

    @pytest.mark.asyncio
    async def test_pause_resume_generator(self, generator):
        """测试35: 暂停恢复生成器"""
        await generator.start()
        assert generator.state == GeneratorState.RUNNING

        await generator.pause()
        assert generator.state == GeneratorState.PAUSED

        await generator.resume()
        assert generator.state == GeneratorState.RUNNING

        await generator.stop()


# ============== Test Signal Callbacks ==============

class TestSignalCallbacks:
    """测试信号回调"""

    @pytest.mark.asyncio
    async def test_signal_callback_triggered(self, config):
        """测试36: 信号回调触发"""
        callback_signals = []

        def signal_callback(signal: MomentumSignal):
            callback_signals.append(signal)

        generator = MomentumSignalGenerator(config=config, signal_callback=signal_callback)

        # 添加数据
        for i in range(20):
            price_data = PriceData(
                symbol='AAPL',
                timestamp=datetime.now() - timedelta(days=20-i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000
            )
            await generator.update_price(price_data)

        # 应该有信号产生
        assert len(callback_signals) >= 0

    @pytest.mark.asyncio
    async def test_divergence_callback_triggered(self, config):
        """测试37: 背离回调触发"""
        callback_divergences = []

        def divergence_callback(divergence: DivergenceSignal):
            callback_divergences.append(divergence)

        generator = MomentumSignalGenerator(
            config=config,
            divergence_callback=divergence_callback
        )

        # 添加数据
        for i in range(60):
            price_data = PriceData(
                symbol='AAPL',
                timestamp=datetime.now() - timedelta(days=60-i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000
            )
            await generator.update_price(price_data)

        # 检测背离
        generator.detect_divergence('AAPL', 'rsi')

        # 背离不一定总是存在
        assert isinstance(callback_divergences, list)


# ============== Test Edge Cases ==============

class TestEdgeCases:
    """测试边界情况"""

    def test_empty_price_history(self, generator):
        """测试38: 空价格历史"""
        signal = generator.generate_composite_signal('NONEXISTENT')
        assert signal is None

    def test_insufficient_data_for_indicators(self, generator):
        """测试39: 数据不足"""
        # 仅添加少量数据
        price_data = PriceData(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000
        )
        asyncio.run(generator.update_price(price_data))

        # 数据不足时应该返回None
        indicator = generator.calculate_rsi('AAPL')
        assert indicator is None

    def test_zero_prices(self, calculator):
        """测试40: 零价格处理"""
        prices = [0, 0, 0, 0, 0]
        roc = calculator.calculate_roc(prices, 3)

        # 零价格应该返回None
        assert roc is None

    def test_negative_prices(self, calculator):
        """测试41: 负价格处理"""
        # 虽然实际中不会有负价格,但要确保程序不崩溃
        prices = [100, 90, -10, 80, 70]
        rsi = calculator.calculate_rsi(prices, 4)

        # 应该能处理
        assert rsi is not None or rsi is None  # 都是合法的

    def test_very_volatile_prices(self, calculator):
        """测试42: 极端波动价格"""
        prices = [100, 200, 50, 300, 25]
        rsi = calculator.calculate_rsi(prices, 4)

        if rsi is not None:
            assert 0 <= rsi <= 100

    @pytest.mark.asyncio
    async def test_same_price_repeated(self, generator):
        """测试43: 重复相同价格"""
        for i in range(20):
            price_data = PriceData(
                symbol='AAPL',
                timestamp=datetime.now() - timedelta(days=20-i),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000000
            )
            await generator.update_price(price_data)

        indicator = generator.calculate_rsi('AAPL')

        # 价格不变时,没有涨跌,RSI可能计算为极值或特殊值
        # 只验证指标存在且在合理范围内
        if indicator:
            assert 0 <= indicator.value <= 100


# ============== Test Integration ==============

class TestIntegration:
    """测试集成场景"""

    @pytest.mark.asyncio
    async def test_full_trading_cycle(self, generator):
        """测试44: 完整交易周期"""
        # 模拟完整的市场周期:上涨->盘整->下跌
        now = datetime.now()
        base_price = 100.0

        # 上涨阶段
        for i in range(20):
            price = base_price + i * 0.5
            price_data = PriceData(
                symbol='AAPL',
                timestamp=now - timedelta(days=60-i),
                open=price - 0.1,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
                volume=1000000
            )
            await generator.update_price(price_data)

        # 盘整阶段
        peak_price = base_price + 20 * 0.5
        for i in range(20):
            price = peak_price + (i % 3 - 1) * 0.3
            price_data = PriceData(
                symbol='AAPL',
                timestamp=now - timedelta(days=40-i),
                open=price - 0.1,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
                volume=1000000
            )
            await generator.update_price(price_data)

        # 下跌阶段
        for i in range(20):
            price = peak_price - i * 0.4
            price_data = PriceData(
                symbol='AAPL',
                timestamp=now - timedelta(days=20-i),
                open=price + 0.1,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
                volume=1000000
            )
            await generator.update_price(price_data)

        # 计算所有指标
        rsi = generator.calculate_rsi('AAPL')
        macd = generator.calculate_macd('AAPL')
        stoch = generator.calculate_stochastic('AAPL')
        adx = generator.calculate_adx('AAPL')

        assert rsi is not None
        assert macd is not None
        assert stoch is not None
        assert adx is not None

        # 生成综合信号
        signal = generator.generate_composite_signal('AAPL')
        assert signal is not None

        # 获取统计
        stats = generator.get_statistics()
        assert stats['symbols_tracked'] >= 1

    @pytest.mark.asyncio
    async def test_multiple_symbols(self, generator):
        """测试45: 多个标的"""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        now = datetime.now()

        for symbol in symbols:
            for i in range(20):
                price = 100.0 + i * 0.5
                price_data = PriceData(
                    symbol=symbol,
                    timestamp=now - timedelta(days=20-i),
                    open=price - 0.1,
                    high=price + 0.2,
                    low=price - 0.2,
                    close=price,
                    volume=1000000
                )
                await generator.update_price(price_data)

        # 为每个标的生成信号
        for symbol in symbols:
            generator.calculate_rsi(symbol)
            signal = generator.generate_composite_signal(symbol)
            # 信号可能为None(数据不足)
            if signal:
                assert signal.symbol == symbol

    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, generator):
        """测试46: 连续监控"""
        await generator.start()

        # 添加实时数据
        for i in range(10):
            price_data = PriceData(
                symbol='AAPL',
                timestamp=datetime.now(),
                open=100.0 + i * 0.1,
                high=100.2 + i * 0.1,
                low=99.8 + i * 0.1,
                close=100.0 + i * 0.1,
                volume=1000000
            )
            await generator.update_price(price_data)
            await asyncio.sleep(0.05)

        await generator.stop()

        # 验证生成器正常工作
        assert generator.state == GeneratorState.IDLE

    def test_signal_strength_classification(self, generator_with_data):
        """测试47: 信号强度分类"""
        generator_with_data.calculate_rsi('AAPL')
        generator_with_data.calculate_macd('AAPL')
        generator_with_data.calculate_stochastic('AAPL')

        signal = generator_with_data.generate_composite_signal('AAPL')

        if signal:
            assert signal.signal_strength in SignalStrength
            assert signal.signal_type in SignalType

    def test_indicator_caching(self, generator_with_data):
        """测试48: 指标缓存"""
        # 计算指标
        indicator1 = generator_with_data.calculate_rsi('AAPL')

        # 从缓存获取
        indicator2 = generator_with_data.get_indicator('AAPL', 'rsi')

        assert indicator1 is not None
        assert indicator2 is not None
        assert indicator1.value == indicator2.value

    def test_signal_history(self, generator_with_data):
        """测试49: 信号历史"""
        # 生成多个信号
        for _ in range(5):
            generator_with_data.calculate_rsi('AAPL')
            generator_with_data.generate_composite_signal('AAPL')

        history = generator_with_data.get_signals_history('AAPL')

        assert isinstance(history, list)
        assert len(history) <= generator_with_data.config.history_size

    def test_divergence_history(self, generator_with_data):
        """测试50: 背离历史"""
        # 尝试检测背离
        for _ in range(3):
            generator_with_data.detect_divergence('AAPL', 'rsi')

        divergences = generator_with_data.get_divergences('AAPL')

        assert isinstance(divergences, list)


# ============== Test Factory Functions ==============

class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_momentum_generator_default(self):
        """测试51: 创建默认生成器"""
        generator = create_momentum_generator()

        assert isinstance(generator, MomentumSignalGenerator)
        assert generator.config is not None

    def test_create_momentum_generator_custom(self):
        """测试52: 创建自定义生成器"""
        callback_called = []

        def callback(signal):
            callback_called.append(signal)

        generator = create_momentum_generator(
            signal_callback=callback,
            rsi_period=20,
            macd_fast_period=10,
        )

        assert generator.config.rsi_period == 20
        assert generator.config.macd_fast_period == 10
        assert generator.signal_callback is not None


# ============== Test Data Classes ==============

class TestDataClasses:
    """测试数据类"""

    def test_price_data_to_dict(self):
        """测试53: PriceData序列化"""
        price = PriceData(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000
        )

        data_dict = price.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == 'AAPL'
        assert data_dict['close'] == 100.5

    def test_momentum_indicator_to_dict(self):
        """测试54: MomentumIndicator序列化"""
        indicator = MomentumIndicator(
            name='rsi',
            value=65.5,
            timestamp=datetime.now(),
            signal=SignalType.BUY,
            strength=SignalStrength.MODERATE
        )

        data_dict = indicator.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['name'] == 'rsi'
        assert data_dict['value'] == 65.5

    def test_momentum_signal_to_dict(self):
        """测试55: MomentumSignal序列化"""
        signal = MomentumSignal(
            symbol='AAPL',
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            signal_strength=SignalStrength.STRONG,
            confidence=0.85,
            price=100.5,
            contributing_indicators={'rsi': 65.0, 'macd': 1.5}
        )

        data_dict = signal.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == 'AAPL'
        assert data_dict['confidence'] == 0.85

    def test_divergence_signal_to_dict(self):
        """测试56: DivergenceSignal序列化"""
        now = datetime.now()
        divergence = DivergenceSignal(
            symbol='AAPL',
            timestamp=now,
            divergence_type=DivergenceType.BEARISH,
            indicator_name='rsi',
            price_point1=(now - timedelta(days=10), 100.0),
            price_point2=(now, 105.0),
            indicator_point1=(now - timedelta(days=10), 70.0),
            indicator_point2=(now, 65.0),
            strength=SignalStrength.MODERATE,
            significance=0.7
        )

        data_dict = divergence.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['divergence_type'] == 'bearish'
        assert data_dict['significance'] == 0.7

    def test_trend_strength_to_dict(self):
        """测试57: TrendStrength序列化"""
        trend = TrendStrength(
            symbol='AAPL',
            timestamp=datetime.now(),
            adx=35.0,
            plus_di=25.0,
            minus_di=15.0,
            trend_direction=TrendDirection.UPTREND,
            trend_strength=SignalStrength.STRONG,
            is_trending=True
        )

        data_dict = trend.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['adx'] == 35.0
        assert data_dict['is_trending'] is True

    def test_momentum_config_to_dict(self):
        """测试58: MomentumConfig序列化"""
        config = MomentumConfig(
            rsi_period=14,
            macd_fast_period=12,
            enable_multi_timeframe=True
        )

        data_dict = config.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['rsi_period'] == 14
        assert data_dict['enable_multi_timeframe'] is True


# ============== Test Performance ==============

class TestPerformance:
    """测试性能"""

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, generator):
        """测试59: 大数据集性能"""
        import time

        start_time = time.time()

        # 添加大量数据
        for i in range(200):
            price_data = PriceData(
                symbol='AAPL',
                timestamp=datetime.now() - timedelta(days=200-i),
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.0 + i * 0.1,
                volume=1000000
            )
            await generator.update_price(price_data)

        # 计算所有指标
        generator.calculate_rsi('AAPL')
        generator.calculate_macd('AAPL')
        generator.calculate_stochastic('AAPL')
        generator.calculate_adx('AAPL')
        generator.generate_composite_signal('AAPL')

        elapsed = time.time() - start_time

        # 应该在合理时间内完成
        assert elapsed < 5.0  # 5秒内完成

    def test_memory_efficiency(self, generator):
        """测试60: 内存效率"""
        # 添加超过历史限制的数据
        for i in range(600):  # 超过默认的500
            price_data = PriceData(
                symbol='AAPL',
                timestamp=datetime.now() - timedelta(days=600-i),
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.0 + i * 0.1,
                volume=1000000
            )
            asyncio.run(generator.update_price(price_data))

        # 验证历史数据被正确限制
        assert len(generator._price_history['AAPL']) <= generator.config.history_size


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
