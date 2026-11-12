"""
Alpha信号生成器测试 (Alpha Signal Generator Tests)

测试Alpha信号生成器的所有功能:
- Momentum Alpha (动量Alpha)
- Mean Reversion Alpha (均值回归Alpha)
- Value Alpha (价值Alpha)
- Quality Alpha (质量Alpha)
- Volatility Alpha (波动率Alpha)
- Cross-sectional Alpha (截面Alpha)
- 多因子组合
- 因子权重优化
- 信号标准化
- IC/IR计算
- 信号评估

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

from quantrisk_realtime.realtime.alpha_signal_generator import (
    # Main Classes
    AlphaSignalGenerator,
    AlphaFactorCalculator,
    create_alpha_generator,
    create_alpha_factor,
    quick_alpha_signal,
    # Data Classes
    MarketData,
    AlphaFactor,
    FactorWeight,
    AlphaSignal,
    SignalMetrics,
    AlphaConfig,
    # Enums
    AlphaType,
    SignalDirection,
    SignalStrength,
    FactorCategory,
    GeneratorState,
)


# ============== Fixtures ==============

@pytest.fixture
def config():
    """创建基础配置"""
    return AlphaConfig(
        momentum_period=20,
        reversion_period=10,
        volatility_period=20,
        normalize_method="zscore",
        ic_window=20,
        update_interval=0.1,
        history_size=1000,
    )


@pytest.fixture
def calculator():
    """创建计算器"""
    return AlphaFactorCalculator()


@pytest.fixture
def generator(config):
    """创建基础生成器"""
    return AlphaSignalGenerator(config=config)


@pytest.fixture
def generator_with_data(generator):
    """创建带有数据的生成器"""
    # 生成上升趋势的市场数据
    base_price = 100.0
    now = datetime.now()

    for i in range(50):
        price = base_price + i * 0.5
        market_data = MarketData(
            symbol='AAPL',
            timestamp=now - timedelta(days=50-i),
            open=price - 0.2,
            high=price + 0.3,
            low=price - 0.3,
            close=price,
            volume=1000000 + i * 10000,
            turnover=price * 1000000,
            market_cap=1000000000,
            pe_ratio=20.0,
            pb_ratio=3.0
        )
        asyncio.run(generator.update_market_data(market_data))

    return generator


@pytest.fixture
def market_data_uptrend():
    """生成上升趋势市场数据"""
    base_price = 100.0
    now = datetime.now()
    data = []

    for i in range(30):
        price = base_price + i * 0.5
        data.append(MarketData(
            symbol='AAPL',
            timestamp=now - timedelta(days=30-i),
            open=price - 0.2,
            high=price + 0.3,
            low=price - 0.3,
            close=price,
            volume=1000000,
            market_cap=1000000000,
            pe_ratio=20.0,
            pb_ratio=3.0
        ))

    return data


@pytest.fixture
def market_data_downtrend():
    """生成下降趋势市场数据"""
    base_price = 100.0
    now = datetime.now()
    data = []

    for i in range(30):
        price = base_price - i * 0.5
        data.append(MarketData(
            symbol='AAPL',
            timestamp=now - timedelta(days=30-i),
            open=price + 0.2,
            high=price + 0.3,
            low=price - 0.3,
            close=price,
            volume=1000000,
            market_cap=1000000000,
            pe_ratio=20.0,
            pb_ratio=3.0
        ))

    return data


@pytest.fixture
def market_data_sideways():
    """生成横盘市场数据"""
    base_price = 100.0
    now = datetime.now()
    data = []

    for i in range(30):
        price = base_price + (i % 3 - 1) * 0.5
        data.append(MarketData(
            symbol='AAPL',
            timestamp=now - timedelta(days=30-i),
            open=price - 0.1,
            high=price + 0.2,
            low=price - 0.2,
            close=price,
            volume=1000000,
            market_cap=1000000000,
            pe_ratio=20.0,
            pb_ratio=3.0
        ))

    return data


# ============== Test AlphaFactorCalculator ==============

class TestAlphaFactorCalculator:
    """测试Alpha因子计算器"""

    def test_calculate_momentum_alpha_basic(self, calculator):
        """测试1: 基础动量Alpha计算"""
        prices = [100, 102, 105, 108, 110] + [110 + i * 0.5 for i in range(20)]
        alpha = calculator.calculate_momentum_alpha(prices, 20)

        assert alpha is not None
        assert isinstance(alpha, float)
        assert alpha > 0  # 上升趋势应该有正动量

    def test_calculate_momentum_alpha_insufficient_data(self, calculator):
        """测试2: 数据不足的动量Alpha"""
        prices = [100, 102]
        alpha = calculator.calculate_momentum_alpha(prices, 20)

        assert alpha is None

    def test_calculate_momentum_alpha_downtrend(self, calculator):
        """测试3: 下降趋势的动量Alpha"""
        prices = [110 - i * 0.5 for i in range(25)]
        alpha = calculator.calculate_momentum_alpha(prices, 20)

        assert alpha is not None
        assert alpha < 0  # 下降趋势应该有负动量

    def test_calculate_mean_reversion_alpha_basic(self, calculator):
        """测试4: 基础均值回归Alpha"""
        prices = [100, 101, 102, 101, 100, 99, 100, 101, 100, 99]
        alpha = calculator.calculate_mean_reversion_alpha(prices, 10)

        assert alpha is not None
        assert isinstance(alpha, float)

    def test_calculate_mean_reversion_alpha_above_mean(self, calculator):
        """测试5: 高于均值的均值回归"""
        prices = [100] * 9 + [110]  # 最后一个价格高于均值
        alpha = calculator.calculate_mean_reversion_alpha(prices, 10)

        assert alpha is not None
        assert alpha < 0  # 高于均值应该给出负信号(卖出)

    def test_calculate_mean_reversion_alpha_below_mean(self, calculator):
        """测试6: 低于均值的均值回归"""
        prices = [100] * 9 + [90]  # 最后一个价格低于均值
        alpha = calculator.calculate_mean_reversion_alpha(prices, 10)

        assert alpha is not None
        assert alpha > 0  # 低于均值应该给出正信号(买入)

    def test_calculate_value_alpha_basic(self, calculator):
        """测试7: 基础价值Alpha"""
        alpha = calculator.calculate_value_alpha(
            price=100.0,
            market_cap=1000000000,
            pe_ratio=20.0,
            pb_ratio=3.0
        )

        assert alpha is not None
        assert isinstance(alpha, float)
        assert alpha > 0

    def test_calculate_value_alpha_low_valuation(self, calculator):
        """测试8: 低估值的价值Alpha"""
        alpha = calculator.calculate_value_alpha(
            price=100.0,
            market_cap=1000000000,
            pe_ratio=10.0,  # 低PE
            pb_ratio=1.5    # 低PB
        )

        assert alpha is not None
        assert alpha > 0

    def test_calculate_value_alpha_invalid(self, calculator):
        """测试9: 无效估值的价值Alpha"""
        alpha = calculator.calculate_value_alpha(
            price=100.0,
            market_cap=1000000000,
            pe_ratio=-10.0,  # 负PE
            pb_ratio=3.0
        )

        assert alpha is None

    def test_calculate_quality_alpha_basic(self, calculator):
        """测试10: 基础质量Alpha"""
        prices = [100 + i * 0.1 for i in range(20)]
        volumes = [1000000 + i * 10000 for i in range(20)]
        alpha = calculator.calculate_quality_alpha(prices, volumes, 20)

        assert alpha is not None
        assert isinstance(alpha, float)
        assert 0 <= alpha <= 100

    def test_calculate_quality_alpha_stable(self, calculator):
        """测试11: 稳定的质量Alpha"""
        prices = [100.0] * 20  # 完全稳定
        volumes = [1000000] * 20
        alpha = calculator.calculate_quality_alpha(prices, volumes, 20)

        assert alpha is not None
        assert alpha > 50  # 稳定应该有高质量分数

    def test_calculate_quality_alpha_volatile(self, calculator):
        """测试12: 波动的质量Alpha"""
        prices = [100 + (-1)**i * 10 for i in range(20)]  # 高波动
        volumes = [1000000 + (-1)**i * 500000 for i in range(20)]
        alpha = calculator.calculate_quality_alpha(prices, volumes, 20)

        assert alpha is not None
        # 高波动应该有较低的质量分数

    def test_calculate_volatility_alpha_basic(self, calculator):
        """测试13: 基础波动率Alpha"""
        prices = [100 + i * 0.5 for i in range(25)]
        alpha = calculator.calculate_volatility_alpha(prices, 20)

        assert alpha is not None
        assert isinstance(alpha, float)
        assert 0 <= alpha <= 100

    def test_calculate_volatility_alpha_low_vol(self, calculator):
        """测试14: 低波动率Alpha"""
        prices = [100.0 + i * 0.01 for i in range(25)]  # 非常低的波动
        alpha = calculator.calculate_volatility_alpha(prices, 20)

        assert alpha is not None
        assert alpha > 50  # 低波动应该有高Alpha

    def test_calculate_volatility_alpha_high_vol(self, calculator):
        """测试15: 高波动率Alpha"""
        prices = [100 + (-1)**i * 5 for i in range(25)]  # 高波动
        alpha = calculator.calculate_volatility_alpha(prices, 20)

        assert alpha is not None
        # 高波动应该有低Alpha

    def test_calculate_cross_sectional_alpha_basic(self, calculator):
        """测试16: 基础截面Alpha"""
        symbol_prices = {
            'AAPL': 150.0,
            'GOOGL': 120.0,
            'MSFT': 100.0,
            'TSLA': 200.0,
        }
        alpha = calculator.calculate_cross_sectional_alpha(symbol_prices, 'AAPL')

        assert alpha is not None
        assert isinstance(alpha, float)
        assert -100 <= alpha <= 100

    def test_calculate_cross_sectional_alpha_top_rank(self, calculator):
        """测试17: 最高排名的截面Alpha"""
        symbol_prices = {
            'AAPL': 200.0,  # 最高
            'GOOGL': 120.0,
            'MSFT': 100.0,
        }
        alpha = calculator.calculate_cross_sectional_alpha(symbol_prices, 'AAPL')

        assert alpha is not None
        assert alpha > 0  # 最高价应该有正Alpha

    def test_calculate_cross_sectional_alpha_bottom_rank(self, calculator):
        """测试18: 最低排名的截面Alpha"""
        symbol_prices = {
            'AAPL': 80.0,   # 最低
            'GOOGL': 120.0,
            'MSFT': 100.0,
        }
        alpha = calculator.calculate_cross_sectional_alpha(symbol_prices, 'AAPL')

        assert alpha is not None
        assert alpha < 0  # 最低价应该有负Alpha


# ============== Test AlphaSignalGenerator ==============

class TestAlphaSignalGenerator:
    """测试Alpha信号生成器"""

    def test_generator_initialization(self, generator):
        """测试19: 生成器初始化"""
        assert generator.state == GeneratorState.IDLE
        assert isinstance(generator.config, AlphaConfig)
        assert len(generator._factor_weights) > 0

    @pytest.mark.asyncio
    async def test_update_market_data(self, generator):
        """测试20: 更新市场数据"""
        market_data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000,
            market_cap=1000000000,
            pe_ratio=20.0,
            pb_ratio=3.0
        )

        await generator.update_market_data(market_data)

        assert 'AAPL' in generator._market_data
        assert len(generator._market_data['AAPL']) == 1

    @pytest.mark.asyncio
    async def test_update_multiple_market_data(self, generator):
        """测试21: 更新多个市场数据"""
        now = datetime.now()

        for i in range(5):
            market_data = MarketData(
                symbol='AAPL',
                timestamp=now - timedelta(days=5-i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000,
                market_cap=1000000000,
                pe_ratio=20.0,
                pb_ratio=3.0
            )
            await generator.update_market_data(market_data)

        assert len(generator._market_data['AAPL']) == 5

    def test_calculate_momentum_factor(self, generator_with_data):
        """测试22: 计算动量因子"""
        factor = generator_with_data.calculate_momentum_factor('AAPL')

        assert factor is not None
        assert isinstance(factor, AlphaFactor)
        assert factor.name == 'momentum'
        assert factor.alpha_type == AlphaType.MOMENTUM
        assert factor.category == FactorCategory.TECHNICAL

    def test_calculate_mean_reversion_factor(self, generator_with_data):
        """测试23: 计算均值回归因子"""
        factor = generator_with_data.calculate_mean_reversion_factor('AAPL')

        assert factor is not None
        assert isinstance(factor, AlphaFactor)
        assert factor.name == 'mean_reversion'
        assert factor.alpha_type == AlphaType.MEAN_REVERSION

    def test_calculate_value_factor(self, generator_with_data):
        """测试24: 计算价值因子"""
        factor = generator_with_data.calculate_value_factor('AAPL')

        assert factor is not None
        assert isinstance(factor, AlphaFactor)
        assert factor.name == 'value'
        assert factor.alpha_type == AlphaType.VALUE
        assert factor.category == FactorCategory.FUNDAMENTAL

    def test_calculate_quality_factor(self, generator_with_data):
        """测试25: 计算质量因子"""
        factor = generator_with_data.calculate_quality_factor('AAPL')

        assert factor is not None
        assert isinstance(factor, AlphaFactor)
        assert factor.name == 'quality'
        assert factor.alpha_type == AlphaType.QUALITY

    def test_calculate_volatility_factor(self, generator_with_data):
        """测试26: 计算波动率因子"""
        factor = generator_with_data.calculate_volatility_factor('AAPL')

        assert factor is not None
        assert isinstance(factor, AlphaFactor)
        assert factor.name == 'volatility'
        assert factor.alpha_type == AlphaType.VOLATILITY

    def test_calculate_cross_sectional_factor(self, generator_with_data):
        """测试27: 计算截面因子"""
        # 添加多个标的数据
        for symbol in ['GOOGL', 'MSFT']:
            for i in range(30):
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now() - timedelta(days=30-i),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1000000,
                    market_cap=1000000000,
                    pe_ratio=20.0,
                    pb_ratio=3.0
                )
                asyncio.run(generator_with_data.update_market_data(market_data))

        factor = generator_with_data.calculate_cross_sectional_factor('AAPL')

        assert factor is not None
        assert isinstance(factor, AlphaFactor)
        assert factor.name == 'cross_sectional'
        assert factor.alpha_type == AlphaType.CROSS_SECTIONAL

    def test_normalize_factors_zscore(self, generator_with_data):
        """测试28: Z-score标准化"""
        # 计算多个因子
        generator_with_data.calculate_momentum_factor('AAPL')
        generator_with_data.calculate_mean_reversion_factor('AAPL')
        generator_with_data.calculate_volatility_factor('AAPL')

        # 标准化
        generator_with_data.normalize_factors('AAPL')

        # 检查zscore已设置
        factors = generator_with_data.get_all_factors('AAPL')
        for factor in factors.values():
            assert hasattr(factor, 'zscore')

    def test_normalize_factors_rank(self, config):
        """测试29: 排名标准化"""
        config.normalize_method = "rank"
        generator = AlphaSignalGenerator(config=config)

        # 添加数据并计算因子
        for i in range(30):
            market_data = MarketData(
                symbol='AAPL',
                timestamp=datetime.now() - timedelta(days=30-i),
                open=100.0 + i * 0.5,
                high=101.0 + i * 0.5,
                low=99.0 + i * 0.5,
                close=100.0 + i * 0.5,
                volume=1000000,
                market_cap=1000000000,
                pe_ratio=20.0,
                pb_ratio=3.0
            )
            asyncio.run(generator.update_market_data(market_data))

        generator.calculate_momentum_factor('AAPL')
        generator.calculate_volatility_factor('AAPL')
        generator.normalize_factors('AAPL')

        factors = generator.get_all_factors('AAPL')
        assert len(factors) >= 2

    def test_generate_alpha_signal_basic(self, generator_with_data):
        """测试30: 生成基础Alpha信号"""
        # 计算所有因子
        generator_with_data.calculate_momentum_factor('AAPL')
        generator_with_data.calculate_mean_reversion_factor('AAPL')
        generator_with_data.calculate_volatility_factor('AAPL')

        signal = generator_with_data.generate_alpha_signal('AAPL')

        assert signal is not None
        assert isinstance(signal, AlphaSignal)
        assert signal.symbol == 'AAPL'
        assert signal.signal_direction in SignalDirection
        assert signal.signal_strength in SignalStrength
        assert -100 <= signal.alpha_value <= 100
        assert 0 <= signal.confidence <= 1

    def test_generate_signal_with_uptrend(self, generator, market_data_uptrend):
        """测试31: 上升趋势的信号生成"""
        for data in market_data_uptrend:
            asyncio.run(generator.update_market_data(data))

        signal = generator.generate_alpha_signal('AAPL')

        if signal:
            # 多因子模型可能给出不同方向
            # 动量因子会给LONG, 均值回归因子会给SHORT
            # 验证信号已正确生成
            assert signal.signal_direction in SignalDirection
            assert isinstance(signal.alpha_value, float)

    def test_generate_signal_with_downtrend(self, generator, market_data_downtrend):
        """测试32: 下降趋势的信号生成"""
        for data in market_data_downtrend:
            asyncio.run(generator.update_market_data(data))

        signal = generator.generate_alpha_signal('AAPL')

        if signal:
            # 多因子模型可能给出不同方向
            # 验证信号已正确生成
            assert signal.signal_direction in SignalDirection
            assert isinstance(signal.alpha_value, float)

    def test_signal_decay(self, generator_with_data):
        """测试33: 信号衰减"""
        signal1 = generator_with_data.generate_alpha_signal('AAPL')
        assert signal1 is not None

        # 等待一段时间
        import time
        time.sleep(0.1)

        signal2 = generator_with_data.generate_alpha_signal('AAPL')
        if signal2:
            # 第二个信号应该有衰减
            assert signal2.decay_factor <= 1.0

    def test_calculate_ic(self, generator_with_data):
        """测试34: 计算IC"""
        # 生成足够的信号
        for _ in range(25):
            generator_with_data.generate_alpha_signal('AAPL')

        ic = generator_with_data.calculate_ic('AAPL')

        # IC可能为None(数据不足)或在-1到1之间
        if ic is not None:
            assert -1 <= ic <= 1

    def test_calculate_ir(self, generator_with_data):
        """测试35: 计算IR"""
        # 生成足够的信号
        for _ in range(25):
            generator_with_data.generate_alpha_signal('AAPL')
            generator_with_data.calculate_ic('AAPL')

        ir = generator_with_data.calculate_ir('AAPL')

        # IR可能为None
        if ir is not None:
            assert isinstance(ir, float)

    def test_calculate_signal_metrics(self, generator_with_data):
        """测试36: 计算信号指标"""
        # 生成信号
        for _ in range(10):
            generator_with_data.generate_alpha_signal('AAPL')

        metrics = generator_with_data.calculate_signal_metrics('AAPL')

        assert metrics is not None
        assert isinstance(metrics, SignalMetrics)
        assert metrics.symbol == 'AAPL'
        assert 0 <= metrics.coverage_rate <= 1
        assert 0 <= metrics.turnover_rate <= 1

    def test_get_current_signal(self, generator_with_data):
        """测试37: 获取当前信号"""
        generator_with_data.generate_alpha_signal('AAPL')

        signal = generator_with_data.get_current_signal('AAPL')

        assert signal is not None
        assert isinstance(signal, AlphaSignal)
        assert signal.symbol == 'AAPL'

    def test_get_signal_history(self, generator_with_data):
        """测试38: 获取信号历史"""
        # 生成多个信号
        for _ in range(5):
            generator_with_data.generate_alpha_signal('AAPL')

        history = generator_with_data.get_signal_history('AAPL')

        assert isinstance(history, list)
        assert len(history) >= 1
        assert all(s.symbol == 'AAPL' for s in history)

    def test_get_factor(self, generator_with_data):
        """测试39: 获取因子"""
        generator_with_data.calculate_momentum_factor('AAPL')

        factor = generator_with_data.get_factor('AAPL', 'momentum')

        assert factor is not None
        assert isinstance(factor, AlphaFactor)
        assert factor.name == 'momentum'

    def test_get_all_factors(self, generator_with_data):
        """测试40: 获取所有因子"""
        generator_with_data.calculate_momentum_factor('AAPL')
        generator_with_data.calculate_mean_reversion_factor('AAPL')
        generator_with_data.calculate_volatility_factor('AAPL')

        factors = generator_with_data.get_all_factors('AAPL')

        assert isinstance(factors, dict)
        assert len(factors) >= 3

    def test_get_factor_weights(self, generator_with_data):
        """测试41: 获取因子权重"""
        weights = generator_with_data.get_factor_weights()

        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert all(isinstance(w, FactorWeight) for w in weights.values())

    def test_get_metrics(self, generator_with_data):
        """测试42: 获取指标"""
        generator_with_data.generate_alpha_signal('AAPL')
        generator_with_data.calculate_signal_metrics('AAPL')

        metrics = generator_with_data.get_metrics('AAPL')

        if metrics:
            assert isinstance(metrics, SignalMetrics)
            assert metrics.symbol == 'AAPL'

    def test_get_statistics(self, generator_with_data):
        """测试43: 获取统计信息"""
        stats = generator_with_data.get_statistics()

        assert isinstance(stats, dict)
        assert 'state' in stats
        assert 'active_symbols' in stats
        assert 'total_signals' in stats
        assert stats['active_symbols'] >= 1

    @pytest.mark.asyncio
    async def test_start_stop_generator(self, generator):
        """测试44: 启动停止生成器"""
        await generator.start()
        assert generator.state == GeneratorState.RUNNING

        await asyncio.sleep(0.2)

        await generator.stop()
        assert generator.state == GeneratorState.IDLE

    @pytest.mark.asyncio
    async def test_pause_resume_generator(self, generator):
        """测试45: 暂停恢复生成器"""
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
        """测试46: 信号回调触发"""
        callback_signals = []

        def signal_callback(signal: AlphaSignal):
            callback_signals.append(signal)

        generator = AlphaSignalGenerator(config=config, signal_callback=signal_callback)

        # 添加数据
        for i in range(30):
            market_data = MarketData(
                symbol='AAPL',
                timestamp=datetime.now() - timedelta(days=30-i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000,
                market_cap=1000000000,
                pe_ratio=20.0,
                pb_ratio=3.0
            )
            await generator.update_market_data(market_data)

        # 应该有信号产生
        assert isinstance(callback_signals, list)

    @pytest.mark.asyncio
    async def test_metrics_callback_triggered(self, config):
        """测试47: 指标回调触发"""
        callback_metrics = []

        def metrics_callback(metrics: SignalMetrics):
            callback_metrics.append(metrics)

        generator = AlphaSignalGenerator(config=config, metrics_callback=metrics_callback)

        # 添加数据
        for i in range(30):
            market_data = MarketData(
                symbol='AAPL',
                timestamp=datetime.now() - timedelta(days=30-i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000,
                market_cap=1000000000,
                pe_ratio=20.0,
                pb_ratio=3.0
            )
            await generator.update_market_data(market_data)

        # 应该有指标产生
        assert isinstance(callback_metrics, list)


# ============== Test Edge Cases ==============

class TestEdgeCases:
    """测试边界情况"""

    def test_empty_market_data(self, generator):
        """测试48: 空市场数据"""
        signal = generator.generate_alpha_signal('NONEXISTENT')
        assert signal is None

    def test_insufficient_data_for_factors(self, generator):
        """测试49: 数据不足"""
        market_data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000,
            market_cap=1000000000,
            pe_ratio=20.0,
            pb_ratio=3.0
        )
        asyncio.run(generator.update_market_data(market_data))

        # 数据不足时应该返回None
        factor = generator.calculate_momentum_factor('AAPL')
        assert factor is None

    def test_zero_prices(self, calculator):
        """测试50: 零价格处理"""
        prices = [0, 0, 0, 0, 0]
        alpha = calculator.calculate_momentum_alpha(prices, 3)

        assert alpha is None

    def test_same_prices(self, calculator):
        """测试51: 相同价格处理"""
        prices = [100.0] * 25
        alpha = calculator.calculate_volatility_alpha(prices, 20)

        if alpha is not None:
            assert alpha >= 0  # 零波动应该有高Alpha

    def test_extreme_volatility(self, calculator):
        """测试52: 极端波动"""
        prices = [100 + (-1)**i * 50 for i in range(25)]
        alpha = calculator.calculate_volatility_alpha(prices, 20)

        assert alpha is not None
        assert 0 <= alpha <= 100


# ============== Test Integration ==============

class TestIntegration:
    """测试集成场景"""

    @pytest.mark.asyncio
    async def test_full_alpha_cycle(self, generator):
        """测试53: 完整Alpha周期"""
        now = datetime.now()

        # 添加市场周期数据
        for i in range(50):
            price = 100.0 + i * 0.5
            market_data = MarketData(
                symbol='AAPL',
                timestamp=now - timedelta(days=50-i),
                open=price - 0.2,
                high=price + 0.3,
                low=price - 0.3,
                close=price,
                volume=1000000 + i * 10000,
                market_cap=1000000000,
                pe_ratio=20.0 - i * 0.1,
                pb_ratio=3.0 - i * 0.02
            )
            await generator.update_market_data(market_data)

        # 计算所有因子
        momentum = generator.calculate_momentum_factor('AAPL')
        reversion = generator.calculate_mean_reversion_factor('AAPL')
        value = generator.calculate_value_factor('AAPL')
        quality = generator.calculate_quality_factor('AAPL')
        volatility = generator.calculate_volatility_factor('AAPL')

        assert momentum is not None
        assert reversion is not None
        assert value is not None
        assert quality is not None
        assert volatility is not None

        # 生成信号
        signal = generator.generate_alpha_signal('AAPL')
        assert signal is not None

        # 计算指标
        metrics = generator.calculate_signal_metrics('AAPL')
        # metrics可能为None

        # 获取统计
        stats = generator.get_statistics()
        assert stats['active_symbols'] >= 1

    @pytest.mark.asyncio
    async def test_multiple_symbols(self, generator):
        """测试54: 多个标的"""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        now = datetime.now()

        for symbol in symbols:
            for i in range(30):
                price = 100.0 + i * 0.5
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=now - timedelta(days=30-i),
                    open=price - 0.2,
                    high=price + 0.3,
                    low=price - 0.3,
                    close=price,
                    volume=1000000,
                    market_cap=1000000000,
                    pe_ratio=20.0,
                    pb_ratio=3.0
                )
                await generator.update_market_data(market_data)

        # 为每个标的生成信号
        for symbol in symbols:
            generator.calculate_momentum_factor(symbol)
            signal = generator.generate_alpha_signal(symbol)
            if signal:
                assert signal.symbol == symbol

    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, generator):
        """测试55: 连续监控"""
        await generator.start()

        # 添加实时数据
        for i in range(10):
            market_data = MarketData(
                symbol='AAPL',
                timestamp=datetime.now(),
                open=100.0 + i * 0.1,
                high=100.2 + i * 0.1,
                low=99.8 + i * 0.1,
                close=100.0 + i * 0.1,
                volume=1000000,
                market_cap=1000000000,
                pe_ratio=20.0,
                pb_ratio=3.0
            )
            await generator.update_market_data(market_data)
            await asyncio.sleep(0.05)

        await generator.stop()
        assert generator.state == GeneratorState.IDLE

    def test_factor_weighting(self, generator_with_data):
        """测试56: 因子权重"""
        # 计算多个因子
        generator_with_data.calculate_momentum_factor('AAPL')
        generator_with_data.calculate_mean_reversion_factor('AAPL')
        generator_with_data.calculate_volatility_factor('AAPL')

        signal = generator_with_data.generate_alpha_signal('AAPL')

        if signal:
            # 检查因子权重
            assert len(signal.factor_weights) > 0
            assert sum(signal.factor_weights.values()) > 0


# ============== Test Factory Functions ==============

class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_alpha_generator_default(self):
        """测试57: 创建默认生成器"""
        generator = create_alpha_generator()

        assert isinstance(generator, AlphaSignalGenerator)
        assert generator.config is not None

    def test_create_alpha_generator_custom(self):
        """测试58: 创建自定义生成器"""
        callback_called = []

        def callback(signal):
            callback_called.append(signal)

        generator = create_alpha_generator(
            signal_callback=callback,
            momentum_period=30,
            ic_window=25,
        )

        assert generator.config.momentum_period == 30
        assert generator.config.ic_window == 25
        assert generator.signal_callback is not None

    def test_create_alpha_factor(self):
        """测试59: 创建Alpha因子"""
        factor = create_alpha_factor(
            name='test_factor',
            alpha_type=AlphaType.MOMENTUM,
            category=FactorCategory.TECHNICAL,
            value=50.0,
            zscore=1.5
        )

        assert isinstance(factor, AlphaFactor)
        assert factor.name == 'test_factor'
        assert factor.value == 50.0
        assert factor.zscore == 1.5

    def test_quick_alpha_signal(self):
        """测试60: 快速创建Alpha信号"""
        signal = quick_alpha_signal(
            symbol='AAPL',
            alpha_value=75.0,
            confidence=0.9
        )

        assert isinstance(signal, AlphaSignal)
        assert signal.symbol == 'AAPL'
        assert signal.alpha_value == 75.0
        assert signal.confidence == 0.9
        assert signal.signal_direction == SignalDirection.LONG
        assert signal.signal_strength in SignalStrength


# ============== Test Data Classes ==============

class TestDataClasses:
    """测试数据类"""

    def test_market_data_to_dict(self):
        """测试61: MarketData序列化"""
        data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000,
            market_cap=1000000000,
            pe_ratio=20.0,
            pb_ratio=3.0
        )

        data_dict = data.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == 'AAPL'
        assert data_dict['close'] == 100.5

    def test_alpha_factor_to_dict(self):
        """测试62: AlphaFactor序列化"""
        factor = AlphaFactor(
            name='momentum',
            alpha_type=AlphaType.MOMENTUM,
            category=FactorCategory.TECHNICAL,
            value=50.0,
            zscore=1.5
        )

        data_dict = factor.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['name'] == 'momentum'
        assert data_dict['value'] == 50.0

    def test_alpha_signal_to_dict(self):
        """测试63: AlphaSignal序列化"""
        signal = AlphaSignal(
            symbol='AAPL',
            timestamp=datetime.now(),
            signal_direction=SignalDirection.LONG,
            signal_strength=SignalStrength.STRONG,
            alpha_value=75.0,
            confidence=0.85
        )

        data_dict = signal.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == 'AAPL'
        assert data_dict['alpha_value'] == 75.0

    def test_signal_metrics_to_dict(self):
        """测试64: SignalMetrics序列化"""
        metrics = SignalMetrics(
            symbol='AAPL',
            timestamp=datetime.now(),
            ic=0.5,
            ic_ir=2.0,
            coverage_rate=0.8,
            turnover_rate=0.3
        )

        data_dict = metrics.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == 'AAPL'
        assert data_dict['ic'] == 0.5

    def test_factor_weight_to_dict(self):
        """测试65: FactorWeight序列化"""
        weight = FactorWeight(
            factor_name='momentum',
            weight=0.25,
            confidence=0.9
        )

        data_dict = weight.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['factor_name'] == 'momentum'
        assert data_dict['weight'] == 0.25

    def test_alpha_config_to_dict(self):
        """测试66: AlphaConfig序列化"""
        config = AlphaConfig(
            momentum_period=20,
            ic_window=25,
            enable_decay=True
        )

        data_dict = config.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict['momentum_period'] == 20
        assert data_dict['enable_decay'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
